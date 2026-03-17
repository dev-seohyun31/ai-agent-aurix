"""
evaluate_ragas.py
RAGAS 기반 RAG 품질 자동 평가 — Vector DB 단독 vs GraphRAG 비교

4개 지표:
  Faithfulness       답변이 컨텍스트에 근거하는가 (할루시네이션 측정)
  Answer Relevancy   답변이 질문에 얼마나 관련있는가
  Context Precision  검색된 청크가 실제로 유용한가
  Context Recall     정답에 필요한 정보가 청크에 포함됐는가

사용법:
    python evaluate_ragas.py                   # Vector DB vs GraphRAG 전체 비교
    python evaluate_ragas.py --mode vector     # Vector DB 단독만
    python evaluate_ragas.py --mode graphrag   # GraphRAG만
    python evaluate_ragas.py --id TC-F01       # 특정 케이스만
"""

import os
import sys
import json
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── RAGAS ────────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

# ── LlamaIndex ────────────────────────────────────────
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle

# ── 프로젝트 내부 ──────────────────────────────────────
import config
from core.models import setup_all
from core.vector_store import get_vector_index
from core.graph_store import get_kg_index
from rag.graphrag import graphrag_query

# ── 색상 출력 ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"{GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"{RED}❌ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"{BLUE}ℹ️  {msg}{RESET}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 1. RAGAS 전용 LLM·임베딩 설정 ────────────────────
def setup_ragas_evaluator():
    """RAGAS 채점용 LLM + 임베딩 — LlamaIndex Settings와 별개"""
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
            temperature=0,
            request_timeout=60,
            max_retries=3,
        )
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    )
    return ragas_llm, ragas_emb


# ── 2. Vector DB 단독 실행 ────────────────────────────
def run_vector(question: str, vector_index, reranker) -> dict:
    retriever      = vector_index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
    nodes          = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(question)
    )
    contexts    = [n.node.text for n in reranked_nodes]
    synthesizer = get_response_synthesizer(response_mode="tree_summarize")
    response    = synthesizer.synthesize(question, nodes=reranked_nodes)

    return {
        "answer":   str(response).strip(),
        "contexts": contexts,
    }


# ── 3. GraphRAG 실행 ──────────────────────────────────
def run_graphrag(question: str, vector_index, reranker) -> dict:
    result = graphrag_query(question, vector_index, reranker)

    retriever      = vector_index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
    nodes          = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(question)
    )
    contexts = [n.node.text for n in reranked_nodes]

    if result["graph_triples"]:
        contexts.append(f"[Knowledge Graph 관계]\n{result['graph_triples']}")

    return {
        "answer":   result["answer"],
        "contexts": contexts,
    }


# ── 4. RAGAS Dataset 구성 ─────────────────────────────
def build_dataset(cases, vector_index, reranker, mode: str) -> Dataset:
    questions, answers, contexts_list, ground_truths = [], [], [], []
    total = len(cases)

    for i, case in enumerate(cases):
        question = case["question"]
        print(f"  [{i+1}/{total}] {question[:40]}...", end=" ", flush=True)
        start = time.time()

        if mode == "vector":
            result = run_vector(question, vector_index, reranker)
        else:
            result = run_graphrag(question, vector_index, reranker)

        print(f"({round(time.time()-start, 1)}s)")

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(case["ground_truth"])

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ── 5. RAGAS 채점 ─────────────────────────────────────
def run_ragas(dataset: Dataset, ragas_llm, ragas_emb) -> dict:
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm        = ragas_llm
        m.embeddings = ragas_emb

    result = evaluate(dataset=dataset, metrics=metrics)
    df     = result.to_pandas()
    keys   = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    return {k: float(df[k].mean()) for k in keys if k in df.columns}


# ── 6. 결과 출력 ──────────────────────────────────────
def print_scores(label: str, scores: dict, color: str):
    print(f"\n{color}{BOLD}── {label} ──{RESET}")
    for key, name in [
        ("faithfulness",      "Faithfulness"),
        ("answer_relevancy",  "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("context_recall",    "Context Recall"),
    ]:
        val = scores.get(key)
        if val is None:
            warn(f"{name:<25} N/A")
            continue
        bar    = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        status = (f"{GREEN}✅{RESET}" if val >= 0.7
                  else f"{YELLOW}⚠️ {RESET}" if val >= 0.5
                  else f"{RED}❌{RESET}")
        print(f"  {name:<25} {val:.3f}  {bar}  {status}")


def print_comparison(v_scores: dict, g_scores: dict):
    metrics = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }

    print(f"\n{BOLD}{'='*65}")
    print("📊 Vector DB 단독 vs GraphRAG 최종 비교")
    print(f"{'='*65}{RESET}")
    print(f"  {'지표':<25} {'Vector DB':>10} {'GraphRAG':>10}  승자")
    print(f"  {'─'*58}")

    g_wins, v_wins = 0, 0
    for key, name in metrics.items():
        v, g = v_scores.get(key), g_scores.get(key)
        if v is None or g is None:
            print(f"  {name:<25} {'N/A':>10} {'N/A':>10}")
            continue
        diff = g - v
        if diff > 0.01:
            winner = f"{GREEN}GraphRAG +{diff:.3f}{RESET}"
            g_wins += 1
        elif diff < -0.01:
            winner = f"{YELLOW}Vector  +{abs(diff):.3f}{RESET}"
            v_wins += 1
        else:
            winner = f"{BLUE}동점{RESET}"
        print(f"  {name:<25} {v:>10.3f} {g:>10.3f}  {winner}")

    print(f"\n  최종: GraphRAG {g_wins}승 / Vector DB {v_wins}승")

    if g_wins > v_wins:
        ok("GraphRAG가 전반적으로 우세합니다.")
    elif g_wins < v_wins:
        warn("Vector DB 단독이 우세합니다. Knowledge Graph 품질 개선을 검토하세요.")
    else:
        warn("두 방식이 동등합니다.")

    print(f"\n{BOLD}개선 제안:{RESET}")
    if min(v_scores.get("faithfulness", 1), g_scores.get("faithfulness", 1)) < 0.7:
        fail("Faithfulness 낮음 → system_prompt 강화 또는 similarity_threshold 추가")
    if min(v_scores.get("context_precision", 1), g_scores.get("context_precision", 1)) < 0.7:
        warn("Context Precision 낮음 → chunk_size 줄이기 또는 similarity_top_k 조정")
    if min(v_scores.get("context_recall", 1), g_scores.get("context_recall", 1)) < 0.7:
        warn("Context Recall 낮음 → chunk_overlap 늘리기 또는 문서 추가 검토")
    if g_scores.get("faithfulness", 1) < v_scores.get("faithfulness", 1):
        warn("GraphRAG Faithfulness가 낮음 → Graph 관계 품질 개선 필요")


# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGAS 평가 스크립트")
    parser.add_argument("--mode", choices=["vector", "graphrag", "both"], default="both")
    parser.add_argument("--id", help="특정 케이스 ID (예: TC-F01)")
    args = parser.parse_args()

    # 테스트 케이스 로드 (hallucination 제외)
    with open(os.path.join(BASE_DIR, "test_cases.json"), "r", encoding="utf-8") as f:
        all_cases = json.load(f)["test_cases"]

    cases = [c for c in all_cases if c["type"] != "hallucination"]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]

    if not cases:
        fail("평가할 케이스가 없습니다.")
        return

    info(f"평가 케이스: {len(cases)}개 (hallucination 제외)")

    # 초기화 — core/ 에서 한 번만
    reranker     = setup_all()
    vector_index = get_vector_index()
    if args.mode in ("graphrag", "both"):
        get_kg_index()  # Neo4j 연결 확인
    ragas_llm, ragas_emb = setup_ragas_evaluator()

    v_scores, g_scores = None, None

    if args.mode in ("vector", "both"):
        print(f"\n{BOLD}🔍 Vector DB 단독 평가 중...{RESET}")
        v_dataset = build_dataset(cases, vector_index, reranker, mode="vector")
        print("  RAGAS 채점 중...")
        v_scores = run_ragas(v_dataset, ragas_llm, ragas_emb)
        print_scores("Vector DB 단독", v_scores, BLUE)

    if args.mode in ("graphrag", "both"):
        print(f"\n{BOLD}🔗 GraphRAG 평가 중...{RESET}")
        g_dataset = build_dataset(cases, vector_index, reranker, mode="graphrag")
        print("  RAGAS 채점 중...")
        g_scores = run_ragas(g_dataset, ragas_llm, ragas_emb)
        print_scores("GraphRAG", g_scores, GREEN)

    if v_scores and g_scores:
        print_comparison(v_scores, g_scores)


if __name__ == "__main__":
    main()