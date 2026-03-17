"""
evaluate_ragas.py
RAGAS 기반 RAG 품질 자동 평가 스크립트

4개 지표:
  - Faithfulness       답변이 컨텍스트에 근거하는가 (할루시네이션 측정)
  - Answer Relevancy   답변이 질문에 얼마나 관련있는가
  - Context Precision  검색된 청크가 실제로 유용한가
  - Context Recall     정답에 필요한 정보가 청크에 포함됐는가

사용법:
    python evaluate_ragas.py                   # Vector DB vs GraphRAG 전체 비교
    python evaluate_ragas.py --mode vector     # Vector DB 단독만
    python evaluate_ragas.py --mode graphrag   # GraphRAG만
    python evaluate_ragas.py --id TC-001       # 특정 케이스만
"""

import os
import sys
import json
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ── RAGAS imports ─────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

# ── LlamaIndex imports ────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
import chromadb

from rag.graphrag import graphrag_query, query_graph

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

# ── 설정 ─────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH       = os.path.join(BASE_DIR, "chroma_db")
CHROMA_COLLECTION = "research_docs"
GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]


# ── 1. 모델 설정 ──────────────────────────────────────
def setup_llamaindex():
    """LlamaIndex용 모델 설정"""
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        system_prompt=(
            "You are a technical document assistant for AURIX microcontroller. "
            "ALWAYS respond in Korean only. "
            "Only use information from the provided documents. "
            "If the answer is not found, say '문서에서 찾을 수 없습니다'."
        )
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")


def setup_ragas():
    """RAGAS 평가용 LLM + 임베딩 설정"""
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            request_timeout=60,
            max_retries=3
        )
    )
    ragas_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    )
    return ragas_llm, ragas_emb


def setup_vector_db():
    """Vector DB 연결"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    if collection.count() == 0:
        fail("chroma_db가 비어있습니다. db/vectordb.py를 먼저 실행하세요.")
        exit(1)

    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index           = VectorStoreIndex.from_vector_store(vector_store)
    info(f"Vector DB 연결 완료 ({collection.count()}개 청크)")
    return index


# ── 2. Vector DB 단독 실행 ────────────────────────────
def run_vector(question: str, vector_index, reranker) -> dict:
    # top_k=10 검색 → Reranker → 상위 5개
    retriever      = vector_index.as_retriever(similarity_top_k=10)
    nodes          = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(question)
    )
    contexts = [n.node.text for n in reranked_nodes]

    # Reranker 통과한 노드로 직접 응답 생성 (query_engine 사용 안 함)
    synthesizer = get_response_synthesizer(response_mode="tree_summarize")
    response    = synthesizer.synthesize(question, nodes=reranked_nodes)

    return {
        "answer":   str(response).strip(),
        "contexts": contexts,
    }

# ── 3. GraphRAG 실행 ──────────────────────────────────
def run_graphrag(question: str, vector_index, reranker) -> dict:
    """GraphRAG 쿼리 — 답변 + 컨텍스트(청크 + 관계) 반환"""
    result   = graphrag_query(question, vector_index, reranker)
    contexts = []

    # Vector DB 청크
    retriever = vector_index.as_retriever(similarity_top_k=10)
    nodes     = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(question)
    )
    contexts = [n.node.text for n in reranked_nodes]

    # Knowledge Graph 관계 → 컨텍스트로 추가
    if result["graph_triples"]:
        contexts.append(f"[Knowledge Graph 관계]\n{result['graph_triples']}")

    return {
        "answer":   result["answer"],
        "contexts": contexts,
    }


# ── 4. RAGAS 데이터셋 구성 ────────────────────────────
def build_dataset(cases, vector_index, reranker, mode: str) -> Dataset:
    """
    RAGAS가 요구하는 Dataset 형식으로 변환
    필드: question, answer, contexts, ground_truth
    """
    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    total = len(cases)
    for i, case in enumerate(cases):
        question = case["question"]
        gt       = case["ground_truth"]

        print(f"  [{i+1}/{total}] {question[:40]}...", end=" ", flush=True)
        start = time.time()

        if mode == "vector":
            result = run_vector(question, vector_index, reranker)
        else:
            result = run_graphrag(question, vector_index, reranker)

        elapsed = round(time.time() - start, 1)
        print(f"({elapsed}s)")

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(gt)

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })


# ── 5. RAGAS 평가 실행 ────────────────────────────────
def run_ragas(dataset: Dataset, ragas_llm, ragas_emb) -> dict:
    """RAGAS 4개 지표 계산"""
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # RAGAS 평가 LLM/임베딩 주입
    for metric in metrics:
        metric.llm       = ragas_llm
        metric.embeddings = ragas_emb

    result = evaluate(dataset=dataset, metrics=metrics)
    return result


# ── 6. 결과 출력 ──────────────────────────────────────
def print_scores(label: str, scores: dict, color: str):
    print(f"\n{color}{BOLD}── {label} ──{RESET}")

    def fmt(key, name):
        val = scores.get(key)
        if val is None:
            warn(f"{name:<25} N/A")
            return
        bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
        score_str = f"{val:.3f}"
        if val >= 0.7:
            status = f"{GREEN}✅{RESET}"
        elif val >= 0.5:
            status = f"{YELLOW}⚠️ {RESET}"
        else:
            status = f"{RED}❌{RESET}"
        print(f"  {name:<25} {score_str}  {bar}  {status}")

    fmt("faithfulness",       "Faithfulness")
    fmt("answer_relevancy",   "Answer Relevancy")
    fmt("context_precision",  "Context Precision")
    fmt("context_recall",     "Context Recall")


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
    print(f"  {'지표':<25} {'Vector DB':>10} {'GraphRAG':>10}  {'승자'}")
    print(f"  {'─'*58}")

    graphrag_wins = 0
    vector_wins   = 0

    for key, name in metrics.items():
        v = v_scores.get(key)
        g = g_scores.get(key)
        if v is None or g is None:
            print(f"  {name:<25} {'N/A':>10} {'N/A':>10}")
            continue

        diff = g - v
        if diff > 0.01:
            winner = f"{GREEN}GraphRAG +{diff:.3f}{RESET}"
            graphrag_wins += 1
        elif diff < -0.01:
            winner = f"{YELLOW}Vector  +{abs(diff):.3f}{RESET}"
            vector_wins += 1
        else:
            winner = f"{BLUE}동점{RESET}"

        print(f"  {name:<25} {v:>10.3f} {g:>10.3f}  {winner}")

    print(f"\n  최종: GraphRAG {graphrag_wins}승 / Vector DB {vector_wins}승")

    if graphrag_wins > vector_wins:
        ok(f"GraphRAG가 전반적으로 우세합니다.")
    elif graphrag_wins < vector_wins:
        warn(f"Vector DB 단독이 우세합니다. Knowledge Graph 품질 개선을 검토하세요.")
    else:
        warn(f"두 방식이 동등합니다.")

    # 개선 제안
    print(f"\n{BOLD}개선 제안:{RESET}")
    v_faith = v_scores.get("faithfulness", 1)
    g_faith = g_scores.get("faithfulness", 1)
    v_cp    = v_scores.get("context_precision", 1)
    g_cp    = g_scores.get("context_precision", 1)
    v_cr    = v_scores.get("context_recall", 1)
    g_cr    = g_scores.get("context_recall", 1)

    if min(v_faith, g_faith) < 0.7:
        fail("Faithfulness 낮음 → system_prompt 강화 또는 similarity_threshold 추가")
    if min(v_cp, g_cp) < 0.7:
        warn("Context Precision 낮음 → chunk_size 줄이기 또는 similarity_top_k 조정")
    if min(v_cr, g_cr) < 0.7:
        warn("Context Recall 낮음 → chunk_overlap 늘리기 또는 문서 추가 검토")
    if g_faith < v_faith:
        warn("GraphRAG Faithfulness가 낮음 → Graph 관계 품질 개선 필요")


# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAGAS 평가 스크립트")
    parser.add_argument("--mode", choices=["vector", "graphrag", "both"], default="both",
                        help="평가 대상 (default: both)")
    parser.add_argument("--id",  help="특정 케이스 ID (예: TC-001)")
    args = parser.parse_args()

    # 테스트 케이스 로드
    cases_path = os.path.join(BASE_DIR, "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)["test_cases"]

    # hallucination 케이스 제외 (ground_truth가 "문서에서 찾을 수 없습니다"라 Context Recall 측정 불가)
    cases = [c for c in all_cases if c["type"] != "hallucination"]
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]

    if not cases:
        fail("평가할 케이스가 없습니다.")
        return

    info(f"평가 케이스: {len(cases)}개 (hallucination 제외)")

    # 모델 초기화
    setup_llamaindex()
    vector_index    = setup_vector_db()
    ragas_llm, ragas_emb = setup_ragas()

    v_scores = None
    g_scores = None

    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=5,
    )

    # Vector DB 단독 평가
    if args.mode in ("vector", "both"):
        print(f"\n{BOLD}🔍 Vector DB 단독 평가 중...{RESET}")
        v_dataset = build_dataset(cases, vector_index, reranker, mode="vector")
        print("  RAGAS 채점 중...")
        v_result  = run_ragas(v_dataset, ragas_llm, ragas_emb)
        v_df = v_result.to_pandas()
        score_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        v_scores = {k: float(v_df[k].mean()) for k in score_keys if k in v_df.columns}

        print_scores("Vector DB 단독", v_scores, BLUE)

    # GraphRAG 평가
    if args.mode in ("graphrag", "both"):
        print(f"\n{BOLD}🔗 GraphRAG 평가 중...{RESET}")
        g_dataset = build_dataset(cases, vector_index, reranker, mode="graphrag")
        print("  RAGAS 채점 중...")
        g_result  = run_ragas(g_dataset, ragas_llm, ragas_emb)
        g_df = g_result.to_pandas()
        score_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        g_scores = {k: float(g_df[k].mean()) for k in score_keys if k in g_df.columns}

        print_scores("GraphRAG", g_scores, GREEN)

    # 비교 출력
    if v_scores and g_scores:
        print_comparison(v_scores, g_scores)


if __name__ == "__main__":
    main()