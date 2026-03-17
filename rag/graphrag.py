"""
rag/graphrag.py
Vector DB + Knowledge Graph 결합 GraphRAG 질문/답변 시스템

사용법:
    python rag/graphrag.py                  # 대화형 질문/답변
    python rag/graphrag.py --compare        # Vector DB 단독 vs GraphRAG 비교
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

import config
from core.models import setup_all
from core.vector_store import get_vector_index
from core.graph_store import get_kg_index, query_graph

# ── 색상 출력 ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def graphrag_query(question: str, vector_index, reranker: FlagEmbeddingReranker,
                   verbose: bool = False) -> dict:
    """
    Vector DB + Knowledge Graph 결합 쿼리
    흐름: Vector(top_k=10) → Reranker(top_n=5) + Graph → LLM → 답변
    """

    # ── STEP 1: Vector DB + Reranker ──────────────
    retriever      = vector_index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
    nodes          = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(
        nodes, query_bundle=QueryBundle(question)
    )

    chunks  = [n.node.text for n in reranked_nodes]
    sources = [n.node.metadata.get("file_name", "알 수 없음") for n in reranked_nodes]
    scores  = [round(n.score, 3) if n.score else 0 for n in reranked_nodes]

    # ── STEP 2: Knowledge Graph 조회 ──────────────
    graph_context = query_graph(question)

    # ── STEP 3: Context 조립 ──────────────────────
    context_parts = []
    if chunks:
        context_parts.append("[문서 내용]")
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[청크 {i+1}]\n{chunk}")
    if graph_context:
        context_parts.append(f"\n[관계 정보 (Knowledge Graph)]\n{graph_context}")

    full_context = "\n\n".join(context_parts)

    if verbose:
        print(f"\n{BLUE}── Vector DB ({len(chunks)}개 청크, Reranker 적용) ──{RESET}")
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            print(f"  [{i+1}] 유사도: {score} | {chunk[:100]}...")
        print(f"\n{BLUE}── Knowledge Graph ──{RESET}")
        print(graph_context if graph_context else "  (관련 관계 없음)")

    # ── STEP 4: LLM 호출 ──────────────────────────
    prompt = (
        "아래 정보만 근거로 질문에 답하세요.\n"
        "문서에 없는 내용은 절대 추가하지 마세요.\n\n"
        f"{full_context}\n\n질문: {question}"
    )
    answer = str(Settings.llm.complete(prompt)).strip()

    return {
        "answer":        answer,
        "sources":       list(set(sources)),
        "scores":        scores,
        "graph_used":    bool(graph_context),
        "graph_triples": graph_context,
    }


def vector_only_query(question: str, vector_index,
                      reranker: FlagEmbeddingReranker) -> dict:
    """Vector DB 단독 쿼리 (비교용)"""
    query_engine = vector_index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K,
        node_postprocessors=[reranker],
        response_mode="tree_summarize",
    )
    response = query_engine.query(question)
    sources  = list(set(n.metadata.get("file_name", "알 수 없음") for n in response.source_nodes))
    scores   = [round(n.score, 3) for n in response.source_nodes if n.score]

    return {
        "answer":  str(response).strip(),
        "sources": sources,
        "scores":  scores,
    }


def compare_mode(vector_index, reranker):
    test_questions = [
        "EVADC는 무엇인가?",
        "EVADC Primary Converter Cluster의 스펙은?",
        "EVADC는 어떤 클러스터를 포함하고 있어?",
    ]

    print(f"\n{BOLD}{'='*60}")
    print("📊 Vector DB 단독 vs GraphRAG 비교")
    print(f"{'='*60}{RESET}\n")

    for question in test_questions:
        print(f"{BOLD}질문: {question}{RESET}")
        print(f"{'─'*60}")

        v = vector_only_query(question, vector_index, reranker)
        print(f"{BLUE}[Vector DB]{RESET}")
        print(f"  답변: {v['answer'][:300]}")
        print(f"  출처: {', '.join(v['sources'])}")
        print()

        g = graphrag_query(question, vector_index, reranker)
        print(f"{GREEN}[GraphRAG]{RESET}")
        print(f"  답변: {g['answer'][:300]}")
        print(f"  그래프 활용: {'✅' if g['graph_used'] else '❌'}")
        if g['graph_triples']:
            for triple in g['graph_triples'].split('\n')[:5]:
                print(f"    {triple}")
        print(f"\n{'='*60}\n")


def interactive_mode(vector_index, reranker):
    print(f"\n{BOLD}💬 GraphRAG 질문/답변 (종료: q){RESET}")
    print(f"  --v 옵션: 상세 출력  예) EVADC란? --v\n")

    while True:
        user_input = input("질문: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            break

        verbose  = "--v" in user_input
        question = user_input.replace("--v", "").strip()
        result   = graphrag_query(question, vector_index, reranker, verbose=verbose)

        print(f"\n{BOLD}📝 답변:{RESET}")
        print(result["answer"])
        print(f"\n{BOLD}📚 참조 출처:{RESET}")
        for src, score in zip(result["sources"], result["scores"]):
            print(f"  - {src} (유사도: {score})")
        if result["graph_used"]:
            print(f"{GREEN}🔗 Knowledge Graph 활용됨{RESET}")
        else:
            print(f"{YELLOW}⚠️  Knowledge Graph 관련 정보 없음{RESET}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="GraphRAG 질문/답변")
    parser.add_argument("--compare", action="store_true", help="Vector DB vs GraphRAG 비교")
    args = parser.parse_args()

    reranker     = setup_all()
    vector_index = get_vector_index()
    get_kg_index()  # 연결 확인

    if args.compare:
        compare_mode(vector_index, reranker)
    else:
        interactive_mode(vector_index, reranker)


if __name__ == "__main__":
    main()