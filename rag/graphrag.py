"""
graphrag.py
Vector DB + Knowledge Graph를 결합한 GraphRAG 질문/답변 시스템

사용법:
    python graphrag.py                  # 대화형 질문/답변
    python graphrag.py --compare        # Vector DB 단독 vs GraphRAG 비교
"""

import os
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

load_dotenv()

# ── 설정값 ────────────────────────────────────────────
NEO4J_URL      = os.environ["NEO4J_URL"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
CHROMA_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
CHROMA_COLLECTION = "research_docs"

# ── 색상 출력 ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


# ── 1. 모델 설정 ──────────────────────────────────────
def setup_models():
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        system_prompt=(
            "You are a technical document assistant for AURIX microcontroller. "
            "ALWAYS respond in Korean only. "
            "NEVER mix other languages into your response. "
            "Base your answer ONLY on the provided document chunks and graph relationships. "
            "If the answer is not found, say '문서에서 찾을 수 없습니다'."
        )
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    print("✅ 모델 설정 완료 (Gemini + bge-m3)")


# ── 2. Vector DB 연결 ─────────────────────────────────
def setup_vector_db():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

    if collection.count() == 0:
        print(f"{RED}❌ chroma_db가 비어있습니다. db/vectordb.py를 먼저 실행하세요.{RESET}")
        exit(1)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print(f"✅ Vector DB 연결 완료 ({collection.count()}개 청크)")
    return index


# ── 3. Knowledge Graph 연결 ───────────────────────────
def setup_knowledge_graph():
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        node_count = session.run(
            "MATCH (n) RETURN count(n) AS cnt"
        ).single()["cnt"]
    driver.close()

    if node_count == 0:
        print(f"{RED}❌ Neo4j가 비어있습니다. db/knowledge_graph.py를 먼저 실행하세요.{RESET}")
        exit(1)

    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
    )
    print(f"✅ Knowledge Graph 연결 완료 ({node_count}개 노드)")
    return kg_index


# ── 4. Knowledge Graph 직접 조회 ──────────────────────
def query_graph(question: str, top_k: int = 10) -> str:
    """
    질문에서 키워드를 추출해 Neo4j에서 관련 관계를 직접 조회.
    LLM 없이 Cypher로 직접 탐색하여 정확도 향상.
    """
    # 질문에서 핵심 키워드 추출 (영어 기술 용어 위주)
    keywords = []
    question_lower = question.lower()

    # 자주 등장하는 AURIX 기술 용어 매핑
    keyword_map = {
        "evadc": "evadc",
        "클러스터": "cluster",
        "cluster": "cluster",
        "primary": "primary",
        "secondary": "secondary",
        "fast compare": "fast compare",
        "sar": "sar",
        "멀티플렉서": "multiplexer",
        "multiplexer": "multiplexer",
        "큐": "queue",
        "queue": "queue",
        "레지스터": "register",
        "register": "register",
        "변환": "converter",
        "converter": "converter",
        "신호": "signal",
        "signal": "signal",
    }

    for kor, eng in keyword_map.items():
        if kor in question_lower and eng not in keywords:
            keywords.append(eng)

    if not keywords:
        # 키워드 없으면 질문 단어 그대로 사용
        keywords = [
            w for w in question_lower.split()
            if len(w) > 2 and w not in ["이야", "뭐야", "뭐", "어떤", "알려줘", "설명해줘", "이란", "이란?", "은?", "는?"]
        ]

    if not keywords:
        return ""

    # Neo4j Cypher 직접 조회
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    results = []

    with driver.session() as session:
        for keyword in keywords[:3]:  # 상위 3개 키워드만
            rows = session.run("""
                MATCH (a)-[r]->(b)
                WHERE toLower(a.id) CONTAINS $keyword
                   OR toLower(b.id) CONTAINS $keyword
                RETURN a.id AS from_node, type(r) AS relation, b.id AS to_node
                LIMIT $limit
            """, keyword=keyword, limit=top_k)

            for row in rows:
                triple = f"{row['from_node']} -[{row['relation']}]→ {row['to_node']}"
                if triple not in results:
                    results.append(triple)

    driver.close()
    return "\n".join(results) if results else ""


# ── 5. GraphRAG 통합 쿼리 ─────────────────────────────
def graphrag_query(question: str, vector_index, verbose: bool = False) -> dict:
    """
    Vector DB + Knowledge Graph 결합 쿼리

    흐름:
    1. Vector DB → 유사 청크 검색
    2. Knowledge Graph → 관련 관계 직접 조회
    3. 둘을 합쳐서 LLM에 전달
    4. 출처 기반 답변 생성
    """

    # ── STEP 1: Vector DB 검색 ──────────────────────
    retriever = vector_index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(question)

    chunks = []
    sources = []
    scores = []
    for node in nodes:
        chunks.append(node.node.text)
        fname = node.node.metadata.get("file_name", "알 수 없음")
        score = round(node.score, 3) if node.score else 0
        sources.append(fname)
        scores.append(score)

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
        print(f"\n{BLUE}── Vector DB 검색 결과 ({len(chunks)}개 청크) ──{RESET}")
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            print(f"  [{i+1}] 유사도: {score} | {chunk[:100]}...")
        print(f"\n{BLUE}── Knowledge Graph 조회 결과 ──{RESET}")
        if graph_context:
            print(graph_context)
        else:
            print("  (관련 관계 없음)")

    # ── STEP 4: LLM 호출 ──────────────────────────
    prompt = f"""아래 정보만 근거로 질문에 답하세요.
문서에 없는 내용은 절대 추가하지 마세요.

{full_context}

질문: {question}
"""

    llm_response = Settings.llm.complete(prompt)
    answer = str(llm_response).strip()

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "scores": scores,
        "graph_used": bool(graph_context),
        "graph_triples": graph_context,
    }


# ── 6. Vector DB 단독 쿼리 (비교용) ──────────────────
def vector_only_query(question: str, vector_index) -> dict:
    query_engine = vector_index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize"
    )
    response = query_engine.query(question)
    sources = list(set(
        n.metadata.get("file_name", "알 수 없음")
        for n in response.source_nodes
    ))
    scores = [round(n.score, 3) for n in response.source_nodes if n.score]

    return {
        "answer": str(response).strip(),
        "sources": sources,
        "scores": scores,
    }


# ── 7. 비교 모드 ─────────────────────────────────────
def compare_mode(vector_index):
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

        # Vector DB 단독
        v_result = vector_only_query(question, vector_index)
        print(f"{BLUE}[Vector DB 단독]{RESET}")
        print(f"  답변: {v_result['answer'][:200]}")
        print(f"  출처: {', '.join(v_result['sources'])}")

        print()

        # GraphRAG
        g_result = graphrag_query(question, vector_index)
        print(f"{GREEN}[GraphRAG (Vector + Graph)]{RESET}")
        print(f"  답변: {g_result['answer'][:200]}")
        print(f"  출처: {', '.join(g_result['sources'])}")
        print(f"  그래프 활용: {'✅' if g_result['graph_used'] else '❌'}")
        if g_result['graph_triples']:
            print(f"  참조 관계:")
            for triple in g_result['graph_triples'].split('\n')[:5]:
                print(f"    {triple}")

        print(f"\n{'='*60}\n")


# ── 8. 대화형 모드 ────────────────────────────────────
def interactive_mode(vector_index):
    print(f"\n{BOLD}💬 GraphRAG 질문/답변 (종료: q){RESET}")
    print(f"  --v 옵션: 상세 출력  예) EVADC란? --v\n")

    while True:
        user_input = input("질문: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            break

        verbose = "--v" in user_input
        question = user_input.replace("--v", "").strip()

        result = graphrag_query(question, vector_index, verbose=verbose)

        print(f"\n{BOLD}📝 답변:{RESET}")
        print(result["answer"])

        print(f"\n{BOLD}📚 참조 출처:{RESET}")
        for src, score in zip(result["sources"], result["scores"]):
            print(f"  - {src} (유사도: {score})")

        if result["graph_used"]:
            print(f"{GREEN}🔗 Knowledge Graph 활용됨{RESET}")
        else:
            print(f"{YELLOW}⚠️  Knowledge Graph 관련 정보 없음 (Vector DB만 사용){RESET}")

        print("-" * 50)


# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GraphRAG 질문/답변")
    parser.add_argument(
        "--compare", action="store_true",
        help="Vector DB 단독 vs GraphRAG 비교 모드"
    )
    args = parser.parse_args()

    setup_models()
    vector_index = setup_vector_db()
    setup_knowledge_graph()  # 연결 확인용

    if args.compare:
        compare_mode(vector_index)
    else:
        interactive_mode(vector_index)


if __name__ == "__main__":
    main()