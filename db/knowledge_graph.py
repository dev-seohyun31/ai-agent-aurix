"""
knowledge_graph.py
PDF 문서에서 엔티티·관계를 추출하여 Neo4j Knowledge Graph를 구축합니다.

사용법:
    python knowledge_graph.py          # 구축 + 결과 확인
    python knowledge_graph.py --reset  # Neo4j 초기화 후 재구축
"""

import os
import argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


# Configurations
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
DOCS_DIR = "../docs"

# Prompt
EXTRACT_PROMPT = """\
Extract up to {max_knowledge_triplets} knowledge triplets from the text below.
Each triplet must follow this strict schema:

ENTITY TYPES:
- Module: Top-level hardware module (e.g. EVADC)
- Cluster: Sub-cluster inside a module (e.g. Primary Converter Cluster)
- Component: Hardware component (e.g. Multiplexer, Queue, Register)
- Principle: Operating principle (e.g. SAR, Successive Approximation)
- Spec: Numerical specification (e.g. 0.5us, 8-stage)
- Signal: Input/output signal

RELATIONSHIP TYPES (use ONLY these):
- CONTAINS: Module contains Cluster, or Cluster contains Component
- HAS_SPEC: Cluster or Component has a Spec
- USES: Module or Cluster uses a Principle
- OUTPUTS: Module or Cluster outputs a Signal
- PART_OF: Component is part of a Cluster

RULES:
- Use ONLY the relationship types listed above
- Do NOT invent new relationship types
- Do NOT extract file paths, page numbers, or version numbers
- Do NOT extract relationships about dates or authors
- Entity names must be concise (3 words max)

Text: {text}

Triplets (format: subject | relationship | object):
"""

def setup_models():
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        system_prompt=(
            "You are a precise technical knowledge extraction assistant. "
            "Extract only factual relationships explicitly stated in the text. "
            "Never invent or infer relationships not present in the text. "
            "Follow the schema strictly."
        )
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    print("✅ 모델 설정 완료 (Gemini + bge-m3)")

def load_existing_graph(storage_context):
    """기존 Neo4j 그래프 재사용"""
    return KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
    )


def build_graph(storage_context):
    """PDF 로드 후 Knowledge Graph 구축"""
    print("📄 PDF 로딩 중...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"✅ {len(documents)}페이지 로드 완료")

    print("🔍 엔티티·관계 추출 중... (Gemini API 호출)")
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=5,
        include_embeddings=True,
        kg_triple_extract_template=EXTRACT_PROMPT,
        show_progress=True,
    )
    print("✅ Knowledge Graph 구축 완료")
    return index


def print_graph_summary(driver):
    """구축 결과 요약 출력"""
    with driver.session() as session:

        node_count = session.run(
            "MATCH (n) RETURN count(n) AS cnt"
        ).single()["cnt"]
        rel_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS cnt"
        ).single()["cnt"]
        print(f"\n📦 노드 수: {node_count} / 관계 수: {rel_count}")

        # 관계 타입 분포
        print("\n📊 관계 타입 분포:")
        rel_types = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC
            LIMIT 10
        """)
        for r in rel_types:
            print(f"  [{r['rel_type']}] {r['cnt']}개")

        # 노드 샘플
        print("\n📋 노드 샘플 (상위 10개):")
        nodes = session.run("""
            MATCH (n)
            WHERE n.id IS NOT NULL
            RETURN n.id AS node_id
            LIMIT 10
        """)
        for r in nodes:
            print(f"  - {r['node_id']}")

        # 핵심 관계 확인 (스키마 기반)
        print("\n🔗 핵심 관계 샘플 (CONTAINS / USES / HAS_SPEC):")
        core_rels = session.run("""
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['CONTAINS', 'USES', 'HAS_SPEC', 'OUTPUTS', 'PART_OF']
            RETURN a.id AS from, type(r) AS rel, b.id AS to
            LIMIT 20
        """)
        rows = list(core_rels)
        if rows:
            for r in rows:
                print(f"  {r['from']} -[{r['rel']}]→ {r['to']}")
        else:
            print("  ⚠️ 핵심 관계 없음 — 스키마 준수율 확인 필요")

        # EVADC 관련 관계
        print("\n🔍 EVADC 관련 관계:")
        evadc_rels = session.run("""
            MATCH (a)-[r]->(b)
            WHERE toLower(a.id) CONTAINS 'evadc'
               OR toLower(b.id) CONTAINS 'evadc'
            RETURN a.id AS from, type(r) AS rel, b.id AS to
            LIMIT 15
        """)
        rows = list(evadc_rels)
        if rows:
            for r in rows:
                print(f"  {r['from']} -[{r['rel']}]→ {r['to']}")
        else:
            print("  ⚠️ EVADC 관련 관계 없음")


def test_query(index):
    """구축된 그래프로 테스트 쿼리"""
    print("\n🧪 쿼리 테스트")
    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        verbose=False,
    )
    questions = [
        "EVADC는 어떤 클러스터를 포함하고 있어?",
        "Primary Converter Cluster의 스펙은?",
    ]
    for q in questions:
        print(f"\n질문: {q}")
        response = query_engine.query(q)
        print(f"답변: {response}")


def reset_neo4j(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🗑️  Neo4j 초기화 완료")


def get_node_count(driver):
    with driver.session() as session:
        return session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="Neo4j 초기화 후 재구축"
    )
    args = parser.parse_args()

    # 1. 모델 설정
    setup_models()

    # 2. Neo4j 연결
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph_store = Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    if args.reset:
        reset_neo4j(driver)

    # 3. 인덱싱 또는 재사용
    node_count = get_node_count(driver)
    if node_count > 0:
        print(f"✅ 기존 Knowledge Graph 재사용 ({node_count}개 노드)")
        index = load_existing_graph(storage_context)
    else:
        index = build_graph(storage_context)

    # 4. 결과 확인 & 쿼리 테스트
    print_graph_summary(driver)
    test_query(index)

    driver.close()

    return


if __name__ == "__main__":
    main()