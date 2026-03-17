"""
db/knowledge_graph.py
PDF → Neo4j Knowledge Graph 구축 (ETL)

사용법:
    python db/knowledge_graph.py          # 구축 + 결과 확인
    python db/knowledge_graph.py --reset  # Neo4j 초기화 후 재구축
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

import config
from core.models import setup_kg_llm
from core.graph_store import (
    get_neo4j_driver, get_graph_storage_context,
    get_node_count, reset_graph
)


def build_graph():
    """PDF 로드 → 엔티티·관계 추출 → Neo4j 저장"""
    print("📄 PDF 로딩 중...")
    documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
    print(f"✅ {len(documents)}페이지 로드 완료")

    print("🔍 엔티티·관계 추출 중... (Gemini API 호출)")
    storage_context = get_graph_storage_context()
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=5,
        include_embeddings=False,
        kg_triple_extract_template=config.KG_EXTRACT_PROMPT,
        show_progress=True,
    )
    print("✅ Knowledge Graph 구축 완료")
    return index


def load_existing_graph():
    """기존 Neo4j 그래프 재사용"""
    storage_context = get_graph_storage_context()
    return KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
    )


def print_graph_summary():
    """구축 결과 요약 출력"""
    driver = get_neo4j_driver()
    with driver.session() as session:

        node_count = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
        print(f"\n📦 노드 수: {node_count} / 관계 수: {rel_count}")

        print("\n📊 관계 타입 분포 (상위 10개):")
        for r in session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC LIMIT 10
        """):
            print(f"  [{r['rel_type']}] {r['cnt']}개")

        print("\n📋 노드 샘플 (상위 10개):")
        for r in session.run("""
            MATCH (n) WHERE n.id IS NOT NULL
            RETURN n.id AS node_id LIMIT 10
        """):
            print(f"  - {r['node_id']}")

        print("\n🔗 핵심 관계 샘플 (CONTAINS / USES / HAS_SPEC):")
        rows = list(session.run("""
            MATCH (a)-[r]->(b)
            WHERE type(r) IN ['CONTAINS', 'USES', 'HAS_SPEC', 'OUTPUTS', 'PART_OF']
            RETURN a.id AS from, type(r) AS rel, b.id AS to LIMIT 20
        """))
        if rows:
            for r in rows:
                print(f"  {r['from']} -[{r['rel']}]→ {r['to']}")
        else:
            print("  ⚠️ 핵심 관계 없음 — 스키마 준수율 확인 필요")

    driver.close()


def test_query(index):
    """구축된 그래프로 테스트 쿼리"""
    print("\n🧪 쿼리 테스트")
    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        verbose=False,
    )
    for q in ["EVADC는 어떤 클러스터를 포함하고 있어?", "Primary Converter Cluster의 스펙은?"]:
        print(f"\n질문: {q}")
        print(f"답변: {query_engine.query(q)}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph 구축")
    parser.add_argument("--reset", action="store_true", help="Neo4j 초기화 후 재구축")
    args = parser.parse_args()

    # 모델 초기화 (KG 전용 system_prompt)
    setup_kg_llm()

    if args.reset:
        reset_graph()

    # 인덱싱 또는 재사용
    node_count = get_node_count()
    if node_count > 0:
        print(f"✅ 기존 Knowledge Graph 재사용 ({node_count}개 노드)")
        index = load_existing_graph()
    else:
        index = build_graph()

    # 결과 확인
    print_graph_summary()
    test_query(index)


if __name__ == "__main__":
    main()