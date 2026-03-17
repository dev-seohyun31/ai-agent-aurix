"""
core/graph_store.py
Neo4j 연결 및 Cypher 조회 — 프로젝트 전체 공통 사용
"""

from neo4j import GraphDatabase
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext

import config

RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_neo4j_driver():
    """Neo4j 드라이버 반환 — 사용 후 반드시 driver.close() 호출"""
    return GraphDatabase.driver(
        config.NEO4J_URL,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )


def get_graph_storage_context() -> StorageContext:
    """Neo4j 기반 StorageContext 반환"""
    graph_store = Neo4jGraphStore(
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URL,
        database="neo4j",
    )
    return StorageContext.from_defaults(graph_store=graph_store)


def get_kg_index() -> KnowledgeGraphIndex:
    """
    기존 Neo4j 그래프 재사용 — 쿼리 전용
    DB가 비어있으면 에러 출력 후 종료
    """
    driver = get_neo4j_driver()
    with driver.session() as session:
        node_count = session.run(
            "MATCH (n) RETURN count(n) AS cnt"
        ).single()["cnt"]
    driver.close()

    if node_count == 0:
        print(f"{RED}❌ Neo4j가 비어있습니다. db/knowledge_graph.py를 먼저 실행하세요.{RESET}")
        exit(1)

    storage_context = get_graph_storage_context()
    kg_index = KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
    )
    print(f"{GREEN}✅ Knowledge Graph 연결 완료 ({node_count}개 노드){RESET}")
    return kg_index


def query_graph(question: str, top_k: int = 10) -> str:
    """
    질문 키워드 기반 Neo4j Cypher 직접 조회
    LLM 없이 빠르게 관련 관계 반환
    """
    keywords       = []
    question_lower = question.lower()

    for kor, eng in config.KEYWORD_MAP.items():
        if kor in question_lower and eng not in keywords:
            keywords.append(eng)

    if not keywords:
        keywords = [
            w for w in question_lower.split()
            if len(w) > 2 and w not in config.STOP_WORDS
        ]

    if not keywords:
        return ""

    driver  = get_neo4j_driver()
    results = []

    with driver.session() as session:
        for keyword in keywords[:3]:
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


def get_node_count() -> int:
    driver = get_neo4j_driver()
    with driver.session() as session:
        count = session.run(
            "MATCH (n) RETURN count(n) AS cnt"
        ).single()["cnt"]
    driver.close()
    return count


def reset_graph():
    """Neo4j 전체 초기화"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("🗑️  Neo4j 초기화 완료")