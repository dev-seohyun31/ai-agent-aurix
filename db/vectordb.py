"""
db/vectordb.py
PDF → ChromaDB 인덱싱 (ETL) + 대화형 질문/답변

사용법:
    python db/vectordb.py         # 인덱싱 후 대화형 질문/답변
    python db/vectordb.py --reset # ChromaDB 초기화 후 재인덱싱
"""

import sys
import os
import argparse
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

import config
from core.models import setup_all
from core.vector_store import get_vector_store_for_indexing, get_vector_index


def build_index() -> VectorStoreIndex:
    """PDF 로드 → 청킹 → 임베딩 → ChromaDB 저장"""
    collection, vector_store, storage_context = get_vector_store_for_indexing()

    Settings.chunk_size    = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    if collection.count() > 0:
        print(f"✅ 기존 인덱스 재사용 ({collection.count()}개 청크)")
        return VectorStoreIndex.from_vector_store(vector_store)

    print("📄 PDF 로딩 중...")
    documents = SimpleDirectoryReader(config.DOCS_DIR).load_data()
    print(f"✅ {len(documents)}페이지 로드 완료")

    print("🔍 인덱싱 중...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"✅ 인덱싱 완료 — 저장된 청크 수: {collection.count()}")
    return index


def interactive(index: VectorStoreIndex, reranker):
    """대화형 질문/답변 루프"""
    from llama_index.core import Settings as S
    query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K,
        node_postprocessors=[reranker],
        response_mode="tree_summarize",
    )

    print("\n💬 질문을 입력하세요 (종료: q)\n")
    while True:
        question = input("질문: ").strip()
        if question.lower() == "q":
            break

        response = query_engine.query(question)
        print(f"\n📝 답변:\n{response}\n")

        print("📚 참조 출처:")
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "알 수 없음")
            score = round(node.score, 3) if node.score else "-"
            print(f"  - {fname} (유사도: {score})")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Vector DB 구축 및 질문/답변")
    parser.add_argument("--reset", action="store_true", help="ChromaDB 초기화 후 재인덱싱")
    args = parser.parse_args()

    if args.reset and os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
        print("🗑️  ChromaDB 초기화 완료")

    # 모델 초기화
    reranker = setup_all()

    # 인덱싱
    index = build_index()

    # 대화형 질문/답변
    interactive(index, reranker)


if __name__ == "__main__":
    main()