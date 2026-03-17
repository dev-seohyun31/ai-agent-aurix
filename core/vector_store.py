"""
core/vector_store.py
ChromaDB 연결 및 VectorStoreIndex 반환 — 프로젝트 전체 공통 사용
"""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_vector_index() -> VectorStoreIndex:
    """
    ChromaDB에 연결하고 VectorStoreIndex 반환
    DB가 비어있으면 에러 출력 후 종료
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)

    if collection.count() == 0:
        print(f"{RED}❌ chroma_db가 비어있습니다. db/vectordb.py를 먼저 실행하세요.{RESET}")
        exit(1)

    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index           = VectorStoreIndex.from_vector_store(vector_store)

    print(f"{GREEN}✅ Vector DB 연결 완료 ({collection.count()}개 청크){RESET}")
    return index


def get_vector_store_for_indexing():
    """
    인덱싱(ETL) 전용 — collection과 storage_context 함께 반환
    vectordb.py에서만 사용
    """
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection    = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return collection, vector_store, storage_context