"""
core/models.py
LLM, 임베딩, Reranker 초기화 — 프로젝트 전체에서 한 번만 호출
"""

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

import config

# ── 색상 출력 ─────────────────────────────────────────
GREEN = "\033[92m"
RESET = "\033[0m"


def setup_llm(system_prompt: str | None = None):
    """
    Gemini LLM 초기화 및 LlamaIndex 전역 설정
    system_prompt가 없으면 기본 RAG용 프롬프트 사용
    """
    if system_prompt is None:
        system_prompt = (
            "You are a technical document assistant for AURIX microcontroller. "
            "ALWAYS respond in Korean only. "
            "NEVER mix other languages into your response. "
            "Base your answer ONLY on the provided document chunks and graph relationships. "
            "If the answer is not found, say '문서에서 찾을 수 없습니다'."
        )

    Settings.llm = GoogleGenAI(
        model=config.LLM_MODEL,
        api_key=config.GEMINI_API_KEY,
        system_prompt=system_prompt,
    )
    print(f"{GREEN}✅ LLM 설정 완료 ({config.LLM_MODEL}){RESET}")


def setup_embedding():
    """bge-m3 임베딩 초기화 및 LlamaIndex 전역 설정"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL
    )
    print(f"{GREEN}✅ 임베딩 설정 완료 ({config.EMBED_MODEL}){RESET}")


def setup_reranker() -> FlagEmbeddingReranker:
    """Reranker 초기화 — 한 번만 로드해서 재사용"""
    reranker = FlagEmbeddingReranker(
        model=config.RERANKER_MODEL,
        top_n=config.RERANKER_TOP_N,
    )
    print(f"{GREEN}✅ Reranker 로드 완료 ({config.RERANKER_MODEL}){RESET}")
    return reranker


def setup_all(system_prompt: str | None = None) -> FlagEmbeddingReranker:
    """
    LLM + 임베딩 + Reranker 한꺼번에 초기화
    대부분의 진입점(graphrag.py, evaluate_ragas.py 등)에서 이걸 호출
    """
    setup_llm(system_prompt)
    setup_embedding()
    reranker = setup_reranker()
    return reranker


def setup_kg_llm():
    """
    Knowledge Graph 구축 전용 LLM 설정
    일반 RAG와 system_prompt가 다름
    """
    setup_llm(
        system_prompt=(
            "You are a precise technical knowledge extraction assistant. "
            "Extract only factual relationships explicitly stated in the text. "
            "Never invent or infer relationships not present in the text. "
            "Follow the schema strictly."
        )
    )
    setup_embedding()