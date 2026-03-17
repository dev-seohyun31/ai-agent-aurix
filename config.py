"""
config.py
프로젝트 전체 공통 설정값 — 경로, 모델명, DB 접속 정보
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── 경로 ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR   = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── ChromaDB ──────────────────────────────────────────
CHROMA_COLLECTION = "research_docs"

# ── Neo4j ─────────────────────────────────────────────
NEO4J_URL      = os.environ["NEO4J_URL"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# ── LLM ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LLM_MODEL      = "gemini-2.5-flash"

# ── 임베딩 ────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-m3"

# ── Reranker ──────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_N = 5

# ── 청킹 ──────────────────────────────────────────────
CHUNK_SIZE    = 256
CHUNK_OVERLAP = 32

# ── 검색 ──────────────────────────────────────────────
SIMILARITY_TOP_K = 10

# ── Knowledge Graph 추출 프롬프트 ─────────────────────
KG_EXTRACT_PROMPT = """\
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

# ── AURIX 기술 용어 키워드 매핑 (한국어 → 영어) ────────
KEYWORD_MAP = {
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
    "캘리브레이션": "calibration",
    "calibration": "calibration",
    "트리거": "trigger",
    "trigger": "trigger",
    "gtm": "gtm",
    "dma": "dma",
    "arbitration": "arbitration",
    "중재": "arbitration",
    "전압": "voltage",
    "voltage": "voltage",
    "gpio": "gpio",
    "adc": "adc",
    "클록": "clock",
    "clock": "clock",
}

STOP_WORDS = {
    "이야", "뭐야", "뭐", "어떤", "알려줘", "설명해줘",
    "이란", "이란?", "은?", "는?", "무엇", "어떻게", "왜",
    "있나요", "있어요", "인가요", "됩니까", "할까요",
}