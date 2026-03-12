# ai-agent-aurix

AURIX 마이크로컨트롤러 기술 문서(PDF) 기반 GraphRAG Q&A 시스템.
한국어로 질문하면 영어 PDF를 검색해 한국어로 답변합니다.

**Vector DB (ChromaDB) + Knowledge Graph (Neo4j) = GraphRAG**

---

## 구조

```
ai-agent-aurix/
├── db/
│   ├── vectordb.py          # ChromaDB 구축
│   └── knowledge_graph.py   # Neo4j Knowledge Graph 구축
├── rag/
│   └── graphrag.py          # GraphRAG Q&A (Vector DB + Knowledge Graph)
├── evaluate.py              # Vector DB 평가
├── evaluate_graphrag.py     # GraphRAG 평가
├── test_cases.json          # 평가 테스트 케이스
├── docs/                    # PDF 문서 (git 제외)
├── chroma_db/               # Vector DB 저장소 (자동 생성)
├── docker-compose.yml       # Neo4j
├── .env                     # API 키 (git 제외)
└── requirements.txt
```

---

## 기술 스택

| 역할 | 사용 기술 |
|---|---|
| LLM | Gemini 2.5 Flash |
| 임베딩 | BAAI/bge-m3 (HuggingFace) |
| Vector DB | ChromaDB |
| Knowledge Graph | Neo4j 5.18 (Docker) |
| 프레임워크 | LlamaIndex |

---

## 세팅 (최초 1회)

### 1. Python 환경

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Python 3.10 또는 3.11 필수 (3.12+ 비호환)

### 2. 환경 변수 설정

`.env` 파일 생성:

```
GEMINI_API_KEY=your_key
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=aurix1234
```

### 3. Neo4j 실행

```bash
docker-compose up -d
```

### 4. PDF 추가

```
docs/
└── 32_EVADC.pdf   ← 분석할 PDF
```

### 5. DB 구축

```bash
python db/vectordb.py         # Vector DB 인덱싱
python db/knowledge_graph.py  # Knowledge Graph 구축
```

---

## 실행

```bash
python rag/graphrag.py           # 대화형 Q&A
python rag/graphrag.py --compare # Vector DB 단독 vs GraphRAG 비교
```

```
질문: EVADC란 무엇인가?
질문: EVADC 클러스터 구조는? --v   # 상세 출력
```

---

## 평가

```bash
python evaluate.py                          # Vector DB 평가
python evaluate_graphrag.py                 # GraphRAG 평가
python evaluate_graphrag.py --compare       # 두 방식 비교
python evaluate_graphrag.py --type factual  # 유형별 (factual / hallucination / retrieval)
python evaluate_graphrag.py --id TC-001     # 특정 케이스
```
