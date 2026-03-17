# ai-agent-aurix

AURIX 마이크로컨트롤러 기술 문서(PDF) 기반 GraphRAG Q&A 시스템.
한국어로 질문하면 영어 PDF를 검색해 한국어로 답변합니다.

**Vector DB (ChromaDB) + Knowledge Graph (Neo4j) = GraphRAG**

---

## 프로젝트 구조

```
ai-agent-aurix/
├── config.py                  # 모든 설정값 (경로, 모델, 파라미터)
│
├── core/
│   ├── models.py              # LLM, 임베딩, Reranker 초기화
│   ├── vector_store.py        # ChromaDB 연결
│   └── graph_store.py         # Neo4j 연결 + Cypher 조회
│
├── db/
│   ├── vectordb.py            # ETL — PDF → ChromaDB 인덱싱 + 질문/답변
│   └── knowledge_graph.py     # ETL — PDF → Neo4j Knowledge Graph 구축
│
├── rag/
│   └── graphrag.py            # GraphRAG 질문/답변 (Vector + Graph)
│
├── evaluate_ragas.py          # RAGAS 품질 평가
├── generate_test_cases.py     # 테스트 케이스 자동 생성
├── test_cases.json            # 평가 케이스
│
├── docs/                      # PDF 문서 폴더 (git 제외)
├── chroma_db/                 # Vector DB 저장소 (git 제외, 자동 생성)
├── .env                       # API 키 (git 제외)
├── .env.example               # API 키 양식
├── docker-compose.yml         # Neo4j + APOC
└── requirements.txt
```

---

## 환경 요구사항

- Python 3.10 또는 3.11 (**3.12 이상 사용 불가 — chromadb 호환 문제**)
- Docker (Neo4j 실행용)

---

## 최초 세팅

### 1. 가상환경 생성 및 패키지 설치

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. .env 설정

```
GEMINI_API_KEY=your_gemini_api_key

NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=aurix1234
```

### 3. Neo4j 실행

```bash
docker-compose up -d
```

### 4. docs 폴더에 PDF 추가

```
ai-agent-aurix/
└── docs/
    └── 분석할_문서.pdf
```

---

## 실행

### Vector DB 구축 및 질문/답변

```bash
python db/vectordb.py           # 인덱싱 + 대화형 질문/답변
python db/vectordb.py --reset   # ChromaDB 초기화 후 재인덱싱
```

### Knowledge Graph 구축

```bash
python db/knowledge_graph.py           # 구축 + 결과 확인
python db/knowledge_graph.py --reset   # Neo4j 초기화 후 재구축
```

### GraphRAG 질문/답변

```bash
python rag/graphrag.py             # 대화형 질문/답변
python rag/graphrag.py --compare   # Vector DB vs GraphRAG 비교
```

질문 입력 시 `--v` 옵션을 붙이면 청크 내용과 그래프 관계를 상세 출력합니다.

```
질문: EVADC란? --v
```

---

## 평가

### RAGAS 평가 실행

```bash
python evaluate_ragas.py                   # Vector DB vs GraphRAG 전체 비교
python evaluate_ragas.py --mode vector     # Vector DB 단독만
python evaluate_ragas.py --mode graphrag   # GraphRAG만
python evaluate_ragas.py --id TC-F01       # 특정 케이스만
```

### 테스트 케이스 유형

| 유형 | 용도 | ground_truth |
|---|---|---|
| `factual` | 정의, 수치, 구조 질문 | PDF 기반 정답 |
| `retrieval` | 여러 개념 간 관계·비교 질문 | PDF 기반 정답 |
| `hallucination` | 문서에 없는 내용 질문 | `"문서에서 찾을 수 없습니다"` 고정 |

### RAGAS 지표 해석

| 지표 | 의미 | 낮을 때 조치 |
|---|---|---|
| Faithfulness | 할루시네이션 억제 | system_prompt 강화 |
| Answer Relevancy | 질문 관련성 | Hybrid Search 도입 |
| Context Precision | 검색 청크 정확도 | chunk_size 조정, Reranker 튜닝 |
| Context Recall | 정보 커버리지 | chunk_overlap 확대 |

---

## 주요 파라미터 (config.py)

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `LLM_MODEL` | gemini-2.5-flash | LLM 모델 |
| `EMBED_MODEL` | BAAI/bge-m3 | 임베딩 모델 (한국어+영어) |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Reranker 모델 |
| `CHUNK_SIZE` | 256 | 청크 크기 (토큰) |
| `CHUNK_OVERLAP` | 32 | 청크 겹침 (토큰) |
| `SIMILARITY_TOP_K` | 10 | Reranker 입력 청크 수 |
| `RERANKER_TOP_N` | 5 | Reranker 출력 청크 수 |

---

## PDF 문서 교체 시

```bash
# 1. chroma_db 삭제
rmdir /s /q chroma_db

# 2. docs 폴더 교체 후 재인덱싱
python db/vectordb.py

# 3. Knowledge Graph 재구축 (문서 바뀌면 필요)
python db/knowledge_graph.py --reset
```

---

## 트러블슈팅

**`chromadb pydantic` 오류**  
Python 3.12 이상은 chromadb와 호환 안 됨.
```bash
py -3.10 -m venv .venv
```

**Neo4j APOC 오류**  
docker-compose.yml로 실행해야 APOC 플러그인이 자동으로 포함됩니다.
```bash
docker compose up -d   # docker run 대신 이걸 사용
```

**Gemini 429 오류**  
API 키 재발급 또는 지출 한도 확인. `gemini-2.5-flash` 모델명 확인.

---

## 진행 현황

```
✅ Phase 1  Vector DB 프로토타입 구축
✅ Phase 2  GraphRAG 프레임워크 구축 (Gemini + Neo4j)
✅ Phase 3  RAGAS 성능 평가 및 튜닝 (chunk_size 256 + Reranker)
⬜ Phase 4  Reader 교체 (표·다이어그램 처리 개선)
⬜ Phase 5  다문서 확장 및 프로덕션 전환
```