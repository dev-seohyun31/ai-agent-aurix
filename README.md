# ai-agent-aurix

AURIX 기술 문서 기반 AI Agent의 프로토타입 입니다.
한국어로 질문하면, 영어로 된 PDF 문서에서 검색하여 한국어 답변을 하는 RAG 시스템입니다.

---

## 프로젝트 구조

```
ai-agent-aurix/
├── vectordb.py        # Vector DB 구축 및 RAG 질문/답변
├── evaluate.py        # 검색 품질 및 할루시네이션 평가
├── test_cases.json    # 평가용 테스트 케이스
├── docs/              # PDF 문서 폴더 (git 제외)
├── chroma_db/         # Vector DB 저장소 (git 제외, 자동 생성)
├── .venv/             # 가상환경 (git 제외)
├── .env               # API 키 (git 제외)
├── .env.example       # API 키 양식 (git 포함)
├── requirements.txt   # 패키지 목록
└── README.md
```

---

## 환경 요구사항

- Python 3.10 또는 3.11 (**3.12 이상 사용 불가 — chromadb 호환 문제**)
- Ollama 설치 필요 ([https://ollama.com/download](https://ollama.com/download))

---

## 최초 세팅 (처음 한 번만)

### 1. Python 버전 확인

```bash
python --version
# Python 3.10.x 또는 3.11.x 여야 함
```

### 2. 가상환경 생성 및 활성화

```bash
# 생성
py -3.10 -m venv .venv

# 활성화 (Windows)
.venv\Scripts\activate

# 활성화 확인 — 터미널 앞에 (.venv) 표시
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. Ollama 모델 다운로드

```bash
ollama pull llama3.2
# 약 2GB 다운로드 (최초 1회만)
```

### 5. docs 폴더에 PDF 넣기

```
ai-agent-aurix/
└── docs/
    └── 분석할_문서.pdf   ← 여기에 PDF 추가
```

---

## 실행

### 매번 실행 시

```bash
# 1. 프로젝트 폴더로 이동
cd C:\Users\DBC141\develop\ai-agent-aurix

# 2. 가상환경 활성화
.venv\Scripts\activate

# 3. 실행
python vectordb.py
```

### 처음 실행 시 (인덱싱)

PDF가 처음 로딩될 때 인덱싱이 자동 실행됩니다.

```
📄 PDF 로딩 중...
✅ N페이지 로드 완료
🔍 인덱싱 중...
✅ 인덱싱 완료          ← bge-m3 모델 특성상 시간 소요
📦 저장된 청크 수: N
```

이후 실행부터는 기존 인덱스를 재사용하여 바로 질문 가능합니다.

```
✅ 기존 인덱스 재사용 (N개 청크)
```

### 질문/답변

```
💬 질문을 입력하세요 (종료: q)

질문: EVADC란 무엇인가?

📝 답변:
...

📚 참조 출처:
  - 32_EVADC.pdf (유사도: 0.82)
  - 32_EVADC.pdf (유사도: 0.79)
--------------------------------------------------
```

종료하려면 `q` 입력.

---

## 평가 실행 (evaluate.py)

`vectordb.py`로 인덱싱이 완료된 상태에서 실행합니다.

### 기본 실행 — 전체 테스트

```bash
python evaluate.py
```

```
🧪 RAG 평가 시작 — 6개 케이스
ℹ️  인덱스 로드 완료 (255개 청크)

[TC-001] 문서에 있는 정의 질문
질문: EVADC가 뭐야?
...
⚠️  유사도 상위: 0.459
❌ 키워드 매칭: 24%

📊 평가 요약
총 테스트      : 6개
평균 유사도    : 0.451  ✅
평균 키워드 매칭: 51%   ✅
할루시네이션 억제: 2/2  ✅
평균 응답시간  : 67.16초
```

### 옵션별 실행

```bash
# 유형별 테스트
python evaluate.py --type factual         # 정의·수치·구조 질문만
python evaluate.py --type hallucination   # 할루시네이션 테스트만
python evaluate.py --type retrieval       # 복합 질문만

# 특정 케이스만
python evaluate.py --id TC-001
```

### 테스트 케이스 추가 (test_cases.json)

PDF 내용을 보면서 정답을 아는 질문을 직접 추가합니다.

```json
{
  "test_cases": [
    {
      "id": "TC-007",
      "type": "factual",
      "description": "추가할 질문 설명",
      "question": "질문 내용",
      "ground_truth": "PDF에서 찾은 실제 정답"
    }
  ]
}
```

케이스 유형은 세 가지입니다.

| 유형 | 용도 |
|---|---|
| `factual` | 문서에 있는 내용 — 정의, 수치, 구조 질문 |
| `hallucination` | 문서에 없는 내용 — 거부 여부 확인, `ground_truth`는 `"문서에서 찾을 수 없습니다"` 고정 |
| `retrieval` | 여러 청크에 걸친 복합 질문 |

### 평가 결과 해석

| 지표 | 양호 기준 | 낮을 때 조치 |
|---|---|---|
| 유사도 | 0.35 이상 | `chunk_size` 줄이기 또는 `similarity_top_k` 늘리기 |
| 키워드 매칭 | 50% 이상 | `chunk_overlap` 늘리기 또는 LLM 교체 |
| 할루시네이션 억제 | 100% | `system_prompt` 강화 |

---

## PDF 문서 교체 시

새 문서로 교체할 경우 기존 Vector DB를 초기화해야 합니다.

```bash
# 1. chroma_db 삭제
rmdir /s /q chroma_db

# 2. docs 폴더 교체
# 기존 PDF 삭제 후 새 PDF 추가

# 3. 재실행 (자동으로 재인덱싱)
python vectordb.py
```

---

## 주요 파라미터 (vectordb.py)

| 파라미터 | 현재값 | 설명 |
|---|---|---|
| `chunk_size` | 512 | 청크 크기 (토큰 수) |
| `chunk_overlap` | 50 | 청크 간 겹치는 토큰 수 |
| `similarity_top_k` | 5 | 검색 시 참조할 청크 수 |
| `embed_model` | BAAI/bge-m3 | 한국어+영어 동시 지원 임베딩 |
| `llm` | llama3.2 | 로컬 LLM (추후 Claude API로 교체 예정) |

---

## 트러블슈팅

**`지정된 경로를 찾을 수 없습니다` 오류**
```bash
# 프로젝트 폴더에 있는지 확인
cd C:\Users\DBC141\develop\ai-agent-aurix
.venv\Scripts\activate   # .venv (점 포함)
```

**`chromadb pydantic` 오류**
Python 버전 문제. 3.12 이상은 chromadb와 호환 안 됨.
```bash
py -3.10 -m venv .venv
```

**인덱싱이 너무 느릴 때**
bge-m3는 CPU 환경에서 느림. 청크 수가 많으면 시간 소요 정상.
추후 OpenAI 임베딩 API로 교체 시 대폭 개선 가능.

---

## 향후 개발 계획

```
✅ Phase 1  Vector DB 구축 (완료)
⬜ Phase 2  Knowledge Graph 구축 (Neo4j)
⬜ Phase 3  GraphRAG 통합 (Vector DB + Knowledge Graph)
⬜ Phase 4  LLM 교체 (llama3.2 → Claude API)
⬜ Phase 5  성능 평가 (RAGAS)
```