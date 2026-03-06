"""
evaluate.py
Vector DB 검색 품질 및 RAG 답변 품질 평가 스크립트

사용법:
    python evaluate.py                        # 전체 테스트
    python evaluate.py --type factual         # 타입별 테스트 (factual / hallucination / retrieval)
    python evaluate.py --id TC-001            # 특정 케이스만 테스트
    python evaluate.py --verbose              # 청크 내용까지 출력
"""

import os
import json
import argparse
import time
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

load_dotenv()

# ── 색상 출력 헬퍼 ────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"{GREEN}✅ {msg}{RESET}")
def fail(msg):  print(f"{RED}❌ {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}⚠️  {msg}{RESET}")
def info(msg):  print(f"{BLUE}ℹ️  {msg}{RESET}")

# ── 모델 설정 ─────────────────────────────────────────
def setup():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.llm = Ollama(
        model="llama3.2",
        request_timeout=120.0,
        system_prompt=(
            "You are a technical document assistant. "
            "ALWAYS respond in Korean only. "
            "NEVER mix other languages into your response. "
            "Only use information from the provided documents. "
            "If the answer is not in the documents, say '문서에서 찾을 수 없습니다'."
        )
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("research_docs")

    if collection.count() == 0:
        fail("chroma_db가 비어있습니다. 먼저 vectordb.py를 실행해 인덱싱하세요.")
        exit(1)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize"
    )

    info(f"인덱스 로드 완료 ({collection.count()}개 청크)")
    return query_engine

# ── 단일 테스트 실행 ──────────────────────────────────
def run_test(query_engine, case, verbose=False):
    question    = case["question"]
    ground_truth = case["ground_truth"]
    test_type   = case["type"]

    start = time.time()
    response = query_engine.query(question)
    elapsed = round(time.time() - start, 2)

    answer  = str(response).strip()
    sources = response.source_nodes
    scores  = [round(n.score, 3) for n in sources if n.score]
    files   = list(set(n.metadata.get("file_name", "?") for n in sources))

    # ── 평가 로직 ─────────────────────────────────────
    result = {}

    # 1. 유사도 검색 품질
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0
    top_score = scores[0] if scores else 0
    score_gap = round(scores[0] - scores[-1], 3) if len(scores) > 1 else 0

    # 2. 할루시네이션 감지 (hallucination 타입)
    if test_type == "hallucination":
        not_found_keywords = ["찾을 수 없", "없습니다", "문서에", "정보가 없"]
        hallucination_ok = any(kw in answer for kw in not_found_keywords)
        result["hallucination_check"] = hallucination_ok
    else:
        result["hallucination_check"] = None

    # 3. 키워드 포함 여부 (ground_truth 핵심 단어 기반)
    gt_keywords = [w for w in ground_truth.replace(",", " ").replace(".", " ").split() if len(w) > 2]
    matched = [kw for kw in gt_keywords if kw.lower() in answer.lower()]
    keyword_score = round(len(matched) / len(gt_keywords), 2) if gt_keywords else 0

    result.update({
        "id": case["id"],
        "type": test_type,
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "avg_similarity": avg_score,
        "top_similarity": top_score,
        "score_gap": score_gap,
        "keyword_match": keyword_score,
        "source_files": files,
        "elapsed_sec": elapsed,
    })

    return result

# ── 결과 출력 ─────────────────────────────────────────
def print_result(r, verbose=False):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}[{r['id']}] {r['description'] if 'description' in r else r['type'].upper()}{RESET}")
    print(f"질문: {r['question']}")
    print(f"답변: {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")
    print(f"정답: {r['ground_truth']}")
    print()

    # 유사도 평가
    if r["top_similarity"] >= 0.5:
        ok(f"유사도 상위: {r['top_similarity']} (평균: {r['avg_similarity']})")
    elif r["top_similarity"] >= 0.35:
        warn(f"유사도 상위: {r['top_similarity']} (평균: {r['avg_similarity']}) — 개선 여지 있음")
    else:
        fail(f"유사도 상위: {r['top_similarity']} (평균: {r['avg_similarity']}) — 낮음")

    # 키워드 매칭
    if r["keyword_match"] >= 0.6:
        ok(f"키워드 매칭: {int(r['keyword_match']*100)}%")
    elif r["keyword_match"] >= 0.3:
        warn(f"키워드 매칭: {int(r['keyword_match']*100)}%")
    else:
        fail(f"키워드 매칭: {int(r['keyword_match']*100)}%")

    # 할루시네이션 체크
    if r["hallucination_check"] is not None:
        if r["hallucination_check"]:
            ok("할루시네이션 억제: 문서 없음을 정확히 인식")
        else:
            fail("할루시네이션 감지: 없는 내용을 답변에 포함")

    print(f"출처: {', '.join(r['source_files'])}  |  응답시간: {r['elapsed_sec']}s")

# ── 최종 요약 ─────────────────────────────────────────
def print_summary(results):
    print(f"\n{BOLD}{'='*60}")
    print("📊 평가 요약")
    print(f"{'='*60}{RESET}")

    total = len(results)
    avg_sim   = round(sum(r["avg_similarity"] for r in results) / total, 3)
    avg_kw    = round(sum(r["keyword_match"] for r in results) / total, 2)
    avg_time  = round(sum(r["elapsed_sec"] for r in results) / total, 2)

    hallucination_cases = [r for r in results if r["hallucination_check"] is not None]
    hallucination_pass  = sum(1 for r in hallucination_cases if r["hallucination_check"])

    print(f"총 테스트      : {total}개")
    print(f"평균 유사도    : {avg_sim}  {'✅' if avg_sim >= 0.4 else '⚠️'}")
    print(f"평균 키워드 매칭: {int(avg_kw*100)}%  {'✅' if avg_kw >= 0.5 else '⚠️'}")
    print(f"할루시네이션 억제: {hallucination_pass}/{len(hallucination_cases)}  {'✅' if hallucination_pass == len(hallucination_cases) else '❌'}")
    print(f"평균 응답시간  : {avg_time}초")

    print(f"\n{BOLD}개선 제안:{RESET}")
    if avg_sim < 0.4:
        warn("유사도 낮음 → chunk_size 줄이기 (512 → 256) 또는 similarity_top_k 늘리기")
    if avg_kw < 0.5:
        warn("키워드 매칭 낮음 → chunk_overlap 늘리기 (50 → 100) 또는 LLM 교체 고려")
    if hallucination_pass < len(hallucination_cases):
        fail("할루시네이션 발생 → system_prompt 강화 또는 similarity_threshold 추가 필요")
    if avg_sim >= 0.4 and avg_kw >= 0.5 and hallucination_pass == len(hallucination_cases):
        ok("전체적으로 양호합니다!")

# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG 평가 스크립트")
    parser.add_argument("--type", help="테스트 타입 필터 (factual / hallucination / retrieval)")
    parser.add_argument("--id",   help="특정 테스트 케이스 ID (예: TC-001)")
    parser.add_argument("--verbose", action="store_true", help="청크 내용 상세 출력")
    args = parser.parse_args()

    # 테스트 케이스 로드
    with open("test_cases.json", "r", encoding="utf-8") as f:
        all_cases = json.load(f)["test_cases"]

    # 필터링
    if args.id:
        cases = [c for c in all_cases if c["id"] == args.id]
    elif args.type:
        cases = [c for c in all_cases if c["type"] == args.type]
    else:
        cases = all_cases

    if not cases:
        fail("해당하는 테스트 케이스가 없습니다.")
        return

    print(f"\n{BOLD}🧪 RAG 평가 시작 — {len(cases)}개 케이스{RESET}")
    query_engine = setup()

    results = []
    for i, case in enumerate(cases):
        print(f"\n진행중... ({i+1}/{len(cases)})", end="", flush=True)
        r = run_test(query_engine, case, verbose=args.verbose)
        r["description"] = case.get("description", "")
        print_result(r, verbose=args.verbose)
        results.append(r)

    print_summary(results)

if __name__ == "__main__":
    main()