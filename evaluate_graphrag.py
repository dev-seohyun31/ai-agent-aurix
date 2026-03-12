"""
evaluate_graphrag.py
GraphRAG (Vector DB + Knowledge Graph) 품질 평가 스크립트

사용법:
    python evaluate_graphrag.py                   # 전체 테스트
    python evaluate_graphrag.py --type factual    # 타입별
    python evaluate_graphrag.py --id TC-001       # 특정 케이스
    python evaluate_graphrag.py --compare         # Vector DB 단독 vs GraphRAG 비교
"""

import os
import sys
import json
import argparse
import time
from dotenv import load_dotenv

# rag/ 폴더의 graphrag.py를 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag.graphrag import setup_models, setup_vector_db, setup_knowledge_graph, graphrag_query, vector_only_query

load_dotenv()

# ── 색상 출력 ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"{GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"{RED}❌ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"{BLUE}ℹ️  {msg}{RESET}")


# ── 단일 케이스 평가 ──────────────────────────────────
def run_test(vector_index, case):
    question     = case["question"]
    ground_truth = case["ground_truth"]
    test_type    = case["type"]

    # GraphRAG 쿼리
    start = time.time()
    result = graphrag_query(question, vector_index)
    elapsed = round(time.time() - start, 2)

    answer = result["answer"]

    # ── 평가 1: 유사도 ────────────────────────────────
    scores     = result["scores"]
    top_score  = scores[0] if scores else 0
    avg_score  = round(sum(scores) / len(scores), 3) if scores else 0

    # ── 평가 2: 키워드 매칭 ───────────────────────────
    gt_keywords = [
        w for w in ground_truth.replace(",", " ").replace(".", " ").split()
        if len(w) > 2
    ]
    matched = [kw for kw in gt_keywords if kw.lower() in answer.lower()]
    keyword_score = round(len(matched) / len(gt_keywords), 2) if gt_keywords else 0

    # ── 평가 3: 할루시네이션 억제 ─────────────────────
    if test_type == "hallucination":
        not_found_kws = ["찾을 수 없", "없습니다", "문서에", "정보가 없"]
        hallucination_ok = any(kw in answer for kw in not_found_kws)
    else:
        hallucination_ok = None

    # ── 평가 4: Graph 활용 여부 ───────────────────────
    graph_used    = result["graph_used"]
    graph_triples = result["graph_triples"]
    graph_count   = len(graph_triples.split("\n")) if graph_triples else 0

    return {
        "id":               case["id"],
        "type":             test_type,
        "description":      case.get("description", ""),
        "question":         question,
        "answer":           answer,
        "ground_truth":     ground_truth,
        "top_similarity":   top_score,
        "avg_similarity":   avg_score,
        "keyword_match":    keyword_score,
        "hallucination_ok": hallucination_ok,
        "graph_used":       graph_used,
        "graph_count":      graph_count,
        "graph_triples":    graph_triples,
        "sources":          result["sources"],
        "elapsed_sec":      elapsed,
    }


# ── 결과 출력 ─────────────────────────────────────────
def print_result(r, verbose=False):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}[{r['id']}] {r['description']}{RESET}")
    print(f"질문: {r['question']}")
    print(f"답변: {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")
    print(f"정답: {r['ground_truth']}")
    print()

    # 유사도
    if r["top_similarity"] >= 0.5:
        ok(f"유사도: {r['top_similarity']} (평균: {r['avg_similarity']})")
    elif r["top_similarity"] >= 0.35:
        warn(f"유사도: {r['top_similarity']} (평균: {r['avg_similarity']})")
    else:
        fail(f"유사도: {r['top_similarity']} (평균: {r['avg_similarity']})")

    # 키워드 매칭
    if r["keyword_match"] >= 0.6:
        ok(f"키워드 매칭: {int(r['keyword_match']*100)}%")
    elif r["keyword_match"] >= 0.3:
        warn(f"키워드 매칭: {int(r['keyword_match']*100)}%")
    else:
        fail(f"키워드 매칭: {int(r['keyword_match']*100)}%")

    # 할루시네이션
    if r["hallucination_ok"] is not None:
        if r["hallucination_ok"]:
            ok("할루시네이션 억제: 문서 없음을 정확히 인식")
        else:
            fail("할루시네이션 감지: 없는 내용을 답변에 포함")

    # Graph 활용
    if r["graph_used"]:
        ok(f"Knowledge Graph 활용: {r['graph_count']}개 관계 참조")
        if verbose and r["graph_triples"]:
            for triple in r["graph_triples"].split("\n")[:5]:
                print(f"    {BLUE}{triple}{RESET}")
    else:
        warn("Knowledge Graph: 관련 관계 없음 (Vector DB만 사용)")

    print(f"출처: {', '.join(r['sources'])}  |  응답시간: {r['elapsed_sec']}s")


# ── Vector DB 단독 vs GraphRAG 비교 ──────────────────
def run_compare(vector_index, case):
    question     = case["question"]
    ground_truth = case["ground_truth"]

    # Vector DB 단독
    v_start  = time.time()
    v_result = vector_only_query(question, vector_index)
    v_elapsed = round(time.time() - v_start, 2)

    # GraphRAG
    g_start  = time.time()
    g_result = graphrag_query(question, vector_index)
    g_elapsed = round(time.time() - g_start, 2)

    # 키워드 매칭 계산
    def kw_score(answer):
        gt_kws  = [w for w in ground_truth.replace(",", " ").replace(".", " ").split() if len(w) > 2]
        matched = [kw for kw in gt_kws if kw.lower() in answer.lower()]
        return round(len(matched) / len(gt_kws), 2) if gt_kws else 0

    v_kw = kw_score(v_result["answer"])
    g_kw = kw_score(g_result["answer"])

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}[{case['id']}] {case.get('description', '')}{RESET}")
    print(f"질문: {question}\n")

    print(f"{BLUE}[Vector DB 단독]{RESET}")
    print(f"  답변: {v_result['answer'][:150]}...")
    print(f"  키워드 매칭: {int(v_kw*100)}%  |  응답시간: {v_elapsed}s")

    print(f"\n{GREEN}[GraphRAG (Vector + Graph)]{RESET}")
    print(f"  답변: {g_result['answer'][:150]}...")
    print(f"  키워드 매칭: {int(g_kw*100)}%  |  응답시간: {g_elapsed}s")
    print(f"  Graph 활용: {'✅ ' + str(len(g_result['graph_triples'].split(chr(10))) if g_result['graph_triples'] else 0) + '개 관계' if g_result['graph_used'] else '❌ 없음'}")

    # 승자 판정
    if g_kw > v_kw:
        ok(f"GraphRAG 우세 ({int(v_kw*100)}% → {int(g_kw*100)}%, +{int((g_kw-v_kw)*100)}%p)")
    elif g_kw == v_kw:
        warn(f"동점 ({int(g_kw*100)}%)")
    else:
        warn(f"Vector DB 우세 ({int(v_kw*100)}% → {int(g_kw*100)}%)")

    return {
        "id":        case["id"],
        "v_kw":      v_kw,
        "g_kw":      g_kw,
        "graph_used": g_result["graph_used"],
        "v_elapsed": v_elapsed,
        "g_elapsed": g_elapsed,
    }


# ── 최종 요약 ─────────────────────────────────────────
def print_summary(results):
    print(f"\n{BOLD}{'='*60}")
    print("📊 GraphRAG 평가 요약")
    print(f"{'='*60}{RESET}")

    total = len(results)
    avg_sim  = round(sum(r["avg_similarity"] for r in results) / total, 3)
    avg_kw   = round(sum(r["keyword_match"]  for r in results) / total, 2)
    avg_time = round(sum(r["elapsed_sec"]    for r in results) / total, 2)

    hallucination_cases = [r for r in results if r["hallucination_ok"] is not None]
    hallucination_pass  = sum(1 for r in hallucination_cases if r["hallucination_ok"])

    graph_used_count = sum(1 for r in results if r["graph_used"])

    print(f"총 테스트           : {total}개")
    print(f"평균 유사도         : {avg_sim}  {'✅' if avg_sim >= 0.4 else '⚠️'}")
    print(f"평균 키워드 매칭    : {int(avg_kw*100)}%  {'✅' if avg_kw >= 0.5 else '⚠️'}")
    print(f"할루시네이션 억제   : {hallucination_pass}/{len(hallucination_cases)}  {'✅' if hallucination_pass == len(hallucination_cases) else '❌'}")
    print(f"Graph 활용          : {graph_used_count}/{total}개 케이스")
    print(f"평균 응답시간       : {avg_time}초")

    print(f"\n{BOLD}개선 제안:{RESET}")
    if avg_sim < 0.4:
        warn("유사도 낮음 → chunk_size 줄이기 또는 similarity_top_k 늘리기")
    if avg_kw < 0.5:
        warn("키워드 매칭 낮음 → chunk_overlap 늘리기")
    if hallucination_pass < len(hallucination_cases):
        fail("할루시네이션 발생 → system_prompt 강화 필요")
    if graph_used_count < total // 2:
        warn("Graph 활용률 낮음 → keyword_map에 도메인 용어 추가 필요")
    if avg_sim >= 0.4 and avg_kw >= 0.5 and hallucination_pass == len(hallucination_cases):
        ok("전체적으로 양호합니다!")


def print_compare_summary(compare_results):
    print(f"\n{BOLD}{'='*60}")
    print("📊 Vector DB 단독 vs GraphRAG 최종 비교")
    print(f"{'='*60}{RESET}")

    total   = len(compare_results)
    v_avg   = round(sum(r["v_kw"] for r in compare_results) / total, 2)
    g_avg   = round(sum(r["g_kw"] for r in compare_results) / total, 2)
    v_time  = round(sum(r["v_elapsed"] for r in compare_results) / total, 2)
    g_time  = round(sum(r["g_elapsed"] for r in compare_results) / total, 2)
    g_wins  = sum(1 for r in compare_results if r["g_kw"] > r["v_kw"])
    v_wins  = sum(1 for r in compare_results if r["v_kw"] > r["g_kw"])
    draws   = total - g_wins - v_wins
    graph_used = sum(1 for r in compare_results if r["graph_used"])

    print(f"{'항목':<20} {'Vector DB':>12} {'GraphRAG':>12}")
    print(f"{'─'*44}")
    print(f"{'평균 키워드 매칭':<20} {int(v_avg*100):>11}% {int(g_avg*100):>11}%")
    print(f"{'평균 응답시간':<20} {v_time:>11}s {g_time:>11}s")
    print(f"{'─'*44}")
    print(f"승/무/패: GraphRAG {g_wins}승 {draws}무 {v_wins}패")
    print(f"Graph 활용: {graph_used}/{total}개 케이스")

    if g_avg > v_avg:
        ok(f"GraphRAG 효과 확인: +{int((g_avg-v_avg)*100)}%p 향상")
    elif g_avg == v_avg:
        warn("동점 — Graph 키워드맵 확장 권장")
    else:
        warn("Vector DB 단독이 우세 — Knowledge Graph 품질 개선 필요")


# ── 메인 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GraphRAG 평가 스크립트")
    parser.add_argument("--type",    help="factual / hallucination / retrieval")
    parser.add_argument("--id",      help="특정 케이스 ID (예: TC-001)")
    parser.add_argument("--compare", action="store_true", help="Vector DB vs GraphRAG 비교")
    parser.add_argument("--verbose", action="store_true", help="Graph 관계 상세 출력")
    args = parser.parse_args()

    # 테스트 케이스 로드
    cases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
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

    # 모델 및 DB 초기화
    setup_models()
    vector_index = setup_vector_db()
    setup_knowledge_graph()

    # 비교 모드
    if args.compare:
        print(f"\n{BOLD}🧪 Vector DB vs GraphRAG 비교 — {len(cases)}개 케이스{RESET}")
        compare_results = []
        for i, case in enumerate(cases):
            print(f"\n진행중... ({i+1}/{len(cases)})", end="", flush=True)
            r = run_compare(vector_index, case)
            compare_results.append(r)
        print_compare_summary(compare_results)
        return

    # 일반 평가 모드
    print(f"\n{BOLD}🧪 GraphRAG 평가 시작 — {len(cases)}개 케이스{RESET}")
    results = []
    for i, case in enumerate(cases):
        print(f"\n진행중... ({i+1}/{len(cases)})", end="", flush=True)
        r = run_test(vector_index, case)
        print_result(r, verbose=args.verbose)
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()