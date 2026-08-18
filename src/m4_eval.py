from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, GOOGLE_API_KEY, GEMINI_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _paced_invoke(llm, prompt: str, delay: float = 5.0) -> str:
    """Invoke LLM with a small delay to stay under free-tier rate limits (Gemini free tier: ~15 RPM)."""
    time.sleep(delay)
    return llm.invoke(prompt).content.strip()


def _llm_score_faithfulness(llm, answer: str, contexts: list[str]) -> float:
    """% of answer statements that are supported by the given contexts (RAGAS faithfulness definition)."""
    context_str = "\n\n".join(contexts)
    prompt = (
        "Cho câu trả lời sau, liệt kê các mệnh đề/khẳng định riêng biệt trong đó (mỗi dòng 1 mệnh đề).\n\n"
        f"Câu trả lời:\n{answer}"
    )
    statements_text = _paced_invoke(llm, prompt)
    statements = [s.strip().lstrip("0123456789.-) ") for s in statements_text.split("\n") if s.strip()][:3]
    if not statements:
        return 0.0

    verdicts = []
    for stmt in statements:
        judge_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Mệnh đề: {stmt}\n\n"
            "Mệnh đề trên có được suy ra/hỗ trợ trực tiếp bởi context không? "
            "Trả lời CHỈ một từ: 'yes' hoặc 'no'."
        )
        verdict = _paced_invoke(llm, judge_prompt).lower()
        verdicts.append(1 if "yes" in verdict else 0)
    return sum(verdicts) / len(verdicts)


def _llm_score_answer_relevancy(llm, embeddings, question: str, answer: str) -> float:
    """Cosine similarity between the original question and questions regenerated from the answer."""
    import numpy as np

    prompt = (
        f"Dựa trên câu trả lời sau, hãy sinh ra 1 câu hỏi mà câu trả lời đó trả lời. "
        f"Chỉ trả về câu hỏi, không giải thích.\n\nCâu trả lời:\n{answer}"
    )
    generated_question = _paced_invoke(llm, prompt)

    v1 = np.array(embeddings.embed_query(question))
    v2 = np.array(embeddings.embed_query(generated_question))
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return max(0.0, sim)


def _llm_score_context_precision(llm, question: str, answer: str, contexts: list[str]) -> float:
    """% of retrieved contexts judged relevant to answering the question (precision@k)."""
    if not contexts:
        return 0.0
    verdicts = []
    for ctx in contexts[:3]:
        judge_prompt = (
            f"Câu hỏi: {question}\n\nContext:\n{ctx}\n\n"
            "Context trên có hữu ích để trả lời câu hỏi không? Trả lời CHỈ 'yes' hoặc 'no'."
        )
        verdict = _paced_invoke(llm, judge_prompt).lower()
        verdicts.append(1 if "yes" in verdict else 0)
    return sum(verdicts) / len(verdicts)


def _llm_score_context_recall(llm, ground_truth: str, contexts: list[str]) -> float:
    """% of ground-truth statements that can be attributed to the retrieved contexts."""
    context_str = "\n\n".join(contexts)
    prompt = (
        f"Câu trả lời chuẩn: {ground_truth}\n\nListe các mệnh đề/khẳng định riêng biệt trong câu trả lời chuẩn "
        "trên (mỗi dòng 1 mệnh đề)."
    )
    statements_text = _paced_invoke(llm, prompt)
    statements = [s.strip().lstrip("0123456789.-) ") for s in statements_text.split("\n") if s.strip()][:3]
    if not statements:
        return 0.0

    verdicts = []
    for stmt in statements:
        judge_prompt = (
            f"Context:\n{context_str}\n\nMệnh đề: {stmt}\n\n"
            "Mệnh đề trên có thể được suy ra từ context không? Trả lời CHỈ 'yes' hoặc 'no'."
        )
        verdict = _paced_invoke(llm, judge_prompt).lower()
        verdicts.append(1 if "yes" in verdict else 0)
    return sum(verdicts) / len(verdicts)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS-style evaluation (4 metrics), computed via direct sync LLM calls.

    ⚠️ Dùng ChatGoogleGenerativeAI.invoke() thay vì ragas.evaluate() vì thư viện ragas 0.1.x
    bị deadlock trong async Executor trên Windows (asyncio.run + gRPC AsyncClient conflict).
    Cùng 4 metrics, cùng phương pháp luận RAGAS (LLM-as-judge trên statements), chỉ khác
    cách thực thi: đồng bộ, tuần tự thay vì qua ragas's internal asyncio executor.
    """
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
    if not GOOGLE_API_KEY:
        print("  ⚠️  GOOGLE_API_KEY not set — skipping RAGAS evaluation")
        return zeros

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

        per_question = []
        for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
            print(f"  [{i+1}/{len(questions)}] scoring: {q[:40]}...", flush=True)
            try:
                f = _llm_score_faithfulness(llm, a, ctx)
                print(f"    faithfulness={f:.2f}", flush=True)
                ar = _llm_score_answer_relevancy(llm, embeddings, q, a)
                print(f"    answer_relevancy={ar:.2f}", flush=True)
                cp = _llm_score_context_precision(llm, q, a, ctx)
                print(f"    context_precision={cp:.2f}", flush=True)
                cr = _llm_score_context_recall(llm, gt, ctx)
                print(f"    context_recall={cr:.2f}", flush=True)
            except Exception as e:
                print(f"  ⚠️  RAGAS scoring failed for question '{q[:40]}...': {e}")
                f = ar = cp = cr = 0.0
            per_question.append(EvalResult(
                question=q, answer=a, contexts=ctx, ground_truth=gt,
                faithfulness=f, answer_relevancy=ar, context_precision=cp, context_recall=cr,
            ))

        if not per_question:
            return zeros

        return {
            "faithfulness": sum(r.faithfulness for r in per_question) / len(per_question),
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / len(per_question),
            "context_precision": sum(r.context_precision for r in per_question) / len(per_question),
            "context_recall": sum(r.context_recall for r in per_question) / len(per_question),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        scored.append((avg, worst_metric, metrics[worst_metric], r))

    scored.sort(key=lambda x: x[0])

    failures = []
    for avg, worst_metric, worst_score, r in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
