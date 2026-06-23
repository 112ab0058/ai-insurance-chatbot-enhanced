"""此腳本會呼叫真實 API，執行前確認 token 額度，預設不自動執行。

這是人工執行的輕量評測工具；pytest 不會載入或執行本檔案。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.rag import answer_question, build_knowledge_base


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
DEFAULT_TESTSET = EVAL_DIR / "qa_testset.json"
DEFAULT_PDF = PROJECT_ROOT / "insurance.pdf"
DEFAULT_OUTPUT = EVAL_DIR / "evaluation_results.json"


def load_testset(path: Path = DEFAULT_TESTSET) -> list[dict[str, Any]]:
    """Load the manually curated QA cases."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_cases(
    cases: list[dict[str, Any]],
    pdf_path: Path = DEFAULT_PDF,
    token: str | None = None,
) -> dict[str, Any]:
    """Call the real RAG pipeline once per case and calculate rule-based metrics."""
    api_token = (token or os.getenv("GITHUB_TOKEN", "")).strip()
    if not api_token:
        raise RuntimeError("請先設定 GITHUB_TOKEN；此評測會消耗真實 API 額度。")

    pdf_bytes = pdf_path.read_bytes()
    vectorstore, _ = build_knowledge_base(pdf_bytes, pdf_path.name)
    rows: list[dict[str, Any]] = []

    for case in cases:
        result = answer_question(
            vectorstore,
            str(case["question"]),
            api_token,
            role="user",
        )
        expected_keywords = [str(item) for item in case.get("expected_keywords", [])]
        matched_keywords = [
            keyword for keyword in expected_keywords if keyword in result.answer
        ]
        source_pages = [source.page for source in result.sources]
        expected_page = case.get("expected_page")
        retrieval_hit = expected_page is None or int(expected_page) in source_pages
        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
        )
        rows.append(
            {
                "question": case["question"],
                "answer": result.answer,
                "source_pages": source_pages,
                "expected_page": expected_page,
                "retrieval_hit": retrieval_hit,
                "matched_keywords": matched_keywords,
                "keyword_coverage": round(keyword_coverage, 4),
                "low_confidence": result.low_confidence,
            }
        )

    total = len(rows)
    return {
        "summary": {
            "case_count": total,
            "retrieval_accuracy": round(
                sum(row["retrieval_hit"] for row in rows) / total, 4
            )
            if total
            else 0.0,
            "keyword_coverage": round(
                sum(row["keyword_coverage"] for row in rows) / total, 4
            )
            if total
            else 0.0,
        },
        "results": rows,
    }


def print_score_table(report: dict[str, Any]) -> None:
    """Print a compact terminal score table without extra dependencies."""
    print("# | Retrieval | Keywords | Question")
    print("--|-----------|----------|---------")
    for index, row in enumerate(report["results"], start=1):
        retrieval = "PASS" if row["retrieval_hit"] else "FAIL"
        coverage = f"{row['keyword_coverage'] * 100:.0f}%"
        print(f"{index:>2} | {retrieval:^9} | {coverage:^8} | {row['question']}")
    summary = report["summary"]
    print(
        "\nRetrieval accuracy: "
        f"{summary['retrieval_accuracy'] * 100:.1f}% | "
        f"Keyword coverage: {summary['keyword_coverage'] * 100:.1f}%"
    )


def main() -> None:
    """Run the opt-in real-API evaluation and save its detailed results."""
    report = evaluate_cases(load_testset())
    print_score_table(report)
    DEFAULT_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"詳細結果：{DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
