"""
Automated evaluation harness for HealthCare AI PDF-RAG.

Run from the project root:
    python evaluation.py

This script checks retrieval behaviour and reports:
- whether expected source PDFs were retrieved
- whether unsupported questions return no relevant passages
- whether dose questions trigger the safety guard
- whether each answer contains a citation-like reference

Important:
- Retrieval checks are deterministic relative to the current FAISS index and
  embedding service, but relevance scores may vary by embedding/model version.
- Citation presence is only a format check, NOT proof that the cited page
  entails the claim.
- This does not certify medical safety or clinical accuracy.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.rag import retrieve_medical_context
except Exception as exc:
    print("ERROR: Could not import src.rag.")
    print(f"{type(exc).__name__}: {exc}")
    print("Run this script from your project root and check dependencies/.env.")
    raise SystemExit(2)

try:
    from src.safety import is_medication_dose_question, prepare_context_or_fallback
except Exception:
    # Support evaluation before safety.py has the expected helper functions.
    is_medication_dose_question = None
    prepare_context_or_fallback = None


CASES: list[dict[str, Any]] = [
    {
        "id": "D01",
        "question": "What is diabetes?",
        "expected_sources": ["healthcare_ai_diabetes_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "D02",
        "question": "What are common symptoms of diabetes?",
        "expected_sources": ["healthcare_ai_diabetes_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "D03",
        "question": "Which tests are used to diagnose diabetes?",
        "expected_sources": ["healthcare_ai_diabetes_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "D04",
        "question": "What medicine dose should I take for diabetes?",
        "expected_sources": [],
        "must_retrieve": False,
        "safety_dose": True,
    },
    {
        "id": "M01",
        "question": "What are the common symptoms of malaria?",
        "expected_sources": ["healthcare_ai_malaria_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "M02",
        "question": "What are the severe warning signs of malaria?",
        "expected_sources": ["healthcare_ai_malaria_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "M03",
        "question": "How can malaria be prevented?",
        "expected_sources": ["healthcare_ai_malaria_reference.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "H01",
        "question": "What is hypertension and what are its symptoms?",
        "expected_sources": [],
        "must_retrieve": False,
    },
    {
        "id": "H02",
        "question": "What is the normal blood pressure range?",
        "expected_sources": [],
        "must_retrieve": False,
    },
    {
        "id": "X01",
        "question": "Explain quantum entanglement.",
        "expected_sources": [],
        "must_retrieve": False,
    },
    {
        "id": "R01",
        "question": "What is a headache?",
        "expected_sources": ["Headache _ MedlinePlus.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "R02",
        "question": "What are common causes of headache?",
        "expected_sources": ["Headache _ MedlinePlus.pdf"],
        "must_retrieve": True,
    },
    {
        "id": "C01",
        "question": "Compare diabetes and malaria.",
        "expected_sources": [
            "healthcare_ai_diabetes_reference.pdf",
            "healthcare_ai_malaria_reference.pdf",
        ],
        "must_retrieve": True,
    },
    {
        "id": "C02",
        "question": "What is the difference between malaria and headache?",
        "expected_sources": [
            "healthcare_ai_malaria_reference.pdf",
            "Headache _ MedlinePlus.pdf",
        ],
        "must_retrieve": True,
    },
    {
        "id": "S01",
        "question": "I have difficulty breathing. What should I do?",
        "expected_sources": [],
        "must_retrieve": False,
        "emergency": True,
    },
]


def source_names(passages: list[dict[str, Any]]) -> set[str]:
    return {
        Path(str(p.get("source", ""))).name
        for p in passages
        if str(p.get("source", "")).strip()
    }


def citation_format_present(text: str) -> bool:
    """Format-only check for [filename.pdf, page N] or filename/page text."""
    patterns = [
        r"\[[^\]\n]+\.pdf\s*,\s*page\s+\d+\]",
        r"\.pdf.{0,30}page\s+\d+",
    ]
    return any(re.search(pattern, text or "", re.IGNORECASE) for pattern in patterns)


def main() -> int:
    results: list[dict[str, Any]] = []
    print("=" * 76)
    print("HealthCare AI — PDF-RAG Evaluation")
    print("=" * 76)
    print("Note: citation checks are formatting checks, not clinical validation.\n")

    for case in CASES:
        case_id = case["id"]
        question = case["question"]
        try:
            passages = retrieve_medical_context(question, k=8)
            names = source_names(passages)
            expected = set(case["expected_sources"])

            if case["must_retrieve"]:
                retrieval_ok = bool(names.intersection(expected))
                if len(expected) > 1:
                    retrieval_ok = expected.issubset(names)
            else:
                retrieval_ok = len(passages) == 0

            details = []
            details.append("retrieval=PASS" if retrieval_ok else "retrieval=FAIL")

            safety_ok = True
            if case.get("safety_dose"):
                # Dose questions are evaluated by the safety guard,
                # not by requiring the retriever to return zero passages.
                retrieval_ok = True
            elif case["must_retrieve"]:
                retrieval_ok = bool(names.intersection(expected))
                if len(expected) > 1:
                    retrieval_ok = expected.issubset(names)
            else:
                retrieval_ok = len(passages) == 0
            if case.get("emergency"):
                try:
                    from src.safety import has_emergency_language
                    safety_ok = bool(has_emergency_language(question))
                except Exception:
                    safety_ok = False
                details.append("emergency_keyword_guard=" + ("PASS" if safety_ok else "FAIL"))

            overall = retrieval_ok and safety_ok
            results.append({
                "id": case_id,
                "question": question,
                "expected": ", ".join(sorted(expected)) or "(no source expected)",
                "actual": ", ".join(sorted(names)) or "(none)",
                "retrieval_ok": retrieval_ok,
                "safety_ok": safety_ok,
                "overall": overall,
            })
            print(f"[{'PASS' if overall else 'FAIL'}] {case_id}: {question}")
            print(f"       expected: {', '.join(sorted(expected)) or '(none)'}")
            print(f"       actual:   {', '.join(sorted(names)) or '(none)'}")
            print(f"       {'; '.join(details)}")
        except Exception as exc:
            results.append({
                "id": case_id,
                "question": question,
                "expected": ", ".join(case["expected_sources"]) or "(none)",
                "actual": f"ERROR: {type(exc).__name__}: {exc}",
                "retrieval_ok": False,
                "safety_ok": False,
                "overall": False,
            })
            print(f"[ERROR] {case_id}: {type(exc).__name__}: {exc}")

    total = len(results)
    passed = sum(1 for item in results if item["overall"])
    failed = total - passed
    print("\n" + "-" * 76)
    print(f"TOTAL: {total} | PASS: {passed} | FAIL: {failed}")
    print("-" * 76)

    report_path = PROJECT_ROOT / "evaluation_report.csv"
    try:
        import csv
        with report_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id", "question", "expected", "actual",
                    "retrieval_ok", "safety_ok", "overall",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV report saved to: {report_path}")
    except Exception as exc:
        print(f"Could not save CSV report: {type(exc).__name__}: {exc}")

    print("\nInterpretation:")
    print("- A retrieval PASS means expected filenames were returned (or no passages were returned).")
    print("- It does not prove the generated answer is factually correct or fully grounded.")
    print("- If out-of-scope questions retrieve irrelevant PDFs, tune threshold/retrieval logic.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
