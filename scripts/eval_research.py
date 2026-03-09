from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} in {path} must be a JSON object.")
            rows.append(obj)
    return rows


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_case_key(value: str) -> str:
    value = normalize_text(value)
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path = re.sub(r"/+", "/", parsed.path.strip("/"))
        return f"{parsed.netloc.lower()}/{path}"
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def normalize_factor(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9_ -]+", "", text)
    return text.replace(" ", "_")


def extract_case_entries(pred_row: Dict[str, Any]) -> List[Dict[str, str]]:
    entries = pred_row.get("retrieved_cases") or pred_row.get("predicted_cases") or []
    if isinstance(entries, dict):
        entries = [entries]

    out: List[Dict[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            out.append({"key": normalize_case_key(entry), "quote": ""})
            continue
        if not isinstance(entry, dict):
            continue
        case_url = entry.get("url") or entry.get("case_url") or ""
        case_name = entry.get("case_name") or entry.get("name") or ""
        key = normalize_case_key(case_url) or normalize_case_key(case_name)
        quote = normalize_text(entry.get("quote") or entry.get("passage") or "")
        out.append({"key": key, "quote": quote})
    return out


def extract_gold_case_keys(gold_row: Dict[str, Any]) -> Set[str]:
    cases = gold_row.get("gold_cases", [])
    if isinstance(cases, str):
        cases = [cases]
    keys = set()
    for case in cases:
        if isinstance(case, dict):
            case = case.get("url") or case.get("case_name") or case.get("id") or ""
        key = normalize_case_key(case)
        if key:
            keys.add(key)
    return keys


def extract_gold_passages(gold_row: Dict[str, Any]) -> List[str]:
    passages = gold_row.get("gold_passages", [])
    if isinstance(passages, str):
        passages = [passages]
    out = []
    for passage in passages:
        if isinstance(passage, dict):
            passage = passage.get("quote") or passage.get("text") or ""
        p = normalize_text(passage)
        if p:
            out.append(p)
    return out


def quote_matches_any_gold(quote: str, gold_passages: List[str], threshold: float = 0.85) -> bool:
    if not quote or not gold_passages:
        return False
    for gold in gold_passages:
        if quote in gold or gold in quote:
            return True
        ratio = SequenceMatcher(a=quote, b=gold).ratio()
        if ratio >= threshold:
            return True
    return False


def evaluate(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    pred_by_id = {str(row.get("id", "")).strip(): row for row in pred_rows if row.get("id")}

    scored = 0
    missing_predictions = []

    recall_hits = 0
    precision_scores: List[float] = []
    citation_scores: List[float] = []
    factor_scores: List[float] = []
    per_case: List[Dict[str, Any]] = []

    for gold in gold_rows:
        case_id = str(gold.get("id", "")).strip()
        if not case_id:
            continue

        gold_case_keys = extract_gold_case_keys(gold)
        gold_passages = extract_gold_passages(gold)
        gold_factors = {normalize_factor(x) for x in (gold.get("factor_tags") or []) if normalize_factor(x)}
        has_gold_signal = bool(gold_case_keys or gold_passages or gold_factors)
        if not has_gold_signal:
            continue

        pred = pred_by_id.get(case_id)
        if pred is None:
            missing_predictions.append(case_id)
            continue

        scored += 1
        pred_cases = extract_case_entries(pred)
        pred_topk = pred_cases[:k]
        pred_keys = [entry["key"] for entry in pred_topk if entry["key"]]

        recall_hit = 0
        if gold_case_keys:
            recall_hit = 1 if any(key in gold_case_keys for key in pred_keys) else 0
        else:
            recall_hit = 1
        recall_hits += recall_hit

        if k <= 0:
            precision = 0.0
        elif gold_case_keys:
            precision = sum(1 for key in pred_keys if key in gold_case_keys) / k
        else:
            precision = 1.0
        precision_scores.append(precision)

        pred_quotes = [entry["quote"] for entry in pred_topk if entry["quote"]]
        if pred_quotes and gold_passages:
            matches = sum(1 for quote in pred_quotes if quote_matches_any_gold(quote, gold_passages))
            citation_score = matches / len(pred_quotes)
        elif not gold_passages:
            citation_score = 1.0
        else:
            citation_score = 0.0
        citation_scores.append(citation_score)

        pred_factors = pred.get("predicted_factors") or pred.get("factor_tags") or []
        if isinstance(pred_factors, str):
            pred_factors = [pred_factors]
        pred_factor_set = {normalize_factor(x) for x in pred_factors if normalize_factor(x)}
        if gold_factors:
            factor_coverage = len(gold_factors & pred_factor_set) / len(gold_factors)
        else:
            factor_coverage = 1.0
        factor_scores.append(factor_coverage)

        per_case.append(
            {
                "id": case_id,
                f"recall@{k}_hit": recall_hit,
                f"precision@{k}": round(precision, 4),
                "citation_correctness": round(citation_score, 4),
                "factor_coverage": round(factor_coverage, 4),
            }
        )

    report = {
        "totals": {
            "gold_rows": len(gold_rows),
            "pred_rows": len(pred_rows),
            "scored_rows": scored,
            "missing_predictions": len(missing_predictions),
            "k": k,
        },
        "metrics": {
            f"recall@{k}": round((recall_hits / scored) if scored else 0.0, 4),
            f"precision@{k}": round((sum(precision_scores) / len(precision_scores)) if precision_scores else 0.0, 4),
            "citation_correctness": round((sum(citation_scores) / len(citation_scores)) if citation_scores else 0.0, 4),
            "factor_coverage": round((sum(factor_scores) / len(factor_scores)) if factor_scores else 0.0, 4),
        },
        "missing_prediction_ids": missing_predictions,
        "per_case": per_case,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate research retrieval predictions against gold JSONL.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path (case_retrieval_test).")
    parser.add_argument("--pred", required=True, help="Prediction JSONL path.")
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K and Precision@K.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    parser.add_argument("--show-cases", action="store_true", help="Print per-case details.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    report = evaluate(gold_rows=gold, pred_rows=pred, k=args.k)

    if not args.show_cases:
        report = {k: v for k, v in report.items() if k != "per_case"}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
