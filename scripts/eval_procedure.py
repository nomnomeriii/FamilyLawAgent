from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


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
    text = re.sub(r"[^a-z0-9\s/_-]", "", text)
    return text


def normalize_items(values: Any) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    out = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("text") or value.get("value") or json.dumps(value, sort_keys=True)
        n = normalize_text(value)
        if n:
            out.add(n)
    return out


def item_overlap(gold_items: Set[str], pred_items: Set[str]) -> float:
    if not gold_items:
        return 1.0
    if not pred_items:
        return 0.0
    matches = 0
    for gold in gold_items:
        if any(gold in pred or pred in gold for pred in pred_items):
            matches += 1
    return matches / len(gold_items)


def evaluate(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_by_id = {str(row.get("id", "")).strip(): row for row in pred_rows if row.get("id")}
    scored = 0
    missing_predictions = []

    workflow_hits = 0
    checklist_scores: List[float] = []
    critical_misses = 0
    per_case: List[Dict[str, Any]] = []

    for gold in gold_rows:
        case_id = str(gold.get("id", "")).strip()
        if not case_id:
            continue

        gold_form = normalize_items(gold.get("gold_form_family"))
        gold_required = normalize_items(gold.get("required_inputs"))
        gold_service = normalize_items(gold.get("service_steps"))
        gold_attach = normalize_items(gold.get("attachments"))
        gold_critical = normalize_items(gold.get("critical_steps"))
        has_gold_signal = bool(gold_form or gold_required or gold_service or gold_attach or gold_critical)
        if not has_gold_signal:
            continue

        pred = pred_by_id.get(case_id)
        if pred is None:
            missing_predictions.append(case_id)
            continue

        scored += 1
        pred_form = normalize_items(pred.get("predicted_form_family"))
        pred_required = normalize_items(pred.get("predicted_required_inputs"))
        pred_service = normalize_items(pred.get("predicted_service_steps"))
        pred_attach = normalize_items(pred.get("predicted_attachments"))
        pred_all = pred_required | pred_service | pred_attach

        form_hit = 0
        if gold_form:
            # Count as hit if any gold form family appears in predicted family list.
            form_hit = 1 if item_overlap(gold_form, pred_form) > 0 else 0
        else:
            form_hit = 1
        workflow_hits += form_hit

        gold_all_checklist = gold_required | gold_service | gold_attach
        checklist = item_overlap(gold_all_checklist, pred_all)
        checklist_scores.append(checklist)

        missing_critical = []
        for critical in gold_critical:
            if not any(critical in p or p in critical for p in pred_all):
                missing_critical.append(critical)
        critical_fail = 1 if missing_critical else 0
        critical_misses += critical_fail

        per_case.append(
            {
                "id": case_id,
                "workflow_hit": form_hit,
                "checklist_completeness": round(checklist, 4),
                "critical_miss": bool(critical_fail),
                "missing_critical_items": missing_critical,
            }
        )

    workflow_accuracy = workflow_hits / scored if scored else 0.0
    checklist_completeness = sum(checklist_scores) / len(checklist_scores) if checklist_scores else 0.0
    critical_miss_rate = critical_misses / scored if scored else 0.0

    return {
        "totals": {
            "gold_rows": len(gold_rows),
            "pred_rows": len(pred_rows),
            "scored_rows": scored,
            "missing_predictions": len(missing_predictions),
        },
        "metrics": {
            "workflow_accuracy": round(workflow_accuracy, 4),
            "checklist_completeness": round(checklist_completeness, 4),
            "critical_miss_rate": round(critical_miss_rate, 4),
        },
        "missing_prediction_ids": missing_predictions,
        "per_case": per_case,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate procedure workflow predictions against gold JSONL.")
    parser.add_argument("--gold", required=True, help="Gold JSONL path (procedure_workflow_test).")
    parser.add_argument("--pred", required=True, help="Prediction JSONL path.")
    parser.add_argument("--out", default="", help="Optional output JSON file path.")
    parser.add_argument("--show-cases", action="store_true", help="Print per-case details.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gold = load_jsonl(Path(args.gold))
    pred = load_jsonl(Path(args.pred))
    report = evaluate(gold, pred)

    if not args.show_cases:
        report = {k: v for k, v in report.items() if k != "per_case"}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
