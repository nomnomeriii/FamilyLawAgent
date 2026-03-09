from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from family_law_agent.procedure import load_vectorstore, run_procedure_engine
from family_law_agent.research import run_research_engine_structured


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{i}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Each JSONL row must be an object: {path}:{i}")
        rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def dedup_keep_order(items: Iterable[str], limit: int = 12) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        val = " ".join(str(item or "").strip().split())
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
        if len(out) >= limit:
            break
    return out


def section_lines(text: str, section_hint: str) -> List[str]:
    lines = text.splitlines()
    if not lines:
        return []

    start = -1
    hint = section_hint.lower()
    for i, line in enumerate(lines):
        normalized = re.sub(r"[*#`_]", "", line).strip().lower()
        if hint in normalized:
            start = i + 1
            break
    if start < 0:
        return []

    out: List[str] = []
    for line in lines[start:]:
        normalized = line.strip()
        if not normalized:
            continue
        if re.match(r"^\s*\d+\)\s+", normalized):
            break
        if re.match(r"^\s*\d+\.\s+", normalized):
            break
        cleaned = re.sub(r"^\s*[-*•]+\s*", "", normalized).strip()
        if cleaned:
            out.append(cleaned)
    return out


def infer_form_family(query: str, response: str, case_type: str) -> List[str]:
    q = f"{query} {response}".lower()
    labels: List[str] = []

    if "relocat" in q:
        labels.extend(["custody_visitation_modification_petition", "relocation_request_motion"])
    if "custody" in q and "emerg" in q:
        labels.extend(["emergency_order_to_show_cause_temporary_custody", "custody_visitation_petition"])
    if "custody" in q and "modif" in q:
        labels.append("custody_visitation_modification_petition")
    if "visitation" in q and "grand" in q:
        labels.append("grandparent_visitation_petition")

    if "support" in q and "modif" in q:
        labels.append("child_support_modification_petition")
    if "support" in q and ("enforc" in q or "violation" in q or "arrear" in q):
        labels.append("child_support_violation_enforcement_petition")
    if "paternity" in q or "parentage" in q:
        labels.extend(["paternity_parentage_petition", "child_support_petition"])

    if "order of protection" in q or "family offense" in q:
        labels.extend(["family_offense_petition", "temporary_order_of_protection_request"])
    if "extend" in q and "order of protection" in q:
        labels.append("order_of_protection_extension_petition_or_motion")
    if "modify" in q and "order of protection" in q:
        labels.append("order_of_protection_modification_petition_or_motion")

    if "uncontested divorce" in q:
        labels.append("uncontested_divorce_packet")
    if "default" in q and "divorce" in q:
        labels.append("default_divorce_application")
    if "name" in q and "divorce" in q:
        labels.append("divorce_judgment_name_change_request")
    if "temporary" in q and "divorce" in q:
        labels.append("pendente_lite_motion_for_temporary_relief")

    # Case-type fallback if nothing matched.
    if not labels:
        fallback = {
            "custody": "initial_custody_visitation_petition",
            "support": "child_support_petition",
            "divorce": "divorce_petition_or_motion",
            "oop": "family_offense_petition",
            "parentage": "paternity_parentage_petition",
        }
        labels.append(fallback.get(case_type, "family_court_petition"))

    return dedup_keep_order(labels, limit=5)


def split_service_and_attachments(items: Iterable[str]) -> tuple[List[str], List[str]]:
    service: List[str] = []
    attachments: List[str] = []
    for item in items:
        lowered = item.lower()
        if any(word in lowered for word in ["serve", "service", "affidavit of service", "hearing", "file"]):
            service.append(item)
        if any(word in lowered for word in ["attach", "exhibit", "document", "copy", "proof", "record", "evidence"]):
            attachments.append(item)
    return dedup_keep_order(service), dedup_keep_order(attachments)


def infer_factors(query: str, case_type: str) -> List[str]:
    q = query.lower()
    factors: List[str] = []
    if case_type == "custody":
        factors.append("best_interests")
        if "relocat" in q:
            factors.append("relocation")
        if "school" in q:
            factors.append("educational_needs")
        if "interstate" in q or "out-of-state" in q:
            factors.append("uccjea")
    if case_type == "support":
        factors.append("support_modification")
        if "lost job" in q or "income" in q:
            factors.append("income")
        if "arrear" in q:
            factors.append("arrears")
        if "interstate" in q:
            factors.append("uifsa")
    if case_type == "divorce":
        factors.extend(["equitable_distribution", "financial_need"])
    if case_type == "oop":
        factors.extend(["family_offense", "harassment", "threats"])
    return dedup_keep_order(factors, limit=6)


def generate_procedure_predictions(
    wf_rows: List[Dict[str, Any]],
    llm: Optional[ChatOpenAI],
    vectorstore: Any,
    max_cases: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(wf_rows, start=1):
        if max_cases and idx > max_cases:
            break
        case_id = row.get("id", "")
        query = row.get("fact_pattern", "")
        case_type = row.get("case_type", "general")

        meta: Dict[str, Any] = {}
        response_text = ""
        if llm is not None and vectorstore is not None:
            try:
                meta = run_procedure_engine(query=query, llm=llm, vectorstore=vectorstore)
                response_text = meta.get("final_response", "") if isinstance(meta, dict) else ""
            except Exception:
                response_text = ""

        required_inputs = section_lines(response_text, "Required Inputs You Still Need")
        checklist_lines = section_lines(response_text, "Attachments and Service Checklist")
        service_steps, attachments = split_service_and_attachments(checklist_lines)

        # Fallback to retrieved docs metadata when sections are not present.
        if not attachments and isinstance(meta, dict):
            attachments = dedup_keep_order(
                [f"source:{d.get('source')}" for d in (meta.get("retrieved_docs") or []) if d.get("source")], limit=6
            )

        out.append(
            {
                "id": case_id,
                "predicted_form_family": infer_form_family(query=query, response=response_text, case_type=case_type),
                "predicted_required_inputs": dedup_keep_order(required_inputs, limit=12),
                "predicted_service_steps": dedup_keep_order(service_steps, limit=10),
                "predicted_attachments": dedup_keep_order(attachments, limit=10),
            }
        )
    return out


def generate_research_predictions(
    cr_rows: List[Dict[str, Any]],
    courtlistener_token: str,
    max_results: int,
    max_cases: int,
    heuristic_only: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(cr_rows, start=1):
        if max_cases and idx > max_cases:
            break
        case_id = row.get("id", "")
        query = row.get("fact_pattern", "")
        case_type = row.get("case_type", "general")

        retrieved_cases: List[Dict[str, str]] = []
        if courtlistener_token and not heuristic_only:
            try:
                meta = run_research_engine_structured(
                    query=query,
                    courtlistener_token=courtlistener_token,
                    case_type=case_type if case_type in {"custody", "support", "divorce", "oop", "general"} else "general",
                    max_results=max_results,
                )
                if isinstance(meta, dict) and meta.get("ok"):
                    for case in (meta.get("stage2", {}).get("case_details") or [])[:max_results]:
                        retrieved_cases.append(
                            {
                                "case_name": case.get("case_name", ""),
                                "url": case.get("url", ""),
                                "quote": case.get("quote", ""),
                            }
                        )
            except Exception:
                retrieved_cases = []

        out.append(
            {
                "id": case_id,
                "retrieved_cases": retrieved_cases,
                "predicted_factors": infer_factors(query=query, case_type=case_type),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation prediction JSONL files from current engines.")
    parser.add_argument("--procedure-gold", default="data/eval/procedure_workflow_test.jsonl")
    parser.add_argument("--research-gold", default="data/eval/case_retrieval_test.jsonl")
    parser.add_argument("--procedure-out", default="data/eval/procedure_predictions.jsonl")
    parser.add_argument("--research-out", default="data/eval/research_predictions.jsonl")
    parser.add_argument("--db-dir", default="./db_procedure")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--heuristic-research-only",
        action="store_true",
        help="Do not call CourtListener API; fill research predictions with heuristic factors and empty case list.",
    )
    parser.add_argument(
        "--heuristic-procedure-only",
        action="store_true",
        help="Do not call procedure engine/LLM; fill procedure predictions from query heuristics only.",
    )
    parser.add_argument("--skip-procedure", action="store_true")
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing prediction files and generate only missing IDs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    load_dotenv(dotenv_path=Path(".env"), override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    courtlistener_token = os.getenv("COURTLISTENER_TOKEN", "").strip()

    llm: Optional[ChatOpenAI] = None
    if openai_api_key and not args.heuristic_procedure_only:
        llm = ChatOpenAI(model=args.model, temperature=0)

    if not args.skip_procedure:
        wf_rows = load_jsonl(Path(args.procedure_gold))
        procedure_out_path = Path(args.procedure_out)
        existing_proc_by_id: Dict[str, Dict[str, Any]] = {}
        if args.resume and procedure_out_path.exists():
            for row in load_jsonl(procedure_out_path):
                rid = str(row.get("id", "")).strip()
                if rid:
                    existing_proc_by_id[rid] = row

        wf_target_rows = [row for row in wf_rows if str(row.get("id", "")).strip() not in existing_proc_by_id]
        if args.max_cases > 0:
            wf_target_rows = wf_target_rows[: args.max_cases]

        vectorstore = None
        if llm is not None and not args.heuristic_procedure_only:
            try:
                vectorstore = load_vectorstore(args.db_dir)
            except Exception as exc:
                print(f"Warning: could not load vectorstore ({exc}). Falling back to heuristic procedure predictions.")
                vectorstore = None
        generated_proc = generate_procedure_predictions(
            wf_rows=wf_target_rows,
            llm=llm,
            vectorstore=vectorstore,
            max_cases=0,
        )
        for row in generated_proc:
            rid = str(row.get("id", "")).strip()
            if rid:
                existing_proc_by_id[rid] = row

        ordered_proc = []
        for row in wf_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid in existing_proc_by_id:
                ordered_proc.append(existing_proc_by_id[rid])

        write_jsonl(procedure_out_path, ordered_proc)
        print(
            f"Wrote {len(ordered_proc)} procedure predictions -> {args.procedure_out} "
            f"(generated this run: {len(generated_proc)})"
        )

    if not args.skip_research:
        cr_rows = load_jsonl(Path(args.research_gold))
        research_out_path = Path(args.research_out)
        existing_res_by_id: Dict[str, Dict[str, Any]] = {}
        if args.resume and research_out_path.exists():
            for row in load_jsonl(research_out_path):
                rid = str(row.get("id", "")).strip()
                if rid:
                    existing_res_by_id[rid] = row

        cr_target_rows = [row for row in cr_rows if str(row.get("id", "")).strip() not in existing_res_by_id]
        if args.max_cases > 0:
            cr_target_rows = cr_target_rows[: args.max_cases]

        generated_res = generate_research_predictions(
            cr_rows=cr_target_rows,
            courtlistener_token=courtlistener_token,
            max_results=args.max_results,
            max_cases=0,
            heuristic_only=args.heuristic_research_only,
        )
        for row in generated_res:
            rid = str(row.get("id", "")).strip()
            if rid:
                existing_res_by_id[rid] = row

        ordered_res = []
        for row in cr_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid in existing_res_by_id:
                ordered_res.append(existing_res_by_id[rid])

        write_jsonl(research_out_path, ordered_res)
        print(
            f"Wrote {len(ordered_res)} research predictions -> {args.research_out} "
            f"(generated this run: {len(generated_res)})"
        )


if __name__ == "__main__":
    main()
