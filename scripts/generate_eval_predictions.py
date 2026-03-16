from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
else:
    ChatOpenAI = Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from family_law_agent.procedure_schema import build_structured_procedure_schema


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
    if ("paternity" in q or "parentage" in q) and any(token in q for token in ["dispute", "contest", "respond"]):
        labels.extend(["paternity_response_or_motion", "request_for_genetic_testing"])

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
    if "divorce" in q and "never responded" in q:
        labels.append("default_divorce_application")
    if "default" in q and "support" in q and "vacate" in q:
        labels.append("motion_to_vacate_default_support_order")
    if "out-of-state" in q and "custody" in q and any(token in q for token in ["register", "enforce"]):
        labels.extend(["uccjea_registration_or_enforcement_petition", "custody_enforcement_petition"])
    if "out-of-state" in q and "support" in q and "enforcement" in q:
        labels.append("uifsa_registration_and_enforcement_petition")

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


CASE_TYPE_SCHEMA_HINTS: Dict[str, Dict[str, List[str]]] = {
    "custody": {
        "required_inputs": [
            "petitioner_identity",
            "respondent_identity",
            "child_identity_and_dob",
            "prior_orders_and_case_numbers",
        ],
        "service_steps": [
            "file_in_proper_family_court_county",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_at_scheduled_hearing",
        ],
        "attachments": [
            "copy_of_prior_order_if_any",
            "supporting_affidavit_or_verified_petition",
        ],
    },
    "support": {
        "required_inputs": [
            "petitioner_identity",
            "respondent_identity",
            "child_identity_and_dob",
            "prior_orders_and_case_numbers",
        ],
        "service_steps": [
            "file_in_proper_family_court_county",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_at_scheduled_hearing",
        ],
        "attachments": [
            "copy_of_prior_order_if_any",
            "supporting_affidavit_or_verified_petition",
        ],
    },
    "oop": {
        "required_inputs": [
            "petitioner_identity",
            "respondent_identity",
            "child_identity_and_dob",
            "prior_orders_and_case_numbers",
        ],
        "service_steps": [
            "file_in_proper_family_court_county",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_at_scheduled_hearing",
        ],
        "attachments": [
            "copy_of_prior_order_if_any",
            "supporting_affidavit_or_verified_petition",
        ],
    },
    "divorce": {
        "required_inputs": [
            "children_information_if_any",
        ],
        "service_steps": [],
        "attachments": [],
    },
    "parentage": {
        "required_inputs": [
            "child_identity_and_dob",
        ],
        "service_steps": [
            "serve_respondent",
            "file_affidavit_of_service",
        ],
        "attachments": [
            "supporting_affidavit",
        ],
    },
}


FORM_SCHEMA_HINTS: Dict[str, Dict[str, List[str]]] = {
    "relocation_request_motion": {
        "required_inputs": [
            "proposed_relocation_details",
            "employment_offer_or_work_change",
            "current_parenting_schedule",
            "proposed_long_distance_parenting_plan",
            "international_relocation_plan",
            "passport_and_travel_information",
            "proposed_parenting_time_after_relocation",
        ],
        "attachments": [
            "proof_supporting_relocation_reason",
            "proposed_parenting_plan",
            "travel_and_school_plan_documents",
            "communication_and_access_plan",
        ],
    },
    "initial_custody_visitation_petition": {
        "required_inputs": [
            "residence_history",
            "requested_legal_and_physical_custody",
            "safety_concerns_if_any",
        ],
        "attachments": [
            "proposed_parenting_schedule_if_requested",
        ],
    },
    "custody_visitation_modification_petition": {
        "required_inputs": [
            "current_custody_order",
            "substantial_change_in_circumstances",
            "requested_schedule_change",
        ],
        "attachments": [
            "current_custody_order_copy",
            "school_or_schedule_supporting_documents",
        ],
    },
    "custody_visitation_enforcement_or_modification_petition": {
        "required_inputs": [
            "current_custody_or_visitation_order",
            "specific_denial_incidents",
            "requested_makeup_or_modified_schedule",
        ],
        "attachments": [
            "order_copy",
            "communication_logs_or_school_records",
        ],
    },
    "custody_visitation_petition": {
        "required_inputs": [
            "requested_legal_and_physical_custody",
        ],
    },
    "emergency_order_to_show_cause_temporary_custody": {
        "required_inputs": [
            "immediate_risk_facts",
            "requested_emergency_relief",
            "supporting_safety_details",
        ],
        "attachments": [
            "emergency_affidavit",
            "supporting_evidence_reports_messages_photos_if_any",
        ],
    },
    "grandparent_visitation_petition": {
        "required_inputs": [
            "grandparent_relationship_facts",
            "existing_custody_orders",
            "basis_for_best_interests_claim",
        ],
        "service_steps": [
            "file_petition",
            "serve_parents_or_guardians",
            "file_affidavit_of_service",
            "appear_for_hearing",
        ],
        "attachments": [
            "supporting_affidavit",
            "evidence_of_existing_relationship",
        ],
    },
    "uccjea_registration_or_enforcement_petition": {
        "required_inputs": [
            "out_of_state_order_copy_certified_if_possible",
            "child_residence_history",
            "current_locations_of_parties",
            "requested_enforcement_relief",
        ],
        "service_steps": [
            "file_registration_or_enforcement_petition",
            "serve_other_party",
            "file_affidavit_of_service",
            "appear_for_jurisdiction_and_enforcement_hearing",
        ],
        "attachments": [
            "certified_copy_of_out_of_state_order",
            "supporting_affidavit",
        ],
    },
    "custody_enforcement_petition": {
        "required_inputs": [
            "current_custody_or_visitation_order",
        ],
        "service_steps": [
            "file_registration_or_enforcement_petition",
            "serve_other_party",
            "file_affidavit_of_service",
            "appear_for_jurisdiction_and_enforcement_hearing",
        ],
    },
    "uccjea_related_affidavit_if_required": {
        "required_inputs": [
            "child_residence_history",
            "current_locations_of_parties",
        ],
    },
    "child_support_modification_petition": {
        "required_inputs": [
            "current_support_order",
            "income_change_details",
            "other_parent_income_increase_facts",
            "financial_disclosure",
            "employment_status_evidence",
        ],
        "attachments": [
            "copy_of_current_support_order",
            "copy_of_support_order",
            "income_and_benefits_documents",
            "income_increase_evidence_if_available",
        ],
    },
    "child_support_violation_enforcement_petition": {
        "required_inputs": [
            "current_support_order",
            "arrears_amount_estimate",
            "payment_history",
        ],
        "attachments": [
            "support_order_copy",
            "payment_records_or_dcss_statement",
        ],
    },
    "child_support_petition": {
        "required_inputs": [
            "parentage_status",
            "income_information",
        ],
        "service_steps": [
            "file_support_petition",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_for_support_hearing",
        ],
        "attachments": [
            "financial_disclosure",
            "child_expense_documents_if_available",
        ],
    },
    "support_response_and_financial_disclosure": {
        "required_inputs": [
            "current_petition_copy",
            "income_information",
            "expense_information",
            "dependents_information",
        ],
        "service_steps": [
            "file_response_or_appearance",
            "exchange_or_file_financial_disclosure",
            "serve_other_side_if_required",
            "appear_at_support_hearing",
        ],
        "attachments": [
            "pay_stubs_or_income_records",
            "tax_return_or_equivalent",
        ],
    },
    "uifsa_registration_and_enforcement_petition": {
        "required_inputs": [
            "out_of_state_support_order",
            "arrears_information",
            "obligor_identifying_information",
            "payment_history",
        ],
        "service_steps": [
            "file_uifsa_registration_or_enforcement",
            "serve_obligor",
            "file_affidavit_of_service",
            "appear_on_enforcement_date",
        ],
        "attachments": [
            "certified_support_order_copy",
            "arrears_statement",
        ],
    },
    "motion_to_vacate_default_support_order": {
        "required_inputs": [
            "default_order_copy",
            "service_defect_facts",
            "meritorious_defense_summary",
            "timing_of_motion",
        ],
        "service_steps": [
            "file_motion_to_vacate",
            "serve_other_party",
            "file_proof_of_service",
            "appear_on_motion_date",
        ],
        "attachments": [
            "default_order_copy",
            "supporting_affidavit",
            "proof_re_service_issue_if_any",
        ],
    },
    "paternity_parentage_petition": {
        "required_inputs": [
            "mother_identity",
            "alleged_father_identity",
            "facts_showing_parentage",
            "requested_relief_support_and_parentage_order",
        ],
        "service_steps": [
            "file_parentage_petition",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_for_hearing_or_genetic_testing",
        ],
        "attachments": [
            "child_birth_records_if_available",
            "supporting_affidavit",
        ],
    },
    "paternity_parentage_petition_if_needed": {
        "required_inputs": [
            "parentage_status",
        ],
        "attachments": [
            "child_birth_records_if_available",
        ],
    },
    "paternity_response_or_motion": {
        "required_inputs": [
            "respondent_identity",
            "child_identity",
            "case_number",
            "grounds_for_contesting_parentage",
        ],
        "service_steps": [
            "file_response_or_appearance",
            "serve_other_party_if_required",
            "appear_for_genetic_testing_or_hearing",
        ],
        "attachments": [
            "supporting_affidavit",
            "any_prior_parentage_documents",
        ],
    },
    "request_for_genetic_testing": {
        "service_steps": [
            "file_response_or_appearance",
            "appear_for_genetic_testing_or_hearing",
        ],
    },
    "family_offense_petition": {
        "required_inputs": [
            "specific_incidents_dates_locations",
            "relationship_basis_for_family_court_jurisdiction",
            "requested_protection_terms",
            "incident_timeline",
            "threat_or_harassment_facts",
            "requested_stay_away_or_no_contact_terms",
        ],
        "attachments": [
            "incident_evidence_messages_photos_reports_if_available",
            "messages_emails_or_other_incident_evidence_if_available",
        ],
    },
    "temporary_order_of_protection_request": {
        "attachments": [
            "incident_evidence_messages_photos_reports_if_available",
        ],
    },
    "order_of_protection_extension_petition_or_motion": {
        "required_inputs": [
            "existing_order_of_protection",
            "new_or_continuing_incidents",
            "requested_extension_duration",
        ],
        "attachments": [
            "copy_of_existing_order_of_protection",
        ],
    },
    "order_of_protection_modification_petition_or_motion": {
        "required_inputs": [
            "existing_order_of_protection",
            "requested_modified_terms",
            "facts_supporting_modification",
        ],
        "service_steps": [
            "file_modification_request",
            "serve_respondent_if_required",
            "file_affidavit_of_service",
            "appear_for_modification_hearing",
        ],
        "attachments": [
            "copy_of_existing_order",
            "supporting_affidavit_or_evidence",
        ],
    },
    "uncontested_divorce_packet": {
        "required_inputs": [
            "marriage_date_and_place",
            "residency_requirements_facts",
            "grounds_or_no_fault_basis",
            "children_information_if_any",
            "agreement_terms",
        ],
        "service_steps": [
            "file_summons_and_complaint_or_joint_packet",
            "serve_defendant_if_required",
            "file_affidavit_of_service",
            "submit_uncontested_judgment_package",
        ],
        "attachments": [
            "settlement_agreement_if_any",
            "child_support_worksheet_if_children",
            "proposed_judgment_documents",
        ],
    },
    "default_divorce_application": {
        "required_inputs": [
            "index_number",
            "proof_of_service",
            "proof_of_default",
            "requested_judgment_terms",
        ],
        "service_steps": [
            "file_default_package",
            "serve_additional_default_notices_if_required",
            "submit_proposed_judgment",
        ],
        "attachments": [
            "affidavit_of_service",
            "affidavit_of_default",
            "proposed_judgment",
        ],
    },
    "divorce_judgment_name_change_request": {
        "required_inputs": [
            "current_legal_name",
            "requested_restored_or_new_name",
            "divorce_case_information",
        ],
        "service_steps": [
            "include_name_change_request_in_divorce_submission",
            "serve_if_required_by_case_posture",
        ],
        "attachments": [
            "proposed_judgment_with_name_change_language",
        ],
    },
    "pendente_lite_motion_for_temporary_relief": {
        "required_inputs": [
            "pending_divorce_index_number",
            "children_information",
            "income_and_expense_information",
            "requested_temporary_relief_terms",
        ],
        "service_steps": [
            "file_motion_in_pending_divorce_case",
            "serve_opposing_party_per_court_rules",
            "file_proof_of_service",
            "appear_on_motion_return_date",
        ],
        "attachments": [
            "financial_affidavit",
            "proposed_temporary_parenting_schedule",
            "supporting_affidavit",
        ],
    },
    "temporary_child_support_request": {
        "required_inputs": [
            "income_and_expense_information",
            "requested_temporary_relief_terms",
        ],
    },
    "temporary_custody_request": {
        "required_inputs": [
            "children_information",
            "requested_temporary_relief_terms",
        ],
        "attachments": [
            "proposed_temporary_parenting_schedule",
        ],
    },
}


def infer_schema_hints(case_type: str, predicted_forms: Iterable[str], query: str, response_text: str) -> Dict[str, List[str]]:
    text = f"{query} {response_text}".lower()
    hints = {
        "required_inputs": list(CASE_TYPE_SCHEMA_HINTS.get(case_type, {}).get("required_inputs", [])),
        "service_steps": list(CASE_TYPE_SCHEMA_HINTS.get(case_type, {}).get("service_steps", [])),
        "attachments": list(CASE_TYPE_SCHEMA_HINTS.get(case_type, {}).get("attachments", [])),
    }

    for form in predicted_forms:
        schema = FORM_SCHEMA_HINTS.get(form, {})
        hints["required_inputs"].extend(schema.get("required_inputs", []))
        hints["service_steps"].extend(schema.get("service_steps", []))
        hints["attachments"].extend(schema.get("attachments", []))

    if "job" in text or "unemploy" in text:
        hints["required_inputs"].extend(["income_change_details", "employment_status_evidence"])
        hints["attachments"].append("income_and_benefits_documents")
    if "salary increase" in text or "income increase" in text:
        hints["required_inputs"].append("other_parent_income_increase_facts")
        hints["attachments"].append("income_increase_evidence_if_available")
    if "arrear" in text or "non-payment" in text or "not paid" in text:
        hints["required_inputs"].extend(["arrears_amount_estimate", "payment_history"])
        hints["attachments"].append("payment_records_or_dcss_statement")
    if "relocat" in text:
        hints["required_inputs"].extend(["proposed_relocation_details", "proposed_long_distance_parenting_plan"])
        hints["attachments"].extend(["proof_supporting_relocation_reason", "proposed_parenting_plan"])
    if "threat" in text or "harass" in text:
        hints["required_inputs"].append("specific_incidents_dates_locations")
        hints["attachments"].append("incident_evidence_messages_photos_reports_if_available")
    if "support petition" in text and "response" in text:
        hints["service_steps"].extend(["file_response_or_appearance", "appear_at_support_hearing"])
    if "proof of service" in text or "affidavit of service" in text:
        hints["service_steps"].extend(["file_affidavit_of_service", "file_proof_of_service"])

    return {k: dedup_keep_order(v, limit=20) for k, v in hints.items()}


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
            from family_law_agent.procedure import run_procedure_engine

            try:
                meta = run_procedure_engine(query=query, llm=llm, vectorstore=vectorstore)
                response_text = meta.get("final_response", "") if isinstance(meta, dict) else ""
            except Exception:
                response_text = ""

        structured_schema = build_structured_procedure_schema(
            query=query,
            response_text=response_text,
            case_type=case_type,
            retrieved_docs=(meta.get("retrieved_docs") or []) if isinstance(meta, dict) else None,
        )
        if isinstance(meta, dict) and isinstance(meta.get("structured_schema"), dict):
            structured_schema = meta["structured_schema"]

        out.append(
            {
                "id": case_id,
                "predicted_form_family": structured_schema.get("predicted_form_family", []),
                "predicted_required_inputs": structured_schema.get("predicted_required_inputs", []),
                "predicted_service_steps": structured_schema.get("predicted_service_steps", []),
                "predicted_attachments": structured_schema.get("predicted_attachments", []),
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
            from family_law_agent.research import run_research_engine_structured

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
        from langchain_openai import ChatOpenAI as _ChatOpenAI

        llm = _ChatOpenAI(model=args.model, temperature=0)

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
                from family_law_agent.procedure import load_vectorstore

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
