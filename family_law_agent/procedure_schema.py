from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


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
    lines = (text or "").splitlines()
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


def detect_procedure_case_type(query: str) -> str:
    q = (query or "").lower()
    if any(token in q for token in ["order of protection", "family offense", "harass", "threat", "violence"]):
        return "oop"
    if any(token in q for token in ["paternity", "parentage", "genetic testing"]):
        return "parentage"
    if any(token in q for token in ["divorce", "spouse", "uncontested", "pendente lite"]):
        return "divorce"
    if any(token in q for token in ["support", "arrears", "uifsa", "income"]):
        return "support"
    if any(token in q for token in ["custody", "visitation", "relocat", "uccjea", "grandparent"]):
        return "custody"
    return "general"


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
        "required_inputs": ["children_information_if_any"],
        "service_steps": [],
        "attachments": [],
    },
    "parentage": {
        "required_inputs": ["child_identity_and_dob"],
        "service_steps": ["serve_respondent", "file_affidavit_of_service"],
        "attachments": ["supporting_affidavit"],
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
        "attachments": ["proposed_parenting_schedule_if_requested"],
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
        "attachments": ["order_copy", "communication_logs_or_school_records"],
    },
    "custody_visitation_petition": {
        "required_inputs": ["requested_legal_and_physical_custody"],
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
        "attachments": ["supporting_affidavit", "evidence_of_existing_relationship"],
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
        "attachments": ["certified_copy_of_out_of_state_order", "supporting_affidavit"],
    },
    "custody_enforcement_petition": {
        "required_inputs": ["current_custody_or_visitation_order"],
        "service_steps": [
            "file_registration_or_enforcement_petition",
            "serve_other_party",
            "file_affidavit_of_service",
            "appear_for_jurisdiction_and_enforcement_hearing",
        ],
    },
    "uccjea_related_affidavit_if_required": {
        "required_inputs": ["child_residence_history", "current_locations_of_parties"],
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
        "attachments": ["support_order_copy", "payment_records_or_dcss_statement"],
    },
    "child_support_petition": {
        "required_inputs": ["parentage_status", "income_information"],
        "service_steps": [
            "file_support_petition",
            "serve_respondent",
            "file_affidavit_of_service",
            "appear_for_support_hearing",
        ],
        "attachments": ["financial_disclosure", "child_expense_documents_if_available"],
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
        "attachments": ["pay_stubs_or_income_records", "tax_return_or_equivalent"],
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
        "attachments": ["certified_support_order_copy", "arrears_statement"],
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
        "attachments": ["child_birth_records_if_available", "supporting_affidavit"],
    },
    "paternity_parentage_petition_if_needed": {
        "required_inputs": ["parentage_status"],
        "attachments": ["child_birth_records_if_available"],
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
        "attachments": ["supporting_affidavit", "any_prior_parentage_documents"],
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
        "attachments": ["incident_evidence_messages_photos_reports_if_available"],
    },
    "order_of_protection_extension_petition_or_motion": {
        "required_inputs": [
            "existing_order_of_protection",
            "new_or_continuing_incidents",
            "requested_extension_duration",
        ],
        "attachments": ["copy_of_existing_order_of_protection"],
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
        "attachments": ["copy_of_existing_order", "supporting_affidavit_or_evidence"],
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
        "attachments": ["affidavit_of_service", "affidavit_of_default", "proposed_judgment"],
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
        "attachments": ["proposed_judgment_with_name_change_language"],
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
        "required_inputs": ["income_and_expense_information", "requested_temporary_relief_terms"],
    },
    "temporary_custody_request": {
        "required_inputs": ["children_information", "requested_temporary_relief_terms"],
        "attachments": ["proposed_temporary_parenting_schedule"],
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


def build_structured_procedure_schema(
    query: str,
    response_text: str,
    case_type: Optional[str] = None,
    retrieved_docs: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    resolved_case_type = case_type or detect_procedure_case_type(query)
    predicted_forms = infer_form_family(query=query, response=response_text, case_type=resolved_case_type)
    required_inputs = section_lines(response_text, "Required Inputs You Still Need")
    workflow_lines = section_lines(response_text, "Step-by-Step Workflow")
    checklist_lines = section_lines(response_text, "Attachments and Service Checklist")
    service_steps, attachments = split_service_and_attachments(workflow_lines + checklist_lines)

    schema_hints = infer_schema_hints(
        case_type=resolved_case_type,
        predicted_forms=predicted_forms,
        query=query,
        response_text=response_text,
    )

    if not attachments and retrieved_docs:
        attachments = dedup_keep_order(
            [f"source:{doc.get('source')}" for doc in retrieved_docs if doc.get("source")],
            limit=6,
        )

    return {
        "case_type": resolved_case_type,
        "predicted_form_family": predicted_forms,
        "predicted_required_inputs": dedup_keep_order(
            list(required_inputs) + schema_hints["required_inputs"],
            limit=20,
        ),
        "predicted_service_steps": dedup_keep_order(
            list(service_steps) + schema_hints["service_steps"],
            limit=20,
        ),
        "predicted_attachments": dedup_keep_order(
            list(attachments) + schema_hints["attachments"],
            limit=20,
        ),
    }
