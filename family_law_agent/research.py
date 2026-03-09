from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
COURTLISTENER_CLUSTERS_URL = "https://www.courtlistener.com/api/rest/v4/clusters/"
COURTLISTENER_OPINIONS_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"

CASE_TYPE_TERMS: Dict[str, List[str]] = {
    "general": [],
    "custody": ["custody", "visitation", "parenting time", "best interests", "relocation"],
    "support": ["child support", "support modification", "substantial change in circumstances", "income"],
    "divorce": ["domestic relations", "equitable distribution", "maintenance", "spousal support"],
    "oop": ["order of protection", "family offense", "harassment", "threats"],
}

FAMILY_POS = [
    "family court",
    "custody",
    "visitation",
    "relocation",
    "child support",
    "domestic relations",
    "best interests",
    "order of protection",
]

FAMILY_NEG = [
    "people v.",
    "criminal",
    "habeas",
    "correctional",
    "indictment",
    "sentencing",
    "conviction",
    "parole",
    "probation",
]

INTENT_HINTS = {
    "custody": ["custody", "visitation", "parenting", "relocation", "best interests"],
    "support": ["child support", "support modification", "income", "arrears"],
    "divorce": ["divorce", "equitable distribution", "spousal support", "maintenance"],
    "oop": ["order of protection", "family offense", "threat", "harass", "violence"],
}


def detect_query_intent(query: str) -> Tuple[str, int, Dict[str, int]]:
    lowered = (query or "").lower()
    scores = {k: sum(1 for keyword in kws if keyword in lowered) for k, kws in INTENT_HINTS.items()}
    best = max(scores, key=scores.get) if scores else "general"
    best_score = scores.get(best, 0)
    if best_score == 0:
        return "general", 0, scores
    return best, best_score, scores


def build_family_query(user_query: str, case_type: str = "general") -> str:
    base = (
        '(family OR "family court" OR custody OR visitation OR relocation OR '
        '"child support" OR "domestic relations" OR "order of protection" OR "best interests")'
    )
    extra_terms = CASE_TYPE_TERMS.get(case_type, [])
    boosts: List[str] = []

    if "relocat" in (user_query or "").lower():
        boosts = ['"tropea"']

    scoped: List[str] = []
    if extra_terms:
        scoped.extend([f'"{term}"' if " " in term else term for term in extra_terms])
    scoped.extend(boosts)

    if scoped:
        return f"{user_query} {base} ({' OR '.join(scoped)})"
    return f"{user_query} {base}"


def family_score(text: str) -> int:
    lowered = (text or "").lower()
    return sum(token in lowered for token in FAMILY_POS) - 2 * sum(token in lowered for token in FAMILY_NEG)


def _to_abs_url(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    if text.startswith("/"):
        return "https://www.courtlistener.com" + text
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return ""


def _extract_id(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    if text.isdigit():
        return text
    match = re.search(r"/(\d+)/?$", text)
    return match.group(1) if match else ""


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _best_quote(raw_text: str, case_type: str, query: str, limit: int = 320) -> str:
    text = _clean_text(raw_text)
    if not text:
        return ""

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return text[:limit]

    query_terms = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4]
    target_terms = set(FAMILY_POS + CASE_TYPE_TERMS.get(case_type, []) + query_terms)

    scored: List[Tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        hits = sum(term in lowered for term in target_terms)
        scored.append((hits, index, sentence))

    scored.sort(reverse=True)
    hits, index, sentence = scored[0]

    if hits <= 0:
        return sentence[:limit]

    out = sentence + (" " + sentences[index + 1] if index + 1 < len(sentences) else "")
    return (out[:limit] + "...") if len(out) > limit else out


def _request_json(url: str, token: Optional[str], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    response = requests.get(url, headers=headers, params=params, timeout=20)

    # If caller provided an invalid/expired token, retry anonymously once so
    # public search can still work for debugging/demo scenarios.
    if response.status_code in (401, 403) and token:
        anon_response = requests.get(url, headers={"Accept": "application/json"}, params=params, timeout=20)
        anon_response.raise_for_status()
        return anon_response.json()

    response.raise_for_status()
    return response.json()


def _build_cluster_api(result: Dict[str, Any]) -> str:
    raw = result.get("cluster")
    if raw:
        text = str(raw)
        if "/api/rest/v4/clusters/" in text:
            return _to_abs_url(text)
        cluster_id = _extract_id(text)
        if cluster_id:
            return f"{COURTLISTENER_CLUSTERS_URL}{cluster_id}/"

    cluster_id = _extract_id(result.get("cluster_id"))
    if cluster_id:
        return f"{COURTLISTENER_CLUSTERS_URL}{cluster_id}/"
    return ""


def _build_opinion_api(result: Dict[str, Any]) -> str:
    resource = result.get("resource_uri")
    if resource and "/api/rest/v4/opinions/" in str(resource):
        return _to_abs_url(resource)

    opinions = result.get("opinions")
    if isinstance(opinions, list) and opinions:
        first = opinions[0]
        text = str(first)
        if "/api/rest/v4/opinions/" in text:
            return _to_abs_url(text)
        opinion_id = _extract_id(text)
        if opinion_id:
            return f"{COURTLISTENER_OPINIONS_URL}{opinion_id}/"

    opinion_id = _extract_id(result.get("opinion_id"))
    if opinion_id:
        return f"{COURTLISTENER_OPINIONS_URL}{opinion_id}/"

    fallback_id = _extract_id(result.get("id"))
    if fallback_id:
        return f"{COURTLISTENER_OPINIONS_URL}{fallback_id}/"
    return ""


def _fetch_case_page_text(url: str) -> str:
    if not url:
        return ""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
            timeout=20,
        )
        response.raise_for_status()
        html = response.text or ""
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        return _clean_text(html)
    except Exception:
        return ""


def dedupe_details(details: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for detail in details:
        key = detail.get("url") or detail.get("case_name", "").lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(detail)
    return output


def _quote_tokens(text: str) -> Set[str]:
    lowered = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {token for token in lowered.split() if len(token) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def diversify_cases(cases: Sequence[Dict[str, Any]], max_results: int = 5, threshold: float = 0.72) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    signatures: List[Set[str]] = []

    for case in cases:
        signature = _quote_tokens(case.get("quote", ""))
        similarity = max((_jaccard(signature, existing) for existing in signatures), default=0.0)
        if signature and similarity >= threshold:
            dropped.append(case)
            continue

        kept.append(case)
        signatures.append(signature)
        if len(kept) >= max_results:
            break

    if len(kept) < max_results:
        need = max_results - len(kept)
        kept.extend(dropped[:need])

    return {"kept": kept[:max_results], "dropped_similar": dropped}


def low_relevance(meta: Dict[str, Any]) -> bool:
    if not isinstance(meta, dict) or not meta.get("ok"):
        return True
    details = meta.get("stage2", {}).get("case_details", [])
    quote_backed = sum(1 for case in details if case.get("quote"))
    top_score = max((case.get("family_score", -99) for case in details), default=-99)
    return quote_backed < 1 or top_score < 1


def _search_passes(
    query: str, case_type: str, token: Optional[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    plans = [
        {
            "name": "scoped_family_with_courts",
            "case_type": case_type,
            "query": build_family_query(query, case_type),
            "court": "ny nyappdiv nyappterm nyfamct",
            "order_by": "score desc",
            "type": "o",
        },
        {
            "name": "general_family_with_courts",
            "case_type": "general",
            "query": build_family_query(query, "general"),
            "court": "ny nyappdiv nyappterm nyfamct",
            "order_by": "score desc",
            "type": "o",
        },
        {
            "name": "general_family_state_appellate",
            "case_type": "general",
            "query": build_family_query(query, "general"),
            "court": "ny nyappdiv nyappterm",
            "order_by": "score desc",
            "type": "o",
        },
        {
            "name": "general_family_all_courts",
            "case_type": "general",
            "query": build_family_query(query, "general"),
            "court": "",
            "order_by": "score desc",
            "type": "o",
        },
        {
            "name": "raw_query_all_courts",
            "case_type": "general",
            "query": query,
            "court": "",
            "order_by": "score desc",
            "type": "o",
        },
        {
            "name": "raw_query_no_type",
            "case_type": "general",
            "query": query,
            "court": "",
            "order_by": "score desc",
            "type": "",
        },
    ]

    search_errors: List[str] = []
    for plan in plans:
        params = {
            "q": plan["query"],
            "order_by": plan["order_by"],
            "page_size": 30,
        }
        if plan["type"]:
            params["type"] = plan["type"]
        if plan["court"]:
            params["court"] = plan["court"]

        try:
            data = _request_json(COURTLISTENER_SEARCH_URL, token, params=params)
            raw = data.get("results", [])
            if raw:
                return raw, plan, search_errors
        except Exception as exc:
            search_errors.append(f"{plan['name']}: {exc}")
            continue

    return [], {"case_type": "general", "court": "", "order_by": "score desc"}, search_errors


def run_research_engine_structured(query: str, courtlistener_token: Optional[str], case_type: str = "general", max_results: int = 5) -> Dict[str, Any]:
    token = (courtlistener_token or "").strip() or None
    raw, used_plan, search_errors = _search_passes(query, case_type, token)

    if not raw:
        return {
            "ok": False,
            "error": "Stage 1 search failed: no results across retrieval passes.",
            "retrieval_mode": used_plan,
            "search_errors": search_errors,
        }

    candidates = []
    seen = set()
    for row in raw[: max_results * 4]:
        case_name = row.get("caseName") or row.get("case_name") or "Unknown Case"
        cluster_api = _build_cluster_api(row)
        opinion_api = _build_opinion_api(row)
        search_url = _to_abs_url(row.get("absolute_url"))
        key = cluster_api or opinion_api or search_url or case_name.lower()

        if key in seen:
            continue

        seen.add(key)
        candidates.append(
            {
                "case_name": case_name,
                "cluster_api": cluster_api,
                "opinion_api": opinion_api,
                "search_url": search_url,
                "search_snippet": row.get("snippet") or "",
                "family_score": family_score(f"{case_name} {row.get('snippet') or ''}"),
            }
        )

    candidates.sort(key=lambda item: item["family_score"], reverse=True)

    details = []
    for candidate in candidates[: max_results * 4]:
        errors: List[str] = []
        cluster: Dict[str, Any] = {}
        opinion: Dict[str, Any] = {}

        try:
            if candidate["opinion_api"]:
                opinion = _request_json(candidate["opinion_api"], token)
        except Exception as exc:
            errors.append(f"opinion_fetch_failed: {exc}")

        try:
            if candidate["cluster_api"]:
                cluster = _request_json(candidate["cluster_api"], token)
        except Exception as exc:
            errors.append(f"cluster_fetch_failed: {exc}")

        if not opinion:
            try:
                sub_opinions = cluster.get("sub_opinions") or []
                if sub_opinions:
                    op_url = _to_abs_url(sub_opinions[0])
                    if op_url:
                        opinion = _request_json(op_url, token)
            except Exception as exc:
                errors.append(f"sub_opinion_fetch_failed: {exc}")

        raw_text = (
            opinion.get("plain_text")
            or opinion.get("html_with_citations")
            or opinion.get("html")
            or cluster.get("headmatter")
            or cluster.get("syllabus")
            or candidate["search_snippet"]
        )

        if not raw_text:
            raw_text = _fetch_case_page_text(candidate["search_url"])

        quote = _best_quote(raw_text, used_plan["case_type"], query=query)
        case_name = cluster.get("case_name") or candidate["case_name"]

        details.append(
            {
                "case_name": case_name,
                "url": _to_abs_url(cluster.get("absolute_url"))
                or candidate["search_url"]
                or candidate["cluster_api"]
                or candidate["opinion_api"],
                "quote": quote,
                "errors": errors,
                "family_score": family_score(f"{case_name} {quote}"),
            }
        )

    details = dedupe_details(details)
    primary = [detail for detail in details if detail["quote"] and not detail["errors"]]
    secondary = [detail for detail in details if detail["quote"] and detail["errors"]]
    tertiary = [detail for detail in details if not detail["quote"]]

    ordered = sorted(primary, key=lambda item: item["family_score"], reverse=True)
    ordered += sorted(secondary, key=lambda item: item["family_score"], reverse=True)
    ordered += sorted(tertiary, key=lambda item: item["family_score"], reverse=True)

    diverse = diversify_cases(ordered, max_results=max_results)
    final_cases = diverse["kept"]

    context = ""
    for i, case in enumerate(final_cases, start=1):
        quote = case["quote"] if case["quote"] else "No verifiable quote in retrieved context."
        context += f"[C{i}] {case['case_name']} | URL: {case['url']}\n[C{i}] Quote: \"{quote}\"\n\n"

    return {
        "ok": True,
        "stage1": {
            "candidate_count": len(candidates),
            "top_candidates": candidates[:max_results],
        },
        "stage2": {
            "count": len(final_cases),
            "case_details": final_cases,
            "quality_primary_count": len(primary),
            "quality_secondary_count": len(secondary),
            "quality_tertiary_count": len(tertiary),
            "dropped_similar_count": len(diverse["dropped_similar"]),
        },
        "retrieved_case_context_for_prompt": context,
        "retrieval_mode": used_plan,
    }


def run_research_engine(
    query: str,
    llm: Any,
    courtlistener_token: Optional[str],
    case_type: str = "general",
    max_results: int = 5,
) -> Dict[str, Any]:
    result = run_research_engine_structured(
        query=query,
        courtlistener_token=courtlistener_token,
        case_type=case_type,
        max_results=max_results,
    )

    if not result.get("ok"):
        result["refined_prompt"] = ""
        result["final_response"] = result.get("error", "Research engine failed.")
        return result

    details = result.get("stage2", {}).get("case_details", [])
    quote_backed = sum(1 for case in details if case.get("quote"))

    if len(details) == 0 or quote_backed == 0:
        result["refined_prompt"] = ""
        result["final_response"] = (
            "I could not retrieve reliable NY case citations for this query right now. "
            "Here is general informational guidance (not case-citation grounded):\n"
            "1) For relocation/custody, NY courts evaluate best interests.\n"
            "2) Common factors include reason for move, impact on parent-child relationship, and feasibility of parenting time.\n"
            "3) Consider consulting NY Family Court forms/professional legal aid."
        )
        result["fallback_mode"] = "general_info_only"
        return result

    context = result["retrieved_case_context_for_prompt"]
    prompt = f"""
You are a New York family-law case research assistant.
Scope and safety:
- Jurisdiction is strictly New York.
- Provide informational research only, not legal advice.
- If evidence is weak, explicitly say uncertainty.

Task:
From the retrieved case context ONLY, identify relevant cases, summarize holdings/factors, and map factors to user facts.
Never fabricate quotes or case facts.

Required output format:
1) Top Relevant Cases (ranked)
2) Holding and Key Factors by Case
3) Fact-to-Factor Mapping
4) Quoted Support
5) Source Citations
6) Limits and Disclaimer

Quote rules:
- Use exact quotes from retrieved context.
- Cite each quote as [C1], [C2], etc.
- If no reliable quote exists, write: "No verifiable quote in retrieved context."

Retrieved Context:
{context}

User Query:
{query}
""".strip()

    synthesis = llm.invoke(prompt)
    response_text = synthesis.content if hasattr(synthesis, "content") else str(synthesis)

    result["refined_prompt"] = prompt
    result["final_response"] = response_text
    return result
