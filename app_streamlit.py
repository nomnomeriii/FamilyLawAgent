from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from family_law_agent.procedure import ingest_procedure_documents, load_vectorstore, run_procedure_engine
from family_law_agent.research import detect_query_intent, low_relevance, run_research_engine
from family_law_agent.safety import safety_classifier

DEFAULT_FORMS_DIR = "./NY Family Law Forms"
DEFAULT_DB_DIR = "./db_procedure"

# Load local environment variables from project root (if present).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


@st.cache_resource
def cached_vectorstore(db_dir: str):
    return load_vectorstore(db_dir)


def _procedure_sources(procedure_meta: dict, limit: int = 8) -> list[str]:
    out = []
    for doc in (procedure_meta.get("retrieved_docs") or [])[:limit]:
        tag = doc.get("tag", "S?")
        source = doc.get("source", "Unknown source")
        page = doc.get("page")
        page_suffix = f" (page {page})" if page is not None else ""
        out.append(f"- [{tag}] {source}{page_suffix}")
    return out


def _research_sources(research_meta: dict, limit: int = 8) -> list[str]:
    out = []
    if not research_meta.get("ok"):
        return out
    for case in (research_meta.get("stage2", {}).get("case_details") or [])[:limit]:
        case_name = case.get("case_name", "Unknown Case")
        url = case.get("url", "")
        quote = (case.get("quote") or "").strip()
        if url:
            line = f"- [{case_name}]({url})"
        else:
            line = f"- {case_name}"
        if quote:
            line += f' | Quote: "{quote}"'
        out.append(line)
    return out


def build_workflow_checklist_markdown(case_type_label: str, user_query: str, procedure_meta: dict) -> str:
    procedure_text = procedure_meta.get("final_response", "")
    if not procedure_meta.get("ok"):
        procedure_text = f"Procedure engine failed: {procedure_meta.get('error', 'Unknown error')}."
    procedure_sources = _procedure_sources(procedure_meta)
    source_block = "\n".join(procedure_sources) if procedure_sources else "- None returned."
    return (
        f"### Workflow Checklist ({case_type_label})\n"
        f"**User question:** {user_query}\n\n"
        f"{procedure_text}\n\n"
        f"### Procedure Sources\n{source_block}\n"
    )


def build_case_research_markdown(research_meta: dict) -> str:
    research_text = research_meta.get("final_response", "")
    if not research_meta.get("ok"):
        research_text = f"Research engine failed: {research_meta.get('error', 'Unknown error')}."
    research_sources = _research_sources(research_meta)
    source_block = "\n".join(research_sources) if research_sources else "- None returned."
    return (
        "### Case Research\n"
        f"{research_text}\n\n"
        f"### Case Sources\n{source_block}\n"
    )


def build_draft_outline_markdown(case_type_label: str, user_query: str, research_meta: dict) -> str:
    case_sources = _research_sources(research_meta, limit=3)
    cited_authorities = "\n".join(case_sources) if case_sources else "- Add authorities after legal review."
    return (
        f"### Draft Outline ({case_type_label})\n"
        f"**Matter summary:** {user_query}\n\n"
        "1) Caption and Court\n"
        "- Court: New York Family Court (County: [TODO])\n"
        "- Petitioner: [TODO]\n"
        "- Respondent: [TODO]\n\n"
        "2) Relief Requested\n"
        "- Primary relief: [TODO]\n"
        "- Temporary/emergency relief (if needed): [TODO]\n\n"
        "3) Key Facts to Plead\n"
        "- Child and party background: [TODO]\n"
        "- Timeline of material events: [TODO]\n"
        "- Prior orders / case numbers: [TODO]\n\n"
        "4) Legal Basis and Factors\n"
        "- Applicable NY standards and factors: [TODO]\n"
        "- Fact-to-factor mapping: [TODO]\n\n"
        "5) Evidence and Attachments\n"
        "- Exhibits list: [TODO]\n"
        "- Service documents: [TODO]\n"
        "- Financial records / supporting docs: [TODO]\n\n"
        "6) Proposed Order Language\n"
        "- Requested terms: [TODO]\n"
        "- Compliance / enforcement terms: [TODO]\n\n"
        "### Suggested Authorities\n"
        f"{cited_authorities}\n\n"
        "_General legal information only. Not legal advice._\n"
    )


def build_filing_packet_markdown(
    case_type_label: str,
    user_query: str,
    workflow_md: str,
    research_md: str,
    draft_md: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "# Filing Packet\n\n"
        f"- Generated: {ts}\n"
        f"- Case Type: {case_type_label}\n"
        f"- User Question: {user_query}\n\n"
        "---\n\n"
        f"{workflow_md}\n\n"
        "---\n\n"
        f"{research_md}\n\n"
        "---\n\n"
        f"{draft_md}\n"
    )


def main() -> None:
    st.set_page_config(page_title="NY Family Law Agent", layout="wide")
    st.title("Family Law Litigation Lawyer Agent")
    st.markdown(
        "Bridging the New York Family Court justice gap with procedure-guided and citation-grounded assistance."
    )

    st.sidebar.header("Configuration")
    env_openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_courtlistener_token = os.getenv("COURTLISTENER_TOKEN", "").strip()

    if env_openai_api_key:
        openai_api_key = env_openai_api_key
        st.sidebar.caption("OPENAI_API_KEY loaded from environment.")
    else:
        st.sidebar.warning("OPENAI_API_KEY not found in environment. Add it in .env or enter it below.")
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password").strip()

    if env_courtlistener_token:
        courtlistener_token = env_courtlistener_token
        st.sidebar.caption("COURTLISTENER_TOKEN loaded from environment.")
    else:
        st.sidebar.warning("COURTLISTENER_TOKEN not found in environment. Entering it is recommended.")
        courtlistener_token = st.sidebar.text_input("CourtListener Token", type="password").strip()

    forms_dir = st.sidebar.text_input("Forms Directory", value=DEFAULT_FORMS_DIR)
    db_dir = st.sidebar.text_input("Procedure DB Directory", value=DEFAULT_DB_DIR)

    if st.sidebar.button("Build / Refresh Procedure DB"):
        try:
            with st.spinner("Indexing local NY family law forms..."):
                result = ingest_procedure_documents(forms_directory=forms_dir, persist_directory=db_dir)
            cached_vectorstore.clear()
            if result.get("ok"):
                st.sidebar.success(
                    f"Indexed {result['chunks_indexed']} chunks from {result['documents_loaded']} documents."
                )
            else:
                st.sidebar.error(result.get("error", "Procedure ingestion failed."))
        except Exception as exc:
            st.sidebar.error(f"Procedure ingestion failed: {exc}")

    jurisdiction = st.sidebar.selectbox("Jurisdiction", ["New York", "California", "Texas", "Other"])
    max_results = st.sidebar.slider("Research retrieval count", 3, 5, 5)

    case_type_map = {
        "Custody": "custody",
        "Support": "support",
        "Divorce": "divorce",
        "Order of Protection": "oop",
        "General": "general",
    }
    type_to_label = {
        "custody": "Custody",
        "support": "Support",
        "divorce": "Divorce",
        "oop": "Order of Protection",
        "general": "General",
    }

    if jurisdiction != "New York":
        st.error("Jurisdiction gate: this agent is currently scoped only for New York family law.")
        st.stop()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am the NY Family Law Litigation Agent. "
                    "I can help with procedure guidance and case-law-grounded legal information.\n\n"
                    "First, choose your case type below to get started."
                ),
            }
        ]
    if "selected_case_type_label" not in st.session_state:
        st.session_state["selected_case_type_label"] = None
    if "latest_run_debug" not in st.session_state:
        st.session_state["latest_run_debug"] = None
    if "latest_workspace" not in st.session_state:
        st.session_state["latest_workspace"] = None

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state["selected_case_type_label"]:
        selected_case_type_label = st.selectbox(
            "Case Type",
            ["Custody", "Support", "Divorce", "Order of Protection", "General"],
            key="onboarding_case_type",
        )
        if st.button("Start Conversation"):
            st.session_state["selected_case_type_label"] = selected_case_type_label
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        f"Great, we will use **{selected_case_type_label}**.\n\n"
                        "What question do you want to ask?"
                    ),
                }
            )
            st.rerun()
        st.stop()

    selected_case_type_label = st.session_state["selected_case_type_label"]
    case_type = case_type_map[selected_case_type_label]

    st.caption(f"Active case type: {selected_case_type_label}")
    if st.button("Change Case Type"):
        st.session_state["selected_case_type_label"] = None
        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": "No problem. Please choose a new case type below."}
        )
        st.rerun()

    user_query = st.chat_input("Type your NY family law question")
    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not openai_api_key:
            error_msg = "OpenAI API key is required. Add OPENAI_API_KEY in .env or enter it in the sidebar."
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state["chat_messages"].append({"role": "assistant", "content": error_msg})
            return

        is_safe, safety_message = safety_classifier(user_query)
        if not is_safe:
            with st.chat_message("assistant"):
                st.error(safety_message)
            st.session_state["chat_messages"].append({"role": "assistant", "content": safety_message})
            return

        os.environ["OPENAI_API_KEY"] = openai_api_key
        if courtlistener_token:
            os.environ["COURTLISTENER_TOKEN"] = courtlistener_token
        effective_courtlistener_token = (courtlistener_token or os.getenv("COURTLISTENER_TOKEN", "")).strip() or None

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing your question with procedure and case-law retrieval..."):
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    vectorstore = cached_vectorstore(db_dir)

                    procedure_meta = run_procedure_engine(user_query, llm, vectorstore)
                    research_meta = run_research_engine(
                        user_query,
                        llm,
                        courtlistener_token=effective_courtlistener_token,
                        case_type=case_type,
                        max_results=max_results,
                    )

                    used_case_type = case_type
                    if case_type != "general" and low_relevance(research_meta):
                        fallback_meta = run_research_engine(
                            user_query,
                            llm,
                            courtlistener_token=effective_courtlistener_token,
                            case_type="general",
                            max_results=max_results,
                        )
                        if fallback_meta.get("ok") and not low_relevance(fallback_meta):
                            research_meta = fallback_meta
                            used_case_type = "general"
            except Exception as exc:
                err_text = str(exc)
                if "invalid_api_key" in err_text or "Incorrect API key provided" in err_text:
                    friendly = (
                        "OpenAI API key is invalid (401). Check OPENAI_API_KEY in .env and remove stale shell values. "
                        "Then restart Streamlit."
                    )
                else:
                    friendly = f"Engine execution failed: {exc}"
                st.error(friendly)
                st.session_state["chat_messages"].append({"role": "assistant", "content": friendly})
                return

            pred_type, pred_score, _ = detect_query_intent(user_query)
            note_lines = []
            if pred_type != "general" and pred_type != case_type and pred_score >= 2:
                note_lines.append(
                    f"Note: your question looks closer to **{type_to_label[pred_type]}** than "
                    f"the selected **{selected_case_type_label}**."
                )
            if used_case_type != case_type:
                note_lines.append(
                    "Note: research retrieval switched to **General** because selected type had low relevance."
                )
            if research_meta.get("fallback_mode") == "general_info_only":
                note_lines.append("Note: citation retrieval failed, so the research response is general information.")

            procedure_text = procedure_meta.get("final_response", "")
            if not procedure_meta.get("ok"):
                procedure_text = f"Procedure engine failed: {procedure_meta.get('error', 'Unknown error')}."

            research_text = research_meta.get("final_response", "")
            if not research_meta.get("ok"):
                research_text = f"Research engine failed: {research_meta.get('error', 'Unknown error')}."

            procedure_sources = []
            for doc in procedure_meta.get("retrieved_docs", [])[:5]:
                tag = doc.get("tag", "PROC")
                source = doc.get("source", "Unknown source")
                page = doc.get("page")
                page_suffix = f" (page {page})" if page is not None else ""
                procedure_sources.append(f"- [{tag}] {source}{page_suffix}")

            research_sources = []
            if research_meta.get("ok"):
                for case in research_meta.get("stage2", {}).get("case_details", [])[:5]:
                    case_name = case.get("case_name", "Unknown Case")
                    url = case.get("url", "")
                    quote = (case.get("quote") or "").strip()
                    if url:
                        case_line = f"- [{case_name}]({url})"
                    else:
                        case_line = f"- {case_name}"
                    if quote:
                        case_line += f' | Quote: "{quote}"'
                    research_sources.append(case_line)

            sources_parts = []
            sources_parts.append("### Retrieved Data / Sources Used")
            if procedure_sources:
                sources_parts.append("**Procedure Sources**")
                sources_parts.extend(procedure_sources)
            else:
                sources_parts.append("**Procedure Sources**")
                sources_parts.append("- None returned.")
            if research_sources:
                sources_parts.append("**Case Law Sources**")
                sources_parts.extend(research_sources)
            else:
                sources_parts.append("**Case Law Sources**")
                sources_parts.append("- None returned.")
            sources_block = "\n".join(sources_parts)

            assistant_reply_parts = []
            if note_lines:
                assistant_reply_parts.append("\n".join(note_lines))
            assistant_reply_parts.append(f"### Procedure Guidance\n{procedure_text}")
            assistant_reply_parts.append(f"### Case Law Research\n{research_text}")
            assistant_reply_parts.append(sources_block)
            assistant_reply = "\n\n".join(assistant_reply_parts)

            st.markdown(assistant_reply)
            st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_reply})
            st.session_state["latest_run_debug"] = {
                "selected_case_type": case_type,
                "selected_case_type_label": selected_case_type_label,
                "used_case_type": used_case_type,
                "procedure_meta": procedure_meta,
                "research_meta": research_meta,
                "user_query": user_query,
            }

            workflow_md = build_workflow_checklist_markdown(selected_case_type_label, user_query, procedure_meta)
            research_md = build_case_research_markdown(research_meta)
            draft_md = build_draft_outline_markdown(selected_case_type_label, user_query, research_meta)
            packet_md = build_filing_packet_markdown(
                case_type_label=selected_case_type_label,
                user_query=user_query,
                workflow_md=workflow_md,
                research_md=research_md,
                draft_md=draft_md,
            )
            st.session_state["latest_workspace"] = {
                "case_type_label": selected_case_type_label,
                "user_query": user_query,
                "workflow_md": workflow_md,
                "research_md": research_md,
                "draft_md": draft_md,
                "packet_md": packet_md,
            }

    latest_workspace = st.session_state.get("latest_workspace")
    if latest_workspace:
        st.markdown("---")
        st.subheader("Case Workspace")
        tab_workflow, tab_research, tab_draft = st.tabs(["Workflow Checklist", "Case Research", "Draft Outline"])
        with tab_workflow:
            st.markdown(latest_workspace["workflow_md"])
        with tab_research:
            st.markdown(latest_workspace["research_md"])
        with tab_draft:
            st.markdown(latest_workspace["draft_md"])

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        safe_case_type = latest_workspace["case_type_label"].replace(" ", "_").lower()
        st.download_button(
            "Export Filing Packet (.md)",
            data=latest_workspace["packet_md"],
            file_name=f"filing_packet_{safe_case_type}_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    latest_run_debug = st.session_state.get("latest_run_debug")
    if latest_run_debug:
        procedure_meta = latest_run_debug["procedure_meta"]
        research_meta = latest_run_debug["research_meta"]
        used_case_type = latest_run_debug["used_case_type"]
        selected_case_type = latest_run_debug["selected_case_type"]

        st.markdown("---")
        st.subheader("Latest Run Debug")
        st.caption(f"Selected case type: {selected_case_type} | Used for retrieval: {used_case_type}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Procedure Engine Debug**")
            with st.expander("Refined Prompt (Procedure)"):
                st.text(procedure_meta.get("refined_prompt", ""))
            with st.expander("Retrieved Data (Procedure)"):
                for doc in procedure_meta.get("retrieved_docs", []):
                    page = doc.get("page")
                    page_suffix = f" (page {page})" if page is not None else ""
                    st.markdown(f"**[{doc['tag']}] Source:** {doc['source']}{page_suffix}")
                    st.write(doc["snippet"])
                    st.markdown("---")
            with st.expander("LLM Prompt Context Sent (Procedure)"):
                st.text(procedure_meta.get("retrieved_context_for_prompt", ""))
            with st.expander("Structured Schema (Procedure)"):
                st.json(procedure_meta.get("structured_schema", {}))

        with col2:
            st.markdown("**Research Engine Debug**")
            if research_meta.get("retrieval_mode"):
                mode = research_meta["retrieval_mode"]
                st.caption(
                    f"Retrieval mode: case_type={mode.get('case_type')} | "
                    f"court={mode.get('court') or 'ALL'} | order_by={mode.get('order_by')}"
                )

            with st.expander("Refined Prompt (Research)"):
                st.text(research_meta.get("refined_prompt", ""))

            if research_meta.get("search_errors"):
                with st.expander("Research Retrieval Errors (Debug)"):
                    for err in research_meta["search_errors"]:
                        st.write(f"- {err}")

            if research_meta.get("ok"):
                with st.expander("Stage 1 Retrieved Candidates"):
                    st.write({"candidate_count": research_meta.get("stage1", {}).get("candidate_count", 0)})
                    for i, case in enumerate(research_meta.get("stage1", {}).get("top_candidates", []), start=1):
                        st.markdown(f"**{i}. {case.get('case_name', 'Unknown Case')}**")
                        st.write(f"Search URL: {case.get('search_url', '')}")
                        st.write(f"Cluster API: {case.get('cluster_api', '')}")
                        st.write(f"Opinion API: {case.get('opinion_api', '')}")
                        st.write(f"Family Score: {case.get('family_score', '')}")
                        st.markdown("---")

                with st.expander("Final Evidence Used (Stage 2)"):
                    for i, case in enumerate(research_meta["stage2"]["case_details"], start=1):
                        st.markdown(f"**{i}. {case.get('case_name', 'Unknown Case')}**")
                        st.write(f"Citation URL: {case.get('url', '')}")
                        st.write(f"Quote: \"{case.get('quote') or 'No verifiable quote in retrieved context.'}\"")
                        st.write(f"Errors: {case.get('errors', [])}")
                        st.markdown("---")

            if research_meta.get("ok"):
                details = research_meta["stage2"]["case_details"]
                quote_backed = sum(1 for case in details if case.get("quote"))
                st.write(
                    {
                        "selected_case_type": selected_case_type,
                        "used_case_type": used_case_type,
                        "final_cases": research_meta["stage2"]["count"],
                        "quote_backed_cases": quote_backed,
                        "quality_primary_count": research_meta["stage2"]["quality_primary_count"],
                        "quality_tertiary_count": research_meta["stage2"]["quality_tertiary_count"],
                    }
                )

    st.markdown("---")
    st.caption("Model layer: OpenAI + CourtListener + Chroma")


if __name__ == "__main__":
    main()
