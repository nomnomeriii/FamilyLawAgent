from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROCEDURE_PROMPT_TEMPLATE = """
You are a New York family-law procedure assistant.
Scope and safety:
- Jurisdiction is strictly New York.
- This is general legal information, not legal advice.
- Never provide coaching for harm, harassment, evasion of court orders, hiding assets, or wrongdoing.
- If required NY-specific support is missing, ask for missing facts or say the source is unavailable.

Task:
Using only the retrieved NY forms/statutes/help-guide context, produce a filing workflow.
Do not invent forms, steps, deadlines, or statutory requirements.

Required output format:
1) Likely Filing Path
2) Required Inputs You Still Need
3) Step-by-Step Workflow
4) Attachments and Service Checklist
5) Source Citations
6) Limits and Disclaimer

Citation rules:
- Every legal/procedural claim must include inline citations like [S1], [S2].
- If unsupported, say "insufficient source support".

Context (each source is tagged; cite only these tags): {context}
User Query: {question}
""".strip()

FAMILY_LAW_HINTS = {
    "custody": ["custody", "visitation", "parenting", "relocation", "best interests"],
    "support": ["child support", "income", "modification", "arrears"],
    "divorce": ["divorce", "maintenance", "equitable distribution"],
    "oop": ["order of protection", "family offense", "harassment", "threat"],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {token for token in tokens if len(token) >= 3 and token not in STOPWORDS}


def _expand_retrieval_queries(query: str) -> list[str]:
    queries = [query]
    lowered = (query or "").lower()

    for intent, hints in FAMILY_LAW_HINTS.items():
        if any(hint in lowered for hint in hints):
            queries.append(f"{query} New York family court {intent} forms instructions")
            break

    queries.append(f"{query} New York family court forms instructions")
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))


def _doc_unique_key(doc: Any) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", ""))
    chunk = str(doc.metadata.get("chunk_id", ""))
    head = (doc.page_content or "")[:120].strip().lower()
    return f"{source}|{page}|{chunk}|{head}"


def _select_procedure_docs(vectorstore: Chroma, query: str, k: int) -> list[Any]:
    retrieval_queries = _expand_retrieval_queries(query)
    dense_candidates: list[Any] = []

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(10, k * 3), "fetch_k": max(30, k * 8)},
    )
    sim_retriever = vectorstore.as_retriever(search_kwargs={"k": max(8, k * 2)})

    for retrieval_query in retrieval_queries:
        for retriever in (mmr_retriever, sim_retriever):
            try:
                dense_candidates.extend(retriever.invoke(retrieval_query))
            except Exception:
                continue

    seen = set()
    deduped: list[Any] = []
    for doc in dense_candidates:
        key = _doc_unique_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    if not deduped:
        return []

    query_tokens = _tokenize(" ".join(retrieval_queries))
    ranked = []
    total = len(deduped)
    for idx, doc in enumerate(deduped):
        doc_tokens = _tokenize(doc.page_content or "")
        overlap = len(query_tokens & doc_tokens)
        overlap_ratio = (overlap / max(1, len(query_tokens))) if query_tokens else 0.0
        dense_rank_boost = (total - idx) / total
        score = 0.65 * dense_rank_boost + 0.35 * overlap_ratio
        ranked.append((score, idx, doc))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [item[2] for item in ranked[:k]]


def ingest_procedure_documents(
    forms_directory: str,
    persist_directory: str = "./db_procedure",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """Ingest NY forms and persist a Chroma vector index."""
    if not os.path.isdir(forms_directory):
        raise FileNotFoundError(f"Forms directory not found: {forms_directory}")

    pdf_loader = PyPDFDirectoryLoader(forms_directory)
    txt_loader = DirectoryLoader(forms_directory, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True)

    raw_documents = []
    raw_documents.extend(pdf_loader.load())
    raw_documents.extend(txt_loader.load())

    if not raw_documents:
        return {
            "ok": False,
            "error": "No PDFs/TXT files found in the forms directory.",
            "forms_directory": forms_directory,
        }

    for doc in raw_documents:
        source_path = str(doc.metadata.get("source", "unknown"))
        doc.metadata["source_path"] = source_path
        doc.metadata["source"] = os.path.basename(source_path) if source_path else "unknown"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(raw_documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)

    return {
        "ok": True,
        "forms_directory": forms_directory,
        "persist_directory": persist_directory,
        "documents_loaded": len(raw_documents),
        "chunks_indexed": len(chunks),
    }


def load_vectorstore(persist_directory: str = "./db_procedure") -> Optional[Chroma]:
    if not os.path.exists(persist_directory):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def run_procedure_engine(query: str, llm: Any, vectorstore: Optional[Chroma], k: int = 5) -> Dict[str, Any]:
    if vectorstore is None:
        return {
            "ok": False,
            "error": "Vector database not found. Build procedure DB before querying.",
        }

    prompt = PromptTemplate(
        template=PROCEDURE_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    docs = _select_procedure_docs(vectorstore=vectorstore, query=query, k=k)
    if not docs:
        return {
            "ok": False,
            "error": "No procedure sources matched this query. Try rephrasing or rebuild the procedure DB.",
        }

    context_blocks = []
    retrieved_docs = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "unknown")
        source_path = doc.metadata.get("source_path", source)
        page_part = f" | page {page}" if page is not None else ""
        snippet = (doc.page_content or "").strip()

        context_blocks.append(f"[S{i}] Source: {source}{page_part}\n{snippet}")
        retrieved_docs.append(
            {
                "tag": f"S{i}",
                "source": source,
                "source_path": source_path,
                "page": page,
                "snippet": snippet[:700],
            }
        )
    context = "\n\n".join(context_blocks)

    try:
        answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": query})
    except Exception as exc:
        return {"ok": False, "error": f"Procedure engine failed: {exc}"}

    return {
        "ok": True,
        "refined_prompt": PROCEDURE_PROMPT_TEMPLATE,
        "retrieved_docs": retrieved_docs,
        "retrieved_context_for_prompt": context,
        "final_response": answer,
    }
