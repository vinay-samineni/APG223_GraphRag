import re
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from apg223_rag.config import (
    EMBED_MODEL_NAME,
    SEMANTIC_BREAK_PERCENTILE,
    MAX_CHILD_CHARS,
    CHUNK_OVERLAP_CHARS,
)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<!\d)(?<!\b[A-Z])(?<=[.?!])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in parts if s.strip()]


def _locate_span_metadata(chunk_text: str, parent: Dict[str, Any]) -> Tuple[int, List[float]]:
    spans = parent.get("block_spans", [])
    if not spans:
        return parent["page_start"], []

    pos = parent["content"].find(chunk_text[:80])
    if pos == -1:
        return spans[0]["page"], spans[0]["bbox"]

    for sp in spans:
        if sp["start_char"] <= pos <= sp["end_char"]:
            return sp["page"], sp["bbox"]

    return spans[-1]["page"], spans[-1]["bbox"]


def build_child_chunks(
    parents: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    embed_model: SentenceTransformer | None = None,
) -> List[Dict[str, Any]]:
    model = embed_model or SentenceTransformer(EMBED_MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHILD_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    children: List[Dict[str, Any]] = []
    chunk_counter = 1

    for parent in parents:
        prose_only = parent["content"].split("\n\n[Table ")[0]
        sentences = _split_sentences(prose_only)
        if not sentences:
            continue

        if len(sentences) == 1:
            semantic_groups = [sentences[0]]
        else:
            embeddings = model.encode(sentences, normalize_embeddings=True)
            sims = [float(np.dot(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
            threshold = np.percentile(sims, SEMANTIC_BREAK_PERCENTILE)

            semantic_groups = []
            curr_group = [sentences[0]]
            for idx, sim in enumerate(sims):
                if sim < threshold and len(" ".join(curr_group)) >= 400:
                    semantic_groups.append(" ".join(curr_group))
                    curr_group = [sentences[idx + 1]]
                else:
                    curr_group.append(sentences[idx + 1])
            if curr_group:
                semantic_groups.append(" ".join(curr_group))

        final_texts = []
        for grp in semantic_groups:
            if len(grp) > MAX_CHILD_CHARS:
                final_texts.extend(splitter.split_text(grp))
            else:
                final_texts.append(grp)

        for c_text in final_texts:
            matched_page, matched_bbox = _locate_span_metadata(c_text, parent)
            children.append({
                "chunk_id": f"C{chunk_counter:03d}",
                "chunk_type": "text",
                "parent_id": parent["parent_id"],
                "section": parent["section"],
                "subsection": parent["subsection"],
                "page": matched_page,
                "bbox": matched_bbox,
                "content": c_text,
            })
            chunk_counter += 1

    for tbl in tables:
        if "parent_id" not in tbl:
            continue
        children.append({
            "chunk_id": f"C{chunk_counter:03d}",
            "chunk_type": "table",
            "parent_id": tbl["parent_id"],
            "section": tbl["section"],
            "subsection": tbl["subsection"],
            "page": tbl["page"],
            "bbox": tbl["bbox"],
            "content": tbl["content"],
        })
        chunk_counter += 1

    return children