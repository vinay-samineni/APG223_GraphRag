import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from apg223_rag.config import (
    EMBED_MODEL_NAME,
    SEMANTIC_BREAK_PERCENTILE,
    MAX_CHILD_CHARS,
    CHUNK_OVERLAP_CHARS,
)


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<!\d)(?<!\b[A-Z])\.(?!\d)\s+(?=[A-Z0-9])', text)
    return [s.strip() if s.endswith(".") else f"{s.strip()}." for s in sentences if s.strip()]


def build_child_chunks(
    parents: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    embed_model: SentenceTransformer | None = None
) -> List[Dict[str, Any]]:
    """
    1. Splits Parent prose into sentences and detects semantic topic shifts via BGE cosine similarity.
    2. Applies RecursiveCharacterTextSplitter (1500 chars, 200 overlap) to oversized semantic groups.
    3. Converts isolated tables into dedicated table child chunks without slicing rows.
    """
    model = embed_model or SentenceTransformer(EMBED_MODEL_NAME)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHILD_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    children: List[Dict[str, Any]] = []
    chunk_counter = 1

    # 1. Prose Semantic + Recursive Chunking
    for parent in parents:
        # Exclude appended [Table ...] block from prose chunking since tables are handled separately
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
            matched_page = parent["page_start"]
            matched_bbox = parent["blocks"][0]["bbox"] if parent["blocks"] else []
            prefix = c_text[:35]
            for blk in parent["blocks"]:
                if prefix in blk["text"]:
                    matched_page = blk["page"]
                    matched_bbox = blk["bbox"]
                    break

            children.append({
                "chunk_id": f"C{chunk_counter:03d}",
                "chunk_type": "text",
                "parent_id": parent["parent_id"],
                "section": parent["section"],
                "subsection": parent["subsection"],
                "page": matched_page,
                "bbox": matched_bbox,
                "content": c_text
            })
            chunk_counter += 1

    # 2. Dedicated Table Child Chunks
    for tbl in tables:
        children.append({
            "chunk_id": f"C{chunk_counter:03d}",
            "chunk_type": "table",
            "parent_id": tbl.get("parent_id", "P001"),
            "section": tbl.get("section", "General"),
            "subsection": tbl.get("subsection", "Overview"),
            "page": tbl["page"],
            "bbox": tbl["bbox"],
            "content": tbl["content"]
        })
        chunk_counter += 1

    return children