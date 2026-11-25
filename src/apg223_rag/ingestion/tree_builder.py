import re
from typing import List, Dict, Any, Tuple

PARA_START_RE = re.compile(r"^(\d{1,3})\.\s+")


def _merge_split_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive BODY blocks belonging to the same APRA numbered paragraph
    when a paragraph spans across a page break or visual block split.
    """
    merged: List[Dict[str, Any]] = []
    for blk in blocks:
        if blk["tag"] != "BODY" or not merged or merged[-1]["tag"] != "BODY":
            merged.append(dict(blk))
            continue

        prev = merged[-1]
        starts_new_para = bool(PARA_START_RE.match(blk["text"]))
        starts_sub_bullet = bool(re.match(r"^[a-z]\)\s+", blk["text"]))

        # If it does not start a new numbered paragraph or lettered sub-clause, join to previous block
        if not starts_new_para and not starts_sub_bullet and not prev["text"].endswith(":"):
            prev["text"] = f"{prev['text']} {blk['text']}"
            prev["page_end"] = blk["page"]
        else:
            merged.append(dict(blk))
    return merged


def build_document_tree_and_parents(
    prose_blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    footnotes: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds the hierarchical Document Tree, creates one Parent Node per subsection,
    and attaches matching tables and footnotes to their parent subsection.
    """
    clean_blocks = _merge_split_paragraphs(prose_blocks)
    doc_tree: Dict[str, Dict[str, List[str]]] = {}
    parents: List[Dict[str, Any]] = []

    current_h1 = "General"
    current_h2 = "Overview"
    buffer_blocks: List[Dict[str, Any]] = []
    parent_counter = 1

    def flush_parent(sec: str, subsec: str, blks: List[Dict[str, Any]]):
        nonlocal parent_counter
        if not blks:
            return
        parent_id = f"P{parent_counter:03d}"
        parent_counter += 1

        page_start = blks[0]["page"]
        page_end = blks[-1].get("page_end", blks[-1]["page"])
        page_fns = [fn["text"] for fn in footnotes if page_start <= fn["page"] <= page_end]

        parents.append({
            "parent_id": parent_id,
            "section": sec,
            "subsection": subsec,
            "page_start": page_start,
            "page_end": page_end,
            "y_start": blks[0]["bbox"][1],
            "content": "\n\n".join(b["text"] for b in blks),
            "footnotes": page_fns,
            "blocks": blks
        })
        doc_tree.setdefault(sec, {})[subsec] = [b["text"] for b in blks]

    for blk in clean_blocks:
        if blk["tag"] == "H1":
            flush_parent(current_h1, current_h2, buffer_blocks)
            buffer_blocks = []
            current_h1 = blk["text"]
            current_h2 = f"{current_h1} - Introduction"
        elif blk["tag"] == "H2":
            flush_parent(current_h1, current_h2, buffer_blocks)
            buffer_blocks = []
            current_h2 = blk["text"]
        else:
            buffer_blocks.append(blk)

    flush_parent(current_h1, current_h2, buffer_blocks)

    # Link isolated tables to their enclosing Parent subsection
    for tbl in tables:
        t_page = tbl["page"]
        t_y0 = tbl["bbox"][1]
        matched_parent = parents[0] if parents else None

        for p in parents:
            if p["page_start"] <= t_page <= p["page_end"]:
                if t_page == p["page_start"] and t_y0 < p["y_start"]:
                    continue
                matched_parent = p

        if matched_parent:
            tbl["parent_id"] = matched_parent["parent_id"]
            tbl["section"] = matched_parent["section"]
            tbl["subsection"] = matched_parent["subsection"]
            matched_parent["content"] += f"\n\n[Table {tbl['table_id']}]\n{tbl['content']}"

    return doc_tree, parents, tables