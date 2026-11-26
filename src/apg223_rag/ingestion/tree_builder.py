import re
from typing import List, Dict, Any, Tuple
from apg223_rag.config import MAX_PARENT_CHARS, PARENT_OVERLAP_PARAS

PARA_START_RE = re.compile(r"^(\d{1,3})\.\s+")


def _merge_split_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for blk in blocks:
        if blk["tag"] != "BODY" or not merged or merged[-1]["tag"] != "BODY":
            merged.append(dict(blk))
            continue

        prev = merged[-1]
        starts_new_para = bool(PARA_START_RE.match(blk["text"]))
        starts_sub_bullet = bool(re.match(r"^[a-z]\)\s+", blk["text"]))

        if not starts_new_para and not starts_sub_bullet and not prev["text"].endswith(":"):
            prev["text"] = f"{prev['text']} {blk['text']}"
            prev["page_end"] = blk["page"]
        else:
            merged.append(dict(blk))
    return merged


def build_document_tree_and_parents(
    prose_blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    footnotes: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    clean_blocks = _merge_split_paragraphs(prose_blocks)
    doc_tree: Dict[str, Dict[str, List[str]]] = {}
    parents: List[Dict[str, Any]] = []

    current_h1 = "General"
    current_h2 = "Overview"
    buffer_blocks: List[Dict[str, Any]] = []
    parent_counter = 1

    def _emit_parent_window(sec: str, subsec: str, window_blks: List[Dict[str, Any]], suffix: str = ""):
        nonlocal parent_counter
        parent_id = f"P{parent_counter:03d}{suffix}"
        if not suffix or suffix == "_1":
            parent_counter += 1

        offset = 0
        block_spans = []
        for b in window_blks:
            start_char = offset
            end_char = start_char + len(b["text"])
            block_spans.append({
                "start_char": start_char,
                "end_char": end_char,
                "page": b["page"],
                "bbox": b["bbox"],
            })
            offset = end_char + 2

        full_content = "\n\n".join(b["text"] for b in window_blks)
        page_start = window_blks[0]["page"]
        page_end = window_blks[-1].get("page_end", window_blks[-1]["page"])

        matched_fns = []
        for fn in footnotes:
            if page_start <= fn["page"] <= page_end:
                fn_num = fn.get("footnote_num")
                if fn_num and re.search(rf"(?<=[a-zA-Z\).]){re.escape(fn_num)}\b", full_content):
                    matched_fns.append(fn["text"])

        parents.append({
            "parent_id": parent_id,
            "section": sec,
            "subsection": subsec,
            "page_start": page_start,
            "page_end": page_end,
            "y_start": window_blks[0]["bbox"][1],
            "char_length": len(full_content),
            "content": full_content,
            "footnotes": matched_fns,
            "block_spans": block_spans,
        })

    def flush_parent(sec: str, subsec: str, blks: List[Dict[str, Any]]):
        if not blks:
            return

        doc_tree.setdefault(sec, {}).setdefault(subsec, []).extend(b["text"] for b in blks)

        total_len = sum(len(b["text"]) + 2 for b in blks)
        if total_len <= MAX_PARENT_CHARS:
            _emit_parent_window(sec, subsec, blks)
            return

        windows: List[List[Dict[str, Any]]] = []
        curr_window: List[Dict[str, Any]] = []
        curr_len = 0

        for b in blks:
            b_len = len(b["text"]) + 2
            if curr_window and (curr_len + b_len) > MAX_PARENT_CHARS:
                windows.append(curr_window)
                overlap = curr_window[-PARENT_OVERLAP_PARAS:] if PARENT_OVERLAP_PARAS > 0 else []
                curr_window = list(overlap)
                curr_len = sum(len(x["text"]) + 2 for x in curr_window)

            curr_window.append(b)
            curr_len += b_len

        if curr_window:
            windows.append(curr_window)

        for idx, win in enumerate(windows, start=1):
            _emit_parent_window(sec, subsec, win, suffix=f"_{idx}")

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

    for tbl in tables:
        t_page = tbl["page"]
        t_y0 = tbl["bbox"][1]
        matched_parent = None

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