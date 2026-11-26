import re
import pymupdf
from typing import List, Dict, Any, Tuple
from apg223_rag.config import LAYOUT


def _overlaps_table(
    block_bbox: Tuple[float, float, float, float],
    table_bboxes: List[Tuple[float, float, float, float]],
    threshold: float = 0.25,
) -> bool:
    bx0, by0, bx1, by1 = block_bbox
    b_area = max((bx1 - bx0) * (by1 - by0), 1e-5)

    for tx0, ty0, tx1, ty1 in table_bboxes:
        ix0, iy0 = max(bx0, tx0), max(by0, ty0)
        ix1, iy1 = min(bx1, tx1), min(by1, ty1)
        if ix1 > ix0 and iy1 > iy0:
            if ((ix1 - ix0) * (iy1 - iy0)) / b_area >= threshold:
                return True
    return False


def parse_prose_and_footnotes(
    pdf_path: str,
    table_bboxes_by_page: Dict[int, List[Tuple[float, float, float, float]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = pymupdf.open(pdf_path)
    prose_blocks: List[Dict[str, Any]] = []
    footnotes: List[Dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num < LAYOUT.start_content_page or page_num > LAYOUT.end_content_page:
            continue

        page_tables = table_bboxes_by_page.get(page_num, [])
        blocks = doc[page_idx].get_text("dict", sort=True)["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue

            bbox = tuple(round(c, 2) for c in b["bbox"])
            if bbox[1] < LAYOUT.header_margin_y or bbox[3] > LAYOUT.footer_margin_y:
                continue
            if page_tables and _overlaps_table(bbox, page_tables):
                continue

            text_parts, font_sizes, flags = [], [], []
            for line in b["lines"]:
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if txt:
                        text_parts.append(txt)
                        font_sizes.append(round(span["size"], 1))
                        flags.append(span["flags"])

            if not text_parts:
                continue

            full_text = " ".join(text_parts)
            if full_text == "APRA" or re.fullmatch(r"\d+", full_text):
                continue

            dominant_size = max(set(font_sizes), key=font_sizes.count)
            is_bold = any(f & 16 for f in flags)

            if dominant_size <= LAYOUT.footnote_max_font:
                fn_match = re.match(r"^(\d{1,2})\s+(.*)", full_text)
                footnotes.append({
                    "footnote_num": fn_match.group(1) if fn_match else None,
                    "page": page_num,
                    "font_size": dominant_size,
                    "bbox": list(bbox),
                    "text": full_text,
                })
                continue

            if dominant_size >= LAYOUT.h1_min_font:
                tag = "H1"
            elif LAYOUT.h2_min_font <= dominant_size < LAYOUT.h1_min_font and is_bold:
                tag = "H2"
            else:
                tag = "BODY"

            prose_blocks.append({
                "page": page_num,
                "text": full_text,
                "font_size": dominant_size,
                "bbox": list(bbox),
                "tag": tag,
            })

    doc.close()
    return prose_blocks, footnotes


def parse_glossary_terms(pdf_path: str) -> List[Dict[str, Any]]:
    doc = pymupdf.open(pdf_path)
    entries: List[Dict[str, Any]] = []

    for page_idx in range(LAYOUT.glossary_start_page - 1, len(doc)):
        page_num = page_idx + 1
        blocks = doc[page_idx].get_text("dict", sort=True)["blocks"]
        terms_col, defs_col = [], []

        for b in blocks:
            if "lines" not in b:
                continue
            bbox = b["bbox"]
            if bbox[1] < LAYOUT.header_margin_y or bbox[3] > LAYOUT.footer_margin_y:
                continue

            txt = " ".join(
                span["text"].strip()
                for line in b["lines"]
                for span in line["spans"]
                if span["text"].strip()
            )
            if not txt or txt in {"APRA", "Glossary"} or re.fullmatch(r"\d+", txt):
                continue

            if bbox[0] < 210.0:
                terms_col.append((bbox[1], page_num, txt))
            else:
                defs_col.append((bbox[1], page_num, txt))

        for y_term, p_num, term_text in terms_col:
            best_def = min(
                defs_col,
                key=lambda d: abs(d[0] - y_term) if d[1] == p_num else 1e6,
                default=None,
            )
            if best_def and abs(best_def[0] - y_term) < 65.0:
                entries.append({
                    "term": term_text,
                    "definition": best_def[2],
                    "page": p_num,
                })

    doc.close()
    return entries