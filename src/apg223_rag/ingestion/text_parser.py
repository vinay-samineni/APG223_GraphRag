import re
import pymupdf
from typing import List, Dict, Any, Tuple
from apg223_rag.config import (
    START_CONTENT_PAGE,
    HEADER_MARGIN_Y,
    FOOTER_MARGIN_Y,
    H1_MIN_FONT,
    H2_MIN_FONT,
    FOOTNOTE_MAX_FONT,
)


def _overlaps_table(
    block_bbox: Tuple[float, float, float, float],
    table_bboxes: List[Tuple[float, float, float, float]],
    threshold: float = 0.25
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
    start_page: int = START_CONTENT_PAGE
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic block-level parser:
    1. Clips top/bottom running margins via bbox coordinates.
    2. Masks out table bounding boxes so tables never bleed into prose.
    3. Separates footnotes (< 9.2pt) from H1, H2, and BODY blocks.
    """
    doc = pymupdf.open(pdf_path)
    prose_blocks: List[Dict[str, Any]] = []
    footnotes: List[Dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num < start_page:
            continue

        page = doc[page_idx]
        page_tables = table_bboxes_by_page.get(page_num, [])
        blocks = page.get_text("dict", sort=True)["blocks"]

        for b in blocks:
            if "lines" not in b:
                continue

            bbox = tuple(round(c, 2) for c in b["bbox"])

            # 1. Margin clipping (removes running "APRA" header & page number footer)
            if bbox[1] < HEADER_MARGIN_Y or bbox[3] > FOOTER_MARGIN_Y:
                continue

            # 2. Exclude table regions
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

            # 3. Separate Footnotes from Structural Hierarchy
            if dominant_size <= FOOTNOTE_MAX_FONT:
                footnotes.append({
                    "page": page_num,
                    "font_size": dominant_size,
                    "bbox": list(bbox),
                    "text": full_text
                })
                continue

            if dominant_size >= H1_MIN_FONT:
                tag = "H1"
            elif H2_MIN_FONT <= dominant_size < H1_MIN_FONT and is_bold:
                tag = "H2"
            else:
                tag = "BODY"

            prose_blocks.append({
                "page": page_num,
                "text": full_text,
                "font_size": dominant_size,
                "bbox": list(bbox),
                "tag": tag
            })

    doc.close()
    return prose_blocks, footnotes