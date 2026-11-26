import pymupdf
from typing import List, Dict, Any, Tuple
from apg223_rag.config import LAYOUT


def _is_valid_data_table(rows: List[List[Any]]) -> bool:
    if not rows or len(rows) < 2:
        return False
    if max(len(r) for r in rows) < 2:
        return False

    total_cells = sum(len(r) for r in rows)
    empty_cells = sum(1 for r in rows for c in r if not c or not str(c).strip())
    if total_cells == 0 or (empty_cells / total_cells) > 0.35:
        return False

    header_cells = [str(c or "").strip() for c in rows[0] if c and str(c).strip()]
    if not header_cells or all(h.startswith("Col") and h[3:].isdigit() for h in header_cells):
        return False

    return True


def extract_tables(
    pdf_path: str,
    start_page: int = LAYOUT.start_content_page,
    end_page: int = LAYOUT.end_content_page,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[float, float, float, float]]]]:
    doc = pymupdf.open(pdf_path)
    tables: List[Dict[str, Any]] = []
    table_bboxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    table_counter = 1

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num < start_page or page_num > end_page:
            continue

        finder = doc[page_idx].find_tables()
        if not finder.tables:
            continue

        for tbl in finder.tables:
            if not _is_valid_data_table(tbl.extract()):
                continue

            bbox = tuple(round(c, 2) for c in tbl.bbox)
            table_bboxes.setdefault(page_num, []).append(bbox)

            tables.append({
                "table_id": f"T{table_counter:03d}",
                "chunk_type": "table",
                "page": page_num,
                "bbox": list(bbox),
                "extractor": "pymupdf",
                "content": tbl.to_markdown().strip(),
            })
            table_counter += 1

    doc.close()
    return tables, table_bboxes