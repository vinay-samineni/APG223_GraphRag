import pymupdf
from typing import List, Dict, Any, Tuple
from apg223_rag.config import START_CONTENT_PAGE

_docling_converter = None


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _is_good_table_extraction(rows: List[List[Any]]) -> bool:
    """Validates PyMuPDF table extraction quality before deciding on Docling fallback."""
    if not rows or len(rows) < 2:
        return False
    num_cols = max(len(r) for r in rows)
    if num_cols < 2:
        return False

    total_cells = sum(len(r) for r in rows)
    empty_cells = sum(1 for r in rows for c in r if not c or not str(c).strip())
    if total_cells == 0 or (empty_cells / total_cells) > 0.40:
        return False

    header_str = "".join(str(c or "") for c in rows[0]).strip()
    return len(header_str) >= 3


def _extract_page_tables_docling(pdf_path: str, page_num: int) -> List[str]:
    """Runs Docling strictly on a single page when PyMuPDF table quality is insufficient."""
    converter = _get_docling_converter()
    result = converter.convert(pdf_path, page_range=(page_num, page_num))
    return [tbl.export_to_markdown() for tbl in result.document.tables]


def extract_tables(
    pdf_path: str,
    start_page: int = START_CONTENT_PAGE
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Tuple[float, float, float, float]]]]:
    """
    Detects tables independently using pymupdf.Page.find_tables().
    Returns:
      - Structured table objects
      - Page-indexed bounding boxes so text_parser.py can mask them out
    """
    doc = pymupdf.open(pdf_path)
    tables: List[Dict[str, Any]] = []
    table_bboxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
    table_counter = 1

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num < start_page:
            continue

        page = doc[page_idx]
        finder = page.find_tables()
        if not finder.tables:
            continue

        table_bboxes[page_num] = []
        docling_cache = None

        for idx, tbl in enumerate(finder.tables):
            bbox = tuple(round(c, 2) for c in tbl.bbox)
            table_bboxes[page_num].append(bbox)

            raw_rows = tbl.extract()
            if _is_good_table_extraction(raw_rows):
                content_md = tbl.to_markdown()
                source_engine = "pymupdf"
            else:
                try:
                    if docling_cache is None:
                        docling_cache = _extract_page_tables_docling(pdf_path, page_num)
                    content_md = docling_cache[idx] if idx < len(docling_cache) else tbl.to_markdown()
                    source_engine = "docling"
                except Exception:
                    content_md = tbl.to_markdown()
                    source_engine = "pymupdf_fallback"

            tables.append({
                "table_id": f"T{table_counter:03d}",
                "chunk_type": "table",
                "page": page_num,
                "bbox": list(bbox),
                "extractor": source_engine,
                "content": content_md.strip()
            })
            table_counter += 1

    doc.close()
    return tables, table_bboxes