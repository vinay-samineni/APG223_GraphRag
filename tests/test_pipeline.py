# tests/test_pipeline.py
from apg223_rag.graph.neo4j_graph import CONCEPT_PATTERNS
from apg223_rag.chunking.semantic_chunker import _split_sentences
from apg223_rag.retrieval.hybrid_retriever import reciprocal_rank_fusion
from apg223_rag.ingestion.tree_builder import build_document_tree_and_parents


def test_whole_word_concept_matching():
    false_text = "The administration of the scheme affects them significantly."
    assert not CONCEPT_PATTERNS["NIS"].search(false_text)
    assert not CONCEPT_PATTERNS["HEM"].search(false_text)

    true_text = "ADIs use NIS models alongside the HEM index and track policy overrides."
    assert CONCEPT_PATTERNS["NIS"].search(true_text)
    assert CONCEPT_PATTERNS["HEM"].search(true_text)
    assert CONCEPT_PATTERNS["Override"].search(true_text)


def test_sentence_splitting_preserves_punctuation():
    sample = "32. Under APS 220, ADIs apply a buffer of at least 3.0 per cent. Good practice includes:"
    sents = _split_sentences(sample)
    assert len(sents) == 2
    assert sents[0].endswith("3.0 per cent.")
    assert sents[1].endswith("includes:")
    assert not sents[1].endswith(":.")


def test_bounded_parent_windowing_splits_oversized_subsections():
    blocks = [
        {"page": 10, "text": "Serviceability assessments", "font_size": 12.5, "bbox": [50, 60, 250, 80], "tag": "H2"},
        {"page": 10, "text": "25. " + ("A" * 1800), "font_size": 10.5, "bbox": [50, 100, 500, 200], "tag": "BODY"},
        {"page": 11, "text": "26. " + ("B" * 1800), "font_size": 10.5, "bbox": [50, 100, 500, 200], "tag": "BODY"},
    ]
    _, parents, _ = build_document_tree_and_parents(blocks, [], [])
    assert len(parents) == 2
    assert parents[0]["parent_id"] == "P001_1"
    assert parents[1]["parent_id"] == "P001_2"


def test_unmatched_tables_do_not_attach_to_p001():
    blocks = [
        {"page": 3, "text": "About this guide", "font_size": 16.0, "bbox": [50, 60, 200, 80], "tag": "H1"},
        {"page": 3, "text": "1. Intro paragraph.", "font_size": 10.5, "bbox": [50, 100, 500, 130], "tag": "BODY"},
    ]
    glossary_table = [{"table_id": "T001", "page": 26, "bbox": [50, 60, 500, 700], "content": "| col |"}]
    _, parents, linked_tables = build_document_tree_and_parents(blocks, glossary_table, [])
    assert "parent_id" not in linked_tables[0]
    assert "[Table T001]" not in parents[0]["content"]


def test_rrf_ranks_overlap_highest():
    dense = ["C003", "C001", "C002"]
    sparse = ["C001", "C004", "C003"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert fused[0][0] == "C001"