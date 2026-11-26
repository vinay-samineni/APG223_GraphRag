# src/apg223_rag/generation/generator.py
import time
from typing import Dict, Any
from google.genai import types
from apg223_rag.config import FALLBACK_MODELS
from apg223_rag.generation.llm_client import get_gemini_client


def generate_answer(query: str, retrieval_bundle: Dict[str, Any]) -> str:
    client = get_gemini_client()
    parent_blocks = "\n\n---\n\n".join(
        f"[{p['parent_id']} | {p['section']} > {p['subsection']} (Pages {p['page_start']}-{p['page_end']})]\n{p['content']}"
        for p in retrieval_bundle["parents"]
    )
    graph_context = "\n".join(retrieval_bundle["graph_triples"]) or "None"
    glossary_context = "\n".join(
        f"- {g['term']}: {g['definition']}" for g in retrieval_bundle.get("glossary_hits", [])
    ) or "None"

    prompt = f"""You are an APRA Prudential Compliance Specialist analyzing APG 223 (Residential Mortgage Lending).
Answer the user's question strictly using the retrieved Parent Subsections, Glossary Definitions, and Neo4j Knowledge Graph relationships below. Cite paragraph numbers (e.g., Paragraph 32, Paragraph 55) where present.

### Glossary Definitions:
{glossary_context}

### Active Knowledge Graph Relationships:
{graph_context}

### Retrieved Full Parent Contexts:
{parent_blocks}

### Question:
{query}
"""
    last_err = None
    for model_name in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            print(f"[Warning] Generator model {model_name} failed ({type(e).__name__}: {e}). Trying fallback...")
            time.sleep(1)

    raise RuntimeError(f"All Gemini generator models failed: {last_err}")