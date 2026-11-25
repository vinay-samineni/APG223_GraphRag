import time
from typing import Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError
from apg223_rag.config import GEMINI_API_KEY, LLM_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=20_000,  # 20-second timeout so it never hangs
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


def generate_answer(query: str, retrieval_bundle: Dict[str, Any]) -> str:
    parent_blocks = "\n\n---\n\n".join(
        f"[{p['parent_id']} | {p['section']} > {p['subsection']} (Pages {p['page_start']}-{p['page_end']})]\n{p['content']}"
        for p in retrieval_bundle["parents"]
    )
    graph_context = "\n".join(retrieval_bundle["graph_triples"]) or "None"

    prompt = f"""You are an APRA Prudential Compliance Specialist analyzing APG 223 (Residential Mortgage Lending).
Answer the user's question strictly using the retrieved Parent Subsections and Knowledge Graph relationships below. Cite paragraph numbers (e.g., Paragraph 32, Paragraph 55) where present.

### Knowledge Graph Relationships:
{graph_context}

### Retrieved Full Parent Contexts:
{parent_blocks}

### Question:
{query}
"""
    last_err = None
    models_to_try = list(dict.fromkeys([LLM_MODEL] + FALLBACK_MODELS))

    for model_name in models_to_try:
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
        except APIError as e:
            last_err = e
            print(f"[Warning] {model_name} failed ({e.code}). Trying next fallback model...")
            time.sleep(1)

    raise RuntimeError(f"All Gemini models failed: {last_err}")