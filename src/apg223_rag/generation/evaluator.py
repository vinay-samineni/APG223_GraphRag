# src/apg223_rag/generation/evaluator.py
import json
import time
from typing import Dict, Any
from pydantic import BaseModel
from google.genai import types
from apg223_rag.config import LLM_JUDGE_MODEL, FALLBACK_MODELS
from apg223_rag.generation.llm_client import get_gemini_client


class EvalMetrics(BaseModel):
    faithfulness: int
    answer_relevance: int
    context_precision: int
    reasoning: str


def evaluate_response(query: str, answer: str, retrieval_bundle: Dict[str, Any]) -> Dict[str, Any]:
    client = get_gemini_client()
    full_context = "\n\n".join(
        f"[{p['parent_id']} | {p['subsection']}]\n{p['content']}"
        for p in retrieval_bundle["parents"]
    )
    if retrieval_bundle.get("glossary_hits"):
        full_context += "\n\n[Glossary]\n" + "\n".join(
            f"{g['term']}: {g['definition']}" for g in retrieval_bundle["glossary_hits"]
        )

    eval_prompt = f"""Evaluate this RAG pipeline response on a 1-5 scale (5 = highest):
1. faithfulness: Are all claims in the answer strictly supported by the provided context?
2. answer_relevance: Does the answer directly and completely address the user's question?
3. context_precision: Do the retrieved sections contain the exact regulatory rules needed to answer the question?

Question: {query}

Retrieved Context:
{full_context}

Generated Answer:
{answer}
"""
    judge_models = list(dict.fromkeys([LLM_JUDGE_MODEL] + FALLBACK_MODELS))
    last_err = None

    for model_name in judge_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=eval_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json",
                    response_schema=EvalMetrics,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            last_err = e
            print(f"[Warning] Judge model {model_name} failed ({type(e).__name__}: {e}). Trying fallback...")
            time.sleep(1)

    raise RuntimeError(f"All Gemini judge models failed: {last_err}")