import json
import time
from typing import Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from apg223_rag.config import GEMINI_API_KEY, LLM_MODEL

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        timeout=20_000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


class EvalMetrics(BaseModel):
    faithfulness: int
    answer_relevance: int
    context_precision: int
    reasoning: str


def evaluate_response(query: str, answer: str, retrieval_bundle: Dict[str, Any]) -> Dict[str, Any]:
    context = "\n\n".join(p["content"] for p in retrieval_bundle["parents"])

    eval_prompt = f"""Evaluate this RAG pipeline response on a 1-5 scale (5 = highest):
1. faithfulness: Are all claims strictly grounded in the retrieved context?
2. answer_relevance: Does the answer directly and thoroughly address the query?
3. context_precision: Did RRF + Graph retrieval select the exact required regulatory sections?

Query: {query}
Context: {context[:7000]}
Answer: {answer}
"""
    last_err = None
    models_to_try = list(dict.fromkeys([LLM_MODEL] + FALLBACK_MODELS))

    for model_name in models_to_try:
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
        except APIError as e:
            last_err = e
            print(f"[Warning] {model_name} failed ({e.code}). Trying next fallback model...")
            time.sleep(1)

    raise RuntimeError(f"All Gemini models failed during evaluation: {last_err}")