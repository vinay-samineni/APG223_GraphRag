# src/apg223_rag/generation/llm_client.py
from google import genai
from google.genai import types
from apg223_rag.config import GEMINI_API_KEY, LLM_TIMEOUT_MS

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Set it in your .env file before running.")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=LLM_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return _client