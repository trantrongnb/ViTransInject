import os

from openai import OpenAI

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
LOCAL_BASE_URL   = "http://localhost:8080/v1"

MODELS = {
    "deepseek":     {"base_url": "https://api.deepseek.com", "model_id": "deepseek-chat", "api_key": DEEPSEEK_API_KEY},
    "qwen2.5-7b":  {"base_url": LOCAL_BASE_URL, "model_id": "qwen2.5-7b",  "api_key": "local"},
    "qwen2.5-3b":  {"base_url": LOCAL_BASE_URL, "model_id": "qwen2.5-3b",  "api_key": "local"},
    "qwen2.5-1.5b":{"base_url": LOCAL_BASE_URL, "model_id": "qwen2.5-1.5b","api_key": "local"},
    "llama-3.2-1b":{"base_url": LOCAL_BASE_URL, "model_id": "llama-3.2-1b", "api_key": "local"},
    "qwen2.5-0.5b":{"base_url": LOCAL_BASE_URL, "model_id": "qwen2.5-0.5b", "api_key": "local"},
    "gemma-2-2b":  {"base_url": LOCAL_BASE_URL, "model_id": "gemma-2-2b",   "api_key": "local"},
    "smollm2-1.7b":{"base_url": LOCAL_BASE_URL, "model_id": "smollm2-1.7b","api_key": "local"},
    "deepseek-r1-1.5b": {"base_url": LOCAL_BASE_URL, "model_id": "deepseek-r1-1.5b", "api_key": "local"},
    "llama-3.2-3b":{"base_url": LOCAL_BASE_URL, "model_id": "llama-3.2-3b", "api_key": "local"},
    "qwen3-1.7b":  {"base_url": LOCAL_BASE_URL, "model_id": "qwen3-1.7b",  "api_key": "local"},
    "sailor2-1b":  {"base_url": LOCAL_BASE_URL, "model_id": "sailor2-1b",  "api_key": "local"},
}

_TRANSLATE_PROMPT = (
    "Translate the following Vietnamese text to English. "
    "Output ONLY the translation, no explanation, no prefix.\n\n"
    "{text}"
)


def call_llm(system: str, user: str, model_name: str, max_tokens: int = 600) -> str:
    cfg = MODELS[model_name]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    r = client.chat.completions.create(
        model=cfg["model_id"],
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def translate(text: str, model_name: str) -> str:
    """Dịch VI→EN dùng cùng model, dùng cho P2/P3/P4 trong TN1."""
    prompt = _TRANSLATE_PROMPT.format(text=text)
    return call_llm("You are a professional translator.", prompt, model_name, max_tokens=800)
