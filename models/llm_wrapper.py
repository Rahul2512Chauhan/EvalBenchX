# models/llm_wrapper.py

import os
from typing import Any
from dotenv import load_dotenv
import openai

# ---------- 🔧 Load .env -----------
load_dotenv()

# ----------🔧 Load configuration-----------
PROVIDER = os.getenv("PROVIDER", "groq").lower()
MODEL = os.getenv("MODEL", "llama3-70b-8192")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

# ---------- 🔑 Set API keys & base URLs -----------
if PROVIDER == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.base_url = "https://api.openai.com/v1"

elif PROVIDER == "groq":
    openai.api_key = os.getenv("GROQ_API_KEY")
    openai.base_url = "https://api.groq.com/openai/v1/"

elif PROVIDER == "together":
    openai.api_key = os.getenv("TOGETHER_API_KEY")
    openai.base_url = "https://api.together.xyz/v1/"

else:
    raise ValueError(f"Unsupported provider: {PROVIDER}")

# ---------- 🧠 Unified LLM call function -----------
def get_completion(prompt: str, model: str = MODEL, temperature: float = TEMPERATURE) -> str:
    """
    Send prompt to selected LLM provider and return response text.

    Args:
        prompt (str): The input prompt for the LLM
        model (str): Model to use (default comes from env)
        temperature (float): Sampling temperature (0 = deterministic)

    Returns:
        str: The LLM's response
    """
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip() # type: ignore

    except Exception as e:
        print(f"❌ LLM call failed via {PROVIDER}: {e}")
        return "[ERROR]"
