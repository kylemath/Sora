#!/usr/bin/env python3
import os
import sys
from typing import List

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception as exc:
    print("Failed to import OpenAI SDK. Activate venv and install requirements:")
    print("  source .venv/bin/activate && pip install -r requirements.txt")
    print(f"Import error: {exc}")
    sys.exit(1)


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENAI_API_KEY. Put it in .env or export it.")
        sys.exit(2)
    return api_key


def main() -> None:
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    # List models and filter likely video-capable ones (heuristic; check docs for official list)
    models = client.models.list()
    candidates: List[str] = []
    for m in models.data:
        model_id = getattr(m, "id", "")
        if not model_id:
            continue
        lower = model_id.lower()
        if "video" in lower or "sora" in lower:
            candidates.append(model_id)

    print("Discovered models that look video-capable:")
    if not candidates:
        print("  (none found; your account may not have access yet)")
        return
    for mid in sorted(candidates):
        print(f"  - {mid}")


if __name__ == "__main__":
    main()


