### Sora 2 video generation — quickstart (Python)

Prereqs: Python 3.9+, macOS. This project uses a virtual environment and avoids global installs.

Setup

1) Create venv and install deps

```bash
cd /Users/kylemathewson/Sora
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure API key and verify organization

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`:

```bash
cp .env.example .env
echo "OPENAI_API_KEY=sk-..." >> .env
```

**Important**: Your OpenAI organization must be verified to use Sora models. Visit `https://platform.openai.com/settings/organization/general` and click "Verify Organization". Access propagates within 15 minutes after verification.

3) Discover Sora-capable models (optional)

```bash
source .venv/bin/activate
python scripts/list_models.py
```

4) Generate a video

```bash
source .venv/bin/activate
python scripts/generate_video.py \
  --prompt "A macro cinematic shot of ocean waves at sunrise, 4K film look" \
  --duration 8 \
  --resolution 1920x1080 \
  --out videos/ocean_sunrise.mp4
```

Notes

- This starter attempts the official Python SDK video method first and falls back to a raw HTTPS call if necessary. If you see a 404/unsupported error, upgrade the `openai` package and re-run.
- Refer to the official guide for parameter details: `https://platform.openai.com/docs/guides/video-generation`.


