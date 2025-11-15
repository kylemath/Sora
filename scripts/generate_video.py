#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Missing OPENAI_API_KEY. Put it in .env or export it.")
        sys.exit(2)
    return api_key


def save_binary(content: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(content)


def parse_resolution_to_size(resolution: str) -> str:
    """Map arbitrary WxH to the closest allowed SDK size.

    Allowed sizes observed via SDK:
      - '1280x720'  (landscape 720p)
      - '720x1280'  (portrait 720p)
      - '1792x1024' (landscape)
      - '1024x1792' (portrait)
    """
    allowed = [
        (1280, 720),
        (720, 1280),
        (1792, 1024),
        (1024, 1792),
    ]
    w = h = None
    if isinstance(resolution, str) and "x" in resolution:
        try:
            parts = resolution.lower().split("x")
            w = int(parts[0])
            h = int(parts[1])
        except Exception:
            w = h = None

    # Default to the smallest area option (720p) matching orientation if possible
    if w and h:
        portrait = h > w
        candidates = [(720, 1280), (1280, 720)] if portrait else [(1280, 720), (720, 1280)]
        for cw, ch in candidates:
            return f"{cw}x{ch}"
    # Fallback to 1280x720
    return "1280x720"


def bucket_seconds(duration: int) -> str:
    """Map arbitrary seconds to closest supported string among {'4','8','12'}."""
    allowed = [4, 8, 12]
    target = min(allowed, key=lambda x: abs(x - (duration or 4)))
    return str(target)


def try_sdk_generate(
    api_key: str,
    prompt: str,
    duration: int,
    resolution: str,
    model: Optional[str],
) -> Optional[bytes]:
    if OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        # Prefer official SDK path
        selected_model = model or "sora-2"

        if hasattr(client, "videos"):
            # Use create_and_poll to simplify job handling
            try:
                seconds_str = bucket_seconds(duration)
                size_str = parse_resolution_to_size(resolution)
                video_obj = client.videos.create_and_poll(
                    model=selected_model,
                    prompt=prompt,
                    seconds=seconds_str,
                    size=size_str,
                    poll_interval_ms=1000,
                )
            except Exception as e:
                # Check if this is a verification error before trying alternatives
                err_str = str(e)
                if "must be verified" in err_str or "403" in err_str:
                    print(f"\n⚠️  Organization verification required!")
                    print("Visit: https://platform.openai.com/settings/organization/general")
                    print("Click 'Verify Organization' and wait up to 15 minutes.\n")
                    return None
                # Older SDKs may not support create_and_poll; try create then poll
                try:
                    seconds_str = bucket_seconds(duration)
                    size_str = parse_resolution_to_size(resolution)
                    created = client.videos.create(
                        model=selected_model,
                        prompt=prompt,
                        seconds=seconds_str,
                        size=size_str,
                    )
                    # Attempt to poll by id if available
                    vid_id = getattr(created, "id", None) or created.get("id") if isinstance(created, dict) else None
                    if vid_id and hasattr(client.videos, "poll"):
                        video_obj = client.videos.poll(vid_id)
                    else:
                        video_obj = created
                except Exception as e2:
                    raise RuntimeError(f"SDK videos create failed: {e2}") from e

            # Try download via SDK helper
            try:
                vid_id = getattr(video_obj, "id", None) or (video_obj.get("id") if isinstance(video_obj, dict) else None)
                if vid_id and hasattr(client.videos, "download_content"):
                    content = client.videos.download_content(vid_id)
                    if isinstance(content, (bytes, bytearray)):
                        return bytes(content)
                    # Fallbacks for response-like objects
                    if hasattr(content, "read"):
                        return content.read()
                    if hasattr(content, "content"):
                        return content.content  # type: ignore
            except Exception:
                pass

            # Try URL fields if present
            for key in ("url", "video_url"):
                if hasattr(video_obj, key):
                    url = getattr(video_obj, key)
                    if isinstance(url, str):
                        r = requests.get(url, timeout=180)
                        r.raise_for_status()
                        return r.content
                if isinstance(video_obj, dict) and key in video_obj and isinstance(video_obj[key], str):
                    r = requests.get(video_obj[key], timeout=180)
                    r.raise_for_status()
                    return r.content

        return None
    except Exception as exc:
        print(f"SDK video generation failed, will try HTTP fallback: {exc}")
        return None


def http_poll(
    api_key: str,
    status_url: str,
    timeout_seconds: int = 600,
    poll_interval: float = 2.0,
) -> Optional[str]:
    deadline = time.time() + timeout_seconds
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    while time.time() < deadline:
        r = requests.get(status_url, headers=headers, timeout=30)
        if r.status_code >= 400:
            print(f"Polling error {r.status_code}: {r.text}")
            return None
        data = r.json()
        state = data.get("status") or data.get("state")
        if state in {"succeeded", "completed", "done"}:
            return data.get("video_url") or data.get("url")
        if state in {"failed", "error"}:
            print(f"Job failed: {data}")
            return None
        time.sleep(poll_interval)
    print("Timed out waiting for video job to complete")
    return None


def http_fallback_generate(
    api_key: str,
    prompt: str,
    duration: int,
    resolution: str,
    model: Optional[str],
) -> Optional[bytes]:
    # Endpoint per official docs
    endpoint = "https://api.openai.com/v1/videos"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Use multipart/form-data as shown in official curl example
    files_payload = {
        "model": (None, model or "sora-2"),
        "prompt": (None, prompt),
    }
    
    # Add optional parameters if provided
    # Map resolution to size if needed
    size_str = parse_resolution_to_size(resolution)
    if size_str:
        files_payload["size"] = (None, size_str)
    
    # Bucket duration to allowed seconds
    if duration:
        seconds_str = bucket_seconds(duration)
        files_payload["seconds"] = (None, seconds_str)

    r = requests.post(endpoint, headers=headers, files=files_payload, timeout=120)
    
    try:
        data = r.json()
    except Exception:
        data = None
    
    if r.status_code >= 400:
        print(f"HTTP create error {r.status_code}: {r.text}")
        # Check if this is a verification/auth error that won't succeed with other methods
        if data and isinstance(data, dict):
            err = data.get("error", {})
            if isinstance(err, dict):
                msg = err.get("message", "")
                if "must be verified" in msg or r.status_code == 403:
                    print("\n⚠️  Organization verification required!")
                    print("Visit: https://platform.openai.com/settings/organization/general")
                    print("Click 'Verify Organization' and wait up to 15 minutes.\n")
        return None
    
    if not data:
        print("HTTP response was not JSON")
        return None

    # Handle immediate URL
    if "video_url" in data:
        url = data["video_url"]
        rr = requests.get(url, timeout=120)
        rr.raise_for_status()
        return rr.content

    # Handle async job with status URL
    status_url = data.get("status_url") or data.get("poll_url")
    if status_url:
        url = http_poll(api_key, status_url)
        if not url:
            return None
        rr = requests.get(url, timeout=120)
        rr.raise_for_status()
        return rr.content

    # Some responses might include nested data
    maybe_url = data.get("url")
    if isinstance(maybe_url, str):
        rr = requests.get(maybe_url, timeout=120)
        rr.raise_for_status()
        return rr.content

    print("Unknown response format; could not obtain video bytes")
    return None


def responses_http_generate(
    api_key: str,
    prompt: str,
    duration: int,
    resolution: str,
    model: Optional[str],
) -> Optional[bytes]:
    # Try the Responses API with an output_video content part
    endpoint = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Parse resolution like 640x360
    width = None
    height = None
    if isinstance(resolution, str) and "x" in resolution:
        try:
            parts = resolution.lower().split("x")
            width = int(parts[0])
            height = int(parts[1])
        except Exception:
            width = None
            height = None

    video_spec = {}
    if width and height:
        video_spec.update({"width": width, "height": height})
    if duration and duration > 0:
        video_spec.update({"duration": duration})
    video_spec.update({"format": "mp4"})

    body = {
        "model": model or "sora-2",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "output_video", "video": video_spec},
                ],
            }
        ],
    }

    r = requests.post(endpoint, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        print(f"Responses API error {r.status_code}: {r.text}")
        return None
    data = r.json()

    # Inspect common shapes for media URLs or base64
    # Try: data["output"][0]["content"][0]["video"]["url"]
    try:
        output = data.get("output") or []
        if output:
            content = output[0].get("content") or []
            for part in content:
                if part.get("type") == "output_video":
                    video = part.get("video") or {}
                    url = video.get("url")
                    if isinstance(url, str):
                        rr = requests.get(url, timeout=180)
                        rr.raise_for_status()
                        return rr.content
                    # Some variants might return base64 bytes
                    b64 = video.get("b64") or video.get("data")
                    if isinstance(b64, str):
                        import base64

                        return base64.b64decode(b64)
    except Exception:
        pass

    print("Unknown Responses API format; could not obtain video bytes")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a video with Sora 2")
    p.add_argument("--prompt", required=True, help="Text prompt for the video")
    p.add_argument("--duration", type=int, default=8, help="Duration in seconds")
    p.add_argument(
        "--resolution",
        default="1280x720",
        help="Resolution like 1280x720 or 1920x1080",
    )
    p.add_argument("--model", default=None, help="Model id (defaults to sora-2)")
    p.add_argument("--out", required=True, help="Output mp4 path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    out_path = Path(args.out)

    # First try SDK if available
    video_bytes = try_sdk_generate(
        api_key=api_key,
        prompt=args.prompt,
        duration=args.duration,
        resolution=args.resolution,
        model=args.model,
    )
    if video_bytes is None:
        # Try Responses API fallback first
        video_bytes = responses_http_generate(
            api_key=api_key,
            prompt=args.prompt,
            duration=args.duration,
            resolution=args.resolution,
            model=args.model,
        )
    if video_bytes is None:
        video_bytes = http_fallback_generate(
            api_key=api_key,
            prompt=args.prompt,
            duration=args.duration,
            resolution=args.resolution,
            model=args.model,
        )

    if not video_bytes:
        print("Video generation failed. See messages above.")
        sys.exit(3)

    save_binary(video_bytes, out_path)
    print(f"Saved video to {out_path}")


if __name__ == "__main__":
    main()


