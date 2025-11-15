#!/usr/bin/env python3
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, send_file

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

app = Flask(__name__)
VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)
METADATA_FILE = VIDEOS_DIR / "metadata.json"


def load_metadata():
    """Load video metadata from JSON file."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(metadata):
    """Save video metadata to JSON file."""
    try:
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Failed to save metadata: {e}")


def load_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")
    return api_key


def bucket_seconds(duration: int) -> str:
    """Map arbitrary seconds to closest supported string among {'4','8','12'}."""
    allowed = [4, 8, 12]
    target = min(allowed, key=lambda x: abs(x - (duration or 4)))
    return str(target)


def parse_resolution_to_size(resolution: str) -> str:
    """Map arbitrary WxH to the closest allowed SDK size."""
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

    if w and h:
        portrait = h > w
        candidates = [(720, 1280), (1280, 720)] if portrait else [(1280, 720), (720, 1280)]
        for cw, ch in candidates:
            return f"{cw}x{ch}"
    return "1280x720"


def generate_video_sdk(
    api_key: str,
    prompt: str,
    duration: int,
    resolution: str,
    model: str,
    input_image_path: Optional[str] = None,
) -> Optional[bytes]:
    """Try SDK-based video generation."""
    if OpenAI is None:
        print("OpenAI SDK not available")
        return None
    try:
        client = OpenAI(api_key=api_key)
        if not hasattr(client, "videos"):
            print("Client does not have 'videos' attribute")
            return None

        seconds_str = bucket_seconds(duration)
        size_str = parse_resolution_to_size(resolution)

        print(f"Calling SDK with: model={model}, seconds={seconds_str}, size={size_str}")

        kwargs = {
            "model": model,
            "prompt": prompt,
            "seconds": seconds_str,
            "size": size_str,
            "poll_interval_ms": 2000,
        }

        # Add input image if provided
        if input_image_path and Path(input_image_path).exists():
            try:
                # Try to pass file path directly
                kwargs["input_reference"] = open(input_image_path, "rb")
                video_obj = client.videos.create_and_poll(**kwargs)
                kwargs["input_reference"].close()
            except Exception as e:
                # If that fails, try without input image
                print(f"Failed to use input image: {e}")
                kwargs.pop("input_reference", None)
                video_obj = client.videos.create_and_poll(**kwargs)
        else:
            print("Creating video without input image...")
            video_obj = client.videos.create_and_poll(**kwargs)
        
        print(f"Video object received: {type(video_obj)}")

        # Try download via SDK helper
        vid_id = getattr(video_obj, "id", None) or (
            video_obj.get("id") if isinstance(video_obj, dict) else None
        )
        if vid_id and hasattr(client.videos, "download_content"):
            content = client.videos.download_content(vid_id)
            if isinstance(content, (bytes, bytearray)):
                return bytes(content)
            if hasattr(content, "read"):
                return content.read()
            if hasattr(content, "content"):
                return content.content

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
    except Exception as e:
        print(f"SDK error: {e}")
        return None


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/videos", methods=["GET"])
def list_videos():
    """List all generated videos with metadata."""
    metadata = load_metadata()
    videos = []
    for vid in sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        video_data = {
            "name": vid.name,
            "size": vid.stat().st_size,
            "created": vid.stat().st_mtime,
        }
        # Add metadata if available
        if vid.name in metadata:
            video_data.update(metadata[vid.name])
        videos.append(video_data)
    return jsonify(videos)


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate a new video."""
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()
        name = data.get("name", "").strip()
        duration = int(data.get("duration", 4))
        resolution = data.get("resolution", "1280x720")
        model = data.get("model", "sora-2")
        input_image_b64 = data.get("input_image")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if not name:
            name = f"video_{int(time.time())}"
        if not name.endswith(".mp4"):
            name += ".mp4"

        print(f"\n🎬 Generating video: {name}")
        print(f"📝 Prompt: {prompt[:100]}...")
        print(f"⏱️  Duration: {duration}s, Resolution: {resolution}, Model: {model}")

        # Save input image if provided
        input_image_path = None
        if input_image_b64:
            try:
                img_data = base64.b64decode(input_image_b64.split(",")[1])
                input_image_path = VIDEOS_DIR / f"input_{int(time.time())}.jpg"
                with open(input_image_path, "wb") as f:
                    f.write(img_data)
                print(f"📷 Using input image: {input_image_path}")
            except Exception as e:
                print(f"❌ Failed to save input image: {e}")
                input_image_path = None

        api_key = load_api_key()
        video_bytes = generate_video_sdk(
            api_key=api_key,
            prompt=prompt,
            duration=duration,
            resolution=resolution,
            model=model,
            input_image_path=str(input_image_path) if input_image_path else None,
        )

        if not video_bytes:
            print("❌ Video generation failed")
            return jsonify({"error": "Video generation failed. Check server logs."}), 500

        out_path = VIDEOS_DIR / name
        with open(out_path, "wb") as f:
            f.write(video_bytes)

        print(f"✅ Video saved: {out_path} ({len(video_bytes)} bytes)\n")

        # Save metadata
        metadata = load_metadata()
        metadata[name] = {
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "model": model,
            "created": time.time(),
        }
        save_metadata(metadata)

        # Clean up temp input image
        if input_image_path and input_image_path.exists():
            input_image_path.unlink()

        return jsonify({
            "success": True,
            "name": name,
            "size": len(video_bytes),
        })
    except Exception as e:
        print(f"❌ Error in generate endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/videos/<name>", methods=["GET"])
def get_video(name):
    """Serve a video file."""
    video_path = VIDEOS_DIR / name
    if not video_path.exists():
        return jsonify({"error": "Video not found"}), 404
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/videos/<name>", methods=["DELETE"])
def delete_video(name):
    """Delete a video file and its metadata."""
    video_path = VIDEOS_DIR / name
    if not video_path.exists():
        return jsonify({"error": "Video not found"}), 404
    video_path.unlink()
    
    # Remove from metadata
    metadata = load_metadata()
    if name in metadata:
        del metadata[name]
        save_metadata(metadata)
    
    return jsonify({"success": True})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sora 2 Video Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .content {
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 30px;
            align-items: start;
        }
        @media (max-width: 968px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        .form-card, .gallery-card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .form-card {
            position: sticky;
            top: 20px;
        }
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            color: #333;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        input[type="text"], textarea, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        .image-upload {
            border: 2px dashed #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }
        .image-upload:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .image-upload.has-image {
            border-color: #667eea;
            padding: 0;
        }
        #imagePreview {
            max-width: 100%;
            border-radius: 8px;
            display: none;
        }
        #imagePreview.show {
            display: block;
        }
        .upload-text {
            color: #666;
        }
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102,126,234,0.4);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 14px;
            display: none;
        }
        .status.show {
            display: block;
        }
        .status.error {
            background: #fee;
            color: #c33;
        }
        .status.success {
            background: #efe;
            color: #3a3;
        }
        .status.loading {
            background: #fef8e7;
            color: #856404;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        .video-card {
            background: #f8f9fa;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .video-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        }
        video {
            width: 100%;
            display: block;
            background: #000;
        }
        .video-info {
            padding: 15px;
        }
        .video-name {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            word-break: break-word;
        }
        .video-meta {
            font-size: 12px;
            color: #666;
        }
        .video-actions {
            padding: 0 15px 15px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .btn-small {
            flex: 1;
            min-width: 80px;
            padding: 8px;
            font-size: 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-view {
            background: #667eea;
            color: white;
        }
        .btn-view:hover {
            background: #5568d3;
        }
        .btn-remix {
            background: #28a745;
            color: white;
        }
        .btn-remix:hover {
            background: #218838;
        }
        .btn-delete {
            background: #dc3545;
            color: white;
        }
        .btn-delete:hover {
            background: #c82333;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.show {
            display: flex;
        }
        .modal-content {
            background: white;
            border-radius: 16px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .modal-header h3 {
            color: #667eea;
            margin: 0;
        }
        .modal-close {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #999;
        }
        .modal-close:hover {
            color: #333;
        }
        .metadata-field {
            margin-bottom: 15px;
        }
        .metadata-field label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
            font-size: 13px;
        }
        .metadata-field .value {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 14px;
            color: #555;
            word-break: break-word;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state svg {
            width: 80px;
            height: 80px;
            margin-bottom: 20px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Sora 2 Video Generator</h1>
        
        <div class="content">
            <div class="form-card">
                <h2>Create Video</h2>
                
                <div class="form-group">
                    <label>Prompt Template</label>
                    <select id="templateSelect" onchange="applyTemplate()">
                        <option value="">Custom prompt (no template)</option>
                        <option value="cinematic">Cinematic Scene</option>
                        <option value="documentary">Documentary Style</option>
                        <option value="product">Product Showcase</option>
                        <option value="abstract">Abstract/Artistic</option>
                        <option value="nature">Nature/Wildlife</option>
                        <option value="stopmotion">Stop Motion Animation</option>
                    </select>
                </div>
                
                <form id="videoForm">
                    <div class="form-group">
                        <label>Prompt * <small style="color: #999; font-weight: normal;">(Be specific: shot type, subject, action, style, lighting)</small></label>
                        <textarea id="prompt" required placeholder="Example: A wide cinematic shot of a golden retriever running through a sunlit meadow at sunset, slow motion, warm tones, shallow depth of field"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>Video Name</label>
                        <input type="text" id="name" placeholder="my_video (optional)">
                    </div>
                    
                    <div class="form-group">
                        <label>Initial Image (Optional)</label>
                        <div class="image-upload" id="imageUploadArea">
                            <input type="file" id="imageInput" accept="image/*" style="display:none">
                            <img id="imagePreview" alt="Preview">
                            <div class="upload-text" id="uploadText">
                                📷 Click to upload an image<br>
                                <small>or drag & drop</small>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Duration</label>
                        <select id="duration">
                            <option value="4" selected>4 seconds</option>
                            <option value="8">8 seconds</option>
                            <option value="12">12 seconds</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Resolution</label>
                        <select id="resolution">
                            <option value="1280x720" selected>1280×720 (Landscape)</option>
                            <option value="720x1280">720×1280 (Portrait)</option>
                            <option value="1792x1024">1792×1024 (Wide)</option>
                            <option value="1024x1792">1024×1792 (Tall)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Model</label>
                        <select id="model">
                            <option value="sora-2" selected>Sora 2</option>
                            <option value="sora-2-pro">Sora 2 Pro</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn" id="generateBtn">
                        Generate Video
                    </button>
                    
                    <div class="status" id="status"></div>
                </form>
                
                <details style="margin-top: 20px; padding: 15px; background: #f8f9ff; border-radius: 8px; font-size: 13px;">
                    <summary style="cursor: pointer; font-weight: 600; color: #667eea; margin-bottom: 10px;">💡 Prompting Tips</summary>
                    <div style="color: #555; line-height: 1.6;">
                        <strong>Key Elements:</strong>
                        <ul style="margin: 8px 0; padding-left: 20px;">
                            <li><strong>Shot Type:</strong> wide, medium, close-up, aerial, POV</li>
                            <li><strong>Camera Movement:</strong> tracking, dolly, crane, handheld, static</li>
                            <li><strong>Subject & Action:</strong> Be specific about what's happening</li>
                            <li><strong>Setting:</strong> Describe the environment in detail</li>
                            <li><strong>Lighting:</strong> golden hour, moody, bright, studio, natural</li>
                            <li><strong>Style:</strong> cinematic, documentary, artistic, vintage</li>
                            <li><strong>Details:</strong> textures, colors, mood, depth of field</li>
                        </ul>
                        <strong>Examples:</strong>
                        <ul style="margin: 8px 0; padding-left: 20px;">
                            <li>"A tracking shot following a cyclist through autumn leaves"</li>
                            <li>"Close-up slow motion of paint splashing on white canvas"</li>
                            <li>"Aerial drone view circling over a misty mountain lake at dawn"</li>
                        </ul>
                    </div>
                </details>
            </div>
            
            <div class="gallery-card">
                <h2>Your Videos</h2>
                <div class="video-grid" id="videoGrid">
                    <div class="empty-state">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        <div>No videos yet. Generate your first one!</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Metadata Modal -->
    <div class="modal" id="metadataModal" onclick="if(event.target === this) closeModal()">
        <div class="modal-content">
            <div class="modal-header">
                <h3>📋 Video Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div id="modalBody"></div>
        </div>
    </div>
    
    <script>
        let currentImageData = null;
        
        // Prompt templates based on Sora 2 best practices
        const templates = {
            cinematic: `[SHOT TYPE: wide/medium/close-up] of [SUBJECT] [ACTION] in [SETTING].
Camera: [camera movement: tracking/dolly/crane/static]
Style: cinematic, [lighting: golden hour/moody/bright], [mood]
Details: [specific visual details, depth of field, motion quality]

Example: A wide tracking shot of a lone figure walking through a neon-lit Tokyo alley at night, rain falling, cinematic bokeh, moody blue and pink tones, 35mm film look`,
            
            documentary: `Documentary footage of [SUBJECT] [ACTION] in [REAL LOCATION].
Camera: handheld/stabilized, natural lighting
Style: realistic, authentic, [time of day]
Details: [specific behaviors, environment details]

Example: Documentary style footage of a street artist spray painting a mural on a brick wall in Brooklyn, late afternoon sun, handheld camera, vibrant colors, authentic urban atmosphere`,
            
            product: `Product showcase of [PRODUCT NAME] on [SURFACE/BACKGROUND].
Camera: [rotating/dolly in/360 spin]
Style: [modern/luxury/minimal], studio lighting, [color palette]
Details: [materials, reflections, specific features to highlight]

Example: Rotating 360 degree view of a sleek silver smartwatch on a marble pedestal, studio lighting with soft shadows, minimal background, reflective surface, modern luxury aesthetic`,
            
            abstract: `Abstract [VISUAL CONCEPT] with [ELEMENTS].
Style: [artistic movement/visual style], [color palette]
Motion: [fluid/geometric/organic] movement
Details: [textures, patterns, transitions]

Example: Abstract fluid simulation of colorful liquid metals merging and separating, iridescent colors shifting between purple, gold and teal, smooth organic motion, macro perspective, dreamlike atmosphere`,
            
            nature: `[SHOT TYPE] of [ANIMAL/NATURAL SUBJECT] [ACTION] in [HABITAT].
Camera: [movement style appropriate to wildlife]
Time: [golden hour/dawn/midday]
Style: nature documentary, [weather conditions]
Details: [specific behaviors, environment, lighting]

Example: Close-up slow motion shot of a hummingbird feeding from a red hibiscus flower, morning dew, soft natural lighting, shallow depth of field, wings creating motion blur, tropical garden background`,
            
            stopmotion: `Stop motion animation of [SUBJECT] [ACTION] in [SETTING].
Style: [realistic/claymation/paper craft/LEGO], tactile textures
Details: [materials, handmade quality, specific movements]
Aesthetic: [whimsical/dark/retro]

Example: Stop motion animation of a tiny clay fox navigating through a miniature autumn forest, realistic textures, warm color palette, whimsical atmosphere, visible handcrafted details, falling leaves in motion`
        };
        
        function applyTemplate() {
            const select = document.getElementById('templateSelect');
            const prompt = document.getElementById('prompt');
            const template = templates[select.value];
            
            if (template) {
                prompt.value = template;
                prompt.focus();
            } else {
                prompt.value = '';
            }
        }
        
        // Image upload handling
        const imageInput = document.getElementById('imageInput');
        const imageUploadArea = document.getElementById('imageUploadArea');
        const imagePreview = document.getElementById('imagePreview');
        const uploadText = document.getElementById('uploadText');
        
        imageUploadArea.addEventListener('click', () => imageInput.click());
        
        imageUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            imageUploadArea.style.borderColor = '#667eea';
        });
        
        imageUploadArea.addEventListener('dragleave', () => {
            imageUploadArea.style.borderColor = '#e0e0e0';
        });
        
        imageUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            imageUploadArea.style.borderColor = '#e0e0e0';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImageFile(file);
            }
        });
        
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleImageFile(file);
        });
        
        function handleImageFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImageData = e.target.result;
                imagePreview.src = currentImageData;
                imagePreview.classList.add('show');
                uploadText.style.display = 'none';
                imageUploadArea.classList.add('has-image');
            };
            reader.readAsDataURL(file);
        }
        
        // Form submission
        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            const name = document.getElementById('name').value.trim();
            const duration = parseInt(document.getElementById('duration').value);
            const resolution = document.getElementById('resolution').value;
            const model = document.getElementById('model').value;
            
            if (!prompt) return;
            
            showStatus('loading', 'Generating video... This may take 1-2 minutes.');
            document.getElementById('generateBtn').disabled = true;
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt,
                        name,
                        duration,
                        resolution,
                        model,
                        input_image: currentImageData
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showStatus('success', `Video "${data.name}" generated successfully!`);
                    document.getElementById('videoForm').reset();
                    currentImageData = null;
                    imagePreview.classList.remove('show');
                    uploadText.style.display = 'block';
                    imageUploadArea.classList.remove('has-image');
                    loadVideos();
                } else {
                    showStatus('error', data.error || 'Generation failed');
                }
            } catch (err) {
                showStatus('error', 'Network error: ' + err.message);
            } finally {
                document.getElementById('generateBtn').disabled = false;
            }
        });
        
        function showStatus(type, message) {
            const status = document.getElementById('status');
            status.className = 'status show ' + type;
            if (type === 'loading') {
                status.innerHTML = '<span class="spinner"></span>' + message;
            } else {
                status.textContent = message;
            }
        }
        
        // Load and display videos
        async function loadVideos() {
            try {
                const response = await fetch('/api/videos');
                const videos = await response.json();
                
                const grid = document.getElementById('videoGrid');
                
                if (videos.length === 0) {
                    grid.innerHTML = `
                        <div class="empty-state">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            <div>No videos yet. Generate your first one!</div>
                        </div>
                    `;
                    return;
                }
                
                grid.innerHTML = videos.map(video => {
                    const hasMetadata = video.prompt;
                    return `
                        <div class="video-card">
                            <video controls preload="metadata">
                                <source src="/api/videos/${video.name}" type="video/mp4">
                            </video>
                            <div class="video-info">
                                <div class="video-name">${video.name}</div>
                                <div class="video-meta">
                                    ${formatSize(video.size)} • ${formatDate(video.created)}
                                    ${hasMetadata ? `<br>${video.duration}s • ${video.resolution} • ${video.model}` : ''}
                                </div>
                            </div>
                            <div class="video-actions">
                                ${hasMetadata ? `
                                    <button class="btn-small btn-view" onclick='viewMetadata(${JSON.stringify(video).replace(/'/g, "&apos;")})'>
                                        View
                                    </button>
                                    <button class="btn-small btn-remix" onclick='remixVideo(${JSON.stringify(video).replace(/'/g, "&apos;")})'>
                                        Remix
                                    </button>
                                ` : ''}
                                <button class="btn-small btn-delete" onclick="deleteVideo('${video.name}')">
                                    Delete
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (err) {
                console.error('Failed to load videos:', err);
            }
        }
        
        function viewMetadata(video) {
            const modal = document.getElementById('metadataModal');
            const modalBody = document.getElementById('modalBody');
            
            modalBody.innerHTML = `
                <div class="metadata-field">
                    <label>Prompt</label>
                    <div class="value">${video.prompt || 'N/A'}</div>
                </div>
                <div class="metadata-field">
                    <label>Duration</label>
                    <div class="value">${video.duration || 'N/A'} seconds</div>
                </div>
                <div class="metadata-field">
                    <label>Resolution</label>
                    <div class="value">${video.resolution || 'N/A'}</div>
                </div>
                <div class="metadata-field">
                    <label>Model</label>
                    <div class="value">${video.model || 'N/A'}</div>
                </div>
                <div class="metadata-field">
                    <label>File Name</label>
                    <div class="value">${video.name}</div>
                </div>
                <div class="metadata-field">
                    <label>Size</label>
                    <div class="value">${formatSize(video.size)}</div>
                </div>
            `;
            
            modal.classList.add('show');
        }
        
        function remixVideo(video) {
            // Populate form with video's settings
            document.getElementById('prompt').value = video.prompt || '';
            document.getElementById('duration').value = video.duration || 4;
            document.getElementById('resolution').value = video.resolution || '1280x720';
            document.getElementById('model').value = video.model || 'sora-2';
            document.getElementById('templateSelect').value = '';
            
            // Scroll to form
            document.querySelector('.form-card').scrollIntoView({ behavior: 'smooth' });
            
            // Focus on prompt
            setTimeout(() => {
                document.getElementById('prompt').focus();
                document.getElementById('prompt').select();
            }, 500);
        }
        
        function closeModal() {
            document.getElementById('metadataModal').classList.remove('show');
        }
        
        async function deleteVideo(name) {
            if (!confirm(`Delete "${name}"?`)) return;
            
            try {
                await fetch(`/api/videos/${name}`, {method: 'DELETE'});
                loadVideos();
            } catch (err) {
                alert('Delete failed: ' + err.message);
            }
        }
        
        function formatSize(bytes) {
            return (bytes / 1024 / 1024).toFixed(1) + ' MB';
        }
        
        function formatDate(timestamp) {
            return new Date(timestamp * 1000).toLocaleString();
        }
        
        // Load videos on page load
        loadVideos();
        
        // Auto-refresh every 30s
        setInterval(loadVideos, 30000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n🎬 Starting Sora 2 Local App...")
    print("📂 Videos saved to:", VIDEOS_DIR.absolute())
    print("🌐 Open: http://localhost:8001\n")
    app.run(debug=True, host="0.0.0.0", port=8001)

