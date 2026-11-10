"""
DEARson Image Generator
-----------------------
Pages:
  1. Image Generator  - text -> image (free Pollinations, or ChatGPT / Gemini with your own API key)
  2. Image to Video   - image + description -> walking, talking human video
                        (free AI walk + lip-sync, or Gemini Veo with your own key)
  3. Text to Audio    - free TTS (Edge voices / gTTS) + voice-cloning buttons

Run:  streamlit run dearson_image_generator.py
"""
import asyncio
import base64
import io
import os
import random
import re
import subprocess
import tempfile
import time
from urllib.parse import quote

import requests
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="DEARson Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# STYLE: colourful buttons, ALL IN ONE LINE (even on a phone)
# ----------------------------------------------------------------------------
BUTTON_COLORS = {
    "gen": "linear-gradient(135deg,#ff512f,#dd2476)",
    "regen": "linear-gradient(135deg,#f7971e,#ffd200)",
    "clear": "linear-gradient(135deg,#434343,#000000)",
    "mkvid": "linear-gradient(135deg,#11998e,#38ef7d)",
    "newimg": "linear-gradient(135deg,#8e2de2,#4a00e0)",
    "resetvid": "linear-gradient(135deg,#434343,#000000)",
    "speak": "linear-gradient(135deg,#00c6ff,#0072ff)",
    "clone": "linear-gradient(135deg,#f953c6,#b91d73)",
    "speakclone": "linear-gradient(135deg,#fc4a1a,#f7b733)",
    "clearclone": "linear-gradient(135deg,#434343,#000000)",
    "pitchdn": "linear-gradient(135deg,#1d976c,#93f9b9)",
    "pitchreset": "linear-gradient(135deg,#434343,#000000)",
    "pitchup": "linear-gradient(135deg,#ee0979,#ff6a00)",
}

_css_btns = "\n".join(
    f".st-key-{k} button{{background:{v} !important;color:#fff !important;border:none !important;"
    f"font-weight:700 !important;border-radius:14px !important;width:100%;"
    f"padding:.55rem .2rem !important;font-size:.82rem !important;"
    f"box-shadow:0 4px 12px rgba(0,0,0,.25) !important;transition:transform .15s;}}"
    f".st-key-{k} button:hover{{transform:translateY(-2px) scale(1.03);}}"
    f".st-key-{k} button p{{color:#fff !important;font-weight:700;}}"
    for k, v in BUTTON_COLORS.items()
)

st.markdown(
    f"""
<style>
/* keep columns side by side on phones */
div[data-testid="stHorizontalBlock"]{{flex-wrap:nowrap !important;gap:.4rem !important;}}
div[data-testid="stHorizontalBlock"] > div{{min-width:0 !important;flex:1 1 0 !important;}}
.block-container{{padding-top:1.2rem;max-width:760px;}}
.dear-title{{text-align:center;font-size:2rem;font-weight:900;margin:0;
  background:linear-gradient(90deg,#ff512f,#dd2476,#8e2de2,#00c6ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.dear-sub{{text-align:center;opacity:.7;margin-bottom:.8rem;font-size:.9rem;}}
div[data-testid="stImage"] img, video{{border-radius:16px;}}
{_css_btns}
</style>
<h1 class="dear-title">DEARson Image Generator</h1>
<div class="dear-sub">Create images • Turn them into video • Turn text into voice</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# CACHED RESOURCES  (speed: reuse connections, avoid repeat downloads)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "DEARson-Image-Generator/1.0"})
    return s


STYLES = {
    "None": "",
    "Realistic photo": ", ultra realistic photo, 8k, sharp focus, natural lighting",
    "Cinematic": ", cinematic lighting, dramatic, film still, depth of field",
    "Anime": ", anime style, vibrant colors, detailed illustration",
    "3D render": ", 3d render, octane, soft shadows, high detail",
    "Digital art": ", digital painting, concept art, trending on artstation",
    "Logo / flyer": ", clean graphic design, bold typography, flat vector style",
}
SIZES = {
    "Square 1:1": (768, 768),
    "Portrait 9:16": (576, 1024),
    "Landscape 16:9": (1024, 576),
}


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def fetch_image(prompt: str, w: int, h: int, seed: int, model: str, hf_token: str = "") -> bytes:
    """Text -> image bytes. Pollinations (free, no key) first, Hugging Face as backup."""
    last_err = None
    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {"width": w, "height": h, "seed": seed, "model": model, "nologo": "true"}
    for _ in range(3):
        try:
            r = get_session().get(url, params=params, timeout=90)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return r.content
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
    if hf_token:
        try:
            from huggingface_hub import InferenceClient

            img = InferenceClient(token=hf_token).text_to_image(
                prompt, model="black-forest-labs/FLUX.1-schnell"
            )
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()
        except Exception as e:  # noqa: BLE001
            last_err = f"{last_err} | HF: {e}"
    raise RuntimeError(f"Image generation failed: {last_err}")


# ---- Premium image engines: ChatGPT (OpenAI) and Gemini (Google). They need YOUR API key. ----
IMG_PROVIDERS = {
    "🆓 Free (Pollinations)": "free",
    "🤖 ChatGPT (OpenAI key)": "openai",
    "💎 Gemini (Google key)": "gemini",
}
PROVIDER_NOTES = {
    "free": "Free, no key. Fine for drafts; human faces can look less real.",
    "openai": "Best realism for people. Paid per image and needs an OpenAI API key "
              "(a ChatGPT subscription alone does not include API access).",
    "gemini": "Very good people images. Needs a Google AI Studio API key; "
              "Google may require billing for image models.",
}
OPENAI_SIZES = {"Square 1:1": "1024x1024", "Portrait 9:16": "1024x1536", "Landscape 16:9": "1536x1024"}
GEMINI_RATIOS = {"Square 1:1": "1:1", "Portrait 9:16": "9:16", "Landscape 16:9": "16:9"}
PEOPLE_MODES = {
    "Off": "",
    "Realistic person": ", photorealistic real person, natural skin texture, realistic proportions, "
                        "detailed face and hands, professional photography, sharp focus",
    "Full body (for walking video)": ", full body shot from head to toe, standing upright, whole outfit and shoes "
                                     "visible, photorealistic real person, natural skin texture, realistic proportions, "
                                     "detailed face and hands, plain simple background, professional fashion photography",
}


def openai_image(prompt: str, size: str, key: str, model: str) -> bytes:
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "prompt": prompt, "size": size, "n": 1},
        timeout=240,
    )
    if r.status_code != 200:
        raise RuntimeError(f"ChatGPT (OpenAI) error {r.status_code}: {r.text[:300]}")
    item = r.json()["data"][0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    return requests.get(item["url"], timeout=120).content


def gemini_image(prompt: str, ratio: str, key: str, model: str) -> bytes:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": ratio}},
    }
    r = requests.post(url, headers=headers, json=body, timeout=240)
    if r.status_code == 400:  # some models reject imageConfig: retry without it
        body["generationConfig"].pop("imageConfig", None)
        r = requests.post(url, headers=headers, json=body, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
    for cand in r.json().get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("Gemini returned no image (the prompt may have been blocked). Try different words.")


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def generate_image(provider: str, prompt: str, w: int, h: int, seed: int, size_name: str, pol_model: str,
                   hf_token: str, openai_key: str, openai_model: str, gemini_key: str, gemini_model: str) -> bytes:
    if provider == "openai":
        return openai_image(prompt, OPENAI_SIZES[size_name], openai_key, openai_model)
    if provider == "gemini":
        return gemini_image(prompt, GEMINI_RATIOS[size_name], gemini_key, gemini_model)
    return fetch_image(prompt, w, h, seed, pol_model, hf_token)


def shrink(img_bytes: bytes, max_side: int = 1280) -> bytes:
    """Downscale big phone photos so everything runs faster."""
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# IMAGE -> VIDEO  (local Ken-Burns engine: no API, no token, never fails)
# ----------------------------------------------------------------------------
VIDEO_SIZES = {"Square 1:1": (640, 640), "Portrait 9:16": (480, 848), "Landscape 16:9": (848, 480)}
EFFECTS = ["Zoom in", "Zoom out", "Pan left", "Pan right", "Pan up", "Pan down", "Zoom + pan"]


@st.cache_data(ttl=1800, max_entries=6, show_spinner=False)
def make_video(img_bytes: bytes, effect: str, seconds: int, fps: int, size: tuple) -> bytes:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    W, H = size
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    sw, sh = img.size
    aspect = W / H
    if sw / sh > aspect:
        bh, bw = sh, sh * aspect
    else:
        bw, bh = sw, sw / aspect

    n = max(2, seconds * fps)
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8)
    try:
        for i in range(n):
            p = i / (n - 1)
            p = p * p * (3 - 2 * p)  # smooth easing
            z, dx, dy = 1.0, 0.0, 0.0
            if effect == "Zoom in":
                z = 1 + 0.35 * p
            elif effect == "Zoom out":
                z = 1.35 - 0.35 * p
            elif effect in ("Pan left", "Pan right"):
                z = 1.25
                dx = (1 - 2 * p) * (1 if effect == "Pan left" else -1)
            elif effect in ("Pan up", "Pan down"):
                z = 1.25
                dy = (1 - 2 * p) * (1 if effect == "Pan up" else -1)
            else:  # Zoom + pan
                z = 1.05 + 0.3 * p
                dx = 0.6 * (1 - 2 * p)
            cw, ch = bw / z, bh / z
            cx = sw / 2 + dx * max(0, (sw - cw) / 2)
            cy = sh / 2 + dy * max(0, (sh - ch) / 2)
            box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
            frame = img.resize((W, H), Image.BICUBIC, box=box)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


# ---- Real AI motion (people actually move) via free LTX-Video Space on Hugging Face ----
LTX_SPACE = "Lightricks/ltx-video-distilled"
LTX_SIZES = {"Landscape 16:9": (704, 512), "Portrait 9:16": (512, 704), "Square 1:1": (640, 640)}
CINEMA = ("cinematic lighting, shallow depth of field, warm film color grading, high detail")
NEG = ("worst quality, blurry, jittery, distorted face, deformed hands, extra limbs, extra legs, morphing, "
       "frozen, static pose, slideshow, sliding picture, zoom only, stiff mannequin, feet gliding, text, watermark")
DEFAULT_SCENE = "The person comes alive, breathing, blinking and moving naturally"
WALK_WORDS = ("walk", "catwalk", "stride", "strut", "runway", "stroll", "step", "pace")
WALK_DETAIL = ("The person takes real steps: legs alternate, feet land naturally, hips sway, arms swing, "
               "torso and shoulders turn, hair and clothes move with every step and weight shifts from foot to foot. "
               "The camera tracks smoothly at eye level.")


def motion_prompt(scene: str, cinematic: bool) -> str:
    """Turn the user's one-line description into a detailed, action-first video prompt."""
    s = scene.strip().rstrip(".")
    parts = [s + "."]
    if any(w in s.lower() for w in WALK_WORDS):
        parts.append(WALK_DETAIL)
    parts.append("Real human motion with natural body movement, lifelike face and expressions, smooth stable camera.")
    if cinematic:
        parts.append(CINEMA + ".")
    return " ".join(parts)


@st.cache_resource(ttl=3000, show_spinner=False)
def get_client(space: str, token: str):
    from gradio_client import Client

    if token:
        try:
            return Client(space, hf_token=token)
        except TypeError:
            return Client(space, token=token)
    return Client(space)


def ltx_clip(img_bytes: bytes, prompt: str, seconds: float, size: tuple, token: str, space: str) -> bytes:
    """One AI clip from one image. People really move, smile, dance (prompt-driven)."""
    from gradio_client import handle_file

    W, H = size
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(img_bytes)
    tmp.close()
    client = get_client(space, token)
    try:
        try:
            res = client.predict(
                prompt=prompt, negative_prompt=NEG, input_image_filepath=handle_file(tmp.name),
                height_ui=H, width_ui=W, mode="image-to-video", duration_ui=float(seconds),
                api_name="/image_to_video",
            )
        except Exception as e1:  # noqa: BLE001
            if any(k in str(e1).lower() for k in ("quota", "gpu", "queue", "exceeded")):
                raise
            res = client.predict(
                prompt, NEG, handle_file(tmp.name), None, H, W, "image-to-video",
                float(seconds), 9, 42, True, 1.0, True, api_name="/image_to_video",
            )
    finally:
        os.remove(tmp.name)
    out = res[0] if isinstance(res, (list, tuple)) else res
    for _ in range(2):
        if isinstance(out, dict):
            out = out.get("video") or out.get("path") or out
    if not isinstance(out, str):
        raise RuntimeError("The video Space returned no video.")
    with open(out, "rb") as f:
        return f.read()


def last_frame(video_bytes: bytes) -> bytes:
    """Last frame of a clip: used so the next scene continues from where this one ended."""
    import imageio.v2 as imageio
    from PIL import Image

    p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    p.write(video_bytes)
    p.close()
    last = None
    reader = imageio.get_reader(p.name)
    for fr in reader:
        last = fr
    reader.close()
    os.remove(p.name)
    buf = io.BytesIO()
    Image.fromarray(last).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _ff() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("Video tool error: " + r.stderr[-300:])


def _probe(path: str) -> str:
    return subprocess.run([_ff(), "-i", path], capture_output=True, text=True).stderr


def _has_audio(path: str) -> bool:
    return "Audio:" in _probe(path)


def _video_size(path: str):
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})[\s,\[]", _probe(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def join_clips(clips: list) -> bytes:
    """Stitch scenes into one video and KEEP the voice (silent clips get a silent track)."""
    if len(clips) == 1:
        return clips[0]
    tmp = tempfile.mkdtemp()
    paths = []
    for i, c in enumerate(clips):
        p = os.path.join(tmp, f"in{i}.mp4")
        with open(p, "wb") as f:
            f.write(c)
        paths.append(p)
    W, H = _video_size(paths[0]) or (640, 640)
    W, H = W - W % 2, H - H % 2
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
          f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=24,setsar=1,format=yuv420p")
    ff = _ff()
    norm = []
    for i, p in enumerate(paths):
        o = os.path.join(tmp, f"n{i}.mp4")
        if _has_audio(p):
            cmd = [ff, "-y", "-i", p, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                   "-c:a", "aac", "-ar", "44100", "-ac", "2", o]
        else:
            cmd = [ff, "-y", "-i", p, "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest",
                   "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                   "-c:a", "aac", "-ar", "44100", "-ac", "2", o]
        _run(cmd)
        norm.append(o)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.write("".join(f"file '{p}'\n" for p in norm))
    out = os.path.join(tmp, "out.mp4")
    _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])
    with open(out, "rb") as f:
        return f.read()


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if any(k in msg.lower() for k in ("quota", "exceeded", "gpu")):
        return ("The free GPU time is used up right now. Paste a free Hugging Face token under "
                "⚙️ Settings (your own daily quota is used), or try again in a few hours.")
    return msg


# ----------------------------------------------------------------------------
# TALKING (lip-sync) + GEMINI VEO
# ----------------------------------------------------------------------------
LIPSYNC_SPACE = "pragnakalp/Wav2lip-ZeroGPU"  # free public Space: image + voice -> mouth moves


def crop_face(img_bytes: bytes, out_size: tuple) -> bytes:
    """Crop head-and-shoulders (same shape as the video) so the mouth is big enough to lip-sync."""
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    ow, oh = out_size
    aspect = ow / oh
    face = None
    try:
        import cv2

        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = det.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces):
            face = max(faces, key=lambda f: f[2] * f[3])
    except Exception:  # noqa: BLE001
        face = None
    if face is not None:
        fx, fy, fw, fh = [int(v) for v in face]
        cw, cx, top = fw * 3.4, fx + fw / 2, fy - fh * 0.8
    else:
        cw, cx, top = W * 0.7, W / 2, 0
    cw = min(max(cw, 160), W)
    ch = cw / aspect
    if ch > H:
        ch = H
        cw = ch * aspect
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(top, 0), H - ch)
    crop = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((ow, oh), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _find_video_path(res):
    items = res if isinstance(res, (list, tuple)) else [res]
    for it in items:
        for _ in range(3):
            if isinstance(it, dict):
                it = it.get("video") or it.get("path") or it.get("value") or it
        if isinstance(it, str) and it.lower().endswith((".mp4", ".webm", ".mov")):
            return it
    raise RuntimeError("The lip-sync Space returned no video (it may be busy). Try again in a minute.")


def lipsync_talk(frame_bytes: bytes, audio_mp3: bytes, token: str, out_size: tuple) -> bytes:
    """Face + voice -> video where the mouth really moves with the words (Wav2Lip, free Space)."""
    from gradio_client import handle_file

    tmp = tempfile.mkdtemp()
    mp3 = os.path.join(tmp, "voice.mp3")
    with open(mp3, "wb") as f:
        f.write(audio_mp3)
    wav = os.path.join(tmp, "voice.wav")
    _run([_ff(), "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav])
    face = os.path.join(tmp, "face.jpg")
    with open(face, "wb") as f:
        f.write(crop_face(frame_bytes, out_size))
    client = get_client(LIPSYNC_SPACE, token)
    res = client.predict(input_image=handle_file(face), input_audio=handle_file(wav), api_name="/run_infrence")
    raw = _find_video_path(res)
    out = os.path.join(tmp, "talk.mp4")
    # always put OUR chosen voice on the clip
    _run([_ff(), "-y", "-i", raw, "-i", mp3, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out])
    with open(out, "rb") as f:
        return f.read()


def still_with_audio(frame_bytes: bytes, audio_mp3: bytes) -> bytes:
    """Fallback when lip-sync is unavailable: the voice plays over the last frame."""
    tmp = tempfile.mkdtemp()
    img, mp3, out = (os.path.join(tmp, n) for n in ("f.jpg", "v.mp3", "o.mp4"))
    with open(img, "wb") as f:
        f.write(frame_bytes)
    with open(mp3, "wb") as f:
        f.write(audio_mp3)
    _run([_ff(), "-y", "-loop", "1", "-i", img, "-i", mp3, "-c:v", "libx264", "-tune", "stillimage",
          "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:a", "aac", "-shortest", out])
    with open(out, "rb") as f:
        return f.read()


def veo_prompt(story: str, speech: str) -> str:
    actions = [s.strip().rstrip(".") for s in story.splitlines() if s.strip()][:4] or [DEFAULT_SCENE]
    text = ". Then ".join(actions) + ". "
    if any(w in text.lower() for w in WALK_WORDS):
        text += WALK_DETAIL + " "
    text += "Realistic human motion, lifelike face, steady camera, cinematic lighting."
    if speech.strip():
        text += (' The person looks at the camera and says clearly, with lips moving in sync with the words: "'
                 + speech.strip() + '"')
    return text


def veo_clip(img_bytes: bytes, prompt: str, key: str, model: str, aspect: str) -> bytes:
    """Gemini Veo image-to-video. One 8-second clip with a real moving body AND spoken voice (paid)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    op = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=types.Image(image_bytes=img_bytes, mime_type="image/jpeg"),
        config=types.GenerateVideosConfig(aspect_ratio=aspect),
    )
    t0 = time.time()
    while not op.done:
        if time.time() - t0 > 600:
            raise RuntimeError("Gemini is taking too long. Try again in a few minutes.")
        time.sleep(10)
        op = client.operations.get(op)
    res = getattr(op, "response", None) or getattr(op, "result", None)
    if callable(res):
        res = res()
    vids = getattr(res, "generated_videos", None)
    if not vids:
        raise RuntimeError("Gemini returned no video (the description may have been blocked). Try different words.")
    v = vids[0].video
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        client.files.download(file=v)
    except Exception:  # noqa: BLE001
        pass
    v.save(path)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def veo_error(e: Exception) -> str:
    m = str(e)
    if any(k in m.lower() for k in ("billing", "quota", "permission", "not found", "403", "429", "limit: 0")):
        return ("Gemini Veo needs a Google API key with billing turned on (it is not free). "
                "Check the key and the model name in ⚙️ Settings. Details: " + m[:200])
    return m


# ----------------------------------------------------------------------------
# TEXT -> AUDIO
# ----------------------------------------------------------------------------
FALLBACK_VOICES = {
    "English (US) - Jenny (F)": "en-US-JennyNeural",
    "English (US) - Guy (M)": "en-US-GuyNeural",
    "English (UK) - Sonia (F)": "en-GB-SoniaNeural",
    "English (UK) - Ryan (M)": "en-GB-RyanNeural",
    "English (Nigeria) - Ezinne (F)": "en-NG-EzinneNeural",
    "English (Nigeria) - Abeo (M)": "en-NG-AbeoNeural",
    "French - Denise (F)": "fr-FR-DeniseNeural",
    "Spanish - Elvira (F)": "es-ES-ElviraNeural",
}


@st.cache_data(ttl=86400, show_spinner=False)
def get_voices() -> dict:
    try:
        import edge_tts

        voices = asyncio.run(edge_tts.list_voices())
        out = {f"{v['Locale']} - {v['ShortName'].split('-')[-1].replace('Neural','')} ({v['Gender'][0]})": v["ShortName"] for v in voices}
        return dict(sorted(out.items())) or FALLBACK_VOICES
    except Exception:  # noqa: BLE001
        return FALLBACK_VOICES


def tts_edge(text: str, voice: str, rate: int, pitch: int) -> bytes:
    import edge_tts

    path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

    async def run():
        await edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz").save(path)

    asyncio.run(run())
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def tts_gtts(text: str, lang: str = "en") -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


def clone_speak(text: str, ref_bytes: bytes, lang: str, space: str, api_name: str) -> bytes:
    """Voice cloning through a public Hugging Face Space (XTTS by default)."""
    from gradio_client import Client, handle_file

    ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    ref.write(ref_bytes)
    ref.close()
    client = Client(space)
    result = client.predict(text, lang, handle_file(ref.name), handle_file(ref.name),
                            False, False, False, True, api_name=api_name)
    os.remove(ref.name)
    items = result if isinstance(result, (list, tuple)) else [result]
    for it in items:
        if isinstance(it, dict):
            it = it.get("value") or it.get("path")
        if isinstance(it, str) and it.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            with open(it, "rb") as f:
                return f.read()
    raise RuntimeError("The cloning Space returned no audio (it may be busy or changed).")


# ----------------------------------------------------------------------------
# NAVIGATION  (only the chosen page runs => faster)
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("gen_img", None)
ss.setdefault("gen_prompt", "")
ss.setdefault("gen_seed", random.randint(1, 10**6))
ss.setdefault("vid_src", None)
ss.setdefault("vid_out", None)
ss.setdefault("clone_ref", None)
ss.setdefault("prov", "🆓 Free (Pollinations)")
ss.setdefault("vid_pitch", 0)
ss.setdefault("vid_prompts", [])

page = st.radio(
    "Page",
    ["🎨 Image", "🎬 Image → Video", "🔊 Text → Audio"],
    horizontal=True,
    label_visibility="collapsed",
)

def _secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # noqa: BLE001
        return os.environ.get(name, "")


with st.expander("⚙️ Settings (API keys)"):
    hf_token = st.text_input(
        "Hugging Face token (free; strongly recommended for AI video, gives you your own GPU quota)",
        type="password",
        value=_secret("HF_TOKEN"),
    )
    openai_key = st.text_input("OpenAI API key (for ChatGPT images)", type="password",
                               value=_secret("OPENAI_API_KEY"))
    openai_model = st.text_input("ChatGPT image model", value="gpt-image-2.5-flare")
    gemini_key = st.text_input("Gemini API key (for Gemini images and Veo video)", type="password",
                               value=_secret("GEMINI_API_KEY"))
    gemini_model = st.text_input("Gemini image model", value="gemini-2.5-flash-image")
    veo_model = st.text_input("Gemini Veo video model", value="veo-3.1-fast-generate-preview")
    st.caption("Keys stay in this session only. On Streamlit Cloud you can also store them under Settings → Secrets.")


def make_image(text: str, size_name: str, provider_label: str, pol_model: str = "flux", seed=None) -> bytes:
    prov = IMG_PROVIDERS.get(provider_label, "free")
    if prov == "openai" and not openai_key.strip():
        raise RuntimeError("Paste your OpenAI API key in ⚙️ Settings first.")
    if prov == "gemini" and not gemini_key.strip():
        raise RuntimeError("Paste your Gemini API key in ⚙️ Settings first.")
    w, h = SIZES[size_name]
    seed = seed if seed is not None else random.randint(1, 10**6)
    return generate_image(prov, text, w, h, seed, size_name, pol_model, hf_token,
                          openai_key.strip(), openai_model.strip(), gemini_key.strip(), gemini_model.strip())

# ============================================================================
# PAGE 1 - IMAGE GENERATOR
# ============================================================================
if page.startswith("🎨"):
    prompt = st.text_area("Describe the image", value=ss.gen_prompt, height=100,
                          placeholder="A beautiful Nigerian woman in a fitted dress standing in a bright studio")
    a1, a2 = st.columns(2)
    provider = a1.selectbox("AI engine", list(IMG_PROVIDERS), key="img_provider")
    people = a2.selectbox("👤 People mode", list(PEOPLE_MODES),
                          help="Full body gives a head-to-toe picture, which is needed for a walking video.")
    ss.prov = provider
    st.caption(PROVIDER_NOTES[IMG_PROVIDERS[provider]])
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox("Style", list(STYLES))
    size_name = c2.selectbox("Size", list(SIZES))
    model = c3.selectbox("Free engine", ["flux", "turbo"])
    if people.startswith("Full") and not size_name.startswith("Portrait"):
        st.caption("💡 Full body works best with Size = Portrait 9:16.")
    suffix = STYLES[style] + PEOPLE_MODES[people]

    b1, b2, b3 = st.columns(3)
    go = b1.button("✨ Generate", key="gen")
    again = b2.button("🔄 New version", key="regen")
    clear = b3.button("🗑 Clear", key="clear")

    if clear:
        ss.gen_img, ss.gen_prompt = None, ""
        st.rerun()

    if (go or again) and prompt.strip():
        ss.gen_prompt = prompt.strip()
        ss.gen_seed = random.randint(1, 10**6)
        with st.spinner("Painting your image..."):
            try:
                ss.gen_img = make_image(ss.gen_prompt + suffix, size_name, provider, model, ss.gen_seed)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    elif (go or again) and not prompt.strip():
        st.warning("Type a description first.")

    if ss.gen_img:
        st.image(ss.gen_img, use_container_width=True)
        st.download_button("⬇️ Download image", ss.gen_img, "dearson_image.png", "image/png")

        st.markdown("**💬 Comment on the result: what should change?**")
        comment = st.text_area("Comment", key="comment_img", height=70, label_visibility="collapsed",
                               placeholder="Make it brighter, add a signboard that says DEARson, remove the people...")
        r1, r2 = st.columns(2)
        if r1.button("🎯 Generate with my comment", key="newimg"):
            if comment.strip():
                new_prompt = f"{ss.gen_prompt}, {comment.strip()}"
                with st.spinner("Applying your comment..."):
                    try:
                        ss.gen_seed = random.randint(1, 10**6)
                        ss.gen_img = make_image(new_prompt + suffix, size_name, provider, model, ss.gen_seed)
                        ss.gen_prompt = new_prompt
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
            else:
                st.warning("Write a comment first.")
        if r2.button("🎬 Send to video", key="mkvid"):
            try:
                ss.vid_src = shrink(ss.gen_img)
            except Exception:  # noqa: BLE001
                ss.vid_src = ss.gen_img
            ss.vid_out = None
            st.success("Sent! Open the 🎬 Image → Video page.")

# ============================================================================
# PAGE 2 - IMAGE TO VIDEO
# ============================================================================
elif page.startswith("🎬"):
    VIDEO_ENGINES = [
        "🆓 Free: AI walking + lip-sync talking",
        "🌟 Best: Gemini Veo (paid key): walks and talks in one clip",
        "🖼 Simple zoom/pan (still picture only)",
    ]
    ACTION_PRESETS = {
        "🚶 Catwalk: walk and show side waist": (
            "She walks confidently toward the camera in a smooth catwalk stride, hips swaying and arms swinging naturally\n"
            "She stops, turns sideways to show her side profile and waistline with one hand on her waist, "
            "then looks back at the camera and smiles"
        ),
        "💃 Walk, turn and pose": (
            "She walks forward with a graceful stride\n"
            "She turns slowly in a full circle to show her whole outfit, then strikes a confident pose"
        ),
        "👋 Walk, wave and smile": (
            "She walks toward the camera smiling\n"
            "She waves hello with one hand and laughs warmly"
        ),
        "🧍 Stand and breathe naturally": (
            "She stands naturally, breathing, blinking and gently shifting her weight while looking at the camera"
        ),
    }

    def _apply_preset():
        p = ss.get("vid_preset")
        if p in ACTION_PRESETS:
            ss["vid_story"] = ACTION_PRESETS[p]

    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if up is not None:
        try:
            ss.vid_src = shrink(up.getvalue())
        except Exception:  # noqa: BLE001
            st.error("That file couldn't be read as an image. Try another one.")

    if ss.vid_src:
        st.image(ss.vid_src, caption="Source image", use_container_width=True)
        st.caption("💡 For walking, use a FULL-BODY picture (head to toe). A close-up face cannot walk.")

        engine = st.selectbox("Video engine", VIDEO_ENGINES)
        is_free, is_veo, is_zoom = engine.startswith("🆓"), engine.startswith("🌟"), engine.startswith("🖼")

        c1, c2 = st.columns(2)
        vsize = c1.selectbox("Shape", list(LTX_SIZES))
        secs = c2.slider("Secs/scene", 2, 8, 5)

        story, speech, cinematic = "", "", True
        if is_zoom:
            m1, m2 = st.columns(2)
            effect = m1.selectbox("Motion", EFFECTS)
            fps = m2.selectbox("FPS", [12, 18, 24], index=1)
        else:
            st.selectbox("⚡ Quick action", ["(write my own)"] + list(ACTION_PRESETS),
                         key="vid_preset", on_change=_apply_preset)
            story = st.text_area(
                "🎬 Describe what happens (one line = one scene, max 4). The AI follows this exactly.",
                key="vid_story", height=140,
                placeholder="She walks confidently toward the camera in a catwalk stride\n"
                            "She turns sideways to show her side waist, then smiles at the camera",
            )
            speech = st.text_area(
                "🗣 What should she say? (her mouth will open and talk)" + (" Keep it under about 20 words." if is_veo else ""),
                key="vid_speech", height=80,
                placeholder="Hello, welcome to Dearson. I am happy to show you our new collection.",
            )
            cinematic = st.checkbox("🎞 Cinematic look", value=True)

        voices, vname, vrate = {}, None, 0
        if is_free and speech.strip():
            voices = get_voices()
            names = list(voices)
            default = next((i for i, n in enumerate(names) if "en-NG" in n and "(F)" in n),
                           next((i for i, n in enumerate(names) if "en-US" in n and "(F)" in n), 0))
            vname = st.selectbox("🎙 Voice of choice", names, index=default, key="vid_voice")
            st.write(f"Voice pitch: **{ss.vid_pitch:+d} Hz**")
            pb1, pb2, pb3 = st.columns(3)
            if pb1.button("⬇️ Too high", key="pitchdn"):
                ss.vid_pitch = max(-30, ss.vid_pitch - 5)
                st.rerun()
            if pb2.button("🔄 Reset", key="pitchreset"):
                ss.vid_pitch = 0
                st.rerun()
            if pb3.button("⬆️ Too low", key="pitchup"):
                ss.vid_pitch = min(30, ss.vid_pitch + 5)
                st.rerun()
            vrate = st.slider("Speed %", -50, 50, 0, key="vid_rate")
            if st.button("🔊 Test voice", key="speak"):
                try:
                    st.audio(tts_edge(speech, voices[vname], vrate, ss.vid_pitch), format="audio/mp3")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Audio failed: {e}")

        st.markdown("**🖼 Want a different picture first? Describe it:**")
        vcomment = st.text_area("vc", key="vid_comment", height=70, label_visibility="collapsed",
                                placeholder="A tall elegant woman in a green dress, full body, standing in a studio")

        v1, v2, v3 = st.columns(3)
        make = v1.button("🎬 Make video", key="mkvid")
        newi = v2.button("🖼 New image", key="newimg")
        rst = v3.button("♻️ Reset", key="resetvid")

        if rst:
            ss.vid_src = ss.vid_out = None
            ss.vid_prompts = []
            st.rerun()

        if newi:
            if vcomment.strip():
                with st.spinner("Generating new image from your description..."):
                    try:
                        ss.vid_src = shrink(make_image(vcomment.strip(), "Portrait 9:16", ss.prov))
                        ss.vid_out = None
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(str(e))
            else:
                st.warning("Describe the new image first.")

        if make:
            ss.vid_out = None
            ss.vid_prompts = []
            if is_zoom:
                with st.spinner("Rendering video..."):
                    try:
                        ss.vid_out = make_video(ss.vid_src, effect, secs, fps, VIDEO_SIZES[vsize])
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Video failed: {e}")

            elif is_veo:
                if not gemini_key.strip():
                    st.error("Paste your Gemini API key in ⚙️ Settings first.")
                else:
                    with st.spinner("Gemini Veo is filming (1 to 3 minutes)..."):
                        try:
                            from PIL import Image

                            w0, h0 = Image.open(io.BytesIO(ss.vid_src)).size
                            vp = veo_prompt(story, speech)
                            ss.vid_prompts = [vp]
                            ss.vid_out = veo_clip(ss.vid_src, vp, gemini_key.strip(), veo_model.strip(),
                                                  "9:16" if h0 > w0 else "16:9")
                        except Exception as e:  # noqa: BLE001
                            st.error(veo_error(e))

            else:  # free: AI walking scenes, then a lip-synced talking scene
                scenes = [s.strip() for s in story.splitlines() if s.strip()][:4]
                if not scenes and not speech.strip():
                    scenes = [DEFAULT_SCENE]
                clips, current = [], ss.vid_src
                with st.status("Creating your video (about 1 minute per scene)...", expanded=True) as status:
                    try:
                        for i, scene in enumerate(scenes, 1):
                            st.write(f"🚶 Scene {i}/{len(scenes)}: {scene}")
                            mp = motion_prompt(scene, cinematic)
                            ss.vid_prompts.append(mp)
                            clip = ltx_clip(current, mp, secs, LTX_SIZES[vsize], hf_token, LTX_SPACE)
                            clips.append(clip)
                            current = last_frame(clip)
                        if speech.strip():
                            st.write("🗣 She starts talking (mouth moving with her voice)...")
                            audio = tts_edge(speech, voices[vname], vrate, ss.vid_pitch)
                            try:
                                clips.append(lipsync_talk(current, audio, hf_token, LTX_SIZES[vsize]))
                            except Exception as e:  # noqa: BLE001
                                st.warning("Lip-sync is not available right now, so her voice plays over the last "
                                           f"picture instead. Try again later. ({str(e)[:160]})")
                                clips.append(still_with_audio(current, audio))
                        ss.vid_out = join_clips(clips)
                        status.update(label="Video ready!", state="complete")
                    except Exception as e:  # noqa: BLE001
                        status.update(label="Video stopped", state="error")
                        st.error(friendly_error(e))
                        if clips:
                            try:
                                ss.vid_out = join_clips(clips)
                                st.warning(f"Showing the {len(clips)} part(s) that finished.")
                            except Exception:  # noqa: BLE001
                                pass

        if ss.vid_out:
            st.video(ss.vid_out)
            st.download_button("⬇️ Download video", ss.vid_out, "dearson_video.mp4", "video/mp4")
            if ss.vid_prompts:
                with st.expander("🔎 What the AI was told (built from your description)"):
                    for p in ss.vid_prompts:
                        st.write(p)
    else:
        st.info("Upload an image, or create one on the 🎨 Image page and tap 'Send to video'.")

# ============================================================================
# PAGE 3 - TEXT TO AUDIO + VOICE CLONING
# ============================================================================
else:
    text = st.text_area("Text to speak", height=130, placeholder="Welcome to Dearson Barbing & Technology...")
    engine = st.radio("Engine", ["Edge voices (best)", "Google TTS (backup)"], horizontal=True)

    if engine.startswith("Edge"):
        voices = get_voices()
        names = list(voices)
        default = next((i for i, n in enumerate(names) if "en-NG" in n or "en-US" in n), 0)
        vname = st.selectbox("Voice", names, index=default)
        s1, s2 = st.columns(2)
        rate = s1.slider("Speed %", -50, 50, 0)
        pitch = s2.slider("Pitch Hz", -30, 30, 0)
    else:
        lang = st.selectbox("Language", ["en", "fr", "es", "de", "pt", "yo", "ha", "sw"])

    a1, _, _ = st.columns(3)
    if a1.button("🔊 Speak", key="speak"):
        if text.strip():
            with st.spinner("Creating audio..."):
                try:
                    audio = (tts_edge(text, voices[vname], rate, pitch) if engine.startswith("Edge")
                             else tts_gtts(text, lang))
                    st.audio(audio, format="audio/mp3")
                    st.download_button("⬇️ Download audio", audio, "dearson_audio.mp3", "audio/mpeg")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Audio failed: {e}")
        else:
            st.warning("Type some text first.")

    st.divider()
    st.subheader("🧬 Voice cloning")
    st.caption("Upload 10–30 seconds of clear speech (WAV/MP3), no background noise.")
    ref = st.file_uploader("Voice sample", type=["wav", "mp3", "m4a", "ogg"], key="refup")
    with st.expander("Cloning engine (advanced)"):
        space = st.text_input("Hugging Face Space", value="coqui/xtts")
        api_name = st.text_input("API name", value="/predict")
        clang = st.selectbox("Cloned voice language", ["en", "fr", "es", "de", "pt", "it"])

    k1, k2, k3 = st.columns(3)
    if k1.button("🧬 Clone voice", key="clone"):
        if ref is not None:
            ss.clone_ref = ref.getvalue()
            st.success("Voice sample saved. Now tap 'Speak cloned'.")
        else:
            st.warning("Upload a voice sample first.")
    if k2.button("🎙 Speak cloned", key="speakclone"):
        if not ss.clone_ref:
            st.warning("Clone a voice first.")
        elif not text.strip():
            st.warning("Type some text first.")
        else:
            with st.spinner("Cloning and speaking (can take a minute)..."):
                try:
                    cloned = clone_speak(text, ss.clone_ref, clang, space, api_name)
                    st.audio(cloned, format="audio/wav")
                    st.download_button("⬇️ Download cloned audio", cloned, "dearson_cloned.wav", "audio/wav")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Cloning failed: {e}. The free Space may be busy; try again shortly.")
    if k3.button("🗑 Clear voice", key="clearclone"):
        ss.clone_ref = None
        st.info("Voice sample removed.")
