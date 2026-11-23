"""LiveAvatar Web：浏览器端推理 + 服务端只做 ASR / LLM / TTS。

没有 WebRTC：视频由浏览器自己用 ONNX Runtime Web 生成，服务端只负责：
  /api/asr       语音识别（Qwen-Audio-ASR）
  /api/chat      Coze 流式对话 + Qwen 实时 TTS，SSE 推文本与 PCM
  /api/interrupt 打断本轮播报
其余是静态资源（模型、素材、前端页面）。

启动：
  python server/app.py            # 默认 127.0.0.1:7861
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from queue import Empty, Queue

from flask import Flask, Response, jsonify, request, send_from_directory

import asr
import chat

# Windows 上 .mjs/.wasm 常被识别成 text/plain，模块和 wasm 流式编译都会失败。
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("image/webp", ".webp")

PROJECT = Path(__file__).resolve().parent.parent
CLIENT_DIR = PROJECT / "client"
ASSETS_DIR = PROJECT / "assets"
MODELS_DIR = PROJECT / "models"

AVATAR = os.environ.get("LIVEAVATAR_AVATAR", "model010-1080-short")
AVATAR_DIR = ASSETS_DIR / AVATAR
MAX_UPLOAD = 32 * 1024 * 1024

# 可选：模型/素材/ORT wasm 放在 OSS 时填对象存储前缀（不含结尾斜杠），
# 例如 https://ref-image.oss-cn-shanghai.aliyuncs.com/liveavatar
ASSET_BASE = os.environ.get("LIVEAVATAR_ASSET_BASE", "").rstrip("/")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


@app.after_request
def _no_cache(response: Response) -> Response:
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif request.path.endswith((".html", ".js", ".mjs", ".css")):
        # 前端代码必须每次回源校验，否则改了 worker.js 浏览器还在跑旧模块。
        response.headers["Cache-Control"] = "no-cache"
    # 跨源隔离：让 onnxruntime-web 的 wasm 线程（SharedArrayBuffer）可用。
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


def _avatar_meta() -> dict:
    path = AVATAR_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少素材索引 {path}，先运行 tools/prepare_assets.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _to_wav16k(source: Path, target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ]
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError("音频转码失败：" + proc.stderr[-500:])


@app.get("/")
def index() -> Response:
    return send_from_directory(CLIENT_DIR, "index.html")


@app.get("/<path:filename>")
def client_files(filename: str) -> Response:
    return send_from_directory(CLIENT_DIR, filename)


@app.get("/api/config")
def config() -> Response:
    try:
        meta = _avatar_meta()
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    avatar = meta["avatar"]
    # 大文件（模型 / 素材 / ORT wasm）配了 OSS 就从对象存储取；留空走本机路由。
    # 注意 OSS 上的目录叫 avatars/，本机路由是 /assets/。
    if ASSET_BASE:
        models_base = f"{ASSET_BASE}/models"
        asset_base = f"{ASSET_BASE}/avatars/{avatar}"
        ort_wasm = f"{ASSET_BASE}/ort/"
        # VAD 是独立的一份 ORT 运行时，和主推理那套不共用
        vad_base = f"{ASSET_BASE}/vad"
    else:
        models_base = "/models"
        asset_base = f"/assets/{avatar}"
        ort_wasm = "/vendor/ort/"
        vad_base = "/vad"
    return jsonify(
        avatar=avatar,
        meta=meta,
        models={
            "feather": f"{models_base}/live_hubert.onnx",
            "visual": f"{models_base}/{avatar}.onnx",
        },
        media={
            "loop": f"{asset_base}/loop.mp4",
            "faces": f"{asset_base}/faces",
            # 离线 RVM 抠像蒙版；素材没跑抠像脚本时为 None，前端回退不透明画面
            "mask": f"{asset_base}/{meta['mask']}" if meta.get("mask") else None,
            "vad": f"{vad_base}/silero_vad.onnx",
        },
        ort={
            "script": "/vendor/ort/ort.webgpu.min.js",
            "wasm": ort_wasm,
            "vad_wasm": f"{vad_base}/",
        },
        audio={},
    )


@app.get("/models/<path:filename>")
def model_files(filename: str) -> Response:
    return send_from_directory(MODELS_DIR, filename)


@app.get("/assets/<avatar>/<path:filename>")
def asset_files(avatar: str, filename: str) -> Response:
    return send_from_directory(ASSETS_DIR / avatar, filename)


@app.get("/vendor/<path:filename>")
def vendor_files(filename: str) -> Response:
    return send_from_directory(CLIENT_DIR / "vendor", filename)


@app.post("/api/asr")
def api_asr() -> Response:
    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return jsonify(error="没有收到音频"), 400
    suffix = Path(upload.filename).suffix.lower() or ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / f"input{suffix}"
        upload.save(raw)
        wav = Path(tmp) / "converted.wav"
        try:
            _to_wav16k(raw, wav)
            text = asr.recognize(wav.read_bytes(), fmt="wav")
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=str(exc)), 502
    return jsonify(text=text, empty=not text.strip())


@app.post("/api/chat")
def api_chat() -> Response:
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    session_id = str(payload.get("sessionid") or uuid.uuid4().hex)
    if not text:
        return jsonify(error="内容为空"), 400

    events: Queue = Queue()
    chat.start(session_id, text, events)

    def stream():
        yield f"data: {json.dumps({'type': 'session', 'sessionid': session_id})}\n\n"
        while True:
            try:
                item = events.get(timeout=15)
            except Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                break
            yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            if item.get("type") in ("done", "error"):
                break

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/interrupt")
def api_interrupt() -> Response:
    payload = request.get_json(silent=True) or {}
    chat.interrupt(str(payload.get("sessionid") or ""))
    return jsonify(ok=True)


@app.post("/api/reset")
def api_reset() -> Response:
    payload = request.get_json(silent=True) or {}
    chat.reset_conversation(str(payload.get("sessionid") or ""))
    return jsonify(ok=True)


@app.get("/api/health")
def api_health() -> Response:
    return jsonify(
        ok=True,
        avatar=AVATAR,
        avatar_ready=AVATAR_DIR.exists(),
        models_ready=all(
            (MODELS_DIR / name).exists()
            for name in ("live_hubert.onnx", f"{AVATAR}.onnx")
        ),
    )


if __name__ == "__main__":
    port = int(os.environ.get("LIVEAVATAR_WEB_PORT", "7861"))
    host = os.environ.get("LIVEAVATAR_WEB_HOST", "127.0.0.1")
    print(f"[web] client   = {CLIENT_DIR}")
    print(f"[web] avatar   = {AVATAR_DIR}")
    print(f"[web] models   = {MODELS_DIR}")
    print(f"[web] http://{host}:{port}/")
    app.run(host=host, port=port, threaded=True)
