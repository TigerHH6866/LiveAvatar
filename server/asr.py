"""语音识别：阿里云百炼 Qwen-Audio-ASR（HTTP）。

移植自参考项目 avatarhuman 的 qwenasr.py，把 aiohttp 换成同步 requests，
以便直接在 Flask 视图里调用。
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _credential(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import key as key_module  # type: ignore[import-not-found]

        return str(getattr(key_module, name, default) or default)
    except Exception:  # noqa: BLE001
        return default


ASR_MODEL = os.environ.get("LIVEAVATAR_ASR_MODEL", "qwen-audio-3.1-asr-flash")
# 参考项目的 key.py 里没有 ALIYUN_WORKSPACE_ID，这里从 TTS 域名反推同一个业务空间。
_DEFAULT_WORKSPACE = "llm-prl2mma08soey3qf"
WORKSPACE_ID = (
    os.environ.get("ALIYUN_WORKSPACE_ID")
    or _credential("ALIYUN_WORKSPACE_ID")
    or _DEFAULT_WORKSPACE
)
ASR_URL = (
    f"https://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com"
    "/api/v1/services/aigc/multimodal-generation/generation"
)

MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "ogg": "audio/ogg",
}


def recognize(audio: bytes, fmt: str = "wav", sample_rate: int = 16000,
              timeout: float = 60) -> str:
    """识别一段音频并返回文本，语种由模型自动判断。"""
    fmt = (fmt or "wav").lower()
    if fmt not in MIME_TYPES:
        raise RuntimeError(f"不支持的音频格式：{fmt}")

    data_uri = f"data:{MIME_TYPES[fmt]};base64,{base64.b64encode(audio).decode()}"
    payload = {
        "model": ASR_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": data_uri}}],
                }
            ]
        },
        "parameters": {"format": fmt, "sample_rate": str(sample_rate)},
    }
    headers = {
        "Authorization": f"Bearer {_credential('ALIYUN_TTS_API_KEY')}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }

    response = requests.post(ASR_URL, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"ASR HTTP {response.status_code}：{response.text[:500]}")

    result = response.json()
    output = result.get("output") or {}
    text = output.get("text") or (output.get("sentence") or {}).get("text", "")
    return (text or "").strip()
