"""实时对话管线：Coze 流式 LLM → 分句 → 阿里 Qwen 实时 TTS → 浏览器 PCM。

与 C++ 版 web/chat.py 的唯一区别在落点：这里不再把 PCM 灌进引擎进程，
而是把音频块通过 SSE 事件发给浏览器（浏览器自己做推理和播放），
因此不需要 StreamSession，只需要一个 session_id 维护对话上下文。

事件格式（SSE data 里的一行 JSON）：
  {"type": "delta",  "content": "..."}          文本增量
  {"type": "audio",  "pcm": "<base64 s16le>"}   16 kHz 单声道 PCM
  {"type": "done"}                              本轮结束
  {"type": "error",  "msg": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

import aiohttp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _credential(name: str, default: str = "") -> str:
    """环境变量优先，其次同目录的 key.py。"""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import key as key_module  # type: ignore[import-not-found]

        return str(getattr(key_module, name, default) or default)
    except Exception:  # noqa: BLE001
        return default


COZE_TOKEN = _credential("COZE_TOKEN")
COZE_BOT_ID = _credential("COZE_BOT_ID")
ALIYUN_TTS_API_KEY = _credential("ALIYUN_TTS_API_KEY")
ALIYUN_TTS_VOICE = _credential("ALIYUN_TTS_VOICE")

COZE_CHAT_URL = "https://api.coze.cn/v3/chat"
COZE_USER_ID = "liveavatar_web_user"

ALIYUN_TTS_WS_URL = os.environ.get(
    "LIVEAVATAR_TTS_WS_URL",
    "wss://llm-prl2mma08soey3qf.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
)
ALIYUN_TTS_MODEL = os.environ.get("LIVEAVATAR_TTS_MODEL", "qwen-audio-3.1-tts-flash")

SENTENCE_PUNCT = "。.？?．"


class SentenceSplitter:
    """累积流式增量文本，按句末标点切分，边切边送 TTS。"""

    def __init__(self) -> None:
        self._buf = ""

    def _protected(self, index: int) -> bool:
        """判断该位置的句点是否属于小数、缩写或省略号。"""
        if self._buf[index] not in ".．":
            return False
        nxt = self._buf[index + 1:index + 2]
        if nxt:
            if nxt.isdigit():
                return True
            if nxt in ".．":
                return True
            if "a" <= nxt <= "z" or "A" <= nxt <= "Z" or "\u0400" <= nxt <= "\u04ff":
                return True
        else:
            prev = self._buf[index - 1:index]
            if prev.isdigit() or "A" <= prev <= "Z":
                return True
        cursor = index - 1
        while cursor >= 0 and (
            "a" <= self._buf[cursor] <= "z"
            or "A" <= self._buf[cursor] <= "Z"
            or "\u0400" <= self._buf[cursor] <= "\u04ff"
        ):
            cursor -= 1
        token = self._buf[cursor + 1:index]
        return len(token) == 1 and token.isupper()

    def push(self, text: str) -> list[str]:
        self._buf += text
        segments: list[str] = []
        start = 0
        for index, char in enumerate(self._buf):
            if char in SENTENCE_PUNCT and not self._protected(index):
                segment = self._buf[start:index + 1].strip()
                if segment:
                    segments.append(segment)
                start = index + 1
        self._buf = self._buf[start:]
        return segments

    def flush(self) -> str:
        tail = self._buf.strip()
        self._buf = ""
        return tail


class AliyunTTSClient:
    """阿里云百炼实时语音合成 WS 客户端，音频以 binary 帧裸 PCM 推送。"""

    def __init__(self, http: aiohttp.ClientSession, submit, voice_id: str) -> None:
        self.http = http
        self.submit = submit
        self.voice_id = voice_id
        self.task_id = uuid.uuid4().hex
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._audio_done = asyncio.Event()
        self._error: dict | None = None
        self._reader_task: asyncio.Task | None = None

    async def _send(self, action: str, payload: dict) -> None:
        assert self.ws is not None
        message = {
            "header": {"action": action, "task_id": self.task_id, "streaming": "duplex"},
            "payload": payload,
        }
        await self.ws.send_str(json.dumps(message, ensure_ascii=False))

    async def _wait_event(self, event_type: str, timeout: float = 30) -> dict:
        assert self.ws is not None
        while True:
            message = await asyncio.wait_for(self.ws.receive(), timeout=timeout)
            if message.type == aiohttp.WSMsgType.BINARY:
                continue
            if message.type != aiohttp.WSMsgType.TEXT:
                if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise ConnectionError(f"TTS 连接已关闭：{message.type}")
                continue
            event = json.loads(message.data)
            name = event.get("header", {}).get("event")
            if name == event_type:
                return event
            if name == "task-failed":
                header = event.get("header", {})
                raise RuntimeError(
                    f"TTS 任务失败：{header.get('error_code')} {header.get('error_message')}"
                )

    async def connect(self) -> None:
        self.ws = await self.http.ws_connect(
            ALIYUN_TTS_WS_URL,
            headers={"Authorization": f"Bearer {ALIYUN_TTS_API_KEY}"},
            heartbeat=30,
            max_msg_size=16 * 1024 * 1024,
        )
        await self._send(
            "run-task",
            {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": ALIYUN_TTS_MODEL,
                "input": {},
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self.voice_id,
                    "format": "pcm",
                    "sample_rate": 16000,
                },
            },
        )
        await self._wait_event("task-started")

    def start_reader(self) -> None:
        self._reader_task = asyncio.create_task(self._reader())

    async def append(self, text: str) -> None:
        await self._send("continue-task", {"input": {"text": text}})

    async def complete(self) -> None:
        await self._send("finish-task", {"input": {}})

    async def wait_audio_done(self, timeout: float = 180) -> None:
        await asyncio.wait_for(self._audio_done.wait(), timeout=timeout)
        if self._error:
            raise RuntimeError(f"TTS 错误：{self._error}")

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        self.ws = None

    async def _reader(self) -> None:
        assert self.ws is not None
        try:
            async for message in self.ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    self.submit(message.data)
                    continue
                if message.type != aiohttp.WSMsgType.TEXT:
                    if message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
                    continue
                event = json.loads(message.data)
                header = event.get("header", {})
                name = header.get("event")
                if name == "task-finished":
                    self._audio_done.set()
                    break
                if name == "task-failed":
                    self._error = {
                        "code": header.get("error_code"),
                        "msg": header.get("error_message"),
                    }
                    self._audio_done.set()
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._error = {"msg": repr(exc)}
            self._audio_done.set()


# session_id -> {"conversation_id": str | None, "runner": ChatRunner | None}
_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()


def _session_state(session_id: str) -> dict:
    with _SESSIONS_LOCK:
        return _SESSIONS.setdefault(session_id, {"conversation_id": None, "runner": None})


def reset_conversation(session_id: str) -> None:
    """结束互动：丢弃 conversation_id，下次对话不再带历史。"""
    _session_state(session_id)["conversation_id"] = None


def get_runner(session_id: str) -> "ChatRunner | None":
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(session_id)
        return state.get("runner") if state else None


class ChatRunner:
    """一轮问答：在独立线程的事件循环里跑完整管线，事件回传给 SSE 生成器。"""

    def __init__(self, session_id: str, text: str, events: Queue) -> None:
        self.session_id = session_id
        self.text = text
        self.events = events
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None
        self.tts: AliyunTTSClient | None = None
        self._connect_task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except asyncio.CancelledError:
            # 打断是正常路径，不往外抛，避免线程打印异常堆栈
            pass

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.current_task()
        try:
            await self._pipeline()
        except asyncio.CancelledError:
            self._emit({"type": "error", "msg": "已打断"})
            raise
        except Exception as exc:  # noqa: BLE001
            self._emit({"type": "error", "msg": str(exc)})
        finally:
            await self._cleanup()
            self._emit(None)
            self._clear_state()

    def _clear_state(self) -> None:
        """本轮结束后摘掉 runner 引用，避免后续打断打到已关闭的事件循环。"""
        with _SESSIONS_LOCK:
            state = _SESSIONS.get(self.session_id)
            if state is not None and state.get("runner") is self:
                state["runner"] = None

    async def _cleanup(self) -> None:
        task = self._connect_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001
                pass
        if self.tts is not None:
            try:
                await self.tts.close()
            except BaseException:  # noqa: BLE001
                pass

    def cancel(self) -> None:
        loop, task = self.loop, self.task
        if loop is None or task is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:  # 事件循环在本轮收尾时已关闭
            pass

    def _emit(self, payload: dict | None) -> None:
        self.events.put(payload)

    async def _pipeline(self) -> None:
        session_id = self.session_id
        state = _session_state(session_id)
        tts: AliyunTTSClient | None = None
        t0 = time.monotonic()
        first_delta: float | None = None
        first_segment: float | None = None
        first_audio: float | None = None

        def submit(pcm: bytes) -> None:
            nonlocal first_audio
            if first_audio is None:
                first_audio = time.monotonic() - t0
                print(
                    f"[chat {session_id}] tts first audio {first_audio:.2f}s "
                    f"({len(pcm) / 32000.0:.2f}s pcm)",
                    flush=True,
                )
            self._emit(
                {
                    "type": "audio",
                    "pcm": base64.b64encode(pcm).decode("ascii"),
                }
            )

        async with aiohttp.ClientSession() as http:
            tts = AliyunTTSClient(http, submit, ALIYUN_TTS_VOICE)
            self.tts = tts
            self._connect_task = asyncio.create_task(tts.connect())

            payload = {
                "bot_id": COZE_BOT_ID,
                "user_id": COZE_USER_ID,
                "stream": True,
                "auto_save_history": True,
                "additional_messages": [
                    {"role": "user", "content": self.text, "content_type": "text"}
                ],
            }
            url = COZE_CHAT_URL
            if state.get("conversation_id"):
                url += f"?conversation_id={state['conversation_id']}"
            headers = {
                "Authorization": f"Bearer {COZE_TOKEN}",
                "Content-Type": "application/json",
            }

            async with http.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"Coze HTTP {response.status}：{body[:300]}")

                print(
                    f"[chat {session_id}] coze http 200 {time.monotonic() - t0:.2f}s",
                    flush=True,
                )
                await self._connect_task
                print(
                    f"[chat {session_id}] tts ready {time.monotonic() - t0:.2f}s",
                    flush=True,
                )
                tts.start_reader()

                splitter = SentenceSplitter()
                current_event: str | None = None
                failure: dict | None = None

                async for raw in response.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "conversation.chat.created":
                        conversation_id = data.get("conversation_id")
                        if conversation_id and not state.get("conversation_id"):
                            state["conversation_id"] = conversation_id
                    elif current_event == "conversation.message.delta":
                        if (
                            data.get("type") == "answer"
                            and data.get("content_type", "text") == "text"
                        ):
                            content = data.get("content", "")
                            if content:
                                if first_delta is None:
                                    first_delta = time.monotonic() - t0
                                    print(
                                        f"[chat {session_id}] coze first delta "
                                        f"{first_delta:.2f}s",
                                        flush=True,
                                    )
                                self._emit({"type": "delta", "content": content})
                                for segment in splitter.push(content):
                                    if first_segment is None:
                                        first_segment = time.monotonic() - t0
                                        print(
                                            f"[chat {session_id}] first sentence "
                                            f"{first_segment:.2f}s len={len(segment)} "
                                            f"{segment[:20]!r}",
                                            flush=True,
                                        )
                                    await tts.append(segment)
                    elif current_event == "conversation.chat.failed":
                        failure = data.get("last_error") or data
                    elif current_event == "conversation.chat.completed":
                        last_error = data.get("last_error")
                        if data.get("status") == "failed" or (
                            last_error and last_error.get("code", 0) != 0
                        ):
                            failure = last_error or data

                if failure:
                    raise RuntimeError(f"Coze 对话失败：{failure}")

                tail = splitter.flush()
                if tail:
                    await tts.append(tail)
                await tts.complete()
                await tts.wait_audio_done()
                print(
                    f"[chat {session_id}] audio done {time.monotonic() - t0:.2f}s",
                    flush=True,
                )

        self._emit({"type": "done"})


def start(session_id: str, text: str, events: Queue) -> ChatRunner:
    """启动一轮问答；同一会话尚未结束的上一轮会先被取消。"""
    runner = ChatRunner(session_id, text, events)
    state = _session_state(session_id)
    previous = state.get("runner")
    if previous is not None and previous is not runner:
        previous.cancel()
    state["runner"] = runner
    runner.start()
    return runner


def interrupt(session_id: str) -> None:
    """打断当前播报：取消管线，音频由前端清空。"""
    runner = get_runner(session_id)
    if runner is not None:
        runner.cancel()
