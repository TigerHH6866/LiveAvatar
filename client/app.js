/**
 * LiveAvatar Web 主线程：
 *   - 音频是主时钟：TTS PCM 从 SSE 收下来，按固定延迟排队播放
 *   - 视频帧由音频时钟推导，提前若干帧丢给 worker 推理
 *   - 画面 = 循环素材视频 + 模型出的 152x152 脸部画面（贴回 bbox）
 */

const FPS = 25;
const SAMPLE_RATE = 16000;
const SAMPLES_PER_FRAME = SAMPLE_RATE / FPS;   // 640
const WINDOW_SAMPLES = 12880;                  // 20.125 帧，与 C++ kStreamWindowSamples 一致
const WINDOW_AHEAD = 6480;                     // 窗口中心之后需要的采样数
const WINDOW_BEHIND = WINDOW_SAMPLES - WINDOW_AHEAD;
const AUDIO_DELAY = 0.9;                       // 首次出声前留出的推理余量
const MIN_LEAD = 0.35;                         // 音频断流后重启时的最小提前量
const PCM_SECONDS = 180;

const el = (id) => document.getElementById(id);

const state = {
  config: null,
  meta: null,
  worker: null,
  ready: false,
  provider: '',
  audioCtx: null,
  pcm: new Float32Array(SAMPLE_RATE * PCM_SECONDS),
  received: 0,
  nextPlayTime: 0,
  sources: [],
  anchor: null,       // { ctxTime, sample, frame }
  epoch: 0,
  nextM: 0,
  busy: false,
  patches: new Map(),
  lastPresent: { frame: 0, ctxTime: 0 },
  frameAbs: 0,
  prevLoopIndex: -1,
  wraps: 0,
  speaking: false,
  local: null,        // { seconds }：本地音频播放中（不走对话）
  sessionId: (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()),
  subtitle: '',
};

// ---------------------------------------------------------------- 视频与画布

/**
 * 只用 rAF 驱动：rVFC 在部分环境（隐藏/1px 视频、弱 GPU、后台节流）会停止回调，
 * 一旦停了 draw 就再也不会被调用，表现就是"动一下就彻底冻住"。
 */
function trackVideoFrames(video) {
  let lastDraw = 0;
  let lastTime = -1;
  let lastKick = 0;
  let lastResync = 0;
  const tick = () => {
    const loopFrames = state.meta.loop_frames;
    const index = ((Math.round(video.currentTime * FPS) % loopFrames) + loopFrames) % loopFrames;
    if (state.prevLoopIndex >= 0 && index < state.prevLoopIndex - loopFrames / 2) {
      state.wraps += 1;
    }
    state.prevLoopIndex = index;
    state.frameAbs = state.wraps * loopFrames + index;
    if (state.audioCtx) {
      state.lastPresent = { frame: state.frameAbs, ctxTime: state.audioCtx.currentTime };
    }
    const now = performance.now();
    // 播报中视频若暂停/卡在缓冲，画面和嘴会一起冻住；每 2 秒主动踢一次播放
    if (state.speaking && video.readyState >= 2) {
      const frozen = video.paused || Math.abs(video.currentTime - lastTime) < 1e-4;
      if (frozen && now - lastKick > 2000) {
        lastKick = now;
        video.play().catch(() => {});
      }
    }
    lastTime = video.currentTime;
    // 每秒按「画面帧 vs 音频位置」校正一次基准，防止视频掉帧导致的偏移累积
    if (now - lastResync > 1000) {
      lastResync = now;
      resyncAnchor();
    }
    // 排队的声音都播完了就收尾，否则最后一张贴片会一直定在脸上等下一句
    if (state.speaking && !state.local && state.audioCtx && state.sources.length === 0 &&
        state.audioCtx.currentTime > state.nextPlayTime + 0.1) {
      endSpeech();
    }
    if (now - lastDraw >= 30) {   // 约 30 fps，够贴 25 fps 素材
      lastDraw = now;
      tickLocal();
      draw();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const scratch = document.createElement('canvas');
scratch.width = state.meta ? state.meta.crop : 152;
scratch.height = state.meta ? state.meta.crop : 152;
const scratchCtx = scratch.getContext('2d');

/** 贴片边缘羽化蒙版：外圈 6 像素 alpha 从 0 过渡到 255，避免硬边。 */
const edgeMask = (() => {
  const size = scratch.width;
  const feather = 6;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, size, size);
  ctx.globalCompositeOperation = 'destination-out';
  const ramp = (x0, y0, x1, y1) => {
    const gradient = ctx.createLinearGradient(x0, y0, x1, y1);
    gradient.addColorStop(0, 'rgba(0,0,0,1)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    return gradient;
  };
  ctx.fillStyle = ramp(0, 0, feather, 0);
  ctx.fillRect(0, 0, feather, size);
  ctx.fillStyle = ramp(size, 0, size - feather, 0);
  ctx.fillRect(size - feather, 0, feather, size);
  ctx.fillStyle = ramp(0, 0, 0, feather);
  ctx.fillRect(0, 0, size, feather);
  ctx.fillStyle = ramp(0, size, 0, size - feather);
  ctx.fillRect(0, size - feather, size, feather);
  return canvas;
})();

// 视频时钟与音频时钟会差几十毫秒，推理也可能略慢于实时；只认完全相等的帧号会让
// 贴片永远命中不到（表现就是"有声无口型"），所以退一档取最近的贴片。
const PATCH_TOLERANCE = 8;   // 允许落后 8 帧（约 320 ms）
const PATCH_FUTURE = 2;      // 视频钟略慢时，允许用未来 2 帧
const PATCH_HOLD = 80;       // 贴片保留 80 帧（约 3.2 s），弱机型落后时还有得用

function pickPatch(frame) {
  // 音频还没开播（AUDIO_DELAY 的预滚）就不贴，否则嘴会比声音早动将近 1 秒
  if (state.speaking && state.anchor && state.audioCtx &&
      state.audioCtx.currentTime < state.anchor.ctxTime) {
    return null;
  }
  const exact = state.patches.get(frame);
  if (exact) return exact;
  let past = null;
  let future = null;
  for (const [key, value] of state.patches) {
    if (key < frame) {
      if (!past || key > past.key) past = { key, value };
    } else if (!future || key < future.key) {
      future = { key, value };
    }
  }
  if (past && frame - past.key <= PATCH_TOLERANCE) return past.value;
  if (future && future.key - frame <= PATCH_FUTURE) return future.value;
  // 播报中：贴片落后（推理慢）或跑在前面（视频慢/卡）都退一档，
  // 否则两种情况下都会出现"贴片存在但一张也用不上"的冻住现象
  if (state.speaking) {
    if (past && frame - past.key <= PATCH_HOLD) return past.value;
    if (future && future.key - frame <= PATCH_HOLD) return future.value;
  }
  return null;
}

/**
 * 贴片按「音频位置 → 帧号」排定，画面帧号来自视频时钟。两者一旦差出几帧，
 * 偏移会一直累积（表现为越说越错位）。每秒实测一次偏差，把基准和已算好的
 * 贴片帧号整体平移，既不丢结果也不用重新推理。
 */
function resyncAnchor() {
  const ctx = state.audioCtx;
  const anchor = state.anchor;
  if (!ctx || !anchor || !state.speaking) return;
  const mNow = Math.round((ctx.currentTime - anchor.ctxTime) * FPS);
  if (mNow < 0) return;
  const delta = state.frameAbs - (anchor.frame + mNow);
  if (delta === 0 || Math.abs(delta) > 40) return;
  anchor.frame += delta;
  const shifted = new Map();
  for (const [key, value] of state.patches) shifted.set(key + delta, value);
  state.patches = shifted;
}

function draw() {
  const canvas = el('stage');
  const ctx = canvas.getContext('2d');
  const video = el('source');
  if (video.readyState >= 2) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  const patch = pickPatch(state.frameAbs);
  if (!patch) return;
  // 贴回位置必须跟「屏幕当前帧」，而不是贴片所属的旧帧：两者最多差 5 px，
  // 用旧帧的框贴会把这块脸整体平移/缩放，看起来就是嘴在轻微游走和呼吸。
  const loopFrames = state.meta.loop_frames;
  const loopIndex = ((state.frameAbs % loopFrames) + loopFrames) % loopFrames;
  const bbox = state.meta.bbox[state.meta.loop_order[loopIndex]];
  const crop = state.meta.crop;
  const image = new ImageData(crop, crop);
  const data = patch.data;
  for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
    image.data[j] = data[i];
    image.data[j + 1] = data[i + 1];
    image.data[j + 2] = data[i + 2];
    image.data[j + 3] = 255;
  }
  scratchCtx.putImageData(image, 0, 0);
  // 边缘羽化：整块硬贴会在下巴/脖子处露出一圈矩形边
  scratchCtx.globalCompositeOperation = 'destination-in';
  scratchCtx.drawImage(edgeMask, 0, 0);
  scratchCtx.globalCompositeOperation = 'source-over';
  ctx.drawImage(scratch, bbox[0], bbox[1], bbox[2], bbox[3]);
}

// ---------------------------------------------------------------- 音频与推理

function frameAtCtxTime(ctxTime) {
  const last = state.lastPresent;
  return last.frame + Math.round((ctxTime - last.ctxTime) * FPS);
}

function pushPcm(int16) {
  const target = state.received;
  if (target + int16.length > state.pcm.length) {
    console.warn('[audio] 缓冲已满，丢弃旧数据');
    return;
  }
  const pcm = state.pcm;
  for (let i = 0; i < int16.length; i += 1) {
    pcm[target + i] = int16[i] / 32768;
  }
  state.received += int16.length;
}

function schedulePcm(int16) {
  if (!state.audioCtx || int16.length === 0) return;
  const chunkStartSample = state.received;
  pushPcm(int16);

  const ctx = state.audioCtx;
  let startAt = Math.max(ctx.currentTime + MIN_LEAD, state.nextPlayTime);
  if (state.nextPlayTime > 0 && startAt > state.nextPlayTime + 0.02) {
    // 音频断流：时间轴重锚，之前按旧时间算出的帧全部作废
    state.epoch += 1;
    state.patches.clear();
    state.nextM = 0;
    state.anchor = null;
  }
  if (!state.anchor) {
    state.anchor = {
      ctxTime: startAt,
      sample: chunkStartSample,
      frame: frameAtCtxTime(startAt),
    };
  }
  const buffer = ctx.createBuffer(1, int16.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < int16.length; i += 1) channel[i] = int16[i] / 32768;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(startAt);
  source.onended = () => {
    state.sources = state.sources.filter((item) => item !== source);
  };
  state.sources.push(source);
  state.nextPlayTime = startAt + int16.length / SAMPLE_RATE;
  state.speaking = true;
  pump();
}

/** 取 window_start..window_end 的采样并做归一化（与 C++ 用最近 2 秒统计量一致）。 */
function extractWindow(startSample, endSample) {
  const out = new Float32Array(WINDOW_SAMPLES);
  const from = Math.max(0, startSample);
  const to = Math.min(state.received, endSample);
  if (to > from) {
    out.set(state.pcm.subarray(from, to), from - startSample);
  }
  const statFrom = Math.max(0, endSample - 2 * SAMPLE_RATE);
  const statTo = Math.min(state.received, endSample);
  let mean = 0;
  for (let i = statFrom; i < statTo; i += 1) mean += state.pcm[i];
  mean = statTo > statFrom ? mean / (statTo - statFrom) : 0;
  let variance = 0;
  for (let i = statFrom; i < statTo; i += 1) {
    const d = state.pcm[i] - mean;
    variance += d * d;
  }
  variance = statTo > statFrom ? variance / (statTo - statFrom) : 0;
  const scale = 1 / Math.sqrt(variance + 1e-8);
  for (let i = 0; i < out.length; i += 1) out[i] = (out[i] - mean) * scale;
  return out;
}

function pump() {
  if (!state.ready || !state.anchor) return;
  if (state.busy) {
    // worker 如果因为某帧异常卡住，3 秒后强行放行，否则整段播报会一直冻住
    if (performance.now() - state.lastSentAt < 3000) return;
    state.busy = false;
  }
  // 推理落后太多（弱机型 / 慢 EP）就直接跳帧，否则口型会越拖越远、最后彻底对不上
  if (state.speaking) {
    const target = state.frameAbs - 4 - state.anchor.frame;
    if (target > state.nextM) state.nextM = target;
  }
  const m = state.nextM;
  const need = m * SAMPLES_PER_FRAME + WINDOW_AHEAD;
  const absoluteEnd = state.anchor.sample + need;
  if (absoluteEnd > state.received) return;
  const frame = state.anchor.frame + m;
  // 本地文件整段音频都在手，不限速会把全片一口气算完；只提前约 2 秒
  if (state.local && frame > state.frameAbs + LOCAL_LEAD_FRAMES) return;
  const window = extractWindow(absoluteEnd - WINDOW_SAMPLES, absoluteEnd);
  if (m % 10 === 0) {
    let energy = 0;
    for (let i = 0; i < window.length; i += 1) energy += window[i] * window[i];
    console.log(`[pump] m=${m} rms=${Math.sqrt(energy / window.length).toFixed(3)} received=${state.received} anchor=${JSON.stringify(state.anchor)}`);
  }
  state.busy = true;
  state.nextM = m + 1;
  state.lastSentAt = performance.now();
  state.worker.postMessage(
    { type: 'frame', frame, epoch: state.epoch, audio: window },
    [window.buffer],
  );
}

function onWorkerFrame(message) {
  state.busy = false;
  if (message.epoch === state.epoch) {
    state.patches.set(message.frame, {
      data: new Uint8Array(message.data),
      assetIndex: message.assetIndex,
    });
    const cutoff = state.frameAbs - PATCH_HOLD;
    for (const key of state.patches.keys()) {
      if (key < cutoff) state.patches.delete(key);
    }
    setStatus(`推理 ${message.ms.toFixed(1)} ms/帧 · ${state.provider}`, 'ok');
  }
  pump();
}

function resetPlayback() {
  for (const source of state.sources) {
    try {
      source.stop();
    } catch (error) {
      /* 已结束 */
    }
  }
  state.sources = [];
  state.epoch += 1;
  state.anchor = null;
  state.nextM = 0;
  state.busy = false;
  state.received = 0;
  state.nextPlayTime = 0;
  state.patches.clear();
  state.speaking = false;
  state.local = null;
  el('send').disabled = false;
}

/** 一段播报说完：撤掉贴片回到素材原生画面，否则最后一张会一直定在脸上。 */
function endSpeech(label) {
  state.speaking = false;
  state.patches.clear();
  state.anchor = null;
  state.nextM = 0;
  setStatus(label || `已就绪，可以开始对话 · ${state.provider}`, 'ok');
}

// ---------------------------------------------------------------- 本地音频（离线驱动）

const LOCAL_LEAD_FRAMES = 50;   // 只提前渲染约 2 秒，避免整段音频一次算完

/** 任意浏览器能解码的音频 → 16 kHz 单声道 Int16。 */
async function decodeToPcm16k(file) {
  const bytes = await file.arrayBuffer();
  const decoded = await state.audioCtx.decodeAudioData(bytes);
  const frames = Math.max(1, Math.round(decoded.duration * SAMPLE_RATE));
  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;             // 多声道连到单声道目标会自动下混
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  const channel = rendered.getChannelData(0);
  const pcm = new Int16Array(channel.length);
  for (let i = 0; i < channel.length; i += 1) {
    const v = Math.max(-1, Math.min(1, channel[i]));
    pcm[i] = Math.round(v * 32767);
  }
  return pcm;
}

/** 音频比预分配的 180 秒缓冲更长时扩容。 */
function ensurePcmCapacity(samples) {
  if (samples <= state.pcm.length) return;
  const grown = new Float32Array(Math.ceil((samples + SAMPLE_RATE) / SAMPLE_RATE) * SAMPLE_RATE);
  grown.set(state.pcm.subarray(0, state.received));
  state.pcm = grown;
}

/** 用本地音频文件驱动：不进 chat，浏览器自己解码、排播、逐帧推理。 */
async function playLocalAudio(file) {
  if (!state.ready) {
    setStatus('模型还没就绪');
    return;
  }
  setStatus('解码音频…');
  let pcm;
  try {
    pcm = await decodeToPcm16k(file);
  } catch (error) {
    setStatus('无法解码这个文件：' + (error.message || error), 'error');
    return;
  }
  if (pcm.length < SAMPLE_RATE / 4) {
    setStatus('音频太短', 'error');
    return;
  }

  resetPlayback();
  setSubtitle('');
  state.subtitle = '';
  el('send').disabled = true;

  ensurePcmCapacity(pcm.length);
  pushPcm(pcm);

  const ctx = state.audioCtx;
  if (ctx.state === 'suspended') await ctx.resume();
  const startAt = ctx.currentTime + AUDIO_DELAY;
  const seconds = pcm.length / SAMPLE_RATE;
  state.anchor = { ctxTime: startAt, sample: 0, frame: frameAtCtxTime(startAt) };

  const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 32768;
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start(startAt);
  source.onended = () => {
    state.sources = state.sources.filter((item) => item !== source);
  };
  state.sources.push(source);
  state.nextPlayTime = startAt + seconds;
  state.speaking = true;
  state.local = { seconds };
  setStatus(`离线播放 ${seconds.toFixed(1)} s`);
  pump();
}

/** 每个视频帧回调推进本地播放：补推理、判结束。 */
function tickLocal() {
  if (!state.local || !state.anchor) return;
  pump();
  const ctx = state.audioCtx;
  if (ctx && ctx.currentTime > state.anchor.ctxTime + state.local.seconds + 0.2) {
    state.local = null;
    el('send').disabled = false;
    endSpeech('本地音频播放结束');
  }
}

// ---------------------------------------------------------------- 网络

async function loadConfig() {
  const response = await fetch('/api/config');
  if (!response.ok) throw new Error(await response.text());
  state.config = await response.json();
  state.meta = state.config.meta;
}

function startWorker(prefer, models) {
  return new Promise((resolve, reject) => {
    if (state.worker) {
      state.worker.terminate();
      state.worker = null;
    }
    const worker = new Worker('./worker.js', { type: 'module' });
    worker.onerror = (event) => reject(new Error(event.message || 'worker 启动失败'));
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === 'warmup') {
        setLoading(`预热模型 ${message.done}/${message.total}…`, 0.92 + 0.06 * (message.done / message.total));
      } else if (message.type === 'ready') {
        state.provider = message.provider;
        state.ready = true;
        setStatus(`推理后端：${message.provider}`, 'ok');
        resolve();
      } else if (message.type === 'frame') {
        onWorkerFrame(message);
      } else if (message.type === 'error') {
        console.error('[worker]', message.message);
        if (message.retryWasm && prefer !== 'wasm') {
          setStatus('WebGPU 初始化失败，改用 wasm…');
          startWorker('wasm', models).then(resolve, reject);
          return;
        }
        const reason = message.message.split('\n')[0];
        if (state.ready) {
          // 已就绪后单帧失败：跳过这一帧继续推，别让 busy 永久卡死（嘴会突然冻住）
          setStatus(`推理异常，已跳过该帧：${reason}`, 'error');
          state.busy = false;
          pump();
          return;
        }
        setStatus('推理出错：' + reason, 'error');
        reject(new Error(message.message));
      }
    };
    worker.postMessage({
      type: 'init',
      provider: prefer || 'auto',
      wasmPath: state.config.ort.wasm,
      featherUrl: state.config.models.feather,
      visualUrl: state.config.models.visual,
      // 已经预下载过就直接把字节给它，worker 不再重复拉一遍
      featherData: models && models.feather,
      visualData: models && models.visual,
      facesBase: state.config.media.faces,
      loopOrder: state.meta.loop_order,
    });
    state.worker = worker;
  });
}

async function speak(text) {
  if (!text.trim()) return;
  resetPlayback();
  setSubtitle('');
  state.subtitle = '';
  el('send').disabled = true;
  setStatus('思考中…');
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, sessionid: state.sessionId }),
    });
    if (!response.ok || !response.body) throw new Error('对话请求失败');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split;
      while ((split = buffer.indexOf('\n\n')) >= 0) {
        const block = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        for (const line of block.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = JSON.parse(line.slice(5).trim());
          if (payload.type === 'delta') {
            appendSubtitle(payload.content);
          } else if (payload.type === 'audio') {
            schedulePcm(decodeBase64Pcm(payload.pcm));
          } else if (payload.type === 'done') {
            finished = true;
          } else if (payload.type === 'error') {
            setStatus(payload.msg, 'error');
            finished = true;
          }
        }
      }
    }
  } catch (error) {
    setStatus(String(error), 'error');
  } finally {
    el('send').disabled = false;
  }
}

function decodeBase64Pcm(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer, 0, bytes.length >> 1);
}

// ---------------------------------------------------------------- 录音与识别

let recorder = null;
let recordTimer = null;
let audioStream = null;

async function toggleRecording() {
  if (recorder) {
    recorder.stop();
    return;
  }
  // 用户开口即打断当前播报，和真人对谈一致。
  if (state.speaking) interrupt();
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (error) {
    setStatus('无法打开麦克风：' + error, 'error');
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };
  recorder.onstop = async () => {
    recorder = null;
    el('mic').classList.remove('recording');
    clearTimeout(recordTimer);
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
    const blob = new Blob(chunks, { type: 'audio/webm' });
    if (blob.size < 2000) {
      setStatus('没听清，再说一次？');
      return;
    }
    setStatus('识别中…');
    try {
      const form = new FormData();
      form.append('audio', blob, 'speech.webm');
      const response = await fetch('/api/asr', { method: 'POST', body: form });
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      if (data.empty || !data.text) {
        setStatus('没听清，再说一次？');
        return;
      }
      setStatus('你说：' + data.text);
      speak(data.text);
    } catch (error) {
      setStatus('识别失败：' + error.message, 'error');
    }
  };
  recorder.start();
  el('mic').classList.add('recording');
  setStatus('正在听…');
  recordTimer = setTimeout(() => recorder && recorder.stop(), 20000);
}

async function interrupt() {
  resetPlayback();
  try {
    await fetch('/api/interrupt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionid: state.sessionId }),
    });
  } catch (error) {
    /* 忽略 */
  }
  setStatus('已打断');
}

// ---------------------------------------------------------------- 界面

function setStatus(text, kind) {
  const node = el('status');
  node.textContent = text;
  node.className = kind || '';
}

function setSubtitle(text) {
  el('subtitle').textContent = text;
}

/** 加载遮罩：ratio 为 0..1，不传只更新文字。 */
function setLoading(text, ratio) {
  const label = el('loading-text');
  // 静态页是旧版（没有遮罩）时退到状态栏，别把真正的启动错误盖掉
  if (!label) {
    setStatus(text);
    return;
  }
  label.textContent = text;
  const fill = el('loading-fill');
  if (fill && typeof ratio === 'number') {
    const value = Math.max(0, Math.min(1, ratio));
    fill.style.width = `${Math.round(value * 100)}%`;
  }
}

function finishLoading() {
  const overlay = el('loading');
  if (!overlay) return;
  overlay.classList.add('done');
  setTimeout(() => overlay.remove(), 400);
}

function failLoading(text) {
  const overlay = el('loading');
  if (overlay) overlay.classList.add('error');
  setLoading(text, 0);
}

/** 兜底等视频能出画面（不阻塞太久）。 */
function waitVideoReady(video) {
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('canplay', done);
      resolve();
    };
    video.addEventListener('loadeddata', done);
    video.addEventListener('canplay', done);
    setTimeout(done, 8000);
  });
}

/** 带进度下载两个 onnx，返回 Uint8Array；会话由 worker 用自己的缓冲区创建。 */
async function loadModels() {
  const keys = ['feather', 'visual'];
  const out = {};
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i];
    const url = state.config.models[key];
    const response = await fetch(url);
    if (!response.ok) throw new Error(`模型下载失败（${response.status}）：${url}`);
    const total = Number(response.headers.get('content-length') || 0);
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      const ratio = total ? received / total : 0;
      const span = (i + ratio) / keys.length;
      setLoading(
        `加载模型 ${i + 1}/${keys.length} · ${(received / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`,
        0.08 + span * 0.84,
      );
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    out[key] = bytes;
  }
  return out;
}

function appendSubtitle(text) {
  state.subtitle = (state.subtitle + text).slice(-160);
  setSubtitle(state.subtitle);
}

async function boot() {
  const controls = ['mic', 'text', 'send', 'file', 'stop'];
  controls.forEach((id) => { el(id).disabled = true; });
  try {
    setLoading('加载配置…', 0.02);
    await loadConfig();
    const meta = state.meta;
    const canvas = el('stage');
    canvas.width = meta.width;
    canvas.height = meta.height;
    Object.assign(canvas.style, {
      aspectRatio: `${meta.width} / ${meta.height}`,
    });

    const video = el('source');
    video.src = state.config.media.loop;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    trackVideoFrames(video);
    setLoading('加载素材视频…', 0.05);
    await waitVideoReady(video);
    await video.play().catch(() => {
      setStatus('点击画面开始');
      document.body.addEventListener('click', () => video.play(), { once: true });
    });

    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    state.lastPresent = { frame: 0, ctxTime: state.audioCtx.currentTime };

    setLoading('加载模型…', 0.08);
    const models = await loadModels();
    setLoading('初始化推理引擎…', 0.92);
    setStatus('初始化推理引擎…');
    await startWorker(new URLSearchParams(location.search).get('ep') || 'auto', models);

    el('send').addEventListener('click', () => {
      const input = el('text');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      speak(text);
    });
    el('text').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.isComposing) el('send').click();
    });
    el('mic').addEventListener('click', toggleRecording);
    el('file').addEventListener('click', () => el('audiofile').click());
    el('audiofile').addEventListener('change', async (event) => {
      const file = event.target.files && event.target.files[0];
      event.target.value = '';        // 允许连续选同一个文件
      if (file) await playLocalAudio(file);
    });
    el('stop').addEventListener('click', () => {
      if (state.local) {
        resetPlayback();
        setStatus('已停止');
        return;
      }
      interrupt();
    });
    document.body.addEventListener(
      'pointerdown',
      () => state.audioCtx && state.audioCtx.state === 'suspended' && state.audioCtx.resume(),
      { once: true },
    );

    setLoading('就绪', 1);
    controls.forEach((id) => { el(id).disabled = false; });
    setStatus(`已就绪，可以开始对话 · ${state.provider}`, 'ok');
    finishLoading();
  } catch (error) {
    setStatus('启动失败：' + error.message, 'error');
    failLoading('启动失败：' + error.message);
  }
}

boot();
