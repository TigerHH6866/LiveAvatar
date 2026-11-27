/**
 * 数字人播放核心（iframe / SDK 版）。
 *
 * 和网页版（app.js）共用同一套资产与推理：worker.js、gl_compose.js、模型、素材。
 * 这里只负责「画面 + 音频驱动口型」，不包含对话、ASR、录音、气泡等交互逻辑，
 * 那些由宿主网站自己处理，通过 embed.js 的 postMessage 协议调用。
 *
 * 对外语义：
 *   - speakPcm / speakAudio 默认「追加」，前一段没播完就无缝接在后面；
 *   - interrupt() 才打断；
 *   - 一段音频播完（队列排空）触发 end 事件。
 */

import { createCompositor } from './gl_compose.js';

const FPS = 25;
const SAMPLE_RATE = 16000;
const SAMPLES_PER_FRAME = SAMPLE_RATE / FPS;   // 640
const WINDOW_SAMPLES = 12880;                  // 与 worker 的窗口一致
const WINDOW_AHEAD = 6480;
const MIN_LEAD = 0.35;                         // 断流后重新起播的最小提前量
const FIRST_DELAY = 0.5;                       // 首次出声前留出的推理余量
const LEAD_FRAMES = 50;                        // 推理最多领先画面约 2 秒
const PATCH_TOLERANCE = 8;
const PATCH_FUTURE = 2;
const PATCH_HOLD = 80;
const MASK_DRIFT = 0.15;                       // 蒙版与正片允许的时间差（秒）
const PCM_SECONDS = 300;                       // PCM 环形缓冲初始容量

function el(id) {
  return document.getElementById(id);
}

function waitVideoReady(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 2) {
      resolve();
      return;
    }
    const done = () => {
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('error', fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('error', fail);
      reject(new Error(`视频加载失败：${video.src}`));
    };
    video.addEventListener('loadeddata', done);
    video.addEventListener('error', fail);
  });
}

export class LiveAvatarPlayer {
  /**
   * @param {object} options
   * @param {HTMLCanvasElement} options.canvas 显示画布
   * @param {HTMLVideoElement} options.source  正片循环视频（隐藏）
   * @param {HTMLVideoElement} options.mask    抠像蒙版视频（隐藏）
   * @param {HTMLElement} options.bgLayer      背景层（透明时留空）
   * @param {string} [options.configUrl]       /api/config 地址
   * @param {(type: string, payload: any) => void} [options.onEvent] 事件回调
   */
  constructor(options) {
    this.canvas = options.canvas;
    this.video = options.source;
    this.maskVideo = options.mask;
    this.bgLayer = options.bgLayer || null;
    this.configUrl = options.configUrl || '/api/config';
    this.onEvent = options.onEvent || (() => {});
    this.state = this.createState();
    this.bgUrl = null;
    this.destroyed = false;
  }

  createState() {
    return {
      config: null,
      meta: null,
      worker: null,
      ready: false,
      provider: '',
      compositor: null,
      audioCtx: null,
      pcm: new Float32Array(SAMPLE_RATE * PCM_SECONDS),
      received: 0,
      nextPlayTime: 0,
      sources: [],
      anchor: null,
      epoch: 0,
      nextM: 0,
      busy: false,
      lastSentAt: 0,
      patches: new Map(),
      lastPresent: { frame: 0, ctxTime: 0 },
      frameAbs: 0,
      prevLoopIndex: -1,
      wraps: 0,
      speaking: false,
      scale: 1,
    };
  }

  // ------------------------------------------------------------ 生命周期

  /** 加载配置、素材、模型并预热推理，完成后触发 ready。 */
  async load() {
    const state = this.state;
    this.emit('progress', { phase: 'config', ratio: 0.02 });
    const response = await fetch(this.configUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`配置加载失败：${response.status}`);
    state.config = await response.json();
    state.meta = state.config.meta;

    const meta = state.meta;
    this.canvas.width = meta.width;
    this.canvas.height = meta.height;
    this.setSize(this.cssWidth(), this.cssHeight());

    this.video.src = state.config.media.loop;
    this.video.loop = true;
    this.video.muted = true;
    this.video.playsInline = true;
    this.emit('progress', { phase: 'video', ratio: 0.04 });
    await waitVideoReady(this.video);

    if (meta.mask && state.config.media.mask) {
      try {
        state.compositor = createCompositor(meta.width, meta.height, {
          low: meta.mask_low ?? 0.15,
          high: meta.mask_high ?? 0.85,
        });
      } catch (error) {
        console.warn('[matte] 合成器不可用，回退不透明画面：', error);
      }
    }
    if (state.compositor) {
      this.maskVideo.src = state.config.media.mask;
      this.maskVideo.loop = true;
      this.maskVideo.muted = true;
      this.maskVideo.playsInline = true;
      this.maskVideo.addEventListener('error', () => {
        console.warn('[matte] 蒙版加载失败，回退不透明画面');
        state.compositor = null;
      }, { once: true });
      this.emit('progress', { phase: 'mask', ratio: 0.06 });
      await waitVideoReady(this.maskVideo);
      await this.maskVideo.play().catch(() => {});
    }

    await this.video.play().catch(() => {});
    this.startTicking();

    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    state.lastPresent = { frame: 0, ctxTime: state.audioCtx.currentTime };

    const models = await this.loadModels();
    this.emit('progress', { phase: 'engine', ratio: 0.92 });
    await this.startWorker('auto', models);
    this.emit('ready', { provider: state.provider });
  }

  destroy() {
    this.destroyed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    if (this.observer) this.observer.disconnect();
    const state = this.state;
    if (state.worker) state.worker.terminate();
    for (const source of state.sources) {
      try { source.stop(); } catch (error) { /* 已结束 */ }
    }
    if (state.audioCtx) state.audioCtx.close().catch(() => {});
    if (this.bgUrl) URL.revokeObjectURL(this.bgUrl);
    this.video.pause();
    this.maskVideo.pause();
  }

  emit(type, payload) {
    if (this.destroyed) return;
    try {
      this.onEvent(type, payload || {});
    } catch (error) {
      console.warn('[player] 事件处理异常', error);
    }
  }

  // ------------------------------------------------------------ 尺寸与背景

  cssWidth() {
    return this.canvas.clientWidth || this.canvas.parentElement?.clientWidth || this.state.meta.width;
  }

  cssHeight() {
    return this.canvas.clientHeight || this.canvas.parentElement?.clientHeight || this.state.meta.height;
  }

  /**
   * 按显示尺寸调整画布分辨率：小窗不必按 1080x1920 满算，
   * 上限仍是素材原始尺寸，保证放大时不糊。
   * 显示尺寸由宿主的 iframe 决定，这里只改画布像素尺寸（保持比例由 CSS 负责）。
   */
  setSize(width, height) {
    const meta = this.state.meta;
    if (!meta) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const scale = Math.max(0.2, Math.min(1, ((width || meta.width) * dpr) / meta.width));
    this.state.scale = scale;
    const w = Math.max(2, Math.round(meta.width * scale));
    const h = Math.max(2, Math.round(meta.height * scale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /** 背景：透明（默认）/ 纯色 / 图片（二进制，避免 COEP 拦截跨源 CSS 背景）。 */
  setBackground(spec = {}) {
    if (!this.bgLayer) return;
    const { kind = 'none', value = '', image = null } = spec;
    if (this.bgUrl) {
      URL.revokeObjectURL(this.bgUrl);
      this.bgUrl = null;
    }
    this.bgLayer.style.backgroundImage = '';
    this.bgLayer.style.backgroundColor = 'transparent';
    if (kind === 'color' && value) {
      this.bgLayer.style.backgroundColor = value;
    } else if (kind === 'image' && image) {
      if (typeof image === 'string') {
        this.bgLayer.style.backgroundImage = `url("${image}")`;
      } else {
        const blob = image instanceof Blob ? image : new Blob([image], { type: spec.mime || 'image/png' });
        this.bgUrl = URL.createObjectURL(blob);
        this.bgLayer.style.backgroundImage = `url("${this.bgUrl}")`;
      }
    }
  }

  // ------------------------------------------------------------ 音频接口

  ensureAudio() {
    const state = this.state;
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      state.lastPresent = { frame: 0, ctxTime: state.audioCtx.currentTime };
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume().catch(() => {});
    }
    return state.audioCtx;
  }

  /**
   * 流式 PCM：int16、约定 16 kHz 单声道（其他采样率会内部重采样）。
   * 默认追加到当前播报末尾，replace 为 true 时先打断。
   */
  speakPcm(input, options = {}) {
    if (this.destroyed) return;
    const { replace = false, sampleRate = SAMPLE_RATE } = options;
    let pcm = input instanceof Int16Array ? input : new Int16Array(input);
    if (!pcm.length) return;
    if (replace) this.interrupt();
    this.ensureAudio();
    if (sampleRate !== SAMPLE_RATE) pcm = resampleInt16(pcm, sampleRate, SAMPLE_RATE);
    this.schedulePcm(pcm);
  }

  /** 整段音频（wav/mp3 等浏览器能解码的格式）→ 解码成 16 kHz 再走同一套调度。 */
  async speakAudio(arrayBuffer, options = {}) {
    if (this.destroyed) return;
    const { replace = false } = options;
    const ctx = this.ensureAudio();
    if (replace) this.interrupt();
    const bytes = arrayBuffer instanceof ArrayBuffer ? arrayBuffer.slice(0) : arrayBuffer;
    const decoded = await ctx.decodeAudioData(bytes);
    const frames = Math.max(1, Math.round(decoded.duration * SAMPLE_RATE));
    const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    const channel = rendered.getChannelData(0);
    const pcm = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i += 1) {
      const v = Math.max(-1, Math.min(1, channel[i]));
      pcm[i] = Math.round(v * 32767);
    }
    this.schedulePcm(pcm);
  }

  /** 立即停止当前播报（带 80 ms 淡出，避免爆音）。 */
  interrupt() {
    const state = this.state;
    const ctx = state.audioCtx;
    if (ctx) {
      const now = ctx.currentTime;
      for (const entry of state.sources) {
        try {
          const gain = entry.gain;
          gain.gain.cancelScheduledValues(now);
          gain.gain.setValueAtTime(gain.gain.value, now);
          gain.gain.linearRampToValueAtTime(0, now + 0.08);
          entry.source.stop(now + 0.09);
        } catch (error) {
          /* 已结束 */
        }
      }
    }
    this.resetPlayback();
    if (state.speaking) this.emit('end', { reason: 'interrupt' });
  }

  // ------------------------------------------------------------ 音频调度

  pushPcm(int16) {
    const state = this.state;
    const need = state.received + int16.length;
    if (need > state.pcm.length) {
      // 单次播报很长时按 1.5 倍扩容，避免丢弃数据让口型断掉
      let size = state.pcm.length;
      while (size < need) size = Math.ceil(size * 1.5);
      const grown = new Float32Array(size);
      grown.set(state.pcm.subarray(0, state.received));
      state.pcm = grown;
    }
    const pcm = state.pcm;
    for (let i = 0; i < int16.length; i += 1) {
      pcm[state.received + i] = int16[i] / 32768;
    }
    state.received += int16.length;
  }

  schedulePcm(int16) {
    const state = this.state;
    const ctx = state.audioCtx;
    if (!ctx || !int16.length) return;
    const chunkStart = state.received;
    this.pushPcm(int16);

    const freshUtterance = !state.speaking;
    let startAt = Math.max(ctx.currentTime + (freshUtterance ? FIRST_DELAY : MIN_LEAD), state.nextPlayTime);
    if (!freshUtterance && state.nextPlayTime > 0 && startAt > state.nextPlayTime + 0.02) {
      // 上一段已经播空、这一段迟到：保留贴片历史，只重建时间基准（软续播）
      state.anchor = null;
      state.nextM = 0;
      state.epoch += 1;
      startAt = ctx.currentTime + MIN_LEAD;
    }
    if (!state.anchor) {
      state.anchor = {
        ctxTime: startAt,
        sample: chunkStart,
        frame: this.frameAtCtxTime(startAt),
      };
    }
    const buffer = ctx.createBuffer(1, int16.length, SAMPLE_RATE);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < int16.length; i += 1) channel[i] = int16[i] / 32768;
    const source = ctx.createBufferSource();
    const gain = ctx.createGain();
    source.buffer = buffer;
    source.connect(gain);
    gain.connect(ctx.destination);
    source.start(startAt);
    const entry = { source, gain };
    source.onended = () => {
      state.sources = state.sources.filter((item) => item !== entry);
    };
    state.sources.push(entry);
    state.nextPlayTime = startAt + int16.length / SAMPLE_RATE;
    if (freshUtterance) {
      state.speaking = true;
      this.emit('start', {});
    }
    this.pump();
  }

  frameAtCtxTime(ctxTime) {
    const last = this.state.lastPresent;
    return last.frame + Math.round((ctxTime - last.ctxTime) * FPS);
  }

  extractWindow(startSample, endSample) {
    const state = this.state;
    const out = new Float32Array(WINDOW_SAMPLES);
    const from = Math.max(0, startSample);
    const to = Math.min(state.received, endSample);
    if (to > from) out.set(state.pcm.subarray(from, to), from - startSample);
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

  pump() {
    const state = this.state;
    if (!state.ready || !state.anchor) return;
    if (state.busy) {
      if (performance.now() - state.lastSentAt < 3000) return;
      state.busy = false;
    }
    if (state.speaking) {
      const target = state.frameAbs - 4 - state.anchor.frame;
      if (target > state.nextM) state.nextM = target;
    }
    const m = state.nextM;
    const absoluteEnd = state.anchor.sample + m * SAMPLES_PER_FRAME + WINDOW_AHEAD;
    if (absoluteEnd > state.received) return;
    const frame = state.anchor.frame + m;
    // 只提前约 2 秒，避免整段音频一次算完把主线程/GPU 占满
    if (frame > state.frameAbs + LEAD_FRAMES) return;
    const window = this.extractWindow(absoluteEnd - WINDOW_SAMPLES, absoluteEnd);
    state.busy = true;
    state.nextM = m + 1;
    state.lastSentAt = performance.now();
    state.worker.postMessage({ type: 'frame', frame, epoch: state.epoch, audio: window }, [window.buffer]);
  }

  resetPlayback() {
    const state = this.state;
    for (const entry of state.sources) {
      try { entry.source.stop(); } catch (error) { /* 已结束 */ }
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
  }

  finishUtterance(reason) {
    const state = this.state;
    state.speaking = false;
    state.patches.clear();
    state.anchor = null;
    state.nextM = 0;
    this.emit('end', { reason: reason || 'drained' });
  }

  // ------------------------------------------------------------ 渲染

  startTicking() {
    let lastDraw = 0;
    let lastKick = 0;
    let lastResync = 0;
    const tick = () => {
      if (this.destroyed) return;
      const state = this.state;
      const video = this.video;
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
      if (state.speaking && video.paused && video.readyState >= 2 && now - lastKick > 2000) {
        lastKick = now;
        video.play().catch(() => {});
      }
      if (state.speaking && state.audioCtx && state.sources.length === 0 &&
          state.audioCtx.currentTime > state.nextPlayTime + 0.15) {
        this.finishUtterance('drained');
      }
      if (now - lastResync > 1000) {
        lastResync = now;
        this.resyncAnchor();
      }
      if (now - lastDraw >= 30) {
        lastDraw = now;
        this.draw();
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);

    if (window.ResizeObserver) {
      this.observer = new ResizeObserver(() => {
        this.setSize(this.cssWidth(), this.cssHeight());
      });
      this.observer.observe(this.canvas.parentElement || this.canvas);
    }
  }

  resyncAnchor() {
    const state = this.state;
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

  pickPatch(frame) {
    const state = this.state;
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
    if (state.speaking) {
      if (past && frame - past.key <= PATCH_HOLD) return past.value;
      if (future && future.key - frame <= PATCH_HOLD) return future.value;
    }
    return null;
  }

  syncMask() {
    const state = this.state;
    const video = this.video;
    const mask = this.maskVideo;
    if (!state.compositor || !mask.src || mask.readyState < 2 || video.readyState < 2) return;
    if (mask.paused) mask.play().catch(() => {});
    const duration = mask.duration;
    if (!duration || !Number.isFinite(duration)) return;
    const loopFrames = state.meta.loop_frames;
    const index = ((Math.round(video.currentTime * FPS) % loopFrames) + loopFrames) % loopFrames;
    const target = index / FPS;
    let diff = Math.abs(mask.currentTime - target);
    diff = Math.min(diff, Math.abs(diff - duration));
    if (diff > MASK_DRIFT && !mask.seeking) mask.currentTime = target;
  }

  draw() {
    const state = this.state;
    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');
    const video = this.video;
    if (state.compositor) {
      this.syncMask();
      if (state.compositor.render(video, this.maskVideo)) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(state.compositor.canvas, 0, 0, canvas.width, canvas.height);
      }
    } else if (video.readyState >= 2) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    const patch = this.pickPatch(state.frameAbs);
    if (!patch) return;
    const meta = state.meta;
    const loopFrames = meta.loop_frames;
    const loopIndex = ((state.frameAbs % loopFrames) + loopFrames) % loopFrames;
    const bbox = meta.bbox[meta.loop_order[loopIndex]];
    const crop = meta.crop;
    const k = canvas.width / meta.width;
    if (!this.scratch) {
      this.scratch = document.createElement('canvas');
      this.scratch.width = crop;
      this.scratch.height = crop;
      this.scratchCtx = this.scratch.getContext('2d');
      this.edgeMask = buildEdgeMask(crop);
    }
    const image = new ImageData(crop, crop);
    const data = patch.data;
    for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
      image.data[j] = data[i];
      image.data[j + 1] = data[i + 1];
      image.data[j + 2] = data[i + 2];
      image.data[j + 3] = 255;
    }
    this.scratchCtx.putImageData(image, 0, 0);
    this.scratchCtx.globalCompositeOperation = 'destination-in';
    this.scratchCtx.drawImage(this.edgeMask, 0, 0);
    this.scratchCtx.globalCompositeOperation = 'source-over';
    ctx.drawImage(
      this.scratch,
      bbox[0] * k,
      bbox[1] * k,
      bbox[2] * k,
      bbox[3] * k,
    );
  }

  // ------------------------------------------------------------ 推理引擎

  async loadModels() {
    const keys = ['feather', 'visual'];
    const out = {};
    for (let i = 0; i < keys.length; i += 1) {
      const key = keys[i];
      const url = this.state.config.models[key];
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
        this.emit('progress', {
          phase: 'models',
          index: i + 1,
          total: keys.length,
          received,
          bytes: total,
          ratio: 0.08 + span * 0.84,
        });
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

  startWorker(prefer, models) {
    const state = this.state;
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
          this.emit('progress', {
            phase: 'warmup',
            done: message.done,
            total: message.total,
            ratio: 0.92 + 0.06 * (message.done / message.total),
          });
        } else if (message.type === 'ready') {
          state.provider = message.provider;
          state.ready = true;
          resolve();
        } else if (message.type === 'frame') {
          this.onWorkerFrame(message);
        } else if (message.type === 'error') {
          console.error('[worker]', message.message);
          if (message.retryWasm && prefer !== 'wasm') {
            this.startWorker('wasm', models).then(resolve, reject);
            return;
          }
          const reason = message.message.split('\n')[0];
          if (state.ready) {
            state.busy = false;
            this.emit('error', { message: reason });
            this.pump();
            return;
          }
          reject(new Error(message.message));
        }
      };
      worker.postMessage({
        type: 'init',
        provider: prefer || 'auto',
        wasmPath: state.config.ort.wasm,
        featherUrl: state.config.models.feather,
        visualUrl: state.config.models.visual,
        featherData: models && models.feather,
        visualData: models && models.visual,
        facesBase: state.config.media.faces,
        loopOrder: state.meta.loop_order,
      });
      state.worker = worker;
    });
  }

  onWorkerFrame(message) {
    const state = this.state;
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
    }
    this.pump();
  }
}

// ---------------------------------------------------------------- 工具

/** 贴片边缘羽化：外圈 6 像素淡出，避免露出矩形硬边。 */
function buildEdgeMask(size) {
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
}

/** int16 PCM 线性重采样，用于宿主给的非 16 kHz 数据。 */
function resampleInt16(input, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i += 1) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const frac = pos - i0;
    const s0 = input[i0] || 0;
    const s1 = i0 + 1 < input.length ? input[i0 + 1] : s0;
    out[i] = Math.round(s0 + (s1 - s0) * frac);
  }
  return out;
}
