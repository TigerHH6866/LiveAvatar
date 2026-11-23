/**
 * Silero VAD（ONNX Runtime Web，wasm 单线程）+ 断句状态机。
 *
 * 用法：
 *   await loadVadOrt();                       // 动态加载 ./vad/ort.min.js
 *   const vad = new SileroVAD({ sampleRate: 16000 });
 *   await vad.init('./vad/silero_vad.onnx');
 *   const prob = await vad.processFrame(frame);   // Int16Array，长度 512
 */

class SileroVAD {
  constructor(options = {}) {
    this.sampleRate = options.sampleRate || 16000;
    this.frameSize = this.sampleRate === 16000 ? 512 : 256;
    this.session = null;
    this.state = new Float32Array(2 * 1 * 128);
    this.initialized = false;
  }

  async init(modelUrl, wasmBase) {
    if (this.initialized) return;
    if (typeof ort === 'undefined') throw new Error('VAD 运行时未加载');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    // wasm 可以放在 OSS（传 liveavatar/vad/ort-wasm-simd.wasm），没配就走本机 ./vad/
    ort.env.wasm.wasmPaths = wasmBase || new URL('./vad/', location.href).href;
    ort.env.logSeverityLevel = 3;
    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    this.reset();
    this.initialized = true;
  }

  reset() {
    this.state = new Float32Array(2 * 1 * 128);
  }

  /** Int16 帧 → 语音概率 0~1 */
  async processFrame(frame) {
    if (!this.initialized) return 0;
    const input = new Float32Array(this.frameSize);
    const n = Math.min(frame.length, this.frameSize);
    for (let i = 0; i < n; i += 1) input[i] = frame[i] / 32768;

    const names = this.session.inputNames;
    const feeds = {};
    feeds[names[0]] = new ort.Tensor('float32', input, [1, this.frameSize]);
    feeds[names[1] || 'state'] = new ort.Tensor('float32', new Float32Array(this.state), [2, 1, 128]);
    feeds[names[2] || 'sr'] = new ort.Tensor(
      'int64',
      new BigInt64Array([BigInt(this.sampleRate)]),
      [1],
    );

    const out = await this.session.run(feeds);
    const prob = out[this.session.outputNames[0]].data[0];
    const next = out[this.session.outputNames[1] || 'stateN'];
    if (next && next.data.length === this.state.length) this.state.set(next.data);
    for (const tensor of Object.values(out)) {
      if (tensor && typeof tensor.dispose === 'function') tensor.dispose();
    }
    return prob;
  }
}

/**
 * 断句状态机：IDLE → RECORDING → ENDING → IDLE
 *   - 连续 2 帧 > 0.6 判定开口
 *   - 连续 25 帧（约 0.8 s）静音进入 ENDING
 *   - ENDING 再等 13 帧（约 0.4 s）确认结束，期间恢复说话会合并回 RECORDING
 *   - 不足 17 帧（约 0.5 s）的片段丢弃
 */
class VADStateMachine {
  constructor(callbacks = {}) {
    this.onSpeechStart = callbacks.onSpeechStart || (() => {});
    this.onSpeechEnd = callbacks.onSpeechEnd || (() => {});
    this.onSpeechActive = callbacks.onSpeechActive || (() => {});

    this.startThreshold = 0.6;
    this.endThreshold = 0.35;
    this.startFrames = 2;
    this.endFrames = 25;
    this.endingGraceFrames = 13;
    this.minFrames = 17;
    this.maxFrames = 640;   // 约 20 s，防止一直说不停导致缓存无上限

    this.state = 'IDLE';
    this.speechFrameCount = 0;
    this.silenceFrameCount = 0;
    this.chunks = [];
    this.preRoll = [];
    this.preRollMax = 8;
  }

  onVADFrame(prob, frame) {
    switch (this.state) {
      case 'IDLE':
        this.preRoll.push(this._toFloat(frame));
        if (this.preRoll.length > this.preRollMax) this.preRoll.shift();
        if (prob >= this.startThreshold) {
          this.speechFrameCount += 1;
          if (this.speechFrameCount >= this.startFrames) {
            this.state = 'RECORDING';
            this.silenceFrameCount = 0;
            this.chunks = this.preRoll.slice();
            this.onSpeechStart();
          }
        } else {
          this.speechFrameCount = 0;
        }
        break;

      case 'RECORDING':
        this.chunks.push(this._toFloat(frame));
        if (this.chunks.length >= this.maxFrames) {
          this._finalize();
          break;
        }
        if (prob <= this.endThreshold) {
          this.silenceFrameCount += 1;
          if (this.silenceFrameCount >= this.endFrames) this.state = 'ENDING';
        } else {
          this.silenceFrameCount = 0;
        }
        break;

      case 'ENDING':
        this.chunks.push(this._toFloat(frame));
        if (prob >= this.startThreshold) {
          this.silenceFrameCount = 0;
          this.state = 'RECORDING';
        } else {
          this.silenceFrameCount += 1;
          if (this.silenceFrameCount >= this.endFrames + this.endingGraceFrames) {
            this._finalize();
          }
        }
        break;

      default:
        break;
    }
    this.onSpeechActive(this.state !== 'IDLE');
  }

  /** 外部强行复位（打断、播报开始等） */
  reset() {
    this.state = 'IDLE';
    this.speechFrameCount = 0;
    this.silenceFrameCount = 0;
    this.chunks = [];
    this.preRoll = [];
  }

  _finalize() {
    const frames = this.chunks.length;
    if (frames >= this.minFrames) this.onSpeechEnd(this._toWavBlob());
    this.reset();
  }

  _toFloat(int16) {
    const out = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i += 1) out[i] = int16[i] / 32768;
    return out;
  }

  _toWavBlob() {
    const total = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const pcm = new Int16Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      for (let i = 0; i < chunk.length; i += 1) {
        const x = Math.max(-1, Math.min(1, chunk[i]));
        pcm[offset + i] = x < 0 ? x * 0x8000 : x * 0x7fff;
      }
      offset += chunk.length;
    }
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const write = (at, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(at + i, text.charCodeAt(i));
    };
    const dataSize = pcm.length * 2;
    write(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    write(8, 'WAVE');
    write(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, 'data');
    view.setUint32(40, dataSize, true);
    return new Blob([header, pcm.buffer], { type: 'audio/wav' });
  }
}

/** 动态引入 VAD 专用 ORT（1.17 UMD 版，自带 simd wasm）。 */
function loadVadOrt() {
  if (window.__vadOrt) return window.__vadOrt;
  window.__vadOrt = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = './vad/ort.min.js';
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('VAD 运行时加载失败'));
    document.head.appendChild(script);
  });
  return window.__vadOrt;
}

export { SileroVAD, VADStateMachine, loadVadOrt };
