/**
 * 推理 worker：LiveHuBERT + 144x144 数字人 UNet，跑在 WebGPU（不可用时退回 wasm）。
 *
 * 主线程只负责时钟与合成，这里负责：
 *   - 按需加载脸部裁片（152x152，与 C++ 的裁剪/缩放结果一致）
 *   - 拼 6 通道输入（前 3 通道参考帧、后 3 通道挖掉嘴部）
 *   - 每帧两次推理，返回合成好的 152x152 RGB 画面（主线程直接贴回 bbox）
 */
import * as ort from './vendor/ort/ort.webgpu.min.mjs';

const CROP = 152;
const BORDER = 4;
const INNER = CROP - 2 * BORDER;      // 144
const MASK_LEFT = 4;
const MASK_TOP = 4;
const MASK_RIGHT = 139;               // 不含
const MASK_BOTTOM = 134;              // 不含
const AUDIO_TOKENS = 40;
const AUDIO_DIM = 1024;
const WINDOW_SAMPLES = 12880;

let feather = null;
let visual = null;
let provider = 'wasm';
let facesBase = '';
let loopOrder = null;
const faceCache = new Map();

function post(type, payload, transfer) {
  self.postMessage({ type, ...payload }, transfer || []);
}

async function loadFaces(index) {
  const cached = faceCache.get(index);
  if (cached) return cached;
  const url = `${facesBase}/${String(index).padStart(3, '0')}.webp`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`脸卡加载失败：${url}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = new OffscreenCanvas(CROP, CROP);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = context.getImageData(0, 0, CROP, CROP).data;
  const rgb = new Uint8Array(CROP * CROP * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  faceCache.set(index, rgb);
  return rgb;
}

/** 6x144x144 输入：前 3 通道是参考脸，后 3 通道把嘴部挖空，模型据此重绘嘴型。 */
function buildImageInput(cardRgb) {
  const plane = INNER * INNER;
  const data = new Float32Array(6 * plane);
  for (let y = 0; y < INNER; y += 1) {
    const srcRow = ((y + BORDER) * CROP + BORDER) * 3;
    const dstRow = y * INNER;
    for (let x = 0; x < INNER; x += 1) {
      const src = srcRow + x * 3;
      const dst = dstRow + x;
      // C++ 里输入是 BGR：R 写第 3 通道、G 第 2、B 第 1
      const b = cardRgb[src + 2] / 255;
      const g = cardRgb[src + 1] / 255;
      const r = cardRgb[src] / 255;
      data[dst] = b;
      data[dst + plane] = g;
      data[dst + 2 * plane] = r;
      // 遮罩坐标是 144 内坐标（与 C++ 的 mask_left/top/right/bottom 一致），不要再加 border
      const masked =
        y >= MASK_TOP &&
        y < MASK_BOTTOM &&
        x >= MASK_LEFT &&
        x < MASK_RIGHT;
      if (!masked) {
        data[dst + 3 * plane] = b;
        data[dst + 4 * plane] = g;
        data[dst + 5 * plane] = r;
      }
    }
  }
  return data;
}

/** 把预测出的 144x144 BGR 贴回 152x152 参考帧，边 4 像素保持原样（与 C++ 一致）。 */
function compose(cardRgb, patchBgr) {
  const out = new Uint8Array(cardRgb);  // 拷贝一份，边距直接沿用原图
  for (let y = 0; y < INNER; y += 1) {
    const srcRow = y * INNER * 3;
    const dstRow = ((y + BORDER) * CROP + BORDER) * 3;
    for (let x = 0; x < INNER; x += 1) {
      const src = srcRow + x * 3;
      const dst = dstRow + x * 3;
      out[dst] = patchBgr[src + 2];
      out[dst + 1] = patchBgr[src + 1];
      out[dst + 2] = patchBgr[src];
    }
  }
  return out;
}

/** 输出是 NCHW（3 个平面），要按平面取再交错成 BGR，直接顺序读会变成马赛克。 */
function toUint8Bgr(tensor) {
  const values = tensor.data;
  const plane = INNER * INNER;
  const out = new Uint8Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    const dst = i * 3;
    const b = values[i];
    const g = values[plane + i];
    const r = values[2 * plane + i];
    out[dst] = b <= 0 ? 0 : b >= 1 ? 255 : Math.round(b * 255);
    out[dst + 1] = g <= 0 ? 0 : g >= 1 ? 255 : Math.round(g * 255);
    out[dst + 2] = r <= 0 ? 0 : r >= 1 ? 255 : Math.round(r * 255);
  }
  return out;
}

async function handleInit(message) {
  ort.env.wasm.wasmPaths = message.wasmPath;
  // 没有 SharedArrayBuffer 时多线程初始化会直接失败，必须退回单线程。
  const canThread = typeof SharedArrayBuffer !== 'undefined' && crossOriginIsolated;
  ort.env.wasm.numThreads = canThread
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
  ort.env.logLevel = 'error';
  facesBase = message.facesBase;
  loopOrder = message.loopOrder;

  // 默认先试 WebGPU。有些环境 requestAdapter() 单独探测会返回 null，
  // 但 ORT 自己带参数去要适配器却能成功，所以探测不参与决策，失败再重建 worker 走 wasm。
  let prefer = message.provider || 'webgpu';
  if (prefer === 'auto') prefer = 'webgpu';

  const options = { graphOptimizationLevel: 'all', executionProviders: [prefer] };
  // 主线程已带进度下载过模型，这里直接用字节建会话，不再重复拉一遍
  const featherSource = message.featherData || message.featherUrl;
  const visualSource = message.visualData || message.visualUrl;
  try {
    feather = await ort.InferenceSession.create(featherSource, options);
    visual = await ort.InferenceSession.create(visualSource, options);
    provider = prefer;
  } catch (error) {
    // 同一个 worker 里换 EP 会复用已失败的 wasm 初始化，只能让主线程重建 worker。
    post('error', {
      message: `[${prefer}] ${error && error.message ? error.message : error}`,
      retryWasm: prefer !== 'wasm',
    });
    return;
  }
  // 预热：先把所有脸卡解码进内存，否则首次播报时每帧都要等一次网络（线上 30~40ms/张），
  // 推理节奏被拖乱，开头会明显对不上音。
  await preloadFaces();
  // WebGPU 首次运行要编译管线，先跑一遍，别让第一句话等它。
  try {
    await handleFrame({ frame: 0, epoch: -1, audio: new Float32Array(WINDOW_SAMPLES) });
  } catch (error) {
    console.warn('[worker] 预热失败', error);
  }
  post('ready', { provider });
}

/** 并发预取 loop_order 里出现过的所有脸卡。 */
async function preloadFaces() {
  const unique = Array.from(new Set(loopOrder));
  let next = 0;
  let done = 0;
  const worker = async () => {
    while (next < unique.length) {
      const index = unique[next];
      next += 1;
      try {
        await loadFaces(index);
      } catch (error) {
        console.warn(`[worker] 脸卡 ${index} 预热失败`, error);
      }
      done += 1;
      if (done % 16 === 0 || done === unique.length) {
        post('warmup', { done, total: unique.length });
      }
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
}

/** frame：绝对帧号；audio：12880 个归一化采样（对应 window_start..window_end）。 */
async function handleFrame(message) {
  const loopFrames = loopOrder.length;
  const loopIndex = ((message.frame % loopFrames) + loopFrames) % loopFrames;
  const assetIndex = loopOrder[loopIndex];
  const card = await loadFaces(assetIndex);

  const started = performance.now();
  const waveform = new ort.Tensor('float32', message.audio, [1, message.audio.length]);
  const hidden = await feather.run({ waveform });
  const tokens = AUDIO_TOKENS * AUDIO_DIM;
  const audio = new Float32Array(tokens);
  audio.set(hidden.hidden.data.subarray(0, tokens));

  const imageInput = buildImageInput(card);
  const output = await visual.run({
    input: new ort.Tensor('float32', imageInput, [1, 6, INNER, INNER]),
    audio: new ort.Tensor('float32', audio, [1, AUDIO_TOKENS, AUDIO_DIM]),
  });
  const patch = toUint8Bgr(output.output);
  const composite = compose(card, patch);
  const elapsed = performance.now() - started;

  post(
    'frame',
    {
      frame: message.frame,
      epoch: message.epoch,
      assetIndex,
      data: composite.buffer,
      ms: elapsed,
    },
    [composite.buffer],
  );
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === 'init') {
      await handleInit(message);
    } else if (message.type === 'frame') {
      await handleFrame(message);
    }
  } catch (error) {
    post('error', { message: String(error && error.stack ? error.stack : error) });
  }
};
