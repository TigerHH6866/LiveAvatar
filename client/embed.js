/**
 * iframe 内桥接：把 postMessage 指令翻译成 LiveAvatarPlayer 调用，并把播放事件回抛给宿主。
 *
 * 宿主 → 这里（source = liveavatar-host）：
 *   { type: 'speak-pcm',  id?, pcm: ArrayBuffer, sampleRate?, replace? }
 *   { type: 'speak-audio', id?, data: ArrayBuffer, replace? }
 *   { type: 'interrupt',  id? }
 *   { type: 'set-background', id?, kind: 'none'|'color'|'image', value?, image?, mime? }
 *   { type: 'ping' }
 * 这里 → 宿主（source = liveavatar）：
 *   { type: 'ack'|'error', id, ... } / { type: 'event', event, payload } / { type: 'pong' }
 */

import { LiveAvatarPlayer } from './player.js';

const HOST_SOURCE = 'liveavatar-host';
const SELF_SOURCE = 'liveavatar';

let player = null;
let ready = false;
let pending = [];

function toHost(message, transfer) {
  if (window.parent === window) return;
  parent.postMessage({ source: SELF_SOURCE, ...message }, '*', transfer || []);
}

function reply(id, payload) {
  if (id === undefined) return;
  toHost({ type: 'ack', id, ...payload });
}

function fail(id, message) {
  toHost({ type: id === undefined ? 'event' : 'error', ...(id === undefined ? { event: 'error', payload: { message } } : { id, message }) });
}

async function boot() {
  const canvas = document.getElementById('stage');
  player = new LiveAvatarPlayer({
    canvas,
    source: document.getElementById('source'),
    mask: document.getElementById('mask'),
    bgLayer: document.getElementById('bg'),
    onEvent: (type, payload) => {
      if (type === 'ready') ready = true;
      if (type === 'ready') {
        const queue = pending;
        pending = [];
        queue.forEach((item) => item());
      }
      toHost({ type: 'event', event: type, payload });
    },
  });

  // 浏览器要求用户手势才能出声；iframe 里任何点击都算，顺手 resume 一次。
  const unlock = () => {
    if (!player) return;
    try {
      player.ensureAudio();
    } catch (error) {
      /* 忽略：下面还会再试 */
    }
  };
  document.addEventListener('pointerdown', unlock, { passive: true });

  try {
    await player.load();
  } catch (error) {
    console.error('[embed] 初始化失败', error);
    toHost({ type: 'event', event: 'error', payload: { message: String(error && error.message || error), fatal: true } });
  }
}

function handle(message) {
  const { type, id } = message;
  if (type === 'ping') {
    toHost({ type: 'pong', ready });
    return;
  }
  if (!player) {
    fail(id, '播放器尚未初始化');
    return;
  }
  const run = () => {
    try {
      if (type === 'speak-pcm') {
        player.speakPcm(message.pcm, {
          sampleRate: message.sampleRate || 16000,
          replace: !!message.replace,
        });
      } else if (type === 'speak-audio') {
        player.speakAudio(message.data, { replace: !!message.replace });
      } else if (type === 'interrupt') {
        player.interrupt();
      } else if (type === 'set-background') {
        player.setBackground({
          kind: message.kind || 'none',
          value: message.value,
          image: message.image,
          mime: message.mime,
        });
      } else if (type === 'resize') {
        player.setSize(player.cssWidth(), player.cssHeight());
      } else {
        fail(id, `未知指令：${type}`);
        return;
      }
      reply(id, { ok: true });
    } catch (error) {
      fail(id, String(error && error.message || error));
    }
  };
  // 模型还在下载时先排队，就绪后按顺序执行，避免宿主过早调用被丢弃
  if (!ready && type !== 'interrupt') pending.push(run);
  else run();
}

window.addEventListener('message', (event) => {
  const data = event.data;
  if (!data || typeof data !== 'object' || data.source !== HOST_SOURCE) return;
  handle(data);
});

// 焦点落在 iframe 内时，浏览器按键不会冒泡到宿主页面：
// Ctrl+Alt+T 在这里接住，转成事件交给宿主去显隐调试面板（真实业务页面忽略即可）。
window.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.altKey && !event.metaKey &&
      (event.key === 't' || event.key === 'T')) {
    event.preventDefault();
    toHost({ type: 'event', event: 'debug-toggle', payload: {} });
  }
});

boot();
