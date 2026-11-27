/**
 * 宿主侧 SDK：在任意网站里嵌一个透明数字人。
 *
 *   <script src="https://liveavatar.daihuajun.cn/liveavatar.js"></script>
 *   <script>
 *     const avatar = LiveAvatar.mount('#host', { width: 360, height: 640 });
 *     await avatar.ready;
 *     avatar.speak({ pcm: int16Array, sampleRate: 16000 });
 *   </script>
 *
 * 内部是一个 iframe：模型、素材、推理都在 iframe 自己的源里跑，
 * 宿主不需要配 COOP/COEP，也不用把域名加进 OSS 的 CORS 白名单。
 *
 * 约定：
 *   - 一页只挂一个实例（每个实例都会加载一份模型会话）；
 *   - 音频可以用流式 PCM（int16，16 kHz 单声道）或整段音频文件（wav/mp3 等）；
 *   - 默认连续排队播放，需要打断时显式调用 interrupt() 或传 replace: true；
 *   - 背景和图片请传二进制，避免宿主域名被 COEP / CORS 拦掉。
 */

(function (global) {
  const HOST_SOURCE = 'liveavatar-host';
  const SELF_SOURCE = 'liveavatar';

  class LiveAvatar {
    constructor(container, options) {
      this.container = typeof container === 'string'
        ? document.querySelector(container)
        : container;
      if (!this.container) throw new Error('LiveAvatar: 找不到挂载容器');
      this.options = options || {};
      this.listeners = new Map();
      this.ready = new Promise((resolve, reject) => {
        this.resolveReady = resolve;
        this.rejectReady = reject;
      });
      this.ready.catch(() => {});
      this.nextId = 1;
      this.pendingAcks = new Map();
      this.destroyed = false;
      this.mount();
    }

    mount() {
      const opts = this.options;
      const src = opts.src
        || (opts.base ? `${String(opts.base).replace(/\/$/, '')}/embed.html` : './embed.html');
      const width = opts.width || 360;
      const height = opts.height || 640;

      const iframe = document.createElement('iframe');
      iframe.src = src;
      iframe.title = 'LiveAvatar';
      // autoplay：允许 iframe 内直接出声；不申请麦克风（录音由宿主负责）
      iframe.setAttribute('allow', 'autoplay');
      iframe.setAttribute('scrolling', 'no');
      iframe.style.border = '0';
      iframe.style.background = 'transparent';
      iframe.style.width = `${width}px`;
      iframe.style.height = `${height}px`;
      iframe.style.display = 'block';
      this.iframe = iframe;

      this.onMessage = (event) => {
        if (event.source !== iframe.contentWindow) return;
        const data = event.data;
        if (!data || typeof data !== 'object' || data.source !== SELF_SOURCE) return;
        this.receive(data);
      };
      window.addEventListener('message', this.onMessage);
      this.container.appendChild(iframe);
      this.ping();
    }

    // ---------------------------------------------------------- 通信

    ping() {
      this.pingTimer = setInterval(() => {
        this.post({ type: 'ping' });
      }, 200);
      this.timeout = setTimeout(() => {
        this.emit('error', { message: '数字人加载超时' });
      }, 120000);
    }

    post(message, transfer) {
      if (!this.iframe || !this.iframe.contentWindow) return;
      this.iframe.contentWindow.postMessage({ source: HOST_SOURCE, ...message }, '*', transfer || []);
    }

    request(type, payload, transfer) {
      const id = this.nextId;
      this.nextId += 1;
      return new Promise((resolve, reject) => {
        this.pendingAcks.set(id, { resolve, reject });
        this.post({ type, id, ...payload }, transfer);
        setTimeout(() => {
          if (this.pendingAcks.has(id)) {
            this.pendingAcks.delete(id);
            reject(new Error(`指令超时：${type}`));
          }
        }, 10000);
      });
    }

    receive(data) {
      if (data.type === 'pong') {
        if (data.ready) this.clearTimers();
        return;
      }
      if (data.type === 'ack' || data.type === 'error') {
        const entry = this.pendingAcks.get(data.id);
        if (!entry) return;
        this.pendingAcks.delete(data.id);
        if (data.type === 'ack') entry.resolve(data);
        else entry.reject(new Error(data.message || '指令失败'));
        return;
      }
      if (data.type === 'event') {
        if (data.event === 'ready') {
          this.clearTimers();
          this.resolveReady(this);
        } else if (data.event === 'error' && data.payload && data.payload.fatal) {
          this.rejectReady(new Error(data.payload.message || '初始化失败'));
        }
        this.emit(data.event, data.payload || {});
      }
    }

    clearTimers() {
      if (this.pingTimer) clearInterval(this.pingTimer);
      if (this.timeout) clearTimeout(this.timeout);
      this.pingTimer = null;
      this.timeout = null;
    }

    // ---------------------------------------------------------- 事件

    on(event, handler) {
      if (!this.listeners.has(event)) this.listeners.set(event, new Set());
      this.listeners.get(event).add(handler);
      return () => this.off(event, handler);
    }

    off(event, handler) {
      const set = this.listeners.get(event);
      if (set) set.delete(handler);
    }

    emit(event, payload) {
      const set = this.listeners.get(event);
      if (!set) return;
      for (const handler of set) {
        try {
          handler(payload);
        } catch (error) {
          console.warn('[LiveAvatar] 事件回调异常', error);
        }
      }
    }

    // ---------------------------------------------------------- 对外接口

    /**
     * 播报。
     *   speak({ pcm: Int16Array, sampleRate?: 16000, replace?: false })
     *   speak({ audio: ArrayBuffer | Blob, replace?: false })   // wav/mp3 等
     */
    async speak(input) {
      const options = input || {};
      const replace = !!options.replace;
      if (options.pcm) {
        const pcm = options.pcm instanceof Int16Array
          ? options.pcm
          : new Int16Array(options.pcm);
        const buffer = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
        return this.request('speak-pcm', {
          pcm: buffer,
          sampleRate: options.sampleRate || 16000,
          replace,
        }, [buffer]);
      }
      if (options.audio) {
        const blob = options.audio instanceof Blob ? options.audio : new Blob([options.audio]);
        const buffer = await blob.arrayBuffer();
        return this.request('speak-audio', { data: buffer, replace }, [buffer]);
      }
      throw new Error('speak 需要 pcm 或 audio');
    }

    interrupt() {
      return this.request('interrupt', {});
    }

    /**
     * 背景。
     *   setBackground({ kind: 'none' })
     *   setBackground({ kind: 'color', value: '#0b0d12' })
     *   setBackground({ kind: 'image', image: ArrayBuffer | Blob, mime?: 'image/jpeg' })
     */
    async setBackground(spec) {
      const options = spec || {};
      const message = { kind: options.kind || 'none', value: options.value || '' };
      let transfer = [];
      let image = options.image;
      if (options.kind === 'image' && image) {
        if (typeof image === 'string') {
          message.image = image;   // 已经是 URL 时由宿主保证跨源可用（建议优先用二进制）
        } else {
          if (image instanceof Blob) image = await image.arrayBuffer();
          else if (image instanceof ArrayBuffer) image = image.slice(0);
          message.image = image;
          message.mime = options.mime || 'image/png';
          transfer = [image];
        }
      }
      return this.request('set-background', message, transfer);
    }

    /** 调整显示尺寸（等比由 iframe 内部 CSS 处理，这里只改外框）。 */
    setSize(width, height) {
      if (!this.iframe) return;
      this.iframe.style.width = `${width}px`;
      this.iframe.style.height = `${height}px`;
      this.post({ type: 'resize' });
    }

    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      this.clearTimers();
      window.removeEventListener('message', this.onMessage);
      for (const entry of this.pendingAcks.values()) {
        entry.reject(new Error('实例已销毁'));
      }
      this.pendingAcks.clear();
      this.listeners.clear();
      if (this.iframe && this.iframe.parentNode) this.iframe.parentNode.removeChild(this.iframe);
      this.iframe = null;
    }
  }

  LiveAvatar.mount = function mount(container, options) {
    return new LiveAvatar(container, options);
  };

  LiveAvatar.version = '1.0.0';

  if (typeof module === 'object' && module.exports) module.exports = LiveAvatar;
  else global.LiveAvatar = LiveAvatar;
})(typeof window !== 'undefined' ? window : this);
