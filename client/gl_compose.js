/**
 * 透明数字人合成器（WebGL）：
 *   视频帧 × mask.mp4 的 R 通道 → 带 alpha 的离屏 canvas。
 *
 * 为什么不用 2D 的 destination-in：mask 是「灰度视频」，像素本身是不透明的，
 * 只有颜色深浅，destination-in 读的是 alpha 通道，抠不出形状。WebGL 里把
 * 灰度当 alpha 用，才能在不重新编码带 alpha 视频的前提下拿到透明背景。
 *
 * 输出的 canvas 是预乘 alpha，直接 ctx.drawImage 叠到舞台画布上即可。
 */

const VERTEX = `
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAGMENT = `
precision mediump float;
varying vec2 vUv;
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform float uLow;
uniform float uHigh;
void main() {
  vec3 color = texture2D(uVideo, vUv).rgb;
  float m = texture2D(uMask, vUv).r;
  // RVM 的软边够自然，这里只做一次收紧，压掉半透明边缘上的背景色渗色
  float a = smoothstep(uLow, uHigh, m);
  gl_FragColor = vec4(color * a, a);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('着色器编译失败：' + log);
  }
  return shader;
}

function makeTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return { texture, width: 0, height: 0 };
}

/**
 * @param {number} width  舞台画布宽（1080）
 * @param {number} height 舞台画布高（1920）
 * @param {{low?: number, high?: number}} options  alpha 收紧区间
 */
export function createCompositor(width, height, options = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const gl = canvas.getContext('webgl', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    // 合成失败时要复用上一帧，必须保留绘制缓冲
    preserveDrawingBuffer: true,
  });
  if (!gl) return null;

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error('着色器链接失败：' + gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  // 一个覆盖全屏的三角形带
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  gl.uniform1i(gl.getUniformLocation(program, 'uVideo'), 0);
  gl.uniform1i(gl.getUniformLocation(program, 'uMask'), 1);
  gl.uniform1f(gl.getUniformLocation(program, 'uLow'), options.low ?? 0.15);
  gl.uniform1f(gl.getUniformLocation(program, 'uHigh'), options.high ?? 0.85);
  // HTMLVideoElement 上传默认上下颠倒，翻转一次才是正的
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.viewport(0, 0, width, height);
  gl.clearColor(0, 0, 0, 0);

  const videoTex = makeTexture(gl);
  const maskTex = makeTexture(gl);
  let lastVideoTime = -1;
  let lastMaskTime = -1;
  let hasVideo = false;
  let hasMask = false;

  function upload(unit, target, source) {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    if (target.width !== source.videoWidth || target.height !== source.videoHeight) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      target.width = source.videoWidth;
      target.height = source.videoHeight;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }
  }

  function ready(source) {
    return !!source && source.readyState >= 2 && source.videoWidth > 0;
  }

  return {
    canvas,
    /**
     * 合成一帧。返回 false 只代表「还从没合成出过画面」。
     * mask 或视频在 seek/缓冲时沿用上一次的纹理继续出画面，避免闪回原背景。
     */
    render(video, maskVideo) {
      const videoOk = ready(video);
      const maskOk = ready(maskVideo);
      if (!videoOk && !hasVideo) return false;
      if (!maskOk && !hasMask) return false;
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (videoOk && video.currentTime !== lastVideoTime) {
        upload(0, videoTex, video);
        lastVideoTime = video.currentTime;
        hasVideo = true;
      } else {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, videoTex.texture);
      }
      if (maskOk && maskVideo.currentTime !== lastMaskTime) {
        upload(1, maskTex, maskVideo);
        lastMaskTime = maskVideo.currentTime;
        hasMask = true;
      } else {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, maskTex.texture);
      }
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return true;
    },
  };
}
