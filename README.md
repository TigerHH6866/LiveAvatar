# LiveAvatar Web

# 让您的网页服务轻松接入实时互动数字人
基于C++引擎开发，游览器webgpu或WASM模式推理，模型跑在客户端上，服务器与客户端都不需要显卡，随便一块cpu就可以。PC端现代浏览器运行流畅，口型良好

#零显卡-响应快-高并发

## Update Log
2026-11-23: demo上线了透明背景功能，右上角可提交新背景图，demo：https://three.daihuajun.cn/webgpu


## 目录

```
client/          前端：WebGPU 推理 worker + 音频时钟渲染 + WebGL 抠像合成
server/          后端：Flask，只做 ASR / 对话 / TTS（app.py, chat.py, asr.py, key.py）
models/          live_hubert.onnx、<形象>.onnx（视觉模型与数字人同名）
assets/<形象>/   loop.mp4、mask.mp4、faces/*.webp、meta.json
tools/           matte_loop.py（离线抠像）、prepare_assets.py、export_onnx.py
```


## 运行

```
pip install -r requirements.txt
python server/app.py            
访问：http://127.0.0.1:7861/

```

当前后端接入了coze agent，百炼asr(qwenqwen-audio-3.1-asr-flash)，百炼tts(qwen-audio-3.1-tts-flash)，需要自行开通key。您也可以自行二次开发接入其他agent和语音模型
凭据在 `server/key.py` 或环境变量读（`COZE_TOKEN`、`COZE_BOT_ID`、`ALIYUN_TTS_API_KEY`、`ALIYUN_TTS_VOICE`）。

demo：https://three.daihuajun.cn/webgpu

调试后端：`?ep=webgpu` 强制 WebGPU、`?ep=wasm` 强制 wasm。



## 关键设计

- **音频是主时钟**：TTS 的 16 kHz PCM 收下来后延迟 0.9 s 排队播放，多出来的时间用来提前推理；
  视频帧号由音频采样位置推导，模型窗口固定为「当前帧前后共 12880 个采样」。
- **只算脸**：模型吃 152x152（内层 144）的脸部裁片，输出同样大小的画面，贴回素材的 bbox。
  1080p 全画面来自素材循环视频，不占推理。
- **素材必须下发**：模型输入就是素材本身的脸（与 C++ 引擎一致），但不发 1080p 原图，
  只发 mp4（7 MB）+ lossless WebP 脸卡（4.3 MB）+ 8 KB 的 bbox 表。
- **降级**：WebGPU 会话创建失败时自动重建 worker 走 wasm（44 ms/帧，帧率不足 25 fps 但可用）。
- 模型是 fp32 ONNX（33.7 MB），首次加载偏重，后续靠浏览器 HTTP 缓存。


## 实测（i5-12490）

| 后端 | 每帧耗时 | 25 fps |
| --- | --- | --- |
| WebGPU | 17.5 ms（首帧 126 ms 用于编译管线，已用预热规避） | 有余量 |
| wasm（SIMD，4 线程） | 43 ms | 略超预算，会掉帧 |


## 数字形象模型限制

- 项目所用数字人形象所有权归属本作者，仅可用于研究、学习、演示，不可直接商用
- demo里的透明及可更换背景功能为商用版
- 如需商用版或训练自己的数字人形象，请联系：vx( tigerhe6866 )
