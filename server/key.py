###############################################################################
# key.py
#
# 第三方服务凭证与账号相关配置集中在此文件，方便独立设置、避免散落在业务代码里。
# 修改后重启服务生效。环境变量优先（COZE_TOKEN / COZE_BOT_ID /
# ALIYUN_TTS_API_KEY / ALIYUN_TTS_VOICE），这里只放没有走环境的默认值。
###############################################################################

# ---------------- Coze ----------------
COZE_TOKEN = ""            # 走环境变量 COZE_TOKEN（轮换后的 sat_ 令牌）
COZE_BOT_ID = "7401082582928588841"

# ---------------- 阿里云百炼实时 TTS（Qwen-Audio-TTS/CosyVoice） ----------------
ALIYUN_TTS_API_KEY = ""    # 走环境变量 ALIYUN_TTS_API_KEY
# 音色（customer_taiyi 复刻音色对应的真实 voice_id）
ALIYUN_TTS_VOICE = "qwen-audio-3.1-tts-flash-bailian-7f1354a1e1d94b3983b429dcd298af2b"
