import requests
import subprocess
import os

# 初始模型狀態記錄
current_sovits_model = [None]
current_gpt_model = [None]

# 模型設定
voice_models = {#這裡是供選擇的聲音模型
    "mann": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mann_e2_s50_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mann-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mann.wav",
        "prompt_text": "作者蘆園老師因為當時連載尚未結束，所以一直不想要改編成其他載體，可是製作方不斷釋出他們的誠意，他終究還是開口答應了"
    },
    "mrd": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mrd_e2_s102_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mrd-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mrd.wav",
        "prompt_text": "開始審理案件，被告我看看，在二零二三年接了太多鯊魚廣告，受到觀眾舉發控告。"
    },
    "andy": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/andy_e2_s98_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/andy-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\andy.wav",
        "prompt_text": "好玩有趣的創意，用一支手機拍攝起來，開心的分享在社群媒體上面，剛開始都沒有人看"
    }
}

# 切換模型功能
def switch_model_if_needed(current_model_var, new_path, switch_url, model_name):
    if current_model_var[0] == new_path:
        print(f"✅ {model_name} 模型已是最新：{new_path}")
        return
    print(f"🔄 切換 {model_name} 模型：{current_model_var[0]} → {new_path}")
    res = requests.get(switch_url, params={"weights_path": new_path})
    if res.status_code == 200:
        print(f"✅ 已切換 {model_name} 模型")
        current_model_var[0] = new_path
    else:
        print(f"❌ {model_name} 模型切換失敗：{res.text}")

# Whisper ASR 呼叫函式（已修正）
def run_whisper_asr(input_path, output_dir, language="en", precision="float32"):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "D:\\GPT-SoVITS-v4\\runtime\\python.exe",
        "-Xutf8",
        "D:\\GPT-SoVITS-v4\\tools\\asr\\fasterwhisper_asr.py",
        "-i", input_path,
        "-o", output_dir,
        "-l", language,
        "-p", precision
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("✅ Whisper ASR 執行完成")
    except subprocess.CalledProcessError as e:
        print("❌ Whisper ASR 執行失敗：", e.stderr)
        return None

    try:
        list_files = [f for f in os.listdir(output_dir) if f.endswith(".list")]
        if not list_files:
            print("❌ 找不到 ASR 的 .list 檔案")
            return None
        list_path = os.path.join(output_dir, list_files[0])
        with open(list_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            parts = line.split("|")
            if len(parts) >= 4:
                return parts[3]
            else:
                print("❌ .list 檔案格式錯誤：", line)
                return None
    except Exception as e:
        print("❌ 讀取 .list 檔案失敗：", str(e))
        return None

# 外部呼叫用的模型切換
def switch_model(model_key):
    model = voice_models.get(model_key)
    if not model:
        raise ValueError(f"未知的模型鍵：{model_key}")
    switch_model_if_needed(current_sovits_model, model["sovits"], "http://127.0.0.1:9880/set_sovits_weights", "SoVITS")
    switch_model_if_needed(current_gpt_model, model["gpt"], "http://127.0.0.1:9880/set_gpt_weights", "GPT")
    return model["ref_audio_path"], model["prompt_text"]

# 合成 TTS 語音
def synthesize_tts(text, ref_audio_path, prompt_text, output_path="static/output.wav"):
    tts_url = "http://127.0.0.1:9880/tts"
    params = {
        "text": text,
        "text_lang": "auto",
        "ref_audio_path": ref_audio_path,
        "prompt_lang": "zh",
        "prompt_text": prompt_text,
        "text_split_method": "cut5",
        "batch_size": 2,
        "sample_steps": 16,
        "media_type": "wav",
        "streaming_mode": "false"
    }

    res = requests.get(tts_url, params=params)
    if res.status_code == 200:
        
        with open(output_path, "wb") as f:
            f.write(res.content)
        print("✅ 語音合成成功：", output_path)
        return True
    else:
        print("❌ 語音合成失敗：", res.status_code, res.text)
        return False
