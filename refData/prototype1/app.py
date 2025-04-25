# app.py - FastAPI 後端
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import json
import asyncio
import azure.cognitiveservices.speech as speechsdk
import tempfile
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

app = FastAPI(title="主題性自由對話語音評測系統-API")

# 啟用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 資料模型
class TopicRequest(BaseModel):
    topic: str

class SpeechAssessmentRequest(BaseModel):
    topic: str
    audio_id: str
    transcription: Optional[str] = None

class AssessmentResult(BaseModel):
    speechQuality: Dict[str, float]
    contentQuality: Dict[str, float]
    overallFeedback: str
    improvementSuggestions: List[str]

# Azure Speech Service 配置
class AzureSpeechConfig:
    def __init__(self):
        self.speech_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", f"https://{os.getenv('AZURE_SPEECH_REGION', 'eastus')}.api.cognitive.microsoft.com/")

# 獲取 Azure Speech Service 配置
def get_speech_config():
    return AzureSpeechConfig()

# 模擬主題-圖片映射數據庫
TOPICS_IMAGES = {
    "nature": "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&w=1440&q=80",
    "technology": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1440&q=80",
    "education": "https://images.unsplash.com/photo-1503676260728-1c00da094a0b?auto=format&fit=crop&w=1440&q=80",
    "health": "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=1440&q=80",
    "culture": "https://images.unsplash.com/photo-1513735492246-483525079686?auto=format&fit=crop&w=1440&q=80"
}

# 確保存儲錄音的目錄存在
os.makedirs("audio_files", exist_ok=True)

# 路由定義
@app.get("/")
async def root():
    return {"status": "success", "message": "主題性自由對話語音評測系統API服務運行中"}

@app.get("/topics")
async def get_topics():
    return {"topics": list(TOPICS_IMAGES.keys())}

@app.post("/generate-image")
async def generate_image(request: TopicRequest):
    if request.topic not in TOPICS_IMAGES:
        raise HTTPException(status_code=404, detail="找不到該主題")
    
    # 模擬生成與主題相關的圖片
    # 在實際應用中，可能會調用AI圖像生成API
    await asyncio.sleep(1)  # 模擬處理時間
    
    return {
        "topic": request.topic,
        "image_url": TOPICS_IMAGES[request.topic]
    }

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    # 生成唯一ID作為檔案名
    audio_id = str(uuid.uuid4())
    file_path = f"audio_files/{audio_id}.wav"
    
    # 保存上傳的音頻檔
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    return {"status": "success", "audio_id": audio_id}

# 測試 Azure Speech Service 連接
@app.get("/test-speech-service")
async def test_speech_service(
    speech_key: str = None, 
    speech_region: str = None, 
    speech_endpoint: str = None, 
    config: AzureSpeechConfig = Depends(get_speech_config)
):
    # 使用參數值覆蓋配置（如果提供）
    key = speech_key or config.speech_key
    region = speech_region or config.speech_region
    endpoint = speech_endpoint or config.speech_endpoint
    
    if not key:
        return {"success": False, "message": "未提供Speech Key"}
    
    try:
        # 建立語音配置
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.endpoint = endpoint
        
        # 測試語音服務連接
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = speech_synthesizer.speak_text_async("測試Azure語音服務連接").get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return {"success": True, "message": "服務可用"}
        else:
            return {"success": False, "message": f"服務不可用，原因: {result.reason}"}
    except Exception as e:
        return {"success": False, "message": f"連接錯誤: {str(e)}"}

# 語音轉文字函數
@app.post("/speech-to-text")
async def speech_to_text(
    audio_id: str, 
    speech_key: str = None, 
    speech_region: str = None, 
    config: AzureSpeechConfig = Depends(get_speech_config)
):
    audio_path = f"audio_files/{audio_id}.wav"
    
    # 檢查音頻檔是否存在
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="找不到該音頻檔")
    
    # 使用參數值覆蓋配置（如果提供）
    key = speech_key or config.speech_key
    region = speech_region or config.speech_region
    
    if not key:
        return {"status": "error", "message": "未提供Speech Key"}
    
    try:
        # 建立語音配置
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        
        # 設置音頻配置
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        
        # 創建語音識別器
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            audio_config=audio_config,
            language="zh-TW"  # 使用繁體中文，可以根據需要調整
        )
        
        # 執行語音識別
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"status": "success", "text": result.text}
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {"status": "error", "message": "無法識別語音"}
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails.from_result(result)
            return {"status": "error", "message": f"語音識別取消: {cancellation.reason}"}
        else:
            return {"status": "error", "message": f"語音識別失敗: {result.reason}"}
    except Exception as e:
        return {"status": "error", "message": f"處理過程中發生錯誤: {str(e)}"}

# 文字校正函數 (空函數，由用戶實現)
@app.post("/correct-transcription")
async def correct_transcription(text: str):
    # 這裡應該是調用LLM進行文字校正的代碼
    # 留空供用戶實現
    return {"status": "success", "corrected_text": text}

# 語音評測函數
@app.post("/assess-speech", response_model=AssessmentResult)
async def assess_speech(
    request: SpeechAssessmentRequest,
    speech_key: str = None, 
    speech_region: str = None, 
    config: AzureSpeechConfig = Depends(get_speech_config)
):
    audio_path = f"audio_files/{request.audio_id}.wav"
    
    # 檢查音頻檔是否存在
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="找不到該音頻檔")
    
    # 使用參數值覆蓋配置（如果提供）
    key = speech_key or config.speech_key
    region = speech_region or config.speech_region
    
    if not key:
        raise HTTPException(status_code=400, detail="未提供Speech Key")
    
    try:
        # 建立語音配置
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        
        # 設置音頻配置
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        
        # 這裡應該是實際的語音評測邏輯
        # 目前使用模擬數據
        
        # 模擬處理時間
        await asyncio.sleep(2)
        
        # 返回模擬的評測結果
        return {
            "speechQuality": {
                "pronunciation": 8.5,
                "rhythm": 7.2,
                "intonation": 9.0
            },
            "contentQuality": {
                "vocabulary": 8.0,
                "relevance": 7.5,
                "coherence": 8.2
            },
            "overallFeedback": "您的發言整體表現優秀。您的發音清晰，語調豐富，能夠有效地表達您的想法。在主題相關性方面還有提升空間，可以嘗試更深入地探討主題的核心概念。",
            "improvementSuggestions": [
                "可以適當放慢語速，使聽眾更容易理解您的觀點",
                "嘗試使用更多與主題直接相關的專業詞彙",
                "可以加強語言的邏輯連貫性，使用更多過渡詞連接不同觀點"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評測過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
