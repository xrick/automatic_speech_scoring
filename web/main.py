from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pydub import AudioSegment
import os
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from typing import Optional
from contextlib import asynccontextmanager

from langchain_ollama.llms import OllamaLLM
from faster_whisper import WhisperModel

app = FastAPI()

# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# 確保錄音文件存儲目錄存在
UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/record.html")


# Azure 語音服務配置
SPEECH_KEY = "7c400507-6b30-4a2f-97f9-5baa6c9e4e28"#os.getenv("AZURE_SPEECH_KEY", "your_speech_key")expired.
SPEECH_REGION = "en-US"#os.getenv("AZURE_SPEECH_REGION", "your_region")

# local faster-whisper model setup
WHISPER_MODEL_TYPE = "distil-large-v3";
WHISPER_MODEL_PATH = "/Users/xrickliao/WorkSpaces/LLM_Repo/models/Whisper/Models/faster_distil_whisper_large_v3_snapdwn/";
whisper_model=None;

# local llm setup
LLM_MODEL='phi4:latest';
SPELL_CORRECT_PROMPT='''
    role:you are a perfect english spelling checker.
    task:
    1.please do spelling checking for the following senteces.
    2.Do not change the style, form, and structure of the sentences.
    3.only return the corrected sentences
    sentences:\n{0}\n
'''
llm=None;

# fastapi startup setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    whisper_model = WhisperModel(WHISPER_MODEL_TYPE, device="cpu", compute_type="int8", 
                                 download_root=WHISPER_MODEL_PATH, local_files_only=True);
    llm = OllamaLLM(model=LLM_MODEL);
    yield
    # Clean up objects and release the resources
    whisper_model = None;
    llm = None;


def perform_pronunciation_assessment(audio_file: str, reference_text: str) -> dict:
    """執行發音評估"""
    # try:
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY, 
        region=SPEECH_REGION
    )
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    
    # 創建發音評估配置
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()
    
    # 創建語音識別器
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=audio_config,
        language="en-US"  # 可根據需求修改語言
    )
    
    # 應用發音評估配置
    pronunciation_config.apply_to(speech_recognizer)
    
    # 執行識別
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        return {
            "recognized_text": result.text,
            "accuracy_score": pronunciation_result.accuracy_score,
            "pronunciation_score": pronunciation_result.pronunciation_score,
            "completeness_score": pronunciation_result.completeness_score,
            "fluency_score": pronunciation_result.fluency_score,
            "prosody_score": pronunciation_result.prosody_score,
            "words": [
                {
                    "word": word.word,
                    "accuracy_score": word.accuracy_score,
                    "error_type": word.error_type
                }
                for word in pronunciation_result.words
            ]
        }
    else:
        raise Exception(f"Speech recognition failed: {result.reason}")
            
    # except Exception as e:
    #     raise Exception(f"Pronunciation assessment failed: {str(e)}")

def gen_ref_text(audio_file:str=None):
    ref_txt = "no content"
    return ref_txt

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...), reference_text: str = Form(...)):
# async def upload_audio(audio: UploadFile = File(...)):
    try:
        # 生成時間戳記
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始 WebM 文件
        webm_filename = f"recording_{timestamp}.webm"
        webm_filepath = os.path.join(UPLOAD_DIR, webm_filename)
        
        with open(webm_filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # 轉換為 WAV 格式
        wav_filename = f"recording_{timestamp}.wav"
        wav_filepath = os.path.join(UPLOAD_DIR, wav_filename)
        
        # 使用 pydub 進行轉換
        audio_segment = AudioSegment.from_file(webm_filepath)
        audio_segment.export(wav_filepath, format="wav")
        
        # reference_text = """
        #     This is speech assessment, pleas say: hello, I am very good
        # """
        reference_text = gen_refence_text()
        
        # 執行發音評估
        assessment_result = perform_pronunciation_assessment(
            wav_filepath, 
            reference_text
        )
        
        return JSONResponse(
            content={
                "message": f"錄音已成功保存並轉換",
                "webm_file": webm_filename,
                "wav_file": wav_filename,
                "assessment": assessment_result
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"處理錄音時發生錯誤: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
