from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import os
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from faster_whisper import WhisperModel
import logging

###setup debug log setting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

'''*********環境變數設置*********'''
load_dotenv()
# Azure 語音服務配置
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SERVICE_REGION")

# local faster-whisper model setup
WHISPER_MODEL = os.getenv("WHISPER_MODEL");
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH");

# local llm setup
LLM_MODEL=os.getenv("LLM_MODEL");

#os.getenv("SPELL_CORRECT_PROMPT");
SPELL_CORRECT_PROMPT='''
    role:you are a perfect english spelling checker.
    task:
    1.please do spelling checking for the following senteces.
    2.Do not change the style, form, and structure of the sentences.
    3.only return the corrected sentences
    sentences:\n{0}\n
'''

# 確保錄音文件存儲目錄存在
UPLOAD_DIR = os.getenv("UPLOAD_DIR")
os.makedirs(UPLOAD_DIR, exist_ok=True)
'''************End*************'''
# fastapi startup setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    global llm
    global speech_config
    try:
        # Load the whisper and llm models
        logging.info(f"初始化whisper model:{WHISPER_MODEL}....")
        whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", 
                                    download_root=WHISPER_MODEL_PATH, local_files_only=True);
        
        logging.info(f"初始化whisper model:{WHISPER_MODEL}完成....")
        
        logging.info(f"初始化LLM:{LLM_MODEL}....")
        llm = OllamaLLM(model=LLM_MODEL);
        logging.info(f"初始化LLM:{LLM_MODEL}完成")
        
        #initialize speech service configs
        logging.info("Speech Config....")
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY, 
            region=SPEECH_REGION
        )
        yield # 應用啟動後，繼續執行其餘邏輯
    except Exception as e:
        logging.error(f"初始化過程中出錯: {e}")
        raise RuntimeError(f"初始化失敗: {e}")
    finally:
        # 如果需要，可以在這裡添加清理邏輯
        logging.info("應用即將關閉，清理資源...")
        

app = FastAPI(lifespan=lifespan)
# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")
# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/mainpage2.html")

def perform_pronunciation_assessment(audio_file: str, reference_text: str) -> dict:
    """執行發音評估"""
    # try:
    
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

def gen_refence_text(audio_file:str=None):
    global SPELL_CORRECT_PROMPT
    segments, info = whisper_model.transcribe(audio_file,  beam_size=5, language="en")
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    trans_txt = "";
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        trans_txt += segment.text
        
    ref_txt = llm.invoke(SPELL_CORRECT_PROMPT.format(trans_txt));
    print(f"the corrected transcribe text:\n{trans_txt}");
    return ref_txt
    

@app.post("/upload-audio")
async def upload_audio(
    audio: UploadFile = File(...),
    topic: str = Form(None),
    level: str = Form(None),
    segmentation: str = Form(None)
):
    # try:
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
    
    #進行取得參照文稿
    reference_text = gen_refence_text(audio_file=wav_filepath);
    
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
            "ref_txt": reference_text,
            "topic": topic,
            "level": level,
            "segmentation": segmentation,
            "assessment": assessment_result
        },
        status_code=200
    )
    # except Exception as e:
    #     return JSONResponse(
    #         content={"message": f"處理錄音時發生錯誤: {str(e)}"},
    #         status_code=500
    #     )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
