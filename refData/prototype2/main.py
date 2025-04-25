from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydub import AudioSegment
import os
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from typing import Optional
import logging
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from faster_whisper import WhisperModel
import uvicorn

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# 加载环境变量
load_dotenv()

# 全局变量声明
whisper_model = None
llm = None
speech_config = None

# 环境变量配置
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SERVICE_REGION")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
LLM_MODEL = os.getenv("LLM_MODEL")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SPELL_CORRECT_PROMPT = '''
role:you are a perfect english spelling checker.
task:
1.please do spelling checking for the following senteces.
2.Do not change the style, form, and structure of the sentences.
3.only return the corrected sentences
sentences:\n{0}\n
'''

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, llm, speech_config
    try:
        # 初始化语音识别模型
        logging.info(f"初始化whisper model:{WHISPER_MODEL}....")
        whisper_model = WhisperModel(
            WHISPER_MODEL, 
            device="cpu", 
            compute_type="int8",
            download_root=WHISPER_MODEL_PATH, 
            local_files_only=True
        )
        
        # 初始化LLM
        logging.info(f"初始化LLM:{LLM_MODEL}....")
        llm = OllamaLLM(model=LLM_MODEL)
        
        # 初始化Azure语音配置
        logging.info("初始化Speech Config....")
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        
        yield
    except Exception as e:
        logging.error(f"初始化过程中出错: {e}")
        raise RuntimeError(f"初始化失败: {e}")
    finally:
        logging.info("应用即将关闭，清理资源...")

app = FastAPI(lifespan=lifespan)

# 中间件配置
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secure-secret-key"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ================= 对话管理路由 =================
@app.get("/", response_class=HTMLResponse)
async def conversation_setup(request: Request):
    """提供对话设置页面"""
    return templates.TemplateResponse("conversation.html", {"request": request})

@app.post("/start-chat")
async def start_chat(
    request: Request,
    topic: str = Form(...),
    level: str = Form(...),
    segmentation: str = Form(...),
    scenario: str = Form(...)
):
    """开始聊天会话"""
    request.session["conversation"] = {
        "topic": topic,
        "level": level,
        "segmentation": segmentation,
        "scenario": scenario
    }
    return RedirectResponse(url="/estimate", status_code=303)

@app.get("/estimate", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """聊天界面"""
    conversation = request.session.get("conversation", {})
    if not conversation:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(
        "estimate.html",
        {
            "request": request,
            "topic": conversation.get("topic"),
            "level": conversation.get("level"),
            "segmentation": conversation.get("segmentation"),
            "scenario": conversation.get("scenario")
        }
    )

# ================= 语音处理路由 =================
@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    """处理音频上传"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存原始WebM文件
        webm_filename = f"recording_{timestamp}.webm"
        webm_filepath = os.path.join(UPLOAD_DIR, webm_filename)
        with open(webm_filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # 转换为WAV格式
        wav_filename = f"recording_{timestamp}.wav"
        wav_filepath = os.path.join(UPLOAD_DIR, wav_filename)
        audio_segment = AudioSegment.from_file(webm_filepath)
        audio_segment.export(wav_filepath, format="wav")
        
        # 生成参考文本
        reference_text = gen_refence_text(wav_filepath)
        
        # 执行发音评估
        assessment_result = perform_pronunciation_assessment(
            wav_filepath,
            reference_text
        )
        
        return JSONResponse(
            content={
                "message": "录音处理成功",
                "webm_file": webm_filename,
                "wav_file": wav_filename,
                "ref_txt": reference_text,
                "assessment": assessment_result
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"处理录音时发生错误: {str(e)}"},
            status_code=500
        )

# ================= 工具函数 =================
def perform_pronunciation_assessment(audio_file: str, reference_text: str) -> dict:
    """执行发音评估"""
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True
    )
    pronunciation_config.enable_prosody_assessment()
    
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        language="en-US"
    )
    
    pronunciation_config.apply_to(speech_recognizer)
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        return {
            "accuracy_score": pronunciation_result.accuracy_score,
            "pronunciation_score": pronunciation_result.pronunciation_score,
            "completeness_score": pronunciation_result.completeness_score,
            "fluency_score": pronunciation_result.fluency_score,
            "prosody_score": pronunciation_result.prosody_score,
            "words": [{
                "word": word.word,
                "accuracy_score": word.accuracy_score,
                "error_type": word.error_type
            } for word in pronunciation_result.words]
        }
    else:
        raise Exception(f"语音识别失败: {result.reason}")

def gen_refence_text(audio_file: str):
    """生成参考文本"""
    segments, info = whisper_model.transcribe(audio_file, beam_size=5, language="en")
    trans_txt = " ".join(segment.text for segment in segments)
    ref_txt = llm.invoke(SPELL_CORRECT_PROMPT.format(trans_txt))
    return ref_txt

# ================= API端点 =================
@app.get("/api/session")
async def get_session(request: Request):
    """获取会话数据"""
    return request.session.get("conversation", {})

@app.post("/api/chat")
async def chat_message(request: Request):
    """处理聊天消息"""
    try:
        data = await request.json()
        message = data.get("message", "")
        
        # 实际应用中应集成AI服务
        response_message = ("Excellent timing! We just received the latest RTX 5090 GPU" 
                           if "graphics card" in message.lower() 
                           else "How can I help you today?")
        
        return {
            "message": response_message,
            "sender": "Amber"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理消息时发生错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
