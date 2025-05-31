from fastapi import FastAPI, UploadFile, File, Form, Request
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

# Add new prompt for chat
CHAT_PROMPT='''
    role: You are a friendly English conversation partner.
    task:
    1. Read the user's speech transcription.
    2. Respond naturally as if in a real conversation.
    3. Keep responses concise and engaging.
    4. Encourage further discussion.
    user's speech:\n{0}\n
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
                                    download_root=WHISPER_MODEL_PATH, local_files_only=True); #home
        # whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", local_files_only=False) #office
        
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

@app.post("/transcribe-chunk")
async def transcribe_chunk(audio: UploadFile = File(...), request: Request = None):
    try:
        # Remove VAD header check that was blocking audio processing
        # has_valid_speech = request.headers.get('X-Has-Valid-Speech', 'false').lower() == 'true'
        # audio_format = request.headers.get('X-Audio-Format', 'unknown')
        # chunk_count = request.headers.get('X-Chunk-Count', '0')
        
        logging.info(f"Received audio file: {audio.filename}, content_type: {audio.content_type}")
        
        # Remove this check that was filtering out audio
        # if not has_valid_speech:
        #     logging.info("Client indicates no valid speech detected, skipping processing")
        #     return JSONResponse(
        #         content={
        #             "transcription": "",
        #             "chat_response": "",
        #             "info": "No valid speech detected"
        #         },
        #         status_code=200
        #     )
        
        # Save the audio chunk temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        chunk_filename = f"chunk_{timestamp}.webm"
        chunk_filepath = os.path.join(UPLOAD_DIR, chunk_filename)
        wav_filename = f"chunk_{timestamp}.wav"
        wav_filepath = os.path.join(UPLOAD_DIR, wav_filename)
        
        try:
            # Save WebM file
            with open(chunk_filepath, "wb") as buffer:
                content = await audio.read()
                logging.info(f"Audio content size: {len(content)} bytes")
                if len(content) == 0:
                    raise Exception("Empty audio file received")
                buffer.write(content)
            
            logging.info(f"Saved audio file to: {chunk_filepath}")
            
            # Validate the audio file format
            file_info = validate_audio_file(chunk_filepath)
            logging.info(f"Audio file validation: {file_info}")
            
            if not file_info["valid"]:
                # Log more details for debugging
                logging.warning(f"File validation failed but continuing with processing. Details: {file_info}")
                # Don't fail here - try to process anyway as the validation might be too strict
                # raise Exception(f"Invalid audio file format. Detected: {file_info.get('format', 'unknown')}. Header: {file_info.get('header', 'N/A')}")
            
            # Convert to WAV using multiple strategies based on detected format
            audio_segment = None
            conversion_error = None
            detected_format = file_info.get("format", "unknown")
            confidence = file_info.get("confidence", "low")
            
            # Strategy selection based on detected format and confidence
            if "webm" in detected_format or "matroska" in detected_format:
                if confidence == "high":
                    # High confidence WebM files - try direct methods first
                    strategies = [
                        ("Direct WebM with Opus", {"format": "webm"}),
                        ("WebM with Opus codec", {"format": "webm", "codec": "opus"}),
                        ("WebM auto-detection", {}),
                    ]
                else:
                    # Lower confidence or unknown variants - try more aggressive methods
                    strategies = [
                        ("WebM auto-detection", {}),
                        ("Force WebM with strict mode off", {"format": "webm", "parameters": ["-strict", "-2"]}),
                        ("WebM with Opus codec", {"format": "webm", "codec": "opus"}),
                        ("Force WebM container with copy", {"parameters": ["-f", "webm", "-c:a", "copy", "-strict", "-2"]}),
                        ("Force WebM with Opus transcode", {"parameters": ["-f", "webm", "-c:a", "libopus", "-strict", "-2"]}),
                        ("Generic WebM fallback", {"format": "webm", "parameters": ["-strict", "-2", "-ignore_unknown"]}),
                    ]
            elif "complete_recording" in detected_format:
                # For complete recordings, try WebM methods with more tolerance
                strategies = [
                    ("Auto-detection for complete recording", {}),
                    ("WebM for complete recording", {"format": "webm", "parameters": ["-strict", "-2"]}),
                    ("Force WebM container", {"parameters": ["-f", "webm", "-c:a", "libopus", "-strict", "-2"]}),
                    ("Raw decode with WebM hint", {"parameters": ["-f", "webm", "-strict", "-2", "-ignore_unknown"]}),
                ]
            elif "mp4" in detected_format:
                strategies = [
                    ("Direct MP4", {"format": "mp4"}),
                    ("MP4 with AAC", {"format": "mp4", "parameters": ["-acodec", "aac"]}),
                ]
            elif "unknown_audio" in detected_format:
                # For files that look like audio but have unknown headers
                strategies = [
                    ("Auto-detection", {}),
                    ("Force WebM (common browser format)", {"format": "webm", "parameters": ["-strict", "-2"]}),
                    ("Force container as WebM", {"parameters": ["-f", "webm", "-c:a", "copy", "-strict", "-2"]}),
                    ("Raw audio stream", {"parameters": ["-f", "webm", "-acodec", "libopus", "-strict", "-2"]}),
                ]
            else:
                # Fallback strategies for unknown formats
                strategies = [
                    ("Auto-detection", {}),
                    ("Force WebM (browser default)", {"format": "webm", "parameters": ["-strict", "-2"]}),
                    ("Raw audio with WebM container", {"parameters": ["-f", "webm", "-acodec", "libopus", "-strict", "-2"]}),
                    ("Ignore format errors", {"parameters": ["-f", "webm", "-strict", "-2", "-ignore_unknown"]}),
                ]
            
            # Try each strategy
            for strategy_name, strategy_params in strategies:
                try:
                    logging.info(f"Attempting {strategy_name}...")
                    audio_segment = AudioSegment.from_file(chunk_filepath, **strategy_params)
                    logging.info(f"Successfully loaded with {strategy_name}: duration={len(audio_segment)}ms")
                    break
                except Exception as e:
                    logging.warning(f"{strategy_name} failed: {e}")
                    conversion_error = str(e)
                    continue
            
            # If all strategies failed, try one final fallback
            if audio_segment is None:
                try:
                    logging.info("Attempting final fallback with auto-detection...")
                    audio_segment = AudioSegment.from_file(chunk_filepath)
                    logging.info(f"Final fallback successful: duration={len(audio_segment)}ms")
                except Exception as e:
                    logging.error(f"All conversion strategies failed. Final error: {e}")
                    raise Exception(f"Cannot process audio format '{detected_format}'. All conversion strategies failed. Last error: {conversion_error}")
            
            # Check if we got a valid audio segment
            if audio_segment is None or len(audio_segment) == 0:
                raise Exception("Audio conversion resulted in empty or invalid audio")
            
            # Normalize audio parameters
            logging.info(f"Original audio: channels={audio_segment.channels}, frame_rate={audio_segment.frame_rate}, duration={len(audio_segment)}ms")
            
            # Set audio parameters for Whisper (16kHz mono)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV with specific parameters for better compatibility
            audio_segment.export(
                wav_filepath,
                format="wav",
                parameters=[
                    "-ac", "1",  # mono
                    "-ar", "16000",  # sample rate
                    "-acodec", "pcm_s16le",  # 16-bit PCM
                    "-f", "wav"
                ]
            )
            
            logging.info(f"Successfully converted to WAV: {wav_filepath}")
            
            # Verify WAV file was created and has content
            if not os.path.exists(wav_filepath):
                raise Exception("WAV conversion failed - no output file created")
            
            wav_size = os.path.getsize(wav_filepath)
            if wav_size == 0:
                raise Exception("WAV conversion failed - empty output file")
            
            logging.info(f"WAV file size: {wav_size} bytes")
            
            # Transcribe using Whisper with enhanced error handling
            transcribed_text = ""
            try:
                logging.info("Starting Whisper transcription...")
                segments, info = whisper_model.transcribe(
                    wav_filepath, 
                    beam_size=5,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=1000,
                        speech_pad_ms=30,
                        threshold=0.005  # Lower threshold to be less aggressive (was 0.015)
                    ),
                    word_timestamps=True,
                    condition_on_previous_text=False  # Improve accuracy for short segments
                )
                
                transcribed_text = " ".join([segment.text.strip() for segment in segments if segment.text.strip()])
                logging.info(f"Transcription successful: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'")
                
            except Exception as whisper_error:
                logging.error(f"Whisper transcription failed: {whisper_error}")
                # Fallback transcription without VAD
                try:
                    logging.info("Attempting fallback transcription without VAD...")
                    segments, info = whisper_model.transcribe(
                        wav_filepath, 
                        beam_size=3,  # Reduce beam size for faster processing
                        language="en",
                        condition_on_previous_text=False
                    )
                    transcribed_text = " ".join([segment.text.strip() for segment in segments if segment.text.strip()])
                    logging.info(f"Fallback transcription successful: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}'")
                except Exception as fallback_error:
                    logging.error(f"Fallback transcription also failed: {fallback_error}")
                    transcribed_text = ""
            
            # Generate chat response only if there's transcribed text
            chat_response = ""
            if transcribed_text.strip():
                try:
                    logging.info("Generating chat response...")
                    chat_response = llm.invoke(CHAT_PROMPT.format(transcribed_text))
                    logging.info(f"Chat response generated: '{chat_response[:100]}{'...' if len(chat_response) > 100 else ''}'")
                except Exception as llm_error:
                    logging.error(f"LLM response generation failed: {llm_error}")
                    chat_response = "Sorry, I couldn't process your message right now. Please try again."
            else:
                logging.info("No transcribed text, skipping chat response generation")
            
            # Clean up temporary files
            try:
                if os.path.exists(chunk_filepath):
                    os.remove(chunk_filepath)
                if os.path.exists(wav_filepath):
                    os.remove(wav_filepath)
                logging.info("Temporary files cleaned up")
            except Exception as cleanup_error:
                logging.warning(f"Error cleaning up temporary files: {cleanup_error}")
            
            return JSONResponse(
                content={
                    "transcription": transcribed_text,
                    "chat_response": chat_response
                },
                status_code=200
            )
            
        except Exception as processing_error:
            logging.error(f"Error processing audio: {str(processing_error)}")
            # Clean up files even if processing failed
            try:
                if os.path.exists(chunk_filepath):
                    os.remove(chunk_filepath)
                if os.path.exists(wav_filepath):
                    os.remove(wav_filepath)
            except:
                pass
            raise processing_error
            
    except Exception as e:
        logging.error(f"Error in transcribe_chunk: {str(e)}")
        
        # Determine if this is a processing error that should be shown to user
        error_message = str(e)
        show_error_to_user = True
        
        # Don't show these common processing errors to user as they're not actionable
        silent_errors = [
            "Empty audio file received",
            "Audio chunk too small",
            "No valid speech detected",
            "Audio conversion resulted in empty",
            "Cannot process audio format"
        ]
        
        for silent_error in silent_errors:
            if silent_error in error_message:
                show_error_to_user = False
                break
        
        response_content = {
            "error": error_message if show_error_to_user else "",
            "transcription": "",
            "chat_response": ""
        }
        
        # Only add generic error message for actual processing failures
        if show_error_to_user and "transcription failed" in error_message.lower():
            response_content["chat_response"] = "Sorry, there was an issue processing your audio. Please try speaking again."
        
        return JSONResponse(
            content=response_content,
            status_code=200  # Return 200 to prevent frontend errors
        )

def validate_audio_file(file_path: str) -> dict:
    """Validate and inspect audio file format with enhanced WebM detection"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(128)  # Read more bytes for better detection
        
        file_size = os.path.getsize(file_path)
        
        # Enhanced WebM/Matroska signature detection
        # Standard EBML header starts with 0x1A45DFA3
        if header.startswith(b'\x1a\x45\xdf\xa3'):
            return {"format": "webm/matroska", "valid": True, "size": file_size, "confidence": "high"}
        
        # Check for EBML signature at different offsets (sometimes there's padding)
        for i in range(min(32, len(header) - 4)):
            if header[i:i+4] == b'\x1a\x45\xdf\xa3':
                return {"format": "webm/matroska", "valid": True, "size": file_size, "confidence": "high"}
        
        # Enhanced WebM detection - look for WebM-specific elements
        webm_indicators = [
            b'\x42\x82',  # DocType element
            b'\x18\x53\x80\x67',  # Segment element  
            b'\x1f\x43\xb6\x75',  # Cluster element
            b'webm',  # DocType string
            b'matroska',  # Alternative DocType string
        ]
        
        webm_score = 0
        for indicator in webm_indicators:
            if indicator in header:
                webm_score += 1
        
        if webm_score >= 2:
            return {"format": "webm/matroska", "valid": True, "size": file_size, "confidence": "medium"}
        elif webm_score >= 1:
            return {"format": "webm/possible", "valid": True, "size": file_size, "confidence": "low"}
        
        # For files that are likely complete recordings (larger size, from our new approach)
        # Check filename pattern for complete recordings vs chunks
        filename = os.path.basename(file_path)
        is_complete_recording = filename.startswith('recording_') or file_size > 50000  # 50KB+
        
        # More lenient validation for complete recordings
        if is_complete_recording and 10000 <= file_size <= 100*1024*1024:  # 10KB to 100MB
            return {"format": "webm/complete_recording", "valid": True, "size": file_size, "confidence": "medium"}
        
        # Check for other common audio formats
        if header[4:8] in [b'ftyp', b'mdat', b'moov']:
            return {"format": "mp4", "valid": True, "size": file_size, "confidence": "high"}
        
        if header.startswith(b'RIFF') and header[8:12] == b'WAVE':
            return {"format": "wav", "valid": True, "size": file_size, "confidence": "high"}
        
        if header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3') or header.startswith(b'\xff\xf2'):
            return {"format": "mp3", "valid": True, "size": file_size, "confidence": "high"}
        
        # For files that might be audio but don't have clear signatures
        if 5000 <= file_size <= 10*1024*1024:  # Between 5KB and 10MB
            return {"format": "unknown_audio", "valid": True, "size": file_size, "confidence": "low", "header": header.hex()[:64]}
        
        return {"format": "unknown", "valid": False, "size": file_size, "confidence": "none", "header": header.hex()[:64]}
        
    except Exception as e:
        return {"format": "error", "valid": False, "error": str(e), "size": 0, "confidence": "none"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
