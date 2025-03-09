from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pydub import AudioSegment
import os
from datetime import datetime

app = FastAPI()

# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

# 確保錄音文件存儲目錄存在
UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return RedirectResponse(url="/static/record.html")

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
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
        
        return JSONResponse(
            content={
                "message": f"錄音已成功保存並轉換",
                "webm_file": webm_filename,
                "wav_file": wav_filename
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
