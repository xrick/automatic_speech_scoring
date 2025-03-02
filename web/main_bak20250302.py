from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import os
from datetime import datetime

app = FastAPI()

# 設置靜態文件目錄，並指定 record.html 為預設頁面
app.mount("/static", StaticFiles(directory="static"), name="static")

# 確保錄音文件存儲目錄存在
UPLOAD_DIR = "recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    # 將根路徑重定向到 record.html
    return RedirectResponse(url="/static/record.html")

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        # 生成唯一的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.webm"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        # 保存文件
        with open(filepath, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        return JSONResponse(
            content={
                "message": f"錄音已成功保存為 {filename}",
                "filename": filename
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"message": f"錄音保存失敗: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
