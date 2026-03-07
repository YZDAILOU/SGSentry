import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our modules
from agents.transcriber import AudioTranscriber

load_dotenv()

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the basic frontend."""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found. Please create index.html</h1>"

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """Receives video/image, extracts text, checks claims, and analyzes for deepfakes."""
    file_path = f"temp_{file.filename}"
    
    try:
        # 1. Save Uploaded File
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract Text (Transcript or OCR)
        content_type = file.content_type or ""
        transcript = ""
        
        if "video" in content_type:
            # Use Whisper for video audio
            transcriber = AudioTranscriber()
            transcript = await transcriber.transcribe(file_path)
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    print("Starting Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
