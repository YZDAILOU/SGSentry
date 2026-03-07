import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our modules
from agents.transcriber import AudioTranscriber
from agents.claim_agent import claim_agent, FactCheckerDeps, analyze_media_integrity, extract_image_text
from agents.scorer import calculate_trust_score

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
            transcriber = AudioTranscriber() # this AudioTranscriber is failing
            transcript = await transcriber.transcribe(file_path)
            print("transcript check for video: ", transcript)
        elif "image" in content_type:
            # Use Gemini for Image OCR/Description
            transcript = await extract_image_text(file_path)

        # 3. Analyze Media for Deepfakes (Gemini)
        media_analysis = await analyze_media_integrity(file_path)

        # 4. Analyze Claims (Agent Loop)
        deps = FactCheckerDeps()
        result = await claim_agent.run(
            f"Analyze this transcript: {transcript}",
            deps=deps
        )
        analysis_data = result.output

        # 5. Score
        score = calculate_trust_score(analysis_data)

        return {
            "transcript": transcript,
            "score": score,
            "analysis": analysis_data.model_dump(),
            "video_analysis": media_analysis.model_dump()
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    print("Starting Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
