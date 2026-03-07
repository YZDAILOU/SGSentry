import os
import uuid
import json
import shutil
import clickhouse_connect
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from pypdf import PdfReader  # Ensure you have 'pip install pypdf'
from google import genai
import yt_dlp
from langfuse import Langfuse
# Import our modules
from agents.transcriber import AudioTranscriber
from agents.claim_agent import claim_agent, FactCheckerDeps, analyze_media_integrity, extract_image_text, extract_video_visual_claims, AnalysisResult
from agents.scorer import calculate_trust_score, generate_hex_metrics

load_dotenv()

app = FastAPI()

# Initialize Langfuse Client
# --- LangFuse Initialization ---

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    print("❌ ERROR: LANGFUSE_PUBLIC_KEY not found in environment!")

class UrlRequest(BaseModel):
    media_url: str

async def download_video_from_url(url: str) -> str:
    """Downloads social media videos to a temporary file for visual analysis."""
    temp_filename = f"dl_{uuid.uuid4()}.mp4"
    ydl_opts = {
        # Ensuring we get a format Gemini likes (mp4)
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': temp_filename,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return temp_filename

def log_status(claim_id, status, transcript, details):
    deps = FactCheckerDeps()
    # Explicitly naming columns prevents "Unrecognized column" or "Offset" errors
    query = """
        INSERT INTO claim_history (claim_id, transcript, status, details) 
        VALUES 
    """
    try:
        deps.ch_client.insert(
            'claim_history', 
            [[claim_id, transcript, status, details]], 
            column_names=['claim_id', 'transcript', 'status', 'details']
        )
    except Exception as e:
        print(f"❌ DB Insert Error: {e}")

# --- ClickHouse Setup Logic ---

# def init_clickhouse():
#     """Check if DB is populated, if not, extract PDF and upload."""
#     try:
#         # 1. Connect to ClickHouse
#         client = clickhouse_connect.get_client(
#             host=os.getenv('CH_HOST', 'localhost'),
#             port=int(os.getenv('CH_PORT', 8443)),
#             username=os.getenv('CH_USER', 'default'),
#             password=os.getenv('CH_PASS', ''),
#             secure=True,
#             connect_timeout=30
#         )

#         print(client.query("SELECT 1"))

#         # 2. Create Table if not exists
#         client.command('''
#             CREATE TABLE IF NOT EXISTS sg_policies (
#                 filename String,
#                 content String,
#                 created_at DateTime DEFAULT now()
#             ) ENGINE = MergeTree()
#             ORDER BY created_at
#         ''')

#         # 3. Check if table has data
#         result = client.query('SELECT count() FROM sg_policies')
#         count = result.result_set[0][0]

#         if count == 0:
#             print("📭 ClickHouse empty. Extracting sg_policies.pdf...")
#             pdf_path = os.path.join("data", "sg_policies.pdf")
            
#             if os.path.exists(pdf_path):
#                 reader = PdfReader(pdf_path)
#                 for page in reader.pages:
#                     print("policies extracted: " , page.extract_text())
#                     client.insert('sg_policies', [[ 'sg_policies.pdf', page.extract_text() ]], 
#                              column_names=['filename', 'content'])
                
#                 print("✅ ClickHouse populated successfully.")
#             else:
#                 print("⚠️ Warning: data/sg_policies.pdf not found. Skipping population.")
#         else:
#             print(f"✅ ClickHouse already contains {count} records. Skipping extraction.")

#     except Exception as e:
#         print(f"❌ ClickHouse Init Error: {e}")

def init_clickhouse():
    """Extract PDF, convert to embeddings, and store in ClickHouse."""
    try:
        # 1. Connect to ClickHouse
        client = clickhouse_connect.get_client(
            host=os.getenv('CH_HOST', 'localhost'),
            port=int(os.getenv('CH_PORT', 8443)),
            username=os.getenv('CH_USER', 'default'),
            password=os.getenv('CH_PASS', ''),
            secure=True,
            connect_timeout=30
        )

        # 2. Create Table with Embedding Column
        # Note: Array(Float32) is how we store vectors in ClickHouse
        client.command('''
            CREATE TABLE IF NOT EXISTS sg_policies (
                filename String,
                content String,
                embedding Array(Float32),
                created_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY created_at
        ''')

        # 3. Check if table has data
        result = client.query('SELECT count() FROM sg_policies')
        count = result.result_set[0][0]

        if count == 0:
            print("📭 ClickHouse empty. Starting Embedding Process...")
            pdf_path = os.path.join("data", "sg_policies.pdf")
            
            if os.path.exists(pdf_path):
                # Initialize Gemini Client for embeddings
                # Assuming you are using the new google-genai SDK
                gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                
                reader = PdfReader(pdf_path)
                data_to_insert = []

                for i, page in enumerate(reader.pages):
                    text = page.extract_text().strip()
                    if not text:
                        continue
                    
                    print(f"Processing page {i+1}/{len(reader.pages)}...")

                    # Generate Embedding using Gemini
                    # model="text-embedding-004" is the current standard
                    embed_res = gemini_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text
                    )
                    vector = embed_res.embeddings[0].values

                    # Prepare row for ClickHouse
                    data_to_insert.append(['sg_policies.pdf', text, vector])

                # 4. Batch Insert
                if data_to_insert:
                    client.insert(
                        'sg_policies', 
                        data_to_insert, 
                        column_names=['filename', 'content', 'embedding']
                    )
                    print(f"✅ ClickHouse populated with {len(data_to_insert)} embedded pages.")
            else:
                print("⚠️ Warning: data/sg_policies.pdf not found.")
        
        else:
            print(f"✅ ClickHouse already contains {count} records. Skipping.")

        # 5. Create Claim History Table for Analytics
        client.command('''
            CREATE TABLE IF NOT EXISTS claim_history (
                claim_id String,
                transcript String,
                status Enum('Unverified' = 1, 'Processing' = 2, 'Verified' = 3, 'Debunked' = 4),
                event_time DateTime DEFAULT now(),
                details String
            ) ENGINE = MergeTree() ORDER BY event_time
        ''')

    except Exception as e:
        print(f"❌ ClickHouse Init Error: {e}")

# Run the initialization on startup
@app.on_event("startup")
async def startup_event():
    init_clickhouse()

# --- End ClickHouse Setup ---

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
async def analyze_media(
    file: Optional[UploadFile] = File(None), 
    request_data: Optional[str] = Form(None)
):
    """Receives video/image or URL, extracts text, checks claims, and analyzes for deepfakes."""
    file_path = ""
    content_type = ""
    
    try:
        # --- Handle File vs URL ---
        if file:
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            content_type = file.content_type or ""
        elif request_data:
            data = json.loads(request_data)
            file_path = await download_video_from_url(data['media_url'])
            content_type = "video/mp4" # Assume video for social links

        if not file_path or not os.path.exists(file_path):
            return {"error": "Failed to retrieve media content"}

        # 2. Extract Text (Transcript or OCR)
        transcript = ""
        if "video" in content_type:
            transcriber = AudioTranscriber()
            transcript = await transcriber.transcribe(file_path)
            
            # TRIGGER VISUAL ANALYSIS: If transcript is empty or music-only
            if not transcript.strip() or "music" in transcript.lower():
                print("Detected music-only audio. Switching to visual analysis...")
                # Use your new fallback tool
                transcript = await extract_video_visual_claims(file_path)

        elif "image" in content_type:
            transcript = await extract_image_text(file_path)

        # 3. Analyze Media for Deepfakes (Gemini)
        media_analysis = await analyze_media_integrity(file_path)

        # 4. Analyze Claims (Agent Loop)
        deps = FactCheckerDeps()
        
        try:
            # Get the prompt template from Langfuse UI
            langfuse_prompt = langfuse_client.get_prompt("default_system")
            
            # This injects the transcript into your {{transcript}} tag in Langfuse
            # If your Langfuse prompt doesn't have {{transcript}}, it will fail
            instruction = langfuse_prompt.compile(transcript=transcript)
            
            # Safety Check: Ensure the transcript is actually in the prompt.
            # If the Langfuse template was just "You are a bot", the model won't see the data.
            if transcript and transcript not in instruction:
                instruction += f"\n\nTRANSCRIPT:\n{transcript}"
        except Exception as e:
            print(f"⚠️ Langfuse Fetch Failed: {e}")
            instruction = f"Analyze this content for factual claims: {transcript}"

        result = await claim_agent.run(
            instruction,
            deps=deps
        )
        analysis_data = result.output or AnalysisResult(summary="No claims found", claims=[], hallucination_risk="Low")

        # 5. Score
        score = calculate_trust_score(analysis_data, media_analysis)
        metrics = generate_hex_metrics(analysis_data, media_analysis)

        # Determine status based on score
        status = "Unverified"
        if score >= 70: status = "Verified"
        elif score <= 30: status = "Debunked"

        # 6. Save to ClickHouse
        claim_id = str(uuid.uuid4())
        log_status(claim_id, status, transcript, analysis_data.summary)

        return {
            "transcript": transcript,
            "score": score,
            "metrics": metrics,
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