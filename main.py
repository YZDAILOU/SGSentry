import os
import shutil
import clickhouse_connect
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader  # Ensure you have 'pip install pypdf'
from google import genai

# Import our modules
from agents.transcriber import AudioTranscriber
from agents.claim_agent import claim_agent, FactCheckerDeps, analyze_media_integrity, extract_image_text, extract_video_visual_claims
from agents.scorer import calculate_trust_score

load_dotenv()

app = FastAPI()

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
        
        # 2. Extract Text (Transcript or OCR)
        if "video" in content_type:
            transcriber = AudioTranscriber()
            transcript = await transcriber.transcribe(file_path)
            
            # NEW LOGIC: If the transcript is empty or says 'only music'
            if not transcript.strip() or "no spoken words" in transcript.lower():
                print("🔈 Audio is silent/music. Switching to Visual Extraction...")
                transcript = await extract_video_visual_claims(file_path)

        elif "image" in content_type:
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
        score = calculate_trust_score(analysis_data, media_analysis)

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