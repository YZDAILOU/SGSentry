import os
import time
from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic_ai import Agent, RunContext
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel
import clickhouse_connect


# Import Tools
from tools.pagerank_api import PageRankAPI
from tools.google_fact import GoogleFactCheckAPI

# 1. Load environment variables at the very top
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# --- Data Models ---
class VerificationData(BaseModel):
    source: str
    status: str  # e.g., "Debunked", "Verified", "Unknown"
    details: str

class Claim(BaseModel):
    text: str = Field(..., description="The extracted factual claim.")
    is_fact: bool = Field(..., description="True if it is a verifiable fact, False if opinion.")
    verification: List[VerificationData] = Field(default_factory=list)
    policy_context: Optional[str] = Field(None, description="Relevant info from SG policies PDF.")

class AnalysisResult(BaseModel):
    summary: str
    claims: List[Claim]
    hallucination_risk: str = Field(..., description="Low, Medium, or High based on verification.")

class VideoAnalysisResult(BaseModel):
    is_ai_generated: bool = Field(..., description="True if the video shows signs of AI generation.")
    confidence_score: int = Field(..., description="0-100 confidence level.")
    visual_anomalies: List[str] = Field(..., description="List of visual artifacts found.")
    audio_sync_issues: str = Field(..., description="Analysis of lip-sync and audio alignment.")

# --- Dependencies Setup ---
class FactCheckerDeps:
    def __init__(self):
        self.pagerank = PageRankAPI()
        self.google = GoogleFactCheckAPI()
        # self.policy_text = self._load_policy_pdf()
        self.client = genai.Client(api_key=api_key)
        # Initialize ClickHouse connection
        self.ch_client = clickhouse_connect.get_client(
            host=os.getenv('CH_HOST', 'localhost'),
            port=int(os.getenv('CH_PORT', 8443)),
            username=os.getenv('CH_USER', 'default'),
            password=os.getenv('CH_PASS', ''),
            secure=True,
            connect_timeout=30
        )

    # def _load_policy_pdf(self) -> str:
    #     """Simple RAG: Load text from data/sg_policies.pdf"""
    #     path = os.path.join("data", "sg_policies.pdf")
    #     if not os.path.exists(path):
    #         return ""
    #     try:
    #         reader = PdfReader(path)
    #         text = ""
    #         for page in reader.pages:
    #             text += page.extract_text() + "\n"
    #         return text
    #     except Exception as e:
    #         print(f"Error reading PDF: {e}")
    #         return ""

# --- Agent Initialization ---

model_instance = GoogleModel('gemini-2.5-flash')

claim_agent = Agent(
    model_instance,  # Pass model instance directly
    deps_type=FactCheckerDeps,
    output_type=AnalysisResult,
    system_prompt=(
        "You are a Fact Checking Bot for Singapore context. "
        "1. Analyze the transcript. "
        "2. Extract key claims. "
        "3. Use tools to verify claims against Google Fact Check and PageRank for domain authority. "
        "4. Check the provided 'consult_policies' tool for local context. "
        "5. Return a structured report."
    )
)

# --- Agent Tools ---

@claim_agent.tool
async def check_google_facts(ctx: RunContext[FactCheckerDeps], query: str) -> str:
    """Search Google Fact Check API for a specific claim."""
    results = await ctx.deps.google.search(query)
    if not results:
        return "No existing fact checks found."
    
    summary = []
    for claim in results[:2]: # Limit to top 2
        review = claim.get("claimReview", [{}])[0]
        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating = review.get("textualRating", "Unknown")
        summary.append(f"{publisher} rated it '{rating}'")
    return "; ".join(summary)

@claim_agent.tool
async def check_domain_authority(ctx: RunContext[FactCheckerDeps], domain: str) -> str:
    """Check PageRank scoring if a website is mentioned."""
    # Ensure get_pagerank exists in your PageRankAPI class
    data = await ctx.deps.pagerank.get_pagerank(domain)
    return f"Domain: {domain}, PageRank: {data.get('page_rank_decimal')}, Rank: {data.get('rank')}"

@claim_agent.tool
async def consult_policies(ctx: RunContext[FactCheckerDeps], query: str) -> str:
    """
    Search across all Singapore government policy documents in ClickHouse 
    to verify claims against official regulations.
    """
    # 1. Generate the embedding for the user's specific question
    embedding_response = ctx.deps.client.models.embed_content(
        model="gemini-embedding-001", # Must match what you used to populate ClickHouse
        contents=query
    )
    query_vector = embedding_response.embeddings[0].values

    # 2. Perform Vector Search in ClickHouse
    # We use cosineDistance: smaller value = more similar
    search_query = """
        SELECT content, filename, cosineDistance(embedding, %(vec)s) AS score
        FROM sg_policies
        ORDER BY score ASC
        LIMIT 5
    """
    
    result = ctx.deps.ch_client.query(search_query, parameters={'vec': query_vector})
    
    if not result.result_rows:
        return "No relevant policy information found in the database."

    # 3. Format the results for the Agent
    context_blocks = []
    for row in result.result_rows:
        content, source, score = row
        context_blocks.append(f"Source: {source} (Similarity: {1-score:.2f})\nContent: {content}")

    return "\n\n---\n\n".join(context_blocks)

# --- Video Analysis Logic ---

async def analyze_media_integrity(media_path: str) -> VideoAnalysisResult:
    """Uploads video/image to Gemini to check for AI generation/deepfakes."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")
        
    # Using the latest genai client
    client = genai.Client(api_key=api_key)
    
    print(f"Uploading {media_path} for AI detection...")
    media_file = client.files.upload(file=media_path)
    
    # Wait for processing
    while media_file.state.name == "PROCESSING":
        time.sleep(2)
        media_file = client.files.get(name=media_file.name)
        
    if media_file.state.name == "FAILED":
        raise RuntimeError("Media processing failed.")
        
    prompt = "Analyze this media (video or image) for signs of AI generation (deepfakes, manipulation), such as unnatural artifacts, lighting inconsistencies, or lip-sync errors. Return JSON."
    
    # Note: Use 'gemini-2.5-pro' for better video analysis
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[media_file, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": VideoAnalysisResult
        }
    )
    
    return VideoAnalysisResult.model_validate_json(response.text)

async def extract_image_text(image_path: str) -> str:
    """Extracts text from an image using Gemini."""
    if not api_key: return ""
    client = genai.Client(api_key=api_key)
    
    img_file = client.files.upload(file=image_path)
    while img_file.state.name == "PROCESSING":
        time.sleep(1)
        img_file = client.files.get(name=img_file.name)
        
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img_file, "Extract all readable text from this image. If no text, describe the image content in detail."],
    )
    return response.text

async def extract_video_visual_claims(video_path: str) -> str:
    """Fallback tool that 'watches' the video when speech is missing."""
    client = genai.Client(api_key=api_key)
    media_file = client.files.upload(file=video_path)
    
    # Wait for processing (Gemini needs to index the frames)
    while media_file.state.name == "PROCESSING":
        time.sleep(2)
        media_file = client.files.get(name=media_file.name)
    
    prompt = (
        "This video has no spoken dialogue or only contains music. "
        "Analyze the visuals: extract all text overlays, captions, "
        "and describe the key factual events occurring in the scene."
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[media_file, prompt]
    )
    return response.text