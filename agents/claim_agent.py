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
from openai import OpenAI
from langfuse import Langfuse

# Import Tools
from tools.pagerank_api import PageRankAPI
from tools.google_fact import GoogleFactCheckAPI

# 1. Load environment variables at the very top
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
langfuse = Langfuse()

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
        # Initialize OpenAI Client for Logic Auditing
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
# --- LangFuse Initialization ---

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

print("Langfuse: ", langfuse_client)
# 1. Fetch the prompt from Langfuse
try:
    # This retrieves the versioned prompt you created in the UI
    langfuse_prompt = langfuse_client.get_prompt("default_system")
    # Extract the actual text string from the Langfuse object
    system_instructions = langfuse_prompt.get_langchain_prompt() 
except Exception as e:
    # Fallback in case Langfuse is unreachable
    system_instructions = "You are a factual analysis agent. Extract and verify claims."

# 2. Pass it into the Agent
claim_agent = Agent(
    model_instance,
    deps_type=FactCheckerDeps,
    output_type=AnalysisResult,
    system_prompt=system_instructions 
)

#Google Fact Check API tool to verify claims against existing fact checks. This provides a quick check for widely debunked or verified claims, and the logic auditor can then analyze any discrepancies in depth.
@claim_agent.tool
async def check_google_facts(ctx: RunContext[FactCheckerDeps], query: str) -> str:
    """
    CRITICAL TOOL: Search the Google Fact Check API to find existing fact-checks for a claim.
    Use this for EVERY factual claim you extract.
    Input: The complete claim text (e.g., 'Singapore's birth rate is below 1.0')
    Returns: Publisher ratings (Debunked/Verified/Mixed) from credible fact-checking organizations.
    This is your primary verification source for global facts.
    """
    results = await ctx.deps.google.search(query)
    if not results:
        return "No existing fact checks found."

    summary = []
    for claim in results[:2]: # Limit to top 2
        review = claim.get("claimReview", [{}])[0]
        publisher = review.get("publisher", {}).get("name", "Unknown")
        rating = review.get("textualRating", "Unknown")
        summary.append(f"{publisher} rated it '{rating}'")
    print("Finished google fact")
    return "google fact; ".join(summary)

#PageRank tool to evaluate the credibility of sources mentioned in claims. This can help the logic auditor weigh evidence based on source reliability.
@claim_agent.tool
async def check_domain_authority(ctx: RunContext[FactCheckerDeps], domain: str) -> str:
    """
    CREDIBILITY TOOL: Evaluate the authority and trustworthiness of a domain using PageRank scoring.
    Use this whenever a claim references a specific website (e.g., 'according to straitstimes.com').
    Input: Domain name (e.g., 'straitstimes.com')
    Returns: PageRank decimal score and ranking position. Higher scores = more authoritative sources.
    This helps you weight evidence based on source reliability.
    """
    # Ensure get_pagerank exists in your PageRankAPI class
    data = await ctx.deps.pagerank.get_pagerank(domain)
    print("Finished pagerank")
    return f"Domain: {domain}, PageRank: {data.get('page_rank_decimal')}, Rank: {data.get('rank')}"

# The consult_policies tool is enhanced to perform a semantic vector search against the ClickHouse database of Singapore government policies. This allows the agent to retrieve highly relevant policy excerpts that can be used as authoritative evidence when verifying claims related to Singapore regulations. The logic auditor can then analyze how well the claim aligns with official policies, especially in cases where Google Fact Check results are inconclusive or conflicting.
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

    print("Finished policy consultation")
    return "\n\n---\n\n".join(context_blocks)

# --- Video Analysis Logic ---

async def analyze_media_integrity(media_path: str) -> VideoAnalysisResult:
    """Uploads video/image to Gemini to check for AI generation/deepfakes."""
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")

    # Using the latest genai client
    client = genai.Client(api_key=api_key)

    # print(f"Uploading {media_path} for AI detection...")
    media_file = client.files.upload(file=media_path)

    # Wait for processing
    while media_file.state.name == "PROCESSING":
        time.sleep(2)
        media_file = client.files.get(name=media_file.name)

    if media_file.state.name == "FAILED":
        raise RuntimeError("Media processing failed.")
        
    media_type = "video" if media_path.endswith(('.mp4', '.mov', '.avi')) else "image"
    
    try:
        prompt_tmpl = langfuse.get_prompt("media_integrity_analyst")
        prompt = prompt_tmpl.compile(media_type=media_type)
    except Exception as e:
        # print(f"⚠️ Langfuse Prompt Error: {e}")
        prompt = f"You are an AI forensics expert. Analyze this {media_type} for signs of manipulation. " \
                 "Check for unnatural skin textures, flickering, or temporal inconsistencies. " \
                 "Return a structured assessment of the likelihood that this media is synthetic."
    
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
    
    try:
        prompt_tmpl = langfuse.get_prompt("visual_claim_extractor")
        prompt = prompt_tmpl.compile(video_frames_description="the video content")
    except Exception as e:
        # print(f"⚠️ Langfuse Prompt Error: {e}")
        prompt = (
            "The audio for this video is unavailable or contains only music. Your task is to 'watch' the visuals. "
            "Extract all text overlays (OCR). Identify factual assertions made via captions or infographics."
        )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[media_file, prompt]
    )
    return response.text
