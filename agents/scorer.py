from agents.claim_agent import AnalysisResult, VideoAnalysisResult, FactCheckerDeps
from openai import OpenAI
import os
import json

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def calculate_trust_score(analysis: AnalysisResult, video_meta: VideoAnalysisResult) -> int:
    """
    Hybrid Trust Score System:
    1. Rule-based baseline scoring
    2. GPT reasoning refinement
    """

    score = 50
    verified_count = 0
    debunked_count = 0
    policy_hits = 0

    claim_summary = []

    if analysis.claims:
        for claim in analysis.claims:

            # Safety check
            if isinstance(claim, str):
                continue

            claim_text = claim.text
            verification_text = ""

            # Count verification signals
            if claim.verification:
                for v in claim.verification:

                    verification_text += f"{v.source}: {v.status}\n"

                    if "Debunked" in v.status:
                        score -= 20
                        debunked_count += 1

                    elif "Verified" in v.status:
                        score += 10
                        verified_count += 1

            # Policy alignment boost
            if claim.policy_context and "No relevant" not in claim.policy_context:
                score += 5
                policy_hits += 1

            claim_summary.append(
                f"""
Claim: {claim_text}
Verification:
{verification_text}
Policy Context:
{claim.policy_context}
"""
            )

    # Media integrity penalty
    if video_meta.is_ai_generated:
        deduction = int(video_meta.confidence_score * 0.4)
        score -= deduction

    # Clamp rule-based score
    score = max(0, min(100, score))

    # ----------------------------
    # GPT REASONING REFINEMENT
    # ----------------------------

    claims_text = "\n".join(claim_summary)

    prompt = f"""
You are a professional misinformation analyst.

Evaluate the credibility of the following media content.

Claims:
{claims_text}

Media Integrity:
AI Generated: {video_meta.is_ai_generated}
Confidence Score: {video_meta.confidence_score}

Initial heuristic score: {score}

Consider:
- fact-check verification results
- number of debunked vs verified claims
- reliability of evidence
- policy alignment
- possible AI manipulation

Return JSON only:

{{
"trust_score": number between 0 and 100
}}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        result = response.choices[0].message.content.strip()

        data = json.loads(result)

        ai_score = int(data["trust_score"])

        final_score = int((score * 0.6) + (ai_score * 0.4))

        return max(0, min(100, final_score))

    except Exception as e:
        # print("GPT scoring failed:", e)
        return score
    
    
def generate_hex_metrics(analysis: AnalysisResult, video_meta: VideoAnalysisResult) -> dict:
    """Calculates 6 specific metrics (0-100) for the hexagonal radar chart."""

    # 1. Policy Alignment (Check if any claim has valid policy context)
    policy_score = 0
    for claim in analysis.claims:
        if claim.policy_context and "No relevant" not in claim.policy_context:
            policy_score = 100
            break

    # 2. Fact-Check (Check for verified status)
    fact_score = 20
    if "Verified" in str(analysis.claims) or "True" in str(analysis.claims):
        fact_score = 100

    return {
        "Source Authority": 70, # Placeholder: Connect to PageRank tool if available
        "Policy Alignment": policy_score,
        "Media Integrity": int(100 - video_meta.confidence_score) if video_meta.is_ai_generated else 100,
        "Fact-Check": fact_score,
        "Cross-Reference": 70, # Static baseline for demo
        "Metadata": 85       # Static baseline for demo
    }
