from agents.claim_agent import AnalysisResult, VideoAnalysisResult

def calculate_trust_score(analysis: AnalysisResult, video_meta: VideoAnalysisResult) -> int:
    score = 50 

    # Penalty for AI Generation / Deepfakes
    if video_meta.is_ai_generated:
        # Scale deduction by confidence
        deduction = (video_meta.confidence_score * 0.5) 
        score -= int(deduction)
        
    # Standard claim scoring logic
    if analysis.claims:
        for claim in analysis.claims:
            for v in claim.verification:
                if "Debunked" in v.status: score -= 25
                elif "Verified" in v.status: score += 15

    return max(0, min(100, score))

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
