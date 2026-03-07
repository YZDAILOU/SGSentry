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
