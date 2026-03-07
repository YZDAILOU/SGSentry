from agents.claim_agent import AnalysisResult

def calculate_trust_score(analysis: AnalysisResult) -> int:
    """
    Calculates a 0-100 trust score based on claims analysis.
    """
    if not analysis.claims:
        return 50  # Neutral if no claims found

    score = 50  # Start neutral

    for claim in analysis.claims:
        # 1. Check Verification Status
        for v in claim.verification:
            if "False" in v.status or "Debunked" in v.status or "Pants on Fire" in v.details:
                score -= 30
            elif "True" in v.status or "Verified" in v.status:
                score += 20
            elif "Mixture" in v.status:
                score -= 10

        # 2. Check Policy Alignment (Heuristic based on agent output)
        if claim.policy_context and "No relevant" not in claim.policy_context:
            # If policy context supports it (simplified logic)
            score += 5

    # 3. Hallucination Risk Penalty
    if analysis.hallucination_risk == "High":
        score -= 20
    elif analysis.hallucination_risk == "Low":
        score += 10

    # Clamp score
    return max(0, min(100, score))
