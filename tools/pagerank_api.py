import os
import httpx
from typing import Dict, Any

class PageRankAPI:
    def __init__(self):
        self.api_key = os.getenv("OPEN_PAGERANK_KEY")
        self.base_url = "https://openpagerank.com/api/v1.0"
        self.headers = {"API-OPR": self.api_key} if self.api_key else {}

    async def get_pagerank(self, domain: str) -> Dict[str, Any]:
        """
        Fetches PageRank metrics from Open PageRank API.
        """
        if not self.api_key:
            # Return a neutral/mock response if API key is missing
            return {
                "domain": domain,
                "page_rank_decimal": 0,
                "rank": "Unknown",
                "error": "API Key missing"
            }

        url = f"{self.base_url}/getPageRank"
        params = {"domains[]": domain}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, headers=self.headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                
                if "response" in data and isinstance(data["response"], list) and len(data["response"]) > 0:
                    result = data["response"][0]
                    return {
                        "domain": result.get("domain", domain),
                        "page_rank_decimal": result.get("page_rank_decimal", 0),
                        "rank": result.get("rank", "Unknown")
                    }
                return {"domain": domain, "error": "No data found", "page_rank_decimal": 0}
            except Exception as e:
                print(f"PageRank API Error: {e}")
                return {"domain": domain, "page_rank_decimal": 0, "error": str(e)}