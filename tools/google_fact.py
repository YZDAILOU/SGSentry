import os
import httpx
from typing import List, Dict, Any

class GoogleFactCheckAPI:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_FACT_CHECK_KEY")
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches for existing fact checks for a given claim.
        """
        if not self.api_key:
            print("Google Fact Check API Key missing.")
            return []

        params = {
            "key": self.api_key,
            "query": query,
            "languageCode": "en"
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(self.base_url, params=params)
                resp.raise_for_status()
                data = resp.json()
                
                # Return list of claims if found
                return data.get("claims", [])
            except Exception as e:
                print(f"Google Fact Check Error: {e}")
                return []
