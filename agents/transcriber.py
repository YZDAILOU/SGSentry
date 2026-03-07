import os
import httpx

class AudioTranscriber:
    def __init__(self):
        # Ensure HF_API_TOKEN and MERALION_API_URL are set in .env
        self.api_token = os.getenv("HF_API_TOKEN")
        self.api_url = os.getenv("MERALION_API_URL")

    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio using MERaLiON via Hugging Face Inference Endpoint.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not self.api_token or not self.api_url:
            raise ValueError("Missing HF_API_TOKEN or MERALION_API_URL in environment variables.")

        print(f"Transcribing {audio_path} with MERaLiON...")
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/octet-stream"
        }
        
        async with httpx.AsyncClient() as client:
            with open(audio_path, "rb") as f:
                data = f.read()
            
            response = await client.post(self.api_url, headers=headers, data=data, timeout=300.0)
            response.raise_for_status()
            
            result = response.json()
            
            # Handle list output (common in HF pipelines)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
                
            # Handle standard HF ASR output or AudioLLM generation
            if isinstance(result, dict):
                return result.get("text", result.get("generated_text", str(result)))
            
            return str(result)
