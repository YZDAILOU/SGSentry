import os
import time
from google import genai
from google.genai import types

class AudioTranscriber:
    def __init__(self):
        # Initialize the Gemini Client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio by uploading to Gemini File API and processing with Gemini 2.5 Flash.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Uploading {audio_path} to Gemini File API...")

        try:
            # 1. Upload the file to Google's File API
            # Gemini requires audio to be uploaded before it can be processed
            uploaded_file = self.client.files.upload(file=audio_path)

            # 2. Wait for the file to be processed by Google (Status: ACTIVE)
            # This is usually instant for small files, but good for safety
            while uploaded_file.state.name == "PROCESSING":
                print("Waiting for file to be processed...")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise ValueError(f"File processing failed: {uploaded_file.name}")

            # 3. Generate Transcription
            print(f"Transcribing {audio_path} with {self.model_id}...")
            
            # Using a specific prompt to ensure verbatim Singapore-aware transcription
            prompt = (
                "Transcribe this audio verbatim. Keep the original sentence structure "
                "and accurately capture Singaporean English (Singlish) terms, accents, and acronyms."
            )

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[prompt, uploaded_file]
            )

            # 4. Clean up (Optional: Delete the file from Google's cloud after use)
            self.client.files.delete(name=uploaded_file.name)

            return response.text

        except Exception as e:
            print(f"❌ Gemini Transcription Error: {e}")
            raise e