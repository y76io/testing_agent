import google.generativeai as genai
from ..config import GEMINI_API_KEY, GEMINI_MODEL

class GeminiClient:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)

    def complete(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.candidates[0].content.parts[0].text if resp and resp.candidates else ""