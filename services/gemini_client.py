import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = "gemini-3-flash-preview"

    def generate(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text


if __name__ == "__main__":
    gemini = GeminiClient()

    response = gemini.generate(
        "Explain demand forecasting in retail in 3 sentences"
    )

    print(response)