from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/analyze-mood")
async def analyze_mood(input: TextInput):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        endpoint = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"What is the main emotion in this text? Respond with one word (e.g., happy, sad, angry):\n\n{input.text}"
                        }
                    ]
                }
            ]
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        result = response.json()
        print("🔍 Gemini raw response:", result)  # 👈 Show full response

        if "candidates" not in result:
            raise ValueError(f"Gemini error: {result.get('error', 'No candidates returned')}")

        emotion = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"emotion": emotion}

    except Exception as e:
        print("❌ Gemini error:", e)
        raise HTTPException(status_code=500, detail=str(e))
