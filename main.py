from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

# Load .env variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Allow frontend to connect (for later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request format
class TextInput(BaseModel):
    text: str

# ✅ Endpoint 1: Analyze Mood
@app.post("/analyze_mood")
async def analyze_mood(input: TextInput):
    try:
        # Hugging Face emotion detection model
        HF_MODEL_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": input.text}

        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        result = response.json()

        print("🔍 Hugging Face raw output:", result)

        # If error from Hugging Face
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}

        # Get top emotion
        emotion = max(result[0], key=lambda x: x['score'])["label"]
        return {"emotion": emotion}

    except Exception as e:
        return {"error": str(e)}
@app.post("/detect_crisis")
async def detect_crisis(input: TextInput):
    try:
        # Crisis detection model (for detecting harmful intent)
        HF_MODEL_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/bert-base-uncased-suicide"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": input.text}

        # Send request to Hugging Face model
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}

        # Check if the model detected any harmful intent
        crisis_detected = False
        if result and result[0]:
            labels = result[0].get('label', [])
            if "suicidal" in labels or "self-harm" in labels:
                crisis_detected = True
        
        return {"crisis_detected": crisis_detected}

    except Exception as e:
        return {"error": str(e)}

@app.post("/summarize")
async def summarize(input: TextInput):
    try:
        HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": input.text}

        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}

        summary = result[0]["summary_text"]
        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}
