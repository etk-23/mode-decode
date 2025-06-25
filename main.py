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
        # Hugging Face crisis detection model (replace this URL with your own model URL)
        HF_MODEL_URL = "https://api-inference.huggingface.co/models/your-crisis-detection-model"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": input.text}

        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        result = response.json()

        print("🔍 Hugging Face raw output:", result)

        # If error from Hugging Face
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"]}

        # Assuming the model returns a "crisis_detected" field
        crisis_detected = result.get("crisis_detected", False)

        return {"crisis_detected": crisis_detected}

    except Exception as e:
        return {"error": str(e)}





#i want to hurt myself

@app.post("/summarize")
async def summarize(input: TextInput):
    HF_MODEL = "facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json={
            "inputs": input.text,
            "parameters": {"max_length": 150, "min_length": 30}
        }
    )
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"]}

    return {"summary": result[0].get("summary_text", "")}
