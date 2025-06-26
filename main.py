from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from transformers import pipeline
import torch

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

# Endpoint 1: Analyze Mood
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
    

# Endpoint 2: Detect Crisis
@app.post("/detect_crisis")
async def detect_crisis(input: TextInput):
    HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": input.text,
        "parameters": {
            "candidate_labels": ["crisis", "non-crisis", "neutral", "support"]
        }
    }

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        print("🔁 Status Code:", response.status_code)
        print("📦 Raw Text:", response.text)

        if response.status_code != 200:
            return {
                "error": f"Model call failed with status {response.status_code}",
                "details": response.text
            }

        result = response.json()
        labels = result.get("labels", [])
        scores = result.get("scores", [])

        top_label = labels[0]
        top_score = scores[0]
        crisis_detected = top_label.lower() == "crisis"

        return {
            "crisis_detected": crisis_detected,
            "top_label": top_label,
            "confidence": round(top_score, 4),
            "raw_output": list(zip(labels, scores))
        }

    except Exception as e:
        return {"error": str(e)}



#i want to hurt myself

# Endpoint 3: Summerize
@app.post("/summarize")
async def summarize(input: TextInput):
    HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    # Optional: Prevent junk output
    if len(input.text.split()) < 10:
        return {
            "summary": input.text,
            "note": "Too short to summarize — returned original text."
        }

    payload = {
        "inputs": input.text,
        "parameters": {
            "max_length": 50,
            "min_length": 10,
            "do_sample": False
        }
    }

    try:
        response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
        result = response.json()

        if response.status_code != 200:
            return {"error": result.get("error", "Unknown error")}

        summary = result[0].get("summary_text", "")
        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}
