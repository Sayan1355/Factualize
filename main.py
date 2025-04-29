from fastapi import FastAPI, status
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.layers import TextVectorization
from fastapi.middleware.cors import CORSMiddleware

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    intent: str
    response: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model after application startup to improve initial response time
@app.on_event("startup")
async def load_model():
    global model, class_names, responses
    model = tf.keras.models.load_model(
        "./models/chatbot.keras",
        custom_objects={"TextVectorization": TextVectorization},
        compile=False
    )
    class_names = np.load("./models/classes.npy", allow_pickle=True)
    with open("./models/dataset.json", "r") as f:
        data = json.load(f)
    responses = {
        intent["tag"]: intent.get("responses", [])
        for intent in data.get("intents", [])
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    texts = np.array([req.text], dtype=object)
    preds = model.predict(texts)
    idx = int(np.argmax(preds, axis=1)[0])
    intent = class_names[idx]
    opts = responses.get(intent, [])
    reply = np.random.choice(opts) if opts else "Sorry, I didn't get that."
    return PredictResponse(intent=intent, response=reply)

@app.get("/")
def root():
    return {"msg": "Therapeutic chatbot API up and running!"}

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}