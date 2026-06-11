from fastapi import FastAPI
import joblib
import re
from training.preprocess import clean_text
app = FastAPI()

# Load model + vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running 🚀"}

@app.post("/predict")
def predict(text: str):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    sentiment = "positive 👍" if prediction == 1 else "negative 👎"

    return {
        "text": text,
        "sentiment": sentiment
    }
