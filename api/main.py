from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import os
import time
import io
import pandas as pd
import nltk

from training.preprocess import clean_text
from api.utils.nlp_helpers import get_sia, detect_emotions, analyze_aspects, highlight_sentences

app = FastAPI(title="Sentiment AI Service", version="2.0.0")

# Mount static files folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Load model + vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.on_event("startup")
def startup_event():
    # Make sure we have the required NLTK packages downloaded
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Error initializing NLTK data: {e}")

@app.get("/")
def home():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sentiment Analysis API is running 🚀. Frontend static assets are being prepared."}

@app.post("/predict")
def predict(text: str, model_type: str = "ml"):
    start_time = time.time()
    
    # 1. Custom ML Model (Logistic Regression + TF-IDF)
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    ml_pred = model.predict(vector)[0]
    
    # Get probability estimate if supported
    if hasattr(model, "predict_proba"):
        ml_prob = model.predict_proba(vector)[0]
    else:
        # Fallback if no probabilities
        ml_prob = [0.0, 1.0] if ml_pred == 1 else [1.0, 0.0]
        
    ml_sentiment = "positive" if ml_pred == 1 else "negative"
    ml_conf = float(ml_prob[1]) if ml_pred == 1 else float(ml_prob[0])

    # 2. VADER Sentiment Analysis
    sia = get_sia()
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    if vader_compound > 0.05:
        vader_sentiment = "positive"
    elif vader_compound < -0.05:
        vader_sentiment = "negative"
    else:
        vader_sentiment = "neutral"
        
    # Choose sentiment and confidence based on model selection
    if model_type == "ml":
        chosen_sentiment = ml_sentiment
        chosen_confidence = ml_conf
    else:
        chosen_sentiment = vader_sentiment
        # Map compound score to confidence
        chosen_confidence = abs(vader_compound) if vader_sentiment != "neutral" else (1.0 - abs(vader_compound))

    # Backward compatibility with original output format (e.g. positive 👍 / negative 👎)
    formatted_sentiment = f"{chosen_sentiment} " + ("👍" if chosen_sentiment == "positive" else "👎" if chosen_sentiment == "negative" else "😐")

    # 3. Advanced NLP Features
    sentences = highlight_sentences(text)
    aspects = analyze_aspects(text)
    emotions = detect_emotions(text, vader_compound)
    
    latency = time.time() - start_time

    return {
        "text": text,
        "sentiment": formatted_sentiment,
        "sentiment_label": chosen_sentiment,
        "confidence": round(chosen_confidence, 4),
        "model_used": model_type,
        "latency_seconds": round(latency, 4),
        
        "models_comparison": {
            "ml": {
                "sentiment": ml_sentiment,
                "confidence": round(ml_conf, 4),
                "probabilities": {
                    "negative": round(float(ml_prob[0]), 4),
                    "positive": round(float(ml_prob[1]), 4)
                }
            },
            "vader": {
                "sentiment": vader_sentiment,
                "confidence": round(abs(vader_compound), 4),
                "scores": vader_scores
            }
        },
        
        "sentence_breakdown": sentences,
        "aspects": aspects,
        "emotions": emotions
    }

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(None),
    model_type: str = "ml",
    texts: str = Query(None) # Can accept a comma-separated list of texts
):
    start_time = time.time()
    batch_texts = []
    
    # Case 1: File Upload (CSV or TXT)
    if file:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                # Search for a column with text data
                possible_cols = ["review", "text", "message", "body", "comment", "content", "review_body"]
                text_col = None
                for col in df.columns:
                    if col.lower() in possible_cols:
                        text_col = col
                        break
                if text_col is None:
                    text_col = df.columns[0] # fallback
                batch_texts = df[text_col].dropna().astype(str).tolist()
            except Exception as e:
                return {"error": f"Failed to parse CSV file: {str(e)}", "results": [], "summary": {}}
        else:
            # Plain text file (line by line)
            try:
                text_content = content.decode("utf-8", errors="ignore")
                batch_texts = [line.strip() for line in text_content.split("\n") if line.strip()]
            except Exception as e:
                return {"error": f"Failed to parse TXT file: {str(e)}", "results": [], "summary": {}}
                
    # Case 2: Comma-separated query param
    elif texts:
        batch_texts = [t.strip() for t in texts.split(",") if t.strip()]
        
    if not batch_texts:
        return {"error": "No review texts provided.", "results": [], "summary": {}}
        
    # Safety ceiling
    limit = 500
    processing_texts = batch_texts[:limit]
    
    results = []
    total_pos = 0
    total_neg = 0
    total_neu = 0
    
    for t in processing_texts:
        # Custom ML prediction
        cleaned = clean_text(t)
        vector = vectorizer.transform([cleaned])
        ml_pred = model.predict(vector)[0]
        ml_sentiment = "positive" if ml_pred == 1 else "negative"
        
        # VADER prediction
        sia = get_sia()
        v_score = sia.polarity_scores(t)['compound']
        vader_sentiment = "positive" if v_score > 0.05 else ("negative" if v_score < -0.05 else "neutral")
        
        selected = ml_sentiment if model_type == "ml" else vader_sentiment
        
        if selected == "positive":
            total_pos += 1
        elif selected == "negative":
            total_neg += 1
        else:
            total_neu += 1
            
        results.append({
            "text": t,
            "sentiment": selected,
            "vader_score": round(v_score, 4)
        })
        
    latency = time.time() - start_time
    total = len(processing_texts)
    
    return {
        "results": results,
        "total_processed": total,
        "truncated": len(batch_texts) > limit,
        "latency_seconds": round(latency, 4),
        "summary": {
            "positive": total_pos,
            "negative": total_neg,
            "neutral": total_neu,
            "positive_pct": round((total_pos / total) * 100, 1) if total else 0.0,
            "negative_pct": round((total_neg / total) * 100, 1) if total else 0.0,
            "neutral_pct": round((total_neu / total) * 100, 1) if total else 0.0
        }
    }
