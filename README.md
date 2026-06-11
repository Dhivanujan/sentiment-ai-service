# 🧠 SentimentAI: Production-Ready Sentiment Analysis Service

Welcome to **SentimentAI**, an end-to-end, production-grade sentiment and cognitive analysis ecosystem. The system leverages a dual-engine architecture combining custom Machine Learning models (Logistic Regression + TF-IDF) and rule-based Lexicon models (NLTK VADER) to extract multi-dimensional insights from customer feedback, social media posts, and product reviews.

The service exposes high-performance REST APIs via FastAPI and provides two user interfaces: a **premium web-based single-page application dashboard** (HTML5, Vanilla CSS, JS, Chart.js) and a **Streamlit analytics dashboard**.

---

## 🚀 Key Features

*   **Dual-Engine Analytics**: Swap dynamically between a custom ML classifier trained on 50,000 IMDB movie reviews (~89.5% accuracy) and the rule-based NLTK VADER lexicon.
*   **Cognitive Emotion Mapping**: Detects intensities of core human emotions (*Joy, Anger, Sadness, Fear, Surprise, Love*) using a hybrid keyword/VADER confidence lookup.
*   **Aspect-Based Sentiment Extraction**: Extracts local sentiments on key feedback areas such as *Quality, Price, Service, Performance, Delivery, and Usability*.
*   **Sentence-by-Sentence Breakdown**: Tokenizes documents to analyze and highlight individual sentence sentiments (Green = Positive, Red = Negative, Muted = Neutral).
*   **Batch Data Processor**: Upload CSV/TXT files to perform bulk analysis on up to 500 reviews with interactive doughnut charts, searchable/paginated tables, and CSV exports.
*   **Developer API Playground**: Built-in visual sandbox in the web app to trigger live REST queries and instantly view response JSONs, including pre-formatted client code snippets for **cURL**, **JavaScript**, and **Python**.

---

## 📂 Project Structure

```directory
SentimentAI/
├── api/
│   ├── main.py                 # FastAPI Application (Endpoints, Static mounting)
│   ├── static/                 # Premium Single-Page Web Frontend
│   │   ├── index.html          # Web UI Dashboard Shell
│   │   ├── styles.css          # Glassmorphic UI Aesthetics (Vanilla CSS)
│   │   └── app.js              # REST Controller, Chart.js, and API sandbox
│   └── utils/
│       ├── __init__.py
│       └── nlp_helpers.py      # VADER wrappers, emotion/aspect lexicons
├── app/
│   └── app.py                  # Streamlit Analytics Dashboard (Alternative UI)
├── data/
│   └── IMDBDataset.csv         # Training source file (IMDb reviews)
├── model/
│   ├── sentiment_model.pkl     # Serialized Logistic Regression Model
│   └── vectorizer.pkl          # Serialized TF-IDF Vectorizer
├── training/
│   ├── __init__.py
│   ├── preprocess.py           # Text regex cleanup routines
│   ├── train.py                # Model training and validation script
│   └── predict.py              # CLI evaluation tool
├── README.md                   # System Documentation
└── requirements.txt            # Python Dependencies
```

---

## 🛠️ Prerequisites & Installation

### 1. Requirements
*   Python 3.8 or higher
*   Virtual environment (recommended)

### 2. Clone & Setup
Clone the repository and set up a Python virtual environment:

```bash
# Navigate to the workspace root
cd SentimentAI

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training & Dataset

The ML model uses a **Logistic Regression** classifier combined with a **TF-IDF Vectorizer** trained on the IMDB reviews dataset containing 50,000 records.

### Running the Training Pipeline:
If you need to retrain or generate the serialized pickle files (`model/sentiment_model.pkl` and `model/vectorizer.pkl`), execute:

```bash
python training/train.py
```

*This script cleans the dataset using `training/preprocess.py`, performs an 80/20 train/test split, trains a Logistic Regression model, prints accuracy metrics (approx. **89.5%**), and saves the model parameters inside `/model`.*

---

## 🖥️ Running the Services

To experience the full capabilities of SentimentAI, run both the backend API and the analytics interfaces.

### Step 1: Start the FastAPI Server
Run the FastAPI application locally using Uvicorn:

```bash
uvicorn api.main:app --reload
```

*   **API Base URL**: `http://localhost:8000`
*   **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
*   **Premium Web UI Dashboard**: Accessible directly at `http://localhost:8000/` (or `http://localhost:8000/static/index.html`)

### Step 2: Run the Streamlit Dashboard (Optional)
Open a new terminal session, activate your virtual environment, and launch the Streamlit frontend:

```bash
streamlit run app/app.py
```

*   **Streamlit Access**: Typically opens at `http://localhost:8501`

---

## 🔌 API Reference

### 1. Predict Single Text
Performs comprehensive sentiment, aspect-based, and emotional analysis on a text query.

*   **Endpoint**: `POST /predict`
*   **Content-Type**: `application/json` (automatically handled by FastAPI query parameters)
*   **Query Parameters**:
    *   `text` (string, required): The target review or comment to evaluate.
    *   `model_type` (string, optional): `"ml"` (Logistic Regression) or `"vader"` (NLTK Lexicon). Defaults to `"ml"`.

#### Request Example (cURL):
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict?text=The%20screen%20and%20camera%20quality%20are%20breathtaking!%20However,%20delivery%20took%20three%20weeks.' \
  -H 'accept: application/json' \
  -d ''
```

#### Response Payload (JSON):
```json
{
  "text": "The screen and camera quality are breathtaking! However, delivery took three weeks.",
  "sentiment": "positive 👍",
  "sentiment_label": "positive",
  "confidence": 0.9534,
  "model_used": "ml",
  "latency_seconds": 0.0031,
  "models_comparison": {
    "ml": {
      "sentiment": "positive",
      "confidence": 0.9534,
      "probabilities": {
        "negative": 0.0466,
        "positive": 0.9534
      }
    },
    "vader": {
      "sentiment": "neutral",
      "confidence": 0.0,
      "scores": {
        "neg": 0.0,
        "neu": 1.0,
        "pos": 0.0,
        "compound": 0.0
      }
    }
  },
  "sentence_breakdown": [
    {
      "text": "The screen and camera quality are breathtaking!",
      "score": 0.0,
      "sentiment": "neutral"
    },
    {
      "text": "However, delivery took three weeks.",
      "score": 0.0,
      "sentiment": "neutral"
    }
  ],
  "aspects": [
    {
      "aspect": "quality",
      "score": 0.0,
      "sentiment": "neutral"
    },
    {
      "aspect": "delivery",
      "score": 0.0,
      "sentiment": "neutral"
    }
  ],
  "emotions": {
    "joy": 20.0,
    "anger": 0.0,
    "sadness": 0.0,
    "fear": 0.0,
    "surprise": 20.0,
    "love": 60.0
  }
}
```

---

### 2. Predict Batch Texts
Classifies a batch of reviews loaded from a file or supplied in query parameters.

*   **Endpoint**: `POST /predict/batch`
*   **Content-Type**: `multipart/form-data`
*   **Form-Data parameters**:
    *   `file` (file, optional): A CSV containing a column named `review`, `text`, `message`, `body`, or `comment`, OR a plain text file `.txt` containing one review per line.
*   **Query Parameters**:
    *   `model_type` (string, optional): `"ml"` or `"vader"`. Defaults to `"ml"`.
    *   `texts` (string, optional): A comma-separated string of texts to run.

#### Response Payload (JSON):
```json
{
  "results": [
    {
      "text": "I absolutely love this product",
      "sentiment": "positive",
      "vader_score": 0.6369
    },
    {
      "text": "This software is terrible and slow",
      "sentiment": "negative",
      "vader_score": -0.4767
    }
  ],
  "total_processed": 2,
  "truncated": false,
  "latency_seconds": 0.0051,
  "summary": {
    "positive": 1,
    "negative": 1,
    "neutral": 0,
    "positive_pct": 50.0,
    "negative_pct": 50.0,
    "neutral_pct": 0.0
  }
}
```

---

## ⚖️ Engine Benchmark Comparison

| Evaluation Metric | Logistic Regression (TF-IDF) | NLTK VADER Lexicon |
| :--- | :--- | :--- |
| **Model Type** | Machine Learning Classification | Rule & Valence-Based Lexicon |
| **Inference Latency** | ⚡ Extremely Fast (< 3ms) | 🚀 Ultra Fast (< 1ms) |
| **IMDB Accuracy** | 🎯 High (~89.5%) | 📈 Moderate (~75.0%) |
| **Punctuation Sensitivity** | ❌ None (lowercased, stripped) | ✅ High (capitalization & punctuation boost intensity) |
| **slang / Emojis** | ❌ Ignores non-word symbols | ✅ Excels at social media expressions/slang |
| **Best Target Data** | Long, descriptive, or formal reviews | Short reviews, social comments, tweets |

---

## 🎨 UI Dashboards Overview

### 1. Dark-Theme Web SPA Frontend (`api/static`)
The primary UI serves as a premium, dashboard. It provides:
1.  **Tabbed Workspace Layout**: Easily switch between single analysis, file batch uploads, code sandbox playground, and models documentation.
2.  **Interactive Visualizations**: Renders dynamic emotional profiling charts via Chart.js.
3.  **Real-Time Snippet Engine**: Generates copyable request scripts instantly based on playground input.

### 2. Streamlit Dashboard Backend (`app/app.py`)
Features a full side-control panel to toggle models on-the-fly and processes reviews through horizontal bar graphs, layout grids, and interactive, colored sentence segments.
