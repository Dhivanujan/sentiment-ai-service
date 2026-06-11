import joblib

from preprocess import clean_text

# Load model
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Input
text = ["I absolutely love this product"]

# Preprocess
text = [clean_text(t) for t in text]

# Vectorize
vector = vectorizer.transform(text)

# Predict
prediction = model.predict(vector)[0]

if prediction == 1:
    print("Positive")
else:
    print("Negative")