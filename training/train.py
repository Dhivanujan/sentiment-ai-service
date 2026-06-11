import pandas as pd
import joblib

from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset
df = pd.read_csv("data/IMDBDataset.csv")

print(df.shape)
print(df["sentiment"].value_counts())

# Preprocess
df["review"] = df["review"].apply(clean_text)

# Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"])

# Labels
y = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")

# Save
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model Saved Successfully")