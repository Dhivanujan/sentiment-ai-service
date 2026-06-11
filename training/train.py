import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/IMDBDataset.csv")

print(df.head())
print(df.shape)
print(df["sentiment"].value_counts())

df["review"] = df["review"].apply(clean_text)
print(df["review"].head())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["review"])
