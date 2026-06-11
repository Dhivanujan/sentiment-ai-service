import streamlit as st
import requests

st.title("😊 AI Sentiment Analyzer")
st.write("Enter any text and get sentiment prediction")

text = st.text_input("Enter your review")

if st.button("Analyze"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        params={"text": text}
    )

    result = response.json()

    st.success(f"Sentiment: {result['sentiment']}")
    