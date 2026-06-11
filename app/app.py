import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="SentimentAI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to align with dark theme aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f3f4f6 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .metric-card {
        background-color: rgba(22, 31, 48, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Options
st.sidebar.title("🧠 SentimentAI Control")
st.sidebar.markdown("Configure analysis models and run custom reviews.")

model_type = st.sidebar.selectbox(
    "Choose Analysis Model",
    options=["ml", "vader"],
    format_func=lambda x: "Logistic Regression (ML)" if x == "ml" else "NLTK VADER Lexicon"
)

st.sidebar.info("💡 **Logistic Regression** works best for longer, formal text reviews. **VADER** is excellent for social media post emojis, punctuation emphasis, and short comments.")

# Main Workspace
st.title("😊 SentimentAI Analysis Hub")
st.subheader("Extract cognitive emotions, aspect sentiments, and sentence metrics from text.")

# Examples helper
st.markdown("### Choose an Example or Type Below")
examples = [
    "Select an example...",
    "The screen and camera quality of this new phone are breathtaking! However, the delivery took almost three weeks and was quite late.",
    "This software is absolutely terrible. It is extremely slow, lags constantly, and customer support refused to refund my subscription cost.",
    "I received the package today. Delivery was speedy and setup was very simple. It works exactly as expected."
]
selected_example = st.selectbox("Pre-fill workspace with a sample review:", examples)

# Text Input
text_val = ""
if selected_example != "Select an example...":
    text_val = selected_example

text = st.text_area("Enter Review Text for Cognitive Processing:", value=text_val, height=140, placeholder="Type review here...")

if st.button("Analyze Review", type="primary"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Processing cognitive NLP pipeline..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    params={"text": text, "model_type": model_type}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Columns layout for overall results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Predicted Sentiment", result["sentiment"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence Level", f"{result['confidence'] * 100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Inference Latency", f"{result['latency_seconds'] * 1000:.2f} ms")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Lower grid splits
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.write("#### 📊 Emotional Profile")
                        emotions_data = pd.DataFrame({
                            "Emotion": [e.capitalize() for e in result["emotions"].keys()],
                            "Intensity (%)": list(result["emotions"].values())
                        })
                        st.bar_chart(emotions_data.set_index("Emotion"), horizontal=True, color="#00ffff")
                        
                    with col_right:
                        st.write("#### 🏷️ Aspect-Based Sentiment")
                        aspects = result["aspects"]
                        if not aspects:
                            st.info("No specific aspects (quality, price, service, etc.) matched in this text.")
                        else:
                            for asp in aspects:
                                sentiment_emoji = "🟢" if asp["sentiment"] == "positive" else ("🔴" if asp["sentiment"] == "negative" else "⚪")
                                st.markdown(f"**{asp['aspect'].capitalize()}**: {sentiment_emoji} {asp['sentiment'].upper()} (Score: {asp['score']:.2f})")
                                
                    # Sentence breakdown highlight rendering
                    st.write("#### 📝 Sentence-by-Sentence Breakdown")
                    st.markdown("Highlights represent local contextual scores (Green = Positive, Red = Negative, Neutral = Muted):")
                    
                    highlighted_html = '<div style="background-color: #101725; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 15px; line-height: 1.6; font-size: 1.05rem;">'
                    for s in result["sentence_breakdown"]:
                        if s["sentiment"] == "positive":
                            bg_color = "rgba(16, 185, 129, 0.18)"
                            border_color = "#10b981"
                        elif s["sentiment"] == "negative":
                            bg_color = "rgba(255, 0, 127, 0.18)"
                            border_color = "#ff007f"
                        else:
                            bg_color = "rgba(255, 255, 255, 0.03)"
                            border_color = "#64748b"
                            
                        highlighted_html += f'<span style="background-color: {bg_color}; border-bottom: 2px solid {border_color}; padding: 2px 4px; margin: 0 2px; border-radius: 4px;" title="Score: {s["score"]:.2f}">{s["text"]}</span> '
                    highlighted_html += '</div>'
                    
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Prediction failed with status code: {response.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}. Ensure the FastAPI server is running on localhost port 8000.")