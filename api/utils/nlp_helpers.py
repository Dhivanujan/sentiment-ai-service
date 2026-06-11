import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Intensity Analyzer
# Note: we ensure this is downloaded in main.py startup
_sia = None

def get_sia():
    global _sia
    if _sia is None:
        try:
            _sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            _sia = SentimentIntensityAnalyzer()
    return _sia

# Lexicons for Emotion and Aspect matching
EMOTION_LEXICON = {
    "joy": ["happy", "glad", "joy", "delight", "love", "wonderful", "great", "excellent", "awesome", "fantastic", "amazing", "good", "nice", "enjoy", "pleased", "satisfy", "satisfied", "thrilled", "excited", "cheerful", "gladly", "happily", "joyful", "perfect", "superb", "fabulous", "outstanding", "pleasure"],
    "anger": ["angry", "mad", "annoyed", "frustrated", "hate", "furious", "irritated", "rage", "terrible", "bad", "worst", "dislike", "annoy", "offended", "crap", "stupid", "idiot", "nonsense", "useless", "disgusting", "scam", "waste", "garbage", "trash", "hate", "hated", "awful"],
    "sadness": ["sad", "unhappy", "depressed", "disappointed", "sorrow", "grief", "cry", "gloomy", "heartbroken", "regret", "pity", "hurt", "pain", "lonely", "miss", "failed", "unfortunately", "loss", "lost", "fail", "hopeless", "miserable", "dull", "bored", "regretful"],
    "fear": ["scared", "afraid", "fear", "terrified", "anxious", "nervous", "worry", "worried", "horror", "spooky", "scary", "threat", "danger", "frightened", "panic", "dread", "suspicious", "insecure", "creepy", "dreadful"],
    "surprise": ["surprised", "shocked", "amazing", "suddenly", "unexpected", "astonished", "wow", "incredible", "unbelievable", "wonder", "surprising", "shocker", "stunning", "extraordinary", "unanticipated", "mindblowing", "astounded"],
    "love": ["love", "adore", "like", "fond", "passion", "affection", "darling", "sweet", "caring", "admire", "cherish", "lovely", "heart", "warmth", "kindness", "compassion", "beloved", "romantic", "loving", "liked"]
}

ASPECT_LEXICON = {
    "quality": ["quality", "build", "material", "durable", "sturdy", "cheap", "broke", "fragile", "premium", "solid", "well-made", "flimsy", "plastic", "craftsmanship", "reliable", "design", "appearance"],
    "price": ["price", "cost", "expensive", "cheap", "worth", "money", "affordable", "value", "deal", "pricing", "budget", "overpriced", "billing", "charge", "refund", "fee", "rates", "economical"],
    "service": ["service", "customer", "staff", "support", "help", "agent", "representative", "employee", "manager", "helpful", "rude", "polite", "assistance", "response", "responsiveness", "friendly"],
    "performance": ["speed", "fast", "slow", "performance", "lag", "smooth", "battery", "camera", "screen", "display", "processor", "hardware", "quick", "responsive", "power", "ram", "memory", "efficiency"],
    "delivery": ["delivery", "shipping", "shipped", "arrived", "package", "late", "time", "speedy", "delay", "post", "mail", "track", "tracking", "received", "transit"],
    "usability": ["easy", "difficult", "hard", "simple", "complex", "usability", "setup", "install", "interface", "ui", "navigation", "confusing", "intuitive", "user-friendly", "learning", "complicated"]
}

def clean_word(word):
    return word.strip(string.punctuation).lower()

def detect_emotions(text, overall_sentiment_score):
    """
    Detects emotion intensities (Joy, Anger, Sadness, Fear, Surprise, Love) based on keyword occurrences,
    guided by the overall VADER compound score to handle baseline fallback cases.
    """
    words = [clean_word(w) for w in re.split(r'\s+', text) if w]
    counts = {emotion: 0 for emotion in EMOTION_LEXICON.keys()}
    
    # Keyword matches
    for word in words:
        for emotion, keywords in EMOTION_LEXICON.items():
            if word in keywords:
                counts[emotion] += 1

    total_matches = sum(counts.values())
    
    # Fallback to general sentiment mapping if no keywords are matched
    if total_matches == 0:
        if overall_sentiment_score > 0.05:
            # Positive baseline
            counts["joy"] = 3
            counts["love"] = 1
            counts["surprise"] = 1
        elif overall_sentiment_score < -0.05:
            # Negative baseline
            counts["anger"] = 2
            counts["sadness"] = 2
            counts["fear"] = 1
        else:
            # Neutral baseline
            counts["joy"] = 0.5
            counts["surprise"] = 0.5
            counts["love"] = 0.2
    
    # Calculate percentages
    total_score = sum(counts.values())
    emotions_pct = {}
    for emotion, val in counts.items():
        emotions_pct[emotion] = round((val / total_score) * 100, 1) if total_score > 0 else 0.0
        
    return emotions_pct

def analyze_aspects(text):
    """
    Identifies if aspects (quality, price, etc.) are discussed in the text,
    and returns their sentiment score based on sentence context.
    """
    sia = get_sia()
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(text)
        
    aspect_scores = {aspect: [] for aspect in ASPECT_LEXICON.keys()}
    
    for sentence in sentences:
        sentence_clean = sentence.lower()
        words = [clean_word(w) for w in re.split(r'\s+', sentence_clean) if w]
        
        # Check matching aspects
        matched_aspects = []
        for aspect, keywords in ASPECT_LEXICON.items():
            for word in words:
                if word in keywords:
                    matched_aspects.append(aspect)
                    break
        
        if matched_aspects:
            # Calculate sentiment score for this sentence
            scores = sia.polarity_scores(sentence)
            compound = scores['compound']
            
            for aspect in matched_aspects:
                aspect_scores[aspect].append(compound)
                
    # Aggregate scores
    results = []
    for aspect, scores in aspect_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.05:
                sentiment = "positive"
            elif avg_score < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            results.append({
                "aspect": aspect,
                "score": round(avg_score, 3),
                "sentiment": sentiment
            })
            
    return results

def highlight_sentences(text):
    """
    Splits text into sentences and scores each individually.
    """
    sia = get_sia()
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(text)
        
    sentence_data = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        scores = sia.polarity_scores(sentence)
        compound = scores['compound']
        
        if compound > 0.05:
            sentiment = "positive"
        elif compound < -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        sentence_data.append({
            "text": sentence,
            "score": round(compound, 3),
            "sentiment": sentiment
        })
        
    return sentence_data
