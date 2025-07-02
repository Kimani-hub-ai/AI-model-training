import spacy
import re

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon product reviews
reviews = [
    "I absolutely love my new Sony headphones. The sound quality is amazing!",
    "Terrible product. My Samsung phone stopped working in two weeks!",
    "This Apple charger works great and charges super fast.",
    "The Logitech keyboard is okay, but the keys are too stiff.",
    "Garbage! This HP printer jammed every time I used it."
]

# Define rule-based positive and negative keywords
positive_keywords = ["love", "great", "amazing", "good", "awesome", "excellent", "super", "fast"]
negative_keywords = ["terrible", "bad", "worst", "awful", "hate", "slow", "stopped", "stiff", "garbage", "jammed"]

# Function to classify sentiment based on keyword rules
def classify_sentiment(text):
    text_lower = text.lower()
    pos_count = sum(word in text_lower for word in positive_keywords)
    neg_count = sum(word in text_lower for word in negative_keywords)
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"

# Process each review
for review in reviews:
    print("\nReview:", review)
    
    # Run spaCy pipeline
    doc = nlp(review)
    
    # Extract entities related to products/brands
    print("Named Entities:")
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f" - {ent.text} ({ent.label_})")
    
    # Sentiment analysis
    sentiment = classify_sentiment(review)
    print("Sentiment:", sentiment)
