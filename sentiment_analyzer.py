from transformers import pipeline

sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def classify_sentiment(text):
    result = sentiment_classifier(text[:512])[0]  # Limit text length to avoid errors
    return result['label'], result['score']

def analyze_sentiment(df):
    df['sentiment_label'], df['sentiment_score'] = zip(*df['processed_text'].apply(classify_sentiment))
    return df