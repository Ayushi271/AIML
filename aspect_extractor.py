def extract_aspects(text, aspect_keywords):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    aspects = []
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in [token.lemma_.lower() for token in doc] for keyword in keywords):
            aspects.append(aspect)
    return aspects

aspect_keywords = {
    'Food Quality': ['food', 'taste', 'flavor', 'delicious', 'menu'],
    'Service': ['service', 'staff', 'waiter', 'waitress', 'customer'],
    'Ambiance': ['ambiance', 'atmosphere', 'decor', 'music', 'setting'],
    'Pricing': ['price', 'value', 'expensive', 'cheap', 'affordable'],
    'Cleanliness': ['clean', 'hygiene', 'sanitary', 'spotless', 'dirty']
}

def extract_aspects_from_reviews(df):
    df['aspects'] = df['processed_text'].apply(lambda x: extract_aspects(x, aspect_keywords))
    return df
