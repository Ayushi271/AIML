import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

def preprocess_data(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df