import json
import pandas as pd

def load_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def load_data():
    reviews_df = load_json_file('data/review.json')
    businesses_df = load_json_file('data/business.json')
    return reviews_df, businesses_df