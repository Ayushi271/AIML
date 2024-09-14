from src.data_loader import load_data
from src.preprocessor import preprocess_data
from src.aspect_extractor import extract_aspects_from_reviews
from src.sentiment_analyzer import analyze_sentiment
from src.visualizer import plot_overall_sentiments, plot_sentiment_distribution
import pandas as pd

def generate_insights(analyzed_df):
    aspect_sentiments = analyzed_df.explode('aspects').groupby(['business_id', 'aspects']).agg({
        'sentiment_score': 'mean',
        'stars': 'mean'
    }).reset_index()
    
    overall_aspect_sentiments = aspect_sentiments.groupby('aspects').agg({
        'sentiment_score': 'mean',
        'stars': 'mean'
    }).reset_index()
    
    top_bottom_businesses = aspect_sentiments.groupby('aspects').apply(
        lambda x: pd.concat([x.nlargest(5, 'sentiment_score'), x.nsmallest(5, 'sentiment_score')])
    ).reset_index(drop=True)
    
    return overall_aspect_sentiments, top_bottom_businesses

def main():
    # Load data
    print("Loading data...")
    reviews_df, businesses_df = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    reviews_df = preprocess_data(reviews_df)
    
    # Extract aspects
    print("Extracting aspects...")
    reviews_df = extract_aspects_from_reviews(reviews_df)
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    reviews_df = analyze_sentiment(reviews_df)
    
    # Generate insights
    print("Generating insights...")
    overall_aspect_sentiments, top_bottom_businesses = generate_insights(reviews_df)
    
    # Visualize results
    print("Creating visualizations...")
    plot_overall_sentiments(overall_aspect_sentiments)
    plot_sentiment_distribution(reviews_df)
    
    # Save results
    reviews_df.to_csv('yelp_aspect_sentiment_results.csv', index=False)
    overall_aspect_sentiments.to_csv('overall_aspect_sentiments.csv', index=False)
    top_bottom_businesses.to_csv('top_bottom_businesses_by_aspect.csv', index=False)
    
    print("Analysis complete. Results and visualizations saved.")

if __name__ == "__main__":
    main()