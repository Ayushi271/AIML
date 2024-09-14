import matplotlib.pyplot as plt
import seaborn as sns

def plot_overall_sentiments(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='aspects', y='sentiment_score', data=df)
    plt.title('Overall Aspect Sentiments')
    plt.xlabel('Aspects')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('overall_aspect_sentiments.png')
    plt.close()

def plot_sentiment_distribution(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='aspects', y='sentiment_score', data=df.explode('aspects'))
    plt.title('Sentiment Distribution by Aspect')
    plt.xlabel('Aspects')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_distribution_by_aspect.png')
    plt.close()