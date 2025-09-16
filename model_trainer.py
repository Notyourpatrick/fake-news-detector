# model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save_model():
    # Load dataset
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/FakeNewsData.csv")
    df = df[['text', 'label']]  # text: news, label: 0 = Real, 1 = Fake

    # Preprocessing
    df.dropna(inplace=True)
    X = df['text']
    y = df['label']

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    # Train model
    model = LogisticRegression()
    model.fit(X_vec, y)

    # Save model and vectorizer
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')

if __name__ == "__main__":
    train_and_save_model()
