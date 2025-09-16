# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# 🔹 Load dataset (CSV must have columns: 'text' and 'label')
df = pd.read_csv("fake_or_real_news.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Save trained model inside /model folder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fake_news_model.pkl")

print("✅ Model trained and saved to model/fake_news_model.pkl")
