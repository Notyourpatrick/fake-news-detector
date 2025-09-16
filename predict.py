# utils/predict.py
import joblib

# Load pipeline (contains both TF-IDF vectorizer + classifier)
model = joblib.load("model/fake_news_model.pkl")

def predict_news(text):
    """
    Predict whether a news/message is FAKE or REAL.
    Returns:
        prediction (str) -> 'FAKE' or 'REAL'
        probability (dict) -> probabilities for each class
    """
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]

    # Create a dict with class probabilities
    prob_dict = {
        label: round(prob, 3) for label, prob in zip(model.classes_, probability)
    }

    return prediction, prob_dict
