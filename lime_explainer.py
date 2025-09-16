import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load the pipeline (TF-IDF + classifier)
model = joblib.load("model/fake_news_model.pkl")

# Class names
class_names = model.classes_

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

def explain_prediction(text):
    """
    Generate LIME explanation for a single text input.
    Returns:
        list of tuples: (word, weight)
    """
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=model.predict_proba,
        num_features=10
    )
    return exp.as_list()
