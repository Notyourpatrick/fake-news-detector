# 📰 Fake News & WhatsApp News Detector

An interactive Python web application that detects **fake news** in articles, messages, and WhatsApp chats. The app also provides **explanations** for predictions using **LIME**, helping users understand why a piece of news is classified as fake or real.

---

## 🔹 Features

- **Text Input Detection:** Paste any news or message and check if it’s fake or real.  
- **URL Detection:** Enter a news article URL and automatically extract and analyze its content.  
- **WhatsApp Chat Detection:** Upload exported WhatsApp `.txt` files to analyze messages in bulk.  
- **Explanation with LIME:** Highlights key words that influenced the prediction for transparency.  
- **Interactive Web App:** Built with **Streamlit** for easy use without coding knowledge.  
- **Machine Learning Model:** Uses **TF-IDF vectorization** + **Logistic Regression pipeline** for accurate predictions.

---

## 🔹 Tech Stack

- **Python 3.13**  
- **Streamlit** (Frontend & Interactive UI)  
- **Scikit-learn** (Machine Learning Model)  
- **LIME** (Explainable AI)  
- **Newspaper3k** (Web article scraping)  
- **Joblib** (Model persistence)

---

## 🔹 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


structre of the folder
fake-news-detector/
│
├── app.py
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── utils/
│   ├── __init__.py        ✅ (this file must exist, even if empty)
│   ├── predict.py
│   ├── lime_explainer.py
│   ├── scraper.py
│   └── whatsapp_parser.py
