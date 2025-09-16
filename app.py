import streamlit as st
from utils.predict import predict_news
from utils.lime_explainer import explain_prediction
from utils.scraper import extract_text_from_url
from utils.whatsapp_parser import parse_whatsapp_file

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News & WhatsApp Detector", layout="wide")

st.title("📰 Fake News & WhatsApp Detector")
st.markdown("Detect fake news in **articles, messages, or WhatsApp chats** with explanations 🔍")

# Sidebar for input type
option = st.sidebar.radio(
    "Choose Input Type:",
    ("Paste News Text", "Enter News URL", "Upload WhatsApp Chat")
)

# -------------------------------
# 1. PASTE TEXT
# -------------------------------
if option == "Paste News Text":
    user_text = st.text_area("✍️ Paste a news/article text below:", key="paste_text")
    if st.button("Analyze Text", key="analyze_text_btn"):
        if user_text.strip() != "":
            pred, confidence = predict_news(user_text)
            label = "🟢 Real" if pred == "REAL" else "🔴 Fake"

            st.subheader("Prediction:")
            st.write(f"{label} (Confidence: {confidence})")

            # Explanation
            with st.expander("See Explanation"):
                explanation = explain_prediction(user_text)
                for word, weight in explanation:
                    st.write(f"**{word}** → {weight:.3f}")
        else:
            st.warning("Please enter some text.")

# -------------------------------
# 2. URL INPUT
# -------------------------------
elif option == "Enter News URL":
    url = st.text_input("🌐 Enter the news article URL:", key="news_url")
    if st.button("Analyze URL", key="analyze_url_btn"):
        if url.strip() != "":
            text = extract_text_from_url(url)
            if text.startswith("Error:"):
                st.error(text)
            else:
                pred, confidence = predict_news(text)
                label = "🟢 Real" if pred == "REAL" else "🔴 Fake"

                st.subheader("Prediction:")
                st.write(f"{label} (Confidence: {confidence})")

                # Explanation
                with st.expander("See Explanation"):
                    explanation = explain_prediction(text)
                    for word, weight in explanation:
                        st.write(f"**{word}** → {weight:.3f}")
        else:
            st.warning("Please enter a URL.")

# -------------------------------
# 3. WHATSAPP CHAT
# -------------------------------
elif option == "Upload WhatsApp Chat":
    uploaded_file = st.file_uploader(
        "📤 Upload WhatsApp .txt export file", 
        type=["txt"], 
        key="whatsapp_upload"
    )
    if uploaded_file is not None:
        # Parse messages from WhatsApp export
        messages = parse_whatsapp_file(uploaded_file)
        st.success(f"Loaded {len(messages)} messages.")

        # Analyze each message
        if st.button("Analyze Chat", key="analyze_chat_btn"):
            fake_count = 0
            for msg in messages[:20]:  # analyze only first 20 messages for speed
                pred, confidence = predict_news(msg["message"])
                if pred == "FAKE":
                    fake_count += 1
                    st.write(f"🔴 **Fake:** {msg['message'][:80]}... (Confidence: {confidence})")
                else:
                    st.write(f"🟢 Real: {msg['message'][:80]}... (Confidence: {confidence})")

            st.info(f"Total Fake Messages Found: {fake_count}")
