import streamlit as st
import joblib
import pandas as pd

# ===============================
# Load Model & Vectorizer
# ===============================
model = joblib.load("fake_message_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake Message Detector", page_icon="📰", layout="centered")

st.title("🔍 AI-Powered Fake Message Detector")
st.write("Enter a message below to check if it's **True** or **Fake**.")

# Sidebar - About
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    This app uses a **Machine Learning model** trained on fake/real message datasets.  
    It applies **TF-IDF (text vectorization)** + a classifier to detect suspicious messages.  
    Built with **Python, scikit-learn & Streamlit**.
    """
)

# Sidebar - New Features Info
st.sidebar.subheader("✨ Features")
st.sidebar.write(
    """
    - Predicts if a message is **True** ✅ or **Fake** 🚫  
    - Shows **Confidence Score**  
    - Displays **Probability Bar Chart**  
    - Clean & interactive UI  
    """
)

# ===============================
# Input Section
# ===============================
user_input = st.text_area("✏️ Type your message here:", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Transform input & predict
        input_tfidf = tfidf.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        proba = model.predict_proba(input_tfidf)[0]

        # Map prediction to index
        label_map = {0: "Fake", 1: "True"}
        reverse_map = {"FAKE": 0, "REAL": 1}

        if prediction in [0, "FAKE"]:
            st.error("🚫 This message is likely **Fake**.")
            pred_index = 0
        elif prediction in [1, "REAL"]:
            st.success("✅ This message looks **True**.")
            pred_index = 1
        else:
            st.warning(f"⚠️ Unknown prediction: {prediction}")
            pred_index = None

        # Show confidence score
        if pred_index is not None:
            st.write(f"**Confidence:** {proba[pred_index] * 100:.2f}%")

            # Show probability chart
            proba_df = pd.DataFrame({
                "Label": ["Fake", "True"],
                "Probability": proba
            })
            st.subheader("📊 Prediction Probability")
            st.bar_chart(proba_df.set_index("Label"))
