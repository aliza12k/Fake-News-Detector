import streamlit as st
import pickle
import os

# 1. APPLICATION SETUP & PAGE CONFIGURATION
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Custom UI Styling with a neutral public background image (No personal username links)
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=1964&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }
    textarea {
        background-color: #121824 !important;
        color: #ffffff !important;
        border: 1px solid #3a4f7c !important;
        border-radius: 10px !important;
    }
    h1 {
        color: #00e5ff !important;
        text-shadow: 0px 0px 12px rgba(0, 229, 255, 0.6);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .app-description {
        color: #d1d5db;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Render Application Headers
st.title(" Fake News Detector")
st.markdown('<p class="app-description">Enter a news article below to check if it is Real or Fake in real-time.</p>', unsafe_allow_html=True)

#  SESSION STATE FOR INPUT FLUSHING
if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = 0

current_widget_key = f"input_field_{st.session_state['reset_trigger']}"

# 2. INPUT INTERFACE
raw_article_text = st.text_area("Paste the news text here:", height=150, key=current_widget_key)

left_btn, right_btn = st.columns([1, 4])

with left_btn:
    analyze_action = st.button("Check News", type="primary")

with right_btn:
    flush_action = st.button("Clear Text")
    if flush_action:
        st.session_state["reset_trigger"] += 1
        st.rerun()

# 3. MACHINE LEARNING INFERENCE PIPELINE
if analyze_action:
    if not raw_article_text.strip():
        st.warning("Please enter some text first!")
    else:
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            st.error(" Error: Missing weights config! Please run 'python train_model.py' in your terminal first.")
        else:
            with open('model.pkl', 'rb') as saved_model:
                news_classifier = pickle.load(saved_model)
            with open('vectorizer.pkl', 'rb') as saved_vec:
                text_transformer = pickle.load(saved_vec)
            
            vector_matrix = text_transformer.transform([raw_article_text])
            class_outcome = news_classifier.predict(vector_matrix)[0]
            
            st.markdown("---")
            if class_outcome == 1:
                st.error(" WARNING: This news is likely FAKE!")
            else:
                st.success(" RELIABLE: This news appears to be REAL.")
