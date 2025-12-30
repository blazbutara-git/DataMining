import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time

# 1. Page Configuration
st.set_page_config(page_title="Brand Reputation Dashboard 2023", layout="wide")

# 2. Hugging Face API Setup
ENDPOINTS = [
    "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest",
    "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
]
HF_TOKEN = "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"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_single_sentiment(text):
    """Fetches sentiment with automatic fallback between endpoints"""
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    for url in ENDPOINTS:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    data = result[0] if isinstance(result[0], list) else result
                    top_result = max(data, key=lambda x: x['score'])
                    return top_result['label'].upper(), top_result['score']
            elif response.status_code == 503:
                return "LOADING", 0.0
            continue 
        except Exception:
            continue
    return "OFFLINE", 0.0

# 3. Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("scraped_data.csv")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Brand Dashboard")
page = st.sidebar.radio("Navigation:", ["Products", "Testimonials", "Reviews"])

if page == "Products":
    st.header("🛒 Product List")
    product_df = df[df['category'] == 'Product'][['text']]
    st.dataframe(product_df, use_container_width=True)

elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    testimonial_df = df[df['category'] == 'Testimonial'][['text', 'rating']]
    st.table(testimonial_df)

elif page == "Reviews":
    st.header("🔍 Sentiment Analysis (2023)")
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.select_slider("Select month:", options=range(1, 13), format_func=lambda x: months[x-1])
    
    review_df = df[(df['category'] == 'Review') & (df['date'].dt.month == selected_month)].copy()
    
    if not review_df.empty:
        # Cleanup: Remove time from date for display
        review_df['date'] = review_df['date'].dt.date
        
        st.write(f"Reviews found: {len(review_df)}")
        
        if st.button("Run AI Analysis"):
            sentiments = []
            scores = []
            bar = st.progress(0)
            msg = st.empty()
            
            for i, row in enumerate(review_df['text']):
                msg.text(f"Processing {i+1}/{len(review_df)}...")
                label, score = get_single_sentiment(str(row))
                
                if label == "LOADING":
                    msg.warning("Model is waking up... retrying in 10s.")
                    time.sleep(10)
                    label, score = get_single_sentiment(str(row))
                
                sentiments.append(label)
                scores.append(score)
                bar.progress((i + 1) / len(review_df))
            
            review_df['Sentiment'] = sentiments
            review_df['Confidence'] = scores
            msg.success("Analysis Complete!")

            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader(f"Review Details - {months[selected_month-1]}")
                # Added 'rating' back to the columns list
                st.dataframe(review_df[['date', 'text', 'rating', 'Sentiment', 'Confidence']], use_container_width=True)
            with c2:
                st.subheader("Sentiment Distribution")
                counts = review_df['Sentiment'].value_counts()
                if not counts.empty:
                    fig, ax = plt.subplots()
                    color_map = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f1c40f'}
                    colors = [color_map.get(l, '#95a5a6') for l in counts.index]
                    counts.plot(kind='bar', color=colors, ax=ax)
                    st.pyplot(fig)
                st.metric("Avg Star Rating", f"{review_df['rating'].mean():.1f}")

            st.divider()
            st.subheader("Word Cloud")
            all_text = " ".join(review_df['text'].astype(str))
            if len(all_text) > 10:
                wc = WordCloud(background_color="white", width=800, height=400).generate(all_text)
                st.image(wc.to_array(), use_container_width=True)
    else:
        st.info("No reviews for this month.")
