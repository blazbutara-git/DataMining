import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="Brand Reputation Dashboard 2023", layout="wide")

# Load data and cache it for performance
@st.cache_data
def load_data():
    df = pd.read_csv("scraped_data.csv")
    # Convert date to datetime objects (Data Cleaning Part)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Load Hugging Face model and cache it (Part 3)
@st.cache_resource
def load_sentiment_model():
    # Specific model requested: distilbert-base-uncased-finetuned-sst-2-english
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize data and model
df = load_data()
sentiment_analyzer = load_sentiment_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section:", ["Products", "Testimonials", "Reviews"])

# --- SECTION 1: PRODUCTS ---
if page == "Products":
    st.header("🛒 Scraped Products")
    st.write("List of products extracted from the sandbox environment.")
    product_df = df[df['category'] == 'Product'][['text']]
    st.dataframe(product_df, use_container_width=True)

# --- SECTION 2: TESTIMONIALS ---
elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    st.write("General feedback and star ratings.")
    testimonial_df = df[df['category'] == 'Testimonial'][['text', 'rating']]
    
    # Display testimonials in a clean table
    st.table(testimonial_df.head(20))

# --- SECTION 3: REVIEWS & SENTIMENT ANALYSIS ---
elif page == "Reviews":
    st.header("🔍 Sentiment Analysis of Reviews (2023)")
    
    # Month slider (1 = January, 12 = December)
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    
    selected_month_num = st.select_slider(
        "Select a month to analyze:",
        options=range(1, 13),
        format_func=lambda x: months[x-1]
    )
    
    # Filter reviews by category and selected month
    review_df = df[(df['category'] == 'Review') & (df['date'].dt.month == selected_month_num)].copy()
    
    if not review_df.empty:
        # Run Sentiment Analysis
        with st.spinner('AI is analyzing sentiment...'):
            texts = review_df['text'].tolist()
            # Perform inference
            results = sentiment_analyzer(texts, truncation=True)
            
            # Add results back to dataframe
            review_df['Sentiment'] = [res['label'] for res in results]
            review_df['Confidence'] = [res['score'] for res in results]
        
        # UI layout with columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Reviews for {months[selected_month_num-1]} 2023")
            st.dataframe(review_df[['date', 'text', 'rating', 'Sentiment', 'Confidence']], use_container_width=True)
            
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_counts = review_df['Sentiment'].value_counts()
            
            # Simple Matplotlib Bar Chart
            fig, ax = plt.subplots()
            colors = ['#2ecc71' if label == 'POSITIVE' else '#e74c3c' for label in sentiment_counts.index]
            sentiment_counts.plot(kind='bar', color=colors, ax=ax)
            ax.set_ylabel("Number of Reviews")
            st.pyplot(fig)
            
            # Metrics
            avg_rating = review_df['rating'].mean()
            st.metric("Average Star Rating", f"{avg_rating:.1f} / 5")
    else:
        st.info(f"No reviews found for {months[selected_month_num-1]} 2023.")