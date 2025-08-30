import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import time

# Sentiment Analysis Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data (run once)
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/vader_lexicon')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

download_nltk_data()

# Initialize VADER analyzer
@st.cache_resource
def load_vader_analyzer():
    return SentimentIntensityAnalyzer()

vader_analyzer = load_vader_analyzer()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text == "":
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_hashtags(text):
    """Extract hashtags from text"""
    return re.findall(r'#\w+', text)

def extract_mentions(text):
    """Extract mentions from text"""
    return re.findall(r'@\w+', text)

def count_emojis(text):
    """Count emojis in text"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(text))

def get_text_metadata(text):
    """Extract metadata from text"""
    return {
        'character_count': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(re.split(r'[.!?]+', text)) - 1,
        'hashtag_count': len(extract_hashtags(text)),
        'mention_count': len(extract_mentions(text)),
        'emoji_count': count_emojis(text),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?')
    }

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        sentiment = 'Positive'
        emoji = '😊'
    elif polarity < -0.1:
        sentiment = 'Negative'
        emoji = '😞'
    else:
        sentiment = 'Neutral'
        emoji = '😐'
    
    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'confidence': abs(polarity)
    }

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'Positive'
        emoji = '😊'
    elif compound <= -0.05:
        sentiment = 'Negative'
        emoji = '😞'
    else:
        sentiment = 'Neutral'
        emoji = '😐'
    
    return {
        'sentiment': sentiment,
        'emoji': emoji,
        'compound': compound,
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg'],
        'confidence': abs(compound)
    }

def get_sentiment_intensity(score):
    """Get sentiment intensity label"""
    abs_score = abs(score)
    if abs_score >= 0.8:
        return "Very Strong"
    elif abs_score >= 0.6:
        return "Strong"
    elif abs_score >= 0.4:
        return "Moderate"
    elif abs_score >= 0.2:
        return "Weak"
    else:
        return "Very Weak"

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .sentiment-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid;
        margin: 1rem 0;
    }
    
    .positive-card {
        border-left-color: #28a745;
    }
    
    .negative-card {
        border-left-color: #dc3545;
    }
    
    .neutral-card {
        border-left-color: #6c757d;
    }
    
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🎭 Sentiment Analysis Tool</h1>
    <p>Analyze emotions and sentiments in text using AI-powered NLP</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Settings")
st.sidebar.markdown("---")

# Analysis options
analysis_method = st.sidebar.selectbox(
    "Choose Analysis Method:",
    ["Both (Recommended)", "TextBlob Only", "VADER Only"]
)

show_metadata = st.sidebar.checkbox("Show Text Metadata", value=True)
show_visualizations = st.sidebar.checkbox("Show Visualizations", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Methods")
st.sidebar.markdown("""
**TextBlob:**
- General purpose sentiment analysis
- Polarity: -1 (negative) to +1 (positive)
- Subjectivity: 0 (objective) to 1 (subjective)

**VADER:**
- Social media focused
- Better with emojis, slang, punctuation
- Compound score: -1 to +1
""")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Your Text")
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["Single Text", "Multiple Texts", "Upload File"],
        horizontal=True
    )
    
    if input_method == "Single Text":
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here... (tweets, reviews, comments, etc.)",
            height=150,
            help="Enter any text you want to analyze for sentiment"
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if user_input:
                with st.spinner("Analyzing sentiment..."):
                    # Perform analysis
                    cleaned_text = clean_text(user_input)
                    metadata = get_text_metadata(user_input)
                    
                    if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                        textblob_result = analyze_sentiment_textblob(cleaned_text)
                    
                    if analysis_method in ["Both (Recommended)", "VADER Only"]:
                        vader_result = analyze_sentiment_vader(cleaned_text)
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    # Main results
                    if analysis_method == "Both (Recommended)":
                        col_tb, col_vader = st.columns(2)
                        
                        with col_tb:
                            sentiment_class = textblob_result['sentiment'].lower()
                            st.markdown(f"""
                            <div class="sentiment-card {sentiment_class}-card">
                                <h3>TextBlob Analysis {textblob_result['emoji']}</h3>
                                <h2>{textblob_result['sentiment']}</h2>
                                <p><strong>Polarity:</strong> {textblob_result['polarity']:.3f}</p>
                                <p><strong>Subjectivity:</strong> {textblob_result['subjectivity']:.3f}</p>
                                <p><strong>Intensity:</strong> {get_sentiment_intensity(textblob_result['polarity'])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_vader:
                            sentiment_class = vader_result['sentiment'].lower()
                            st.markdown(f"""
                            <div class="sentiment-card {sentiment_class}-card">
                                <h3>VADER Analysis {vader_result['emoji']}</h3>
                                <h2>{vader_result['sentiment']}</h2>
                                <p><strong>Compound:</strong> {vader_result['compound']:.3f}</p>
                                <p><strong>Positive:</strong> {vader_result['positive']:.3f}</p>
                                <p><strong>Negative:</strong> {vader_result['negative']:.3f}</p>
                                <p><strong>Intensity:</strong> {get_sentiment_intensity(vader_result['compound'])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Agreement check
                        agreement = textblob_result['sentiment'] == vader_result['sentiment']
                        if agreement:
                            st.success(f"✅ Both methods agree: **{textblob_result['sentiment']}**")
                        else:
                            st.warning(f"⚠️ Methods disagree: TextBlob says **{textblob_result['sentiment']}**, VADER says **{vader_result['sentiment']}**")
                    
                    elif analysis_method == "TextBlob Only":
                        sentiment_class = textblob_result['sentiment'].lower()
                        st.markdown(f"""
                        <div class="sentiment-card {sentiment_class}-card">
                            <h3>TextBlob Analysis {textblob_result['emoji']}</h3>
                            <h2>{textblob_result['sentiment']}</h2>
                            <p><strong>Polarity:</strong> {textblob_result['polarity']:.3f}</p>
                            <p><strong>Subjectivity:</strong> {textblob_result['subjectivity']:.3f}</p>
                            <p><strong>Intensity:</strong> {get_sentiment_intensity(textblob_result['polarity'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:  # VADER Only
                        sentiment_class = vader_result['sentiment'].lower()
                        st.markdown(f"""
                        <div class="sentiment-card {sentiment_class}-card">
                            <h3>VADER Analysis {vader_result['emoji']}</h3>
                            <h2>{vader_result['sentiment']}</h2>
                            <p><strong>Compound:</strong> {vader_result['compound']:.3f}</p>
                            <p><strong>Positive:</strong> {vader_result['positive']:.3f}</p>
                            <p><strong>Negative:</strong> {vader_result['negative']:.3f}</p>
                            <p><strong>Intensity:</strong> {get_sentiment_intensity(vader_result['compound'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metadata section
                    if show_metadata:
                        st.markdown("### 📊 Text Metadata")
                        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                        
                        with meta_col1:
                            st.metric("Characters", metadata['character_count'])
                            st.metric("Words", metadata['word_count'])
                        
                        with meta_col2:
                            st.metric("Sentences", metadata['sentence_count'])
                            st.metric("Hashtags", metadata['hashtag_count'])
                        
                        with meta_col3:
                            st.metric("Mentions", metadata['mention_count'])
                            st.metric("Emojis", metadata['emoji_count'])
                        
                        with meta_col4:
                            st.metric("Exclamations", metadata['exclamation_count'])
                            st.metric("Questions", metadata['question_count'])
                    
                    # Visualizations
                    if show_visualizations and analysis_method == "Both (Recommended)":
                        st.markdown("### 📈 Visualizations")
                        
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Polarity comparison
                            fig_comparison = go.Figure()
                            fig_comparison.add_trace(go.Bar(
                                x=['TextBlob', 'VADER'],
                                y=[textblob_result['polarity'], vader_result['compound']],
                                marker_color=['skyblue', 'lightgreen'],
                                text=[f"{textblob_result['polarity']:.3f}", f"{vader_result['compound']:.3f}"],
                                textposition='auto'
                            ))
                            fig_comparison.update_layout(
                                title="Sentiment Score Comparison",
                                yaxis_title="Score",
                                yaxis_range=[-1, 1]
                            )
                            fig_comparison.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        with viz_col2:
                            # VADER components
                            fig_vader = go.Figure(data=[
                                go.Bar(x=['Positive', 'Neutral', 'Negative'],
                                      y=[vader_result['positive'], vader_result['neutral'], vader_result['negative']],
                                      marker_color=['green', 'gray', 'red'])
                            ])
                            fig_vader.update_layout(
                                title="VADER Sentiment Components",
                                yaxis_title="Score"
                            )
                            st.plotly_chart(fig_vader, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze!")
    
    elif input_method == "Multiple Texts":
        st.info("Enter multiple texts (one per line) for batch analysis")
        multiple_texts = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Text 1\nText 2\nText 3\n...",
            height=200
        )
        
        if st.button("🔍 Analyze All Texts", type="primary"):
            if multiple_texts:
                texts = [text.strip() for text in multiple_texts.split('\n') if text.strip()]
                
                if texts:
                    with st.spinner(f"Analyzing {len(texts)} texts..."):
                        results = []
                        
                        for i, text in enumerate(texts):
                            cleaned_text = clean_text(text)
                            metadata = get_text_metadata(text)
                            
                            result = {
                                'index': i + 1,
                                'text': text[:50] + "..." if len(text) > 50 else text,
                                'full_text': text
                            }
                            
                            if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                                tb_result = analyze_sentiment_textblob(cleaned_text)
                                result.update({
                                    'textblob_sentiment': tb_result['sentiment'],
                                    'textblob_polarity': tb_result['polarity'],
                                    'textblob_subjectivity': tb_result['subjectivity']
                                })
                            
                            if analysis_method in ["Both (Recommended)", "VADER Only"]:
                                vader_result = analyze_sentiment_vader(cleaned_text)
                                result.update({
                                    'vader_sentiment': vader_result['sentiment'],
                                    'vader_compound': vader_result['compound'],
                                    'vader_positive': vader_result['positive'],
                                    'vader_negative': vader_result['negative']
                                })
                            
                            result.update(metadata)
                            results.append(result)
                        
                        # Create DataFrame
                        df = pd.DataFrame(results)
                        
                        # Display results
                        st.success(f"Analyzed {len(texts)} texts successfully!")
                        
                        # Summary statistics
                        st.markdown("### 📊 Summary Statistics")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                            tb_counts = df['textblob_sentiment'].value_counts()
                            with summary_col1:
                                st.markdown("**TextBlob Results:**")
                                for sentiment, count in tb_counts.items():
                                    percentage = (count / len(df)) * 100
                                    st.write(f"{sentiment}: {count} ({percentage:.1f}%)")
                        
                        if analysis_method in ["Both (Recommended)", "VADER Only"]:
                            vader_counts = df['vader_sentiment'].value_counts()
                            with summary_col2:
                                st.markdown("**VADER Results:**")
                                for sentiment, count in vader_counts.items():
                                    percentage = (count / len(df)) * 100
                                    st.write(f"{sentiment}: {count} ({percentage:.1f}%)")
                        
                        if analysis_method == "Both (Recommended)":
                            agreement = (df['textblob_sentiment'] == df['vader_sentiment']).sum()
                            agreement_pct = (agreement / len(df)) * 100
                            with summary_col3:
                                st.markdown("**Agreement:**")
                                st.write(f"Both agree: {agreement}/{len(df)} ({agreement_pct:.1f}%)")
                        
                        # Data table
                        st.markdown("### 📋 Detailed Results")
                        
                        # Select columns to display
                        display_cols = ['index', 'text']
                        if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                            display_cols.extend(['textblob_sentiment', 'textblob_polarity'])
                        if analysis_method in ["Both (Recommended)", "VADER Only"]:
                            display_cols.extend(['vader_sentiment', 'vader_compound'])
                        
                        if show_metadata:
                            display_cols.extend(['word_count', 'character_count', 'emoji_count'])
                        
                        st.dataframe(df[display_cols], use_container_width=True)
                        
                        # Download option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations for batch analysis
                        if show_visualizations:
                            st.markdown("### 📈 Batch Analysis Visualizations")
                            
                            if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                                # TextBlob sentiment distribution
                                fig_tb_dist = px.pie(
                                    values=tb_counts.values,
                                    names=tb_counts.index,
                                    title="TextBlob Sentiment Distribution"
                                )
                                st.plotly_chart(fig_tb_dist, use_container_width=True)
                            
                            if analysis_method in ["Both (Recommended)", "VADER Only"]:
                                # VADER sentiment distribution
                                fig_vader_dist = px.pie(
                                    values=vader_counts.values,
                                    names=vader_counts.index,
                                    title="VADER Sentiment Distribution"
                                )
                                st.plotly_chart(fig_vader_dist, use_container_width=True)
                else:
                    st.warning("Please enter at least one text to analyze!")
            else:
                st.warning("Please enter some texts to analyze!")
    
    else:  # Upload File
        st.info("Upload a CSV file with a text column for batch analysis")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with a column containing text to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Found {len(df_upload)} rows.")
                
                # Select text column
                text_column = st.selectbox(
                    "Select the column containing text to analyze:",
                    df_upload.columns.tolist()
                )
                
                # Preview data
                st.markdown("### 👀 Data Preview")
                st.dataframe(df_upload.head(), use_container_width=True)
                
                if st.button("🔍 Analyze CSV Data", type="primary"):
                    with st.spinner(f"Analyzing {len(df_upload)} texts from CSV..."):
                        # Process similar to multiple texts but with uploaded data
                        results = []
                        
                        for i, row in df_upload.iterrows():
                            text = str(row[text_column])
                            cleaned_text = clean_text(text)
                            metadata = get_text_metadata(text)
                            
                            result = {
                                'index': i + 1,
                                'text': text[:50] + "..." if len(text) > 50 else text,
                                'full_text': text
                            }
                            
                            if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                                tb_result = analyze_sentiment_textblob(cleaned_text)
                                result.update({
                                    'textblob_sentiment': tb_result['sentiment'],
                                    'textblob_polarity': tb_result['polarity'],
                                    'textblob_subjectivity': tb_result['subjectivity']
                                })
                            
                            if analysis_method in ["Both (Recommended)", "VADER Only"]:
                                vader_result = analyze_sentiment_vader(cleaned_text)
                                result.update({
                                    'vader_sentiment': vader_result['sentiment'],
                                    'vader_compound': vader_result['compound'],
                                    'vader_positive': vader_result['positive'],
                                    'vader_negative': vader_result['negative']
                                })
                            
                            result.update(metadata)
                            results.append(result)
                        
                        # Create results DataFrame
                        df_results = pd.DataFrame(results)
                        
                        st.success(f"Analyzed {len(df_results)} texts from CSV successfully!")
                        
                        # Show summary and detailed results (similar to multiple texts section)
                        # Summary statistics
                        st.markdown("### 📊 Summary Statistics")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        
                        if analysis_method in ["Both (Recommended)", "TextBlob Only"]:
                            tb_counts = df_results['textblob_sentiment'].value_counts()
                            with summary_col1:
                                st.markdown("**TextBlob Results:**")
                                for sentiment, count in tb_counts.items():
                                    percentage = (count / len(df_results)) * 100
                                    st.write(f"{sentiment}: {count} ({percentage:.1f}%)")
                        
                        if analysis_method in ["Both (Recommended)", "VADER Only"]:
                            vader_counts = df_results['vader_sentiment'].value_counts()
                            with summary_col2:
                                st.markdown("**VADER Results:**")
                                for sentiment, count in vader_counts.items():
                                    percentage = (count / len(df_results)) * 100
                                    st.write(f"{sentiment}: {count} ({percentage:.1f}%)")
                        
                        if analysis_method == "Both (Recommended)":
                            agreement = (df_results['textblob_sentiment'] == df_results['vader_sentiment']).sum()
                            agreement_pct = (agreement / len(df_results)) * 100
                            with summary_col3:
                                st.markdown("**Agreement:**")
                                st.write(f"Both agree: {agreement}/{len(df_results)} ({agreement_pct:.1f}%)")
                        
                        # Results table and download
                        st.markdown("### 📋 Analysis Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download results
                        csv_results = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Analysis Results as CSV",
                            data=csv_results,
                            file_name=f"csv_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

# Right sidebar with info
with col2:
    st.markdown("### 🔍 Quick Examples")
    
    example_texts = [
        "I absolutely love this new product! It's amazing! 😍",
        "This service is terrible. Worst experience ever! 😠",
        "The weather today is okay, nothing special.",
        "OMG! Just got the best news ever! So excited! 🎉",
        "Feeling a bit disappointed with the results.",
    ]
    
    for i, example in enumerate(example_texts, 1):
        if st.button(f"Try Example {i}", key=f"example_{i}"):
            st.session_state.example_text = example
    
    st.markdown("### 📈 Interpretation Guide")
    st.markdown("""
    **Polarity Scores:**
    - +1.0: Very Positive
    - +0.5: Positive  
    - 0.0: Neutral
    - -0.5: Negative
    - -1.0: Very Negative
    
    **Subjectivity:**
    - 0.0: Objective (factual)
    - 1.0: Subjective (opinion)
    
    **Confidence:**
    - Higher absolute values = more confident
    - Values near 0 = less confident
    """)
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - VADER works better with social media text
    - TextBlob is good for formal text
    - Use both methods for comparison
    - Consider context for sarcasm
    - Emojis can affect results
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with Streamlit • Powered by TextBlob & VADER</p>
    <p>🎭 Sentiment Analysis Tool v1.0</p>
</div>
""", unsafe_allow_html=True)