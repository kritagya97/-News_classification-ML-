# import streamlit as st
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt') # If tokenization is used
# nltk.download('wordnet')  # If lemmatization is used
# nltk.download('omw-1.4')
# import joblib
# import re 
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# # Load pre-trained model
# model = joblib.load('news_classification/nb_model.joblib')

# # Streamlit app layout
# st.set_page_config(page_title="News Categorization", page_icon="📚", layout="centered")

# # App header
# st.title("News Categorization App")
# st.markdown("""
#     ### Enter a news headline or text to categorize it
#     This app uses a machine learning model to predict the category of the given news text.
#     """)

# # Add some space
# st.markdown("<hr>", unsafe_allow_html=True)

# # User input text
# input_text = st.text_area("Enter News Text", height=200)

# def preprocess_text(text):
#     # Lowercasing
#     text = text.lower()
#     # Removing special characters and punctuation
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Tokenization
#     words = word_tokenize(text)
#     # Remove stopwords and lemmatize
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     # Join back into a string
#     return ' '.join(words)

# # Function to predict category
# def predict_category(text):
#     # Preprocess the text
#     processed_text = preprocess_text(text)
#     # Make the prediction using the model
#     prediction = model.predict([processed_text])
#     return prediction[0]

# # Display prediction result when button is pressed
# if st.button("Predict Category"):
#     if input_text:
#         category = predict_category(input_text)
#         st.markdown(f"**Predicted Category:** {category}")
#     else:
#         st.warning("Please enter a news text to categorize.")

# # Adding a footer
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("""
#     Built by Kritagya_Ghimire!
#     """)



import streamlit as st
import nltk
import os
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Set custom NLTK data path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Download required NLTK datasets
nltk.download('stopwords', download_dir="nltk_data")
nltk.download('punkt', download_dir="nltk_data")
nltk.download('wordnet', download_dir="nltk_data")
nltk.download('omw-1.4', download_dir="nltk_data")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load pre-trained model
model = joblib.load('nb_model.joblib')  # Ensure this file is in the root directory

# Streamlit app layout
st.set_page_config(page_title="News Categorization", page_icon="📚", layout="centered")

st.title("News Categorization App")
st.markdown("""
    ### Enter a news headline or text to categorize it
    This app uses a machine learning model to predict the category of the given news text.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# User input text
input_text = st.text_area("Enter News Text", height=200)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)  # If still failing, replace with words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_category(text):
    processed_text = preprocess_text(text)
    print(f"Processed Text: {processed_text}")  # Debugging
    prediction = model.predict([processed_text])
    return prediction[0]

if st.button("Predict Category"):
    if input_text:
        category = predict_category(input_text)
        st.markdown(f"**Predicted Category:** {category}")
    else:
        st.warning("Please enter a news text to categorize.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""Built by Kritagya_Ghimire!""")


