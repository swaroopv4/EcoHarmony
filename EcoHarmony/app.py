import streamlit as st
from joblib import load
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
import speech_recognition as sr

# Load the model and vectorizer
clf = load('hate_speech_model.pkl')
cv = load('count_vectorizer.pkl')

def hate_speech_detection(tweet):
    data = cv.transform([tweet]).toarray()
    st.write(f"Transformed Input: {data}")  # Debug statement to print the transformed input
    prediction = clf.predict(data)
    st.write(f"Model Prediction: {prediction}")  # Debug statement to print the model prediction
    return prediction[0]

# Language support
LANGUAGES = {
    'Auto-Detect': 'auto',
    'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az',
    'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca',
    'Cebuano': 'ceb', 'Chichewa': 'ny', 'Chinese (Simplified)': 'zh-cn', 'Chinese (Traditional)': 'zh-tw',
    'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
    'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy',
    'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'iw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn',
    'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it',
    'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko',
    'Kurdish (Kurmanji)': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv',
    'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms',
    'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn',
    'Myanmar (Burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Odia': 'or', 'Pashto': 'ps',
    'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru',
    'Samoan': 'sm', 'Scots Gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd',
    'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su',
    'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr',
    'Turkmen': 'tk', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}

# Custom CSS for better styling
st.markdown("""
    <style>
    body {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSp06-wZ-zr9zFlUlSVRtein-UDYtnaz_RIDw&s');
        background-size: cover;
        color: white; /* Adjust text color */
    }
    .main {
        background-color: rgba(0, 0, 0, 0.5); /* Adjust background color */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
    }
    h1 {
        color: white;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    textarea {
        width: 100%;
        height: 150px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #dddddd;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
        color: black; /* Adjust text color */
        background-color: white; /* Adjust background color */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .prediction {
        font-size: 18px;
        color: white;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Language selection
language = st.sidebar.selectbox("Select Language", list(LANGUAGES.keys()))

# App title and description
st.title("EcoHarmony")
st.markdown("Enter a tweet in the text area below to determine if it contains hate speech. This application supports multiple languages and provides a comprehensive analysis to identify potentially harmful or offensive content.")

# Initialize user input
user_input = ""

# Search mode
search_mode = st.sidebar.radio("Select Search Mode", ('Text Search', 'Voice Search'))

if search_mode == "Voice Search":
    # Language options for voice search
    input_language = st.selectbox('Select Input Language', list(LANGUAGES.keys()))
    input_language_code = LANGUAGES[input_language]

    if st.button("Ask", key="voice_search_button"):
        st.subheader("Voice Search")
        with st.spinner("Listening..."):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            try:
                recognized_text = r.recognize_google(audio, language=input_language_code)
                
                # Auto-detect the language of recognized text
                detected_language = detect(recognized_text)
                
                if detected_language != 'en':
                    # Translate recognized text to English
                    translator = GoogleTranslator(source=detected_language, target='en')
                    recognized_text = translator.translate(recognized_text)
                st.write(f"Recognized Text in English: {recognized_text}")
                user_input = recognized_text
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError:
                st.error("Could not request results")
        # Directly proceed to detection after recognizing voice input
        if user_input:
            prediction = hate_speech_detection(user_input)
else:
    # Text Search
    user_input = st.text_area("Enter a Tweet:", key="text_search")

    # Detect button for text search
    if st.button("Detect", key="detect_button"):
        if user_input:
            # Detect language if auto-detect is selected
            if language == 'Auto-Detect':
                detected_language = detect(user_input)
            else:
                detected_language = LANGUAGES[language]
            
            # Translate to English if necessary
            if detected_language != 'en':
                translator = GoogleTranslator(source=detected_language, target='en')
                user_input = translator.translate(user_input)
            
            prediction = hate_speech_detection(user_input)
