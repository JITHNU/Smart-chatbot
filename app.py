import streamlit as st
st.set_page_config(page_title="JithBot", page_icon="🤖", layout="wide")

import json
import random
import pickle
#import pyttsx3
import wikipedia
import torch
import os
from gtts import gTTS
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline

#Load trained models
with open("mini_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

#Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
model._target_device = torch.device("cpu")

#QA Pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

#Load knowledge base
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    context = f.read().strip()

if not context:
    context = "Paris is the capital of France."

logo = Image.open("logo.png")  
st.image(logo, width=100)

def update_knowledge_base_from_wikipedia(query, filename="knowledge_base.txt"):
    try:
        summary = wikipedia.summary(query, sentences=5)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n{summary}\n")
        return f"✅ Added information about '{query}' to the knowledge base."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"⚠️ Too many topics found. Try being more specific: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Topic not found. ❌"

def get_intent(text):
    try:
        embedding = model.encode([text])
        probs = clf.predict_proba(embedding)[0]
        max_prob = max(probs)
        pred = clf.predict(embedding)[0]
        label = le.inverse_transform([pred])[0]
        return label, max_prob
    except Exception as e:
        st.error(f"⚠️ Error in intent prediction: {str(e)}")
        return "unknown", 0.0

def search_knowledge_base(question):
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        kb_lines = f.readlines()
    for line in kb_lines:
        if any(word.lower() in line.lower() for word in question.split()):
            return line.strip()
    return ""

def get_response(text):
    try:
        label, prob = get_intent(text)
        
        question_keywords = ["what", "who", "when", "where", "how", "which", "name", "Tell me about"]
        if any(text.lower().startswith(q) for q in question_keywords):
            try:
                answer = qa_pipeline(question=text, context=context)
                return answer["answer"]
            except Exception as e:
                st.error(f"⚠️ QA Pipeline Error: {str(e)}")
                return "❌ Sorry, I couldn't process that. Try asking something else."

        if prob > 0.6:
            for intent in intents["intents"]:
                if intent["tag"] == label:
                    return random.choice(intent["responses"])
        else:
            return "🤖 I’m not sure how to answer that, but I’m learning more every day!"
    except Exception as e:
        st.error(f"⚠️ get_response() failed: {str(e)}")
        return "❌ Something went wrong while generating a response."

def speak(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    st.audio(fp.getvalue(), format='audio/mp3')

#Sidebar
st.sidebar.title("🔧 Controls")
voice_enabled = st.sidebar.checkbox("🔊 Enable Voice", value=True)

st.sidebar.markdown("---")
wiki_query = st.sidebar.text_input("📚 Add Topic to Knowledge Base")
if st.sidebar.button("Fetch Wiki Info"):
    if wiki_query:
        result = update_knowledge_base_from_wikipedia(wiki_query)
        st.sidebar.success(result)
    else:
        st.sidebar.warning("Please enter a topic!")

# Chat history display
st.subheader("💬 Chat History")
if 'history' in st.session_state and st.session_state['history']:
    for sender, message in st.session_state['history']:
        st.markdown(f"**{sender}:** {message}")
else:
    st.info("No messages yet. Start the conversation below!")


if st.button("🧹 Clear Chat"):
    st.session_state['history'] = []
    st.rerun()

st.title("💬 Smart Chatbot with Knowledge Base 🤖")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me anything!")
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

#Chat display
for sender, msg in st.session_state.chat_history:
    with st.chat_message("👤" if sender == "user" else "🤖"):
        st.markdown(msg)
        if sender == "bot" and voice_enabled:
            speak(msg)
