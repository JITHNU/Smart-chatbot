import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')

with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])

vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(texts)

le = LabelEncoder()
y = le.fit_transform(labels)

model = LogisticRegression()
model.fit(X, y)

def predict_intent(user_input):
    input_vector = vectorizer.transform([user_input])
    predicted_label = model.predict(input_vector)[0]
    tag = le.inverse_transform([predicted_label])[0]
    return tag

def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("🤖 Jith Chatbot is ready! Type 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye 👋")
        break
    try:
        tag = predict_intent(user_input)
        response = get_response(tag)
        print("Bot:", response)
    except:
        print("Bot: Sorry, I didn’t understand that.")