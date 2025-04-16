import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch

with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X = model.encode(sentences, show_progress_bar=True)

le = LabelEncoder()
y = le.fit_transform(labels)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

with open('mini_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model trained and saved using MiniLM")
print("CUDA Available:", torch.cuda.is_available())
print("Embedding test:", model.encode(["hello"]))
