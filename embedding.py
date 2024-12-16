import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IMDB
from sklearn.model_selection import train_test_split
import numpy as np
import re
import string
from collections import Counter

# Prétraitement des textes
def preprocess_text(texts):
    stop_words = set(["a", "and", "the", "of", "is", "in", "to", "it", "on", "for", "with", "this", "that", "an", "by", "as"])
    processed_texts = []
    for text in texts:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        tokens = [word for word in text.split() if word not in stop_words]
        processed_texts.append(tokens)
    return processed_texts

# Charger les embeddings GloVe
def load_glove_embeddings(glove_path, embedding_dim):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Dataset IMDb
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, glove_embeddings, embedding_dim):
        self.texts = texts
        self.labels = labels
        self.glove_embeddings = glove_embeddings
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        embedding = self.get_embedding(text)
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def get_embedding(self, tokens):
        vectors = []
        for token in tokens:
            if token in self.glove_embeddings:
                vectors.append(self.glove_embeddings[token])
            else:
                vectors.append(np.random.uniform(-0.01, 0.01, self.embedding_dim))
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)

# Modèle MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

train_iter, test_iter = IMDB(split=('train', 'test'))
texts, labels = [], []
for label, line in train_iter:
    texts.append(line)
    labels.append(1 if label == "pos" else 0)


processed_texts = preprocess_text(texts)

glove_path = "glove.6B.100d.txt"
embedding_dim = 100
glove_embeddings = load_glove_embeddings(glove_path, embedding_dim)


train_texts, val_texts, train_labels, val_labels = train_test_split(processed_texts, labels, test_size=0.2, random_state=42)


train_dataset = IMDBDataset(train_texts, train_labels, glove_embeddings, embedding_dim)
val_dataset = IMDBDataset(val_texts, val_labels, glove_embeddings, embedding_dim)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


input_dim = embedding_dim
hidden_dim = 256
output_dim = 1
model = MLPClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for embeddings, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = torch.round(torch.sigmoid(outputs))
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

# Évaluer le modèle
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for embeddings, labels in val_loader:
        outputs = model(embeddings).squeeze()
        predictions = torch.round(torch.sigmoid(outputs))
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
val_accuracy = total_correct / total_samples
print(f"Validation Accuracy: {val_accuracy:.4f}")
