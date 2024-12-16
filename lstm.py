import gdown
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


url = "https://drive.google.com/uc?id=1vnf0SL2ucnABzL5nWUbZRvyto8rS2rMz"
output = "imdb.csv"
gdown.download(url, output, quiet=False)


def clean_data(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()


data = pd.read_csv("imdb.csv")
data['review'] = data['review'].apply(clean_data)


label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment']) 


max_vocab_size = 10000 
max_seq_length = 200 

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data['review'])
sequences = tokenizer.texts_to_sequences(data['review'])
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, data['sentiment'], test_size=0.2, random_state=42
)


embedding_dim = 64

model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


num_epochs = 5
batch_size = 32

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=1
)

#Évaluation finale
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Précision sur l'ensemble de test : {accuracy:.2f}")

