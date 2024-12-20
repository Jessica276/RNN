# -*- coding: utf-8 -*-
"""NGram.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SS_4Ek0MnQAG10EIspr2_mcMYom13AxV
"""

import gdown
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer

nltk.download('reuters')
nltk.download('punkt')
nltk.download('wordnet')

url = "https://drive.google.com/uc?id=1vnf0SL2ucnABzL5nWUbZRvyto8rS2rMz"
output = "imdb.csv"
gdown.download(url, output)

class Ngram:
    def __init__(self, ngram_range=(1, 1), max_features=1000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self.model = None

    def process_data(self, df, feat_name, target_name):
        x = self.vectorizer.fit_transform(df[feat_name])
        count_vect_df = pd.DataFrame(x.toarray(), columns=self.vectorizer.get_feature_names_out())

        df[target_name] = df[target_name].replace({"positive": 1, "negative": 0})
        df = pd.concat([df[target_name], count_vect_df], axis=1)

        y = df[target_name]
        X = df.drop([target_name], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"\nTraining accuracy: {train_accuracy}")
        print(f"Test accuracy: {test_accuracy}")

def clean_data(text):
  text = text.lower()
  text = re.sub(r"\W", " ", text)
  text = re.sub(r"\d", "", text)
  lemmatizer = WordNetLemmatizer()
  text = lemmatizer.lemmatize(text)

  return text

def main():
    data = pd.read_csv("imdb.csv")
    data["review"] = data["review"].apply(lambda x: clean_data(x))

    ngram_ranges = [(2, 2), (3, 3), (5, 5)]

    for ngram_range in ngram_ranges:
        print(f"\nEntraînement avec des n-grammes de taille : {ngram_range[0]}")
        ngram_model = Ngram(ngram_range=ngram_range, max_features=1000)
        ngram_model.process_data(data, "review", "sentiment")
        ngram_model.train()
        ngram_model.predict()

if __name__ == "__main__":
  main()