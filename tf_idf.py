# -*- coding: utf-8 -*-
"""TF_IDF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1llUvEQ_Xm0ZoDb2_tITsvzZ11LS7ckGT
"""

import gdown
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt_tab')
nltk.download('wordnet')

url = "https://drive.google.com/uc?id=1vnf0SL2ucnABzL5nWUbZRvyto8rS2rMz"
output = "imdb.csv"
gdown.download(url, output)

data = pd.read_csv("imdb.csv")

class TfIdf:
  def __init__(self,df):
    self.df = df

  def preprocess(self,feature_name=None,target_name=None,max_feat=100):
    vectorizer = TfidfVectorizer(
        lowercase = True,
        stop_words = "english",
        max_features = max_feat,
    )

    #Convert the text documents to a TF-IDF matrix
    tf_idf_matrix = vectorizer.fit_transform(self.df[feature_name])
    df_tfidf = pd.DataFrame(tf_idf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    #Replace positive by 1 and negative by 2
    self.df[target_name] = self.df[target_name].replace({"positive":1, "negative":0})

    #Concat dataframe
    self.df = pd.concat([self.df[target_name], df_tfidf], axis=1)

    #Split data
    y = self.df[target_name]
    X = self.df.drop([target_name],axis = 1)

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

    print(f"\n\nTraining accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

def clean_data(text):
  text = re.sub(r"\W"," ",text)
  text = re.sub(r"\d","",text)
  lemmatizer = WordNetLemmatizer()
  text = lemmatizer.lemmatize(text)

  return text

def main():
  data = pd.read_csv("imdb.csv")
  data["review"] = data["review"].apply(lambda x: clean_data(x))

  tfidf = TfIdf(data)
  tfidf.preprocess("review", "sentiment", 1000)

  #Train and evaluate model
  tfidf.train()
  tfidf.predict()

#Entry point for the script
if __name__ == "__main__":
    main()

