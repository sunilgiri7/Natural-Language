# -*- coding: utf-8 -*-
"""Fine Tuning Pretrained Model using transformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Fqw30t7T1_7S3KHRiAdRyLR0GJtpCtjP
"""

import pandas as pd
import numpy as np
df = pd.read_csv('RealOrFakeNews.csv')
df.head()

df.drop(['Unnamed: 0', 'text'], axis=1, inplace=True)

df.head()

df.dropna(inplace=True)

X = list(df['title'])
y = list(df['label'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

X_train

# !pip install transformers

# this model is used to convert text data to numeric
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# giving to tokenizer model
train_encoding = tokenizer(X_train, truncation=True, padding=True)
test_encoding = tokenizer(X_test, truncation=True, padding=True)


 
