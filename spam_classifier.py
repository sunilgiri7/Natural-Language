import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
df = pd.read_csv('SMSSpamCollection', sep='\t',  names=["label", "message"])
ps = PorterStemmer()
lm = WordNetLemmatizer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize()(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

dum = pd.get_dummies(df['label'])
y = dum.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)