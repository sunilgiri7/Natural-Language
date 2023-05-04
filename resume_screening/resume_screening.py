import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

df = pd.read_csv('resume.csv')
df.shape

df['length'] = df['Resume'].str.len()

df.isnull().sum()

df['Category'].unique()
df.nunique()

plt.figure(figsize=(5,5))
plt.xticks(rotation=90)
sns.countplot(data=df, y='Category', palette='Reds')

plt.figure(figsize=(12,6))
sns.distplot(df['length']).set_title('resume length distrbution')

df['Resume'][0]
nlp = spacy.load('en_core_web_sm')
# Now we have to remove unnecessary tokens from Resume column
def preprocess(text):
    doc = nlp(text)
    no_punct = [token.text for token in doc if not token.is_punct]
    return ' '.join(no_punct)

df['clean_resume'] = df['Resume'].apply(preprocess)

# Now we need to give numerical level to Category
le = LabelEncoder()
cat_var = ['Category']
for i in cat_var:
    df[i] = le.fit_transform(df[i])
    
# Now we need to convert 'clean_resume' text in vectorizer format
cleanText = df['clean_resume']
y = df['Category']

tf = TfidfVectorizer(sublinear_tf=True,
    stop_words='english',
    max_features=1500)
X = tf.fit_transform(cleanText)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Accuracy of KNearest on training set", knn.score(X_train, y_train))
print("Accuracy of KNearest on test set", knn.score(X_test, y_test))

print("accuracy score", accuracy_score(y_test, y_pred))
print("MSE loss", mean_squared_error(y_test, y_pred))

mse_loss = mean_squared_error(y_test, y_pred)

data = {'X_test': y_pred, 'y_pred':mse_loss}
dfs = pd.DataFrame(data)
sns.lineplot(data=dfs)

class JobPredictor:
    def __init__(self) -> None:
        self.le = le
        self.word_vectorizer = tf
        self.clf = knn

    def predict(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted = self.clf.predict(feature)
        resume_position = self.le.inverse_transform(predicted)[0]
        return resume_position

    def predict_proba(self, resume):
        feature = self.word_vectorizer.transform([resume])
        predicted_prob = self.clf.predict_proba(feature)
        return predicted_prob[0]
    
job_description = '''

Summary: Experienced software engineer with expertise in full-stack web development and experience leading development teams. Proficient in JavaScript, Python, React, and Node.js.

Experience:

Senior Software Engineer at XYZ Corp, 2018-present
Full Stack Developer at ABC Inc, 2016-2018
Software Developer at DEF Corp, 2014-2016
Education:

Bachelor of Science in Computer Science, XYZ University, 2014
Skills: JavaScript, Python, React, Node.js, Git, Agile Methodologies
'''
result = JobPredictor().predict(job_description)
print(f"Position: {result}")