import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
import spacy
nlp = spacy.load('en_core_web_sm')
# Now we have to remove unnecessary tokens from Resume column
def preprocess(text):
    doc = nlp(text)
    no_punct = [token.text for token in doc if not token.is_punct]
    return ' '.join(no_punct)

df['clean_resume'] = df['Resume'].apply(preprocess)

# Now we need to give numerical level to Category
'''le = LabelEncoder()
cat_var = ['Category']
for i in cat_var:
    df[i] = le.fit_transform(df[i])'''
    
# Now we need to convert 'clean_resume' text in vectorizer format
from sklearn.cluster import KMeans
tf = TfidfVectorizer()
X = tf.fit_transform(df['clean_resume'])
y = df['Category']

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

clusters = kmeans.predict(X)

X = np.array(X.toarray())

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot the clusters using a scatter plot
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
plt.title('KMeans Clustering (n_clusters=5)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

input_resume = input("Enter Your Resume")
df['cluster'] = clusters
cluster_expertise = {}
for cluster in range(n_clusters):
    cluster_df = df[df['cluster'] == cluster]
    expertise_counts = {}
    for expertise in cluster_df['Category']:
        if expertise in expertise_counts:
            expertise_counts[expertise] += 1
        else:
            expertise_counts[expertise] = 1
    cluster_expertise[cluster] = max(expertise_counts, key=expertise_counts.get)

# Recommend subject matter experts based on the most common expertise within each cluster
for cluster in range(n_clusters):
    if cluster in cluster_expertise:
        print(f"Subject Matter Expertise In: {cluster_expertise[cluster]}")
        break
    else:
        pass
import pickle
with open('kmeans.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

with open('pca.pkl', 'wb') as file:
    pickle.dump(pca, file)    






