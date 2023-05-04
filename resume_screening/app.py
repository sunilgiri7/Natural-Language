import streamlit as st
import pandas as pd
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

with open('kmeans.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pred = pickle.load(f)  

with open('pca.pkl', 'rb') as f:
    preprocess = pickle.load(f)  

# Define the Streamlit app
st.title("Resume Expertise Recommender")
def app():
    # Create the input field for the resume
    resume_text = st.text_area("Enter your resume")
    preprocess(resume_text)

    tf = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
    transformed = tf.fit_transform([resume_text])

    # When the user clicks the "Recommend" button, cluster the resume and recommend subject matter experts
    if st.button("Recommend"):
        # Call the recommend_experts function with the user's resume
        model.fit(transformed)
        clusters = model.predict(transformed)   

        df = clusters
        cluster_expertise = {}
        for cluster in range(5):
            cluster_df = df == cluster
            expertise_counts = {}
            for expertise in cluster_df:
                if expertise in expertise_counts:
                    expertise_counts[expertise] += 1
                else:
                    expertise_counts[expertise] = 1
            cluster_expertise[cluster] = max(expertise_counts, key=expertise_counts.get)

        # Display the recommended experts
        for cluster in range(5):
            if cluster in cluster_expertise:
                st.write(f"Subject Matter Expertise In: {cluster_expertise[cluster]}")
                break
            else:
                st.write("No subject matter experts found.")

app()
