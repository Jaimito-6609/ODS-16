# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:01:13 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 01 ODS 16
#==============================================================================
"""
Biermann, Kanie and Kim (2017) highlight the potential of AI in evidence 
collection and analysis to optimize criminal investigations. AI in digital 
forensics allows law enforcement to discover crucial data quickly, increasing 
the chances of securing convictions. This advance improves judicial processes, 
supporting UN SDG 16 for just and peaceful societies.

Biermann, F., Kanie, N., & Kim, R. E. (2017). Global governance by goal-
setting: the novel approach of the UN Sustainable Development Goals. Current 
Opinion In Environmental Sustainability, 26-27, 26-31. 
https://doi.org/10.1016/j.cosust.2017.01.010. 
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Simulation of message data
data = {
    'message': [
        "Meet me tonight at the usual spot",
        "I need a good lawyer, asap!",
        "Don't forget to erase those files",
        "Can you deliver the package tomorrow?",
        "This is a totally normal message, nothing suspicious.",
        "The operation was successful",
        "Is the money transferred?",
        "I haven’t done anything illegal!",
        "Destroy the evidence immediately"
    ],
    'label': [1, 0, 1, 1, 0, 1, 1, 0, 1]  # 1 indicates relevant to the investigation, 0 not relevant
}
df = pd.DataFrame(data)

# Text preprocessing
vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model building
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions))

# Visualization of keywords
features = vectorizer.get_feature_names_out()
feature_counts = np.asarray(X.sum(axis=0)).flatten()
sorted_features = sorted(list(zip(features, feature_counts)), key=lambda x: x[1], reverse=True)
print("Top 5 most frequent keywords in relevant messages:")
for feature, count in sorted_features[:5]:
    print(f"{feature}: {count}")
