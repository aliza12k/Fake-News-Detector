import pandas as pd
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- STEP 1: VERIFY DATA SOURCE ACCESSIBILITY ---
if not os.path.exists('True.csv') or not os.path.exists('Fake.csv'):
    print(" Error: 'True.csv' or 'Fake.csv' not found in folder! Please check your dataset path.")
    exit()

print(">> Loading Dataset Files for Fake News Detector...")
data_real = pd.read_csv('True.csv')
data_fake = pd.read_csv('Fake.csv')

# --- STEP 2: BINARY TARGET LABELING ---
data_real['target_class'] = 0
data_fake['target_class'] = 1

print(">> Merging Real and Fake datasets...")
dataset = pd.concat([data_real, data_fake], axis=0).reset_index(drop=True)
dataset = dataset[['text', 'target_class']]

# Shuffling with seed 101 to alter data split slightly
dataset = dataset.sample(frac=1, random_state=101).reset_index(drop=True)

# --- STEP 3: TEXT PREPROCESSING PIPELINE ---
def process_string(raw_text):
    if isinstance(raw_text, str):
        raw_text = raw_text.lower()
        raw_text = re.sub(r'https?://\S+|www\.\S+', '', raw_text)  
        raw_text = re.sub(r'\[.*?\]', '', raw_text)               
        raw_text = re.sub(r'<.*?>+', '', raw_text)                
        raw_text = re.sub(r'\w*\d\w*', '', raw_text)              
        raw_text = re.sub(r'[^a-zA-Z\s]', '', raw_text)           
        raw_text = raw_text.strip()
        return raw_text
    return ""

print(" Preprocessing text dataset rows (Please wait)...")
dataset['cleaned_text'] = dataset['text'].apply(process_string)

features = dataset['cleaned_text']
labels = dataset['target_class']

# --- STEP 4: DATA SPLIT AND VECTORIZATION ---
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=101)

print(">> Transforming Text Data using TF-IDF Vectorizer...")
tfidf_transformer = TfidfVectorizer(stop_words='english', max_features=4500) 
X_train_transformed = tfidf_transformer.fit_transform(X_train)
X_test_transformed = tfidf_transformer.transform(X_test)

# --- STEP 5: CLASSIFIER TRAINING ---
print(">> Training Logistic Regression Classifier...")
classifier = LogisticRegression(max_iter=1200)
classifier.fit(X_train_transformed, y_train)

# --- STEP 6: MODEL PERFORMANCE EVALUATION ---
predictions = classifier.predict(X_test_transformed)
print("\n===== MODEL PERFORMANCE EVALUATION REPORT =====")
print(f"Overall Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Classification Metrics:")
print(classification_report(y_test, predictions))
print("=====================================================")

# --- STEP 7: SERIALIZATION ---
with open('model.pkl', 'wb') as model_out:
    pickle.dump(classifier, model_out)
with open('vectorizer.pkl', 'wb') as vec_out:
    pickle.dump(tfidf_transformer, vec_out)

print("\n Success! Model weights and vectorizer configuration saved successfully!")
