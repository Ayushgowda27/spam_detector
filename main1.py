import pandas as pd
import numpy as np
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load and clean dataset
df = pd.read_csv('spam_cleaned.csv')
df[['label', 'text']] = df['v1'].str.split('\t', expand=True)
df = df[['label', 'text']].dropna()

# Print first and last spam & ham messages
print("\n📩 First 5 Spam Messages:")
print(df[df['label'] == 'spam'][['label', 'text']].head())

print("\n📨 Last 5 Spam Messages:")
print(df[df['label'] == 'spam'][['label', 'text']].tail())

print("\n✅ First 5 Ham Messages:")
print(df[df['label'] == 'ham'][['label', 'text']].head())

print("\n📥 Last 5 Ham Messages:")
print(df[df['label'] == 'ham'][['label', 'text']].tail())

# Clean the text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)

nb.fit(X_train, y_train)
lr.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
y_pred_lr = lr.predict(X_test)

# Evaluation function
def evaluate(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Bar graph
    metrics = [acc, prec, rec]
    names = ['Accuracy', 'Precision', 'Recall']
    plt.figure(figsize=(6, 4))
    sns.barplot(x=names, y=metrics, palette='viridis')
    plt.ylim(0, 1)
    plt.title(f'{model_name} - Performance Metrics')
    plt.show()

    # Scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(y_true)), y_true, label='Actual', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.4)
    plt.legend()
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Label (0 = Ham, 1 = Spam)')
    plt.show()

# Run evaluations
evaluate("Naive Bayes", y_test, y_pred_nb)
evaluate("Logistic Regression", y_test, y_pred_lr)
