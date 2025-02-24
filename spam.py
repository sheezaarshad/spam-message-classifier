import pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string

# Load dataset
df = pd.read_csv('E:/spam.csv', encoding="Latin1")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Rename columns
df.rename(columns={"v1": "target", "v2": "message"}, inplace=True)

def preprocess(text):
    """Preprocess the text by converting to lowercase, removing punctuation and stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing
df['message'] = df['message'].apply(preprocess)

# Convert text to feature vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['target']  

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

def classify_message(message):
    """Classify a new message as spam or ham."""
    message = preprocess(message)  # Preprocess the input message
    message_vector = vectorizer.transform([message])  # Convert the message to vector
    prediction = model.predict(message_vector)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    message = input("Enter the message: ")
    result = classify_message(message)
    print(f"The message is: {result}")
