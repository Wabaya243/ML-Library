import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem  import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Charger une partie du dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.sample(frac=0.4, random_state=42).reset_index(drop=True)

# Garder les bonnes colonnes
df = df[["label", "text"]]
df.columns = ["label", "message"]

# Nettoyer les labels avant conversion
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Convertir en 0 / 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text) #supprimé les characteres specials
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words] 
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


def predict_email(email):
    processed_text = preprocess_text(email)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not spam"

#example
email = "Congratulations! you've won a free Iphone. click here to claim now."
result = predict_email(email)
print(result)

