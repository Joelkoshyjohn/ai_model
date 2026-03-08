import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("reviews_dataset.csv")

# Features and labels
X = data["review"]
y = data["label"]

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")

X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()

model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully")