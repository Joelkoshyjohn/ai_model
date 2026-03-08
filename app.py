from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Request format
class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake Review Detection API Running"}


@app.post("/predict")
def predict(review: Review):

    review_vector = vectorizer.transform([review.text])

    prediction = model.predict(review_vector)[0]
    probability = model.predict_proba(review_vector)[0][1]

    result = "Fake Review" if prediction == 1 else "Genuine Review"

    return {
        "review": review.text,
        "prediction": result,
        "ai_probability": float(probability)
    }