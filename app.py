from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Allow requests from any origin
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

    return {
        "review": review.text,
        "prediction": int(prediction),
        "ai_probability": float(probability)
    }