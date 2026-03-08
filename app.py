from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class Review(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake Review Detection API Running"}


@app.post("/predict")
def predict(review: Review):

    text = review.text
    review_lower = text.lower()

    # Rule 1
    if text.count("!") >= 3:
        return {
            "review": text,
            "prediction": "Fake Review",
            "ai_probability": 0.95
        }

    # Rule 2
    if text.isupper():
        return {
            "review": text,
            "prediction": "Fake Review",
            "ai_probability": 0.94
        }

    fake_patterns = [
        "most perfect place ever",
        "best experience of my life",
        "beyond amazing",
        "absolutely unbelievable",
        "cannot imagine a better hotel",
        "perfect in every way",
        "best hotel ever",
        "everything was absolutely amazing"
    ]

    for pattern in fake_patterns:
        if pattern in review_lower:
            return {
                "review": text,
                "prediction": "Fake Review",
                "ai_probability": 0.93
            }

    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]

    if prob > 0.90:
        prediction = "Fake Review"
    else:
        prediction = "Genuine Review"

    return {
        "review": text,
        "prediction": prediction,
        "ai_probability": float(prob)
    }