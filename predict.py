import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

review = input("Enter review: ")

review_lower = review.lower()

# Rule 1: Too many exclamation marks
if review.count("!") >= 3:
    print("Fake Review")
    exit()

# Rule 2: ALL CAPS spam
if review.isupper():
    print("Fake Review")
    exit()

# Fake patterns
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
        print("Fake Review")
        exit()

# ML prediction
X = vectorizer.transform([review])

prob = model.predict_proba(X)[0][1]

if prob > 0.90:
    print("Fake Review")
else:
    print("Genuine Review")