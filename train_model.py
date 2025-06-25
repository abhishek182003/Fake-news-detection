# train_model.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Provide proper meaningful news text samples
texts = [
    "Breaking: Aliens invade New York City!",     # Fake
    "The government announced a new education policy.",  # Real
    "Scientists discover unicorns in Africa!",    # Fake
    "Apple launches new iPhone with AI features." # Real
]
labels = ["fake", "real", "fake", "real"]

# Train TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
