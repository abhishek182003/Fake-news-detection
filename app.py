# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    data = vectorizer.transform([news])
    result = model.predict(data)[0]
    return render_template("index.html", prediction=result, news=news)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

