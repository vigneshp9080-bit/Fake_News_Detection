from flask import Flask, render_template, request
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

model_path = "distilbert_v2_indian_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Correct mapping
    return "Real News" if prediction == 0 else "Fake News"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        news_text = request.form["news"]

        if news_text.strip() == "":
            prediction = "Please enter news text."

        elif len(news_text.split()) < 5:
            prediction = "Please enter a valid news sentence."

        else:
            prediction = predict_news(news_text)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)