from flask import Flask, render_template, request
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# In production (HuggingFace Spaces), set HF_MODEL_ID env variable to
# your uploaded model e.g. "vigneshp9080-bit/distilbert-fake-news-indian"
# Locally, it falls back to the local model folder.
model_path = os.environ.get("HF_MODEL_ID", "distilbert_v2_indian_model")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ── Medical Domain Boundary ──────────────────────────────────────────────────
MEDICAL_KEYWORDS = [
    # Diseases & conditions
    "disease", "virus", "bacteria", "infection", "cancer", "diabetes", "hypertension",
    "covid", "corona", "flu", "influenza", "ebola", "malaria", "tuberculosis", "hiv",
    "aids", "hepatitis", "dengue", "cholera", "typhoid", "pneumonia", "asthma",
    "arthritis", "alzheimer", "parkinson", "epilepsy", "stroke", "heart attack",
    "cardiac", "tumor", "leukemia", "anemia", "obesity", "depression", "anxiety",
    "schizophrenia", "autism", "adhd", "dementia", "meningitis", "sepsis",
    # Medical professionals & settings
    "doctor", "physician", "surgeon", "nurse", "hospital", "clinic", "pharmacy",
    "emergency room", "icu", "operation theatre", "ambulance", "patient", "ward",
    "medical", "healthcare", "health care", "medicine", "medication", "prescription",
    "diagnosis", "treatment", "therapy", "surgery", "operation", "procedure",
    # Drugs & treatments
    "drug", "vaccine", "vaccination", "antibiotic", "antiviral", "chemotherapy",
    "radiation", "insulin", "paracetamol", "aspirin", "ibuprofen", "hydroxychloroquine",
    "remdesivir", "ivermectin", "dose", "dosage", "side effect", "clinical trial",
    "fda", "who", "cdc", "icmr", "aiims",
    # Body parts & biology
    "blood", "lung", "liver", "kidney", "brain", "heart", "bone", "muscle", "skin",
    "gene", "dna", "rna", "protein", "cell", "immune", "antibody", "pathogen",
    "symptom", "symptoms", "fever", "cough", "pain", "fatigue", "nausea", "vomiting",
    "headache", "dizziness", "rash", "swelling", "bleeding",
    # Health topics
    "nutrition", "diet", "mental health", "public health", "epidemic", "pandemic",
    "outbreak", "quarantine", "lockdown health", "sanitizer", "mask", "ppe",
    "ventilator", "oxygen", "x-ray", "mri", "scan", "ultrasound", "biopsy",
    "blood pressure", "cholesterol", "sugar level", "immunity", "ayurveda", "homeopathy",
]

def is_medical_domain(text: str) -> bool:
    """Return True if the text contains medical/health-related keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in MEDICAL_KEYWORDS)

# ─────────────────────────────────────────────────────────────────────────────

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

        elif not is_medical_domain(news_text):
            prediction = "OUT_OF_DOMAIN"

        else:
            prediction = predict_news(news_text)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)