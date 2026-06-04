# 📰 TruthScan AI — Fake News Detection

> AI-powered fake news detector using **DistilBERT** fine-tuned on Indian news articles.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)

---

## 🚀 Features

- ⚡ **Fast inference** — DistilBERT runs in ~300ms
- 🇮🇳 **Indian News Corpus** — fine-tuned on verified Indian news datasets
- 🎯 **95%+ Accuracy** — state-of-the-art classification
- 💻 **Professional UI** — carbon-themed dark mode with animations
- 🔍 **Instant Verdict** — Real News or Fake News with confidence bar

---

## 🏗️ Project Structure

```
Fake_News_Detection/
├── app.py                    # Flask web application
├── templates/
│   └── index.html            # Frontend UI (Carbon Theme)
├── static/
│   └── style.css             # Full CSS design system
├── fake_news_distilbert.ipynb # Training notebook (DistilBERT v1)
├── fake_news_v2_indian.ipynb  # Training notebook (DistilBERT v2 Indian)
├── requirements.txt           # Python dependencies
└── .gitignore
```

> **Note:** Model weights (`distilbert_v2_indian_model/`) and datasets are excluded from this repo due to size. Download from HuggingFace Hub or train using the provided notebooks.

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/vigneshp9080-bit/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model weights
Place the fine-tuned model folder `distilbert_v2_indian_model/` in the project root.

### 5. Run the app
```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Base Model | `distilbert-base-uncased` |
| Task | Binary Text Classification |
| Labels | `0 = Real News`, `1 = Fake News` |
| Max Token Length | 128 |
| Training Data | Indian News Corpus (50K+ articles) |
| Framework | HuggingFace Transformers + PyTorch |

---

## 📦 Requirements

```
flask
torch
transformers
```

---

## 👨‍💻 Author

**Vignesh P** — [GitHub](https://github.com/vigneshp9080-bit)

---

## 📄 License

MIT License
