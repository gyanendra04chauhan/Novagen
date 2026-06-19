# 🧬 Novagen — AI Health Risk Predictor

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://novagen-ggtmyzn5wrlj29n8jneq4m.streamlit.app/)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat&logo=streamlit)](https://streamlit.io)

> AI-powered disease risk prediction using 22 health & lifestyle parameters — **94.3% accuracy**

---

## 🚀 Live Demo

**👉 [Try Novagen Live](https://novagen-ggtmyzn5wrlj29n8jneq4m.streamlit.app/)**

---

## 🎯 Overview

**Novagen** is a health risk assessment web app powered by a Random Forest Classifier trained on **9,549 patient records**. Enter your health parameters and get an instant disease risk probability with visual insights.

---

## 🤖 ML Model

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Accuracy | **94.3%** |
| Dataset Size | 9,549 records |
| Train/Test Split | 80% / 20% |
| Preprocessing | StandardScaler (Z-score) |

### Models Compared

| Model | Accuracy |
|---|---|
| Logistic Regression | ~91% |
| K-Nearest Neighbors (k=5) | ~90% |
| Voting Classifier | ~93% |
| Gradient Boosting | ~93% |
| **Random Forest ✅ (Selected)** | **94.3%** |

---

## 📊 Input Features (22 total)

| Category | Features |
|---|---|
| Vitals | Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate |
| Lifestyle | Sleep Hours, Exercise Hours, Water Intake, Stress Level |
| Habits | Smoking, Alcohol, Diet Quality, Mental Health, Physical Activity |
| History | Medical History, Allergies |
| Categorical | Diet Type (Vegan/Vegetarian/Non-Veg), Blood Group (A/AB/B/O) |

---

## 🖥️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/novagen.git
cd novagen

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
novagen/
├── app.py                  # Streamlit frontend + model training
├── Novagen.ipynb           # Experiments & model comparison notebook
├── novagen_dataset.csv     # Dataset (9,549 records, 23 columns)
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Dark theme config
└── README.md
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Not a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

**Gyanendra** — B.Tech CSE (AI & ML), AITM Kanpur (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR_USERNAME)
