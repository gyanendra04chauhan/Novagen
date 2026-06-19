# 🧬 Novagen — AI Health Risk Predictor

A machine learning web application that predicts disease risk based on 22 health and lifestyle parameters.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)

![Novagen Screenshot](https://via.placeholder.com/1000x500/0a0e1a/38bdf8?text=Novagen+Health+Risk+Predictor)

---

## 🎯 Overview

**Novagen** is a health risk assessment tool powered by a Random Forest Classifier trained on 9,549 patient records with **94.3% accuracy**. It accepts 22 health and lifestyle features and outputs a disease risk probability with visual insights.

---

## 🤖 ML Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 200 trees |
| Accuracy | **94.3%** |
| Dataset Size | 9,549 records |
| Train/Test Split | 80% / 20% |
| Preprocessing | StandardScaler |

### Models Compared
| Model | Accuracy |
|---|---|
| Logistic Regression | ~91% |
| K-Nearest Neighbors (k=5) | ~90% |
| **Random Forest** ✅ | **94.3%** |
| Voting Classifier | ~93% |
| Gradient Boosting | ~93% |

---

## 📊 Features Used

**Vitals:** Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate

**Lifestyle:** Sleep Hours, Exercise Hours, Water Intake, Stress Level

**Habits:** Smoking, Alcohol, Diet Quality, Mental Health, Physical Activity

**History:** Medical History, Allergies

**Categorical:** Diet Type (Vegan/Vegetarian/Non-Veg), Blood Group (A/AB/B/O)

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/novagen.git
cd novagen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file:** `app.py`
5. Click **Deploy** — live in ~2 minutes!

---

## 📁 Project Structure

```
novagen/
├── app.py                  # Main Streamlit application
├── Novagen.ipynb           # Model training notebook
├── novagen_dataset.csv     # Training dataset
├── model.pkl               # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
├── feature_names.pkl       # Feature names list
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme config
└── README.md
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional medical diagnosis or advice.

---

## 👨‍💻 Author

**Gyanendra** — B.Tech CSE (AI & ML), AITM Kanpur (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)
