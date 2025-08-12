#  Fake News Detection Model:

## 📌 Project Overview
This project is an **AI-powered Fake News Detection System** that predicts whether a given news headline or article is **real** or **fake**.  
It uses a **Machine Learning model** trained on a labeled dataset with **TF-IDF vectorization** for feature extraction.  
The app is built using **Streamlit** and can be run locally or with **Docker**.


## 🔄 End-to-End Workflow
1. **Data Collection** – Obtain a labeled dataset of real and fake news.
2. **Data Preprocessing** – Clean and prepare text.
3. **Feature Extraction** – TF-IDF vectorization.
4. **Model Training** – Train a classification model.
5. **Model Evaluation** – Accuracy, Precision, Recall, F1-score.
6. **Model Saving** – Save model & vectorizer with `joblib`.
7. **Frontend** – Streamlit app for predictions.
8. **Deployment** – Run locally or in Docker.


## 📋 Prerequisites
- Python **3.8+**
- Git
- pip
- (Optional) Docker


## Setup, Installation & Run (All-in-One)
```bash
# 1️⃣ Clone the repository
```
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# 2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # On Windows
# source venv/bin/activate   # On Mac/Linux

# 3️⃣ Install dependencies:
pip install -r requirements.txt

# 4️⃣ Run the app locally:
streamlit run app.py

## Run Using the Docker:
# Build Docker image:
docker build -t fake-news-app .

# Run container:
docker run -p 8501:8501 fake-news-app

# Using the Interface:
Enter a news headline or article text in the box.

Click Predict.

The app shows whether it’s Real ✅ or Fake ❌.

# Data Source:
# Kagggle:
https://www.kaggle.com/code/therealsampat/fake-news-detection/notebook.
