# Fake News Detection Model

##  Project Overview

This is an **AI-powered Fake News Detection System** that predicts whether a given news headline or article is **Real** or **Fake**.
It uses a **Machine Learning model** trained on a labeled dataset with **TF-IDF vectorization** for feature extraction.
The application is built with **Streamlit** and runs locally.

## End-to-End Workflow

1. **Data Collection** – Obtain a labeled dataset of real and fake news.
2. **Data Preprocessing** – Clean and prepare text.
3. **Feature Extraction** – Apply TF-IDF vectorization.
4. **Model Training** – Train a classification model.
5. **Model Evaluation** – Evaluate using Accuracy, Precision, Recall, and F1-score.
6. **Model Saving** – Save the trained model and vectorizer using `joblib`.
7. **Frontend** – Streamlit app for predictions.
8. **Deployment** – Run locally.

## Prerequisites

* Python **3.8+**
* Git
* pip

## Setup, Installation & Run

```
# Step 1: Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Step 2: Create a virtual environment
python -m venv venv

# Step 3: Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Step 4: Install required dependencies
pip install -r requirements.txt

# Step 5: Run the Streamlit app
streamlit run app.py
```
## Data Source

Dataset used for training:
[Kaggle – Fake News Detection](https://www.kaggle.com/code/therealsampat/fake-news-detection/notebook)
