# Titanic Survival Prediction System using Machine Learning

A machine learning project that predicts whether a passenger would survive the Titanic disaster based on passenger information such as age, gender, passenger class, fare, and family details.

The project demonstrates the complete machine learning workflow including data exploration, preprocessing, model training, evaluation, and deployment.

---

## Project Overview

This project follows an end-to-end machine learning pipeline:

- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- Streamlit Web Application

---

## Features

- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Categorical Data Encoding
- Multiple Machine Learning Models
- Model Performance Comparison
- Model Evaluation using Classification Metrics
- Trained Model Serialization using Pickle
- Interactive Streamlit Web Application

---

## Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## Dataset

Dataset: Titanic Dataset

Features include:

- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Embarked Port

Target Variable:

- Survived

---

## Project Structure

```text
Titanic-Survival-Prediction-System/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   └── train_model.py
│
├── data/
│   └── Titanic-Dataset.csv
│
├── models/
│   └── titanic_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```
## Development Workflow

```text
main
│
├── feature/eda
├── feature/preprocessing
├── feature/model-training
├── feature/model-evaluation
├── feature/model-saving
└── feature/streamlit-app
```
---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle
- Streamlit
- Git
- GitHub

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd Titanic-Survival-Prediction-System
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

Windows

```bash
.venv\Scripts\activate
```

Install the dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python app/train_model.py
```

---

## Run the Streamlit Application

```bash
streamlit run app/app.py
```

---

## Machine Learning Workflow

```
Dataset
   │
   ▼
EDA
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Model Selection
   │
   ▼
Model Saving
   │
   ▼
Streamlit Deployment
```

---

## Results

The trained machine learning models were compared using accuracy, and the best-performing model was selected for deployment.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Selection
- Model Explainability
- Cloud Deployment
- Docker Support

---

## Learning Outcomes

Through this project, the following concepts were implemented:

- Data Analysis
- Data Cleaning
- Feature Engineering
- Classification Algorithms
- Model Evaluation
- Model Serialization
- Web Application Deployment
- Git Feature Branch Workflow
- Pull Request Based Development

---

## Author

**Vinita Patil**

GitHub: https://github.com/VinitaPatil2005

LinkedIn: https://www.linkedin.com/in/vinita-patil-a87052303/

---
