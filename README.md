# Machine Learning Classification Dashboard (Streamlit)
        
 Project Overview

This project is an interactive Machine Learning classification web application built using Streamlit.
It allows users to:

Train and evaluate multiple ML models

Handle imbalanced datasets using SMOTE

View performance metrics

Visualize class distributions

Compare models in a clean UI

The application is designed for educational, research, and demo purposes, making ML evaluation easy and visual.

 Features

✅ Interactive Streamlit UI

✅ Multiple classification models

✅ SMOTE applied to balance training data

✅ Model performance comparison

✅ Clean sidebar insights

✅ Smooth and readable outputs (no raw symbols like s)

🧠 Machine Learning Pipeline

Data Loading

Train/Test Split

SMOTE Oversampling

Model Training

Model Evaluation

Visualization in Streamlit

📂 Project Structure
project/
│
├── app.py                 # Main Streamlit application
├── model_utils.py         # Model training & evaluation logic
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── data/
    └── dataset.csv        # Dataset (if included)

⚙️ Technologies Used

Python 3

Streamlit

Scikit-learn

Imbalanced-learn (SMOTE)

Pandas

NumPy

Matplotlib / Seaborn

📊 Sidebar Information

The sidebar displays:

📌 Class distribution info

📌 SMOTE usage confirmation

📌 Model details

Example:

st.sidebar.subheader("📊 Class Distribution (Training)")
st.sidebar.write("Balanced using SMOTE during training")

🧪 Models Used

Depending on your implementation, models may include:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

K-Nearest Neighbors

Decision Tree

Each model is trained and evaluated under the same conditions for fair comparison.

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

All results are displayed in human-readable format (fixed from raw or symbol-based outputs).

▶️ How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit app
streamlit run app.py

3️⃣ Open in browser
http://localhost:8501
