import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE

# ----------------------------
# Load dataset
# ----------------------------
data = pd.read_csv("mushroom.csv")

# ----------------------------
# Check class balance BEFORE
# ----------------------------
print("\n=== Dataset Balance Check (Before Balancing) ===")
print(data['class'].value_counts())
print("\nPercentages:")
print(data['class'].value_counts(normalize=True) * 100)

# ----------------------------
# Selected features (15)
# ----------------------------
selected_features = [
    'cap-shape',
    'cap-surface',
    'cap-color',
    'bruises',
    'odor',
    'gill-attachment',
    'gill-spacing',
    'gill-size',
    'gill-color',
    'stalk-shape',
    'stalk-surface-above-ring',
    'stalk-surface-below-ring',
    'ring-number',
    'ring-type',
    'habitat'
]

# ----------------------------
# Encode categorical features
# ----------------------------
encoders = {}

for column in selected_features + ['class']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# ----------------------------
# Train-test split (stratified)
# ----------------------------
X = data[selected_features]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# Handle class imbalance using SMOTE
# ----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\n=== Dataset Balance Check (After SMOTE) ===")
print(pd.Series(y_train).value_counts())

# ----------------------------
# Logistic Regression model
# ----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_metrics = {
    "accuracy": accuracy_score(y_test, lr_pred),
    "precision": precision_score(y_test, lr_pred),
    "recall": recall_score(y_test, lr_pred),
    "model": lr
}

# ----------------------------
# Random Forest model
# ----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_metrics = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "precision": precision_score(y_test, rf_pred),
    "recall": recall_score(y_test, rf_pred),
    "model": rf
}

# ----------------------------
# Compare models
# Priority: Recall → Precision → Accuracy
# ----------------------------
models = {
    "Logistic Regression": lr_metrics,
    "Random Forest": rf_metrics
}

best_model_name = max(
    models,
    key=lambda m: (
        models[m]['recall'],
        models[m]['precision'],
        models[m]['accuracy']
    )
)

best_model = models[best_model_name]['model']

# ----------------------------
# Save trained components
# ----------------------------
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(models, open("model_results.pkl", "wb"))

# ----------------------------
# Print model comparison
# ----------------------------
print("\n=== Model Comparison ===")
for name, scores in models.items():
    print(
        f"{name}: Accuracy={scores['accuracy']:.3f}, "
        f"Precision={scores['precision']:.3f}, "
        f"Recall={scores['recall']:.3f}"
    )

print("\nBest Model Selected:", best_model_name)
print("Training Finished Successfully")
