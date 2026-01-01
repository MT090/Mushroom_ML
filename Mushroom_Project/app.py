import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load saved components
# ----------------------------
model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
model_results = pickle.load(open("model_results.pkl", "rb"))

# ----------------------------
# Feature list
# ----------------------------
features = [
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
# Human-readable mappings
# ----------------------------
value_mappings = {
    "cap-shape": {
        "bell": "b", "conical": "c", "convex": "x",
        "flat": "f", "knobbed": "k", "sunken": "s"
    },
    "cap-surface": {
        "fibrous": "f", "grooves": "g",
        "scaly": "y", "smooth": "s"
    },
    "cap-color": {
        "brown": "n", "buff": "b", "cinnamon": "c",
        "gray": "g", "green": "r", "pink": "p",
        "purple": "u", "red": "e", "white": "w", "yellow": "y"
    },
    "bruises": {"yes": "t", "no": "f"},
    "odor": {
        "almond": "a", "anise": "l", "creosote": "c",
        "fishy": "y", "foul": "f", "musty": "m",
        "none": "n", "pungent": "p", "spicy": "s"
    },
    "gill-attachment": {"attached": "a", "free": "f"},
    "gill-spacing": {"close": "c", "crowded": "w"},
    "gill-size": {"broad": "b", "narrow": "n"},
    "gill-color": {
        "black": "k", "brown": "n", "buff": "b",
        "chocolate": "h", "gray": "g", "green": "r",
        "orange": "o", "pink": "p", "purple": "u",
        "red": "e", "white": "w", "yellow": "y"
    },
    "stalk-shape": {"enlarging": "e", "tapering": "t"},
    "stalk-surface-above-ring": {
        "fibrous": "f", "scaly": "y", "silky": "k", "smooth": "s"
    },
    "stalk-surface-below-ring": {
        "fibrous": "f", "scaly": "y", "silky": "k", "smooth": "s"
    },
    "ring-number": {"none": "n", "one": "o", "two": "t"},
    "ring-type": {
        "cobwebby": "c", "evanescent": "e", "flaring": "f",
        "large": "l", "none": "n", "pendant": "p",
        "sheathing": "s", "zone": "z"
    },
    "habitat": {
        "grasses": "g", "leaves": "l", "meadows": "m",
        "paths": "p", "urban": "u", "waste": "w", "woods": "d"
    }
}

# ============================
# Best model
# ============================
best_model_name = max(
    model_results,
    key=lambda k: (
        model_results[k]['recall'],
        model_results[k]['precision'],
        model_results[k]['accuracy']
    )
)

# ============================
# SIDEBAR
# ============================
st.sidebar.title("📊 Model Comparison")
st.sidebar.markdown(f"**Best Model: {best_model_name}** ✅")

for name, scores in model_results.items():
    st.sidebar.subheader(name)
    st.sidebar.write(f"Accuracy  : {scores['accuracy']:.3f}")
    st.sidebar.write(f"Precision : {scores['precision']:.3f}")
    st.sidebar.write(f"Recall    : {scores['recall']:.3f}")
    st.sidebar.markdown("---")

st.sidebar.subheader("📊 Class Distribution (Training)")
st.sidebar.write("Balanced using SMOTE during training")

# ============================
# MAIN UI
# ============================
st.title("🍄 Mushroom Classification System")

st.write("""
This system predicts whether a mushroom is **Edible** or **Poisonous**
using machine learning models trained on **balanced data**.
""")

st.subheader("Enter Mushroom Features")

# ----------------------------
# User Input (Human readable)
# ----------------------------
user_input = {}

for feature in features:
    selected_label = st.selectbox(
        feature.replace("-", " ").title(),
        list(value_mappings[feature].keys())
    )
    user_input[feature] = value_mappings[feature][selected_label]

# ----------------------------
# Encode for model
# ----------------------------
input_df = pd.DataFrame([user_input])

for col in input_df.columns:
    input_df[col] = encoders[col].transform(input_df[col])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)
    result = encoders['class'].inverse_transform(prediction)[0]

    if result == 'e':
        st.success("✅ Edible Mushroom")
    else:
        st.error("❌ Poisonous Mushroom")
