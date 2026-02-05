import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mushroom Classifier 🍄",
    layout="centered"
)

st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is **Edible** or **Poisonous**")

# ---------------- FILE PATHS ----------------
MODEL_PATH = "mushroom_model.pkl"
DATA_PATH = "mushroom_classification.csv"

# ---------------- FILE CHECKS ----------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ mushroom_model.pkl not found")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("❌ mushroom_classification.csv not found")
    st.stop()

# ---------------- LOAD MODEL ----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# Separate features
X = df.drop("class", axis=1)

st.subheader("🔎 Enter Mushroom Features")

# ---------------- USER INPUT ----------------
user_input = {}
for col in X.columns:
    user_input[col] = st.selectbox(
        label=col.replace("-", " ").title(),
        options=sorted(df[col].unique())
    )

input_df = pd.DataFrame([user_input])

# ---------------- ENCODING ----------------
encoded_df = input_df.copy()

for col in encoded_df.columns:
    le = LabelEncoder()
    le.fit(df[col])          # fit encoder on training data
    encoded_df[col] = le.transform(encoded_df[col])

# ---------------- PREDICTION ----------------
if st.button("Predict 🍄"):
    prediction = model.predict(encoded_df)[0]

    if prediction == 0:
        st.success("✅ This mushroom is **EDIBLE**")
    else:
        st.error("☠️ This mushroom is **POISONOUS**")
