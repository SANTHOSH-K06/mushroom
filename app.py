import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Mushroom Classifier 🍄", layout="centered")

st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is **Edible** or **Poisonous**")

# Load model
with open("mushroom_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset (only to get column names & categories)
df = pd.read_csv("mushroom_classification.csv")

# Separate features
X = df.drop("class", axis=1)

st.subheader("🔎 Enter Mushroom Features")

user_input = {}

# Create dropdowns for each feature
for col in X.columns:
    user_input[col] = st.selectbox(
        col,
        options=sorted(df[col].unique())
    )

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Encode input same way as training
encoder = LabelEncoder()
for col in input_df.columns:
    encoder.fit(df[col])
    input_df[col] = encoder.transform(input_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ This mushroom is **EDIBLE**")
    else:
        st.error("☠️ This mushroom is **POISONOUS**")