import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#  I want to Configure the dashboard page by alligning to center and including title

st.set_page_config(page_title="Diabetes Dashboard", layout="centered")

st.title("🩺 Diabetes Prediction Dashboard")
st.write("Machine Learning Dashboard for Diabetes Risk Prediction.")

# Then i load the dataset and and XGBoost Model

df = pd.read_csv("diabetes_prediction_dataset.csv")
model = joblib.load("xgboost_model.pkl")

# I added on sidebar fileter to the dahboard

st.sidebar.header("📊 Data Filters")

age_range = st.sidebar.slider("Age Range", 0, 100, (20, 80))
bmi_range = st.sidebar.slider("BMI Range", 10.0, 50.0, (18.5, 35.0))
glucose_range = st.sidebar.slider("Blood Glucose Range", 70, 300, (90, 180))
gender_filter = st.sidebar.selectbox("Gender Filter", ["All", "Female", "Male", "Other"])

# Next, I added on filters

filtered_df = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["bmi"].between(bmi_range[0], bmi_range[1])) &
    (df["blood_glucose_level"].between(glucose_range[0], glucose_range[1]))
]

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]


# I added on Diabetes Class Distribution Plot as 1st visual

st.subheader("Diabetes Class Distribution")

fig1, ax1 = plt.subplots()
filtered_df["diabetes"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
st.pyplot(fig1)

# I added on Age Distribution Histogram as 2nd visual

st.subheader("Age Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(filtered_df["age"], bins=20, edgecolor="black")
st.pyplot(fig2)

# I added on BMI vs Blood glucose level as 3rd visual

st.subheader("BMI vs Blood Glucose by Diabetes")

fig3, ax3 = plt.subplots()

scatter = ax3.scatter(
    filtered_df["bmi"],
    filtered_df["blood_glucose_level"],
    c=filtered_df["diabetes"],
    cmap="coolwarm",
    alpha=0.6
)

ax3.set_xlabel("BMI")
ax3.set_ylabel("Blood Glucose Level")

legend = ax3.legend(*scatter.legend_elements(),
                    title="Diabetes (0=No, 1=Yes)")
ax3.add_artist(legend)

st.pyplot(fig3)

# I added prediction section to select variables based on patients and input predicted diabetes as low and high

st.subheader("🧠 Diabetes Prediction")

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.slider("Age (Patient)", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.slider("HbA1c Level", 3.0, 10.0, 5.5)
glucose = st.slider("Blood Glucose Level", 70, 300, 120)

gender_map = {"Female": 0, "Male": 1, "Other": 2}
smoking_map = {
    "never": 0,
    "current": 1,
    "former": 2,
    "ever": 3,
    "not current": 4,
    "No Info": 5
}

if st.button("Predict"):

    input_data = pd.DataFrame([[
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        smoking_map[smoking],
        bmi,
        hba1c,
        glucose
    ]], columns=[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Result")

    st.metric("Risk Probability", f"{prob:.2%}")

    if prediction == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")