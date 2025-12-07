

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download 

# ---------------------------------------------------
# Load trained tourism model
# ---------------------------------------------------
MODEL_REPO_ID = "neerajig/tourism_pkg_prediction_model" # Hugging Face Model Hub repo ID
MODEL_FILENAME = "best_tourism_pkg_prediction_model_v1.joblib" # Name of the model file in the repo

# Download the model file from Hugging Face Hub
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model")

# Load the model from the local downloaded path
model = joblib.load(model_path)

st.set_page_config(page_title="Tour Package Purchase Prediction", layout="centered")

st.title("Tour Package Purchase Prediction App")
st.write("""
Fill in the customer details below and the model will predict
whether the customer is likely to **take the tour package (ProdTaken)**.
""")

# --------------------------
# Input widgets (features)
# --------------------------

age = st.number_input("Age", min_value=18, max_value=100, value=35)

typeof_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox("City Tier", [1, 2, 3])

duration_of_pitch = st.number_input(
    "Duration Of Pitch (minutes)", min_value=0, max_value=60, value=10
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Self Employed", "Free Lancer", "Other"]
)

# UI label vs actual value used in model
gender_ui = st.selectbox("Gender", ["Male", "Female"])
gender = "Male" if gender_ui == "Male" else "FeMale"   # matches cleaned training data

number_of_person_visiting = st.number_input(
    "Number Of Person Visiting", min_value=1, max_value=10, value=3
)

number_of_followups = st.number_input(
    "Number Of Followups", min_value=0, max_value=10, value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star", [1, 2, 3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Widowed"]
)

number_of_trips = st.number_input(
    "NumberOfTrips", min_value=0, max_value=50, value=3
)

passport_ui = st.selectbox("Passport", ["No", "Yes"])
passport = 1 if passport_ui == "Yes" else 0   # your data uses 0/1

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

own_car_ui = st.selectbox("Own Car", ["No", "Yes"])
own_car = 1 if own_car_ui == "Yes" else 0

number_of_children_visiting = st.number_input(
    "Number Of Children Visiting", min_value=0, max_value=10, value=1
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"]
)

monthly_income = st.number_input(
    "Monthly Income", min_value=0, max_value=1000000, value=30000, step=100
)

# ---------------------------------------------------
# Assemble input into DataFrame (no ProdTaken here)
# ---------------------------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

st.subheader("Input Preview")
st.dataframe(input_data)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict ProdTaken"):
    # Model expects 'TypeofContact' as numeric (label encoded)
    # Assuming 'TypeofContact' in input_data is a string, we need to encode it.
    # For simplicity, assuming 'Company Invited' is 0 and 'Self Enquiry' is 1 as per prep.py
    input_data['TypeofContact'] = input_data['TypeofContact'].apply(lambda x: 1 if x == 'Self Enquiry' else 0)

    pred = model.predict(input_data)[0]          # 0 or 1
    prob = None
    try:
        prob = model.predict_proba(input_data)[:, 1][0]  # probability of ProdTaken = 1
    except Exception as e:
        st.error(f"Error getting probabilities: {e}")

    result_text = "Customer WILL take the package (ProdTaken = 1)" if pred == 1 \
                  else "Customer will NOT take the package (ProdTaken = 0)"

    st.subheader("Prediction Result:")
    st.success(result_text)

    if prob is not None:
        st.write(f"Probability of taking the package: **{prob:.2%}**")
