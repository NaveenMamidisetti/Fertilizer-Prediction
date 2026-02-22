import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# LOAD SAVED FILES
# -----------------------------

model = pickle.load(open("rf_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🌾 Fertilizer Recommendation System")
st.write("Enter soil and crop details to get fertilizer recommendation")

# -----------------------------
# INPUT SECTION
# -----------------------------

soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Silt", "Loamy"])
soil_ph = st.number_input("Soil pH", 4.0, 9.0, 6.5)
soil_moisture = st.number_input("Soil Moisture")
organic_carbon = st.number_input("Organic Carbon")
electrical_conductivity = st.number_input("Electrical Conductivity")

nitrogen = st.number_input("Nitrogen Level")
phosphorus = st.number_input("Phosphorus Level")
potassium = st.number_input("Potassium Level")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
rainfall = st.number_input("Rainfall")

crop_type = st.selectbox(
    "Crop Type",
    ["Rice","Wheat","Maize","Cotton","Sugarcane","Potato","Tomato"]
)

crop_stage = st.selectbox(
    "Crop Growth Stage",
    ["Sowing","Vegetative","Flowering","Harvest"]
)

season = st.selectbox("Season", ["Kharif","Rabi","Zaid"])
irrigation = st.selectbox("Irrigation Type", ["Canal","Drip","Rainfed","Sprinkler"])
previous_crop = st.selectbox(
    "Previous Crop",
    ["Rice","Wheat","Maize","Cotton","Sugarcane","Potato","Tomato"]
)

region = st.selectbox("Region", ["North","South","East","West","Central"])

fert_last = st.number_input("Fertilizer Used Last Season")
yield_last = st.number_input("Yield Last Season")

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Fertilizer"):

    # Create DataFrame
    input_data = pd.DataFrame([{
        "Soil_Type": soil_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Nitrogen_Level": nitrogen,
        "Phosphorus_Level": phosphorus,
        "Potassium_Level": potassium,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": crop_stage,
        "Season": season,
        "Irrigation_Type": irrigation,
        "Previous_Crop": previous_crop,
        "Region": region,
        "Fertilizer_Used_Last_Season": fert_last,
        "Yield_Last_Season": yield_last
    }])

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    input_data['N_P_Ratio'] = input_data['Nitrogen_Level'] / (input_data['Phosphorus_Level'] + 1)
    input_data['N_K_Ratio'] = input_data['Nitrogen_Level'] / (input_data['Potassium_Level'] + 1)
    input_data['P_K_Ratio'] = input_data['Phosphorus_Level'] / (input_data['Potassium_Level'] + 1)

    input_data['Total_Nutrients'] = (
        input_data['Nitrogen_Level'] +
        input_data['Phosphorus_Level'] +
        input_data['Potassium_Level']
    )

    input_data['Soil_Health'] = (
        input_data['Organic_Carbon'] * 2 +
        input_data['Nitrogen_Level'] * 0.3
    )

    input_data['Moisture_Temp'] = (
        input_data['Soil_Moisture'] *
        input_data['Temperature']
    )

    input_data['Rainfall_per_Humidity'] = (
        input_data['Rainfall'] /
        (input_data['Humidity'] + 1)
    )

    input_data['Same_Crop_Repeat'] = (
        input_data['Crop_Type'] ==
        input_data['Previous_Crop']
    ).astype(int)

    input_data['Yield_per_Fertilizer'] = (
        input_data['Yield_Last_Season'] /
        (input_data['Fertilizer_Used_Last_Season'] + 1)
    )

    # Ensure correct column order
    input_data = input_data.reindex(columns=preprocessor.feature_names_in_)

    # Apply preprocessing
    input_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(input_processed)

    # Decode label
    fertilizer_name = le.inverse_transform(prediction)

    # Show result
    st.success(f"🌱 Recommended Fertilizer: {fertilizer_name[0]}")