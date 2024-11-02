import streamlit as st
import pandas as pd
import joblib

# Load the training dataset
df = pd.read_csv("train.csv")

# Set the title and icon of the Streamlit app
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# Title of the app
st.title("🚘 Car Price Predictor")

# Input fields for user directly in the main app
model = st.selectbox("🚗 Select Car Model", df['model'].unique())
series = st.selectbox("🛠️ Select Car Series", df[df['model'] == model]['series'].unique())
cc = st.number_input("⚙️ Engine CC", min_value=0, value=0)
width = st.number_input("📏 Width (mm)", min_value=0, value=0)
length = st.number_input("📐 Length (mm)", min_value=0, value=0)
frim = st.number_input("🔧 Front Rim (inches)", min_value=0.0, value=0.0)
rrim = st.number_input("🔩 Rear Rim (inches)", min_value=0.0, value=0.0)
weight = st.number_input("⚖️ Kerb Weight (kg)", min_value=0, value=0)
num_gears = st.number_input("🔄 Number of Gears", min_value=0, value=0)
pp = st.number_input("🏎️ Peak Power (hp)", min_value=0, value=0)
pt = st.number_input("🔋 Peak Torque (Nm)", min_value=0, value=0)
fthread = st.number_input("🛞 Front Thread", min_value=0.0, value=0.0)
rthread = st.number_input("🛞 Rear Thread", min_value=0.0, value=0.0)
parking = st.selectbox("🅿️ Parking Sensor", options=["Yes", "No"])
side = st.selectbox("➡️ Side Mirror Turning Indicators", options=["Yes", "No"])

# Convert categorical input to integer
parking = 1 if parking == "Yes" else 0
side = 1 if side == "Yes" else 0

# Button to make prediction
if st.button("🔍 Predict Price"):
    # Prepare the input data
    res = {}

    df2 = df.loc[(df.model == model) & (df.series == series)]

    # Fill with mode values
    for col in df2.columns:
        res[col] = df2[col].mode()[0]  # Get the first mode value

    # Update res dictionary with user input
    res['Engine CC'] = cc
    res['Width (mm)'] = width
    res['Length (mm)'] = length
    res['Front Rim (inches)'] = frim
    res['Rear Rim (inches)'] = rrim
    res['Kerb Weight (kg)'] = weight
    res['Number of Gears'] = num_gears
    res['Peak Power (hp)'] = pp
    res['Peak Torque (Nm)'] = pt
    res['Front Thread'] = fthread
    res['Rear Thread'] = rthread
    res['Parking sensor'] = parking
    res['Side mirror turning indicators'] = side

    df2 = pd.DataFrame(res, index=[0])  # Create DataFrame for prediction

    # Load the model pipeline
    pipeline = joblib.load('pipeline.pkl')
    price = round(pipeline.predict(df2)[0], 2)

    # Show the prediction result
    st.success(f"The predicted price is: RM {price}")

