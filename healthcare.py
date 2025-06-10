import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import os

# =============== CONFIGURE GEMINI API ===============
# Replace with your actual Gemini API Key
genai.configure(api_key="AIzaSyDSOBFmasJ4a_wX4Pu91bO5W4WhO1tFaHI")

# =============== GEMINI FUNCTION ====================
def get_health_tip(data):
    prompt = (
        f"Based on the following health data:\n"
        f"Steps walked: {data['steps']} steps\n"
        f"Calories consumed: {data['calories']} kcal\n"
        f"Water intake: {data['water']} liters\n"
        f"Sleep: {data['sleep']} hours\n"
        f"Weight: {data['weight']} kg, Height: {data['height']} cm\n"
        f"Give a short health tip to improve wellness. Keep it positive and under 100 words."
    )
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error fetching health tip: {e}"

# =============== STREAMLIT LAYOUT ===================
st.set_page_config(page_title="Health Dashboard", page_icon="🏥")
st.title("🏥 Personal Health Dashboard")

st.sidebar.header("📋 Enter your daily health stats")

# ======== Sidebar Form Inputs ========
steps = st.sidebar.number_input("Steps Walked", min_value=0, value=5000)
calories = st.sidebar.number_input("Calories Consumed", min_value=0, value=1800)
water = st.sidebar.number_input("Water Intake (liters)", min_value=0.0, value=2.0)
sleep = st.sidebar.number_input("Sleep (hours)", min_value=0.0, value=7.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, value=70.0)
height = st.sidebar.number_input("Height (cm)", min_value=0.0, value=170.0)

if st.sidebar.button("✅ Submit"):
    st.success("Data Submitted!")

    # ======== BMI Calculation ========
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    st.subheader(f"🧮 Your BMI: {bmi}")

    if bmi < 18.5:
        st.write("🔍 You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.write("✅ You have a normal weight.")
    elif 25 <= bmi < 29.9:
        st.write("⚠️ You are overweight.")
    else:
        st.write("🚨 You are in the obese range.")

    # ======== Plotting Health Stats ========
    st.subheader("📊 Health Stats Chart")
    labels = ['Steps', 'Calories', 'Water (L)', 'Sleep (hrs)']
    values = [steps, calories, water, sleep]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'cyan', 'purple'])
    ax.set_ylabel('Values')
    ax.set_title('Your Daily Health Stats')
    st.pyplot(fig)

    # ======== Generate AI Health Tip ========
    st.subheader("💡 AI Health Tip")
    user_data = {
        "steps": steps,
        "calories": calories,
        "water": water,
        "sleep": sleep,
        "weight": weight,
        "height": height
    }
    tip = get_health_tip(user_data)
    st.info(tip)
