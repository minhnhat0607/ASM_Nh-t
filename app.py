import streamlit as st
import pandas as pd
import pickle

# Load mô hình đã huấn luyện
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎮 Game Global Sales Prediction App")
st.markdown("Dự đoán doanh số bán toàn cầu của trò chơi dựa trên các đặc điểm đầu vào.")

# Nhập dữ liệu từ người dùng
platform = st.number_input("Platform (encoded)", min_value=0, max_value=100, value=10)
genre = st.number_input("Genre (encoded)", min_value=0, max_value=20, value=5)
critic_score = st.slider("Critic Score", 0, 100, 75)
user_score = st.slider("User Score", 0.0, 10.0, 8.0, step=0.1)
year = st.number_input("Year of Release", min_value=1980, max_value=2025, value=2015)

# Dự đoán
if st.button("Dự đoán doanh số"):
    input_data = [[platform, genre, critic_score, user_score, year]]
    prediction = model.predict(input_data)
    st.success(f"💰 Doanh số dự đoán: {prediction[0]:.2f} triệu bản")
