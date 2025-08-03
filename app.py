import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    df = df.dropna(subset=['Global_Sales', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Year_of_Release'])
    return df

df = load_data()

# Huấn luyện mô hình trực tiếp
features = ['Platform', 'Genre', 'Critic_Score', 'User_Score', 'Year_of_Release']
X = df[features]
y = df['Global_Sales']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Giao diện người dùng
st.title("🎮 Video Game Sales Predictor")
st.markdown("Dự đoán doanh số bán toàn cầu của một trò chơi.")

platform = st.number_input("Platform (mã hoá)", min_value=0, max_value=100, value=10)
genre = st.number_input("Genre (mã hoá)", min_value=0, max_value=20, value=5)
critic = st.slider("Critic Score", 0, 100, 75)
user = st.slider("User Score", 0.0, 10.0, 8.0, step=0.1)
year = st.number_input("Year of Release", min_value=1980, max_value=2025, value=2015)

if st.button("Dự đoán"):
    input_data = [[platform, genre, critic, user, year]]
    prediction = model.predict(input_data)
    st.success(f"💰 Doanh số dự đoán: {prediction[0]:.2f} triệu bản")

# Thêm một biểu đồ (tuỳ chọn)
st.subheader("📊 Doanh số trung bình theo thể loại")
fig, ax = plt.subplots()
df.groupby('Genre')['Global_Sales'].mean().sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)
