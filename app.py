import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")
st.title("🎮 Video Game Global Sales App")
st.markdown("Ứng dụng dự đoán doanh số toàn cầu của trò chơi điện tử và hiển thị biểu đồ.")

# Load và xử lý dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

    # Chuyển đổi kiểu dữ liệu số
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')
    df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')

    # Mã hóa cột phân loại
    for col in ['Platform', 'Genre']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Xoá dữ liệu thiếu
    df = df.dropna(subset=['Global_Sales', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Year_of_Release'])
    return df

df = load_data()

# Huấn luyện mô hình
features = ['Platform', 'Genre', 'Critic_Score', 'User_Score', 'Year_of_Release']
X = df[features]
y = df['Global_Sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Giao diện dự đoán
st.subheader("📈 Dự đoán doanh số trò chơi")

platform = st.number_input("Platform (mã hóa)", min_value=0, max_value=100, value=10)
genre = st.number_input("Genre (mã hóa)", min_value=0, max_value=20, value=5)
critic_score = st.slider("Critic Score", 0, 100, 75)
user_score = st.slider("User Score", 0.0, 10.0, 8.0, step=0.1)
year = st.number_input("Year of Release", min_value=1980, max_value=2025, value=2015)

if st.button("🔮 Dự đoán"):
    input_data = [[platform, genre, critic_score, user_score, year]]
    prediction = model.predict(input_data)
    st.success(f"💰 Doanh số dự đoán: {prediction[0]:.2f} triệu bản")

# Hiển thị biểu đồ
st.subheader("📊 Biểu đồ trực quan dữ liệu")

# Biểu đồ 1: Số lượng game theo năm
st.markdown("**1. Số lượng game phát hành theo năm**")
fig1, ax1 = plt.subplots()
df['Year_of_Release'].value_counts().sort_index().plot(kind='line', marker='o', ax=ax1)
ax1.set_xlabel("Năm")
ax1.set_ylabel("Số lượng")
ax1.grid(True)
st.pyplot(fig1)

# Biểu đồ 2: Doanh số trung bình theo thể loại
st.markdown("**2. Doanh số trung bình theo thể loại**")
fig2, ax2 = plt.subplots()
df.groupby('Genre')['Global_Sales'].mean().sort_values().plot(kind='barh', color='skyblue', ax=ax2)
ax2.set_xlabel("Doanh số trung bình (triệu bản)")
st.pyplot(fig2)

# Biểu đồ 3: Critic Score vs Global Sales
st.markdown("**3. Điểm đánh giá chuyên môn vs Doanh số**")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='Critic_Score', y='Global_Sales', ax=ax3)
ax3.set_title("Critic Score vs Global Sales")
ax3.grid(True)
st.pyplot(fig3)

# Biểu đồ 4: Heatmap tương quan
st.markdown("**4. Ma trận tương quan**")
fig4, ax4 = plt.subplots()
sns.heatmap(df[['Critic_Score', 'User_Score', 'Global_Sales']].corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Biểu đồ 5: Boxplot theo nền tảng
st.markdown("**5. Phân phối doanh số theo nền tảng**")
fig5, ax5 = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x='Platform', y='Global_Sales', ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
st.pyplot(fig5)
