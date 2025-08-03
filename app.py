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
# Biểu đồ 1: Line Chart
st.subheader("1️⃣ Number of Games Released per Year")
fig1, ax1 = plt.subplots()
df['Year_of_Release'].value_counts().sort_index().plot(kind='line', marker='o', ax=ax1)
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Games")
ax1.grid(True)
st.pyplot(fig1)

# Biểu đồ 2: Bar Chart
st.subheader("2️⃣ Average Global Sales by Genre")
fig2, ax2 = plt.subplots()
df.groupby('Genre')['Global_Sales'].mean().sort_values().plot(kind='barh', color='skyblue', ax=ax2)
ax2.set_xlabel("Avg Sales (millions)")
st.pyplot(fig2)

# Biểu đồ 3: Scatter Plot
st.subheader("3️⃣ Critic Score vs Global Sales")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='Critic_Score', y='Global_Sales', ax=ax3)
ax3.grid(True)
st.pyplot(fig3)

# Biểu đồ 4: Heatmap
st.subheader("4️⃣ Correlation Heatmap")
fig4, ax4 = plt.subplots()
sns.heatmap(df[['Critic_Score', 'User_Score', 'Global_Sales']].corr(), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)

# Biểu đồ 5: Boxplot
st.subheader("5️⃣ Sales Distribution by Platform")
fig5, ax5 = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x='Platform', y='Global_Sales', ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
st.pyplot(fig5)


from sklearn.ensemble import RandomForestRegressor

# Huấn luyện mô hình trực tiếp trong app.py (vì đơn giản và nhanh)
features = ['Platform', 'Genre', 'Critic_Score', 'User_Score', 'Year_of_Release']
X = df[features]
y = df['Global_Sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)  # huấn luyện toàn bộ (vì app demo)
