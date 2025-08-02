import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Video Game Sales Dashboard")

# Load dữ liệu
df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")

# Dọn dẹp dữ liệu
df = df.dropna(subset=['Year_of_Release', 'Genre', 'Global_Sales'])
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

# 1. Tổng số game phát hành theo năm
games_per_year = df.groupby('Year_of_Release')['Name'].count().reset_index()
fig1 = px.line(games_per_year, x='Year_of_Release', y='Name', title='Số lượng game phát hành theo năm')
st.plotly_chart(fig1)

# 2. Doanh thu toàn cầu theo thể loại
sales_by_genre = df.groupby('Genre')['Global_Sales'].sum().reset_index()
fig2 = px.bar(sales_by_genre, x='Genre', y='Global_Sales', title='Doanh thu toàn cầu theo thể loại')
st.plotly_chart(fig2)

# 3. Top 10 game bán chạy nhất
top_10 = df.sort_values('Global_Sales', ascending=False).head(10)
fig3 = px.bar(top_10, x='Name', y='Global_Sales', title='Top 10 game bán chạy nhất')
st.plotly_chart(fig3)
