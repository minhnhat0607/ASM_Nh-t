import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

# --- 1. Environment Setup and Required Libraries ---
st.title("P7: Product Demand Forecasting for ABC Manufacturing")

# Configure plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

st.sidebar.header("User Settings")
forecast_steps = st.sidebar.slider("Number of forecast days:", 30, 180, 30)

# --- 2. Mock Data Generation ---
@st.cache_data
def generate_mock_sales_data(start_date, end_date, num_products=3):
    # (The same data generation function as before)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    for prod_id in range(1, num_products + 1):
        base_sales = np.random.randint(50, 200)
        for i, date in enumerate(dates):
            trend = i / 1000.0
            month_effect = 10 * np.sin(2 * np.pi * date.month / 12)
            year_effect = 20 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, 15)
            promotion_effect = 0
            if np.random.rand() < 0.05:
                promotion_effect = np.random.randint(30, 80)
            sales = max(0, base_sales + trend + month_effect + year_effect + noise + promotion_effect)
            data.append([date, f'PROD_{prod_id}', int(sales)])
    df = pd.DataFrame(data, columns=['Date', 'ProductID', 'SalesQuantity'])
    return df

df_sales = generate_mock_sales_data('2020-01-01', '2024-12-31', num_products=3)

# --- 3. Data Preprocessing ---
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_sales = df_sales.set_index('Date').sort_index()
df_daily_sales = df_sales.groupby(['Date', 'ProductID'])['SalesQuantity'].sum().unstack(fill_value=0)
df_daily_sales = df_daily_sales.stack().reset_index()
df_daily_sales.columns = ['Date', 'ProductID', 'SalesQuantity']
df_daily_sales = df_daily_sales.set_index('Date').sort_index()
df_daily_sales['Year'] = df_daily_sales.index.year
df_daily_sales['Month'] = df_daily_sales.index.month

# --- 4. EDA & Visualization ---
st.header("1. Exploratory Data Analysis (EDA)")

st.subheader("Overall Monthly Sales Trend")
fig1, ax1 = plt.subplots(figsize=(12, 6))
df_daily_sales.resample('M')['SalesQuantity'].sum().plot(ax=ax1)
st.pyplot(fig1)

st.subheader("Monthly Sales Trend for Each Product")
fig2, ax2 = plt.subplots(figsize=(12, 6))
for product_id in df_daily_sales['ProductID'].unique():
    df_daily_sales[df_daily_sales['ProductID'] == product_id]['SalesQuantity'].resample('M').sum().plot(label=product_id, ax=ax2)
st.pyplot(fig2)

# --- 5. Build and Train Forecasting Model (ARIMA) ---
st.header("2. Forecasting with SARIMA Model")
target_product = st.selectbox("Select a product to forecast:", df_daily_sales['ProductID'].unique())

ts_data = df_daily_sales[df_daily_sales['ProductID'] == target_product]['SalesQuantity'].resample('D').sum()
train_data = ts_data['2020-01-01':'2023-12-31']
test_data = ts_data['2024-01-01':'2024-12-31']

order = (1, 1, 1)
seasonal_order = (1, 1, 0, 7)

@st.cache_resource
def train_sarima_model(data, order, seasonal_order):
    model = ARIMA(data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    return model_fit

st.write(f"Training SARIMA({order})({seasonal_order}, S=7) model for {target_product}...")
model_fit = train_sarima_model(train_data, order, seasonal_order)
st.success("Model training complete!")

# --- 6. Model Evaluation ---
st.header("3. Model Evaluation and Future Forecast")

forecast_result = model_fit.get_forecast(steps=len(test_data))
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()
forecast_mean.index = test_data.index
forecast_ci.index = test_data.index

mae = mean_absolute_error(test_data, forecast_mean)
rmse = np.sqrt(mean_squared_error(test_data, forecast_mean))
st.write(f"**Model Evaluation for {target_product}:**")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# --- 7. Future Forecasting and Decision Support ---
future_steps = st.slider("Select number of days to forecast into the future:", 1, 365, 30)
last_date = ts_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
future_forecast_result = model_fit.get_forecast(steps=future_steps)
future_forecast_mean = future_forecast_result.predicted_mean
future_forecast_ci = future_forecast_result.conf_int()
future_forecast_mean.index = future_dates
future_forecast_ci.index = future_dates

fig3, ax3 = plt.subplots(figsize=(15, 7))
ax3.plot(train_data.index[-90:], train_data[-90:], label='Recent Training Data')
ax3.plot(test_data.index, test_data, label='Actual Data (2024)', color='orange')
ax3.plot(forecast_mean.index, forecast_mean, label='Forecast (2024)', color='red', linestyle='--')
ax3.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
ax3.plot(future_forecast_mean.index, future_forecast_mean, label=f'Future Forecast ({future_steps} days)', color='green')
ax3.fill_between(future_forecast_ci.index, future_forecast_ci.iloc[:, 0], future_forecast_ci.iloc[:, 1], color='lightgreen', alpha=0.3)
st.pyplot(fig3)

st.header("4. Decision Support")
st.write("Based on the forecast, ABC Manufacturing can make the following decisions:")
st.write(f"**- Production Planning:** The production department can schedule output to match the forecasted demand of `{future_forecast_mean.mean():.0f}` units/day for the next `{future_steps}` days.")
st.write("**- Inventory Management:** The warehouse can adjust safety stock levels to meet anticipated demand and optimize storage costs. The confidence interval provides a risk range for this planning.")
