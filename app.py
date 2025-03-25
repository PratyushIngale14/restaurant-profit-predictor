# restaurant_profit_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from flask import Flask, render_template, request, jsonify
import joblib
import os

df = pd.read_csv("restaurant_raw_data.csv")

# ------------------------------
# Step 1: EDA & Data Cleaning
# ------------------------------

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle invalid and missing values
df = df[df['Total_Sales'] > 0]
df = df[df['Utilities'] > 0]
df = df[df['Rent'] < 40000]  # Remove extreme rent outliers
df = df[df['Staff_Cost'] < 150000]  # Remove staff cost outliers

# Fill missing marketing costs with median
df['Marketing_Cost'].fillna(df['Marketing_Cost'].median(), inplace=True)

# Recalculate total expenses and profit
df['Total_Expenses'] = df['Marketing_Cost'] + df['Rent'] + df['Staff_Cost'] + df['Utilities']
df['Profit'] = df['Total_Sales'] - df['Total_Expenses']

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# ------------------------------
# Step 2: Profit Prediction Model
# ------------------------------
features = ['Total_Sales', 'Marketing_Cost', 'Rent', 'Staff_Cost', 'Utilities']
X = df[features]
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'profit_model.pkl')

# Evaluate model
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# ------------------------------
# Step 3: Sales Forecasting with Prophet
# ------------------------------
forecast_df = df.groupby(['Year', 'Month'])[['Total_Sales']].sum().reset_index()
forecast_df['ds'] = pd.to_datetime(forecast_df[['Year', 'Month']].assign(DAY=1))
forecast_df = forecast_df.rename(columns={'Total_Sales': 'y'})[['ds', 'y']]
prophet_model = Prophet()
prophet_model.fit(forecast_df)
future = prophet_model.make_future_dataframe(periods=36, freq='MS')
forecast = prophet_model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("sales_forecast.csv", index=False)

# ------------------------------
# Step 4: Flask Web App
# ------------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['Total_Sales'],
        data['Marketing_Cost'],
        data['Rent'],
        data['Staff_Cost'],
        data['Utilities']
    ]
    model = joblib.load('profit_model.pkl')
    prediction = model.predict([features])[0]
    return jsonify({'predicted_profit': round(prediction, 2)})

@app.route('/forecast')
def forecast_view():
    forecast_data = pd.read_csv("sales_forecast.csv").tail(36)
    return forecast_data.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/forecast-page')
def forecast_page():
    return render_template('forecast.html')

@app.route('/forecast-data')
def forecast_data():
    forecast_df = pd.read_csv("sales_forecast.csv").tail(36)
    return forecast_df.to_json(orient='records')
