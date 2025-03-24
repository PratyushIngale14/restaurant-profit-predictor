# restaurant_profit_prediction_with_forecast_and_dashboard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from flask import Flask, render_template, request, jsonify
import joblib
import os

# Load dataset
df = pd.read_csv("restaurant_financial_data.csv")

# Feature selection for profit prediction
features = ['Total_Sales', 'Marketing_Cost', 'Rent', 'Staff_Cost', 'Utilities']
X = df[features]
y = df['Profit']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model for Flask app
joblib.dump(model, 'profit_model.pkl')

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot Actual vs Predicted Profit
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_profit.png")
plt.close()

# Save the model coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coefficients)

# Prophet forecasting setup
forecast_df = df.groupby(['Year', 'Month'])[['Total_Sales']].sum().reset_index()
forecast_df['ds'] = pd.to_datetime(forecast_df[['Year', 'Month']].assign(DAY=1))
forecast_df = forecast_df.rename(columns={'Total_Sales': 'y'})[['ds', 'y']]

prophet_model = Prophet()
prophet_model.fit(forecast_df)

future = prophet_model.make_future_dataframe(periods=36, freq='MS')
forecast = prophet_model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("sales_forecast.csv", index=False)

# Flask app setup
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
