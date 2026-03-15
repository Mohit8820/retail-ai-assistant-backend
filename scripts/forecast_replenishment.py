import pandas as pd
from prophet import Prophet
import os

# 1️⃣ Load prepared weekly dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir,  "..", "data", "product_weekly_sales.csv")

# print(csv_path)
df = pd.read_csv(csv_path, parse_dates=['Date'])
# 2️⃣ Prepare output DataFrame
forecast_results = []
# 3️⃣ Loop through each product + country
for (stockcode, country), group in df.groupby(['StockCode', 'Country']):
    
    if len(group) < 5:
      print(f"Skipping {stockcode} ({country}) - insufficient data")
      continue

    product_name = group['Description'].iloc[0]
  
    # Prepare data for Prophet
    prophet_df = group[['Date', 'QuantitySold']].rename(columns={'Date': 'ds', 'QuantitySold': 'y'})
  
    # Initialize Prophet model
    model = Prophet(weekly_seasonality=True)
  
    # Add regressors if needed
    if 'IsHoliday' in group.columns:
        prophet_df['IsHoliday'] = group['IsHoliday'].astype(int)
        model.add_regressor('IsHoliday')
    if 'UnitPrice' in group.columns:
        prophet_df['UnitPrice'] = group['UnitPrice']
        model.add_regressor('UnitPrice')
    if 'Temperature' in group.columns:
        prophet_df['Temperature'] = group['Temperature']
        model.add_regressor('Temperature')
    if 'Fuel_Price' in group.columns:
        prophet_df['Fuel_Price'] = group['Fuel_Price']
        model.add_regressor('Fuel_Price')
  
    # Fit model
    model.fit(prophet_df)
  
    # Predict next 4 weeks
    future = model.make_future_dataframe(periods=4, freq='W')
  
    # Include regressors for future (simple: use last known values)
    for reg in ['IsHoliday', 'UnitPrice', 'Temperature', 'Fuel_Price']:
        if reg in group.columns:
            future[reg] = group[reg].iloc[-1]
  
    forecast = model.predict(future)
  
    # Only take future 4 weeks
    future_forecast = forecast.tail(4)
  
    # Compute ReorderQuantity using last available stock
    last_stock = group['AvailableStock'].iloc[-1]
  
    for _, row in future_forecast.iterrows():
        predicted = max(row['yhat'], 0)
        reorder_qty = max(predicted - last_stock, 0)
        forecast_results.append({
            'StockCode': stockcode,
            'Description': product_name,
            'Country': country,
            'WeekStart': row['ds'].date(),
            'PredictedQuantity': round(predicted),
            'AvailableStock': last_stock,
            'ReorderQuantity': round(reorder_qty)
        })
# 4️⃣ Save results
forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_csv(os.path.join(script_dir, "..", "data", "product_forecast_reorder.csv"), index=False)
print("Forecast and reorder quantities saved to 'product_forecast_reorder.csv'")
print(forecast_df.head())