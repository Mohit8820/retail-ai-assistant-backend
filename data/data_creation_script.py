import pandas as pd
import numpy as np

# Load dataset
initial_df = pd.read_csv("retail.csv", parse_dates=['InvoiceDate'])
print(initial_df.head())

sample_products = initial_df['StockCode'].unique()[:10]

df=initial_df[initial_df['StockCode'].isin(sample_products)]

print(df)

# 1️⃣ Aggregate weekly per product
df['WeekStart'] = (df['InvoiceDate'] - pd.to_timedelta(df['InvoiceDate'].dt.dayofweek, unit='D')).dt.normalize()
print(df['WeekStart'])

df['Week'] = df['InvoiceDate'].dt.isocalendar().week
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year
# 2️⃣ Generate IsHoliday column (basic simulation)
# Assume major holidays: month of dec for christmas
df['IsHoliday'] = False
df.loc[(df['InvoiceDate'].dt.month == 12), 'IsHoliday'] = True

print(df)
# # 3️⃣ Generate Temperature and Fuel_Price columns (simulate)
np.random.seed(42)
df['Temperature'] = np.random.uniform(5, 30, size=len(df))  # °C
df['Fuel_Price'] = np.random.uniform(2.5, 4.0, size=len(df))  # $/unit
# 4️⃣ Aggregate weekly per StockCode + Description + Country
weekly_df = df.groupby(['StockCode', 'Description', 'WeekStart', 'Country']).agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'Temperature': 'mean',
    'Fuel_Price': 'mean',
    'IsHoliday': 'max'
}).reset_index()
# 5️⃣ Rename for clarity
weekly_df.rename(columns={'Quantity': 'QuantitySold', 'WeekStart': 'Date'}, inplace=True)
# 6️⃣ Generate AvailableStock (simulate)
weekly_df['AvailableStock'] = np.random.randint(50, 200, size=len(weekly_df))

# Save prepared dataset
weekly_df.to_csv("product_weekly_sales.csv", index=False)
print("Dataset ready for forecasting and AI assistant!")
print(weekly_df.head())