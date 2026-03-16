import pandas as pd
import os
import json

# Load dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir,  "..", "data", "product_weekly_sales.csv")
df = pd.read_csv(csv_path, parse_dates=['Date'])

insights = []

# GROUP BY PRODUCT + COUNTRY
grouped = df.groupby(["StockCode", "Description", "Country"])

for (stock, desc, country), group in grouped:

    avg_weekly_sales = group["QuantitySold"].mean()
    latest_stock = group.sort_values("Date").iloc[-1]["AvailableStock"]
    
    stockout_risk = "Low"
    
    if avg_weekly_sales > 0:
        weeks_remaining = latest_stock / avg_weekly_sales
        
        if weeks_remaining < 1:
            stockout_risk = "Very High"
        elif weeks_remaining < 2:
            stockout_risk = "High"
        elif weeks_remaining < 4:
            stockout_risk = "Medium"

    # Holiday demand
    holiday_sales = group[group["IsHoliday"] == True]["QuantitySold"].mean()
    
    insight = f"""
Product: {desc}
Country: {country}

Average weekly demand: {avg_weekly_sales:.2f} units
Current available stock: {latest_stock}

Stockout risk: {stockout_risk}

Average holiday demand: {holiday_sales if not pd.isna(holiday_sales) else 0:.2f}

Recommendation:
If stockout risk is high, replenish inventory soon.
"""

    insights.append(insight)


print(f"Generated {len(insights)} insights")

# Save insights
with open(os.path.join(script_dir,  "..", "data", "insights.json"), "w") as f:
    json.dump(insights, f, indent=2)