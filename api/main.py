import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

from services.rag_service import RetailRAGService

app = FastAPI(title="Retail AI Demand Forecast API")

# Load forecast data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "product_forecast_reorder.csv")

df = pd.read_csv(csv_path)


@app.get("/")
def home():
    return {"message": "Retail AI Assistant running 🚀"}


@app.get("/replenishment-alerts")
def get_replenishment_alerts():
    
    alerts = df[df["ReorderQuantity"] > 0]

    return alerts.head(20).to_dict(orient="records")


@app.get("/forecast/{stock_code}")
def get_product_forecast(stock_code: str):

    product_data = df[df["StockCode"] == stock_code]

    if product_data.empty:
        return {"message": "Product not found"}

    return product_data.to_dict(orient="records")

ai = RetailRAGService()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):

    answer = ai.ask(query.question)

    return {
        "question": query.question,
        "answer": answer
    }