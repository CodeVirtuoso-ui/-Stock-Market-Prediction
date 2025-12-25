# -Stock-Market-PredictionSTOCK-MARKET-PREDICTIOR

A minimal Flask web app that fetches historical stock data with yfinance, engineers simple moving-average features, trains a Linear Regression model, and predicts the next trading day close. It also renders a compact chart and basic evaluation metrics.

Features

Fetch daily OHLCV data via yfinance
Feature engineering: Close, MA5, MA20; target is next-day Close
Train/test split with Linear Regression and basic metrics (MAE, MSE, R²)
Plot recent Close prices and mark predicted next close
Simple Flask UI with two routes: GET / and POST /predict
Tech Stack

Python, Flask
yfinance, pandas, numpy, scikit-learn, matplotlib (Agg backend)
Project Structure

E:/DMDW CASE STUDY/
├─ app.py
├─ requirements.txt
├─ static/
│  └─ style.css
├─ templates/
│  ├─ index.html
│  └─ result.html
└─ README.md
Setup

Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies
pip install -r requirements.txt
Run

python app.py
The app starts in debug mode on http://127.0.0.1:5000/.

Usage

Open the home page and enter a ticker (e.g., RELIANCE.NS, TCS.NS, AAPL).
Optionally adjust the period (e.g., 1y, 2y).
Submit to view the predicted next close, metrics, and chart.
Notes

Matplotlib uses the non-GUI Agg backend to avoid GUI/thread issues on servers.
Predictions are for demonstration only and not financial advice.
Requirements See requirements.txt for pinned package versions.
