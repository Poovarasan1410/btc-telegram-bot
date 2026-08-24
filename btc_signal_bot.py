import os
import joblib
import requests
from binance.client import Client

# Load model and features
model = joblib.load("btc_xgb_model_v3.pkl")
features = joblib.load("btc_features.pkl")

# Telegram settings
TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": msg
        }
    )

# Binance data
client = Client()

klines = client.get_klines(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_15MINUTE,
    limit=100
)

print(f"Downloaded {len(klines)} candles")

# NOTE:
# Your RSI, MACD, EMA20, EMA50,
# BB_HIGH, BB_LOW, ATR, ADX, STOCH
# calculation code must be copied here
# from your notebook.

# After indicators are calculated:
latest_live = df_live[features].tail(1)

prediction = model.predict(latest_live)[0]
probability = model.predict_proba(latest_live)[0]

buy_prob = probability[1]
sell_prob = probability[0]

if buy_prob > 0.70:
    signal = "STRONG BUY"
elif buy_prob > 0.60:
    signal = "BUY"
elif sell_prob > 0.70:
    signal = "SELL"
else:
    signal = "NO TRADE"

message = f"""
BTC SIGNAL

Signal : {signal}

BUY Probability : {buy_prob:.2%}
SELL Probability : {sell_prob:.2%}
"""

print(message)

send_telegram(message)
