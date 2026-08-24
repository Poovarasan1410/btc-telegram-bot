import os
import joblib
import requests
import pandas as pd

from binance.client import Client

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# Load model
model = joblib.load("btc_xgb_model_v3.pkl")
features = joblib.load("btc_features.pkl")

# Telegram secrets
TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# Binance
client = Client()

# Download latest candles
klines = client.get_klines(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_15MINUTE,
    limit=100
)

# Build dataframe
df = pd.DataFrame(klines)

df = df.iloc[:, :6]

df.columns = [
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = df[col].astype(float)

# Indicators
df["MA20"] = df["Close"].rolling(20).mean()

df["RSI"] = RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

macd = MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

bb = BollingerBands(close=df["Close"])

df["BB_HIGH"] = bb.bollinger_hband()
df["BB_LOW"] = bb.bollinger_lband()

df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()

df["VOL_CHANGE"] = df["Volume"].pct_change()
df["PRICE_CHANGE"] = df["Close"].pct_change()

atr = AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

df["ATR"] = atr.average_true_range()

adx = ADXIndicator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

df["ADX"] = adx.adx()

stoch = StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

df["STOCH"] = stoch.stoch()

df = df.dropna()

latest_live = df[features].tail(1)

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

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

requests.post(
    url,
    data={
        "chat_id": CHAT_ID,
        "text": message
    }
)
