import os
import time
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
import pyupbit
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@dataclass
class Config:
    market: str
    candle: str

def load_config() -> Config:
    # Load .env first, then optional env.local (non-dotfile fallback)
    load_dotenv()
    load_dotenv("env.local", override=False)
    # Also load from this module's directory so running from project root still picks up bot/.env
    base_dir = os.path.dirname(__file__)
    try:
        load_dotenv(os.path.join(base_dir, ".env"), override=False)
        load_dotenv(os.path.join(base_dir, "env.local"), override=False)
    except Exception:
        pass
    return Config(
        market=os.getenv("MARKET", "KRW-BTC"),
        candle=os.getenv("CANDLE", "minute10"),
    )

def get_candles(market: str, candle: str, count: int = 200) -> pd.DataFrame:
    # PyUpbit uses parameter name 'ticker' not 'market'
    if candle.startswith("minute"):
        unit = int(candle.replace("minute", ""))
        data = pyupbit.get_ohlcv(ticker=market, interval=f"minute{unit}", count=count)
    else:
        data = pyupbit.get_ohlcv(ticker=market, interval=candle, count=count)
    if data is None or data.empty:
        raise RuntimeError("Failed to fetch OHLCV")
    return data

@app.route('/api/price', methods=['GET'])
def get_price():
    try:
        cfg = load_config()
        df = get_candles(cfg.market, cfg.candle, count=10)
        price = float(df["close"].iloc[-1])
        
        return jsonify({
            'price': price,
            'market': cfg.market,
            'candle': cfg.candle,
            'timestamp': int(time.time() * 1000)
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'price': 0,
            'market': 'KRW-BTC',
            'candle': 'minute10',
            'timestamp': int(time.time() * 1000)
        }), 500

@app.route('/api/price/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'running',
        'service': 'price-api',
        'timestamp': int(time.time() * 1000)
    })

if __name__ == '__main__':
    print("💰 Price API server starting...")
    app.run(host='127.0.0.1', port=5058, debug=False)
