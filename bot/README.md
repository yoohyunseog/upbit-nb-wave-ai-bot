# 8BIT Upbit Bot (PyUpbit)

Quick-start PyUpbit bot with EMA crossover example.

## Setup (Windows PowerShell)

```powershell
cd bot
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # then edit keys
python main.py  # runs in paper mode by default
```

## .env

```
UPBIT_ACCESS_KEY=
UPBIT_SECRET_KEY=
# PAPER=true -> no real orders
PAPER=true
MARKET=KRW-BTC
CANDLE=minute10
EMA_FAST=10
EMA_SLOW=30
INTERVAL_SEC=30
ORDER_KRW=5000
```

- PAPER=true: simulate fills without sending real orders
- Set keys and set PAPER to false to enable live trading

## Notes
- Respect Upbit rate limits (use 0.2~0.5s sleep between calls)
- Backtest externally (this example is live/paper executor only)

