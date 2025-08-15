## upbit-nb-wave-ai-bot

An Upbit auto-trading bot with a real-time Flask UI, NB-wave-based signals, optional ML modeling, backtesting, and live order markers.

### Key features
- Real-time chart UI with NB wave, EMA, SMA, Ichimoku overlays and order markers
- Auto trading loop (paper/live) with partial close protection and sizing from UI profit/loss bars
- ML training and prediction (per timeframe), time-series CV, light hyper-parameter search, probability calibration
- Backtesting on the current chart view; rolling Win% panel and top PnL slider
- Assets panel (KRW, top holdings), auto refresh

### Project structure
```
bot/
  server.py        # Flask server and trading/ML APIs
  static/ui.html   # Web UI
  static/ui.js     # UI logic (charts, backtest, ML controls, assets)
  trade.py         # Order sizing and live/paper trade execution
  strategy.py      # Signal helpers
  main.py          # Standalone simple loop (optional)
  requirements.txt # Python dependencies
  models/          # Saved ML models per timeframe (created at runtime)
  data/            # NB params, etc.
```

### Requirements
- Python 3.10+
- pip
- Optional for live trading: Upbit API keys

### Installation
```powershell
cd bot
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration (.env)
Create `bot/.env` (never commit this) with values for live trading and UI options.
```
# Upbit standard keys (for pyupbit)
UPBIT_ACCESS_KEY=...
UPBIT_SECRET_KEY=...

# UI
UI_PORT=5057
UI_HTTPS=false

# NB thresholds (optional override)
NB_HIGH=0.55
NB_LOW=0.45
```

Runtime config can also be set from the UI toolbar (paper mode, order KRW, EMA, timeframe). UI pushes changes to the server without restart.

### Run
```powershell
python server.py
# Open the UI in your browser:
# http://127.0.0.1:5057/ui
```

### UI guide (high level)
- Start Bot / Stop Bot: control auto trading loop
- Auto BT: periodic backtest on the current chart view
- Train (Auto Period) / Auto-Tune NB: quick NB optimization
- ML Train / ML Predict / ML Random: train, predict, and random-train cycles
- ML Metrics: shows last model’s in-sample accuracy/confusion, CV F1-macro and CV PnL
- Assets panel: total KRW estimate, buyable KRW, sellable tickers, top holdings bars (live mode)
- Win% panel: rolling items from backtests and live trades; top slider reflects aggregate win/loss

### Auto-trade sizing logic (summary)
- Minimum notional: Upbit requires ≥ 5,000 KRW per order
- BUY KRW: max(fixed 5,000 KRW, available_KRW × profit_ratio%) rounded down to 1,000 steps
- SELL size:
  - If loss_ratio% > 0: sell holdings × loss_ratio%
  - Else if pnl_ratio% > 0: sell holdings × pnl_ratio%
  - Else: sell up to the bot’s estimated open position (prevents full liquidation)
  - Orders below ~5,000 KRW notional are skipped

The UI pushes `pnl_profit_ratio` and `pnl_loss_ratio` based on the top PnL bar aggregate.

### ML training details
- Features X (from recent OHLCV):
  - r (0..1 position in rolling window), w (range/avg), ema_f, ema_s, ema_diff
  - r_ema3, r_ema5, dr, ret1, ret3, ret5
- Labels y (selectable internally):
  - nb_zone (default), fwd_return(tau/horizon), nb_best_trade
- Modeling pipeline:
  - GradientBoosting candidates + TimeSeriesSplit (3) → select by F1-macro, tie-breaker CV PnL
  - Calibrated probabilities (sigmoid)
  - Saved per timeframe: `bot/models/nb_ml_{interval}.pkl`
- APIs:
  - POST `/api/ml/train` (train with params)
  - GET  `/api/ml/predict` (predict action on the latest window)
  - GET  `/api/ml/metrics` (in-sample report, confusion, CV F1/PnL, params)

### REST API (selected)
- UI/state
  - GET  `/api/state` – price/signal snapshot
  - GET  `/api/stream` – server-sent events (price, signal, last order)
  - GET  `/api/ohlcv?count=...&interval=...`
  - POST `/api/bot/config` – runtime overrides (paper, order_krw, pnl ratios, EMA, candle, keys)
  - POST `/api/bot/start`, `/api/bot/stop`, GET `/api/bot/status`
- Orders & markers
  - GET  `/api/orders` – recent orders
  - POST `/api/order` – external order notify (optional)
  - POST `/api/orders/clear`
- NB
  - POST `/api/nb/optimize`, `/api/nb/train`, GET/POST `/api/nb/params`
- ML
  - POST `/api/ml/train`, GET `/api/ml/predict`, GET `/api/ml/metrics`

### Safety & disclaimer
- For live trading, test thoroughly in paper mode first
- Always keep `.env` and private keys out of version control
- This software is provided “as is,” without warranty of any kind. Trading involves risk.

### License
MIT — see `../LICENSE`.

