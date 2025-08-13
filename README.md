## upbit-nb-wave-ai-bot

An Upbit auto-trading bot with a real-time Flask UI, NB-wave signals, optional ML modeling, backtesting, and live order markers.

### Highlights
- Real-time chart UI (NB wave, EMA/SMA/Ichimoku, order markers)
- Auto trading loop (paper/live), partial close protection
- Sizing from UI top PnL bar (profit/loss ratios)
- ML: time-series CV, light hyper-parameter search, calibrated probs, per-timeframe models
- One-click backtest on current chart; rolling Win% and top PnL slider
- Assets panel (KRW and holdings), auto refresh

### Quick start
```bash
# Python 3.10+
cd bot
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python server.py
# Open UI: http://127.0.0.1:5057/ui
```

### Configure (do not commit secrets)
Create `bot/.env` with your keys and options:
```
UPBIT_ACCESS_KEY=...
UPBIT_SECRET_KEY=...
UI_PORT=5057
UI_HTTPS=false
NB_HIGH=0.55
NB_LOW=0.45
```

You can also override paper/live, order size, EMAs, timeframe from the UI; changes are pushed to the server at runtime.

### Repo structure
```
bot/
  server.py        # Flask server and trading/ML APIs
  static/ui.html   # Web UI
  static/ui.js     # UI logic (charts, backtest, ML controls, assets)
  trade.py         # Order sizing and execution (paper/live)
  strategy.py      # Signal helpers
  main.py          # Simple loop (optional)
  requirements.txt # Python deps
  models/          # Saved ML models per timeframe (runtime)
  data/            # NB params, etc.
```

More details are documented in `bot/README.md`.

### Safety
- Test in paper mode first. Trading involves risk.
- Never commit `.env` or private keys.


