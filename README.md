## upbit-nb-wave-ai-bot

An Upbit auto-trading bot with a real-time Flask UI, NB-wave signals, optional ML modeling, backtesting, and live order markers.

### Highlights
- Real-time chart UI (NB wave, EMA/SMA/Ichimoku, order markers)
- Auto trading loop (paper/live), partial close protection
- Sizing from UI top PnL bar (profit/loss ratios)
- ML: time-series CV, light hyper-parameter search, calibrated probs, per-timeframe models
- One-click backtest on current chart; rolling Win% and top PnL slider
- Assets panel (KRW and holdings), auto refresh

### What’s new (2025-08)
- Zone-aware learning and insight
  - New features: `zone_min_r`, `zone_max_r`, `zone_extreme_r`, `zone_extreme_age`, `dist_high`, `dist_low`, `extreme_gap`, `zone_conf`.
  - ML model persists `feature_names` to prevent feature-dimension mismatch across versions.
  - Model Insight shows raw/adjusted BLUE/ORANGE, zone extrema and age.
- UI/UX
  - Insight badge updated; text color improvements (`text-white`).
  - Real-time log autoscroll toggle honored; no forced scroll when OFF. Log capped to 50 lines.
- API
  - `/api/ml/predict` insight includes zone extrema fields; probabilities include raw and trend-adjusted values.

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

### Links
- GitHub repository: [yoohyunseog/upbit-nb-wave-ai-bot](https://github.com/yoohyunseog/upbit-nb-wave-ai-bot.git)


