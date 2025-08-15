## upbit-nb-wave-ai-bot

An Upbit auto-trading bot with a real-time Flask UI, NB-wave signals, optional ML modeling, backtesting, and live order markers.

### Version
- 0.9.4 (2025-08-15)

### What's new in 0.9.4
- N/B COIN per-candle tracking
  - One “coin” per candle bucket (timeframe-aware) recorded in-memory
  - Marked automatically on auto-trade, manual trade, and ML signal log
  - New API `GET /api/nb/coin?interval=...&market=...&n=50` returns current and recent coins
  - `GET /api/bot/status` includes current coin under `coin`
- UI N/B COIN panel
  - Separate card showing current bar’s coin (BUY/SELL/NONE) and a recent strip (older→newer)
  - Trade Readiness also shows inline coin status
  - Auto-refresh every few seconds
- ML-only and ML segment-only toggles (runtime config)
  - Server accepts `ml_only`, `ml_seg_only` via `/api/bot/config`
- Manual trading endpoints hardened
  - Zone 100% gating honored; proper size/price echo; order interval tagging and `live_ok` propagation

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

#### Optional runtime overrides (via UI → Config API)
- `ml_only=true|false`: Follow ML NB direction strictly within the loop
- `ml_seg_only=true|false`: Trade on NB line crosses (extreme-only gate)

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

### License
MIT — see `LICENSE` file.


