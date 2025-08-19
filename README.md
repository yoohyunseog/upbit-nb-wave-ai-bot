# 🏰 8BIT Village Trading System

An advanced Upbit auto-trading bot with a unique village-based trading system, real-time Flask UI, NB-wave signals, ML modeling, backtesting, and live order markers.

## 🎯 Version

* 0.10.0 (2025-01-18) - 8BIT Village Integration

## 🆕 What's New in 0.10.0

### 🏰 8BIT Village Trading System
* **Village-based Trading**: Unique village system with residents having specific roles and strategies
* **Mayor's Guidance System**: Real-time guidance with trust-based decision making
* **Bitcar Energy System**: Energy injection and management for trading
* **Trainer Warehouse**: Real-time trade recording and pattern analysis
* **ML Model Learning**: Automatic learning of Mayor's guidance rules
* **AI Trading Explanation**: AI-generated explanations for trading decisions

### 🏛️ Mayor's Real-time Guidance
* **Zone-Side Only**: BUY@BLUE / SELL@ORANGE rules
* **Trust System**: ML Model Trust 40% + N/B Guild Trust 82%
* **Real-time Updates**: 5-second interval updates with state persistence
* **Trust Balance Display**: ML: 40% | N/B: 82% with detailed status

### 👥 Village Residents
* **Scout (Explorer)**: Quick signals, 1m & 3m monitoring
* **Guardian (Protector)**: Trend protection, 5m & 10m analysis
* **Analyst (Strategist)**: Strategic analysis, 15m & 30m patterns
* **Elder (Advisor)**: Long-term wisdom, 1h & daily perspectives
* **Trader_A ~ Trader_F**: Additional trainers with unique strategies

### 🔄 Real-time Synchronization
* **UI State Persistence**: localStorage-based state management
* **Auto Restoration**: Automatic state restoration on page refresh
* **jQuery Integration**: Enhanced AJAX and DOM manipulation
* **Real-time Trust Display**: Current time, minute candle info, zone status

## 🎮 Features

### Core Trading System
* Real-time chart UI (NB wave, EMA/SMA/Ichimoku, order markers)
* Auto trading loop (paper/live), partial close protection
* Sizing from UI top PnL bar (profit/loss ratios)
* ML: time-series CV, light hyper-parameter search, calibrated probs, per-timeframe models
* One-click backtest on current chart; rolling Win% and top PnL slider
* Assets panel (KRW and holdings), auto refresh

### 8BIT Village System
* **Village Energy Management**: HP and Stamina system
* **Bitcar Energy Injection**: Energy collection and injection process
* **Trainer Warehouse**: Real-time trade recording and analysis
* **Trade Journal**: Mayor's guidance compliance and ML model judgments
* **AI Trading Explanations**: Why buy/sell decisions are made
* **Auto Learning System**: Automatic Mayor's guidance learning

### N/B COIN System (from 0.9.5)
* N/B COIN S.L (Save/Load-like) UI cards per timeframe
* Masonry layout (3 columns) with smooth animation (1.2s)
* Current chart timeframe card is featured (full-width, double height)
* Per-card actions: BUY / SELL (bound to current bucket), Copy (card text)
* N/B COIN accounting per timeframe (coin_count)

## 🚀 Quick Start

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

## ⚙️ Configuration

Create `bot/.env` with your keys and options:

```env
UPBIT_ACCESS_KEY=...
UPBIT_SECRET_KEY=...
UI_PORT=5057
UI_HTTPS=false
NB_HIGH=0.55
NB_LOW=0.45
```

### Optional Runtime Overrides (via UI → Config API)

* `ml_only=true|false`: Follow ML NB direction strictly within the loop
* `ml_seg_only=true|false`: Trade on NB line crosses (extreme-only gate)

## 🏗️ Repository Structure

```
8BIT/
├── bot/
│   ├── server.py              # Flask server and trading/ML APIs
│   ├── static/
│   │   ├── ui.html            # Web UI
│   │   ├── ui.js              # UI logic
│   │   └── mayor-guidance.js  # Mayor's guidance system
│   ├── trade.py               # Order sizing and execution
│   ├── strategy.py            # Signal helpers
│   ├── main.py                # Simple loop (optional)
│   └── requirements.txt       # Python dependencies
├── STORY/
│   ├── 8BIT_VILLAGE_SCENARIO.md      # Korean scenario
│   └── 8BIT_VILLAGE_SCENARIO_EN.md   # English scenario
├── data/                      # Chart data and NB params
├── img/                       # Images and screenshots
└── music/                     # Audio files
```

## 🏰 8BIT Village System Details

### Mayor's Guidance System
The Mayor provides real-time guidance based on:
- **Zone-Side Only Rules**: BUY only in BLUE zone, SELL only in ORANGE zone
- **Trust-based Decisions**: Weighted confidence calculation
- **Real-time Updates**: 5-second interval updates with state persistence

### Village Residents
Each resident has:
- **Unique Role**: Specific trading specialty and strategy
- **Energy System**: HP and Stamina management
- **Bitcar Integration**: Energy injection for trading
- **Warehouse Records**: Real-time trade recording and analysis

### ML Model Learning
- **Automatic Learning**: Learns Mayor's guidance rules
- **Trust Integration**: ML 40% + N/B Guild 82% trust system
- **AI Explanations**: Provides reasoning for trading decisions

## 🔒 Safety

* Test in paper mode first. Trading involves risk.
* Never commit `.env` or private keys.
* The 8BIT Village system includes multiple safety mechanisms:
  - Position Lock System
  - Energy-based trade limitations
  - Trust-based decision making
  - Real-time monitoring and alerts

## 📚 Documentation

* [8BIT Village Scenario (Korean)](STORY/8BIT_VILLAGE_SCENARIO.md)
* [8BIT Village Scenario (English)](STORY/8BIT_VILLAGE_SCENARIO_EN.md)
* [Original Bot Documentation](bot/README.md)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and ensure all tests pass before submitting a pull request.

## 📄 License

MIT — see `LICENSE` file.

---

*8BIT Village Trading System - Where AI meets community-driven trading strategies* 🏰✨


