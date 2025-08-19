# 🏰 8BIT Village Trading System Scenario

## 📖 Overview

8BIT Village is a unique village system for Bitcoin trading. Village residents have their own roles and strategies, and they perform trades according to the Mayor's guidance.

## 🏘️ Village Structure

### 🏛️ Mayor
- **Role**: Village leader, N/B Guild Branch Manager
- **Trust System**:
  - 🤖 ML Model Trust: 40% (Low trust)
  - 🏛️ N/B Guild Trust: 82% (High trust, 82 history records)
- **Guidance**: Zone-Side Only (BUY@BLUE / SELL@ORANGE)
- **Weighted Confidence Calculation**: (Personal Confidence * 0.6) + (ML Trust * 0.2) + (N/B Guild Trust * 0.2)

### 👥 Village Residents

#### 1. Scout (Explorer) [Gate]
- **Level**: 1.0-2.9
- **Specialty**: Quick Signals
- **Strategy**: 1m & 3m chart monitoring, rapid opportunity capture
- **Features**: momentum strategy, 60% frequency

#### 2. Guardian (Protector) [Market]
- **Level**: 1.0
- **Specialty**: Trend Protection
- **Strategy**: 5m & 10m trend protection and risk management
- **Features**: meanrev strategy, 50% frequency

#### 3. Analyst (Strategist) [Tower]
- **Level**: 1.0
- **Specialty**: Strategic Analysis
- **Strategy**: 15m & 30m pattern analysis
- **Features**: breakout strategy, 70% frequency

#### 4. Elder (Advisor) [Inn]
- **Level**: 1.0
- **Specialty**: Long-term Wisdom
- **Strategy**: Provides wisdom from 1h & daily perspectives
- **Features**: scalping strategy, 40% frequency

#### 5. Trader_A ~ Trader_F
- **Additional Trainers**: Each with unique strategies and characteristics

## 🚗 Bitcar Energy System

### Energy Injection Process
1. **Village Energy Collection**: Village residents collect energy from the village
2. **Bitcar Injection**: Inject collected energy into personal Bitcar
3. **Market Entry**: Ride the injected Bitcar to the bit market
4. **Trade Execution**: Execute trades according to Mayor's guidance

### Energy Management
- **HP (Health Points)**: 60-95/100
- **Stamina**: 70-90/100
- **Energy Consumption**: Energy consumption during trades
- **Recovery**: Energy recovery during rest

## 🏪 Trainer Warehouse System

### Real-time Trade Recording
- **Profit/Loss Recording**: Store profit rates and loss rates of all trades
- **Pattern Analysis**: Trade pattern analysis and learning
- **Strategy Enhancement**: Strategy improvement through warehouse data

### Trade Journal
- **Mayor's Guidance Compliance**: Record whether guidance was followed
- **Learning Model Judgment**: Record ML model's decision-making
- **Trade Reasons**: Record why trades were made or not made

## 🧠 ML Model Learning System

### Mayor's Guidance Learning
- **Learning Content**: Learn "BUY@BLUE / SELL@ORANGE" rules
- **Auto Learning**: Automatic learning via "🏛️ Mayor's Guidance Learning" button in UI
- **Trust-based**: Based on ML 40% + N/B Guild 82% trust

### AI Trading Explanation
- **Trade Reason Explanation**: Explain why not buying, when to buy, why selling, when to sell
- **Real-time Analysis**: Provide AI analysis of current situation

## 🎯 Trading System

### N/B (Narrative/Belief) System
- **Primary Signal**: Based on r value (0-1)
- **Zone Transition**: BLUE/ORANGE zone transition
- **Thresholds**: HIGH (0.55), LOW (0.45)
- **Zone Characteristics**:
  - 🔵 BLUE Zone: BUY bias
  - 🟠 ORANGE Zone: SELL/HOLD bias

### ML Model
- **Confirmation Layer**: Confirmation of N/B signals
- **Prediction**: Action prediction (-1, 0, 1)
- **Zone Prediction**: BLUE/ORANGE zone prediction

### Position Lock System
- **Cycle Enforcement**: BUY→SELL cycle enforcement
- **Risk Management**: Risk management through position locking

## 🔄 Real-time Synchronization

### UI Updates
- **5-second intervals**: Real-time data updates
- **State Persistence**: State storage through localStorage
- **Auto Restoration**: Automatic state restoration on page refresh

### Trust Display
- **Real-time Trust**: Real-time display of ML Model Trust, N/B Guild Trust
- **Trust Balance**: ML: 40% | N/B: 82%
- **N/B Zone Status**: 1h BLUE/ORANGE
- **Current Time**: Real-time time display
- **Minute Candle Info**: Current minute candle information

## 🏛️ Mayor's Real-time Guidance

### Zone-specific Guidance
- **🔵 BLUE Zone**: BUY only allowed (SELL prohibited)
- **🟠 ORANGE Zone**: SELL only allowed (BUY prohibited)

### Trust System
```
🤖 ML Model Trust: 40%
🏛️ N/B Guild Trust: 82% (82 history records)
⚖️ Trust Balance: ML: 40% | N/B: 82%
📍 N/B Zone Status: 1h [current zone]
⏰ Current Time: [real-time]
📊 Minute Candle Info: [current minute]
```

## 🎮 UI System

### Guild Members Status
- **Real-time Status**: Display current status of each resident
- **Trade Records**: Profit rate, win rate, recent trade information
- **Energy Status**: HP, Stamina display
- **Mayor's Guidance Compliance**: Display guidance compliance status

### Real-time Synchronization
- **N/B Zone**: 🟠ORANGE / 🔵BLUE
- **ML Zone**: 🟠ORANGE / 🔵BLUE
- **Sync Status**: Real-time synchronization status display

## 🔧 Technical Implementation

### Server Side (Python/Flask)
- **bot_ctrl**: Global state management
- **trade_loop**: Real-time trading loop
- **API Endpoints**: Various API provision
- **ML Model**: Learning and prediction system

### Client Side (JavaScript/jQuery)
- **Real-time Updates**: 5-second interval data updates
- **State Storage**: State management through localStorage
- **UI Synchronization**: Real-time UI synchronization
- **jQuery Usage**: AJAX and DOM manipulation

## 🎯 System Features

### Automation
- **Auto Learning**: Automatic Mayor's guidance learning
- **Auto Trading**: Automatic trade execution when conditions are met
- **Auto Recording**: Automatic recording of all trades

### Safety
- **Position Lock**: Position locking system
- **Energy Management**: Energy-based trade limitations
- **Trust-based**: Trust-based decision making

### Scalability
- **Modularization**: Function-based module separation
- **API-based**: RESTful API structure
- **Real-time**: Real-time data processing

## 🚀 Future Development Directions

### Feature Expansion
- **Additional Residents**: Add more trainers
- **Advanced Strategies**: More complex trading strategies
- **AI Enhancement**: More sophisticated AI analysis

### System Improvement
- **Performance Optimization**: Faster processing speed
- **Stability Enhancement**: More stable system
- **User Experience**: Better UI/UX

---

*This scenario is a complete guideline for the 8BIT Village Trading System. It includes all aspects of the system and is continuously updated.*
