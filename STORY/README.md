# Village Residents and Mayor Scenario Documentation

## Overview

This folder contains comprehensive documentation for the Village Residents and Mayor system implemented in the N/B Wave AI Bot. The system creates an engaging trading experience by personifying AI trading strategies as village residents with unique roles, specialties, and learning capabilities.

## Files

### 1. VILLAGE_RESIDENTS_AND_MAYOR.md
**Main scenario document** containing:
- Complete overview of the village system
- Detailed descriptions of the Mayor and all village residents
- N/B Guild directives and their implementation
- Trading system mechanics and learning algorithms
- Integration with the N/B Wave AI Bot

### 2. TECHNICAL_IMPLEMENTATION.md
**Technical documentation** including:
- Data structures and object definitions
- Core functions and algorithms
- UI integration code examples
- API integration points
- Performance considerations and future enhancements

## Key Features

### 🏛️ The Mayor (촌장)
- **Role**: Village Leader and N/B Guild Branch Manager
- **Location**: Town Hall
- **Specialty**: Village Leadership and Financial Management
- **N/B Guild Directives**:
  - **Orange Zone**: Ultra-cautious approach, 60% HOLD, quick profit taking
  - **Blue Zone**: Aggressive approach, 70% BUY bias, alpha strategy

### 👥 Village Residents (Guild Members)

#### Scout (정찰병) - Explorer
- **Location**: Village Gate
- **Specialty**: Quick Signals (1m & 3m charts)
- **Strategy**: Momentum-based quick trades

#### Guardian (수호자) - Protector  
- **Location**: Market
- **Specialty**: Trend Protection (5m & 10m charts)
- **Strategy**: Conservative risk management

#### Analyst (분석가) - Strategist
- **Location**: Tower
- **Specialty**: Strategic Analysis (15m & 30m charts)
- **Strategy**: Pattern-based strategic decisions

#### Elder (장로) - Advisor
- **Location**: Inn
- **Specialty**: Long-term Wisdom (1h & daily charts)
- **Strategy**: Conservative long-term approach

## System Mechanics

### N/B Stamina System
- Governs trading actions and prevents overtrading
- Recovery through profitable mock tests
- Emergency reset functionality

### Mock Trading System
- Real market data-based simulation
- Profit calculation using volatility, bias, and member expertise
- Automated learning and skill improvement

### Auto-Learning System
- Skill level increases based on performance
- Experience accumulation over time
- Strategy evolution and specialty enhancement

## Integration

The Village Residents and Mayor system is fully integrated with:
- **Real-time UI**: Live status monitoring and controls
- **API System**: Status updates and trade execution
- **Learning Algorithms**: Performance-based improvement
- **Emergency Controls**: Quick reset and healing functions

## Usage

1. **Start the Bot**: Launch the N/B Wave AI Bot server
2. **Access UI**: Open the web interface at `http://127.0.0.1:5057/ui`
3. **Monitor Status**: View real-time guild members and auto-trading status
4. **Emergency Controls**: Use reset and healing functions as needed
5. **Performance Tracking**: Monitor individual and collective trading performance

## Technical Requirements

- **Python 3.10+**: For the bot server
- **Modern Browser**: For the web UI
- **Upbit API Keys**: For live trading (optional)
- **Internet Connection**: For market data and API access

## Future Development

Planned enhancements include:
- Advanced AI integration with machine learning models
- Enhanced UI features with real-time charts
- Multi-market support and scalability improvements
- Database storage for comprehensive trade history

---

*This documentation provides a complete guide to understanding and using the Village Residents and Mayor system in the N/B Wave AI Bot.*
