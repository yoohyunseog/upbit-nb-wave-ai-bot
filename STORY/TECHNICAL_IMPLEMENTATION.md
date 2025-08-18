# Technical Implementation: Village Residents and Mayor System

## Overview

This document details the technical implementation of the Village Residents and Mayor system in the N/B Wave AI Bot, including data structures, algorithms, and integration points.

## Data Structures

### Guild Members Object

```javascript
let guildMembers = {
  mayor: {
    name: 'Mayor',
    hp: 100,
    maxHp: 100,
    stamina: 100,
    maxStamina: 100,
    location: 'Town Hall',
    role: 'Leader',
    trainerCards: ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'day'],
    specialty: 'Village Leadership',
    description: 'Oversees all trading strategies and coordinates village efforts',
    // Trading records
    realTrades: [],
    mockTrades: [],
    totalProfit: 0,
    winRate: 0,
    lastTrade: null,
    // Auto learning system
    skillLevel: 2.0,
    experience: 0,
    learningRate: 0.15,
    autoTradingEnabled: true,
    lastAutoTrade: null,
    tradeFrequency: 0.4,
    strategy: 'balanced'
  },
  scout: {
    name: 'Scout',
    hp: 85,
    maxHp: 100,
    stamina: 70,
    maxStamina: 100,
    location: 'Gate',
    role: 'Explorer',
    trainerCards: ['minute1', 'minute3'],
    specialty: 'Quick Signals',
    description: 'Monitors 1m & 3m charts for rapid opportunities',
    // Trading records
    realTrades: [],
    mockTrades: [],
    totalProfit: 0,
    winRate: 0,
    lastTrade: null,
    // Auto learning system
    skillLevel: 1.0,
    experience: 0,
    learningRate: 0.1,
    autoTradingEnabled: true,
    lastAutoTrade: null,
    tradeFrequency: 0.3,
    strategy: 'momentum'
  },
  // ... similar structure for guardian, analyst, elder
};
```

## Core Functions

### 1. Member Decision Making

```javascript
function makeMemberDecision(member, price, interval) {
  const role = member.role;
  const skillLevel = member.skillLevel || 1.0;
  
  let decision = 'HOLD';
  
  if (role === 'Leader') {
    // Mayor's N/B Guild directive implementation
    const currentZone = Math.random() > 0.5 ? 'ORANGE' : 'BLUE';
    
    if (currentZone === 'ORANGE') {
      // Ultra-cautious approach in Orange zone
      const holdBias = 0.60;
      const tradeDecision = Math.random() > 0.6 ? 'BUY' : 'SELL';
      
      if (Math.random() < holdBias) {
        decision = 'HOLD';
      } else {
        decision = tradeDecision;
      }
      member.strategy = 'ultra_cautious';
    } else {
      // Aggressive approach in Blue zone
      const buyBias = 0.70;
      decision = Math.random() > buyBias ? 'SELL' : 'BUY';
      member.strategy = 'aggressive';
    }
  } else if (role === 'Explorer') {
    decision = Math.random() > 0.6 ? 'BUY' : 'SELL';
  } else if (role === 'Protector') {
    decision = Math.random() > 0.7 ? 'BUY' : 'SELL';
  } else if (role === 'Strategist') {
    decision = Math.random() > 0.5 ? 'BUY' : 'SELL';
  } else if (role === 'Advisor') {
    decision = Math.random() > 0.8 ? 'BUY' : 'SELL';
  }
  
  // Skill level affects decision quality
  if (skillLevel > 1.5) {
    if (Math.random() < (skillLevel - 1.0) * 0.2) {
      decision = 'HOLD';
    }
  }
  
  return decision;
}
```

### 2. Strategy Effectiveness Calculation

```javascript
function getStrategyEffectiveness(strategy) {
  const effectiveness = {
    'ultra_cautious': 0.25, // Mayor's Orange zone strategy
    'aggressive': 0.22,     // Mayor's Blue zone strategy
    'defensive': 0.18,      // Legacy defensive strategy
    'balanced': 0.12,       // Balanced approach
    'momentum': 0.1,        // Scout's strategy
    'meanrev': 0.05,        // Mean reversion
    'breakout': 0.15,       // Breakout strategy
    'scalping': 0.08        // Scalping strategy
  };
  return effectiveness[strategy] || 0;
}
```

### 3. Auto Trading Scheduler

```javascript
function autoMockTradingScheduler() {
  if (typeof guildMembers === 'undefined' || !guildMembers) return;
  
  const now = Date.now();
  const fiveMinutes = 5 * 60 * 1000;
  
  Object.values(guildMembers).forEach(member => {
    if (!member.autoTradingEnabled || member.stamina < 10) return;
    
    // Check if enough time has passed since last trade
    if (member.lastAutoTrade && (now - member.lastAutoTrade) < fiveMinutes) return;
    
    // Check trade frequency
    if (Math.random() > member.tradeFrequency) return;
    
    // Execute auto mock trade
    executeAutoMockTrade(member);
  });
}
```

### 4. Mock Trade Execution

```javascript
async function executeAutoMockTrade(member) {
  if (member.stamina < 10) return;
  
  // Consume stamina
  member.stamina = Math.max(0, member.stamina - 10);
  
  // Get current market data
  const price = lastPrice || 160000000;
  const interval = member.trainerCards[Math.floor(Math.random() * member.trainerCards.length)];
  
  // Make trading decision
  const decision = makeMemberDecision(member, price, interval);
  
  if (decision === 'HOLD') {
    member.lastAutoTrade = Date.now();
    return;
  }
  
  // Calculate confidence and simulate trade result
  const confidence = calculateMemberConfidence(member);
  const profitPercent = simulateTradeResult(decision, member, price, confidence);
  
  // Record the trade
  recordMockTrade(profitPercent);
  
  // Update member stats
  member.lastAutoTrade = Date.now();
  member.lastTrade = {
    type: 'MOCK',
    decision: decision,
    profit: profitPercent,
    timestamp: new Date().toLocaleTimeString()
  };
  
  // Update member stats
  updateMemberStats(member);
}
```

### 5. Trade Recording System

```javascript
function recordMockTrade(profitPercent) {
  const activeMembers = Object.values(guildMembers).filter(m => m.stamina > 0);
  if (activeMembers.length === 0) return;
  
  const randomMember = activeMembers[Math.floor(Math.random() * activeMembers.length)];
  
  randomMember.mockTrades.push({
    profit: profitPercent,
    timestamp: Date.now(),
    decision: profitPercent > 0 ? 'WIN' : 'LOSS'
  });
  
  updateMemberStats(randomMember);
}

function recordRealTrade(side, price, size, profit) {
  // Determine responsible member based on interval or random selection
  const responsibleMember = determineResponsibleMember();
  
  responsibleMember.realTrades.push({
    side: side,
    price: price,
    size: size,
    profit: profit,
    timestamp: Date.now()
  });
  
  responsibleMember.lastTrade = {
    type: 'REAL',
    side: side,
    profit: profit,
    timestamp: new Date().toLocaleTimeString()
  };
  
  updateMemberStats(responsibleMember);
}
```

### 6. Member Statistics Update

```javascript
function updateMemberStats(member) {
  const allTrades = [...member.realTrades, ...member.mockTrades];
  
  if (allTrades.length === 0) {
    member.totalProfit = 0;
    member.winRate = 0;
    return;
  }
  
  // Calculate total profit
  member.totalProfit = allTrades.reduce((sum, trade) => {
    return sum + (trade.profit || 0);
  }, 0);
  
  // Calculate win rate
  const winningTrades = allTrades.filter(trade => (trade.profit || 0) > 0);
  member.winRate = (winningTrades.length / allTrades.length) * 100;
}
```

## UI Integration

### 1. Status Display Function

```javascript
function updateGuildMembersStatus() {
  const statusDiv = document.getElementById('integratedGuildStatus');
  if (!statusDiv) return;
  
  let html = '';
  
  // N/B Stamina Status
  html += `<div class="mb-2">
    <strong>⚡ N/B Stamina:</strong> ${nbStamina || 0}/100 (${Math.round(((nbStamina || 0) / 100) * 100)}%)
    <br><small class="text-muted">Treasury Access: ${(nbStamina || 0) >= 80 ? 'Unlocked' : 'Locked'}</small>
  </div>`;
  
  // Guild Members Status
  html += '<div class="guild-members-grid">';
  
  Object.values(guildMembers).forEach(member => {
    const hpPercent = Math.round((member.hp / member.maxHp) * 100);
    const staminaPercent = Math.round((member.stamina / member.maxStamina) * 100);
    const autoStatus = member.autoTradingEnabled && member.stamina >= 10 ? '🟢 활성' : '🔴 비활성';
    
    html += `
      <div class="member-card">
        <div class="member-header">
          <strong>${member.name}</strong> (${member.role}) [${member.location}]
        </div>
        <div class="member-specialty">
          ${member.specialty}: ${member.description}
        </div>
        <div class="member-cards">
          Cards: ${member.trainerCards.join(', ')}
        </div>
        <div class="member-stats">
          Profit: ${member.totalProfit > 0 ? '+' : ''}${member.totalProfit.toFixed(2)}%
          Win Rate: ${member.winRate.toFixed(1)}%
          Real: ${member.realTrades.length} Mock: ${member.mockTrades.length}
        </div>
        <div class="member-last-trade">
          Last: ${member.lastTrade ? `${member.lastTrade.type} ${member.lastTrade.profit > 0 ? '+' : ''}${member.lastTrade.profit?.toFixed(2)}%` : '없음'}
        </div>
        <div class="member-status">
          HP: ${member.hp}/${member.maxHp} Stamina: ${member.stamina}/${member.maxStamina}
          <br>${autoStatus} | Skill: ${member.skillLevel.toFixed(1)}
        </div>
      </div>
    `;
  });
  
  html += '</div>';
  statusDiv.innerHTML = html;
}
```

### 2. Emergency Reset Function

```javascript
async function emergencyStaminaReset() {
  try {
    // Reset N/B Stamina
    nbStamina = 100;
    
    // Reset all guild members
    Object.values(guildMembers).forEach(member => {
      member.hp = member.maxHp;
      member.stamina = member.maxStamina;
      member.lastAutoTrade = null; // Reset cooldown
    });
    
    // Update UI
    updateGuildMembersStatus();
    
    // Log the reset
    pushOrderLogLine(`🆘 Emergency Stamina Reset completed at ${new Date().toLocaleTimeString()}`);
    
  } catch (error) {
    console.error('Emergency reset error:', error);
    pushOrderLogLine(`❌ Emergency reset failed: ${error.message}`);
  }
}
```

## Learning System

### 1. Trainer Learning System

```javascript
function trainerLearningSystem() {
  if (typeof guildMembers === 'undefined' || !guildMembers) return;
  
  Object.values(guildMembers).forEach(member => {
    // Increase experience
    member.experience += 1;
    
    // Update skill level based on performance
    if (member.winRate > 60 && member.totalProfit > 10) {
      member.skillLevel += member.learningRate * 0.1;
    } else if (member.winRate < 40 && member.totalProfit < -5) {
      member.skillLevel = Math.max(0.5, member.skillLevel - member.learningRate * 0.05);
    }
    
    // Enhance specialty based on skill level
    member.specialty = enhanceSpecialty(member.specialty, member.skillLevel);
    
    // Evolve strategy occasionally
    if (Math.random() < 0.1) {
      evolveStrategy(member);
    }
  });
}
```

### 2. Specialty Enhancement

```javascript
function enhanceSpecialty(specialty, skillLevel) {
  if (skillLevel > 2.0) {
    return `Master ${specialty}`;
  } else if (skillLevel > 1.5) {
    return `Advanced ${specialty}`;
  } else if (skillLevel > 1.0) {
    return `Experienced ${specialty}`;
  }
  return specialty;
}
```

## API Integration Points

### 1. Status API Integration

```javascript
async function runTrainerDiagnostics() {
  try {
    // Fetch bot status
    const statusResponse = await fetchJsonStrict('/api/bot/status');
    const status = statusResponse;
    
    // Update N/B Stamina
    nbStamina = status.nb_stamina || 0;
    
    // Update guild members with real data
    updateGuildMembersWithRealData(status);
    
    // Update UI
    updateGuildMembersStatus();
    
  } catch (error) {
    console.error('Trainer diagnostics error:', error);
  }
}
```

### 2. Real Trade Integration

```javascript
// Override global pushOrderLogLine to capture real trades
window.pushOrderLogLine = function(line) {
  // Original functionality
  if (window.originalPushOrderLogLine) {
    window.originalPushOrderLogLine(line);
  }
  
  // Parse trade information
  const buyMatch = line.match(/BUY.*?(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)/);
  const sellMatch = line.match(/SELL.*?(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)/);
  
  if (buyMatch) {
    const price = parseFloat(buyMatch[1]);
    const size = parseFloat(buyMatch[2]);
    recordRealTrade('BUY', price, size, 0);
  } else if (sellMatch) {
    const price = parseFloat(sellMatch[1]);
    const size = parseFloat(sellMatch[2]);
    recordRealTrade('SELL', price, size, 0);
  }
};
```

## Performance Considerations

### 1. Memory Management
- Guild members data is kept in memory for fast access
- Trade history is limited to prevent memory bloat
- Regular cleanup of old trade records

### 2. Update Frequency
- Status updates: Every 30 seconds
- Auto trading scheduler: Every 1 minute
- Learning system: Every 5 minutes

### 3. Error Handling
- Graceful degradation when APIs fail
- Fallback to simulated data
- Comprehensive error logging

## Future Enhancements

### 1. Advanced AI Integration
- Machine learning models for each member
- Predictive analytics for market conditions
- Adaptive strategy selection

### 2. Enhanced UI Features
- Real-time charts for each member
- Performance comparison tools
- Interactive strategy configuration

### 3. Scalability Improvements
- Database storage for trade history
- Multi-market support
- Distributed processing capabilities

---

*This technical implementation provides a robust foundation for the Village Residents and Mayor system, ensuring reliable operation and future extensibility.*
