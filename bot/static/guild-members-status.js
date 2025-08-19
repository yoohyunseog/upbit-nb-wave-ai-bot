// ========================================
// 길드 멤버 상태 관리 시스템 (Guild Members Status)
// ========================================

// Guild Members Status 업데이트 메인 함수
async function updateGuildMembersStatus() {
  try {
    console.log('🔄 Updating Guild Members Status...');
    
    const guildContainer = document.getElementById('integratedGuildStatus');
    if (!guildContainer) {
      console.error('❌ integratedGuildStatus not found in updateGuildMembersStatus');
      return;
    }

    // Check if guildMembers is available
    if (typeof window.guildMembers === 'undefined' || !window.guildMembers) {
      console.log('⚠️ Guild members not initialized yet in updateGuildMembersStatus');
      return;
    }

    const guildMembers = window.guildMembers;
    console.log('📊 Found guild members:', Object.keys(guildMembers));

    // Clear existing content
    guildContainer.innerHTML = '';

    // Create header
    const headerDiv = document.createElement('div');
    headerDiv.style.cssText = 'font-size: 11px; color: #d9e2f3; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);';
    headerDiv.textContent = 'Guild Members Status';
    guildContainer.appendChild(headerDiv);

    // Create member cards
    Object.values(guildMembers).forEach(member => {
      const memberDiv = createMemberCard(member);
      guildContainer.appendChild(memberDiv);
    });

    // 모든 길드 멤버의 상태 업데이트 (저장된 상태 복원 우선)
    Object.values(guildMembers).forEach(member => {
      // 먼저 저장된 상태 복원 시도 - check if functions are available
      let guidanceRestored = false;
      let aiExplanationRestored = false;
      
      if (typeof window.restoreMayorGuidanceStatus === 'function') {
        guidanceRestored = window.restoreMayorGuidanceStatus(member.name);
      }
      
      if (typeof window.restoreAIExplanation === 'function') {
        aiExplanationRestored = window.restoreAIExplanation(member.name);
      }

      // 실시간 업데이트 (복원되지 않은 경우에만)
      if (!guidanceRestored && typeof window.getMayorGuidanceStatus === 'function') {
        window.getMayorGuidanceStatus(member).then(guidanceHtml => {
          const guidanceElement = document.getElementById(`mayor-guidance-${member.name}`);
          if (guidanceElement) {
            guidanceElement.innerHTML = guidanceHtml;
          }
        }).catch(e => console.error('Error updating mayor guidance status:', e));
      }

      if (!aiExplanationRestored && typeof window.getAIExplanation === 'function') {
        window.getAIExplanation(member.name).catch(e => console.error('Error updating AI explanation:', e));
      }
    });

  } catch (e) {
    console.error('Error updating integrated guild status:', e);
  }
}

// 개별 멤버 카드 생성
function createMemberCard(member) {
  const memberDiv = document.createElement('div');
  memberDiv.className = 'guild-member-card';
  memberDiv.style.cssText = `
    background: linear-gradient(135deg, rgba(25,118,210,0.15), rgba(25,118,210,0.05));
    border: 1px solid rgba(25,118,210,0.3);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
  `;

  // 멤버 정보 생성
  const memberInfo = generateMemberInfoHTML(member);
  const tradeSlide = generateTradeSlideHTML(member);
  const memberStatus = generateMemberStatusHTML(member);

  memberDiv.innerHTML = `
    ${memberInfo}
    ${tradeSlide}
    ${memberStatus}
  `;

  return memberDiv;
}

// 멤버 기본 정보 HTML 생성
function generateMemberInfoHTML(member) {
  const currentPrice = typeof window.getCurrentPrice === 'function' ? window.getCurrentPrice() : 160000000;
  const warehouseValue = member.nbCoins * currentPrice;
  const profitColor = member.totalProfit > 0 ? '#0ecb81' : member.totalProfit < 0 ? '#f6465d' : '#ffffff';
  
  return `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-weight: 600; color: #ffffff; font-size: 12px;">${member.name}</span>
        <span style="font-size: 10px; color: #888888;">(${member.role})</span>
        <span style="font-size: 9px; color: #ffb703; background: rgba(255,183,3,0.1); padding: 2px 6px; border-radius: 10px;">
          Level ${member.skillLevel.toFixed(1)}
        </span>
      </div>
      <div style="font-size: 10px; color: ${profitColor};">
        ${member.totalProfit > 0 ? '+' : ''}${member.totalProfit.toFixed(2)}%
      </div>
    </div>
    
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
      <div style="font-size: 10px; color: #888888;">
        N/B 코인: ${member.nbCoins.toFixed(8)} (≈ ${Math.round(warehouseValue).toLocaleString()} KRW)
      </div>
      <div style="font-size: 10px; color: #888888;">
        승률: ${member.winRate.toFixed(1)}% (${member.totalTrades}회)
      </div>
    </div>
  `;
}

// 트레이드 슬라이드 HTML 생성
function generateTradeSlideHTML(member) {
  const hasPosition = member.openPosition !== null;
  const currentPrice = typeof window.getCurrentPrice === 'function' ? window.getCurrentPrice() : 160000000;
  
  if (!hasPosition) {
    return `
      <div style="font-size:11px; color:#888888; padding:8px; background:rgba(255,255,255,0.05); border-radius:4px; text-align:center;">
        📊 No active position
      </div>
    `;
  }
  
  const entryPrice = member.averagePrice || member.openPosition.price;
  const coinAmount = member.totalPositionSize || member.openPosition.coinAmount;
  const positionSide = member.openPosition.side;
  const tradeStartTime = new Date(member.openPosition.timestamp);
  const timeHeld = Date.now() - tradeStartTime.getTime();
  const minutesHeld = Math.floor(timeHeld / (1000 * 60));
  
  // Calculate P&L
  let currentPnl = 0;
  if (positionSide === 'BUY') {
    currentPnl = ((currentPrice - entryPrice) / entryPrice) * 100;
  } else {
    currentPnl = ((entryPrice - currentPrice) / entryPrice) * 100;
  }
  
  const pnlColor = currentPnl > 0 ? '#0ecb81' : currentPnl < 0 ? '#f6465d' : '#ffffff';
  const pnlBgColor = currentPnl > 0 ? 'rgba(14,203,129,0.1)' : currentPnl < 0 ? 'rgba(246,70,93,0.1)' : 'rgba(255,255,255,0.05)';
  
  // Sell prediction logic
  const sellPrediction = calculateSellPrediction(member, currentPnl, minutesHeld);
  
  return `
    <div style="font-size:11px; color:#ffffff; background:rgba(0,209,255,0.1); border-radius:6px; padding:8px; border-left:3px solid #00d1ff;">
      <!-- Trade Progress Bar -->
      <div class="mb-2">
        <div class="d-flex justify-content-between align-items-center mb-1">
          <span style="font-size:10px; color:#ffffff;">📊 ${positionSide} ${coinAmount.toFixed(8)} BTC</span>
          <span style="font-size:10px; color:${pnlColor}; font-weight:600;">
            ${currentPnl > 0 ? '+' : ''}${currentPnl.toFixed(2)}%
          </span>
        </div>
        
        <!-- P&L Progress Bar -->
        <div style="width:100%; height:6px; background:#2b3139; border-radius:3px; overflow:hidden;">
          <div style="width:${Math.min(100, Math.abs(currentPnl) * 10)}%; height:100%; background:${pnlColor}; transition: width 0.3s ease;"></div>
        </div>
        
        <div style="font-size:9px; color:#888888; margin-top:2px;">
          진입: ${Math.round(entryPrice).toLocaleString()} KRW | 현재: ${Math.round(currentPrice).toLocaleString()} KRW
        </div>
        <div style="font-size:9px; color:#888888;">
          보유시간: ${minutesHeld}분 | ${sellPrediction}
        </div>
      </div>
    </div>
  `;
}

// 멤버 상태 HTML 생성
function generateMemberStatusHTML(member) {
  return `
    <!-- 촌장 지침 상태 -->
    <div style="font-size: 9px; color: #888888; margin-top: 8px;" id="mayor-guidance-${member.name}">
      🏛️ 촌장 지침: 로딩 중...
    </div>
    
    <!-- 자동 학습 상태 표시 -->
    <div style="font-size: 8px; color: #888888; margin-top: 2px;" id="auto-learning-status-${member.name}">
      자동 학습: 로딩 중...
    </div>
    
    <!-- AI 거래 설명 -->
    <div style="font-size: 9px; color: #888888; margin-top: 4px; padding: 4px; background: rgba(255,255,255,0.03); border-radius: 3px;" id="ai-explanation-${member.name}">
      AI 거래 판단: 로딩 중...
    </div>
    
    <!-- 촌장 지침 학습 모델 훈련 버튼 -->
    <div style="margin-top: 6px; display: flex; gap: 4px;">
      <button class="btn btn-sm btn-outline-primary" onclick="trainMayorGuidanceModel()" style="font-size: 8px; padding: 2px 4px;">
        촌장 지침 학습
      </button>
      <button class="btn btn-sm btn-outline-success" onclick="toggleAutoLearning()" style="font-size: 8px; padding: 2px 4px; margin-left: 2px;">
        자동 학습
      </button>
    </div>
  `;
}

// 매도 예측 계산
function calculateSellPrediction(member, currentPnl, minutesHeld) {
  // 기본 매도 예측 로직
  if (currentPnl > 5) {
    return "매도 신호 강함";
  } else if (currentPnl > 2) {
    return "매도 고려";
  } else if (currentPnl < -3) {
    return "손절 고려";
  } else if (minutesHeld > 60) {
    return "장기 보유";
  } else {
    return "관찰 중";
  }
}

// Get Guild Members Status for specific interval
function getGuildMembersStatusForInterval(interval) {
  try {
    // Check if guildMembers is available
    if (typeof window.guildMembers === 'undefined' || !window.guildMembers) {
      return {
        nbEnergy: 50,
        nbEnergyColor: '#ffb703',
        activeMembers: 0,
        treasuryAccess: false
      };
    }
    
    const guildMembers = window.guildMembers;
    
    // Calculate active members (those with stamina > 30)
    const activeMembers = Object.values(guildMembers).filter(member => member.stamina > 30).length;
    
    // Calculate N/B Energy percentage - check if nbEnergy is available
    let nbEnergyPercent = 50; // default value
    if (typeof window.nbEnergy !== 'undefined' && window.nbEnergy) {
      nbEnergyPercent = Math.round((window.nbEnergy.current / window.nbEnergy.max) * 100);
    }
    
    // Determine N/B Energy color
    let nbEnergyColor = '#f6465d'; // red
    if (nbEnergyPercent > 70) {
      nbEnergyColor = '#4285f4'; // blue
    } else if (nbEnergyPercent > 40) {
      nbEnergyColor = '#ffb703'; // yellow
    }
    
    // Different intervals have different guild member distributions
    const intervalModifiers = {
      'minute1': { energyBonus: 5, activeBonus: 1 },
      'minute3': { energyBonus: 3, activeBonus: 1 },
      'minute5': { energyBonus: 2, activeBonus: 0 },
      'minute10': { energyBonus: 0, activeBonus: 0 },
      'minute15': { energyBonus: -2, activeBonus: -1 },
      'minute30': { energyBonus: -3, activeBonus: -1 },
      'minute60': { energyBonus: -5, activeBonus: -2 },
      'day': { energyBonus: -10, activeBonus: -3 }
    };
    
    const modifier = intervalModifiers[interval] || { energyBonus: 0, activeBonus: 0 };
    const adjustedEnergy = Math.max(0, Math.min(100, nbEnergyPercent + modifier.energyBonus));
    const adjustedActive = Math.max(0, Math.min(4, activeMembers + modifier.activeBonus));
    
    return {
      nbEnergy: adjustedEnergy,
      nbEnergyColor: adjustedEnergy > 70 ? '#4285f4' : adjustedEnergy > 40 ? '#ffb703' : '#f6465d',
      activeMembers: adjustedActive,
      treasuryAccess: adjustedEnergy >= 80
    };

  } catch (e) {
    console.error('Error getting guild members status for interval:', e);
    return {
      nbEnergy: 50,
      nbEnergyColor: '#ffb703',
      activeMembers: 2,
      treasuryAccess: false
    };
  }
}

// Update Auto Trading Status Display (now integrated into updateGuildMembersStatus)
function updateAutoTradingStatus() {
  // This function is now integrated into updateGuildMembersStatus
  // Keeping it for compatibility but it's no longer needed
}

// Force start auto trading for testing
function forceStartAutoTrading() {
  try {
    if (typeof window.guildMembers === 'undefined' || !window.guildMembers) {
      console.log('Guild members not initialized yet');
      return;
    }
    
    const guildMembers = window.guildMembers;
    
    Object.values(guildMembers).forEach(member => {
      if (!member.autoTrading) {
        member.autoTrading = true;
        member.lastTradeTime = Date.now();
        console.log(`🚀 Force started auto trading for ${member.name}`);
      }
    });
    
    // Update display
    updateGuildMembersStatus().catch(e => console.error('Error updating guild members status:', e));
    
  } catch (e) {
    console.error('Error force starting auto trading:', e);
  }
}

// Initialize guild members status system
function initializeGuildMembersStatusSystem() {
  console.log('🏰 Guild Members Status System 초기화 시작...');
  
  // Check if required elements exist
  const guildContainer = document.getElementById('integratedGuildStatus');
  if (!guildContainer) {
    console.error('❌ integratedGuildStatus element not found');
    return;
  }
  console.log('✅ integratedGuildStatus found');
  
  // Check if guildMembers is available
  if (typeof window.guildMembers === 'undefined' || !window.guildMembers) {
    console.log('⚠️ Guild members not initialized yet, waiting...');
    // Retry after 3 seconds
    setTimeout(() => {
      initializeGuildMembersStatusSystem();
    }, 3000);
    return;
  }
  console.log('✅ guildMembers available:', Object.keys(window.guildMembers));
  
  // Set up periodic updates
  setInterval(() => {
    updateGuildMembersStatus().catch(e => console.error('Error updating guild members status:', e));
    updateAutoTradingStatus();
  }, 5 * 1000); // Every 5 seconds

  // Initial update
  setTimeout(() => {
    console.log('🔄 Initial guild members status update...');
    updateGuildMembersStatus().catch(e => console.error('Error updating guild members status:', e));
  }, 2000); // 2초 후 초기 업데이트

  // Expose functions globally
  window.updateGuildMembersStatus = updateGuildMembersStatus;
  window.createMemberCard = createMemberCard;
  window.generateMemberInfoHTML = generateMemberInfoHTML;
  window.generateTradeSlideHTML = generateTradeSlideHTML;
  window.generateMemberStatusHTML = generateMemberStatusHTML;
  window.calculateSellPrediction = calculateSellPrediction;
  window.getGuildMembersStatusForInterval = getGuildMembersStatusForInterval;
  window.updateAutoTradingStatus = updateAutoTradingStatus;
  window.forceStartAutoTrading = forceStartAutoTrading;
  
  console.log('✅ Guild Members Status System initialized successfully');
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    updateGuildMembersStatus,
    createMemberCard,
    generateMemberInfoHTML,
    generateTradeSlideHTML,
    generateMemberStatusHTML,
    calculateSellPrediction,
    getGuildMembersStatusForInterval,
    updateAutoTradingStatus,
    forceStartAutoTrading,
    initializeGuildMembersStatusSystem
  };
}
