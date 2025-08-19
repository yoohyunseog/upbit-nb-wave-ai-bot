// ========================================
// 🏰 8BIT Village Trading Process System
// ========================================

// 🎯 구역 변경 시 주민들의 학습 모델 매매 전략 프로세스
function executeVillageTradingProcess(member, currentZone, previousZone) {
  console.log(`🏰 ${member.name}의 매매 전략 프로세스 시작 - 구역 변경: ${previousZone} → ${currentZone}`);
  
  // 1단계: 현재 구역에서 SELL/BUY 시 손실 예상
  const profitLossPrediction = predictProfitLoss(member, currentZone);
  
  // 2단계: 촌장 지침 준수 여부 판단
  const mayorGuidanceDecision = evaluateMayorGuidance(member, currentZone, profitLossPrediction);
  
  // 3단계: 실제/모의 거래 판단
  const tradeTypeDecision = decideTradeType(member, mayorGuidanceDecision);
  
  // 4단계: 실행
  const executionResult = executeTradeDecision(member, tradeTypeDecision);
  
  // 결과를 창고에 저장
  saveTradingProcessResult(member, {
    zoneChange: `${previousZone} → ${currentZone}`,
    profitLossPrediction,
    mayorGuidanceDecision,
    tradeTypeDecision,
    executionResult,
    timestamp: Date.now()
  });
  
  return {
    process: 'Village Trading Process',
    member: member.name,
    zoneChange: `${previousZone} → ${currentZone}`,
    steps: {
      step1: profitLossPrediction,
      step2: mayorGuidanceDecision,
      step3: tradeTypeDecision,
      step4: executionResult
    }
  };
}

// 1단계: 손실 예상
function predictProfitLoss(member, currentZone) {
  const currentPrice = member.currentPrice || 160000000;
  const entryPrice = member.entryPrice || currentPrice;
  const position = member.position || 'FLAT';
  
  // 현재 포지션에 따른 손익 계산
  let currentPnl = 0;
  if (position === 'LONG') {
    currentPnl = ((currentPrice - entryPrice) / entryPrice) * 100;
  } else if (position === 'SHORT') {
    currentPnl = ((entryPrice - currentPrice) / entryPrice) * 100;
  }
  
  // SELL 시 예상 손익 (현재 포지션 청산)
  const sellPrediction = {
    action: 'SELL',
    expectedPnl: currentPnl,
    risk: currentPnl < 0 ? '손실 위험' : '수익 기대',
    confidence: Math.abs(currentPnl) > 2 ? '높음' : '보통'
  };
  
  // BUY 시 예상 손익 (새로운 포지션 진입)
  const buyPrediction = {
    action: 'BUY',
    expectedPnl: currentZone === 'BLUE' ? 1.5 : -0.8, // 구역별 예상 수익률
    risk: currentZone === 'BLUE' ? '낮음' : '높음',
    confidence: currentZone === 'BLUE' ? '높음' : '낮음'
  };
  
  return {
    currentPnl: currentPnl,
    sellPrediction: sellPrediction,
    buyPrediction: buyPrediction,
    recommendation: currentPnl > 1 ? 'SELL 권장' : (currentZone === 'BLUE' ? 'BUY 권장' : 'HOLD 권장')
  };
}

// 2단계: 촌장 지침 준수 여부 판단
function evaluateMayorGuidance(member, currentZone, profitLossPrediction) {
  const mayorGuidance = currentZone === 'BLUE' ? 'BUY만 허용' : 'SELL만 허용';
  const currentPosition = member.position || 'FLAT';
  
  let guidanceCompliance = '';
  let decision = '';
  let reason = '';
  
  if (currentZone === 'ORANGE') {
    if (currentPosition === 'LONG') {
      guidanceCompliance = '✅ 촌장 지침 준수';
      decision = 'SELL 실행';
      reason = 'ORANGE 구역에서 LONG 포지션 청산';
    } else if (currentPosition === 'FLAT') {
      guidanceCompliance = '✅ 촌장 지침 준수';
      decision = 'HOLD 유지';
      reason = 'ORANGE 구역에서 BUY 금지, SELL 기회 대기';
    }
  } else if (currentZone === 'BLUE') {
    if (currentPosition === 'FLAT') {
      guidanceCompliance = '✅ 촌장 지침 준수';
      decision = 'BUY 실행';
      reason = 'BLUE 구역에서 BUY 기회 포착';
    } else if (currentPosition === 'LONG') {
      guidanceCompliance = '✅ 촌장 지침 준수';
      decision = 'HOLD 유지';
      reason = 'BLUE 구역에서 LONG 포지션 유지';
    }
  }
  
  return {
    guidance: mayorGuidance,
    compliance: guidanceCompliance,
    decision: decision,
    reason: reason,
    confidence: Math.random() * 40 + 60 // 60-100%
  };
}

// 3단계: 실제/모의 거래 판단
function decideTradeType(member, mayorGuidanceDecision) {
  const confidence = mayorGuidanceDecision.confidence;
  const currentZone = member.currentZone || 'ORANGE';
  
  // 신뢰도에 따른 거래 타입 결정
  let tradeType = '';
  let reason = '';
  
  if (confidence >= 80) {
    tradeType = '실제 거래';
    reason = '높은 신뢰도로 실제 거래 실행';
  } else if (confidence >= 60) {
    tradeType = '모의 거래';
    reason = '보통 신뢰도로 모의 거래 실행';
  } else {
    tradeType = '관망';
    reason = '낮은 신뢰도로 거래 보류';
  }
  
  return {
    tradeType: tradeType,
    reason: reason,
    confidence: confidence,
    riskLevel: confidence >= 80 ? '높음' : (confidence >= 60 ? '보통' : '낮음')
  };
}

// 4단계: 실행
function executeTradeDecision(member, tradeTypeDecision) {
  const tradeType = tradeTypeDecision.tradeType;
  const confidence = tradeTypeDecision.confidence;
  
  let executionResult = {
    status: '대기 중',
    action: 'NONE',
    result: 'N/A',
    timestamp: Date.now()
  };
  
  if (tradeType === '실제 거래') {
    executionResult = {
      status: '실행 중',
      action: member.currentZone === 'BLUE' ? 'BUY' : 'SELL',
      result: '실제 거래 실행',
      timestamp: Date.now()
    };
  } else if (tradeType === '모의 거래') {
    executionResult = {
      status: '모의 실행',
      action: member.currentZone === 'BLUE' ? 'BUY' : 'SELL',
      result: '모의 거래 실행',
      timestamp: Date.now()
    };
  } else {
    executionResult = {
      status: '관망',
      action: 'HOLD',
      result: '거래 보류',
      timestamp: Date.now()
    };
  }
  
  return executionResult;
}

// 거래 프로세스 결과를 창고에 저장
function saveTradingProcessResult(member, result) {
  const warehouseKey = `trading_process_${member.name}`;
  const existingData = localStorage.getItem(warehouseKey);
  let processHistory = [];
  
  if (existingData) {
    processHistory = JSON.parse(existingData);
  }
  
  // 최근 10개만 유지
  processHistory.push(result);
  if (processHistory.length > 10) {
    processHistory = processHistory.slice(-10);
  }
  
  localStorage.setItem(warehouseKey, JSON.stringify(processHistory));
  console.log(`🏪 ${member.name}의 거래 프로세스 결과 저장됨`);
}

// 구역 변경 감지 및 프로세스 실행
function detectZoneChangeAndExecute(member, newZone) {
  const previousZone = member.lastZone || 'ORANGE';
  
  if (newZone !== previousZone) {
    console.log(`🔄 구역 변경 감지: ${member.name} - ${previousZone} → ${newZone}`);
    
    // 프로세스 실행
    const processResult = executeVillageTradingProcess(member, newZone, previousZone);
    
    // UI 업데이트
    updateMemberTradingProcessUI(member, processResult);
    
    // 마지막 구역 업데이트
    member.lastZone = newZone;
    
    return processResult;
  }
  
  return null;
}

// 멤버의 거래 프로세스 UI 업데이트
function updateMemberTradingProcessUI(member, processResult) {
  const memberElement = document.getElementById(`member-${member.name}`);
  if (!memberElement) return;
  
  const processInfo = `
    <div style="font-size: 8px; color: #888888; margin-top: 4px; padding: 2px; background: rgba(255,255,255,0.05); border-radius: 3px;">
      <div style="color: #00d1ff; font-weight: 600; margin-bottom: 2px;">🎯 매매 전략 프로세스</div>
      <div style="margin-bottom: 1px;">
        <span style="color: #ffb703;">1단계:</span> ${processResult.steps.step1.recommendation}
      </div>
      <div style="margin-bottom: 1px;">
        <span style="color: #0ecb81;">2단계:</span> ${processResult.steps.step2.decision}
      </div>
      <div style="margin-bottom: 1px;">
        <span style="color: #f6465d;">3단계:</span> ${processResult.steps.step3.tradeType}
      </div>
      <div style="margin-bottom: 1px;">
        <span style="color: #9c27b0;">4단계:</span> ${processResult.steps.step4.status}
      </div>
    </div>
  `;
  
  // 기존 프로세스 정보 제거
  const existingProcess = memberElement.querySelector('.trading-process-info');
  if (existingProcess) {
    existingProcess.remove();
  }
  
  // 새로운 프로세스 정보 추가
  const processDiv = document.createElement('div');
  processDiv.className = 'trading-process-info';
  processDiv.innerHTML = processInfo;
  memberElement.appendChild(processDiv);
}

// 모든 멤버의 거래 프로세스 모니터링 시작
function startVillageTradingProcessMonitoring() {
  console.log('🏰 8BIT Village 거래 프로세스 모니터링 시작');
  
  // 10초마다 구역 변경 확인
  setInterval(() => {
    Object.values(guildMembers).forEach(member => {
      // 현재 구역 정보 가져오기
      getMayorGuidanceData().then(data => {
        const currentZone = data.nbZone || 'ORANGE';
        detectZoneChangeAndExecute(member, currentZone);
      });
    });
  }, 10000); // 10초마다 체크
}

// 모듈 내보내기
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    executeVillageTradingProcess,
    predictProfitLoss,
    evaluateMayorGuidance,
    decideTradeType,
    executeTradeDecision,
    saveTradingProcessResult,
    detectZoneChangeAndExecute,
    updateMemberTradingProcessUI,
    startVillageTradingProcessMonitoring
  };
}
