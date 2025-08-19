// ========================================
// 🏛️ 촌장의 실시간 지침 시스템
// ========================================

// jQuery를 사용한 촌장 지침 데이터 가져오기
function getMayorGuidanceData() {
  return $.ajax({
    url: '/api/village/current-zone',
    method: 'GET',
    dataType: 'json',
    timeout: 5000
  }).then(function(result) {
    return {
      currentZone: result.current_zone || '',
      lastSignal: result.last_signal || '',
      position: result.position || '',
      nbZone: result.nb_zone || '',
      mlZone: result.ml_zone || '',
             rValue: result.r_value || 0.5,
       mlTrust: result.ml_trust || 40,
       nbTrust: result.nb_trust || 82,
      timestamp: result.timestamp || Date.now()
    };
  }).fail(function(xhr, status, error) {
    console.error('촌장 지침 데이터 가져오기 실패:', error);
    // 기본값 반환
    return {
      currentZone: 'ORANGE',
      lastSignal: 'HOLD',
      position: 'FLAT',
      nbZone: 'ORANGE',
      mlZone: 'BLUE',
      rValue: 0.5,
      mlTrust: 40,
      nbTrust: 82,
      timestamp: Date.now()
    };
  });
}

// jQuery를 사용한 실시간 촌장 지침 업데이트
function updateRealtimeMayorGuidance() {
  getMayorGuidanceData().then(function(data) {
    console.log('API 응답 데이터:', data);
    
    // 현재 구역 표시 업데이트 (jQuery 사용) - API 응답의 실제 값 사용
    $('#current-zone-display').each(function() {
      const actualNbZone = data.nbZone || 'ORANGE';
      const zoneColor = actualNbZone === 'BLUE' ? '#0ecb81' : '#f6465d';
      const zoneEmoji = actualNbZone === 'BLUE' ? '🔵' : '🟠';
      $(this).html(`<span style="color: ${zoneColor}; font-weight: 600;">${zoneEmoji} ${actualNbZone}</span>`);
    });
    
    // 실시간 동기화 상태 업데이트 (jQuery 사용) - API 응답의 실제 값 사용
    $('#zoneConsistencyInfo').each(function() {
      const nbColor = data.nbZone === 'BLUE' ? '🔵' : '🟠';
      const mlColor = data.mlZone === 'BLUE' ? '🔵' : '🟠';
      $(this).html(`
        <div style="font-size: 9px; color: #333; font-weight: 500; line-height: 1.2; padding: 2px 4px; background: #f8f9fa; border-radius: 3px; border-left: 2px solid #0ecb81;">
          🔄 <span style="color: #0ecb81; font-weight: 600;">실시간 동기화</span> | 
          N/B: ${nbColor}${data.nbZone} | 
          ML: ${mlColor}${data.mlZone}
        </div>
      `);
    });
    
         // 신뢰도 정보 업데이트 (jQuery 사용) - API에서 받은 실제 신뢰도 값 사용
     $('#mayor-trust-display').each(function() {
       // API에서 받은 신뢰도 값 사용
       const mlTrust = data.mlTrust || 40;
       const nbTrust = data.nbTrust || 82;
       
       // 현재 시간과 분봉 정보 계산
       const now = new Date();
       const currentTime = now.toLocaleTimeString('ko-KR', { 
         hour: '2-digit', 
         minute: '2-digit', 
         second: '2-digit',
         hour12: false 
       });
       const currentMinute = now.getMinutes();
       const candleTime = `${currentMinute.toString().padStart(2, '0')}분봉`;
       
       $(this).html(`
         <div style="margin-bottom: 4px;">
           <span style="color: #00d1ff;">🤖 ML Model Trust: </span><span style="color: #00d1ff; font-weight: 600; background: rgba(0,209,255,0.1); padding: 1px 3px; border-radius: 2px;">${mlTrust}%</span>
         </div>
         <div style="margin-bottom: 4px;">
           <span style="color: #ffb703;">🏛️ N/B Guild Trust: </span><span style="color: #ffb703; font-weight: 600; background: rgba(255,183,3,0.1); padding: 1px 3px; border-radius: 2px;">${nbTrust}%</span> (${nbTrust}개 히스토리)
         </div>
         <div style="margin-bottom: 4px;">
           <span style="color: #0ecb81;">⚖️ Trust Balance: </span><span style="color: #0ecb81; font-weight: 600; background: rgba(14,203,129,0.1); padding: 1px 3px; border-radius: 2px;">ML: ${mlTrust}% | N/B: ${nbTrust}%</span>
         </div>
         <div style="margin-bottom: 4px;">
           <span style="color: #f6465d;">📍 N/B Zone Status: </span><span style="color: #f6465d; font-weight: 600; background: rgba(246,70,93,0.1); padding: 1px 3px; border-radius: 2px;">1h ${data.nbZone}</span>
         </div>
         <div style="margin-bottom: 4px;">
           <span style="color: #9c27b0;">⏰ 현재 시간: </span><span style="color: #9c27b0; font-weight: 600; background: rgba(156,39,176,0.1); padding: 1px 3px; border-radius: 2px;">${currentTime}</span>
         </div>
         <div style="margin-bottom: 4px;">
           <span style="color: #ff9800;">📊 분봉 정보: </span><span style="color: #ff9800; font-weight: 600; background: rgba(255,152,0,0.1); padding: 1px 3px; border-radius: 2px;">${candleTime}</span>
         </div>
       `);
     });
    
         // localStorage에 저장 - API 응답의 실제 값 사용
     localStorage.setItem('realtime_mayor_guidance', JSON.stringify({
       current_zone: data.nbZone,
       nb_zone: data.nbZone,
       ml_zone: data.mlZone,
       last_signal: data.lastSignal,
       position: data.position,
       ml_trust: data.mlTrust,
       nb_trust: data.nbTrust,
       r_value: data.rValue,
       timestamp: data.timestamp
     }));
    
    console.log('촌장 지침 실시간 업데이트 완료:', data);
  });
}

// jQuery를 사용한 저장된 실시간 촌장 지침 복원
function restoreRealtimeMayorGuidance() {
  try {
    const savedGuidance = localStorage.getItem('realtime_mayor_guidance');
    if (savedGuidance) {
      const guidance = JSON.parse(savedGuidance);
      const nbZone = guidance.nb_zone || guidance.current_zone || 'ORANGE';
      const mlZone = guidance.ml_zone || 'BLUE';
      
      // 현재 구역 표시 복원 (jQuery 사용)
      $('#current-zone-display').each(function() {
        const zoneColor = nbZone === 'BLUE' ? '#0ecb81' : '#f6465d';
        const zoneEmoji = nbZone === 'BLUE' ? '🔵' : '🟠';
        $(this).html(`<span style="color: ${zoneColor}; font-weight: 600;">${zoneEmoji} ${nbZone}</span>`);
      });
      
      // 실시간 동기화 상태 복원 (jQuery 사용)
      $('#zoneConsistencyInfo').each(function() {
        const nbColor = nbZone === 'BLUE' ? '🔵' : '🟠';
        const mlColor = mlZone === 'BLUE' ? '🔵' : '🟠';
        $(this).html(`
          <div style="font-size: 9px; color: #333; font-weight: 500; line-height: 1.2; padding: 2px 4px; background: #f8f9fa; border-radius: 3px; border-left: 2px solid #0ecb81;">
            🔄 <span style="color: #0ecb81; font-weight: 600;">실시간 동기화</span> | 
            N/B: ${nbColor}${nbZone} | 
            ML: ${mlColor}${mlZone}
          </div>
        `);
      });
      
      // 신뢰도 정보 복원 (jQuery 사용)
      $('#mayor-trust-display').each(function() {
        const mlTrust = guidance.ml_trust || 40;
        const nbTrust = guidance.nb_trust || 82;  // 기본값을 82로 수정
        const nbZone = guidance.nb_zone || 'ORANGE';
        
        // 현재 시간과 분봉 정보 계산
        const now = new Date();
        const currentTime = now.toLocaleTimeString('ko-KR', { 
          hour: '2-digit', 
          minute: '2-digit', 
          second: '2-digit',
          hour12: false 
        });
        const currentMinute = now.getMinutes();
        const candleTime = `${currentMinute.toString().padStart(2, '0')}분봉`;
        
        $(this).html(`
          <div style="margin-bottom: 4px;">
            <span style="color: #00d1ff;">🤖 ML Model Trust: </span><span style="color: #00d1ff; font-weight: 600; background: rgba(0,209,255,0.1); padding: 1px 3px; border-radius: 2px;">${mlTrust}%</span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="color: #ffb703;">🏛️ N/B Guild Trust: </span><span style="color: #ffb703; font-weight: 600; background: rgba(255,183,3,0.1); padding: 1px 3px; border-radius: 2px;">${nbTrust}%</span> (${nbTrust}개 히스토리)
          </div>
          <div style="margin-bottom: 4px;">
            <span style="color: #0ecb81;">⚖️ Trust Balance: </span><span style="color: #0ecb81; font-weight: 600; background: rgba(14,203,129,0.1); padding: 1px 3px; border-radius: 2px;">ML: ${mlTrust}% | N/B: ${nbTrust}%</span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="color: #f6465d;">📍 N/B Zone Status: </span><span style="color: #f6465d; font-weight: 600; background: rgba(246,70,93,0.1); padding: 1px 3px; border-radius: 2px;">1h ${nbZone}</span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="color: #9c27b0;">⏰ 현재 시간: </span><span style="color: #9c27b0; font-weight: 600; background: rgba(156,39,176,0.1); padding: 1px 3px; border-radius: 2px;">${currentTime}</span>
          </div>
          <div style="margin-bottom: 4px;">
            <span style="color: #ff9800;">📊 분봉 정보: </span><span style="color: #ff9800; font-weight: 600; background: rgba(255,152,0,0.1); padding: 1px 3px; border-radius: 2px;">${candleTime}</span>
          </div>
        `);
      });
      
      console.log('촌장 지침 복원 완료:', guidance);
      return true;
    }
  } catch (e) {
    console.error('저장된 실시간 촌장 지침 복원 실패:', e);
  }
  return false;
}

// 실시간 촌장 지침 주기적 업데이트 시작
function startRealtimeMayorGuidanceUpdates() {
  // 기존 타이머가 있다면 제거
  if (window.realtimeMayorGuidanceTimer) {
    clearInterval(window.realtimeMayorGuidanceTimer);
  }
  
  // 5초마다 실시간 촌장 지침 업데이트
  window.realtimeMayorGuidanceTimer = setInterval(() => {
    updateRealtimeMayorGuidance().catch(e => console.error('Error in periodic realtime mayor guidance update:', e));
  }, 5000); // 5초마다 업데이트
  
  console.log('실시간 촌장 지침 주기적 업데이트 시작 (5초 간격)');
}

// 실시간 촌장 지침 주기적 업데이트 중지
function stopRealtimeMayorGuidanceUpdates() {
  if (window.realtimeMayorGuidanceTimer) {
    clearInterval(window.realtimeMayorGuidanceTimer);
    window.realtimeMayorGuidanceTimer = null;
    console.log('실시간 촌장 지침 주기적 업데이트 중지');
  }
}

// jQuery를 사용한 개별 길드 멤버의 촌장 지침 상태 생성
function getMayorGuidanceStatus(member) {
  return getMayorGuidanceData().then(function(data) {
    const currentZone = data.nbZone || 'ORANGE';
    
    // 개인 확신도 (N/B 시스템의 구역 신뢰도)
    const personalConfidence = Math.round(Math.random() * 40 + 60); // 60-100%
    
    // 촌장의 신뢰도 시스템 (API에서 받은 값)
    const mlTrust = data.mlTrust || 40; // ML 모델 신뢰도
    const nbGuildTrust = data.nbTrust || 82; // N/B 길드 신뢰도
    
    // 가중 신뢰도 계산
    const weightedConfidence = (personalConfidence * 0.6) + (mlTrust * 0.2) + (nbGuildTrust * 0.2);
    
    // 현재 포지션 상태 확인
    const hasPosition = member.openPosition || (member.lastTrade && member.lastTrade.type !== 'CLOSE');
    const positionSide = hasPosition ? (member.openPosition ? member.openPosition.side : member.lastTrade.type) : 'NONE';
    
    // 촌장 지침 준수 여부 판단 (Zone-Side Only: BUY@BLUE / SELL@ORANGE)
    let guidanceStatus = '';
    let guidanceColor = '#888888';
    
    if (currentZone === 'ORANGE') {
      if (positionSide === 'SELL') {
        guidanceStatus = '✅ 촌장 지침 준수 (ORANGE에서 SELL)';
        guidanceColor = '#0ecb81';
      } else if (positionSide === 'BUY') {
        guidanceStatus = '❌ 촌장 지침 위반 (ORANGE에서 BUY 금지)';
        guidanceColor = '#f6465d';
      } else if (positionSide === 'HOLD') {
        guidanceStatus = '🛡️ ORANGE 구역 - SELL만 허용';
        guidanceColor = '#ffb703';
      } else {
        guidanceStatus = '🛡️ ORANGE 구역 - SELL만 허용';
        guidanceColor = '#4285f4';
      }
    } else if (currentZone === 'BLUE') {
      if (positionSide === 'BUY') {
        guidanceStatus = '✅ 촌장 지침 준수 (BLUE에서 BUY)';
        guidanceColor = '#0ecb81';
      } else if (positionSide === 'SELL') {
        guidanceStatus = '❌ 촌장 지침 위반 (BLUE에서 SELL 금지)';
        guidanceColor = '#f6465d';
      } else if (positionSide === 'HOLD') {
        guidanceStatus = '⚡ BLUE 구역 - BUY만 허용';
        guidanceColor = '#ffb703';
      } else {
        guidanceStatus = '⚡ BLUE 구역 - BUY만 허용';
        guidanceColor = '#4285f4';
      }
    }
    
    // 신뢰도 정보 (촌장 지침에 맞게 수정)
    // 현재 시간과 분봉 정보 계산
    const now = new Date();
    const currentTime = now.toLocaleTimeString('ko-KR', { 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit',
      hour12: false 
    });
    const currentMinute = now.getMinutes();
    const candleTime = `${currentMinute.toString().padStart(2, '0')}분봉`;
    
         const trustInfo = `
       <div style="margin-bottom: 2px;">
         <span style="color: #00d1ff;">🤖 ML Model Trust: </span><span style="color: #00d1ff; font-weight: 600; background: rgba(0,209,255,0.1); padding: 1px 3px; border-radius: 2px;">${mlTrust}%</span>
       </div>
       <div style="margin-bottom: 2px;">
         <span style="color: #ffb703;">🏛️ N/B Guild Trust: </span><span style="color: #ffb703; font-weight: 600; background: rgba(255,183,3,0.1); padding: 1px 3px; border-radius: 2px;">${nbGuildTrust}%</span> (${nbGuildTrust}개 히스토리)
       </div>
      <div style="margin-bottom: 2px;">
        <span style="color: #0ecb81;">⚖️ Trust Balance: </span><span style="color: #0ecb81; font-weight: 600; background: rgba(14,203,129,0.1); padding: 1px 3px; border-radius: 2px;">ML: ${mlTrust}% | N/B: ${nbGuildTrust}%</span>
      </div>
      <div style="margin-bottom: 2px;">
        <span style="color: #f6465d;">📍 N/B Zone Status: </span><span style="color: #f6465d; font-weight: 600; background: rgba(246,70,93,0.1); padding: 1px 3px; border-radius: 2px;">1h ${currentZone}</span>
      </div>
      <div style="margin-bottom: 2px;">
        <span style="color: #9c27b0;">⏰ 현재 시간: </span><span style="color: #9c27b0; font-weight: 600; background: rgba(156,39,176,0.1); padding: 1px 3px; border-radius: 2px;">${currentTime}</span>
      </div>
      <div style="margin-bottom: 2px;">
        <span style="color: #ff9800;">📊 분봉 정보: </span><span style="color: #ff9800; font-weight: 600; background: rgba(255,152,0,0.1); padding: 1px 3px; border-radius: 2px;">${candleTime}</span>
      </div>
    `;
    
         const guidanceData = {
       guidanceStatus: guidanceStatus,
       guidanceColor: guidanceColor,
       trustInfo: trustInfo,
       currentZone: currentZone,
       mlZone: data.mlZone,
       timestamp: Date.now(),
       memberName: member.name
     };

    // 촌장 지침 상태를 localStorage에 저장
    localStorage.setItem(`mayor_guidance_${member.name}`, JSON.stringify(guidanceData));

         return `
       <div style="color: ${guidanceColor}; font-weight: 600; margin-bottom: 4px;">
         🏛️ ${guidanceStatus}
       </div>
       <div style="color: #888888; font-size: 8px; margin-bottom: 4px;">
         ${trustInfo}
       </div>
       <div style="color: #888888; font-size: 8px; margin-bottom: 2px;">
         🔄 실시간 동기화 | N/B: ${data.nbZone === 'BLUE' ? '🔵' : '🟠'}${data.nbZone} | ML: ${data.mlZone === 'BLUE' ? '🔵' : '🟠'}${data.mlZone}
       </div>
       <div style="color: #888888; font-size: 8px;">
         Zone-Side Only: BUY@BLUE / SELL@ORANGE
       </div>
     `;
  }).fail(function(xhr, status, error) {
    console.error('Error generating mayor guidance status:', error);
    return '<div style="color: #888888;">촌장 지침 상태 확인 중...</div>';
  });
}

// jQuery를 사용한 저장된 촌장 지침 상태 복원
function restoreMayorGuidanceStatus(memberName) {
  try {
    const savedGuidance = localStorage.getItem(`mayor_guidance_${memberName}`);
    if (savedGuidance) {
      const guidance = JSON.parse(savedGuidance);
      
      // jQuery를 사용하여 요소 업데이트
      $(`#mayor-guidance-${memberName}`).each(function() {
                 $(this).html(`
           <div style="color: ${guidance.guidanceColor}; font-weight: 600; margin-bottom: 4px;">
             🏛️ ${guidance.guidanceStatus}
           </div>
           <div style="color: #888888; font-size: 8px; margin-bottom: 4px;">
             ${guidance.trustInfo}
           </div>
           <div style="color: #888888; font-size: 8px; margin-bottom: 2px;">
             🔄 실시간 동기화 | N/B: ${guidance.currentZone === 'BLUE' ? '🔵' : '🟠'}${guidance.currentZone} | ML: ${guidance.mlZone === 'BLUE' ? '🔵' : '🟠'}${guidance.mlZone}
           </div>
           <div style="color: #888888; font-size: 8px;">
             Zone-Side Only: BUY@BLUE / SELL@ORANGE
           </div>
         `);
      });
      
      console.log('촌장 지침 상태 복원 완료:', memberName);
      return true;
    }
  } catch (e) {
    console.error('저장된 촌장 지침 상태 복원 실패:', e);
  }
  return false;
}

// 촌장 지침 학습 모델 훈련
async function trainMayorGuidanceModel() {
  try {
    console.log('🏛️ 촌장 지침 학습 모델 훈련 시작...');
    
    const response = await fetch('/api/ml/train-mayor-guidance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        window: 50,
        ema_fast: 10,
        ema_slow: 30,
        horizon: 5,
        count: 1800,
        interval: 'minute1'
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('🏛️ 촌장 지침 학습 모델 훈련 완료:', result);
      return result;
    } else {
      console.error('🏛️ 촌장 지침 학습 모델 훈련 실패');
      return null;
    }
  } catch (e) {
    console.error('🏛️ 촌장 지침 학습 모델 훈련 오류:', e);
    return null;
  }
}

// AI 트레이딩 설명 가져오기
async function getAIExplanation(memberName) {
  try {
    const response = await fetch(`/api/village/ai-explanation/${memberName}`);
    
    if (response.ok) {
      const result = await response.json();
      const explanationElement = document.getElementById(`ai-explanation-${memberName}`);
      if (explanationElement) {
        explanationElement.innerHTML = `
          <div style="font-size: 9px; color: #888888; padding: 4px; background: rgba(255,255,255,0.05); border-radius: 3px; margin-top: 2px;">
            <div style="color: #00d1ff; font-weight: 600; margin-bottom: 2px;">🤖 AI 트레이딩 설명</div>
            <div style="font-size: 8px; line-height: 1.3;">${result.explanation}</div>
          </div>
        `;
      }
      
      // localStorage에 저장
      localStorage.setItem(`ai_explanation_${memberName}`, JSON.stringify({
        explanation: result.explanation,
        timestamp: Date.now()
      }));
      
      return result;
    } else {
      console.error('AI 트레이딩 설명 가져오기 실패');
    }
  } catch (e) {
    console.error('AI 트레이딩 설명 가져오기 오류:', e);
  }
}

// 저장된 AI 트레이딩 설명 복원
function restoreAIExplanation(memberName) {
  try {
    const savedExplanation = localStorage.getItem(`ai_explanation_${memberName}`);
    if (savedExplanation) {
      const explanation = JSON.parse(savedExplanation);
      const explanationElement = document.getElementById(`ai-explanation-${memberName}`);
      if (explanationElement) {
        explanationElement.innerHTML = `
          <div style="font-size: 9px; color: #888888; padding: 4px; background: rgba(255,255,255,0.05); border-radius: 3px; margin-top: 2px;">
            <div style="color: #00d1ff; font-weight: 600; margin-bottom: 2px;">🤖 AI 트레이딩 설명</div>
            <div style="font-size: 8px; line-height: 1.3;">${explanation.explanation}</div>
          </div>
        `;
      }
      return true;
    }
  } catch (e) {
    console.error('저장된 AI 트레이딩 설명 복원 실패:', e);
  }
  return false;
}

// 자동 학습 상태 업데이트
async function updateAutoLearningStatus(memberName) {
  try {
    const response = await fetch('/api/village/system/overview');
    
    if (response.ok) {
      const result = await response.json();
      const autoLearningStatus = result.auto_learning_enabled ? '활성화' : '비활성화';
      const statusColor = result.auto_learning_enabled ? '#0ecb81' : '#f6465d';
      
      const statusElement = document.getElementById(`auto-learning-status-${memberName}`);
      if (statusElement) {
        statusElement.innerHTML = `
          <span style="color: ${statusColor}; font-weight: 600;">🤖 자동 학습: ${autoLearningStatus}</span>
        `;
      }
      
      // localStorage에 저장
      localStorage.setItem(`auto_learning_${memberName}`, JSON.stringify({
        enabled: result.auto_learning_enabled,
        timestamp: Date.now()
      }));
      
      return result;
    } else {
      console.error('자동 학습 상태 업데이트 실패');
    }
  } catch (e) {
    console.error('자동 학습 상태 업데이트 오류:', e);
  }
}

// 저장된 자동 학습 상태 복원
function restoreAutoLearningStatus(memberName) {
  try {
    const savedStatus = localStorage.getItem(`auto_learning_${memberName}`);
    if (savedStatus) {
      const status = JSON.parse(savedStatus);
      const statusElement = document.getElementById(`auto-learning-status-${memberName}`);
      if (statusElement) {
        const statusText = status.enabled ? '활성화' : '비활성화';
        const statusColor = status.enabled ? '#0ecb81' : '#f6465d';
        statusElement.innerHTML = `
          <span style="color: ${statusColor}; font-weight: 600;">🤖 자동 학습: ${statusText}</span>
        `;
      }
      return true;
    }
  } catch (e) {
    console.error('저장된 자동 학습 상태 복원 실패:', e);
  }
  return false;
}

// 자동 학습 토글
async function toggleAutoLearning() {
  try {
    const response = await fetch('/api/village/auto-learning/toggle', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('자동 학습 토글 결과:', result);
      
      // 모든 길드 멤버의 자동 학습 상태 업데이트
      Object.values(guildMembers).forEach(member => {
        updateAutoLearningStatus(member.name).catch(e => console.error('Error updating auto learning status:', e));
      });
      
      return result;
    } else {
      console.error('자동 학습 토글 실패');
    }
  } catch (e) {
    console.error('자동 학습 토글 오류:', e);
  }
}

// 모든 저장된 상태 복원
function restoreAllSavedStates() {
  // 실시간 촌장 지침 복원
  restoreRealtimeMayorGuidance();
  
  // 각 길드 멤버의 상태 복원
  Object.values(guildMembers).forEach(member => {
    restoreMayorGuidanceStatus(member.name);
    restoreAutoLearningStatus(member.name);
    restoreAIExplanation(member.name);
  });
}

// 모듈 내보내기 (다른 파일에서 사용할 수 있도록)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    updateRealtimeMayorGuidance,
    restoreRealtimeMayorGuidance,
    startRealtimeMayorGuidanceUpdates,
    stopRealtimeMayorGuidanceUpdates,
    getMayorGuidanceStatus,
    restoreMayorGuidanceStatus,
    trainMayorGuidanceModel,
    getAIExplanation,
    restoreAIExplanation,
    updateAutoLearningStatus,
    restoreAutoLearningStatus,
    toggleAutoLearning,
    restoreAllSavedStates
  };
}
