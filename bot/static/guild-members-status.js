// ========================================
// 길드 멤버 카드 시스템 (Guild Members Card System)
// ========================================

// Guild Members Card System 업데이트 메인 함수
async function updateGuildMembersStatus() {
  try {
    console.log('🃏 Updating Guild Members Card System...');
    
    const guildContainer = document.getElementById('integratedGuildStatus');
    if (!guildContainer) {
      console.error('❌ integratedGuildStatus not found in updateGuildMembersStatus');
      return;
    }

    // 카드 시스템 상태 가져오기
    const cardSystemStatus = await fetchCardSystemStatus();
    if (!cardSystemStatus) {
      console.error('❌ Failed to fetch card system status');
      return;
    }

    // Clear existing content
    guildContainer.innerHTML = '';

    // Create header
    const headerDiv = document.createElement('div');
    headerDiv.style.cssText = 'font-size: 11px; color: #d9e2f3; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1);';
    headerDiv.textContent = 'Guild Members Card System';
    guildContainer.appendChild(headerDiv);

    // Create member cards
    Object.values(cardSystemStatus.members).forEach(member => {
      const memberDiv = createMemberCard(member);
      guildContainer.appendChild(memberDiv);
    });

    // 전체 카드 시스템 통계 표시
    const statsDiv = createCardSystemStats(cardSystemStatus);
    guildContainer.appendChild(statsDiv);

  } catch (e) {
    console.error('Error updating guild members card system:', e);
  }
}

// 카드 시스템 상태 가져오기
async function fetchCardSystemStatus() {
  try {
    const response = await fetch('/api/village/card-system/status');
    if (response.ok) {
      return await response.json();
    } else {
      console.error('Failed to fetch card system status:', response.status);
      return null;
    }
  } catch (e) {
    console.error('Error fetching card system status:', e);
    return null;
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
  const cardStatus = generateCardStatusHTML(member);
  const cardActions = generateCardActionsHTML(member);

  memberDiv.innerHTML = `
    ${memberInfo}
    ${cardStatus}
    ${cardActions}
  `;

  return memberDiv;
}

// 멤버 기본 정보 HTML 생성
function generateMemberInfoHTML(member) {
  const successRate = (member.analysisSuccessRate * 100).toFixed(1);
  const successColor = member.analysisSuccessRate > 0.7 ? '#0ecb81' : member.analysisSuccessRate > 0.5 ? '#ffb703' : '#f6465d';
  
  return `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-weight: 600; color: #ffffff; font-size: 12px;">${member.memberName}</span>
        <span style="font-size: 10px; color: #888888;">(${member.role})</span>
        <span style="font-size: 9px; color: #ffb703; background: rgba(255,183,3,0.1); padding: 2px 6px; border-radius: 10px;">
          카드 ${member.totalCardsAnalyzed}개
        </span>
      </div>
      <div style="font-size: 10px; color: ${successColor};">
        성공률: ${successRate}%
      </div>
    </div>
    
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
      <div style="font-size: 10px; color: #888888;">
        담당 분봉: ${member.assignedTimeframes.join(', ')}
      </div>
      <div style="font-size: 10px; color: #888888;">
        평균 수익: ${member.averageProfit > 0 ? '+' : ''}${member.averageProfit.toFixed(2)}%
      </div>
    </div>
  `;
}

// 카드 상태 HTML 생성
function generateCardStatusHTML(member) {
  const activeCards = member.activeCards;
  const completedCards = member.completedCards;
  const failedCards = member.failedCards;
  const currentAnalysis = member.currentAnalysis;
  
  return `
    <div style="margin-bottom: 8px;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 10px; color: #00d1ff;">활성 카드: ${activeCards}개</span>
        <span style="font-size: 10px; color: #0ecb81;">완료: ${completedCards}개</span>
        <span style="font-size: 10px; color: #f6465d;">실패: ${failedCards}개</span>
      </div>
      
      ${currentAnalysis ? `
        <div style="font-size: 9px; color: #ffb703; background: rgba(255,183,3,0.1); padding: 4px; border-radius: 4px; text-align: center;">
          🔍 분석 중: 카드 #${currentAnalysis}
        </div>
      ` : `
        <div style="font-size: 9px; color: #888888; background: rgba(255,255,255,0.05); padding: 4px; border-radius: 4px; text-align: center;">
          대기 중
        </div>
      `}
    </div>
  `;
}

// 카드 액션 HTML 생성
function generateCardActionsHTML(member) {
  return `
    <div style="margin-top: 8px; display: flex; gap: 4px;">
      <button class="btn btn-sm btn-outline-primary" onclick="createNewCard('${member.memberName}')" style="font-size: 8px; padding: 2px 4px;">
        새 카드 생성
      </button>
      <button class="btn btn-sm btn-outline-success" onclick="viewCardHistory('${member.memberName}')" style="font-size: 8px; padding: 2px 4px;">
        카드 히스토리
      </button>
      <button class="btn btn-sm btn-outline-info" onclick="viewCardStats('${member.memberName}')" style="font-size: 8px; padding: 2px 4px;">
        통계 보기
      </button>
    </div>
  `;
}

// 카드 시스템 통계 생성
function createCardSystemStats(cardSystemStatus) {
  const totalActive = cardSystemStatus.activeCards;
  const totalCompleted = cardSystemStatus.completedCards;
  const totalFailed = cardSystemStatus.failedCards;
  const totalCards = cardSystemStatus.totalCards;
  
  const successRate = totalCompleted > 0 ? ((totalCompleted - totalFailed) / totalCompleted * 100).toFixed(1) : 0;
  
  return `
    <div style="background: rgba(0,209,255,0.1); border: 1px solid rgba(0,209,255,0.3); border-radius: 8px; padding: 12px; margin-top: 12px;">
      <div style="font-size: 11px; color: #00d1ff; margin-bottom: 8px; text-align: center;">
        📊 카드 시스템 통계
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 9px; color: #888888;">총 카드:</span>
        <span style="font-size: 9px; color: #ffffff;">${totalCards}개</span>
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 9px; color: #888888;">활성 카드:</span>
        <span style="font-size: 9px; color: #00d1ff;">${totalActive}개</span>
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 9px; color: #888888;">완료된 카드:</span>
        <span style="font-size: 9px; color: #0ecb81;">${totalCompleted}개</span>
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 9px; color: #888888;">실패한 카드:</span>
        <span style="font-size: 9px; color: #f6465d;">${totalFailed}개</span>
      </div>
      
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 9px; color: #888888;">전체 성공률:</span>
        <span style="font-size: 9px; color: #ffb703;">${successRate}%</span>
      </div>
    </div>
  `;
}

// 새 카드 생성 함수
async function createNewCard(memberName) {
  try {
    console.log(`🃏 Creating new card for ${memberName}...`);
    
    // 현재 선택된 분봉 가져오기
    const currentTimeframe = document.getElementById('timeframe')?.value || 'minute1';
    
    const response = await fetch('/api/village/card-system/create', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        member_name: memberName,
        timeframe: currentTimeframe,
        pattern_data: {
          timestamp: Date.now(),
          timeframe: currentTimeframe
        }
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`✅ New card created: ${result.card_id}`);
      
      // 카드 분석 시작
      await analyzeCard(result.card_id, memberName);
      
      // UI 업데이트
      updateGuildMembersStatus();
    } else {
      console.error('Failed to create card:', response.status);
    }
  } catch (e) {
    console.error('Error creating new card:', e);
  }
}

// 카드 분석 함수
async function analyzeCard(cardId, memberName) {
  try {
    console.log(`🔍 Analyzing card ${cardId} for ${memberName}...`);
    
    const response = await fetch(`/api/village/card-system/analyze/${cardId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        member_name: memberName
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`✅ Card analysis completed: ${result.strategy.buyCondition}`);
    } else {
      console.error('Failed to analyze card:', response.status);
    }
  } catch (e) {
    console.error('Error analyzing card:', e);
  }
}

// 카드 히스토리 보기 함수
function viewCardHistory(memberName) {
  console.log(`📚 Viewing card history for ${memberName}...`);
  // TODO: 카드 히스토리 모달 또는 페이지 구현
  alert(`${memberName}의 카드 히스토리를 보여줍니다.`);
}

// 카드 통계 보기 함수
function viewCardStats(memberName) {
  console.log(`📊 Viewing card stats for ${memberName}...`);
  // TODO: 카드 통계 모달 또는 페이지 구현
  alert(`${memberName}의 카드 통계를 보여줍니다.`);
}

// Get Guild Members Status for specific interval (카드 시스템 기반)
async function getGuildMembersStatusForInterval(interval) {
  try {
    // 카드 시스템 상태를 기반으로 길드 상태 계산
    const cardSystemStatus = await fetchCardSystemStatus();
    if (!cardSystemStatus || !cardSystemStatus.members) {
      return {
        nbEnergy: 50,
        nbEnergyColor: '#ffb703',
        activeMembers: 0,
        treasuryAccess: false
      };
    }
    
    // 활성 카드 수에 따른 에너지 계산
    const totalActiveCards = cardSystemStatus.activeCards || 0;
    const totalCompletedCards = cardSystemStatus.completedCards || 0;
    const totalFailedCards = cardSystemStatus.failedCards || 0;
    
    // 카드 성과에 따른 에너지 계산
    let nbEnergyPercent = 50; // 기본값
    if (totalCompletedCards > 0) {
      const successRate = (totalCompletedCards - totalFailedCards) / totalCompletedCards;
      nbEnergyPercent = Math.round(successRate * 100);
    }
    
    // 활성 멤버 수 계산 (활성 카드가 있는 멤버)
    const activeMembers = Object.values(cardSystemStatus.members).filter(member => member && member.activeCards > 0).length;
    
    // Determine N/B Energy color
    let nbEnergyColor = '#f6465d'; // red
    if (nbEnergyPercent > 70) {
      nbEnergyColor = '#4285f4'; // blue
    } else if (nbEnergyPercent > 40) {
      nbEnergyColor = '#ffb703'; // yellow
    }
    
    return {
      nbEnergy: nbEnergyPercent,
      nbEnergyColor: nbEnergyColor,
      activeMembers: activeMembers,
      treasuryAccess: nbEnergyPercent >= 80
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

// Initialize guild members card system
function initializeGuildMembersStatusSystem() {
  console.log('🃏 Guild Members Card System 초기화 시작...');
  
  // Check if required elements exist
  const guildContainer = document.getElementById('integratedGuildStatus');
  if (!guildContainer) {
    console.error('❌ integratedGuildStatus element not found');
    return;
  }
  console.log('✅ integratedGuildStatus found');
  
  // Set up periodic updates
  setInterval(() => {
    updateGuildMembersStatus().catch(e => console.error('Error updating guild members card system:', e));
  }, 5 * 1000); // Every 5 seconds

  // Initial update
  setTimeout(() => {
    console.log('🔄 Initial guild members card system update...');
    updateGuildMembersStatus().catch(e => console.error('Error updating guild members card system:', e));
  }, 2000); // 2초 후 초기 업데이트

  // Expose functions globally
  window.updateGuildMembersStatus = updateGuildMembersStatus;
  window.createMemberCard = createMemberCard;
  window.generateMemberInfoHTML = generateMemberInfoHTML;
  window.generateCardStatusHTML = generateCardStatusHTML;
  window.generateCardActionsHTML = generateCardActionsHTML;
  window.createCardSystemStats = createCardSystemStats;
  window.getGuildMembersStatusForInterval = getGuildMembersStatusForInterval;
  window.createNewCard = createNewCard;
  window.analyzeCard = analyzeCard;
  window.viewCardHistory = viewCardHistory;
  window.viewCardStats = viewCardStats;
  
  console.log('✅ Guild Members Card System initialized successfully');
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    updateGuildMembersStatus,
    createMemberCard,
    generateMemberInfoHTML,
    generateCardStatusHTML,
    generateCardActionsHTML,
    createCardSystemStats,
    getGuildMembersStatusForInterval,
    createNewCard,
    analyzeCard,
    viewCardHistory,
    viewCardStats,
    initializeGuildMembersStatusSystem
  };
}
