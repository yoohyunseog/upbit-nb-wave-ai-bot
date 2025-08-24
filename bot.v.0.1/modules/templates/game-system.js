// Game System Module - AI Models and Game Logic

// 구역별 의사 결정 헬퍼 함수들
function getCurrentZone(x, y, startX, topY, spacing, config) {
    // 매수 영역 감지
    if (Math.abs(x - startX) < 50 && Math.abs(y - topY) < 50) {
        return '매수영역';
    }
    // 매도 영역 감지
    else if (Math.abs(x - (startX + spacing)) < 50 && Math.abs(y - topY) < 50) {
        return '매도영역';
    }
    // 대기 영역 감지
    else if (Math.abs(x - (startX + spacing * 2)) < 50 && Math.abs(y - topY) < 50) {
        return '대기영역';
    }
    // 신호 대기 센터 감지 (화면 중앙)
    else if (Math.abs(x - (config.width / 2)) < 60 && Math.abs(y - (config.height / 2)) < 60) {
        return '신호대기센터';
    }
    // N/B 길드 감지
    else if (Math.abs(x - 100) < 60 && Math.abs(y - 100) < 60) {
        return 'N/B길드';
    }
    // BTC 시장 감지
    else if (Math.abs(x - (config.width - 100)) < 60 && Math.abs(y - (config.height - 100)) < 60) {
        return 'BTC시장';
    }
    // 기타 영역
    else {
        return '기타영역';
    }
}

function getZoneDecision(zone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
    switch (zone) {
        case '매수영역':
            // 매수 영역에서는 BLUE 신호이고 매수 전 예상 수익률이 계산된 경우
            if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
                const currentProfitRate = getCurrentProfitRate();
                
                // 매수 전 예상 수익률이 현재 수익률보다 낮으면 즉시 매수 (더 유리한 가격)
                if (buyProfitRate < currentProfitRate) {
                    console.log(`📈 매수영역에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) < 현재 수익률(${currentProfitRate.toFixed(2)}%) → 즉시 매수!`);
                    return {
                        action: '매수',
                        targetX: startX,
                        targetY: topY
                    };
                }
                // 매수 전 예상 수익률이 양수이고 현재 수익률보다 높아도 매수 (상승 기대)
                else if (buyProfitRate > 0) {
                    console.log(`📈 매수영역에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) > 0 → 매수 진행`);
                    return {
                        action: '매수',
                        targetX: startX,
                        targetY: topY
                    };
                }
            }
            return null; // 매수 조건이 맞지 않으면 의사 결정 없음
            
        case '매도영역':
            // 매도 영역에서는 N/B 코인이 있고 매도 전 예상 수익률이 있을 때만 매도
            if (nbCoins > 0 && sellProfitRate !== 0) {
                return {
                    action: '매도',
                    targetX: startX + spacing,
                    targetY: topY
                };
            }
            return null; // 매도 조건이 맞지 않으면 의사 결정 없음
            
        case 'N/B길드':
            // N/B 길드에서는 N/B 코인이 있고 매도 전 예상 수익률이 계산되지 않았을 때만 계산
            if (nbCoins > 0 && sellProfitRate === 0) {
                return {
                    action: 'N/B 길드 방문',
                    targetX: 100,
                    targetY: 100
                };
            }
            return null; // 계산 조건이 맞지 않으면 의사 결정 없음
            
        case 'BTC시장':
            // BTC 시장에서는 BLUE 신호일 때 매수 관련 의사결정
            if (currentMajority === 'BLUE') {
                // 매수 전 예상 수익률이 계산되지 않았을 때는 정보 수집 완료로 설정
                if (buyProfitRate === 0) {
                    return {
                        action: '정보 수집 완료',
                        targetX: config.width - 100,
                        targetY: config.height - 100
                    };
                }
                // 매수 전 예상 수익률이 이미 계산된 경우 매수 의사결정
                else if (buyProfitRate !== 0) {
                    const currentProfitRate = getCurrentProfitRate();
                    
                    // BLUE 신호에서는 예상 수익률이 계산되면 매수 (음수여도 매수)
                    if (buyProfitRate < currentProfitRate || buyProfitRate > 0) {
                        console.log(`📈 BTC 시장에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) → 매수 진행`);
                        return {
                            action: '매수',
                            targetX: startX,
                            targetY: topY
                        };
                    }
                }
            }
            return null; // 매수 조건이 맞지 않으면 의사 결정 없음
            
        default:
            return null; // 기타 영역에서는 의사 결정 없음
    }
}

// 현재 수익률 계산 함수
function getCurrentProfitRate() {
    const pnlElement = document.getElementById('selected-coin-pnl');
    if (pnlElement) {
        const pnlText = pnlElement.textContent;
        const pnlMatch = pnlText.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
        if (pnlMatch) {
            return parseFloat(pnlMatch[1]);
        }
    }
    return 0;
}

// 게임 상태 저장 함수
function saveGameState() {
    try {
        const gameState = {
            timestamp: Date.now(),
            aiModels: window.aiModels ? window.aiModels.map(model => ({
                x: model.circle.x,
                y: model.circle.y,
                discoveredCoords: model.discoveredCoins || [],
                isTrainer: model.isTrainer,
                isExplorer: model.isExplorer
            })) : [],
            currentMajority: window.currentMajority || 'BLUE',
            nbCoins: window.nbCoins || 0,
            buyProfitRate: window.buyProfitRate || 0,
            sellProfitRate: window.sellProfitRate || 0
        };
        
        localStorage.setItem('aiGameState', JSON.stringify(gameState));
        console.log('💾 게임 상태 저장 완료');
    } catch (error) {
        console.error('❌ 게임 상태 저장 실패:', error);
    }
}

// 게임 상태 로드 함수
function loadGameState() {
    try {
        const savedState = localStorage.getItem('aiGameState');
        if (savedState) {
            const gameState = JSON.parse(savedState);
            console.log('📂 저장된 게임 상태 로드 완료');
            return gameState;
        }
    } catch (error) {
        console.error('❌ 게임 상태 로드 실패:', error);
    }
    return null;
}

// AI 시스템 알고리즘
function aiSystemAlgorithm() {
    // AI 시스템 로직은 별도로 구현
    console.log('🤖 AI 시스템 알고리즘 실행 중...');
}

// 게임 시스템 초기화
function initializeGameSystem() {
    console.log('🎮 게임 시스템 초기화 완료');
}

// 전역으로 함수들 노출
window.getCurrentZone = getCurrentZone;
window.getZoneDecision = getZoneDecision;
window.getCurrentProfitRate = getCurrentProfitRate;
window.saveGameState = saveGameState;
window.loadGameState = loadGameState;
window.aiSystemAlgorithm = aiSystemAlgorithm;
window.initializeGameSystem = initializeGameSystem;
