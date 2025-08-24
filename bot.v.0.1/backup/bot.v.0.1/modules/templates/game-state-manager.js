// Game State Manager Module
// 게임 상태 저장 및 로드 관리

class GameStateManager {
    constructor() {
        this.storageKey = 'aiTradingGameState';
        this.autoSaveInterval = 5000; // 5초마다 자동 저장
        this.lastSaveTime = 0;
    }

    // 게임 상태 저장
    saveGameState(gameData) {
        try {
            const gameState = {
                nbCoins: gameData.nbCoins || 0,
                nbMinerals: gameData.nbMinerals || 0.0,
                buyPrice: gameData.buyPrice || 0,
                buyProfitRate: gameData.buyProfitRate || 0,
                sellProfitRate: gameData.sellProfitRate || 0,
                lastBuyAction: gameData.lastBuyAction || false,
                lastSellAction: gameData.lastSellAction || false,
                aiModels: gameData.aiModels || [],
                nbCoinItems: gameData.nbCoinItems || [],
                timestamp: Date.now()
            };
            
            localStorage.setItem(this.storageKey, JSON.stringify(gameState));
            console.log('💾 게임 상태 저장 완료:', gameState);
            
            // 로그 매니저에 저장 완료 로그 추가
            if (window.logManager) {
                window.logManager.addLog(`💾 게임 상태 자동 저장 완료 - N/B 코인: ${gameState.nbCoins}개, N/B 미네랄: ${gameState.nbMinerals.toFixed(2)}%`);
            }
            
            return true;
        } catch (error) {
            console.error('❌ 게임 상태 저장 실패:', error);
            return false;
        }
    }

    // 게임 상태 로드
    loadGameState() {
        try {
            const savedState = localStorage.getItem(this.storageKey);
            if (!savedState) {
                console.log('📂 저장된 게임 상태가 없습니다.');
                return null;
            }
            
            const gameState = JSON.parse(savedState);
            console.log('📂 게임 상태 로드 완료:', gameState);
            
            // 로그 매니저에 로드 완료 로그 추가
            if (window.logManager) {
                window.logManager.addLog(`📂 게임 상태 로드 완료 - N/B 코인: ${gameState.nbCoins}개, N/B 미네랄: ${gameState.nbMinerals.toFixed(2)}%`);
            }
            
            return gameState;
        } catch (error) {
            console.error('❌ 게임 상태 로드 실패:', error);
            return null;
        }
    }

    // 게임 상태 삭제
    clearGameState() {
        try {
            localStorage.removeItem(this.storageKey);
            console.log('🗑️ 게임 상태 삭제 완료');
            
            // 로그 매니저에 삭제 완료 로그 추가
            if (window.logManager) {
                window.logManager.addLog('🗑️ 게임 상태 삭제 완료');
            }
            
            return true;
        } catch (error) {
            console.error('❌ 게임 상태 삭제 실패:', error);
            return false;
        }
    }

    // 자동 저장 체크 및 실행
    checkAutoSave(gameData) {
        const currentTime = Date.now();
        if (currentTime - this.lastSaveTime >= this.autoSaveInterval) {
            this.saveGameState(gameData);
            this.lastSaveTime = currentTime;
        }
    }

    // AI 모델 데이터 변환 (저장용)
    convertAiModelsForStorage(aiModels) {
        return aiModels.map(model => ({
            x: model.circle.x,
            y: model.circle.y,
            targetX: model.targetX,
            targetY: model.targetY,
            discoveredCoords: model.discoveredCoords,
            memoryIndex: model.memoryIndex,
            explorationTimer: model.explorationTimer
        }));
    }

    // AI 모델 데이터 복원 (로드용)
    restoreAiModelsFromStorage(savedModels, aiModels) {
        if (savedModels && savedModels.length === aiModels.length) {
            savedModels.forEach((savedModel, index) => {
                const model = aiModels[index];
                if (model) {
                    model.circle.x = savedModel.x;
                    model.circle.y = savedModel.y;
                    model.name.x = savedModel.x;
                    model.name.y = savedModel.y - (index === 4 ? 6 : 4);
                    model.role.x = savedModel.x;
                    model.role.y = savedModel.y + (index === 4 ? 6 : 4);
                    model.targetX = savedModel.targetX;
                    model.targetY = savedModel.targetY;
                    model.discoveredCoords = savedModel.discoveredCoords || [];
                    model.memoryIndex = savedModel.memoryIndex || 0;
                    model.explorationTimer = savedModel.explorationTimer || 0;
                }
            });
            console.log('🤖 AI 모델 데이터 복원 완료');
        }
    }

    // N/B 코인 아이템 데이터 변환 (저장용)
    convertNBCoinItemsForStorage(nbCoinItems) {
        return nbCoinItems.map(item => ({
            x: item.polygon.x,
            y: item.polygon.y,
            collected: item.collected
        }));
    }

    // N/B 코인 아이템 데이터 복원 (로드용)
    restoreNBCoinItemsFromStorage(savedItems, scene, nbCoinItems) {
        if (savedItems) {
            // 기존 아이템들 제거
            nbCoinItems.forEach(item => {
                if (item.polygon && item.polygon.destroy) {
                    item.polygon.destroy();
                }
                item.connectionLines.forEach(line => {
                    if (line && line.destroy) {
                        line.destroy();
                    }
                });
            });
            nbCoinItems.length = 0;
            
            // 저장된 아이템들 복원
            savedItems.forEach(savedItem => {
                if (!savedItem.collected) {
                    const item = {
                        polygon: scene.add.polygon(savedItem.x, savedItem.y, [0, -8, 6, -4, 6, 4, 0, 8, -6, 4, -6, -4], 0xffaa00),
                        collected: false,
                        connectionLines: []
                    };
                    item.polygon.setOrigin(0.5, 0.5);
                    
                    // 회전 애니메이션
                    scene.tweens.add({
                        targets: item.polygon,
                        rotation: Math.PI * 2,
                        duration: 3000,
                        repeat: -1,
                        ease: 'Linear'
                    });
                    
                    nbCoinItems.push(item);
                }
            });
            console.log('🪙 N/B 코인 아이템 데이터 복원 완료');
        }
    }

    // 게임 상태 초기화
    resetGameState() {
        const defaultState = {
            nbCoins: 0,
            nbMinerals: 0.0,
            buyPrice: 0,
            buyProfitRate: 0,
            sellProfitRate: 0,
            lastBuyAction: false,
            lastSellAction: false,
            aiModels: [],
            nbCoinItems: [],
            timestamp: Date.now()
        };
        
        this.saveGameState(defaultState);
        console.log('🔄 게임 상태 초기화 완료');
        
        // 로그 매니저에 초기화 완료 로그 추가
        if (window.logManager) {
            window.logManager.addLog('🔄 게임 상태 초기화 완료');
        }
        
        return defaultState;
    }

    // 저장된 게임 상태 정보 가져오기
    getGameStateInfo() {
        const savedState = this.loadGameState();
        if (savedState) {
            return {
                nbCoins: savedState.nbCoins,
                nbMinerals: savedState.nbMinerals,
                lastSaveTime: new Date(savedState.timestamp).toLocaleString(),
                hasData: true
            };
        }
        return {
            nbCoins: 0,
            nbMinerals: 0.0,
            lastSaveTime: '없음',
            hasData: false
        };
    }

    // 저장 간격 설정
    setAutoSaveInterval(interval) {
        this.autoSaveInterval = interval;
        console.log(`⚙️ 자동 저장 간격 설정: ${interval}ms`);
    }

    // 저장 키 변경
    setStorageKey(key) {
        this.storageKey = key;
        console.log(`🔑 저장 키 변경: ${key}`);
    }
}

// 전역 인스턴스 생성
window.gameStateManager = new GameStateManager();
