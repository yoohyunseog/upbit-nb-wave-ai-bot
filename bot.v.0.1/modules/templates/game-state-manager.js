// 게임 상태 관리자 모듈
// 게임의 모든 상태를 자동 저장하고 복원하는 기능을 담당

class GameStateManager {
    constructor() {
        this.saveInterval = 5000; // 5초마다 자동 저장
        this.storageKey = 'hankookin_game_state';
        this.maxSaveHistory = 10; // 최대 저장 히스토리 개수
        this.isInitialized = false;
        this.isRestoring = false; // 복원 중인지 확인하는 플래그
        
        // 자동 저장 타이머
        this.autoSaveTimer = null;
        
        //console.log('🎮 Game State Manager: 초기화 완료');
        
        // 페이지 로드 시 자동 복원 설정
        this.setupAutoRestore();
    }

    // 페이지 로드 시 자동 복원 설정
    setupAutoRestore() {
        // DOMContentLoaded 이벤트 리스너 추가
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.autoRestoreOnPageLoad();
            });
        } else {
            // 이미 로드된 경우 즉시 실행
            this.autoRestoreOnPageLoad();
        }
        
        // beforeunload 이벤트에서 상태 저장
        window.addEventListener('beforeunload', () => {
            this.saveGameStateOnUnload();
        });
    }

    // 페이지 로드 시 자동 복원
    autoRestoreOnPageLoad() {
        //console.log('🔄 페이지 로드 시 자동 복원 시작...');
        
        // 약간의 지연 후 복원 실행 (모든 모듈이 로드될 시간 확보)
        setTimeout(() => {
            this.restoreGameStateOnRefresh();
        }, 1000);
    }

    // 새로고침 시 게임 상태 복원 (N/B 길드에서 새로 시작)
    restoreGameStateOnRefresh() {
        if (this.isRestoring) {
            //console.log('🔄 이미 복원 중입니다...');
            return;
        }
        
        this.isRestoring = true;
        //console.log('🔄 새로고침 감지 - N/B 길드에서 새로 시작...');
        
        try {
            const savedState = this.loadGameState();
            if (savedState) {
                // 게임 데이터 복원 (N/B 코인, 미네랄 등)
                this.restoreGameData(savedState.gameData);
                
                // UI 상태 복원
                this.restoreUIState(savedState.uiElements);
                
                // 트레이너 상태 복원
                this.restoreTrainerState(savedState.trainerState);
                
                // AI 모델들은 N/B 길드에서 새로 생성 (저장된 데이터 무시)
                this.restoreAiModels(savedState.aiModels);
                
                // 모든 시스템 재시작
                this.restartAllSystems();
                
                if (window.logManager) {
                    window.logManager.addLog(`✅ 새로고침 후 N/B 길드에서 새로 시작 완료 - 모든 시스템 재시작됨`);
                }
                
                //console.log('✅ 새로고침 후 N/B 길드에서 새로 시작 완료');
            } else {
                //console.log('📂 저장된 게임 상태가 없어 N/B 길드에서 새 게임을 시작합니다.');
                this.initializeNewGame();
            }
        } catch (error) {
            console.error('❌ 새로고침 후 N/B 길드 시작 실패:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 새로고침 후 N/B 길드 시작 실패: ${error.message}`);
            }
            this.initializeNewGame();
        } finally {
            this.isRestoring = false;
        }
    }

    // 게임 데이터 복원
    restoreGameData(gameData) {
        if (!gameData) return;
        
        // 전역 게임 데이터 복원 (중앙 gameData 우선)
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData = { ...window.gameInitializer.gameData, ...gameData };
        }
        
        // 개별 변수들 복원 (하위 호환성)
        if (window.nbCoins !== undefined) window.nbCoins = gameData.nbCoins || 0;
        if (window.nbMinerals !== undefined) window.nbMinerals = gameData.nbMinerals || 0.0;
        if (window.buyPrice !== undefined) window.buyPrice = gameData.buyPrice || 0;
        if (window.buyProfitRate !== undefined) window.buyProfitRate = gameData.buyProfitRate || 0;
        if (window.sellProfitRate !== undefined) window.sellProfitRate = gameData.sellProfitRate || 0;
        if (window.lastBuyAction !== undefined) window.lastBuyAction = gameData.lastBuyAction || false;
        if (window.lastSellAction !== undefined) window.lastSellAction = gameData.lastSellAction || false;
        
        if (window.logManager) {
            window.logManager.addLog(`📊 게임 데이터 복원: N/B코인 ${gameData.nbCoins || 0}개, 미네랄 ${(gameData.nbMinerals || 0).toFixed(2)}%, 드랍아이템 ${gameData.dropItemsCount || 0}개`);
        }
        
        //console.log('📊 게임 데이터 복원 완료:', gameData);
    }

    // AI 모델들 복원 및 움직임 재시작 (N/B 길드에서 새로 시작)
    restoreAiModels(savedModels) {
        // 저장된 모델 데이터는 무시하고 N/B 길드에서 새로 시작
        //console.log('🔄 N/B 길드에서 새로운 탐색원들과 트레이너 생성 중...');
        
        // 게임 씬이 있는지 확인
        if (!window.gameInitializer || !window.gameInitializer.game) {
            //console.log('🎮 게임 씬이 아직 초기화되지 않았습니다. 나중에 복원을 시도합니다.');
            setTimeout(() => this.restoreAiModels(savedModels), 2000);
            return;
        }
        
        const scene = window.gameInitializer.game.scene.getScene('GameScene');
        if (!scene) {
            //console.log('🎮 게임 씬을 찾을 수 없습니다.');
            return;
        }
        
        // 기존 AI 모델들 완전 제거
        this.clearAllAiModels();
        
        // N/B 길드에서 새로운 AI 모델들 생성
        this.initializeDefaultAiModels();
        
        // 각 모델의 움직임 재시작
        if (window.aiModels && Array.isArray(window.aiModels)) {
            window.aiModels.forEach((model, index) => {
                if (model.isExplorer) {
                    // 탐색자 움직임 재시작
                    this.restartExplorerMovement(model, index);
                } else if (model.isTrainer) {
                    // 트레이너 움직임 재시작
                    this.restartTrainerMovement(model);
                }
            });
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🤖 N/B 길드에서 새로운 AI 모델들 생성 완료 - 4개 탐색원, 1개 트레이너`);
        }
        
        //console.log(`🤖 N/B 길드에서 새로운 AI 모델들 생성 완료 - 4개 탐색원, 1개 트레이너`);
    }

    // 탐색자 움직임 재시작
    restartExplorerMovement(model, index) {
        if (!model || !model.circle) return;
        
        // 탐색 시스템 재시작
        if (window.explorerMovementSystem) {
            window.explorerMovementSystem.restartExplorer(model, index);
        }
        
        // 의사결정 시스템 재시작
        if (window.explorerDecisionSystem) {
            window.explorerDecisionSystem.restartExplorerDecision(model);
        }
        
        //console.log(`🔍 탐색자 ${index + 1} 움직임 재시작 완료`);
    }

    // 트레이너 움직임 재시작
    restartTrainerMovement(model) {
        if (!model || !model.circle) return;
        
        // 트레이너 시스템 재시작
        if (window.trainerSystemMain) {
            window.trainerSystemMain.restart();
        }
        
        // 트레이너 의사결정 시스템 재시작
        if (window.trainerDecisionHandler) {
            window.trainerDecisionHandler.restart();
        }
        
        // 트레이너 이동 컨트롤러 재시작
        if (window.trainerMovementController) {
            window.trainerMovementController.restart();
        }
        
        //console.log(`🎯 트레이너 움직임 재시작 완료`);
    }

    // 모든 시스템 재시작
    restartAllSystems() {
        // 탐색 시스템 재시작
        if (window.explorerMovementSystem) {
            window.explorerMovementSystem.restart();
        }
        
        // 주민 수집 시스템 재시작
        if (window.residentCollectionSystem) {
            window.residentCollectionSystem.restart();
        }
        
        // 트레이너 시스템 재시작
        if (window.trainerSystemMain) {
            window.trainerSystemMain.restart();
        }
        
        // 게임 루프 재시작
        this.restartGameLoop();
        
        //console.log('🔄 모든 시스템 재시작 완료');
    }

    // 게임 루프 재시작
    restartGameLoop() {
        // 기존 게임 루프가 있다면 중지
        if (window.gameLoopInterval) {
            clearInterval(window.gameLoopInterval);
        }
        
        // 새로운 게임 루프 시작
        window.gameLoopInterval = setInterval(() => {
            this.updateGameLoop();
        }, 100); // 100ms마다 업데이트
        
        //console.log('🔄 게임 루프 재시작 완료');
    }

    // 게임 루프 업데이트
    updateGameLoop() {
        // AI 모델들 업데이트
        if (window.aiModels && Array.isArray(window.aiModels)) {
            window.aiModels.forEach((model, index) => {
                if (model.isExplorer) {
                    // 탐색자 업데이트
                    if (window.explorerMovementSystem) {
                        window.explorerMovementSystem.updateExplorer(model, index);
                    }
                } else if (model.isTrainer) {
                    // 트레이너 업데이트
                    if (window.trainerMovementController) {
                        window.trainerMovementController.updateTrainerMovement(model, window.gameInitializer?.game?.config);
                    }
                }
            });
        }
    }

    // 새 게임 초기화 (N/B 길드에서 새로 시작)
    initializeNewGame() {
        //console.log('🎮 N/B 길드에서 새 게임 초기화 시작...');
        
        // 기본값으로 초기화
        if (window.gameInitializer) {
            window.gameInitializer.resetGame();
        }
        
        // N/B 길드에서 AI 모델들 기본 위치로 초기화
        this.initializeDefaultAiModels();
        
        //console.log('🎮 N/B 길드에서 새 게임 초기화 완료');
    }

    // 기본 AI 모델들 초기화 (자동 생성 비활성화)
    initializeDefaultAiModels() {
        if (!window.gameInitializer || !window.gameInitializer.game) {
            setTimeout(() => this.initializeDefaultAiModels(), 1000);
            return;
        }
        
        const scene = window.gameInitializer.game.scene.getScene('GameScene');
        if (!scene) return;
        
        // 기존 AI 모델들 완전 제거
        this.clearAllAiModels();
        
        // AI 모델 자동 생성 비활성화 - 빈 배열로 초기화
        window.aiModels = [];
        
        if (window.logManager) {
            window.logManager.addLog(`🤖 AI 모델 자동 생성이 비활성화되었습니다. 수동으로 생성해주세요.`);
        }
        
        //console.log('🤖 AI 모델 자동 생성이 비활성화되었습니다. 수동으로 생성해주세요.');
    }

    // AI 모델 수동 생성 (기존 자동 생성 로직)
    createAiModelsManually() {
        if (!window.gameInitializer || !window.gameInitializer.game) {
            setTimeout(() => this.createAiModelsManually(), 1000);
            return;
        }
        
        const scene = window.gameInitializer.game.scene.getScene('GameScene');
        if (!scene) return;
        
        // 기존 AI 모델들 완전 제거
        this.clearAllAiModels();
        
        const config = window.gameInitializer.game.config;
        const modelColors = [0xff8800, 0x00ff88, 0x8800ff, 0xff0088, 0xffff00];
        const modelNames = ['Explorer-1', 'Explorer-2', 'Explorer-3', 'Explorer-4', 'Trainer'];
        const modelRoles = ['탐색', '탐색', '탐색', '탐색', '트레이너'];
        
        window.aiModels = [];
        
        // 4개의 탐색원과 1개의 트레이너 생성 (총 5개)
        for (let i = 0; i < 5; i++) {
            const circleRadius = i === 4 ? 20 : 10;
            const fontSize = i === 4 ? '8px' : '6px';
            const roleFontSize = i === 4 ? '6px' : '5px';
            
            // 탐색원들을 화면의 다른 구역에 분산 배치
            let defaultX, defaultY;
            
            if (i < 4) {
                // 탐색원들 - 4개 구역에 배치
                switch (i) {
                    case 0: // 좌측 상단
                        defaultX = config.width * 0.25;
                        defaultY = config.height * 0.25;
                        break;
                    case 1: // 우측 상단
                        defaultX = config.width * 0.75;
                        defaultY = config.height * 0.25;
                        break;
                    case 2: // 좌측 하단
                        defaultX = config.width * 0.25;
                        defaultY = config.height * 0.75;
                        break;
                    case 3: // 우측 하단
                        defaultX = config.width * 0.75;
                        defaultY = config.height * 0.75;
                        break;
                }
            } else {
                // 트레이너 - 중앙에 배치
                defaultX = config.width / 2;
                defaultY = config.height / 2;
            }
            
            // 화면 경계 내로 제한
            const margin = 50;
            defaultX = Math.max(margin, Math.min(config.width - margin, defaultX));
            defaultY = Math.max(margin, Math.min(config.height - margin, defaultY));
            
            const model = {
                circle: scene.add.circle(defaultX, defaultY, circleRadius, modelColors[i]),
                name: scene.add.text(defaultX, defaultY - (i === 4 ? 6 : 4), modelNames[i], {
                    fontSize: fontSize,
                    fill: '#ffffff',
                    fontStyle: 'bold'
                }).setOrigin(0.5),
                role: scene.add.text(defaultX, defaultY + (i === 4 ? 6 : 4), modelRoles[i], {
                    fontSize: roleFontSize,
                    fill: '#ffffff'
                }).setOrigin(0.5),
                targetX: defaultX,
                targetY: defaultY,
                targetAction: i === 4 ? '신호 대기' : '',
                isExplorer: i < 4,
                isTrainer: i === 4,
                discoveredCoords: [],
                memoryIndex: 0,
                explorationTimer: 0,
                arrivalLogged: false,
                needsNewDecision: false
            };
            
            window.aiModels.push(model);
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🤖 AI 모델 수동 생성 완료 - 4개 탐색원, 1개 트레이너`);
        }
        
        //console.log('🤖 AI 모델 수동 생성 완료 - 4개 탐색원, 1개 트레이너');
    }

    // 모든 AI 모델들 제거
    clearAllAiModels() {
        if (window.aiModels && Array.isArray(window.aiModels)) {
            window.aiModels.forEach(model => {
                if (model.circle) {
                    model.circle.destroy();
                }
                if (model.name) {
                    model.name.destroy();
                }
                if (model.role) {
                    model.role.destroy();
                }
            });
            window.aiModels = [];
            
            if (window.logManager) {
                window.logManager.addLog(`🗑️ 기존 AI 모델들 모두 제거 완료`);
            }
            
            //console.log('🗑️ 기존 AI 모델들 모두 제거 완료');
        }
    }

    // 페이지 언로드 시 상태 저장
    saveGameStateOnUnload() {
        try {
            // 중앙 gameData에서 데이터 가져오기
            const gameData = {
                nbCoins: window.gameInitializer?.gameData?.nbCoins || 0,
                nbMinerals: window.gameInitializer?.gameData?.nbMinerals || 0.0,
                dropItemsCount: window.gameInitializer?.gameData?.dropItemsCount || 0,
                dropItemsCollected: window.gameInitializer?.gameData?.dropItemsCollected || 0,
                buyPrice: window.gameInitializer?.gameData?.buyPrice || 0,
                buyProfitRate: window.gameInitializer?.gameData?.buyProfitRate || 0,
                sellProfitRate: window.gameInitializer?.gameData?.sellProfitRate || 0,
                lastBuyAction: window.gameInitializer?.gameData?.lastBuyAction || false,
                lastSellAction: window.gameInitializer?.gameData?.lastSellAction || false
            };
            
            this.saveGameState(gameData);
        } catch (error) {
            console.error('❌ 페이지 언로드 시 상태 저장 실패:', error);
        }
    }

    // 게임 상태 저장
    saveGameState(gameData) {
        try {
            // 매수/매도 액션이 있을 때만 수익률 저장
            let buyProfitRate = 0;
            let sellProfitRate = 0;
            
            // 매수 액션이 있을 때만 매수 수익률 저장
            if (gameData.lastBuyAction && gameData.buyProfitRate !== undefined) {
                buyProfitRate = gameData.buyProfitRate;
            }
            
            // 매도 액션이 있을 때만 매도 수익률 저장
            if (gameData.lastSellAction && gameData.sellProfitRate !== undefined) {
                sellProfitRate = gameData.sellProfitRate;
            }
            
            const state = {
                timestamp: Date.now(),
                gameData: {
                    nbCoins: gameData.nbCoins || 0,
                    nbMinerals: gameData.nbMinerals || 0.0,
                    dropItemsCount: gameData.dropItemsCount || 0,
                    dropItemsCollected: gameData.dropItemsCollected || 0,
                    buyPrice: gameData.buyPrice || 0,
                    buyProfitRate: buyProfitRate,
                    sellProfitRate: sellProfitRate,
                    lastBuyAction: gameData.lastBuyAction || false,
                    lastSellAction: gameData.lastSellAction || false
                },
                aiModels: this.convertAiModelsForStorage(gameData.aiModels || []),
                uiElements: this.captureUIState(),
                trainerState: this.captureTrainerState(),
                timestamp: Date.now()
            };

            // 로컬 스토리지에 저장
            localStorage.setItem(this.storageKey, JSON.stringify(state));
            
            // 저장 히스토리 관리
            this.manageSaveHistory();
            
            if (window.logManager) {
                const actionType = gameData.lastBuyAction ? '매수' : gameData.lastSellAction ? '매도' : '상태';
                const profitInfo = gameData.lastBuyAction ? `매수수익률: ${buyProfitRate.toFixed(2)}%` : 
                                 gameData.lastSellAction ? `매도수익률: ${sellProfitRate.toFixed(2)}%` : '수익률 없음';
                window.logManager.addLog(`💾 게임 상태 ${actionType} 저장 완료 (${new Date().toLocaleTimeString()}) - ${profitInfo}`);
            }
            
            //console.log('💾 Game State Manager: 게임 상태 저장 완료');
            return true;
        } catch (error) {
            console.error('❌ Game State Manager: 저장 오류:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 게임 상태 저장 실패: ${error.message}`);
            }
            return false;
        }
    }

    // 게임 상태 복원
    loadGameState() {
        try {
            const savedState = localStorage.getItem(this.storageKey);
            if (!savedState) {
                //console.log('💾 Game State Manager: 저장된 게임 상태 없음');
                return null;
            }

            const state = JSON.parse(savedState);
            
            // 저장된 시간 확인 (24시간 이내)
            const savedTime = new Date(state.timestamp);
            const currentTime = new Date();
            const hoursDiff = (currentTime - savedTime) / (1000 * 60 * 60);
            
            if (hoursDiff > 24) {
                //console.log('💾 Game State Manager: 저장된 상태가 24시간을 초과하여 무시됨');
                localStorage.removeItem(this.storageKey);
                return null;
            }

            if (window.logManager) {
                window.logManager.addLog(`🔄 게임 상태 복원 시작 (저장 시간: ${savedTime.toLocaleString()})`);
            }
            
            //console.log('💾 Game State Manager: 게임 상태 복원 완료');
            return state;
        } catch (error) {
            console.error('❌ Game State Manager: 복원 오류:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 게임 상태 복원 실패: ${error.message}`);
            }
            return null;
        }
    }

    // AI 모델들을 저장용 형태로 변환 (탐색원과 트레이너 데이터는 저장하지 않음)
    convertAiModelsForStorage(aiModels) {
        // 탐색원과 트레이너의 위치 및 상태 데이터는 저장하지 않음
        // N/B 길드에서 새로 시작하기 위해 빈 배열 반환
        return [];
    }

    // AI 모델들을 복원용 형태로 변환 (저장된 데이터는 사용하지 않음)
    convertAiModelsFromStorage(savedModels, scene) {
        // 저장된 탐색원과 트레이너 데이터는 사용하지 않음
        // N/B 길드에서 새로 시작하기 위해 빈 배열 반환
        return [];
    }

    // UI 상태 캡처
    captureUIState() {
        const uiState = {};
        
        // 학습 상태
        if (window.learningStatus) {
            uiState.learningStatus = window.learningStatus.text;
        }
        
        // 트레이너 대화창
        if (window.trainerDialog) {
            uiState.trainerDialog = window.trainerDialog.text;
        }
        
        // 트레이너 위치 정보
        if (window.trainerPositionInfo) {
            uiState.trainerPositionInfo = window.trainerPositionInfo.text;
        }
        
        // N/B 코인 표시
        if (window.nbCoinDisplay) {
            uiState.nbCoinDisplay = window.nbCoinDisplay.text;
        }
        
        // N/B 미네랄 표시
        if (window.nbMineralDisplay) {
            uiState.nbMineralDisplay = window.nbMineralDisplay.text;
        }
        
        // 매수 수익률 표시
        if (window.buyProfitRateDisplay) {
            uiState.buyProfitRateDisplay = window.buyProfitRateDisplay.text;
        }
        
        // 매도 수익률 표시
        if (window.sellProfitRateDisplay) {
            uiState.sellProfitRateDisplay = window.sellProfitRateDisplay.text;
        }
        
        return uiState;
    }

    // UI 상태 복원
    restoreUIState(uiState) {
        if (!uiState) return;
        
        // 학습 상태 복원
        if (window.learningStatus && uiState.learningStatus) {
            window.learningStatus.setText(uiState.learningStatus);
        }
        
        // 트레이너 대화창 복원
        if (window.trainerDialog && uiState.trainerDialog) {
            window.trainerDialog.setText(uiState.trainerDialog);
        }
        
        // 트레이너 위치 정보 복원
        if (window.trainerPositionInfo && uiState.trainerPositionInfo) {
            window.trainerPositionInfo.setText(uiState.trainerPositionInfo);
        }
        
        // N/B 코인 표시 복원
        if (window.nbCoinDisplay && uiState.nbCoinDisplay) {
            window.nbCoinDisplay.setText(uiState.nbCoinDisplay);
        }
        
        // N/B 미네랄 표시 복원
        if (window.nbMineralDisplay && uiState.nbMineralDisplay) {
            window.nbMineralDisplay.setText(uiState.nbMineralDisplay);
        }
        
        // 매수 수익률 표시 복원
        if (window.buyProfitRateDisplay && uiState.buyProfitRateDisplay) {
            window.buyProfitRateDisplay.setText(uiState.buyProfitRateDisplay);
        }
        
        // 매도 수익률 표시 복원
        if (window.sellProfitRateDisplay && uiState.sellProfitRateDisplay) {
            window.sellProfitRateDisplay.setText(uiState.sellProfitRateDisplay);
        }
    }

    // 트레이너 상태 캡처
    captureTrainerState() {
        const trainerState = {};
        
        // 트레이너 의사결정 핸들러 상태
        if (window.trainerDecisionHandler) {
            trainerState.waitCheckTimer = window.trainerDecisionHandler.waitCheckTimer || 0;
            trainerState.waitStartTime = window.trainerDecisionHandler.waitStartTime || null;
            trainerState.countdownStarted = window.trainerDecisionHandler.countdownStarted || false;
            trainerState.btcExplorationMode = window.trainerDecisionHandler.btcExplorationMode || false;
            trainerState.arrivalLogged = window.trainerDecisionHandler.arrivalLogged || false;
        }
        
        // 트레이너 이동 컨트롤러 상태
        if (window.trainerMovementController) {
            trainerState.movementSpeed = window.trainerMovementController.movementSpeed || 0.03;
            trainerState.arrivalThreshold = window.trainerMovementController.arrivalThreshold || 2;
            trainerState.minMovementDistance = window.trainerMovementController.minMovementDistance || 0.5;
        }
        
        return trainerState;
    }

    // 트레이너 상태 복원
    restoreTrainerState(trainerState) {
        if (!trainerState) return;
        
        // 트레이너 의사결정 핸들러 상태 복원
        if (window.trainerDecisionHandler) {
            window.trainerDecisionHandler.waitCheckTimer = trainerState.waitCheckTimer || 0;
            window.trainerDecisionHandler.waitStartTime = trainerState.waitStartTime || null;
            window.trainerDecisionHandler.countdownStarted = trainerState.countdownStarted || false;
            window.trainerDecisionHandler.btcExplorationMode = trainerState.btcExplorationMode || false;
            window.trainerDecisionHandler.arrivalLogged = trainerState.arrivalLogged || false;
        }
        
        // 트레이너 이동 컨트롤러 상태 복원
        if (window.trainerMovementController) {
            window.trainerMovementController.movementSpeed = trainerState.movementSpeed || 0.03;
            window.trainerMovementController.arrivalThreshold = trainerState.arrivalThreshold || 2;
            window.trainerMovementController.minMovementDistance = trainerState.minMovementDistance || 0.5;
        }
    }

    // 저장 히스토리 관리
    manageSaveHistory() {
        const historyKey = `${this.storageKey}_history`;
        let history = JSON.parse(localStorage.getItem(historyKey) || '[]');
        
        // 현재 상태를 히스토리에 추가
        history.push({
            timestamp: Date.now(),
            key: this.storageKey
        });
        
        // 최대 개수 제한
        if (history.length > this.maxSaveHistory) {
            history = history.slice(-this.maxSaveHistory);
        }
        
        localStorage.setItem(historyKey, JSON.stringify(history));
    }

    // 자동 저장 시작
    startAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setInterval(() => {
            if (window.gameInitializer && window.gameInitializer.gameData) {
                const gameData = {
                    ...window.gameInitializer.gameData,
                    aiModels: window.gameInitializer.aiModels || []
                };
                this.saveGameState(gameData);
                
                // 자동 저장 로그 추가
                if (window.logManager) {
                    window.logManager.addLog(`💾 자동 저장 완료: N/B코인 ${gameData.nbCoins || 0}개, 드랍아이템 ${gameData.dropItemsCount || 0}개`);
                }
            } else {
                // 자동 저장 실패 로그
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ 자동 저장 실패: gameInitializer 또는 gameData가 없음`);
                }
            }
        }, this.saveInterval);
        
        //console.log('💾 Game State Manager: 자동 저장 시작 (5초 간격)');
        if (window.logManager) {
            window.logManager.addLog(`💾 자동 저장 시스템 시작 (${this.saveInterval/1000}초 간격)`);
        }
    }

    // 자동 저장 중지
    stopAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
        //console.log('💾 Game State Manager: 자동 저장 중지');
    }

    // 게임 상태 초기화
    initializeGameState() {
        if (this.isInitialized) return;
        
        // 저장된 상태 복원 시도
        const savedState = this.loadGameState();
        if (savedState) {
            // 게임 데이터 복원
            if (window.gameInitializer && savedState.gameData) {
                window.gameInitializer.gameData = { ...savedState.gameData };
            }
            
            // UI 상태 복원
            this.restoreUIState(savedState.uiElements);
            
            // 트레이너 상태 복원
            this.restoreTrainerState(savedState.trainerState);
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 게임 상태 복원 완료 - 이전 세션에서 복구됨`);
            }
        }
        
        // 자동 저장 시작
        this.startAutoSave();
        
        this.isInitialized = true;
        //console.log('💾 Game State Manager: 게임 상태 초기화 완료');
    }

    // 게임 상태 완전 삭제
    clearGameState() {
        localStorage.removeItem(this.storageKey);
        localStorage.removeItem(`${this.storageKey}_history`);
        
        if (window.logManager) {
            window.logManager.addLog(`🗑️ 게임 상태 완전 삭제됨`);
        }
        
        //console.log('💾 Game State Manager: 게임 상태 삭제 완료');
    }

    // 저장된 상태 정보 조회
    getSaveInfo() {
        const savedState = localStorage.getItem(this.storageKey);
        if (!savedState) {
            return { exists: false };
        }
        
        try {
            const state = JSON.parse(savedState);
            return {
                exists: true,
                timestamp: new Date(state.timestamp),
                gameData: state.gameData,
                aiModelsCount: state.aiModels ? state.aiModels.length : 0
            };
        } catch (error) {
            return { exists: false, error: error.message };
        }
    }
}

// 전역 인스턴스 생성
window.gameStateManager = new GameStateManager();

// 페이지 로드 시 자동 초기화
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (window.gameStateManager) {
            window.gameStateManager.initializeGameState();
        }
    }, 2000); // 2초 후 초기화 (다른 모듈들이 로드될 시간 확보)
});

// 페이지 언로드 시 자동 저장
window.addEventListener('beforeunload', () => {
    if (window.gameStateManager && window.gameInitializer && window.gameInitializer.gameData) {
        const gameData = {
            ...window.gameInitializer.gameData,
            aiModels: window.gameInitializer.aiModels || []
        };
        window.gameStateManager.saveGameState(gameData);
    }
});
