// 트레이너 관리자 모듈
// 트레이너의 의사결정, 이동, 상태 관리를 담당

class TrainerManager {
    constructor() {
        this.movementSpeed = 0.1; // 이동 속도 조정 (매우 부드러운 이동)
        this.arrivalThreshold = 30;
        this.waitCheckInterval = 100; // 100ms마다 상태 체크
    }

    // 트레이너 모델 초기화
    initializeTrainerModel(model, config) {
        model.targetAction = '신호 대기';
        model.targetX = config.width / 2;
        model.targetY = config.height / 2;
        model.waitCheckTimer = 0;
        model.countdownStarted = false;
        model.arrivalLogged = false;
        model.btcExplorationMode = false;
        model.btcExplorationCompleted = false;
        model.infoCollectionMode = false;
        model.postBuyDecisionMade = false;
        model.postSellDecisionMade = false;
        
        // 색상 변경 로그 플래그들
        model.buyColorLogged = false;
        model.sellColorLogged = false;
        model.nbGuildColorLogged = false;
        
        return model;
    }

    // 트레이너 상태 업데이트
    updateTrainerState(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        const modelX = model.circle.x;
        const modelY = model.circle.y;
        
        // targetAction이 undefined이면 기본값 설정
        if (typeof model.targetAction === 'undefined' || model.targetAction === '') {
            this.initializeTrainerState(model, config);
        }
        
        let targetAction = model.targetAction;
        
        // 트레이너 활동 로그
        this.logTrainerActivity(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
        
        // BTC 시장에서 매수 전 예상 수익률 계산
        this.handleBTCMarketCalculation(model, config, currentMajority, buyProfitRate, trainerDialog);
        
        // 의사결정 시스템 처리
        targetAction = this.processDecisionSystem(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
        
        // 신호 대기 상태 처리
        targetAction = this.handleSignalWaiting(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        
        // BTC 시장 탐색 모드 처리
        targetAction = this.handleBTCExploration(model, config, currentMajority, buyProfitRate, trainerDialog);
        
        // 목표 도달 처리
        this.handleTargetArrival(model, config, targetAction, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        
        // 트레이너 이동 처리
        this.updateTrainerMovement(model, config);
        
        return targetAction;
    }

    // 트레이너 상태 초기화
    initializeTrainerState(model, config) {
        model.targetAction = 'N/B 코인 확인';
        model.targetX = 150;
        model.targetY = 150;
        model.circle.setFillStyle(0x88ccff); // 하늘색 (신호 대기)
        
        if (window.logManager) {
            window.logManager.addLog(`🔵 트레이너: targetAction 초기화 → N/B 길드에서 시작 (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
        }
    }

    // 트레이너 활동 로그
    logTrainerActivity(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (window.trainerActivityLogger && typeof window.trainerActivityLogger.logTrainerActivity === 'function') {
            window.trainerActivityLogger.logTrainerActivity(
                model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate
            );
        }
    }

    // BTC 시장 계산 처리
    handleBTCMarketCalculation(model, config, currentMajority, buyProfitRate, trainerDialog) {
        if (window.btcMarketCalculator && typeof window.btcMarketCalculator.calculateBuyProfitRateAtMarket === 'function') {
            const calculationResult = window.btcMarketCalculator.calculateBuyProfitRateAtMarket(
                model.circle.x, model.circle.y, config, currentMajority, null, trainerDialog
            );
            
            if (calculationResult && model.infoCollectionMode) {
                model.infoCollectionMode = false;
                setTimeout(() => {
                    model.targetAction = '정보 수집 완료';
                }, 2000);
            }
        }
    }

    // 의사결정 시스템 처리
    processDecisionSystem(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        
        // 현재 구역에서 의사 결정 매칭 확인
        const currentZone = window.decisionSystem && typeof window.decisionSystem.getCurrentZone === 'function' 
            ? window.decisionSystem.getCurrentZone(model.circle.x, model.circle.y, startX, topY, spacing, config)
            : '기타영역';
            
        const zoneDecision = window.decisionSystem && typeof window.decisionSystem.getZoneDecision === 'function'
            ? window.decisionSystem.getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
            : null;
        
        // 현재 구역에서 해당하는 의사 결정이 없으면 신호 대기 상태로 전환
        if (!zoneDecision && !model.countdownStarted && !model.btcExplorationMode) {
            if (model.targetAction !== '신호 대기') {
                const previousAction = model.targetAction;
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                model.circle.setFillStyle(0x88ccff);
                
                if (window.logManager) {
                    window.logManager.addLog(`🔵 트레이너: 현재 구역(${currentZone})에서 의사 결정 없음 → 신호 대기 센터로 이동 (이전 액션: ${previousAction})`);
                }
                
                if (window.trainerActivityLogger) {
                    window.trainerActivityLogger.logStateChange(previousAction, '신호 대기', `현재 구역(${currentZone})에서 의사 결정 없음`);
                }
            }
        } else if (zoneDecision && !model.countdownStarted) {
            model.targetAction = zoneDecision.action;
            model.targetX = zoneDecision.targetX;
            model.targetY = zoneDecision.targetY;
            
            if (window.trainerActivityLogger) {
                window.trainerActivityLogger.logDecision(model.targetAction, `구역: ${currentZone}, 신호: ${currentMajority}`);
            }
            
            this.updateTrainerColor(model, model.targetAction);
        } else if (model.countdownStarted) {
            model.targetAction = '신호 대기';
            model.targetX = config.width / 2;
            model.targetY = config.height / 2;
            model.circle.setFillStyle(0x88ccff);
        }
        
        return model.targetAction;
    }

    // 트레이너 색상 업데이트
    updateTrainerColor(model, targetAction) {
        let color = 0x88ccff; // 기본 색상
        
        switch (targetAction) {
            case '매도':
                color = 0xff8800;
                if (window.logManager && !model.sellColorLogged) {
                    window.logManager.addLog(`🟠 트레이너: 매도 의사결정! 주황색으로 변경`);
                    model.sellColorLogged = true;
                }
                break;
            case '매수':
                color = 0x0088ff;
                if (window.logManager && !model.buyColorLogged) {
                    window.logManager.addLog(`🔵 트레이너: 매수 의사결정! 파란색으로 변경`);
                    model.buyColorLogged = true;
                }
                break;
            case 'BTC 시장 방문':
                color = 0x0088ff;
                if (window.logManager) {
                    window.logManager.addLog(`🔵 트레이너: BTC 시장 방문! 파란색으로 변경`);
                }
                break;
            case 'N/B 길드 방문':
                color = 0xff8800;
                if (window.logManager && !model.nbGuildColorLogged) {
                    window.logManager.addLog(`🟠 트레이너: N/B 길드 방문! 주황색으로 변경`);
                    model.nbGuildColorLogged = true;
                }
                break;
            case '대기':
                color = 0xffff00;
                if (window.logManager) {
                    window.logManager.addLog(`🟡 트레이너: 대기 상태! 노란색으로 변경`);
                }
                break;
            case '신호 대기':
                color = 0x88ccff;
                if (window.logManager) {
                    window.logManager.addLog(`🔵 트레이너: 신호 대기 상태! 하늘색으로 변경`);
                }
                break;
        }
        
        model.circle.setFillStyle(color);
    }

    // 신호 대기 상태 처리
    handleSignalWaiting(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        if (model.targetAction === '신호 대기') {
            // 랜덤 움직임을 위한 타이머 초기화
            if (!model.randomMoveTimer) {
                model.randomMoveTimer = 0;
                model.randomMoveInterval = Math.floor(Math.random() * 3000) + 2000; // 2-5초마다 랜덤 이동
            }
            
            model.randomMoveTimer++;
            
            // 랜덤 이동 간격에 도달하면 새로운 랜덤 목표 설정
            if (model.randomMoveTimer >= model.randomMoveInterval) {
                this.setRandomTarget(model, config);
                model.randomMoveTimer = 0;
                model.randomMoveInterval = Math.floor(Math.random() * 3000) + 2000;
                
                if (window.logManager) {
                    window.logManager.addLog(`🎲 트레이너 랜덤 이동: 새로운 목표 설정 (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
                }
            }
            
            // 현재 위치와 목표 위치 간의 거리 계산
            const distanceToTarget = Math.sqrt((model.circle.x - model.targetX) ** 2 + (model.circle.y - model.targetY) ** 2);
            
            // 목표에 도달했는지 확인
            if (distanceToTarget < 30) {
                if (!model.waitCheckTimer) {
                    model.waitCheckTimer = 0;
                    model.waitStartTime = Date.now();
                }
                model.waitCheckTimer++;
                model.countdownStarted = true;
                
                const elapsedSeconds = (Date.now() - model.waitStartTime) / 1000;
                
                if (elapsedSeconds >= 5) {
                    model.targetAction = 'BTC 시장 탐색';
                    model.targetX = config.width - 100;
                    model.targetY = config.height - 100;
                    model.circle.setFillStyle(0x0088ff);
                    model.btcExplorationMode = true;
                    model.countdownStarted = false;
                    model.randomMoveTimer = 0;
                    
                    if (window.logManager) {
                        window.logManager.addLog(`🔵 트레이너: 랜덤 탐색 완료 → BTC 시장 탐색으로 이동!`);
                    }
                    
                    const dialogMessage = `🔵 [랜덤 탐색 완료] BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정`;
                    trainerDialog.setText(dialogMessage);
                    if (window.logManager) {
                        window.logManager.addLog(dialogMessage);
                    }
                } else {
                    const remainingSeconds = Math.ceil(5 - elapsedSeconds);
                    const dialogMessage = `🎲 [랜덤 탐색] 현재 위치에서 대기 중... (${remainingSeconds}초 후 BTC 시장 탐색) N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
                    trainerDialog.setText(dialogMessage);
                    if (window.logManager) {
                        window.logManager.addLog(dialogMessage);
                    }
                }
            } else {
                if (distanceToTarget > 50 && !model.countdownStarted) {
                    model.waitCheckTimer = 0;
                }
                
                // 랜덤 이동 중 대화창 업데이트
                const dialogMessage = `🎲 [랜덤 탐색] 이동 중... (${Math.round(distanceToTarget)}px 남음) N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
                trainerDialog.setText(dialogMessage);
            }
        }
        
        return model.targetAction;
    }
    
    // 랜덤 목표 위치 설정
    setRandomTarget(model, config) {
        // 화면 경계 내에서 랜덤한 위치 선택
        const margin = 100; // 화면 가장자리에서 100px 여백
        const minX = margin;
        const maxX = config.width - margin;
        const minY = margin;
        const maxY = config.height - margin;
        
        // 랜덤한 목표 위치 생성
        model.targetX = Math.floor(Math.random() * (maxX - minX + 1)) + minX;
        model.targetY = Math.floor(Math.random() * (maxY - minY + 1)) + minY;
        
        // 학습 시스템을 통해 랜덤 이동 학습
        if (window.learningSystem && typeof window.learningSystem.learnRandomMovement === 'function') {
            window.learningSystem.learnRandomMovement(model, model.targetX, model.targetY);
        }
    }

    // BTC 시장 탐색 모드 처리
    handleBTCExploration(model, config, currentMajority, buyProfitRate, trainerDialog) {
        if (model.targetAction === 'BTC 시장 탐색') {
            const distanceToBTCMarket = Math.sqrt((model.circle.x - (config.width - 100)) ** 2 + (model.circle.y - (config.height - 100)) ** 2);
            
            // BTC 시장 충돌 검사
            let isCollidingWithBTCMarket = false;
            if (window.btcMarketPolygon && model.circle) {
                const circleBounds = model.circle.getBounds();
                const polygonBounds = window.btcMarketPolygon.getBounds();
                isCollidingWithBTCMarket = Phaser.Geom.Rectangle.Overlaps(circleBounds, polygonBounds);
                
                if (isCollidingWithBTCMarket) {
                    const centerDistance = Math.sqrt(
                        (model.circle.x - window.btcMarketPolygon.x) ** 2 + 
                        (model.circle.y - window.btcMarketPolygon.y) ** 2
                    );
                    if (centerDistance > 50) {
                        isCollidingWithBTCMarket = false;
                    }
                }
            }
            
            if (isCollidingWithBTCMarket || distanceToBTCMarket < 60) {
                if (model.btcExplorationMode) {
                    model.btcExplorationMode = false;
                    if (window.logManager) {
                        window.logManager.addLog(`🎯 트레이너가 BTC 시장에 도달! 학습 모델 기반 처리 시작`);
                    }
                }
                
                if (window.btcMarketLearningHandler) {
                    window.btcMarketLearningHandler.handleBTCMarketArrival(model, config, trainerDialog, currentMajority, null);
                } else {
                    // 기본 처리: 즉시 신호 대기 센터로 복귀
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    model.circle.setFillStyle(0x88ccff);
                }
            } else {
                const statusText = `🔵 [BTC 탐색] BTC 시장 탐색으로 이동 중... (${Math.round(distanceToBTCMarket)}px 남음)`;
                trainerDialog.setText(statusText);
                if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
                    window.logManager.addLog(`🔵 BTC 시장 탐색 진행 중: 거리 ${Math.round(distanceToBTCMarket)}px, 현재 신호: ${currentMajority}`);
                }
            }
            
            // BTC 탐색 관리자 모듈이 없을 때의 기본 처리
            if (!window.btcExplorationManager) {
                if (model.btcExplorationMode && !model.btcExplorationCompleted) {
                    model.btcExplorationCompleted = true;
                    
                    if (window.logManager) {
                        window.logManager.addLog(`🔵 BTC 탐색 완료: 2초 후 신호 대기 센터로 복귀 (현재 신호: ${currentMajority})`);
                    }
                    
                    setTimeout(() => {
                        const previousAction = model.targetAction;
                        model.targetAction = '신호 대기';
                        model.targetX = config.width / 2;
                        model.targetY = config.height / 2;
                        model.circle.setFillStyle(0x88ccff);
                        model.waitCheckTimer = 0;
                        model.countdownStarted = false;
                        model.arrivalLogged = false;
                        model.btcExplorationMode = false;
                        model.btcExplorationCompleted = false;
                        
                        if (window.logManager) {
                            window.logManager.addLog(`🔵 BTC 탐색 완료: targetAction 변경 (${previousAction} → 신호 대기)`);
                        }
                        
                        const dialogMessage = `🔵 [탐색 완료] 신호 대기 센터로 복귀 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
                        trainerDialog.setText(dialogMessage);
                        if (window.logManager) {
                            window.logManager.addLog(dialogMessage);
                        }
                    }, 2000);
                }
            }
        }
        
        return model.targetAction;
    }

    // 목표 도달 처리
    handleTargetArrival(model, config, targetAction, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        const distanceToTarget = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
        
        if (distanceToTarget < 30) {
            if (!model.arrivalLogged) {
                let arrivalMessage = '';
                if (targetAction === 'BTC 시장 방문') {
                    arrivalMessage = `🎯 트레이너: BTC 시장에 도착! 매수 전 예상 수익률 계산 시작...`;
                } else if (targetAction === 'N/B 길드 방문') {
                    arrivalMessage = `🎯 트레이너: N/B 길드에 도착! 매도 전 예상 수익률 계산 시작...`;
                } else if (targetAction === '신호 대기') {
                    arrivalMessage = `🎯 트레이너: 신호 대기 센터에 도착! 신호 대기 시작...`;
                }
                
                if (arrivalMessage && window.logManager) {
                    window.logManager.addLog(arrivalMessage);
                }
                model.arrivalLogged = true;
            }
            
            model.role.setText(`트레이너 (${targetAction})`);
            
            // N/B 길드 도달 시 처리
            if (targetAction === 'N/B 길드 방문') {
                this.handleNBGuildArrival(model, config, currentMajority, buyProfitRate, sellProfitRate, trainerDialog);
            }
        }
    }

    // N/B 길드 도달 처리
    handleNBGuildArrival(model, config, currentMajority, buyProfitRate, sellProfitRate, trainerDialog) {
        // 구역 불일치 체크: BLUE 구역에서는 매도 준비 불가
        if (currentMajority === 'BLUE') {
            model.targetAction = '신호 대기';
            model.targetX = config.width / 2;
            model.targetY = config.height / 2;
            model.circle.setFillStyle(0x88ccff);
            
            if (window.logManager) {
                window.logManager.addLog(`🔵 N/B 길드에서 매도 준비 중이지만 BLUE 구역임 → 신호 대기 센터로 이동`);
            }
            return;
        }
        
        // 매수한 적이 있는 경우에만 매도 전 예상 수익률 계산
        if (window.buyPrice > 0) {
            this.calculateSellProfitRate(model, config, trainerDialog);
        }
    }

    // 매도 전 예상 수익률 계산
    calculateSellProfitRate(model, config, trainerDialog) {
        if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
            const sellProfitRate = window.currentPriceManager.calculateSellProfitRate(window.buyPrice);
            
            // 매도 전 예상 수익률 표시 업데이트
            const profitColor = sellProfitRate >= 0 ? '#00ff88' : '#ff0088';
            if (window.sellProfitRateDisplay) {
                window.sellProfitRateDisplay.setFill(profitColor);
                window.sellProfitRateDisplay.setText(`매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
            }
            
            // 상태 즉시 저장
            if (window.saveGameState) {
                window.saveGameState();
            }
            
            // 트레이너 역할 텍스트 업데이트
            model.role.setText(`수익률: ${sellProfitRate.toFixed(2)}%`);
            
            const currentPriceLog = window.currentPriceManager.generateCurrentPriceLog();
            console.log(`📊 트레이너: N/B 길드에서 매도 전 예상 수익률 계산 완료: ${sellProfitRate.toFixed(2)}% (현재가: ₩${currentPriceLog.currentPrice.toLocaleString()})`);
            
            // 트레이너 대화창 업데이트
            const dialogMessage = `N/B 길드: 매도 전 예상 수익률 계산 완료! ${sellProfitRate.toFixed(2)}% (${currentPriceLog.currentPriceText})`;
            trainerDialog.setText(dialogMessage);
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        }
    }

    // 트레이너 이동 처리
    updateTrainerMovement(model, config) {
        // TrainerStateHandler가 이미 처리하고 있으면 건너뛰기 (중복 방지)
        if (window.trainerStateHandler && typeof window.trainerStateHandler.updateTrainerMovement === 'function') {
            return; // TrainerStateHandler가 처리하도록 함
        }
        
        // TrainerStateHandler가 없는 경우에만 기본 이동 처리 (fallback)
        // 하지만 현재는 TrainerStateHandler가 항상 존재하므로 이 부분은 실행되지 않음
        if (window.logManager) {
            window.logManager.addLog(`⚠️ TrainerManager 이동 처리 스킵: TrainerStateHandler가 처리 중`);
        }
        
        // 화면 위치 정보 업데이트만 수행
        if (window.trainerPositionInfo) {
            const currentPos = `(${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)})`;
            const targetPos = `(${model.targetX.toFixed(1)}, ${model.targetY.toFixed(1)})`;
            const actualDistance = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
            
            // 구역 정보 가져오기
            const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const targetZone = window.gameInitializer?.getCurrentZoneName(model.targetX, model.targetY) || '기타영역';
            
            const positionText = `📍 위치: ${currentPos} (${currentZone}) | 목표: ${targetPos} (${targetZone}) | 거리: ${Math.round(actualDistance)}px`;
            window.trainerPositionInfo.setText(positionText);
        }
    }

    // 트레이너 액션 처리 (매수/매도/대기)
    handleTrainerActions(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        
        // 매수 액션 처리 (BLUE 구역에서만)
        if (model.targetAction === '매수' && !window.lastBuyAction && currentMajority === 'BLUE') {
            this.handleBuyAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        }
        // 매도 액션 처리 (ORANGE 구역에서만)
        else if (model.targetAction === '매도' && !window.lastSellAction && nbCoins > 0 && currentMajority === 'ORANGE') {
            this.handleSellAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        }
        // 대기 상태
        else if (model.targetAction === '대기') {
            this.handleWaitAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        }
        // 정보 수집 후 신호 대기 센터로 복귀
        else if (model.targetAction === '정보 수집 완료') {
            this.handleInfoCollectionComplete(model, config, currentMajority, buyProfitRate, trainerDialog);
        }
    }

    // 매수 액션 처리
    handleBuyAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        window.lastBuyAction = true;
        window.lastSellAction = false;
        
        // 플래그 리셋
        model.postBuyDecisionMade = false;
        model.postSellDecisionMade = false;
        model.buyColorLogged = false;
        model.sellColorLogged = false;
        model.nbGuildColorLogged = false;
        
        // 매수가격 저장
        if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
            window.buyPrice = window.currentPriceManager.parseCurrentPrice();
            console.log(`💰 매수가격 저장: ₩${window.buyPrice.toLocaleString()}`);
        }
        
        // 매수 완료 후 매수 전 예상 수익률 리셋
        window.buyProfitRate = 0;
        if (window.buyProfitRateDisplay) {
            window.buyProfitRateDisplay.setFill('#00ff88');
            window.buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
        }
        
        // N/B 코인 아이템 생성
        if (window.createNBCoinItem) {
            window.createNBCoinItem();
        }
        
        // UI 업데이트
        if (window.nbCoinDisplay) {
            window.nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${window.nbCoinItems ? window.nbCoinItems.length : 0}개)`);
        }
        
        // 대화창 업데이트
        let dialogMessage = '';
        if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
            dialogMessage = window.decisionSystem.generateDecisionLogMessage('매수 완료', currentMajority, nbCoins, window.nbMinerals, window.nbCoinItems, window.currentPriceText, buyProfitRate, sellProfitRate);
        } else {
            dialogMessage = `💰 [매수 완료] N/B 코인: ${nbCoins}개, 미네랄: ${window.nbMinerals ? window.nbMinerals.toFixed(2) : '0.00'}%, 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
        }
        trainerDialog.setText(dialogMessage);
        
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
        
        console.log(`💰 매수 완료! N/B 코인 드랍 아이템 생성, 매수가: ${window.currentPriceText}`);
        
        // 매수 완료 후 다음 의사결정 실행
        if (!model.postBuyDecisionMade) {
            model.postBuyDecisionMade = true;
            const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                ? window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, (config.width - (120 * 2)) / 2, 60, 120, config)
                : null;
                
            if (nextDecision) {
                model.targetAction = nextDecision.action;
                model.targetX = nextDecision.targetX;
                model.targetY = nextDecision.targetY;
                
                let actionColor = 0x88ccff;
                if (window.decisionSystem && typeof window.decisionSystem.getActionColor === 'function') {
                    actionColor = window.decisionSystem.getActionColor(model.targetAction);
                } else {
                    switch (model.targetAction) {
                        case '매수': actionColor = 0x0088ff; break;
                        case '매도': actionColor = 0xff8800; break;
                        case '신호 대기': actionColor = 0x88ccff; break;
                        case 'BTC 시장 탐색': actionColor = 0x88ff88; break;
                        default: actionColor = 0x88ccff;
                    }
                }
                model.circle.setFillStyle(actionColor);
                
                if (window.logManager) {
                    window.logManager.addLog(`🔄 매수 완료 후 다음 의사결정: ${model.targetAction}`);
                }
            }
        }
        
        // 상태 즉시 저장
        if (window.saveGameState) {
            window.saveGameState();
        }
    }

    // 매도 액션 처리
    handleSellAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        window.lastSellAction = true;
        window.lastBuyAction = false;
        
        // 플래그 리셋
        model.postBuyDecisionMade = false;
        model.postSellDecisionMade = false;
        model.buyColorLogged = false;
        model.sellColorLogged = false;
        model.nbGuildColorLogged = false;
        
        window.nbCoins--;
        
        // 매도 완료 시 현재 수익률을 N/B 미네랄에 누적
        let currentPnl = 0;
        if (window.learningSystem && typeof window.learningSystem.getCurrentProfitRate === 'function') {
            currentPnl = window.learningSystem.getCurrentProfitRate();
        } else {
            const pnlElement = document.getElementById('selected-coin-pnl');
            if (pnlElement) {
                const pnlMatch = pnlElement.textContent.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
                if (pnlMatch) {
                    currentPnl = parseFloat(pnlMatch[1]);
                }
            }
        }
        window.nbMinerals += currentPnl;
        
        // N/B 미네랄 변경됨 (game-state-manager.js에서 자동 저장됨)
        
        // 매도 시 수익률 리셋
        window.buyPrice = 0;
        window.buyProfitRate = 0;
        window.sellProfitRate = 0;
        
        if (window.buyProfitRateDisplay) {
            window.buyProfitRateDisplay.setFill('#00ff88');
            window.buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
        }
        if (window.sellProfitRateDisplay) {
            window.sellProfitRateDisplay.setFill('#ff0088');
            window.sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
        }
        
        // UI 업데이트
        if (window.nbCoinDisplay) {
            window.nbCoinDisplay.setText(`N/B 코인: ${window.nbCoins}개 (드랍 아이템: ${window.nbCoinItems ? window.nbCoinItems.length : 0}개)`);
        }
        if (window.nbMineralDisplay) {
            window.nbMineralDisplay.setText(`N/B 미네랄: ${window.nbMinerals.toFixed(2)}%`);
        }
        
        // 대화창 업데이트
        let dialogMessage = '';
        if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
            dialogMessage = window.decisionSystem.generateDecisionLogMessage('매도 완료', currentMajority, window.nbCoins, window.nbMinerals, window.nbCoinItems, window.currentPriceText, buyProfitRate, sellProfitRate);
        } else {
            dialogMessage = `💸 [매도 완료] N/B 코인: ${window.nbCoins}개, 미네랄: ${window.nbMinerals.toFixed(2)}%, 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
        }
        trainerDialog.setText(dialogMessage);
        
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
        
        console.log(`💸 매도 완료! N/B 코인: ${window.nbCoins}개, N/B 미네랄 누적: ${window.nbMinerals.toFixed(2)}%, 수익률 리셋`);
        
        // 매도 완료 후 다음 의사결정 실행
        if (!model.postSellDecisionMade) {
            model.postSellDecisionMade = true;
            const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                ? window.decisionSystem.getNextDecision(currentMajority, window.nbCoins, buyProfitRate, sellProfitRate, (config.width - (120 * 2)) / 2, 60, 120, config)
                : null;
                
            if (nextDecision) {
                model.targetAction = nextDecision.action;
                model.targetX = nextDecision.targetX;
                model.targetY = nextDecision.targetY;
                
                let actionColor = 0x88ccff;
                if (window.decisionSystem && typeof window.decisionSystem.getActionColor === 'function') {
                    actionColor = window.decisionSystem.getActionColor(model.targetAction);
                } else {
                    switch (model.targetAction) {
                        case '매수': actionColor = 0x0088ff; break;
                        case '매도': actionColor = 0xff8800; break;
                        case '신호 대기': actionColor = 0x88ccff; break;
                        case 'BTC 시장 탐색': actionColor = 0x88ff88; break;
                        default: actionColor = 0x88ccff;
                    }
                }
                model.circle.setFillStyle(actionColor);
                
                if (window.logManager) {
                    window.logManager.addLog(`🔄 매도 완료 후 다음 의사결정: ${model.targetAction}`);
                }
            }
        }
        
        // 상태 즉시 저장
        if (window.saveGameState) {
            window.saveGameState();
        }
    }

    // 대기 액션 처리
    handleWaitAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        window.lastBuyAction = false;
        window.lastSellAction = false;
        
        let dialogMessage = '';
        if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
            dialogMessage = window.decisionSystem.generateDecisionLogMessage('대기', currentMajority, nbCoins, window.nbMinerals, window.nbCoinItems, window.currentPriceText, buyProfitRate, sellProfitRate);
        } else {
            dialogMessage = `⏳ [대기] N/B 코인: ${nbCoins}개, 미네랄: ${window.nbMinerals ? window.nbMinerals.toFixed(2) : '0.00'}%, 현재 구역: ${currentMajority}`;
        }
        trainerDialog.setText(dialogMessage);
        
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
    }

    // 정보 수집 완료 처리
    handleInfoCollectionComplete(model, config, currentMajority, buyProfitRate, trainerDialog) {
        // 예상 수익률이 계산되었고 BLUE 신호라면 즉시 매수
        if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
            model.targetAction = '매수';
            model.targetX = (config.width - (120 * 2)) / 2;
            model.targetY = 60;
            model.circle.setFillStyle(0x0088ff);
            
            console.log(`📈 트레이너: 정보 수집 완료 후 즉시 매수! 예상 수익률: ${buyProfitRate.toFixed(2)}%`);
            const dialogMessage = `📈 [정보 수집 완료] 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%) → 즉시 매수!`;
            trainerDialog.setText(dialogMessage);
            
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        } else {
            // 예상 수익률이 계산되지 않았거나 BLUE 신호가 아니면 신호 대기 센터로 복귀
            model.targetAction = '신호 대기';
            model.targetX = config.width / 2;
            model.targetY = config.height / 2;
            model.circle.setFillStyle(0x88ccff);
            model.waitCheckTimer = 0;
            model.countdownStarted = false;
            
            console.log(`🔵 트레이너: 정보 수집 완료, 신호 대기 센터로 복귀!`);
            const dialogMessage = `🔵 [정보 수집 완료] 신호 대기 센터로 복귀 중...`;
            trainerDialog.setText(dialogMessage);
            
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        }
    }
}

// 전역 인스턴스 생성
window.trainerManager = new TrainerManager();
