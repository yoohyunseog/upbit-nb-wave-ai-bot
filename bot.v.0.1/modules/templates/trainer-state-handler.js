// 트레이너 상태 처리 모듈
// 트레이너의 상태 관리와 액션 처리를 담당

class TrainerStateHandler {
    constructor() {
        this.movementSpeed = 0.3; // 이동 속도 조정 (부드러운 이동)
        this.arrivalThreshold = 30;
    }
    // 5W1H 유틸
    _fiveW1H(actionLabel, model, zone, currentMajority, why, how) {
        return {
            who: 'TrainerModel',
            what: actionLabel,
            when: Date.now(),
            where: { zone, x: Math.round(model?.circle?.x||0), y: Math.round(model?.circle?.y||0) },
            why: why || '',
            how: how || ''
        };
    }
    _composeWhyForBuy(currentMajority, nbCoins, buyProfitRate, threshold, extra) {
        const parts = [];
        parts.push(`signal=${currentMajority}`);
        parts.push(`nbCoins=${nbCoins}`);
        if (typeof buyProfitRate === 'number') parts.push(`buyRate=${buyProfitRate.toFixed(2)}%`);
        if (typeof threshold === 'number') parts.push(`threshold=${threshold.toFixed(2)}%`);
        if (extra) parts.push(extra);
        return parts.join(', ');
    }
    _composeWhyForSell(currentMajority, nbCoins, sellProfitRate, extra) {
        const parts = [];
        parts.push(`signal=${currentMajority}`);
        parts.push(`nbCoins=${nbCoins}`);
        if (typeof sellProfitRate === 'number') parts.push(`sellRate=${sellProfitRate.toFixed(2)}%`);
        if (extra) parts.push(extra);
        return parts.join(', ');
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
        
        return targetAction;
    }

    // 트레이너 상태 초기화
    initializeTrainerState(model, config) {
        model.targetAction = 'BTC 시장 탐색';
        // BTC 시장 탐색 구역에서 시작하도록 변경
        model.targetX = config.width - 100;
        model.targetY = config.height - 100;
        model.circle.setFillStyle(0x0088ff); // 파란색 (BTC 시장 탐색)
        // 서버 파일 로그 기록
        try { fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'INIT', targetAction: model.targetAction, zone: 'INIT', ts: Date.now() }) }); } catch(_){ }
        
        if (window.logManager) {
            window.logManager.addLog(`🔵 트레이너: targetAction 초기화 → BTC 시장 탐색 구역에서 시작 (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
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
        
        // 코인 보유 중이고 BLUE 신호라면: 매수는 제한되어 있으므로 신호 대기 센터에 머무름
        try {
            const liveMaj = (currentMajority || (document.getElementById('majority-zone')?.textContent || '')).toUpperCase();
            const coinsHeld = (typeof nbCoins === 'number' ? nbCoins : 0) || (window.gameInitializer?.gameData?.nbCoins || 0);
            if (coinsHeld > 0 && liveMaj.includes('BLUE')) {
                if (model.targetAction !== '신호 대기') {
                    const previousAction = model.targetAction;
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    model.circle.setFillStyle(0x88ccff);
                    try {
                        const why = `holding_nbcoins_wait_for_sell (nbCoins=${coinsHeld}, majority=${liveMaj})`;
                        const five = this._fiveW1H('STATE_CHANGE', model, currentZone, currentMajority, why, 'processDecisionSystem');
                        fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: previousAction, to: model.targetAction, zone: currentZone, majority: currentMajority, nbCoins: coinsHeld, reason: 'holding_nbcoins_wait_for_sell', ts: Date.now(), ...five }) });
                    } catch(_) { }
                    if (window.logManager) {
                        window.logManager.addLog(`🔵 코인 보유(${coinsHeld}) + BLUE 신호 → 신호 대기 센터 유지`);
                    }
                }
                return model.targetAction;
            }
        } catch(_) { }

        // 현재 분봉에서 매수할 분봉(조건 충족)이 없으면 신호 대기 센터 유지
        try {
            const maj = (currentMajority || '').toUpperCase();
            if (maj.includes('BLUE')) {
                // 좌측 패널 선택 분봉과 보유 상태 확인
                const selected = document.querySelector('.left-panel .timeframe-card-list .selected') 
                    || document.querySelector('#timeframe-cards-container .timeframe-card.active');
                const tf = selected ? (selected.getAttribute('data-timeframe')||'').trim() : (window.timeframeCards?.getCurrentTimeframe?.() || null);
                const leftHeld = tf && window.nbCoinStatus && (typeof window.nbCoinStatus[tf] !== 'undefined') ? (window.nbCoinStatus[tf] ? 1 : 0) : 0;
                const threshold = (window.gameInitializer && window.gameInitializer.gameData && typeof window.gameInitializer.gameData.buyThresholdPercent === 'number')
                    ? window.gameInitializer.gameData.buyThresholdPercent : 0.5;
                const isBuyable = (leftHeld === 0) && (typeof buyProfitRate === 'number') && !isNaN(buyProfitRate) && (buyProfitRate >= threshold);
                if (!isBuyable) {
                    const prev = model.targetAction;
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    model.circle.setFillStyle(0x88ccff);
                    // 로그
                    try {
                        const five = this._fiveW1H('STATE_CHANGE', model, currentZone, currentMajority, `reason=no_buyable_timeframe(tf=${tf||'unknown'}, held=${leftHeld}, rate=${(buyProfitRate||0).toFixed?.(2)||buyProfitRate}%, thr=${threshold.toFixed(2)}%)`, 'processDecisionSystem');
                        fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: prev, to: model.targetAction, ts: Date.now(), ...five }) });
                    } catch(_) {}
                    return model.targetAction;
                }
            }
        } catch(_) { /* ignore */ }
            
        const zoneDecision = window.decisionSystem && typeof window.decisionSystem.getZoneDecision === 'function'
            ? window.decisionSystem.getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
            : null;
        
        // BTC 시장 탐색 구역에 도달했을 때 수익률 계산 트리거
        if (currentZone === 'BTC시장탐색구역' && !model.btcMarketArrivalLogged) {
            model.btcMarketArrivalLogged = true;
            model.btcMarketWaitTimer = 0;
            
            if (window.logManager) {
                window.logManager.addLog(`🔵 BTC 시장 탐색 구역 도착 - 수익률 계산 트리거`);
            }
            
            // BTC 시장 수익률 계산기 호출
            if (window.btcMarketProfitCalculator) {
                const calculationResult = window.btcMarketProfitCalculator.calculateBuyProfitRateAtMarket(
                    currentMajority, 
                    null, // buyProfitRateDisplay
                    null, // trainerDialog
                    model, 
                    config, 
                    config.width - 100, 
                    config.height - 100
                );
                
                if (calculationResult) {
                    if (window.logManager) {
                        window.logManager.addLog(`✅ BTC 시장 수익률 계산 완료`);
                    }
                } else {
                    if (window.logManager) {
                        window.logManager.addLog(`⚠️ BTC 시장 수익률 계산 실패`);
                    }
                }
            }
            
            // 학습 모델 기반 수익률 계산도 시도
            if (window.btcMarketLearningHandler) {
                window.btcMarketLearningHandler.calculateProfitRateWithLearning(
                    model, config, null, currentMajority, null
                );
            }
        }
        
        // 현재 구역에서 해당하는 의사 결정이 없으면 신호 대기 상태로 전환
        if (!zoneDecision && !model.countdownStarted && !model.btcExplorationMode) {
            if (model.targetAction !== '신호 대기') {
                const previousAction = model.targetAction;
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                model.circle.setFillStyle(0x88ccff);
                // 서버 파일 로그 기록 (5W1H)
                try {
                    const five = this._fiveW1H('STATE_CHANGE', model, currentZone, currentMajority, 'reason=no_zone_decision', 'system');
                    fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: previousAction, to: model.targetAction, zone: currentZone, majority: currentMajority, nbCoins: nbCoins, reason: 'no_zone_decision', ts: Date.now(), ...five }) });
                } catch(_){ }
                
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
            // 서버 파일 로그 기록 (의사결정 + 5W1H)
            try {
                const why = this._composeWhyForBuy(currentMajority, nbCoins, buyProfitRate, (window.gameInitializer?.gameData?.buyThresholdPercent ?? undefined));
                const five = this._fiveW1H('DECISION', model, currentZone, currentMajority, why, 'decisionSystem');
                fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'DECISION', targetAction: model.targetAction, zone: currentZone, majority: currentMajority, nbCoins: nbCoins, buyProfitRate: buyProfitRate, sellProfitRate: sellProfitRate, decisionSource: 'decisionSystem', isLearningActive: !!window.btcMarketLearningHandler, ts: Date.now(), ...five }) });
            } catch(_){ }
            
            if (window.trainerActivityLogger) {
                window.trainerActivityLogger.logDecision(model.targetAction, `구역: ${currentZone}, 신호: ${currentMajority}`);
            }
            
            this.updateTrainerColor(model, model.targetAction);
        } else if (model.countdownStarted) {
            model.targetAction = '신호 대기';
            model.targetX = config.width / 2;
            model.targetY = config.height / 2;
            model.circle.setFillStyle(0x88ccff);
            try {
                const five = this._fiveW1H('STATE_CHANGE', model, currentZone, currentMajority, 'reason=countdown', 'system');
                fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: 'countdown', to: model.targetAction, zone: currentZone, majority: currentMajority, nbCoins: nbCoins, reason: 'countdown', ts: Date.now(), ...five }) });
            } catch(_){ }
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
        // 서버 활동 로그(상태/색상 변경)
        try {
            fetch('/api/trainer/activity-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ type:'STATE', action: targetAction, color, ts: Date.now() }) });
        } catch(_) {}
    }

    // 신호 대기 상태 처리
    handleSignalWaiting(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        if (model.targetAction === '신호 대기') {
            // 코인 보유 + BLUE 신호 또는 좌측 패널 풀 매수 시, 신호 대기 센터 고정
            try {
                const liveMaj = (currentMajority || (document.getElementById('majority-zone')?.textContent || '')).toUpperCase();
                const coinsHeld = (typeof nbCoins === 'number' ? nbCoins : 0) || (window.gameInitializer?.gameData?.nbCoins || 0);
                // 좌측 패널 전 분봉 보유 여부 계산
                let allHeld = false;
                try {
                    const candidates = ['minute1','minute3','minute5','minute10','minute15','minute30','minute60','day'];
                    const fromDom = Array.from(document.querySelectorAll('#timeframe-cards-container .timeframe-card[data-timeframe], .left-panel .timeframe-card[data-timeframe]'))
                        .map(n => n.getAttribute('data-timeframe')).filter(Boolean);
                    const tfList = (fromDom.length ? fromDom : (window.nbCoinStatus ? Object.keys(window.nbCoinStatus) : candidates));
                    const hasAny = tfList.length > 0;
                    const heldCount = tfList.reduce((acc, tf) => acc + ((window.nbCoinStatus && window.nbCoinStatus[tf]) ? 1 : 0), 0);
                    allHeld = hasAny && heldCount === tfList.length;
                } catch(_){ allHeld = false; }

                if ((coinsHeld > 0 && liveMaj.includes('BLUE')) || allHeld) {
                    // 위치/색상 유지하며 대기
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    model.circle.setFillStyle(0x88ccff);
                    // 대기 유지 사유 로그(주기적 스팸 방지용 타임스탬프 체크)
                    if (!model._lastStayLogAt || (Date.now() - model._lastStayLogAt) > 5000) {
                        model._lastStayLogAt = Date.now();
                        const why = allHeld ? `all_timeframes_held` : `holding_nbcoins_wait_for_sell (nbCoins=${coinsHeld})`;
                        try {
                            const five = this._fiveW1H('STATE_CHANGE', model, window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y)||'기타영역', currentMajority, why, 'handleSignalWaiting');
                            fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: '신호 대기', to: '신호 대기', reason: why, ts: Date.now(), ...five }) });
                        } catch(_){ }
                        if (window.logManager) window.logManager.addLog(`🔵 신호 대기 유지: ${why}`);
                    }
                }
            } catch(_){ }

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
                    // 코인 보유 또는 전분봉 보유 시에는 계속 신호 대기, 아니면 BTC 탐색 전환
                    const coinsHeld = (typeof nbCoins === 'number' ? nbCoins : 0) || (window.gameInitializer?.gameData?.nbCoins || 0);
                    let allHeld = false;
                    try {
                        const candidates = ['minute1','minute3','minute5','minute10','minute15','minute30','minute60','day'];
                        const fromDom = Array.from(document.querySelectorAll('#timeframe-cards-container .timeframe-card[data-timeframe], .left-panel .timeframe-card[data-timeframe]'))
                            .map(n => n.getAttribute('data-timeframe')).filter(Boolean);
                        const tfList = (fromDom.length ? fromDom : (window.nbCoinStatus ? Object.keys(window.nbCoinStatus) : candidates));
                        const hasAny = tfList.length > 0;
                        const heldCount = tfList.reduce((acc, tf) => acc + ((window.nbCoinStatus && window.nbCoinStatus[tf]) ? 1 : 0), 0);
                        allHeld = hasAny && heldCount === tfList.length;
                    } catch(_){ allHeld = false; }

                    if (coinsHeld === 0 && !allHeld) {
                        model.targetAction = 'BTC 시장 탐색';
                        model.targetX = config.width - 100;
                        model.targetY = config.height - 100;
                        model.circle.setFillStyle(0x0088ff);
                        model.btcExplorationMode = true;
                        model.countdownStarted = false;
                        model.randomMoveTimer = 0;
                        if (window.logManager) window.logManager.addLog(`🔵 트레이너: 신호 대기 후 BTC 시장 탐색으로 이동`);
                        const dialogMessage = `🔵 [탐색으로 전환] BTC 시장 탐색으로 이동 중...`;
                        trainerDialog.setText(dialogMessage);
                    } else {
                        // 계속 센터 대기 유지
                        model.waitCheckTimer = 0;
                        model.countdownStarted = false;
                        if (window.logManager) window.logManager.addLog(`🔵 트레이너: 보유/풀매수 상태 → 신호 대기 계속 유지`);
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
            
            // BTC 시장 감지 검사 (충돌 → 감지)
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
                        window.logManager.addLog(`🔔 트레이너가 BTC 시장을 감지! 학습 모델 기반 처리 시작`);
                    }
                }
                
                if (window.btcMarketLearningHandler) {
                    try { fetch('/api/trainer/activity-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ type:'LEARNING_TRIGGER', majority: currentMajority, zone: 'BTC시장탐색구역', ts: Date.now() }) }); } catch(_) {}
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

            // 매도영역 감지 시 즉시 매도 실행 (ORANGE일 때만, 코인 보유 시)
            const currentZoneAtArrival = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            if (currentZoneAtArrival === '매도영역' && !window.lastSellAction) {
                const coins = (window.gameInitializer?.gameData?.nbCoins ?? window.nbCoins ?? 0);
                const majorityText = (document.getElementById('majority-zone')?.textContent || '').trim();
                if (coins > 0 && typeof majorityText === 'string' && majorityText.includes('ORANGE')) {
                    if (window.logManager) {
                        window.logManager.addLog(`📍 매도영역 감지됨 → 즉시 매도 실행`);
                    }
                    // 다이얼로그 객체 전달
                    const dialogRef = trainerDialog || window.trainerDialog || null;
                    // 현재 다수/수익률 값 확보
                    const buyRateVal = typeof window.buyProfitRate === 'number' ? window.buyProfitRate : (buyProfitRate || 0);
                    const sellRateVal = typeof window.sellProfitRate === 'number' ? window.sellProfitRate : (sellProfitRate || 0);
                    // 목표 액션 보정
                    model.targetAction = '매도';
                    this.handleSellAction(model, config, majorityText, coins, buyRateVal, sellRateVal, dialogRef);
                } else if (coins > 0 && window.logManager) {
                    window.logManager.addLog(`⚠️ 매도영역 감지되었지만 현재 신호가 ORANGE가 아님 → 매도 보류`);
                }
            }
        }
    }

    // N/B 길드 도달 처리
    handleNBGuildArrival(model, config, currentMajority, buyProfitRate, sellProfitRate, trainerDialog) {
        // 매도 전 예상 수익률 계산
        this.calculateSellProfitRate(model, config, trainerDialog);

                    // N/B 길드에 도달했고, ORANGE 신호일 때 코인이 1개 이상이면 즉시 매도
            const coins = (window.gameInitializer?.gameData?.nbCoins ?? window.nbCoins ?? 0);
            const majorityText = (document.getElementById('majority-zone')?.textContent || '').trim() || currentMajority;
            if (!window.lastSellAction && coins > 0 && majorityText.includes('ORANGE')) {
                model.targetAction = '매도';
                if (window.logManager) {
                    window.logManager.addLog(`🏛️ N/B 길드 도달: ORANGE 신호 + 코인 ${coins}개 보유 → 즉시 매도 실행`);
                }
                this.handleSellAction(model, config, majorityText, coins, buyProfitRate || 0, sellProfitRate || 0, trainerDialog);
                return;
            } else if (coins > 0 && !majorityText.includes('ORANGE')) {
                if (window.logManager) {
                    window.logManager.addLog(`⏸️ N/B 길드 도달: 코인 ${coins}개 보유하지만 ORANGE 신호가 아님 → 매도 대기`);
                }
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
        // 디버그 로그 추가 (5초마다만 출력하여 로그 스팸 방지)
        if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
            window.logManager.addLog(`🔍 TrainerStateHandler.updateTrainerMovement 호출됨 - 현재위치: (${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)}), 목표위치: (${model.targetX}, ${model.targetY})`);
        }
        
        // 서버 활동 로그(세부 이동/상태) 수집 (1초 주기)
        try {
            if (!model._lastActivityLogAt || (Date.now() - model._lastActivityLogAt) > 1000) {
                model._lastActivityLogAt = Date.now();
                const payload = {
                    type: 'MOVE',
                    x: Number(model.circle.x.toFixed(1)),
                    y: Number(model.circle.y.toFixed(1)),
                    targetX: Number((model.targetX||0).toFixed(1)),
                    targetY: Number((model.targetY||0).toFixed(1)),
                    speed: this.movementSpeed,
                    action: model.targetAction,
                    zone: window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역',
                    nbCoins: window.gameInitializer?.gameData?.nbCoins || 0,
                    majority: (document.getElementById('majority-zone')?.textContent||'').trim(),
                    ts: Date.now()
                };
                fetch('/api/trainer/activity-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
            }
        } catch(_) {}
        
        const modelX = model.circle.x;
        const modelY = model.circle.y;
        const dx = model.targetX - modelX;
        const dy = model.targetY - modelY;
        
        // 목표 위치가 설정되어 있고 유효한지 확인
        if (typeof model.targetX !== 'undefined' && typeof model.targetY !== 'undefined' && 
            !isNaN(model.targetX) && !isNaN(model.targetY) && 
            model.targetX >= 0 && model.targetY >= 0 && 
            model.targetX <= config.width && model.targetY <= config.height) {
            
            // 거리 계산
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // 도착 판정 거리 (더 정확하게 설정)
            const arrivalThreshold = 3; // 3px 이내면 도착으로 판정
            
            // 목표 위치에 충분히 가까우면 정확히 목표 위치로 이동
            if (distance <= arrivalThreshold) {
                // 정확히 목표 위치로 스냅
                model.circle.x = model.targetX;
                model.circle.y = model.targetY;
                model.name.x = model.circle.x;
                model.name.y = model.circle.y - 6;
                model.role.x = model.circle.x;
                model.role.y = model.circle.y + 6;
                
                // 도착 로그 (한 번만 출력)
                if (!model.arrivalLogged && window.logManager) {
                    window.logManager.addLog(`✅ 트레이너 도착: (${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)}) | 액션: ${model.targetAction}`);
                    model.arrivalLogged = true;
                }
            } else {
                // 이동 중일 때만 로그 출력
                if (model.arrivalLogged) {
                    model.arrivalLogged = false; // 다시 이동 시작
                }
                
                // 최소 이동 거리 설정 (너무 작은 움직임 방지)
                const minMoveDistance = 0.8;
                
                if (distance > minMoveDistance) {
                    // 정규화된 방향 벡터 계산
                    const normalizedDx = dx / distance;
                    const normalizedDy = dy / distance;
                    
                    // 이동 속도 적용 (부드러운 이동을 위해 조정)
                    const adjustedSpeed = Math.min(this.movementSpeed, distance * 0.1); // 거리에 비례한 속도
                    const moveX = normalizedDx * adjustedSpeed;
                    const moveY = normalizedDy * adjustedSpeed;
                    
                    // 위치 업데이트 (안전장치 추가)
                    const newX = model.circle.x + moveX;
                    const newY = model.circle.y + moveY;
                    
                    // 화면 경계 내에 있는지 확인
                    if (newX >= 0 && newX <= config.width && newY >= 0 && newY <= config.height) {
                        model.circle.x = newX;
                        model.circle.y = newY;
                        model.name.x = model.circle.x;
                        model.name.y = model.circle.y - 6;
                        model.role.x = model.circle.x;
                        model.role.y = model.circle.y + 6;
                        
                        // 이동 로그 추가 (거리가 클 때만 출력, 2초마다)
                        if (distance > 15 && (!model.lastMovementLogTime || Date.now() - model.lastMovementLogTime > 2000)) {
                            const currentPos = `(${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)})`;
                            const remainingDistance = Math.round(distance);
                            if (window.logManager) {
                                window.logManager.addLog(`🚶 트레이너 이동 중: ${currentPos} | 남은거리: ${remainingDistance}px | 속도: ${adjustedSpeed.toFixed(2)} | 액션: ${model.targetAction}`);
                            }
                            model.lastMovementLogTime = Date.now();
                        }
                    } else {
                        // 화면 경계를 벗어나면 로그 출력
                        if (window.logManager) {
                            window.logManager.addLog(`⚠️ 트레이너 이동 제한: 화면 경계를 벗어남 (${newX.toFixed(1)}, ${newY.toFixed(1)})`);
                        }
                    }
                }
            }
        } else {
            // 목표 위치가 유효하지 않으면 로그 출력
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 트레이너 이동 중단: 유효하지 않은 목표 위치 (${model.targetX}, ${model.targetY})`);
            }
        }
        
        // 이동 후 실제 위치를 사용하여 거리 계산 및 UI 업데이트
        const actualDistanceToTarget = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
        
        // 목표 지점에 도착했는지 확인
        const arrivalThreshold = 10;
        if (actualDistanceToTarget <= arrivalThreshold && !model.arrivalLogged) {
            model.arrivalLogged = true;
            // 작업 완료 기반 시스템으로 변경됨 - trainer-decision-handler에서 처리
            
            if (window.logManager) {
                window.logManager.addLog(`🎯 트레이너 목표 지점 도착! 구역 작업 시작...`);
            }
        }
        
        // 화면 위치 정보 업데이트
        if (window.trainerPositionInfo) {
            const currentPos = `(${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)})`;
            const targetPos = `(${model.targetX.toFixed(1)}, ${model.targetY.toFixed(1)})`;
            
            // 구역 정보 가져오기
            const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const targetZone = window.gameInitializer?.getCurrentZoneName(model.targetX, model.targetY) || '기타영역';
            
            const positionText = `📍 위치: ${currentPos} (${currentZone}) | 목표: ${targetPos} (${targetZone}) | 거리: ${Math.round(actualDistanceToTarget)}px`;
            window.trainerPositionInfo.setText(positionText);
        }
        
        // 이동 중 대화창 업데이트
        if (actualDistanceToTarget > 30) {
            model.arrivalLogged = false;
            
            const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
                
            const nextDecision = null;
            
            let movingMessage = '';
            {
                // 더 구체적인 메시지 생성
                if (model.targetAction === 'BTC 시장 탐색') {
                    movingMessage = `🔵 [BTC 시장 탐색] BTC 시장 탐색 구역으로 이동 중... (${Math.round(actualDistanceToTarget)}px 남음)`;
                } else if (model.targetAction === '매수') {
                    movingMessage = `💰 [매수 준비] 매수 영역으로 이동 중... (${Math.round(actualDistanceToTarget)}px 남음)`;
                } else if (model.targetAction === '매도') {
                    movingMessage = `📈 [매도 준비] 매도 영역으로 이동 중... (${Math.round(actualDistanceToTarget)}px 남음)`;
                } else {
                    movingMessage = `🎯 [의사결정: ${model.targetAction}] 이동 중... (${Math.round(actualDistanceToTarget)}px 남음)`;
                }
            }
            
            if (window.trainerDialog) {
                window.trainerDialog.setText(movingMessage);
            }
            if (window.logManager && typeof window.logManager.addLog === 'function') {
                window.logManager.addLog(movingMessage);
            }
        }
    }

    // 트레이너 액션 처리 (매수/매도/대기)
    handleTrainerActions(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        
        // 매수 액션 처리 (BLUE에서만 허용) + 구성 가능한 조건 (보유/드랍/임계치)
        if (model.targetAction === '매수' && !window.lastBuyAction) {
            if (currentMajority !== 'BLUE') {
                return;
            }
            
            // 구성 옵션 (전역 토글)
            const cfg = window.TrainerBuyConfig || {};
            const allowBuyWhenHolding = !!cfg.allowBuyWhenHolding; // 기본 false
            const ignoreDropItems = !!cfg.ignoreDropItems; // 기본 false
            
            // N/B 코인과 드랍 아이템 개수 확인
            const currentNbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
            const dropItemsCount = window.gameInitializer?.gameData?.dropItemsCount || 0;
            
            // 보유 코인 제한
            if (!allowBuyWhenHolding && currentNbCoins > 0) {
                if (window.logManager) {
                    window.logManager.addLog(`⛔ 매수 보류: N/B 코인 ${currentNbCoins}개 보유 중 (allowBuyWhenHolding=false)`);
                }
                return;
            }
            // 드랍 아이템 제한
            if (!ignoreDropItems && dropItemsCount > 0) {
                if (window.logManager) {
                    window.logManager.addLog(`⛔ 매수 보류: 드랍 아이템 ${dropItemsCount}개 남음 (ignoreDropItems=false)`);
                }
                return;
            }
            
            // 임계치 결정: 동적 임계치 vs 사용자 오버라이드
            const dynamicThreshold = (window.gameInitializer && window.gameInitializer.gameData && typeof window.gameInitializer.gameData.buyThresholdPercent === 'number')
                ? window.gameInitializer.gameData.buyThresholdPercent
                : 0.5;
            let minThreshold = dynamicThreshold;
            if (typeof cfg.minBuyProfitThreshold === 'number' && !isNaN(cfg.minBuyProfitThreshold)) {
                minThreshold = cfg.minBuyProfitThreshold;
            }
            
            if (typeof buyProfitRate !== 'number' || isNaN(buyProfitRate) || buyProfitRate < minThreshold) {
                if (window.logManager) {
                    window.logManager.addLog(`⛔ 매수 보류: 매수 전 예상 수익률이 임계치 미만 (${(buyProfitRate||0).toFixed(2)}% < ${minThreshold.toFixed(2)}%)`);
                }
                return;
            }
            this.handleBuyAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog);
        }
        // 매도 액션 처리 (ORANGE에서만 허용) + 계산 결과 없으면 금지
        else if (model.targetAction === '매도' && !window.lastSellAction && nbCoins > 0) {
            if (currentMajority !== 'ORANGE') {
                return;
            }
            if (typeof sellProfitRate !== 'number' || isNaN(sellProfitRate)) {
                if (window.logManager) {
                    window.logManager.addLog(`⛔ 매도 보류: 매도 전 예상 수익률 미계산`);
                }
                return;
            }
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
        // 매수 구역이 숨김 상태이면 작동 금지
        if (window.buyPolygon && window.buyPolygon.visible === false) {
            if (window.logManager) {
                window.logManager.addLog(`⛔ 매수 구역 비활성화(숨김) 상태 → 매수 동작 취소`);
            }
            return;
        }
        window.lastBuyAction = true;
        window.lastSellAction = false;
        // 서버 파일 로그 기록 (매수 실행 + 5W1H)
        try {
            const zone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const threshold = (window.gameInitializer?.gameData?.buyThresholdPercent ?? undefined);
            const why = this._composeWhyForBuy(currentMajority, nbCoins, buyProfitRate, threshold, 'action=BUY_EXECUTE');
            const five = this._fiveW1H('BUY_EXECUTE', model, zone, currentMajority, why, 'handleTrainerActions');
            fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'BUY_EXECUTE', majority: currentMajority, nbCoinsBefore: nbCoins, buyProfitRate: buyProfitRate, zone, ts: Date.now(), ...five }) });
        } catch(_){ }
        
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
            
            // 카드 저장소에 매수 기록 추가
            if (window.cardStorageSystem && typeof window.cardStorageSystem.addBuyRecord === 'function') {
                // 현재 선택된 분봉 확인
                let currentTimeframe = null;
                const activeCard = document.querySelector('.timeframe-card.active');
                if (activeCard) {
                    currentTimeframe = activeCard.getAttribute('data-timeframe');
                } else {
                    const selectedCard = document.querySelector('.timeframe-card.selected');
                    if (selectedCard) {
                        currentTimeframe = selectedCard.getAttribute('data-timeframe');
                    }
                }
                
                if (currentTimeframe) {
                    window.cardStorageSystem.addBuyRecord(currentTimeframe, window.buyPrice, buyProfitRate);
                    
                    // N/B 미네랄도 추가 (매수 시 수익률을 미네랄로 추가)
                    if (typeof window.cardStorageSystem.addNBMineral === 'function' && buyProfitRate > 0) {
                        window.cardStorageSystem.addNBMineral(currentTimeframe, buyProfitRate);
                    }
                }
            }
        }
        
        // 매수 완료 후 매수 전 예상 수익률 리셋
        window.buyProfitRate = 0;
        if (window.buyProfitRateDisplay) {
            window.buyProfitRateDisplay.setFill('#00ff88');
            window.buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
        }
        
        // N/B 코인 아이템 생성 (현재 선택된 분봉 정보와 함께)
        if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.dropNBCoin === 'function') {
            // 현재 선택된 분봉 확인
            let currentTimeframe = null;
            
            // 방법 1: 활성화된 분봉 카드에서 확인
            const activeCard = document.querySelector('.timeframe-card.active');
            if (activeCard) {
                currentTimeframe = activeCard.getAttribute('data-timeframe');
            }
            
            // 방법 2: 선택된 분봉 카드에서 확인
            if (!currentTimeframe) {
                const selectedCard = document.querySelector('.timeframe-card.selected');
                if (selectedCard) {
                    currentTimeframe = selectedCard.getAttribute('data-timeframe');
                }
            }
            
            // 방법 3: 기본값 설정
            if (!currentTimeframe) {
                currentTimeframe = '1m'; // 기본값
            }
            
            // 매수 구역 좌표 계산 (매수 구역에서만 드랍)
            const startX = 100; // 매수 구역 X 좌표
            const topY = 50;    // 매수 구역 Y 좌표
            const buyAreaRadius = 30; // 매수 구역 반지름
            
            // 매수 구역 내에서 랜덤 위치 생성
            const angle = Math.random() * Math.PI * 2;
            const distance = Math.random() * buyAreaRadius;
            const dropX = startX + Math.cos(angle) * distance;
            const dropY = topY + Math.sin(angle) * distance;
            
            const createdItem = window.nbCoinDropSystem.dropNBCoin(dropX, dropY, currentTimeframe);
            if (createdItem) {
                if (window.logManager) {
                    window.logManager.addLog(`🪙 매수 완료: N/B 코인 드랍 아이템 생성 → 매수 구역 내 위치 (${Math.round(createdItem.position.x)}, ${Math.round(createdItem.position.y)}), 분봉: ${currentTimeframe}`);
                }
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ 매수 완료: N/B 코인 드랍 아이템 생성 실패 (쿨다운 또는 최대 개수 제한)`);
                }
            }
        }
        
        // UI 업데이트는 NB Coin Drop System에 위임 (충돌 방지)
        if (window.nbCoinDropSystem && window.nbCoinDropSystem.updateNBCoinDisplay) {
            window.nbCoinDropSystem.updateNBCoinDisplay();
        }
        
        // 좌측 패널 분봉 보유 상태 업데이트 (현재 분봉 → held=1)
        try {
            const selected = document.querySelector('.left-panel .timeframe-card.active')
                || document.querySelector('#timeframe-cards-container .timeframe-card.active');
            const tf = selected ? (selected.getAttribute('data-timeframe')||'').trim() : (window.timeframeCards?.getCurrentTimeframe?.() || null);
            if (tf) {
                if (!window.nbCoinStatus || typeof window.nbCoinStatus !== 'object') window.nbCoinStatus = {};
                window.nbCoinStatus[tf] = 1;
                if (window.logManager) window.logManager.addLog(`🪙 분봉 상태 업데이트: ${tf} → held=1`);
                try { fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'TIMEFRAME_HELD_UPDATE', timeframe: tf, held: 1, ts: Date.now(), ...this._fiveW1H('TIMEFRAME_HELD_UPDATE', model, window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y)||'기타영역', currentMajority, 'buy_set_held', 'handleBuyAction') }) }); } catch(_){ }
            }
        } catch(_){ }

        // 대화창 업데이트
        let dialogMessage = '';
        if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
            dialogMessage = window.decisionSystem.generateDecisionLogMessage('매수 완료', currentMajority, nbCoins, window.nbMinerals, window.nbCoinItems, window.currentPriceText, buyProfitRate, sellProfitRate);
        } else {
            dialogMessage = `💰 [매수 완료] N/B 코인: ${nbCoins}개, 미네랄: ${window.nbMinerals ? window.nbMinerals.toFixed(2) : '0.00'}%, 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
        }
        if (trainerDialog && typeof trainerDialog.setText === 'function') {
            trainerDialog.setText(dialogMessage);
        }
        
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
        
        console.log(`💰 매수 완료! N/B 코인 드랍 아이템 생성, 매수가: ${window.currentPriceText}`);
        
        // 매수 완료 후 N/B 길드로 이동 (드랍 아이템 수집을 위해)
        if (!model.postBuyDecisionMade) {
            model.postBuyDecisionMade = true;
            
            // N/B 길드로 이동
            model.targetAction = 'N/B 길드 방문';
            model.targetX = 100; // N/B 길드 위치
            model.targetY = 100;
            model.circle.setFillStyle(0xff8800); // 주황색 (N/B 길드)
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 매수 완료 후 N/B 길드로 이동 → 드랍 아이템 수집 대기`);
            }
        }
        
        // 상태 즉시 저장
        if (window.gameInitializer && window.gameInitializer.saveGameData) {
            window.gameInitializer.saveGameData();
        }
        // 서버 파일 로그 기록 (매수 완료 + 5W1H)
        try {
            const zone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const why = this._composeWhyForBuy(currentMajority, (window.gameInitializer?.gameData?.nbCoins || 0), 0, undefined, 'action=BUY_DONE');
            const five = this._fiveW1H('BUY_DONE', model, zone, currentMajority, why, 'handleBuyAction');
            fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'BUY_DONE', majority: currentMajority, nbCoinsAfter: (window.gameInitializer?.gameData?.nbCoins || 0), ts: Date.now(), ...five }) });
        } catch(_){ }

        // 좌측 패널 현재 선택된 분봉 포함 전 분봉이 모두 매수 완료(보유 상태)인지 확인 → 모두 보유면 신호 대기 센터로 이동
        try {
            const list = document.querySelector('.left-panel .timeframe-card-list') || document.getElementById('timeframe-cards-container');
            const cards = list ? Array.from(list.querySelectorAll('[data-timeframe]')) : [];
            const allHeld = cards.length > 0 && cards.every(el => {
                const tf = (el.getAttribute('data-timeframe')||'').trim();
                const s = (tf && window.nbCoinStatus && typeof window.nbCoinStatus[tf] !== 'undefined') ? window.nbCoinStatus[tf] : 0;
                return !!s; // 1이면 보유
            });
            if (allHeld) {
                const prev = model.targetAction;
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                model.circle.setFillStyle(0x88ccff);
                if (window.logManager) {
                    window.logManager.addLog(`🟦 모든 분봉 매수 보유 확인 → 신호 대기 센터로 이동 (이전: ${prev})`);
                }
                try {
                    const five = this._fiveW1H('STATE_CHANGE', model, window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y)||'기타영역', currentMajority, 'reason=all_timeframes_held', 'postBuyCheck');
                    fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'STATE_CHANGE', from: prev, to: model.targetAction, reason: 'all_timeframes_held', ts: Date.now(), ...five }) });
                } catch(_) {}
            }
        } catch(_) {}
    }

    // 매도 액션 처리
    handleSellAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
        // 매도 구역이 숨김 상태이면 작동 금지
        if (window.sellPolygon && window.sellPolygon.visible === false) {
            if (window.logManager) {
                window.logManager.addLog(`⛔ 매도 구역 비활성화(숨김) 상태 → 매도 동작 취소`);
            }
            return;
        }
        window.lastSellAction = true;
        window.lastBuyAction = false;
        // 서버 파일 로그 기록 (매도 실행 + 5W1H)
        try {
            const zone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const why = this._composeWhyForSell(currentMajority, nbCoins, sellProfitRate, 'action=SELL_EXECUTE');
            const five = this._fiveW1H('SELL_EXECUTE', model, zone, currentMajority, why, 'handleTrainerActions');
            fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'SELL_EXECUTE', majority: currentMajority, nbCoinsBefore: nbCoins, sellProfitRate: sellProfitRate, zone, ts: Date.now(), ...five }) });
        } catch(_){ }
        
        // 플래그 리셋
        model.postBuyDecisionMade = false;
        model.postSellDecisionMade = false;
        model.buyColorLogged = false;
        model.sellColorLogged = false;
        model.nbGuildColorLogged = false;
        
        // 1. 좌측 패널의 N/B COIN 확인
        const leftPanelNbCoins = this.getLeftPanelNbCoins();
        
        // 2. 게임 속 트레이너의 N/B COIN 확인
        const gameNbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
        
        // 3. 둘 다 1개 이상 있을 때만 매도 실행
        if (leftPanelNbCoins <= 0 || gameNbCoins <= 0) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 매도 조건 불만족: 좌측패널 N/B 코인 ${leftPanelNbCoins}개, 게임 N/B 코인 ${gameNbCoins}개`);
            }
            return;
        }
        
        // 4. 좌측 패널 N/B COIN -1
        this.decreaseLeftPanelNbCoins();
        
        // 5. 게임 속 N/B COIN -1
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.nbCoins = gameNbCoins - 1;
            
            if (window.logManager) {
                window.logManager.addLog(`💰 매도 실행: 좌측패널 N/B 코인 ${leftPanelNbCoins}개 → ${leftPanelNbCoins - 1}개, 게임 N/B 코인 ${gameNbCoins}개 → ${gameNbCoins - 1}개`);
            }
        }
        
        // 좌측 패널 분봉 보유 상태 업데이트 (현재/최근 분봉 → held=0)
        try {
            let tf = null;
            const selected = document.querySelector('.left-panel .timeframe-card.active')
                || document.querySelector('#timeframe-cards-container .timeframe-card.active');
            tf = selected ? (selected.getAttribute('data-timeframe')||'').trim() : (window.timeframeCards?.getCurrentTimeframe?.() || null);
            if (!tf && window.lastBoughtTimeframe) tf = window.lastBoughtTimeframe;
            if (tf) {
                if (!window.nbCoinStatus || typeof window.nbCoinStatus !== 'object') window.nbCoinStatus = {};
                window.nbCoinStatus[tf] = 0;
                if (window.logManager) window.logManager.addLog(`🪙 분봉 상태 업데이트: ${tf} → held=0`);
                try { fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'TIMEFRAME_HELD_UPDATE', timeframe: tf, held: 0, ts: Date.now(), ...this._fiveW1H('TIMEFRAME_HELD_UPDATE', model, window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y)||'기타영역', currentMajority, 'sell_clear_held', 'handleSellAction') }) }); } catch(_){ }
            }
        } catch(_){ }

        // UI 갱신: N/B 코인 표시 업데이트
        if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.updateNBCoinDisplay === 'function') {
            window.nbCoinDropSystem.updateNBCoinDisplay();
        }
        
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
        
        // 수익률 검증 및 평균 갱신 (중앙 gameData 기반)
        if (!isNaN(currentPnl)) {
            if (window.gameInitializer && window.gameInitializer.gameData) {
                const sumPrev = window.gameInitializer.gameData.nbMineralsSum || 0;
                const cntPrev = window.gameInitializer.gameData.nbMineralsCount || 0;
                const sumNew = sumPrev + currentPnl;
                const cntNew = cntPrev + 1;
                window.gameInitializer.gameData.nbMineralsSum = sumNew;
                window.gameInitializer.gameData.nbMineralsCount = cntNew;
                window.gameInitializer.gameData.nbMinerals = sumNew / cntNew;

                if (window.nbMineralDisplay && typeof window.nbMineralDisplay.setText === 'function') {
                    window.nbMineralDisplay.setText(`N/B 미네랄(평균): ${window.gameInitializer.gameData.nbMinerals.toFixed(2)}%`);
                }

                if (window.logManager) {
                    window.logManager.addLog(`💰 매도 완료: N/B 미네랄 평균 갱신 +${currentPnl.toFixed(2)}% → 평균 ${window.gameInitializer.gameData.nbMinerals.toFixed(2)}% (n=${cntNew})`);
                }
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 매도 완료: 수익률 추출 실패 - N/B 미네랄 누적 안됨`);
            }
        }
        
        // 수익률 평균 집계 및 표시 업데이트
        if (!window.pnlStats) {
            window.pnlStats = { sum: 0, count: 0, avg: 0 };
        }
        if (!isNaN(currentPnl)) {
            window.pnlStats.sum += currentPnl;
            window.pnlStats.count += 1;
            window.pnlStats.avg = window.pnlStats.sum / window.pnlStats.count;
        }
        const pnlAvgElement = document.getElementById('selected-coin-pnl');
        if (pnlAvgElement) {
            const avgText = `수익율 평균: ${isFinite(window.pnlStats.avg) ? window.pnlStats.avg.toFixed(2) : '0.00'}%`;
            pnlAvgElement.textContent = avgText;
            pnlAvgElement.style.color = (window.pnlStats.avg >= 0) ? 'lime' : 'red';
        }
        if (window.logManager) {
            window.logManager.addLog(`📈 수익률 평균 업데이트: ${isFinite(window.pnlStats.avg) ? window.pnlStats.avg.toFixed(2) : '0.00'}% (거래 수: ${window.pnlStats.count})`);
        }

        // N/B 미네랄 변경됨 (game-state-manager.js에서 자동 저장됨)
        
        // 매도 시 수익률 리셋
        window.buyPrice = 0;
        window.buyProfitRate = 0;
        window.sellProfitRate = 0;
        
        if (window.buyProfitRateDisplay && typeof window.buyProfitRateDisplay.setText === 'function') {
            window.buyProfitRateDisplay.setFill('#00ff88');
            window.buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
        }
        if (window.sellProfitRateDisplay && typeof window.sellProfitRateDisplay.setText === 'function') {
            window.sellProfitRateDisplay.setFill('#ff0088');
            window.sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
        }
        
        // UI 업데이트는 NB Coin Drop System에 위임 (충돌 방지)
        if (window.nbCoinDropSystem && window.nbCoinDropSystem.updateNBCoinDisplay) {
            window.nbCoinDropSystem.updateNBCoinDisplay();
        }
        if (window.nbMineralDisplay && typeof window.nbMineralDisplay.setText === 'function') {
            window.nbMineralDisplay.setText(`N/B 미네랄: ${window.nbMinerals.toFixed(2)}%`);
        }
        
        // 대화창 업데이트
        let dialogMessage = '';
        const currentNbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
        if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
            dialogMessage = window.decisionSystem.generateDecisionLogMessage('매도 완료', currentMajority, currentNbCoins, window.nbMinerals, window.nbCoinItems, window.currentPriceText, buyProfitRate, sellProfitRate);
        } else {
            dialogMessage = `💸 [매도 완료] N/B 코인: ${currentNbCoins}개, 미네랄: ${window.nbMinerals.toFixed(2)}%, 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
        }
        if (trainerDialog && typeof trainerDialog.setText === 'function') {
            trainerDialog.setText(dialogMessage);
        }
        
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
        
        console.log(`💸 매도 완료! N/B 코인: ${window.nbCoins}개, N/B 미네랄 누적: ${window.nbMinerals.toFixed(2)}%, 수익률 리셋`);
        
        // 매도 완료 후 상태 요약 로그
        if (window.logManager) {
            const summaryLog = `📊 매도 완료 후 상태 요약: N/B 코인 ${window.nbCoins}개, N/B 미네랄 ${window.nbMinerals.toFixed(2)}%, 드랍 아이템 ${window.nbCoinItems ? window.nbCoinItems.length : 0}개`;
            window.logManager.addLog(summaryLog);
        }
        
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
        if (window.gameInitializer && window.gameInitializer.saveGameData) {
            window.gameInitializer.saveGameData();
        }
        // 서버 파일 로그 기록 (매도 완료 + 5W1H)
        try {
            const zone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const why = this._composeWhyForSell(currentMajority, (window.gameInitializer?.gameData?.nbCoins || 0), sellProfitRate, 'action=SELL_DONE');
            const five = this._fiveW1H('SELL_DONE', model, zone, currentMajority, why, 'handleSellAction');
            fetch('/api/trainer/decision-log', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ level:'info', action:'SELL_DONE', majority: currentMajority, nbCoinsAfter: (window.gameInitializer?.gameData?.nbCoins || 0), ts: Date.now(), ...five }) });
        } catch(_){ }
        
        // 매도 완료 후 매도 전 예상 수익률 리셋
        window.sellProfitRate = 0;
        if (window.sellProfitRateDisplay) {
            window.sellProfitRateDisplay.setFill('#ff0088');
            window.sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
        }
        
        // 카드 저장소에 매도 기록 추가
        if (window.cardStorageSystem && typeof window.cardStorageSystem.addSellRecord === 'function') {
            // 현재 선택된 분봉 확인
            let currentTimeframe = null;
            const activeCard = document.querySelector('.timeframe-card.active');
            if (activeCard) {
                currentTimeframe = activeCard.getAttribute('data-timeframe');
            } else {
                const selectedCard = document.querySelector('.timeframe-card.selected');
                if (selectedCard) {
                    currentTimeframe = selectedCard.getAttribute('data-timeframe');
                }
            }
            
            if (currentTimeframe) {
                // 현재 가격 가져오기
                let currentPrice = 0;
                if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                    currentPrice = window.currentPriceManager.parseCurrentPrice();
                }
                
                // 수익률 계산
                let profitRate = 0;
                if (window.buyPrice > 0 && currentPrice > 0) {
                    profitRate = ((currentPrice - window.buyPrice) / window.buyPrice) * 100;
                }
                
                window.cardStorageSystem.addSellRecord(currentTimeframe, currentPrice, profitRate);
                
                // N/B 미네랄도 추가 (매도 시 수익률을 미네랄로 추가)
                if (typeof window.cardStorageSystem.addNBMineral === 'function' && profitRate > 0) {
                    window.cardStorageSystem.addNBMineral(currentTimeframe, profitRate);
                }
            }
        }
        
        // 매도 기록 저장 (실제 매도가 발생했을 때만)
        if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
            const sellPrice = window.currentPriceManager.parseCurrentPrice();
            console.log(`💰 매도가격 저장: ₩${sellPrice.toLocaleString()}`);
            
            // 카드 저장소에 매도 기록 추가
            if (window.cardStorageSystem && typeof window.cardStorageSystem.addSellRecord === 'function') {
                // 현재 선택된 분봉 확인
                let currentTimeframe = null;
                const activeCard = document.querySelector('.timeframe-card.active');
                if (activeCard) {
                    currentTimeframe = activeCard.getAttribute('data-timeframe');
                } else {
                    const selectedCard = document.querySelector('.timeframe-card.selected');
                    if (selectedCard) {
                        currentTimeframe = selectedCard.getAttribute('data-timeframe');
                    }
                }
                
                if (currentTimeframe) {
                    // 실제 매도 수익률 계산 (매수가 대비 매도가)
                    let actualSellProfitRate = 0;
                    if (window.buyPrice && window.buyPrice > 0) {
                        actualSellProfitRate = ((sellPrice - window.buyPrice) / window.buyPrice) * 100;
                    } else {
                        // 매수가가 없으면 현재 수익률 사용
                        actualSellProfitRate = currentPnl;
                    }
                    
                    window.cardStorageSystem.addSellRecord(currentTimeframe, sellPrice, actualSellProfitRate);
                    
                    // N/B 미네랄도 추가 (매도 시 실제 수익률을 미네랄로 추가)
                    if (typeof window.cardStorageSystem.addNBMineral === 'function' && actualSellProfitRate !== 0) {
                        window.cardStorageSystem.addNBMineral(currentTimeframe, actualSellProfitRate);
                    }
                    
                    if (window.logManager) {
                        window.logManager.addLog(`📈 카드 ${currentTimeframe} 매도 기록 저장: ₩${sellPrice.toLocaleString()}, 실제 수익률: ${actualSellProfitRate.toFixed(2)}%`);
                    }
                }
            }
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
        if (trainerDialog && typeof trainerDialog.setText === 'function') {
            trainerDialog.setText(dialogMessage);
        }
        
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
            if (trainerDialog && typeof trainerDialog.setText === 'function') {
                trainerDialog.setText(dialogMessage);
            }
            
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
            if (trainerDialog && typeof trainerDialog.setText === 'function') {
                trainerDialog.setText(dialogMessage);
            }
            
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        }
    }
    
    // 좌측 패널의 N/B COIN 개수 가져오기
    getLeftPanelNbCoins() {
        try {
            // 좌측 패널에서 N/B 코인 개수 확인
            const nbCoinElement = document.getElementById('nb-coin-count') || document.getElementById('nb-coin-display');
            if (nbCoinElement) {
                const text = nbCoinElement.textContent || '';
                const match = text.match(/(\d+)/);
                return match ? parseInt(match[1]) : 0;
            }
            
            // 대안: N/B 코인 드롭 시스템에서 확인
            if (window.gameInitializer?.gameData?.nbCoins !== undefined) {
                return window.gameInitializer.gameData.nbCoins;
            }
            
            return 0;
        } catch (e) {
            return 0;
        }
    }
    
    // 좌측 패널 N/B COIN 감소
    decreaseLeftPanelNbCoins() {
        try {
            const nbCoinElement = document.getElementById('nb-coin-count') || document.getElementById('nb-coin-display');
            if (nbCoinElement) {
                const currentText = nbCoinElement.textContent || '';
                const match = currentText.match(/(\d+)/);
                if (match) {
                    const currentCount = parseInt(match[1]);
                    const newCount = Math.max(0, currentCount - 1);
                    const newText = currentText.replace(/\d+/, newCount.toString());
                    nbCoinElement.textContent = newText;
                }
            }
        } catch (e) {
            // 좌측 패널 업데이트 실패 시 무시
        }
    }
}

// 전역 인스턴스 생성
window.trainerStateHandler = new TrainerStateHandler();
