// 신호 대기 관리자 모듈
// 신호 대기 센터에서의 대기, 타이머, BTC 시장 탐색 시작을 담당

class SignalWaitingManager {
    constructor() {
        this.waitTime = 5; // 5초 대기
        this.centerThreshold = 30; // 센터 도달 임계값
        this.resetThreshold = 50; // 타이머 리셋 임계값
    }

    // 신호 대기 상태 업데이트
    updateSignalWaiting(model, config, trainerDialog, currentMajority, nbCoins) {
        // 디버깅: targetAction 상태 확인 (10초마다)
        if (Math.floor(Date.now() / 1000) % 10 === 0) {
            console.log('🔍 DEBUG: 현재 targetAction 상태:', {
                targetAction: model.targetAction,
                modelTargetAction: model.targetAction,
                elapsedSeconds: model.waitStartTime ? Math.round((Date.now() - model.waitStartTime) / 100) / 10 : 0
            });
        }

        // 신호 대기 센터에 도달했을 때만 조건 확인
        const distanceToCenter = Math.sqrt(
            (model.circle.x - (config.width / 2)) ** 2 + 
            (model.circle.y - (config.height / 2)) ** 2
        );
        
        // 디버깅: 거리 확인 (10초마다, 0px일 때는 출력하지 않음)
        if (model.isTrainer && Math.floor(Date.now() / 1000) % 10 === 0 && distanceToCenter > 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔍 신호 대기 센터 거리: ${Math.round(distanceToCenter)}px, 조건: ${distanceToCenter < this.centerThreshold}`);
            }
        }
        
        if (distanceToCenter < this.centerThreshold) {
            return this.handleCenterArrival(model, config, trainerDialog, currentMajority, nbCoins);
        } else {
            return this.handleOutsideCenter(model, distanceToCenter);
        }
    }

    // 센터 도달 처리
    handleCenterArrival(model, config, trainerDialog, currentMajority, nbCoins) {
        // 신호 대기 센터에 도달했을 때 주기적으로 조건 확인
        if (!model.waitCheckTimer) {
            model.waitCheckTimer = 0;
            model.waitStartTime = Date.now(); // 시작 시간 기록
        }
        model.waitCheckTimer++;
        
        // 카운트다운이 시작되었음을 표시
        model.countdownStarted = true;
        
        // 경과 시간 계산
        const elapsedSeconds = (Date.now() - model.waitStartTime) / 1000;
        
        // 디버깅: 센터 도달 확인
        if (model.isTrainer && model.waitCheckTimer === 1) {
            if (window.logManager) {
                window.logManager.addLog(`🎯 트레이너가 신호 대기 센터에 도달! 카운트다운 시작`);
            }
        }
        
        // 디버깅: 타이머 값 확인 (1초마다)
        if (model.isTrainer && Math.floor(elapsedSeconds) % 1 === 0 && elapsedSeconds > 0 && elapsedSeconds < this.waitTime) {
            if (window.logManager) {
                window.logManager.addLog(`🔍 타이머 디버깅: elapsedSeconds = ${elapsedSeconds.toFixed(1)}, countdownStarted = ${model.countdownStarted}, remainingSeconds = ${Math.ceil(this.waitTime - elapsedSeconds)}`);
            }
        }
        
        // 5초 이상 머무르면 BTC 시장으로 이동
        if (elapsedSeconds >= this.waitTime) {
            return this.startBTCExploration(model, config, trainerDialog, currentMajority, nbCoins);
        } else {
            // 카운트다운 표시
            return this.showCountdown(model, config, trainerDialog, currentMajority, nbCoins, elapsedSeconds);
        }
    }

    // 센터 외부 처리
    handleOutsideCenter(model, distanceToCenter) {
        // 신호 대기 센터에서 멀리 떨어져 있을 때만 타이머 리셋 (50px 이상)
        // 단, 카운트다운이 시작되지 않았을 때만 리셋
        if (distanceToCenter > this.resetThreshold && !model.countdownStarted) {
            model.waitCheckTimer = 0;
        }
        
        // 디버깅: 센터에서 멀리 떨어져 있을 때
        if (model.isTrainer && Math.floor(Date.now() / 1000) % 3 === 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔍 트레이너가 신호 대기 센터에서 멀리 떨어짐: ${Math.round(distanceToCenter)}px, countdownStarted: ${model.countdownStarted}`);
            }
        }
        
        return false; // 아직 대기 중
    }

    // BTC 시장 탐색 시작
    startBTCExploration(model, config, trainerDialog, currentMajority, nbCoins) {
        console.log('🔍 DEBUG: 5초 이상 대기 조건 만족, BTC 시장 탐색으로 이동 시작');
        
        // BTC 시장 학습 핸들러 사용
        if (window.btcMarketLearningHandler) {
            // BTC 시장 탐색 모드 설정
            model.targetAction = 'BTC 시장 탐색';
            model.targetX = config.width - 100;
            model.targetY = config.height - 100;
            model.circle.setFillStyle(0x0088ff);
            model.btcExplorationMode = true;
            model.countdownStarted = false;
        } else {
            // 폴백: 직접 설정
            model.targetAction = 'BTC 시장 탐색';
            model.targetX = config.width - 100;
            model.targetY = config.height - 100;
            model.circle.setFillStyle(0x0088ff);
            model.btcExplorationMode = true;
            model.countdownStarted = false;
            
            if (window.logManager) {
                window.logManager.addLog(`🔵 트레이너: 신호 대기 센터에서 5초 이상 대기 → BTC 시장 탐색으로 이동! targetAction: ${model.targetAction}, targetX: ${model.targetX}, targetY: ${model.targetY}`);
            }
            
            const dialogMessage = `🔵 [장시간 대기] BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정`;
            trainerDialog.setText(dialogMessage);
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        }
        
        return true; // BTC 시장 탐색 시작됨
    }

    // 카운트다운 표시
    showCountdown(model, config, trainerDialog, currentMajority, nbCoins, elapsedSeconds) {
        const remainingSeconds = Math.ceil(this.waitTime - elapsedSeconds);
        
        if (remainingSeconds <= 0) {
            const dialogMessage = `🔵 [신호 대기] BTC 시장 탐색으로 이동 중... N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
            trainerDialog.setText(dialogMessage);
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        } else {
            const dialogMessage = `🔵 [신호 대기] 센터에서 대기 중... (${remainingSeconds}초 후 BTC 시장 탐색으로 이동) N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
            trainerDialog.setText(dialogMessage);
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
        }
        
        return false; // 아직 대기 중
    }

    // 대기 상태 초기화
    resetWaitingState(model) {
        model.waitCheckTimer = 0;
        model.waitStartTime = null;
        model.countdownStarted = false;
    }

    // 대기 시간 설정
    setWaitTime(seconds) {
        this.waitTime = seconds;
    }

    // 센터 임계값 설정
    setCenterThreshold(pixels) {
        this.centerThreshold = pixels;
    }

    // 리셋 임계값 설정
    setResetThreshold(pixels) {
        this.resetThreshold = pixels;
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.signalWaitingManager = new SignalWaitingManager();
}

// 모듈 로딩 완료
