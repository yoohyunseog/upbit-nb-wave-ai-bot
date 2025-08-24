// 트레이너 의사결정 핸들러 모듈
// 트레이너의 의사결정, 목표 설정, 상태 관리를 담당

class TrainerDecisionHandler {
    constructor() {
        this.waitCheckTimer = 0;
        this.waitStartTime = null;
        this.countdownStarted = false;
        this.btcExplorationMode = false;
        this.arrivalLogged = false;
        this.btcMarketArrivalLogged = false; // BTC 시장 도착 로그 플래그
        
        // 목표 지점 도착 후 대기 시간 관리 (기존 5초 대기 시스템)
        this.arrivalTime = null;
        this.waitAfterArrival = 5000; // 5초 대기 (밀리초)
        this.arrivalWaitComplete = false;
        
        // 새로운 작업 완료 기반 의사결정 시스템
        this.taskCompletionSystem = {
            currentZone: null,
            taskCompleted: false,
            taskStartTime: null,
            taskTimeout: 10000, // 10초 타임아웃 (밀리초)
            zoneTasks: {
                '매수영역': {
                    name: 'N/B 코인 드랍',
                    completed: false,
                    handler: 'handleBuyAreaTask'
                },
                '매도영역': {
                    name: '매도 액션 처리',
                    completed: false,
                    handler: 'handleSellAreaTask'
                },
                'BTC시장': {
                    name: '매수 전 예상 수익률 계산',
                    completed: false,
                    handler: 'handleBTCMarketTask'
                },
                'N/B길드': {
                    name: '매도 전 예상 수익률 계산',
                    completed: false,
                    handler: 'handleNBGuildTask'
                },
                '신호대기센터': {
                    name: '신호 대기',
                    completed: false,
                    handler: 'handleSignalCenterTask'
                }
            }
        };
        
        // 한 자리에 머무르는 시간 추적을 위한 새로운 속성들
        this.idleStartTime = null;
        this.lastPosition = null;
        this.idleThreshold = 5000; // 5초 (밀리초)
        this.isIdle = false;
        this.idleActionTriggered = false;
        
        // 수익률 계산 로그 누적 저장용
        this.buyProfitLogs = [];
        this.sellProfitLogs = [];
        
        // 3초 간격 저장을 위한 타이머
        this.lastSaveTime = 0; // 마지막 저장 시간
        this.saveInterval = 3000; // 저장 간격 (3초)
        
        // 수익률 계산 로그 파일 초기화
        this.initProfitRateLogs();
        
        // 전역 함수 추가
        TrainerDecisionHandler.addGlobalFunctions();
    }
    
    // 수익률 계산 로그 파일 초기화
    initProfitRateLogs() {
        const today = new Date().toISOString().split('T')[0];
        this.buyProfitLogFile = `log/trainer/buy-profit-rate-logs-${today}.txt`;
        this.sellProfitLogFile = `log/trainer/sell-profit-rate-logs-${today}.txt`;
        
        //console.log(`📊 수익률 계산 로그 파일 초기화: ${this.buyProfitLogFile}, ${this.sellProfitLogFile}`);
        
        // 로그 매니저에 수익률 계산 로그 파일 정보 저장
        if (window.logManager) {
            window.logManager.addLog(`📊 수익률 계산 로그 파일 생성: ${this.buyProfitLogFile}, ${this.sellProfitLogFile}`);
        }
    }
    
    // 수익률 계산 로그 작성 (파일 저장)
    writeProfitRateLog(filename, content) {
        const timestamp = `[${new Date().toLocaleTimeString()}] `;
        const logEntry = timestamp + content + '\n';
        
        // 브라우저에서는 //console.log로 출력
        //console.log(`📊 ${filename}: ${timestamp}${content}`);
        
        // 전역 로그 매니저에도 기록
        if (window.logManager) {
            window.logManager.addLog(`📊 ${filename}: ${content}`);
        }
        
        // 로그 누적 저장 (100줄 제한)
        if (filename.includes('buy-profit-rate')) {
            this.buyProfitLogs.push(logEntry);
            
            // 100줄 제한: 초과하면 가장 오래된 로그부터 제거
            if (this.buyProfitLogs.length > 100) {
                const removedCount = this.buyProfitLogs.length - 100;
                this.buyProfitLogs.splice(0, removedCount);
                if (window.logManager) {
                    window.logManager.addLog(`📊 매수 수익률 로그 100줄 제한: ${removedCount}개 오래된 로그 제거됨`);
                }
            }
            
            // 3초 간격으로 파일에 저장 (과도한 저장 방지)
            const currentTime = Date.now();
            if (currentTime - this.lastSaveTime > this.saveInterval) {
                this.saveLogToFile(this.buyProfitLogFile, this.buyProfitLogs);
                this.lastSaveTime = currentTime;
            }
        } else if (filename.includes('sell-profit-rate')) {
            this.sellProfitLogs.push(logEntry);
            
            // 100줄 제한: 초과하면 가장 오래된 로그부터 제거
            if (this.sellProfitLogs.length > 100) {
                const removedCount = this.sellProfitLogs.length - 100;
                this.sellProfitLogs.splice(0, removedCount);
                if (window.logManager) {
                    window.logManager.addLog(`📊 매도 수익률 로그 100줄 제한: ${removedCount}개 오래된 로그 제거됨`);
                }
            }
            
            // 3초 간격으로 파일에 저장 (과도한 저장 방지)
            const currentTime = Date.now();
            if (currentTime - this.lastSaveTime > this.saveInterval) {
                this.saveLogToFile(this.sellProfitLogFile, this.sellProfitLogs);
                this.lastSaveTime = currentTime;
            }
        }
    }
    
    // 로그 파일 자동 저장 (실제 파일 시스템에 저장)
    saveLogToFile(filename, logs) {
        const content = logs.join('');
        
        // 실제 파일 시스템에 저장
        this.saveToServer(filename, content, logs.length);
        
        // 백업으로 localStorage에도 저장
        try {
            this.saveToLocalStorage(filename, content, logs.length);
        } catch (error) {
            console.error(`❌ localStorage 저장 실패: ${error.message}`);
        }
    }
    
    // 서버에 실제 파일로 저장
    saveToServer(filename, content, logCount) {
        if (typeof fetch !== 'undefined') {
            fetch('/save-log-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: filename,
                    content: content,
                    logCount: logCount
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    //console.log(`💾 ${filename} 실제 파일 저장 완료 (${logCount}개 로그) - 파일 경로: ${data.filepath}`);
                    
                    if (window.logManager) {
                        window.logManager.addLog(`💾 수익률 로그 실제 파일 저장: ${filename} (${logCount}개 로그) - 파일 경로: ${data.filepath}`);
                    }
                } else {
                    console.error(`❌ 실제 파일 저장 실패: ${data.error}`);
                    if (window.logManager) {
                        window.logManager.addLog(`❌ 실제 파일 저장 실패: ${filename} - ${data.error}`);
                    }
                }
            })
            .catch(error => {
                console.error(`❌ 실제 파일 저장 요청 실패: ${error.message}`);
                if (window.logManager) {
                    window.logManager.addLog(`❌ 실제 파일 저장 요청 실패: ${filename} - ${error.message}`);
                }
            });
        } else {
            console.error(`❌ fetch API를 사용할 수 없음 - localStorage에만 저장됨`);
        }
    }
    
    // localStorage에 저장 (폴백)
    saveToLocalStorage(filename, content, logCount) {
        const storageKey = `profit_log_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
        localStorage.setItem(storageKey, content);
        //console.log(`💾 ${filename} localStorage 저장 완료 (${logCount}개 로그) - localStorage: ${storageKey}`);
        
        // 로그 매니저에도 저장 정보 기록
        if (window.logManager) {
            window.logManager.addLog(`💾 수익률 로그 localStorage 저장: ${filename} (${logCount}개 로그) - 키: ${storageKey}`);
        }
    }
    
    // localStorage에서 로그 확인
    getLogFromLocalStorage(filename) {
        const storageKey = `profit_log_${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;
        const content = localStorage.getItem(storageKey);
        if (content) {
            //console.log(`📊 ${filename} localStorage에서 로그 확인: ${content.split('\n').length}개 로그`);
            return content;
        } else {
            //console.log(`📊 ${filename} localStorage에 로그 없음`);
            return null;
        }
    }
    
    // 모든 수익률 로그 확인
    checkAllProfitLogs() {
        //console.log('📊 모든 수익률 로그 확인:');
        
        // 매수 수익률 로그
        const buyLogContent = this.getLogFromLocalStorage(this.buyProfitLogFile);
        if (buyLogContent) {
            //console.log(`📊 매수 수익률 로그 (${this.buyProfitLogFile}):`);
            //console.log(buyLogContent);
        }
        
        // 매도 수익률 로그
        const sellLogContent = this.getLogFromLocalStorage(this.sellProfitLogFile);
        if (sellLogContent) {
            //console.log(`📊 매도 수익률 로그 (${this.sellProfitLogFile}):`);
            //console.log(sellLogContent);
        }
        
        // 메모리 내 로그 상태
        //console.log(`📊 메모리 내 로그 상태: 매수 ${this.buyProfitLogs.length}개, 매도 ${this.sellProfitLogs.length}개`);
    }

    // 한 자리에 머무르는 시간 추적 및 처리 (비활성화됨)
    handleIdleDetection(model, config) {
        // Idle detection 기능이 제거되었습니다.
        // 트레이너는 각 구역에서 자유롭게 계속 작업할 수 있습니다.
        return null; // 특별한 액션이 필요하지 않음
    }

    // 아이들 타이머 리셋 (트레이너가 새로운 목표로 이동할 때 호출)
    resetIdleTimer() {
        this.idleStartTime = null;
        this.isIdle = false;
        this.idleActionTriggered = false;
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 트레이너 아이들 타이머 리셋됨`);
        }
    }

    // 트레이너 의사결정 처리
    handleTrainerDecision(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing) {
        // 순서 기반 의사결정 파이프라인 적용
        try {
            const liveMajorityEl = document.getElementById('majority-zone');
            const liveMajority = (liveMajorityEl ? (liveMajorityEl.textContent || '').trim() : currentMajority || '').toUpperCase();
            const getZone = (x, y) => (window.gameInitializer?.getCurrentZoneName(x, y) || '기타영역');
            const currentZone = getZone(model.circle.x, model.circle.y);
            const isPaper = false;

            // 좌측 패널 현재 분봉의 N/B COIN 상태를 항상 우선 사용 (0: 매수만, 1: 매도만)
            const getSelectedTimeframe = () => {
                try {
                    const list = document.querySelector('.left-panel .timeframe-card-list') || document.getElementById('timeframe-cards-container');
                    const sel = list && list.querySelector('.selected');
                    return sel ? (sel.getAttribute('data-timeframe') || '').trim() : null;
                } catch (_) { return null; }
            };
            const tfUi = getSelectedTimeframe();
            let leftPanelNbStatus = 0;
            if (tfUi && window.nbCoinStatus && typeof window.nbCoinStatus[tfUi] !== 'undefined') {
                leftPanelNbStatus = window.nbCoinStatus[tfUi] ? 1 : 0;
            } else {
                // Fallback: 중앙 gameData 보유 상태로 추정
                leftPanelNbStatus = (window.gameInitializer?.gameData?.nbCoins || 0) > 0 ? 1 : 0;
            }

            // BLUE 파이프라인: (수익률 계산 완료 && 0이 아님) → 매수 우선, 그 외엔 BTC 시장 탐색
            if (liveMajority.includes('BLUE')) {
                // 좌측 패널 상태가 1이면 (보유중) 매수 금지 → 매도 대기: BTC 시장 탐색으로 이동
                if (leftPanelNbStatus === 1) {
                    model.targetAction = 'BTC 시장 탐색';
                    model.targetX = config.width - 100;
                    model.targetY = config.height - 100;
                    return model.targetAction;
                }
                // 실제 조건: 수익률 요건 충족 필요
                // 1순위: 유효한 매수 수익률이 있으면 위치와 무관하게 매수 진행
                if (typeof buyProfitRate === 'number' && !isNaN(buyProfitRate) && buyProfitRate !== 0) {
                    model.targetAction = '매수';
                    model.targetX = startX;
                    model.targetY = topY;
                    return model.targetAction;
                }
                // 2순위: 아직 수익률이 없으면 BTC 시장 탐색 구역으로 이동해 계산
                if (currentZone !== 'BTC시장탐색구역') {
                    model.targetAction = 'BTC 시장 탐색';
                    model.targetX = config.width - 100;
                    model.targetY = config.height - 100;
                    return model.targetAction;
                }
                // 3순위: 구역 안인데 수익률이 미계산/0이면 보류
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                return model.targetAction;
            }

            // ORANGE 파이프라인: (코인>0 && 수익률 계산 완료) → 매도 우선, 그 외엔 N/B 길드 방문
            if (liveMajority.includes('ORANGE')) {
                // 좌측 패널 상태가 0이면 (무보유) 매도 금지 → 매수 대기: N/B 길드로 이동
                if (leftPanelNbStatus === 0) {
                    if (currentZone !== 'N/B길드') {
                        model.targetAction = 'N/B 길드 방문';
                        model.targetX = 100;
                        model.targetY = 100;
                        return model.targetAction;
                    }
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    return model.targetAction;
                }
                // 실제 조건: 수익률 요건 충족 필요
                // 1순위: 유효한 매도 수익률과 코인이 있으면 위치와 무관하게 매도 진행
                if (nbCoins > 0 && typeof sellProfitRate === 'number' && !isNaN(sellProfitRate)) {
                    model.targetAction = '매도';
                    model.targetX = startX + spacing;
                    model.targetY = topY;
                    return model.targetAction;
                }
                // 2순위: 계산/코인이 준비되지 않았으면 N/B 길드로 이동해 계산
                if (currentZone !== 'N/B길드') {
                    model.targetAction = 'N/B 길드 방문';
                    model.targetX = 100;
                    model.targetY = 100;
                    return model.targetAction;
                }
                // 3순위: 구역 안인데 수익률 미계산이면 보류
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                return model.targetAction;
            }
        } catch (e) {
            // 안전망: 기존 로직으로 폴백
        }
        const modelX = model.circle.x;
        const modelY = model.circle.y;
        
        // 목표 지점까지의 거리 계산
        const distanceToTarget = Math.sqrt((model.targetX - modelX) ** 2 + (model.targetY - modelY) ** 2);
        const arrivalThreshold = 10; // 도착 판정 거리
        
        // 현재 위치와 목표 정보 로그
        if (window.logManager) {
            const currentPos = `(${Math.round(modelX)}, ${Math.round(modelY)})`;
            const targetPos = `(${Math.round(model.targetX)}, ${Math.round(model.targetY)})`;
            const currentZone = this.getCurrentZone(modelX, modelY, startX, topY, spacing, config);
            const targetZone = this.getCurrentZone(model.targetX, model.targetY, startX, topY, spacing, config);
            window.logManager.addLog(`🎯 트레이너 의사결정: 현재위치 ${currentPos} (${currentZone}) | 목표위치 ${targetPos} (${targetZone}) | 목표까지거리 ${Math.round(distanceToTarget)}px`);
        }
        
        // 목표 지점에 도착하지 않았으면 기존 의사결정 유지
        if (distanceToTarget > arrivalThreshold) {
            if (window.logManager) {
                window.logManager.addLog(`⏳ 목표 지점 도착 대기 중... (거리: ${Math.round(distanceToTarget)}px) - 기존 의사결정 유지`);
            }
            
            // 기존 의사결정이 있으면 그대로 반환
            if (model.targetAction) {
                return model.targetAction;
            }
        } else {
            // 목표 지점에 도착했을 때 - 작업 완료 기반 시스템 사용
            const currentZone = this.getCurrentZone(modelX, modelY, startX, topY, spacing, config);
            
            // 작업이 시작되지 않았으면 작업 시작
            if (!this.taskCompletionSystem.currentZone || this.taskCompletionSystem.currentZone !== currentZone) {
                this.startZoneTask(currentZone, model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing);
            }
            
            // 작업 완료 확인
            if (this.checkTaskCompletion(currentZone)) {
                // 작업 완료 - 새로운 의사결정 필요
                model.needsNewDecision = true;
                
                if (window.logManager) {
                    window.logManager.addLog(`🎲 ${currentZone} 작업 완료! 새로운 의사결정 시작...`);
                }
                
                // 작업 시스템 리셋
                this.resetTaskSystem();
                
                // 도착 대기 상태 리셋 (기존 시스템과 호환성)
                this.arrivalTime = null;
                this.arrivalWaitComplete = false;
            } else {
                // 작업 진행 중 - 기존 의사결정 유지
                if (model.targetAction) {
                    return model.targetAction;
                }
            }
        }
        
        // 새로운 의사결정이 필요한 경우 처리 (목표 지점에 도착했거나 의사결정이 없는 경우)
        if (model.needsNewDecision || !model.targetAction) {
            if (window.logManager) {
                window.logManager.addLog(`🔄 새로운 의사결정 시작 - 현재 액션: ${model.targetAction || '없음'}`);
            }
            
            // 현재 구역 확인
            const currentZone = this.getCurrentZone(modelX, modelY, startX, topY, spacing, config);
            
            // 새로운 랜덤 액션 선택
            const actions = ['매수', '매도', 'BTC 시장 탐색', 'N/B 길드 방문', '신호 대기'];
            const randomAction = actions[Math.floor(Math.random() * actions.length)];
            
            if (window.logManager) {
                window.logManager.addLog(`🎲 새로운 의사결정: ${randomAction} 선택`);
            }
            
            // 랜덤 액션에 따른 목표 설정 및 강제 이동
            if (window.trainerMovementController) {
                if (randomAction === '매수') {
                    window.trainerMovementController.moveToBuyArea(model, startX, topY);
                    // 목표 위치만 설정 (강제 이동 제거)
                    model.targetX = startX;
                    model.targetY = topY;
                } else if (randomAction === '매도') {
                    window.trainerMovementController.moveToSellArea(model, startX, topY, spacing);
                    // 목표 위치만 설정 (강제 이동 제거)
                    model.targetX = startX + spacing;
                    model.targetY = topY;
                } else if (randomAction === 'BTC 시장 탐색') {
                    window.trainerMovementController.moveToBTCMarket(model, config);
                    // 목표 위치만 설정 (강제 이동 제거)
                    model.targetX = config.width - 100;
                    model.targetY = config.height - 100;
                } else if (randomAction === 'N/B 길드 방문') {
                    window.trainerMovementController.moveToNBGuild(model);
                    // 목표 위치만 설정 (강제 이동 제거)
                    model.targetX = 100;
                    model.targetY = 100;
                } else {
                    window.trainerMovementController.moveToSignalCenter(model, config);
                    // 목표 위치만 설정 (강제 이동 제거)
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                }
                
                // 강제 이동 로그 제거하고 목표 설정 로그로 변경
                if (window.logManager) {
                    window.logManager.addLog(`🎯 트레이너 목표 설정: ${randomAction} 구역으로 이동 시작 (목표: ${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
                }
            }
            
            // 상태 초기화
            model.needsNewDecision = false;
            model.arrivalLogged = false;
            this.countdownStarted = false;
            this.btcExplorationMode = false;
            
            // 도착 대기 상태 리셋
            this.arrivalTime = null;
            this.arrivalWaitComplete = false;
            
            // 아이들 타이머 리셋 (새로운 의사결정으로 이동 시작)
            this.resetIdleTimer();
            
            // 색상 변경
            if (window.trainerVisualEffects) {
                window.trainerVisualEffects.changeTrainerColor(model, randomAction);
            }
            
            // 트레이너 대화창 업데이트
            if (window.trainerDialog) {
                const currentTime = new Date().toLocaleTimeString();
                const dialogText = `🎲 트레이너: 새로운 의사결정 → ${randomAction} | 시간: ${currentTime}`;
                if (window.trainerDialog && typeof window.trainerDialog.setText === 'function') {
                    window.trainerDialog.setText(dialogText);
                }
                
                // 화면 출력 내용을 로그에 저장
                if (window.logManager) {
                    window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }
            }
            
            return randomAction;
        }
        
        // 현재 구역에서 의사 결정 매칭 확인
        const currentZone = this.getCurrentZone(modelX, modelY, startX, topY, spacing, config);
        
        // 구역별 프로세스 단계 처리 (새로운 시스템)
        const processResult = this.handleZoneProcess(model, currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
        
        if (processResult) {
            // 프로세스 단계에서 반환된 액션 처리
            let targetAction = processResult;
            
            // 신호 대기 상태 처리
            if (targetAction === '신호 대기') {
                this.handleSignalWaiting(model, config);
            }
            
            // BTC 시장 탐색 모드 처리
            if (targetAction === 'BTC 시장 탐색') {
                this.handleBTCMarketExploration(model, config);
            }
            
            return targetAction;
        }
        
        // 기존 의사결정 시스템 (폴백)
        const zoneDecision = this.getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
        
        let targetAction = model.targetAction || '신호 대기';
        
        // 의사결정 처리
        if (!zoneDecision && !this.countdownStarted && !this.btcExplorationMode) {
            targetAction = this.handleNoDecision(model, config, targetAction, currentZone);
        } else if (zoneDecision && !this.countdownStarted) {
            targetAction = this.handleZoneDecision(model, zoneDecision, currentZone, currentMajority);
        } else if (this.countdownStarted) {
            targetAction = this.handleCountdown(model, config);
        }
        
        // 신호 대기 상태 처리
        if (targetAction === '신호 대기') {
            this.handleSignalWaiting(model, config);
        }
        
        // BTC 시장 탐색 모드 처리
        if (targetAction === 'BTC 시장 탐색') {
            this.handleBTCMarketExploration(model, config);
        }
        
        return targetAction;
    }

    // 현재 구역 확인 (개선된 버전)
    getCurrentZone(modelX, modelY, startX, topY, spacing, config) {
        // 디버깅: 위치 정보 로그
        if (window.logManager) {
            window.logManager.addLog(`📍 위치 분석: (${modelX}, ${modelY}) | 기준점: (${startX}, ${topY}) | 간격: ${spacing}`);
        }
        
        // 매수 영역 (더 넓은 범위로 설정)
        if (Math.abs(modelX - startX) < 120 && Math.abs(modelY - topY) < 120) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: 매수영역 (거리: X=${Math.abs(modelX - startX)}, Y=${Math.abs(modelY - topY)})`);
            }
            return '매수영역';
        }
        // 매도 영역 (더 넓은 범위로 설정)
        else if (Math.abs(modelX - (startX + spacing)) < 120 && Math.abs(modelY - topY) < 120) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: 매도영역 (거리: X=${Math.abs(modelX - (startX + spacing))}, Y=${Math.abs(modelY - topY)})`);
            }
            return '매도영역';
        }
        // 대기 영역 (더 넓은 범위로 설정)
        else if (Math.abs(modelX - (startX + spacing * 2)) < 120 && Math.abs(modelY - topY) < 120) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: 대기영역 (거리: X=${Math.abs(modelX - (startX + spacing * 2))}, Y=${Math.abs(modelY - topY)})`);
            }
            return '대기영역';
        }
        // N/B 길드 (더 넓은 범위로 설정)
        else if (Math.abs(modelX - 100) < 150 && Math.abs(modelY - 100) < 150) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: N/B길드 (거리: X=${Math.abs(modelX - 100)}, Y=${Math.abs(modelY - 100)})`);
            }
            return 'N/B길드';
        }
        // BTC 시장 탐색 구역 (더 넓은 범위로 설정)
        else if (Math.abs(modelX - (config.width - 100)) < 150 && Math.abs(modelY - (config.height - 100)) < 150) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: BTC시장탐색구역 (거리: X=${Math.abs(modelX - (config.width - 100))}, Y=${Math.abs(modelY - (config.height - 100))})`);
            }
            return 'BTC시장탐색구역';
        }
        // 신호 대기 센터 (더 넓은 범위로 설정)
        else if (Math.abs(modelX - config.width / 2) < 150 && Math.abs(modelY - config.height / 2) < 150) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 구역 판단: 신호대기센터 (거리: X=${Math.abs(modelX - config.width / 2)}, Y=${Math.abs(modelY - config.height / 2)})`);
            }
            return '신호대기센터';
        }
        
        // 기타 영역
        if (window.logManager) {
            window.logManager.addLog(`⚠️ 구역 판단: 기타영역 (모든 구역 조건 불만족)`);
        }
        return '기타영역';
    }

    // 구역별 의사결정
    getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        // 디버깅: 의사결정 입력값 로그
        if (window.logManager) {
            window.logManager.addLog(`🔍 의사결정 입력값: 구역=${currentZone}, 신호=${currentMajority}, 코인=${nbCoins}, 매수수익률=${buyProfitRate?.toFixed(2) || 'N/A'}%, 매도수익률=${sellProfitRate?.toFixed(2) || 'N/A'}%`);
        }
        
        // 의사결정 로직 (기존 decision-system 모듈과 연동)
        if (window.decisionSystem && typeof window.decisionSystem.getZoneDecision === 'function') {
            const decision = window.decisionSystem.getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
            
            if (decision) {
                if (window.logManager) {
                    window.logManager.addLog(`✅ 의사결정 성공: ${decision.action} → (${decision.targetX}, ${decision.targetY})`);
                }
                return decision;
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`❌ 의사결정 없음: 현재 구역(${currentZone})에서 조건 불만족`);
                }
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ decisionSystem 모듈을 찾을 수 없음 - 개선된 폴백 로직 사용`);
            }
        }
        
        // 개선된 폴백 의사결정 로직 사용
        const fallbackDecision = this.getFallbackDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
        
        // 의사결정 결과 로깅
        if (window.logManager) {
            window.logManager.addLog(`🎯 폴백 의사결정 결과: ${fallbackDecision.action} → (${fallbackDecision.targetX}, ${fallbackDecision.targetY})`);
        }
        
        return fallbackDecision;
    }

    // 폴백 의사결정 로직 (개선된 버전)
    getFallbackDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        // 전역 우선순위:
        // - BLUE 존: BTC 시장 탐색 구역에서 매수 전 예상 수익률 계산 → 매수 우선
        // - ORANGE 존: N/B 길드 구역에서 매도 전 예상 수익률 계산 → 매도 우선
        if (currentMajority && currentMajority.includes('BLUE')) {
            if (currentZone !== 'BTC시장탐색구역') {
                if (window.logManager) {
                    window.logManager.addLog(`🔵 우선순위(BLUE): BTC 시장 탐색 구역으로 이동 (매수 수익률 계산 우선)`);
                }
                return {
                    action: 'BTC 시장 탐색',
                    targetX: config.width - 100,
                    targetY: config.height - 100
                };
            }
            if (window.logManager) {
                window.logManager.addLog(`🔵 우선순위(BLUE): 매수 실행 우선`);
            }
            return {
                action: '매수',
                targetX: startX,
                targetY: topY
            };
        }

        if (currentMajority && currentMajority.includes('ORANGE')) {
            if (currentZone !== 'N/B길드' && currentZone !== 'N/B 길드') {
                if (window.logManager) {
                    window.logManager.addLog(`🟠 우선순위(ORANGE): N/B 길드로 이동 (매도 수익률 계산 우선)`);
                }
                return {
                    action: 'N/B 길드 방문',
                    targetX: 100,
                    targetY: 100
                };
            }
            // N/B 길드에 있으면 매도영역 좌표로 정확히 이동 유도
            if (window.logManager) {
                window.logManager.addLog(`🟠 ORANGE in N/B 길드: 매도영역으로 이동 후 매도 실행`);
            }
            return {
                action: '매도',
                targetX: startX + spacing,
                targetY: topY
            };
        }

        // 보호 로직: 신호에 따라 허용되지 않은 반대 포지션은 유도하지 않음
        if (currentMajority && currentMajority.includes('BLUE')) {
            return {
                action: '매수',
                targetX: startX,
                targetY: topY
            };
        }
        if (currentMajority && currentMajority.includes('ORANGE')) {
            return {
                action: 'N/B 길드 방문',
                targetX: 100,
                targetY: 100
            };
        }

        // 현재 시간 기반 랜덤 시드 생성
        const timeSeed = Math.floor(Date.now() / 2000); // 2초마다 변경 (더 빠른 변화)
        const randomValue = (timeSeed * 9301 + 49297) % 233280; // 간단한 랜덤 생성기
        const randomPercent = (randomValue / 233280) * 100;
        
        // 신호 대기 센터에서 오래 머물러 있으면 강제 이동
        if (currentZone === '신호대기센터') {
            this.waitCheckTimer = (this.waitCheckTimer || 0) + 1;
            
            // 3초(30프레임) 이상 신호 대기 센터에 머물러 있으면 강제 이동 (더 빠른 이동)
            if (this.waitCheckTimer > 30) {
                this.waitCheckTimer = 0;
                
                // 강제 이동 대상 선택 (더 다양한 액션)
                const forceMoveTargets = [
                    { action: '매수 수익률 계산', targetX: startX, targetY: topY, reason: '강제 매수 구역 이동' },
                    { action: '매도 수익률 계산', targetX: startX + spacing, targetY: topY, reason: '강제 매도 구역 이동' },
                    { action: 'N/B 코인 확인', targetX: 100, targetY: 100, reason: '강제 N/B 길드 방문' },
                    { action: '시장 분석 완료', targetX: config.width - 100, targetY: config.height - 100, reason: '강제 BTC 시장 탐색' },
                    { action: '시장 신호 분석', targetX: config.width / 2, targetY: config.height / 2, reason: '강제 신호 분석' }
                ];
                
                const selectedTarget = forceMoveTargets[Math.floor(randomPercent / 20)];
                
                if (window.logManager) {
                    window.logManager.addLog(`🚀 강제 이동: ${selectedTarget.reason} (신호 대기 센터 장기 체류)`);
                }
                
                return {
                    action: selectedTarget.action,
                    targetX: selectedTarget.targetX,
                    targetY: selectedTarget.targetY
                };
            }
        }
        // N/B 길드에서 오래 머물러 있으면 강제 이동
        else if (currentZone === 'N/B길드') {
            this.nbGuildWaitTimer = (this.nbGuildWaitTimer || 0) + 1;
            
            // 2초(20프레임) 이상 N/B 길드에 머물러 있으면 강제 이동
            if (this.nbGuildWaitTimer > 20) {
                this.nbGuildWaitTimer = 0;
                
                const forceMoveTargets = [
                    { action: '매수 수익률 계산', targetX: startX, targetY: topY, reason: 'N/B 길드에서 매수 구역 이동' },
                    { action: '매도 수익률 계산', targetX: startX + spacing, targetY: topY, reason: 'N/B 길드에서 매도 구역 이동' },
                    { action: '시장 분석 완료', targetX: config.width - 100, targetY: config.height - 100, reason: 'N/B 길드에서 BTC 시장 탐색' },
                    { action: '신호 대기', targetX: config.width / 2, targetY: config.height / 2, reason: 'N/B 길드에서 신호 대기 센터 이동' }
                ];
                
                const selectedTarget = forceMoveTargets[Math.floor(randomPercent / 25)];
                
                if (window.logManager) {
                    window.logManager.addLog(`🚀 강제 이동: ${selectedTarget.reason} (N/B 길드 장기 체류)`);
                }
                
                return {
                    action: selectedTarget.action,
                    targetX: selectedTarget.targetX,
                    targetY: selectedTarget.targetY
                };
            }
        }
        // BTC 시장 탐색 구역에서 오래 머물러 있으면 강제 이동
        else if (currentZone === 'BTC시장탐색구역') {
            // BTC 시장 탐색 구역에 처음 도착했을 때 수익률 계산 트리거
            if (!this.btcMarketArrivalLogged) {
                this.btcMarketArrivalLogged = true;
                this.btcMarketWaitTimer = 0;
                
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BTC 시장 탐색 구역 도착 - 수익률 계산 트리거`);
                }
                
                // BTC 시장 탐색 구역에서는 항상 매수 수익률 계산 (신호와 관계없이)
                const buyProfitRate = this.calculateBuyProfitRate(model, config);
                window.buyProfitRate = buyProfitRate; // 전역 변수에 저장
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BTC 시장 탐색 구역: 매수 수익률 ${buyProfitRate.toFixed(2)}% 계산 완료`);
                }
                
                // 매도 수익률도 계산 (조건부)
                const sellProfitRate = this.calculateSellProfitRate(model, config);
                if (window.logManager) {
                    window.logManager.addLog(`📊 BTC 시장 탐색: 매도 수익률 ${sellProfitRate.toFixed(2)}% 계산 완료`);
                }
            }
            
            // 대기 타이머 증가
            this.btcMarketWaitTimer = (this.btcMarketWaitTimer || 0) + 1;
            
            // 5초(50프레임)마다 수익률 재계산
            if (this.btcMarketWaitTimer % 50 === 0 && this.btcMarketWaitTimer > 0) {
                if (window.logManager) {
                    window.logManager.addLog(`🔄 BTC 시장 탐색 구역 장기 체류 - 수익률 재계산 트리거`);
                }
                
                // BTC 시장 탐색 구역에서는 항상 매수 수익률 계산 (신호와 관계없이)
                const buyProfitRate = this.calculateBuyProfitRate(model, config);
                window.buyProfitRate = buyProfitRate; // 전역 변수에 저장
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BTC 시장 탐색 구역: 매수 수익률 재계산 ${buyProfitRate.toFixed(2)}% 완료`);
                }
                
                // 매도 수익률도 재계산 (조건부)
                const sellProfitRate = this.calculateSellProfitRate(model, config);
                if (window.logManager) {
                    window.logManager.addLog(`📊 BTC 시장 탐색: 매도 수익률 재계산 ${sellProfitRate.toFixed(2)}% 완료`);
                }
            }
            
            // 10초(100프레임) 이상 BTC 시장 탐색 구역에 머물러 있으면 강제 이동
            if (this.btcMarketWaitTimer > 100) {
                this.btcMarketWaitTimer = 0;
                
                // 강제 이동 대상 선택 (BTC 시장에서 벗어나는 방향)
                const forceMoveTargets = [
                    { action: '매수', targetX: startX, targetY: topY, reason: '강제 매수 구역 이동 (BTC 시장에서 벗어남)' },
                    { action: 'N/B 길드 방문', targetX: 100, targetY: 100, reason: '강제 N/B 길드 방문 (BTC 시장에서 벗어남)' },
                    { action: '신호 대기', targetX: config.width / 2, targetY: config.height / 2, reason: '강제 신호 대기 (BTC 시장에서 벗어남)' }
                ];
                
                const selectedTarget = forceMoveTargets[Math.floor(randomPercent / 33.33)];
                
                if (window.logManager) {
                    window.logManager.addLog(`🚀 강제 이동: ${selectedTarget.reason} (BTC 시장 탐색 구역 장기 체류)`);
                }
                
                return {
                    action: selectedTarget.action,
                    targetX: selectedTarget.targetX,
                    targetY: selectedTarget.targetY
                };
            }
        } else {
            // 다른 구역에 있으면 타이머들 리셋
            this.waitCheckTimer = 0;
            this.btcMarketWaitTimer = 0;
            this.btcMarketArrivalLogged = false; // BTC 시장 도착 로그 플래그 리셋
        }
        
        // 1. 매도 우선 (N/B 코인이 있고 매도 수익률이 있는 경우)
        if (nbCoins > 0 && sellProfitRate !== 0) {
            if (window.logManager) {
                window.logManager.addLog(`📉 폴백 매도 의사결정: 코인 ${nbCoins}개, 매도수익률 ${sellProfitRate.toFixed(2)}%`);
            }
            return {
                action: '매도',
                targetX: startX + spacing,
                targetY: topY
            };
        }
        
        // 2. 매수 (BLUE 신호이고 매수 수익률이 있는 경우) - BLUE 구역 특별 처리
        if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔵 BLUE 구역 매수 의사결정: BLUE 신호, 매수수익률 ${buyProfitRate.toFixed(2)}%`);
            }
            
            // BLUE 구역에서는 BTC 시장 탐색을 우선적으로 수행
            if (currentZone !== 'BTC시장탐색구역') {
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 특별 처리: BTC 시장 탐색 우선 수행`);
                }
                return {
                    action: 'BTC 시장 탐색',
                    targetX: config.width - 100,
                    targetY: config.height - 100
                };
            } else {
                // 이미 BTC 시장 탐색 구역에 있으면 매수 구역으로 이동
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 BTC 시장 탐색 완료: 매수 구역으로 이동`);
                }
                return {
                    action: '매수',
                    targetX: startX,
                    targetY: topY
                };
            }
        }
        
        // 3. ORANGE 신호일 때 BTC 시장 탐색 (개선된 버전)
        if (currentMajority === 'ORANGE') {
            // BTC 시장 탐색 구역에 이미 있으면 다른 구역으로 이동
            if (currentZone === 'BTC시장탐색구역') {
                const alternativeTargets = [
                    { action: '매수 구역 탐색', targetX: startX, targetY: topY, reason: 'ORANGE 신호 - 매수 구역 탐색' },
                    { action: 'N/B 길드 방문', targetX: 100, targetY: 100, reason: 'ORANGE 신호 - N/B 길드 방문' },
                    { action: '신호 대기', targetX: config.width / 2, targetY: config.height / 2, reason: 'ORANGE 신호 - 신호 대기' }
                ];
                
                const selectedTarget = alternativeTargets[Math.floor(randomPercent / 33.33)];
                
                if (window.logManager) {
                    window.logManager.addLog(`🔄 폴백 대안 이동: ${selectedTarget.reason} (BTC 시장 탐색 구역에서 벗어남)`);
                }
                
                return {
                    action: selectedTarget.action,
                    targetX: selectedTarget.targetX,
                    targetY: selectedTarget.targetY
                };
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`🔍 폴백 BTC 시장 탐색: ORANGE 신호 감지`);
                }
                return {
                    action: 'BTC 시장 탐색',
                    targetX: config.width - 100,
                    targetY: config.height - 100
                };
            }
        }
        
        // 4. N/B 길드 방문 (코인이 있고 매도 수익률이 계산되지 않은 경우)
        if (nbCoins > 0 && sellProfitRate === 0) {
            if (window.logManager) {
                window.logManager.addLog(`🏛️ 폴백 N/B 길드 방문: 코인 ${nbCoins}개, 매도수익률 계산 필요`);
            }
            return {
                action: 'N/B 길드 방문',
                targetX: 100,
                targetY: 100
            };
        }
        
        // 5. BTC 시장 탐색 (BLUE 신호이고 매수 수익률이 계산되지 않은 경우) - BLUE 구역 특별 처리
        if (currentMajority === 'BLUE' && buyProfitRate === 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔵 BLUE 구역 BTC 시장 탐색: BLUE 신호, 매수수익률 계산 필요`);
            }
            
            // BLUE 구역에서는 BTC 시장 탐색을 우선적으로 수행하여 매수 수익률 계산
            if (currentZone !== 'BTC시장탐색구역') {
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 특별 처리: BTC 시장 탐색으로 이동하여 매수 수익률 계산`);
                }
                return {
                    action: 'BTC 시장 탐색',
                    targetX: config.width - 100,
                    targetY: config.height - 100
                };
            } else {
                // 이미 BTC 시장 탐색 구역에 있으면 수익률 계산 후 매수 구역으로 이동
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 BTC 시장 탐색 중: 수익률 계산 후 매수 구역으로 이동`);
                }
                return {
                    action: '매수',
                    targetX: startX,
                    targetY: topY
                };
            }
        }
        
        // 6. 랜덤 탐색 (신호 대기 센터에서 벗어나기)
        if (currentZone === '신호대기센터') {
            const explorationTargets = [
                { action: '매수 구역 탐색', targetX: startX, targetY: topY },
                { action: '매도 구역 탐색', targetX: startX + spacing, targetY: topY },
                { action: 'N/B 길드 탐색', targetX: 100, targetY: 100 },
                { action: 'BTC 시장 탐색', targetX: config.width - 100, targetY: config.height - 100 }
            ];
            
            const selectedTarget = explorationTargets[Math.floor(randomPercent / 25)];
            
            if (window.logManager) {
                window.logManager.addLog(`🔍 랜덤 탐색: ${selectedTarget.action} (신호 대기 센터에서 벗어남)`);
            }
            
            return {
                action: selectedTarget.action,
                targetX: selectedTarget.targetX,
                targetY: selectedTarget.targetY
            };
        }
        
        // 7. 신호 대기 (기타 경우)
        if (window.logManager) {
            window.logManager.addLog(`⏳ 폴백 신호 대기: 현재 조건으로는 대기 필요`);
        }
        return {
            action: '신호 대기',
            targetX: config.width / 2,
            targetY: config.height / 2
        };
    }

    // 의사결정이 없을 때 처리
    handleNoDecision(model, config, targetAction, currentZone) {
        // 현재 위치에서 의사결정이 없으면 지능적인 다음 액션 선택
        let nextAction = this.getIntelligentNextAction(currentZone, targetAction);
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 지능적 다음 액션 선택: ${nextAction} (현재 구역: ${currentZone})`);
        }
        
        // 액션에 따른 목표 설정
        if (window.trainerMovementController) {
            if (nextAction === '매수 수익률 계산') {
                window.trainerMovementController.moveToBuyArea(model, 100, 100);
            } else if (nextAction === '매도 수익률 계산') {
                window.trainerMovementController.moveToSellArea(model, 100, 100, 200);
            } else if (nextAction === '시장 분석 완료') {
                window.trainerMovementController.moveToBTCMarket(model, config);
            } else if (nextAction === 'N/B 코인 확인') {
                window.trainerMovementController.moveToNBGuild(model);
            } else {
                window.trainerMovementController.moveToSignalCenter(model, config);
            }
        } else {
            // 폴백: 직접 목표 위치 설정
            if (nextAction === '매수 수익률 계산') {
                model.targetX = 100;
                model.targetY = 100;
            } else if (nextAction === '매도 수익률 계산') {
                model.targetX = 300;
                model.targetY = 100;
            } else if (nextAction === '시장 분석 완료') {
                model.targetX = config.width - 100;
                model.targetY = config.height - 100;
            } else if (nextAction === 'N/B 코인 확인') {
                model.targetX = 100;
                model.targetY = 100;
            } else {
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
            }
        }
        
        // 색상 변경
        if (window.trainerVisualEffects) {
            if (nextAction.includes('매수')) {
                window.trainerVisualEffects.setTrainerColor(model, 'red');
            } else if (nextAction.includes('매도')) {
                window.trainerVisualEffects.setTrainerColor(model, 'orange');
            } else if (nextAction.includes('N/B')) {
                window.trainerVisualEffects.setTrainerColor(model, 'purple');
            } else if (nextAction.includes('시장')) {
                window.trainerVisualEffects.setTrainerColor(model, 'blue');
            } else {
                window.trainerVisualEffects.setTrainerColor(model, 'cyan');
            }
        }
        
        return nextAction;
    }

    // 지능적인 다음 액션 선택
    getIntelligentNextAction(currentZone, currentAction) {
        // 현재 시간 기반 랜덤 시드 생성
        const timeSeed = Math.floor(Date.now() / 1500); // 1.5초마다 변경 (더 빠른 변화)
        const randomValue = (timeSeed * 9301 + 49297) % 233280;
        const randomPercent = (randomValue / 233280) * 100;
        
        // 현재 구역과 액션에 따른 지능적인 다음 액션 선택 (더 다양한 액션)
        const zoneActionMap = {
            '매수영역': [
                '매수 수익률 계산',
                '매도 수익률 계산', 
                'N/B 코인 확인',
                '시장 분석 완료',
                '시장 신호 분석'
            ],
            '매도영역': [
                '매도 수익률 계산',
                '매수 수익률 계산',
                'N/B 코인 확인', 
                '시장 분석 완료',
                '시장 신호 분석'
            ],
            'N/B길드': [
                'N/B 코인 확인',
                '매수 수익률 계산',
                '매도 수익률 계산',
                '시장 분석 완료',
                '시장 신호 분석'
            ],
            'BTC시장탐색구역': [
                '시장 분석 완료',
                '매수 수익률 계산',
                '매도 수익률 계산',
                'N/B 코인 확인',
                '시장 신호 분석'
            ],
            '신호대기센터': [
                '시장 신호 분석',
                '매수 수익률 계산',
                '매도 수익률 계산',
                'N/B 코인 확인',
                '시장 분석 완료'
            ],
            '기타영역': [
                '시장 신호 분석',
                '매수 수익률 계산',
                '매도 수익률 계산',
                'N/B 코인 확인',
                '시장 분석 완료'
            ]
        };
        
        // 현재 구역에 따른 액션 목록 선택
        const availableActions = zoneActionMap[currentZone] || zoneActionMap['기타영역'];
        
        // 랜덤하게 다음 액션 선택 (현재 액션과 다른 액션 우선)
        let nextAction;
        if (currentAction && availableActions.length > 1) {
            // 현재 액션과 다른 액션들만 필터링
            const differentActions = availableActions.filter(action => action !== currentAction);
            if (differentActions.length > 0) {
                nextAction = differentActions[Math.floor(randomPercent / (100 / differentActions.length))];
            } else {
                nextAction = availableActions[Math.floor(randomPercent / (100 / availableActions.length))];
            }
        } else {
            nextAction = availableActions[Math.floor(randomPercent / (100 / availableActions.length))];
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🧠 지능적 액션 선택: 구역=${currentZone}, 현재액션=${currentAction} → 다음액션=${nextAction} (랜덤%)`);
        }
        
        return nextAction;
    }

    // 구역 의사결정 처리
    handleZoneDecision(model, zoneDecision, currentZone, currentMajority) {
        const targetAction = zoneDecision.action;
        model.targetAction = targetAction;
        model.targetX = zoneDecision.targetX;
        model.targetY = zoneDecision.targetY;
        
        // 디버깅: 목표 위치 설정 확인
        if (window.logManager) {
            window.logManager.addLog(`🎯 의사결정: ${targetAction} → 목표 위치 (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
        }
        
        // 색상 변경
        if (window.trainerVisualEffects) {
            window.trainerVisualEffects.changeTrainerColor(model, targetAction);
        }
        
        // 트레이너 의사결정 로그
        if (window.trainerActivityLogger) {
            window.trainerActivityLogger.logDecision(targetAction, `구역: ${currentZone}, 신호: ${currentMajority}`);
        }
        
        // 트레이너 대화창 업데이트
        if (window.trainerDialog) {
            const currentTime = new Date().toLocaleTimeString();
            const dialogText = `🎯 트레이너: 의사결정 → ${targetAction} | 구역: ${currentZone} | 시간: ${currentTime}`;
            window.trainerDialog.setText(dialogText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
            }
        }
        
        return targetAction;
    }

    // 카운트다운 처리
    handleCountdown(model, config) {
        const targetAction = '신호 대기';
        model.targetAction = targetAction;
        model.targetX = config.width / 2;
        model.targetY = config.height / 2;
        
        if (window.trainerVisualEffects) {
            window.trainerVisualEffects.changeTrainerColor(model, targetAction);
        }
        
        return targetAction;
    }

    // 신호 대기 상태 처리
    handleSignalWaiting(model, config) {
        const distanceToCenter = Math.sqrt((model.circle.x - (config.width / 2)) ** 2 + (model.circle.y - (config.height / 2)) ** 2);
        
        // 현재 위치와 신호 대기 센터까지의 거리 로그
        if (window.logManager) {
            const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
            const centerPos = `(${Math.round(config.width / 2)}, ${Math.round(config.height / 2)})`;
            window.logManager.addLog(`⏳ 신호 대기: 현재위치 ${currentPos} | 센터위치 ${centerPos} | 센터까지거리 ${Math.round(distanceToCenter)}px`);
        }
        
        if (distanceToCenter < 30) {
            // 신호 대기 센터에 도달했을 때
            if (!this.waitCheckTimer) {
                this.waitCheckTimer = 0;
                this.waitStartTime = Date.now();
                
                // 신호 대기 센터 도착 시 아이들 타이머 리셋
                this.resetIdleTimer();
                
                if (window.logManager) {
                    window.logManager.addLog(`🎯 트레이너가 신호 대기 센터에 도착 → 아이들 타이머 리셋됨`);
                }
            }
            this.waitCheckTimer++;
            this.countdownStarted = true;
            
            const elapsedSeconds = (Date.now() - this.waitStartTime) / 1000;
            
            // 5초 이상 대기하면 BTC 시장으로 이동
            if (elapsedSeconds >= 5) {
                this.btcExplorationMode = true;
                this.countdownStarted = false;
                
                if (window.trainerMovementController) {
                    window.trainerMovementController.moveToBTCMarket(model, config);
                }
                
                if (window.logManager) {
                    window.logManager.addLog(`🔵 트레이너: 신호 대기 센터에서 5초 이상 대기 → BTC 시장 탐색 구역으로 이동!`);
                }
            }
        } else if (distanceToCenter > 50 && !this.countdownStarted) {
            this.waitCheckTimer = 0;
        }
    }

    // BTC 시장 탐색 모드 처리 (BLUE 구역 특별 처리 포함)
    handleBTCMarketExploration(model, config) {
        // 이미 BTC 시장에 도착했고 새로운 의사결정이 필요한 경우 처리
        if (model.btcMarketArrived && model.needsNewDecision) {
            this.handleBTCMarketArrival(model, config);
            return;
        }
        
        // BTC 시장에 도착했지만 아직 처리되지 않은 경우 처리
        if (model.btcMarketArrived && !model.btcMarketArrivalLogged) {
            this.handleBTCMarketArrival(model, config);
            return;
        }
        
        // 강제 BTC 시장 도착 처리 (목표 거리가 0px이고 BTC 시장 탐색 액션인 경우)
        const distanceToTarget = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
        if (model.targetAction === 'BTC 시장 탐색' && distanceToTarget <= 5 && !model.btcMarketArrivalLogged) {
            if (window.logManager) {
                window.logManager.addLog(`🎯 강제 BTC 시장 도착 처리 시작 - 목표거리: ${Math.round(distanceToTarget)}px`);
            }
            model.btcMarketArrived = true;
            model.needsNewDecision = true;
            this.handleBTCMarketArrival(model, config);
            return;
        }
        
        const distanceToBTCMarket = Math.sqrt((model.circle.x - (config.width - 100)) ** 2 + (model.circle.y - (config.height - 100)) ** 2);
        
        // 현재 위치와 목표 정보 로그 (5초마다만 출력하여 로그 중복 방지)
        if (window.logManager && (!model.lastBTCLogTime || Date.now() - model.lastBTCLogTime > 5000)) {
            const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
            const targetPos = `(${Math.round(model.targetX)}, ${Math.round(model.targetY)})`;
            const distanceToTarget = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
            window.logManager.addLog(`📍 BTC 시장 탐색 중: 현재위치 ${currentPos} | 목표위치 ${targetPos} | 목표까지거리 ${Math.round(distanceToTarget)}px | BTC시장까지거리 ${Math.round(distanceToBTCMarket)}px`);
            model.lastBTCLogTime = Date.now();
        }
        
        // 현재 시장 신호 확인 (BLUE 구역 특별 처리용)
        const majorityElement = document.getElementById('majority-zone');
        const currentMajority = majorityElement ? majorityElement.textContent.trim() : '';
        const isBlueZone = currentMajority.includes('BLUE');
        
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
        
        // BTC 시장에 도달했는지 확인 (더 넓은 범위로 설정)
        if (isCollidingWithBTCMarket || distanceToBTCMarket < 50) {
            // 이미 수익률 계산이 완료되었는지 확인
            if (!model.btcMarketArrivalLogged) {
                model.btcMarketArrivalLogged = true;
                model.btcMarketArrived = true;
                
                if (window.logManager) {
                    const zoneInfo = isBlueZone ? '🔵 BLUE 구역에서' : '⚪ 일반 구역에서';
                    window.logManager.addLog(`🎯 트레이너가 ${zoneInfo} BTC 시장 탐색 구역에 도달! 수익률 계산 시작`);
                }
                
                // 수익률 계산 및 다음 단계 처리
                this.handleBTCMarketArrival(model, config);
            }
        } else if (model.btcMarketArrived && !model.btcMarketArrivalLogged) {
            // 이미 도착 플래그가 설정되어 있지만 아직 처리되지 않은 경우
            model.btcMarketArrivalLogged = true;
            
            if (window.logManager) {
                const zoneInfo = isBlueZone ? '🔵 BLUE 구역에서' : '⚪ 일반 구역에서';
                window.logManager.addLog(`🎯 트레이너가 ${zoneInfo} BTC 시장 탐색 구역에 도달! 수익률 계산 시작 (플래그 기반)`);
            }
            
            // 수익률 계산 및 다음 단계 처리
            this.handleBTCMarketArrival(model, config);
        } else {
            // BTC 시장에서 멀어지면 로그 리셋
            if (distanceToBTCMarket > 120) {
                model.btcMarketArrivalLogged = false;
                model.btcMarketArrived = false;
            }
        }
    }
    
    // BTC 시장 도착 후 처리 로직
    handleBTCMarketArrival(model, config) {
        // 현재 시장 신호 확인
        const majorityElement = document.getElementById('majority-zone');
        const currentMajority = majorityElement ? majorityElement.textContent.trim() : '';
        const isBlueZone = currentMajority.includes('BLUE');
        
        if (window.logManager) {
            const zoneInfo = isBlueZone ? '🔵 BLUE 구역' : '⚪ 일반 구역';
            window.logManager.addLog(`📊 ${zoneInfo} BTC 시장에서 수익률 계산 시작`);
        }
        
        // BLUE 구역 특별 처리
        if (isBlueZone) {
            if (window.logManager) {
                window.logManager.addLog(`🔵 BLUE 구역 특별 처리: 매수 전 예상 수익률 우선 계산`);
            }
            
            // BLUE 구역에서는 매수 수익률을 우선적으로 계산
            const buyProfitRate = this.calculateBuyProfitRate(model, config);
            
            // 매수 수익률이 양수이고 충분히 높으면 매도 수익률도 계산
            if (buyProfitRate > 0.5) {
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 매수 수익률 양수 (${buyProfitRate.toFixed(2)}%) → 매도 수익률도 계산`);
                }
                this.calculateSellProfitRate(model, config);
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BLUE 구역 매수 수익률 낮음 (${buyProfitRate.toFixed(2)}%) → 매도 수익률 계산 생략`);
                }
            }
        } else {
            // 일반 구역에서는 기존 로직 사용
            if (window.logManager) {
                window.logManager.addLog(`⚪ 일반 구역 처리: 매수/매도 수익률 모두 계산`);
            }
            
            // 매수/매도 전 예상 수익률 계산
            this.calculateBuyProfitRate(model, config);
            this.calculateSellProfitRate(model, config);
        }
        
        // 수익률 계산 완료 후 신호 대기 센터로 이동
        setTimeout(() => {
            if (window.trainerMovementController) {
                // 액션을 신호 대기로 변경
                model.targetAction = '신호 대기';
                window.trainerMovementController.setTargetPosition(model, config);
                
                if (window.logManager) {
                    const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
                    const signalCenterPos = `(${Math.round(config.width / 2)}, ${Math.round(config.height / 2)})`;
                    const distanceToSignalCenter = Math.sqrt((model.circle.x - (config.width / 2)) ** 2 + (model.circle.y - (config.height / 2)) ** 2);
                    const zoneInfo = isBlueZone ? '🔵 BLUE 구역' : '⚪ 일반 구역';
                    window.logManager.addLog(`🔄 ${zoneInfo} 수익률 계산 완료 - 신호 대기 센터로 이동: 현재위치 ${currentPos} | 센터위치 ${signalCenterPos} | 센터까지거리 ${Math.round(distanceToSignalCenter)}px`);
                }
            }
            
            // 상태 리셋
            model.btcMarketArrivalLogged = false;
            model.btcMarketArrived = false;
            model.needsNewDecision = false;
            
            if (window.logManager) {
                const zoneInfo = isBlueZone ? '🔵 BLUE 구역' : '⚪ 일반 구역';
                window.logManager.addLog(`🔄 ${zoneInfo} BTC 시장 탐색 완료 - 신호 대기 센터로 이동 시작`);
            }
        }, 1000); // 1초 후 이동
    }
    
    // 매수 구역 도착 처리
    handleBuyAreaArrival(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (!model.arrivalLogged) {
            model.arrivalLogged = true;
            
            if (window.logManager) {
                window.logManager.addLog(`🎯 트레이너 매수 구역 도착: 현재위치 (${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`);
            }
            
            // N/B 코인 드랍 시스템 호출
            if (window.handleBuyAreaArrival) {
                window.handleBuyAreaArrival();
            } else if (window.nbCoinDropSystem) {
                window.nbCoinDropSystem.handleBuyAreaArrival();
            }
            
            // 매수 액션 처리
            if (window.trainerStateHandler) {
                window.trainerStateHandler.handleBuyAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            }
            
            // 트레이너 색상 변경
            if (window.trainerVisualEffects) {
                window.trainerVisualEffects.changeTrainerColor(model, '매수');
            }
            
            // 대화창 업데이트
            if (window.trainerDialog) {
                const currentTime = new Date().toLocaleTimeString();
                const dialogText = `🎯 트레이너: 매수 구역 도착 | N/B 코인 드랍 완료 | 시간: ${currentTime}`;
                window.trainerDialog.setText(dialogText);
                
                if (window.logManager) {
                    window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }
            }
        }
    }
    
    // 매수 전 예상 수익률 계산 (BLUE 구역에서 BTC 시장 탐색 시)
    calculateBuyProfitRate(model, config) {
        // 현재 위치 정보 로그
        if (window.logManager) {
            const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
            const btcMarketPos = `(${Math.round(config.width - 100)}, ${Math.round(config.height - 100)})`;
            const distanceToBTCMarket = Math.sqrt((model.circle.x - (config.width - 100)) ** 2 + (model.circle.y - (config.height - 100)) ** 2);
            window.logManager.addLog(`🔵 BLUE 구역 BTC 시장 탐색: 매수 전 예상 수익률 계산 시작`);
            window.logManager.addLog(`📍 현재위치 ${currentPos} | BTC시장위치 ${btcMarketPos} | BTC시장까지거리 ${Math.round(distanceToBTCMarket)}px`);
        }
        
        // 실제 거래 데이터 가져오기
        const majorityElement = document.getElementById('majority-zone');
        const currentPriceElement = document.getElementById('right-trading-current-price');
        const priceChangeElement = document.getElementById('right-trading-price-change');
        const currentZoneElement = document.getElementById('right-trading-current-zone');
        const zoneStrengthElement = document.getElementById('right-trading-zone-strength');
        const btcBalanceElement = document.getElementById('btc-balance');
        const krwBalanceElement = document.getElementById('krw-balance');
        const avgPriceElement = document.getElementById('selected-coin-avg-price');
        const pnlElement = document.getElementById('selected-coin-pnl');
        
        if (!majorityElement || !currentPriceElement) {
            if (window.logManager) {
                window.logManager.addLog(`❌ BTC 시장 데이터를 가져올 수 없음 - 매수 수익률 계산 중단`);
            }
            return 0;
        }
        
        // 실제 거래 데이터 파싱
        const currentMajority = majorityElement.textContent.trim();
        const currentPriceText = currentPriceElement.textContent;
        const priceChangeText = priceChangeElement ? priceChangeElement.textContent : '0%';
        const currentZone = currentZoneElement ? currentZoneElement.textContent : 'Unknown';
        const zoneStrength = zoneStrengthElement ? zoneStrengthElement.textContent.match(/\d+/)?.[0] || '0' : '0';
        const btcBalance = btcBalanceElement ? parseFloat(btcBalanceElement.textContent.match(/[\d.]+/)?.[0] || '0') : 0;
        const krwBalance = krwBalanceElement ? parseFloat(krwBalanceElement.textContent.match(/[\d,]+/)?.[0].replace(/,/g, '') || '0') : 0;
        const avgPriceText = avgPriceElement ? avgPriceElement.textContent : '';
        const pnlText = pnlElement ? pnlElement.textContent : '';
        
        // 현재 가격 파싱
        const currentPrice = parseFloat(currentPriceText.replace(/[₩,]/g, ''));
        
        // 평균 단가 파싱
        const avgPriceMatch = avgPriceText.match(/[\d,]+/);
        const avgPrice = avgPriceMatch ? parseFloat(avgPriceMatch[0].replace(/,/g, '')) : currentPrice;
        
        // N/B 코인 개수 가져오기
        const nbCoins = window.gameInitializer ? window.gameInitializer.gameData.nbCoins : 0;
        
        // BLUE 구역에서 BTC 시장 탐색 시 매수 전 예상 수익률 계산 로직
        let buyProfitRate = 0;
        
        // 1. BLUE 신호 기반 기본 수익률 (매수 유리)
        if (currentMajority.includes('BLUE')) {
            buyProfitRate += 1.2 + Math.random() * 1.8; // 1.2% ~ 3.0% (BLUE 신호 시 더 높은 수익률)
            if (window.logManager) {
                window.logManager.addLog(`🔵 BLUE 신호 감지: 기본 수익률 ${buyProfitRate.toFixed(2)}% 추가`);
            }
        } else if (currentMajority.includes('ORANGE')) {
            buyProfitRate += 0.3 + Math.random() * 0.7; // 0.3% ~ 1.0% (ORANGE 신호 시 보수적)
            if (window.logManager) {
                window.logManager.addLog(`🟠 ORANGE 신호 감지: 기본 수익률 ${buyProfitRate.toFixed(2)}% 추가`);
            }
        } else {
            buyProfitRate += -0.2 + Math.random() * 0.4; // -0.2% ~ 0.2% (기타 신호 시 매우 보수적)
            if (window.logManager) {
                window.logManager.addLog(`⚪ 기타 신호 감지: 기본 수익률 ${buyProfitRate.toFixed(2)}% 추가`);
            }
        }
        
        // 2. 현재 보유 자산 수익률 반영 (실제 포트폴리오 기반)
        if (btcBalance > 0 && avgPrice > 0) {
            const currentProfitRate = ((currentPrice - avgPrice) / avgPrice) * 100;
            const reflectedProfit = currentProfitRate * 0.4; // 현재 수익률의 40% 반영
            buyProfitRate += reflectedProfit;
            if (window.logManager) {
                window.logManager.addLog(`💰 현재 보유 자산 수익률: ${currentProfitRate.toFixed(2)}% → 반영: ${reflectedProfit.toFixed(2)}%`);
            }
        }
        
        // 3. 가격 변동률 반영 (시장 모멘텀)
        const priceChangeMatch = priceChangeText.match(/-?[\d.]+/);
        if (priceChangeMatch) {
            const priceChange = parseFloat(priceChangeMatch[0]);
            const momentumEffect = priceChange * 0.4; // 가격 변동률의 40% 반영
            buyProfitRate += momentumEffect;
            if (window.logManager) {
                window.logManager.addLog(`📈 가격 변동률: ${priceChange.toFixed(2)}% → 모멘텀 효과: ${momentumEffect.toFixed(2)}%`);
            }
        }
        
        // 4. 구역 강도 반영 (시장 강도)
        const strength = parseInt(zoneStrength);
        if (strength > 0) {
            const strengthEffect = (strength - 50) * 0.015; // 강도 50 기준으로 ±0.75% 보정
            buyProfitRate += strengthEffect;
            if (window.logManager) {
                window.logManager.addLog(`💪 구역 강도: ${strength} → 강도 효과: ${strengthEffect.toFixed(2)}%`);
            }
        }
        
        // 5. N/B 코인 개수에 따른 보정 (시장 신뢰도)
        if (nbCoins > 0) {
            const coinEffect = nbCoins * 0.03; // 코인 1개당 0.03% 추가
            buyProfitRate += coinEffect;
            if (window.logManager) {
                window.logManager.addLog(`🪙 N/B 코인 ${nbCoins}개 → 신뢰도 보정: +${coinEffect.toFixed(2)}%`);
            }
        }
        
        // 6. 포트폴리오 비율 반영 (리스크 관리)
        const totalValue = (btcBalance * currentPrice) + krwBalance;
        if (totalValue > 0) {
            const btcRatio = (btcBalance * currentPrice) / totalValue;
            let portfolioEffect = 0;
            
            if (btcRatio > 0.8) {
                portfolioEffect = -0.4; // BTC 비중이 높으면 매수 신중
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ BTC 비중 높음 (${(btcRatio * 100).toFixed(1)}%) → 리스크 관리: ${portfolioEffect.toFixed(2)}%`);
                }
            } else if (btcRatio < 0.2) {
                portfolioEffect = 0.4; // BTC 비중이 낮으면 매수 적극
                if (window.logManager) {
                    window.logManager.addLog(`✅ BTC 비중 낮음 (${(btcRatio * 100).toFixed(1)}%) → 매수 기회: +${portfolioEffect.toFixed(2)}%`);
                }
            }
            
            buyProfitRate += portfolioEffect;
        }
        
        // 7. 시간 기반 랜덤 요소 (시장 변동성)
        const timeSeed = Math.floor(Date.now() / 3000); // 3초마다 변경
        const timeRandom = ((timeSeed * 9301 + 49297) % 233280) / 233280;
        const volatilityEffect = (timeRandom - 0.5) * 0.3; // ±0.15% 랜덤 변동
        buyProfitRate += volatilityEffect;
        
        if (window.logManager) {
            window.logManager.addLog(`🎲 시장 변동성: ${volatilityEffect.toFixed(2)}% (랜덤 요소)`);
        }
        
        // 8. BLUE 구역 특별 보너스 (BLUE 구역에서 BTC 시장 탐색 시)
        if (currentZone.includes('BLUE') || currentMajority.includes('BLUE')) {
            const blueBonus = 0.2 + Math.random() * 0.3; // 0.2% ~ 0.5% BLUE 구역 보너스
            buyProfitRate += blueBonus;
            if (window.logManager) {
                window.logManager.addLog(`🔵 BLUE 구역 특별 보너스: +${blueBonus.toFixed(2)}%`);
            }
        }
        
        // 수익률 범위 제한 (-3% ~ 10%) (BLUE 구역에서는 더 낙관적)
        buyProfitRate = Math.max(-3, Math.min(10, buyProfitRate));
        
        // 비정상적인 수익률 값 검증
        if (buyProfitRate < -3 || buyProfitRate > 10) {
            console.warn(`⚠️ 비정상적인 매수 수익률 감지: ${buyProfitRate.toFixed(2)}%, 0으로 재설정`);
            buyProfitRate = 0;
        }
        
        // 계산된 수익률을 전역 변수에 저장
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.buyProfitRate = buyProfitRate;
        }
        
        // UI 업데이트
        if (window.buyProfitRateDisplay) {
            const profitRateText = `🔵 BLUE 구역 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
            window.buyProfitRateDisplay.setText(profitRateText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(매수수익률): ${profitRateText}`);
            }
        }
        
        // 상세 계산 로그
        if (window.logManager) {
            window.logManager.addLog(`📊 BLUE 구역 BTC 시장 탐색 - 매수 전 예상 수익률 계산 완료:`);
            window.logManager.addLog(`  - 현재가: ₩${currentPrice.toLocaleString()}`);
            window.logManager.addLog(`  - 평균단가: ₩${avgPrice.toLocaleString()}`);
            window.logManager.addLog(`  - BTC 보유량: ${btcBalance} BTC`);
            window.logManager.addLog(`  - KRW 보유량: ₩${krwBalance.toLocaleString()}`);
            window.logManager.addLog(`  - 시장 신호: ${currentMajority}`);
            window.logManager.addLog(`  - 구역: ${currentZone} (강도: ${zoneStrength})`);
            window.logManager.addLog(`  - 가격변동: ${priceChangeText}`);
            window.logManager.addLog(`  - N/B 코인: ${nbCoins}개`);
            window.logManager.addLog(`  - 최종 예상 수익률: ${buyProfitRate.toFixed(2)}%`);
        }
        
        // 매수 수익률 계산 상세 로그 파일 작성
        const buyLogContent = `🔵 BLUE 구역 BTC 시장 탐색 - 매수 전 예상 수익률 계산 상세 분석:
1. BLUE 신호 기반: ${currentMajority} (${currentMajority.includes('BLUE') ? '1.2~3.0%' : currentMajority.includes('ORANGE') ? '0.3~1.0%' : '-0.2~0.2%'})
2. 현재 보유 자산 수익률: ${btcBalance > 0 && avgPrice > 0 ? ((currentPrice - avgPrice) / avgPrice * 100).toFixed(2) + '% (40% 반영)' : 'N/A'}
3. 가격 변동률 반영: ${priceChangeText} (40% 반영)
4. 구역 강도 반영: ${zoneStrength} (강도 50 기준 ±0.75% 보정)
5. N/B 코인 보정: ${nbCoins}개 (코인 1개당 +0.03%)
6. 포트폴리오 비율: BTC ${btcBalance > 0 ? ((btcBalance * currentPrice) / ((btcBalance * currentPrice) + krwBalance) * 100).toFixed(1) + '%' : '0%'}
7. BLUE 구역 특별 보너스: ${currentZone.includes('BLUE') || currentMajority.includes('BLUE') ? '0.2~0.5%' : '0%'}
8. 최종 예상 수익률: ${buyProfitRate.toFixed(2)}%

계산 근거:
- 현재가: ₩${currentPrice.toLocaleString()}
- 평균단가: ₩${avgPrice.toLocaleString()}
- BTC 보유량: ${btcBalance} BTC
- KRW 보유량: ₩${krwBalance.toLocaleString()}
- 시장 신호: ${currentMajority}
- 구역: ${currentZone} (강도: ${zoneStrength})
- 가격변동: ${priceChangeText}
- N/B 코인: ${nbCoins}개
- 계산 위치: BLUE 구역 BTC 시장 탐색 구역`;
        
        this.writeProfitRateLog(this.buyProfitLogFile, buyLogContent);
        
        // 트레이너 대화창 업데이트
        if (window.trainerDialog) {
            const currentTime = new Date().toLocaleTimeString();
            const dialogText = `🔵 BLUE 구역 BTC 시장 탐색 완료: 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}% | 현재가: ₩${currentPrice.toLocaleString()} | 시간: ${currentTime}`;
            window.trainerDialog.setText(dialogText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
            }
        }
        
        // 계산 완료 후 신호 대기 센터로 이동
        if (window.trainerMovementController) {
            window.trainerMovementController.moveToSignalCenter(model, config);
            
            // 이동 목표 정보 로그
            if (window.logManager) {
                const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
                const signalCenterPos = `(${Math.round(config.width / 2)}, ${Math.round(config.height / 2)})`;
                const distanceToSignalCenter = Math.sqrt((model.circle.x - (config.width / 2)) ** 2 + (model.circle.y - (config.height / 2)) ** 2);
                window.logManager.addLog(`🔄 BLUE 구역 탐색 완료 - 신호 대기 센터로 이동: 현재위치 ${currentPos} | 센터위치 ${signalCenterPos} | 센터까지거리 ${Math.round(distanceToSignalCenter)}px`);
            }
        }
        
        // 트레이너 역할 텍스트 업데이트
        model.targetAction = 'BLUE 구역 매수 수익률 계산 완료';
        if (model.role) {
            model.role.setText(`트레이너 (BLUE 구역 매수 수익률 계산 완료)`);
        }
        
        // 새로운 의사결정 필요 플래그 설정
        model.needsNewDecision = true;
        model.arrivalLogged = false;
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 BLUE 구역 BTC 시장 탐색 완료 - 새로운 의사결정 필요 플래그 설정`);
        }
        
        // 수익률 반환
        return buyProfitRate;
    }
    
    // 통합 수익률 계산 함수 (충돌 감지용)
    calculateProfitRates(model, config) {
        try {
            // 매수 수익률 계산
            const buyRate = this.calculateBuyProfitRate(model, config);
            
            // 매도 수익률 계산 (기본 매수가 사용)
            const sellRate = this.calculateSellProfitRate(model, config);
            
            return {
                buyRate: buyRate,
                sellRate: sellRate
            };
        } catch (error) {
            console.error('수익률 계산 오류:', error);
            return {
                buyRate: 0,
                sellRate: 0
            };
        }
    }
    
    // 매도 전 예상 수익률 계산
    calculateSellProfitRate(model, config) {
        // 실제 거래 데이터 가져오기
        const majorityElement = document.getElementById('majority-zone');
        const currentPriceElement = document.getElementById('right-trading-current-price');
        const priceChangeElement = document.getElementById('right-trading-price-change');
        const currentZoneElement = document.getElementById('right-trading-current-zone');
        const zoneStrengthElement = document.getElementById('right-trading-zone-strength');
        const btcBalanceElement = document.getElementById('btc-balance');
        const krwBalanceElement = document.getElementById('krw-balance');
        const avgPriceElement = document.getElementById('selected-coin-avg-price');
        const pnlElement = document.getElementById('selected-coin-pnl');
        
        if (!majorityElement || !currentPriceElement) {
            if (window.logManager) {
                window.logManager.addLog(`❌ BTC 시장 데이터를 가져올 수 없음`);
            }
            return;
        }
        
        // 실제 거래 데이터 파싱
        const currentMajority = majorityElement.textContent.trim();
        const currentPriceText = currentPriceElement.textContent;
        const priceChangeText = priceChangeElement ? priceChangeElement.textContent : '0%';
        const currentZone = currentZoneElement ? currentZoneElement.textContent : 'Unknown';
        const zoneStrength = zoneStrengthElement ? zoneStrengthElement.textContent.match(/\d+/)?.[0] || '0' : '0';
        const btcBalance = btcBalanceElement ? parseFloat(btcBalanceElement.textContent.match(/[\d.]+/)?.[0] || '0') : 0;
        const krwBalance = krwBalanceElement ? parseFloat(krwBalanceElement.textContent.match(/[\d,]+/)?.[0].replace(/,/g, '') || '0') : 0;
        const avgPriceText = avgPriceElement ? avgPriceElement.textContent : '';
        
        // 현재 가격 파싱
        const currentPrice = parseFloat(currentPriceText.replace(/[₩,]/g, ''));
        
        // 평균 단가 파싱
        const avgPriceMatch = avgPriceText.match(/[\d,]+/);
        const avgPrice = avgPriceMatch ? parseFloat(avgPriceMatch[0].replace(/,/g, '')) : currentPrice;
        
        // N/B 코인 개수 가져오기
        const nbCoins = window.gameInitializer ? window.gameInitializer.gameData.nbCoins : 0;
        
        // 매도 전 예상 수익률 계산 로직 (실제 데이터 기반)
        let sellProfitRate = 0;
        
        // 1. 현재 수익률 기반 (실제 보유 자산 기준)
        if (btcBalance > 0 && avgPrice > 0) {
            const currentProfitRate = ((currentPrice - avgPrice) / avgPrice) * 100;
            sellProfitRate += currentProfitRate * 0.5; // 현재 수익률의 50% 반영 (매도는 더 민감)
        }
        
        // 2. 시장 신호에 따른 기본 수익률 (매도는 반대)
        if (currentMajority.includes('BLUE')) {
            sellProfitRate += -1.0 + Math.random() * 1; // -1% ~ 0%
        } else if (currentMajority.includes('ORANGE')) {
            sellProfitRate += -0.5 + Math.random() * 0.5; // -0.5% ~ 0%
        } else {
            sellProfitRate += 0.5 + Math.random() * 1; // 0.5% ~ 1.5%
        }
        
        // 3. 가격 변동률 반영 (매도는 반대)
        const priceChangeMatch = priceChangeText.match(/-?[\d.]+/);
        if (priceChangeMatch) {
            const priceChange = parseFloat(priceChangeMatch[0]);
            sellProfitRate -= priceChange * 0.3; // 가격 상승 시 매도 유리
        }
        
        // 4. 구역 강도 반영 (매도는 반대)
        const strength = parseInt(zoneStrength);
        if (strength > 0) {
            sellProfitRate -= (strength - 50) * 0.01; // 강도 높을 때 매도 유리
        }
        
        // 5. N/B 코인 개수에 따른 보정
        if (nbCoins > 0) {
            sellProfitRate += (nbCoins * 0.03); // 코인 1개당 0.03% 추가
        }
        
        // 6. 포트폴리오 비율 반영
        const totalValue = (btcBalance * currentPrice) + krwBalance;
        if (totalValue > 0) {
            const btcRatio = (btcBalance * currentPrice) / totalValue;
            if (btcRatio > 0.8) {
                sellProfitRate += 0.3; // BTC 비중이 높으면 매도 유리
            } else if (btcRatio < 0.2) {
                sellProfitRate -= 0.3; // BTC 비중이 낮으면 매도 불리
            }
        }
        
        // 수익률 범위 제한 (-15% ~ 10%)
        sellProfitRate = Math.max(-15, Math.min(10, sellProfitRate));
        
        // 비정상적인 수익률 값 검증
        if (sellProfitRate < -15 || sellProfitRate > 10) {
            console.warn(`⚠️ 비정상적인 매도 수익률 감지: ${sellProfitRate.toFixed(2)}%, 0으로 재설정`);
            sellProfitRate = 0;
        }
        
        // 계산된 수익률을 전역 변수에 저장
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.sellProfitRate = sellProfitRate;
        }
        
        // UI 업데이트
        if (window.sellProfitRateDisplay) {
            const profitRateText = `매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
            window.sellProfitRateDisplay.setText(profitRateText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(매도수익률): ${profitRateText}`);
            }
        }
        
        // 상세 계산 로그
        if (window.logManager) {
            window.logManager.addLog(`📊 매도 전 예상 수익률 계산 상세:`);
            window.logManager.addLog(`  - 현재가: ₩${currentPrice.toLocaleString()}`);
            window.logManager.addLog(`  - 평균단가: ₩${avgPrice.toLocaleString()}`);
            window.logManager.addLog(`  - BTC 보유량: ${btcBalance} BTC`);
            window.logManager.addLog(`  - KRW 보유량: ₩${krwBalance.toLocaleString()}`);
            window.logManager.addLog(`  - 시장 신호: ${currentMajority}`);
            window.logManager.addLog(`  - 구역: ${currentZone} (강도: ${zoneStrength})`);
            window.logManager.addLog(`  - 가격변동: ${priceChangeText}`);
            window.logManager.addLog(`  - N/B 코인: ${nbCoins}개`);
            window.logManager.addLog(`  - 최종 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
        }
        
        // 매도 수익률 계산 상세 로그 파일 작성
        const sellLogContent = `매도 수익률 계산 상세 분석:
1. 현재 수익률 기반: ${btcBalance > 0 && avgPrice > 0 ? ((currentPrice - avgPrice) / avgPrice * 100).toFixed(2) + '% (50% 반영)' : 'N/A'}
2. 시장 신호 기반: ${currentMajority} (${currentMajority.includes('BLUE') ? '-1~0%' : currentMajority.includes('ORANGE') ? '-0.5~0%' : '0.5~1.5%'})
3. 가격 변동률 반영: ${priceChangeText} (반대 방향 30% 반영)
4. 구역 강도 반영: ${zoneStrength} (강도 높을 때 매도 유리)
5. N/B 코인 보정: ${nbCoins}개 (코인 1개당 +0.03%)
6. 포트폴리오 비율: BTC ${btcBalance > 0 ? ((btcBalance * currentPrice) / ((btcBalance * currentPrice) + krwBalance) * 100).toFixed(1) + '%' : '0%'}
7. 최종 예상 수익률: ${sellProfitRate.toFixed(2)}%

계산 근거:
- 현재가: ₩${currentPrice.toLocaleString()}
- 평균단가: ₩${avgPrice.toLocaleString()}
- BTC 보유량: ${btcBalance} BTC
- KRW 보유량: ₩${krwBalance.toLocaleString()}
- 시장 신호: ${currentMajority}
- 구역: ${currentZone} (강도: ${zoneStrength})
- 가격변동: ${priceChangeText}
- N/B 코인: ${nbCoins}개`;
        
        this.writeProfitRateLog(this.sellProfitLogFile, sellLogContent);
        
        // 트레이너 대화창 업데이트
        if (window.trainerDialog) {
            const currentTime = new Date().toLocaleTimeString();
            const dialogText = `📊 BTC 시장 탐색: 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}% | 현재가: ₩${currentPrice.toLocaleString()} | 시간: ${currentTime}`;
            window.trainerDialog.setText(dialogText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
            }
        }
        
        // 수익률 반환
        return sellProfitRate;
    }
    
    // 전역 함수로 로그 확인 기능 추가
    static addGlobalFunctions() {
        // 모든 수익률 로그 확인
        window.checkProfitLogs = () => {
            if (window.trainerDecisionHandler) {
                window.trainerDecisionHandler.checkAllProfitLogs();
            } else {
                //console.log('❌ trainerDecisionHandler가 초기화되지 않음');
            }
        };
        
        // 특정 로그 파일 확인
        window.checkSpecificLog = (filename) => {
            if (window.trainerDecisionHandler) {
                const content = window.trainerDecisionHandler.getLogFromLocalStorage(filename);
                if (content) {
                    //console.log(`📊 ${filename} 로그 내용:`);
                    //console.log(content);
                }
            } else {
                //console.log('❌ trainerDecisionHandler가 초기화되지 않음');
            }
        };
        
        //console.log('🔧 전역 로그 확인 함수 추가됨: checkProfitLogs(), checkSpecificLog(filename)');
        
        // 테스트용 로그 생성 함수
        window.testProfitLogs = () => {
            if (window.trainerDecisionHandler) {
                // 테스트 매수 로그 생성
                window.trainerDecisionHandler.writeProfitRateLog(
                    window.trainerDecisionHandler.buyProfitLogFile, 
                    '테스트 매수 수익률 계산: 2.5% (테스트용)'
                );
                
                // 테스트 매도 로그 생성
                window.trainerDecisionHandler.writeProfitRateLog(
                    window.trainerDecisionHandler.sellProfitLogFile, 
                    '테스트 매도 수익률 계산: 1.8% (테스트용)'
                );
                
                //console.log('🧪 테스트 로그 생성 완료');
            } else {
                //console.log('❌ trainerDecisionHandler가 초기화되지 않음');
            }
        };
        
        // 작업 완료 시스템 디버그 함수
        window.debugTaskSystem = () => {
            if (window.trainerDecisionHandler) {
                window.trainerDecisionHandler.debugTaskSystem();
            } else {
                //console.log('❌ trainerDecisionHandler가 초기화되지 않음');
            }
        };
    }

    // 상태 리셋
    resetState() {
        this.waitCheckTimer = 0;
        this.waitStartTime = null;
        this.countdownStarted = false;
        this.btcExplorationMode = false;
        this.arrivalLogged = false;
        // 구역 단계도 리셋
        this.zoneSteps = {};
        // 작업 완료 시스템도 리셋
        this.resetTaskSystem();
    }

    // 구역별 프로세스 단계 처리 (새로 추가)
    handleZoneProcess(model, currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        // 구역별 프로세스 단계 정의
        const zoneProcesses = {
            '매수영역': {
                steps: [
                    { name: '매수 구역 도착 확인', action: '매수 구역 도착 확인' },
                    { name: '매수 수익률 계산', action: '매수 수익률 계산' },
                    { name: '매수 의사결정', action: '매수 의사결정' },
                    { name: '매수 실행', action: '매수 실행' },
                    { name: 'N/B 코인 드랍', action: 'N/B 코인 드랍' },
                    { name: '신호 대기 센터 이동', action: '신호 대기 센터 이동' }
                ],
                currentStep: this.getZoneStep(currentZone, '매수영역')
            },
            '매도영역': {
                steps: [
                    { name: '매도 구역 도착 확인', action: '매도 구역 도착 확인' },
                    { name: '매도 수익률 계산', action: '매도 수익률 계산' },
                    { name: '매도 의사결정', action: '매도 의사결정' },
                    { name: '매도 실행', action: '매도 실행' },
                    { name: '신호 대기 센터 이동', action: '신호 대기 센터 이동' }
                ],
                currentStep: this.getZoneStep(currentZone, '매도영역')
            },
            'BTC시장탐색구역': {
                steps: [
                    { name: 'BTC 시장 도착 확인', action: 'BTC 시장 도착 확인' },
                    { name: '매수 수익률 계산', action: '매수 수익률 계산' },
                    { name: '매도 수익률 계산', action: '매도 수익률 계산' },
                    { name: '시장 분석 완료', action: '시장 분석 완료' },
                    { name: '신호 대기 센터 이동', action: '신호 대기 센터 이동' }
                ],
                currentStep: this.getZoneStep(currentZone, 'BTC시장탐색구역')
            },
            'N/B길드': {
                steps: [
                    { name: 'N/B 길드 도착 확인', action: 'N/B 길드 도착 확인' },
                    { name: 'N/B 코인 확인', action: 'N/B 코인 확인' },
                    { name: '매도 수익률 계산', action: '매도 수익률 계산' },
                    { name: '매도 의사결정', action: '매도 의사결정' },
                    { name: '신호 대기 센터 이동', action: '신호 대기 센터 이동' }
                ],
                currentStep: this.getZoneStep(currentZone, 'N/B길드')
            },
            '신호대기센터': {
                steps: [
                    { name: '신호 대기 센터 도착 확인', action: '신호 대기 센터 도착 확인' },
                    { name: '시장 신호 분석', action: '시장 신호 분석' },
                    { name: '다음 목적지 결정', action: '다음 목적지 결정' },
                    { name: '목적지로 이동', action: '목적지로 이동' }
                ],
                currentStep: this.getZoneStep(currentZone, '신호대기센터')
            }
        };

        const process = zoneProcesses[currentZone];
        if (!process) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 구역 프로세스 없음: ${currentZone}`);
            }
            return null;
        }

        // process.steps가 존재하고 유효한지 확인
        if (!process.steps || !Array.isArray(process.steps) || process.steps.length === 0) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 구역 프로세스 단계가 없음: ${currentZone}`);
            }
            return null;
        }

        // process.currentStep이 유효한 범위인지 확인
        if (process.currentStep < 0 || process.currentStep >= process.steps.length) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 구역 프로세스 단계 인덱스가 범위를 벗어남: ${currentZone} - 단계 ${process.currentStep + 1}/${process.steps.length}`);
            }
            return null;
        }

        // 현재 단계 실행
        const currentStep = process.steps[process.currentStep];
        
        // currentStep이 존재하는지 확인
        if (!currentStep) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 현재 단계가 존재하지 않음: ${currentZone} - 단계 ${process.currentStep + 1}/${process.steps.length}`);
            }
            return null;
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 구역 프로세스: ${currentZone} - 단계 ${process.currentStep + 1}/${process.steps.length}: ${currentStep.name}`);
        }

        // 단계별 처리
        if (!currentStep.action) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 현재 단계에 action이 없음: ${currentZone} - 단계 ${process.currentStep + 1}`);
            }
            return null;
        }
        
        switch (currentStep.action) {
            case '매수 구역 도착 확인':
                return this.handleBuyZoneArrival(model, startX, topY, spacing, config);
            case '매수 수익률 계산':
                return this.handleBuyProfitCalculation(model, currentMajority, config);
            case '매수 의사결정':
                return this.handleBuyDecision(model, buyProfitRate, startX, topY, spacing, config);
            case '매수 실행':
                return this.handleBuyExecution(model, config);
            case 'N/B 코인 드랍':
                return this.handleNBCoinDrop(model, config);
            case '매도 구역 도착 확인':
                return this.handleSellZoneArrival(model, startX, topY, spacing, config);
            case '매도 수익률 계산':
                return this.handleSellProfitCalculation(model, currentMajority, config);
            case '매수 수익률 계산':
                return this.handleBuyProfitCalculation(model, currentMajority, config);
            case '매도 의사결정':
                return this.handleSellDecision(model, sellProfitRate, startX, topY, spacing, config);
            case '매도 실행':
                return this.handleSellExecution(model, config);
            case 'BTC 시장 도착 확인':
                return this.handleBTCMarketArrival(model, config);
            case '시장 분석 완료':
                return this.handleMarketAnalysisComplete(model, config);
            case 'N/B 길드 도착 확인':
                return this.handleNBGuildArrival(model, config);
            case 'N/B 코인 확인':
                return this.handleNBCoinCheck(model, nbCoins, config);
            case '신호 대기 센터 도착 확인':
                return this.handleSignalCenterArrival(model, config);
            case '시장 신호 분석':
                return this.handleMarketSignalAnalysis(model, currentMajority, config);
            case '다음 목적지 결정':
                return this.handleNextDestinationDecision(model, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
            case '목적지로 이동':
                return this.handleDestinationMove(model, config);
            case '신호 대기 센터 이동':
                return this.handleMoveToSignalCenter(model, config);
            default:
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ 알 수 없는 프로세스 단계: ${currentStep.action}`);
                }
                return null;
        }
    }

    // 구역별 단계 관리
    getZoneStep(currentZone, zoneType) {
        if (!this.zoneSteps) {
            this.zoneSteps = {};
        }
        if (!this.zoneSteps[zoneType]) {
            this.zoneSteps[zoneType] = 0;
        }
        return this.zoneSteps[zoneType];
    }

    // 다음 단계로 진행
    nextZoneStep(zoneType) {
        if (!this.zoneSteps) {
            this.zoneSteps = {};
        }
        if (!this.zoneSteps[zoneType]) {
            this.zoneSteps[zoneType] = 0;
        }
        this.zoneSteps[zoneType]++;
        
        if (window.logManager) {
            window.logManager.addLog(`➡️ 구역 단계 진행: ${zoneType} → 단계 ${this.zoneSteps[zoneType]}`);
        }
    }

    // 구역 단계 리셋
    resetZoneStep(zoneType) {
        if (!this.zoneSteps) {
            this.zoneSteps = {};
        }
        this.zoneSteps[zoneType] = 0;
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 구역 단계 리셋: ${zoneType}`);
        }
    }

    // 구역별 단계 처리 메서드들
    handleBuyZoneArrival(model, startX, topY, spacing, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 구역 도착 확인 완료`);
        }
        this.nextZoneStep('매수영역');
        return '매수 수익률 계산';
    }

    handleBuyProfitCalculation(model, currentMajority, config) {
        // 수정된 부분: 올바른 매개변수로 호출
        const buyProfitRate = this.calculateBuyProfitRate(model, config);
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 수익률 계산 완료: ${buyProfitRate.toFixed(2)}%`);
        }
        this.nextZoneStep('매수영역');
        return '매수 의사결정';
    }

    handleBuyDecision(model, buyProfitRate, startX, topY, spacing, config) {
        // 현재 시장 신호 확인
        const majorityElement = document.getElementById('majority-zone');
        const currentMajority = majorityElement ? majorityElement.textContent.trim() : '';
        
        // N/B 코인과 드랍 아이템 개수 확인
        const nbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
        const dropItemsCount = window.gameInitializer?.gameData?.dropItemsCount || 0;
        
        // 매수 조건: BLUE 신호 + N/B 코인 0개 + 드랍 아이템 0개 + 수익률 조건
        const shouldBuy = currentMajority.includes('BLUE') && 
                         nbCoins <= 0 && 
                         dropItemsCount <= 0 && 
                         (buyProfitRate > 0 || buyProfitRate > -1.0);
        
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 의사결정 완료: 신호=${currentMajority}, N/B코인=${nbCoins}개, 드랍아이템=${dropItemsCount}개, 수익률=${buyProfitRate?.toFixed(2) || 'N/A'}%, 매수결정=${shouldBuy ? '예' : '아니오'}`);
        }
        
        this.nextZoneStep('매수영역');
        
        // 매수 조건이 맞으면 매수 실행
        if (shouldBuy) {
            return '매수 실행';
        } 
        // N/B 코인이 0개이지만 드랍 아이템이 남아있으면 N/B 길드로 이동
        else if (currentMajority.includes('BLUE') && nbCoins <= 0 && dropItemsCount > 0) {
            if (window.logManager) {
                window.logManager.addLog(`⏸️ 매수 조건 불만족: BLUE 신호 + N/B코인 0개이지만 드랍 아이템 ${dropItemsCount}개 남음 → N/B 길드로 이동`);
            }
            this.nextZoneStep('N/B길드');
            return 'N/B 길드 방문';
        }
        // 그 외의 경우 신호 대기 센터로 복귀
        else {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 매수 조건 불만족 - 신호 대기 센터로 복귀`);
            }
            return '신호 대기 센터 이동';
        }
    }

    handleBuyExecution(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 실행 시작`);
        }
        
        // 실제 매수 실행 로직 추가
        if (window.trainerStateHandler) {
            // 현재 거래 데이터 가져오기
            const majorityElement = document.getElementById('majority-zone');
            const currentMajority = majorityElement ? majorityElement.textContent.trim() : 'BLUE';
            const nbCoins = window.gameInitializer ? window.gameInitializer.gameData.nbCoins : 0;
            const buyProfitRate = window.buyProfitRate || 0;
            const sellProfitRate = window.sellProfitRate || 0;
            
            // 실제 매수 액션 처리
            window.trainerStateHandler.handleBuyAction(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            
            if (window.logManager) {
                window.logManager.addLog(`💰 실제 매수 실행 완료 - N/B 코인: ${nbCoins}개`);
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`❌ 매수 실행 실패: trainerStateHandler가 초기화되지 않았습니다.`);
            }
        }
        
        this.nextZoneStep('매수영역');
        return 'N/B 코인 드랍';
    }

    handleNBCoinDrop(model, config) {
        if (window.nbCoinDropSystem) {
            // 현재 타임프레임 정보 가져오기
            let currentTimeframe = null;
            
            // 방법 1: 모델에서 타임프레임 정보 가져오기
            if (model.timeframe) {
                currentTimeframe = model.timeframe;
            }
            // 방법 2: 전역 상태에서 현재 타임프레임 가져오기
            else if (window.currentTimeframe) {
                currentTimeframe = window.currentTimeframe;
            }
            // 방법 3: 좌측 패널에서 현재 선택된 타임프레임 가져오기
            else {
                const selectedCard = document.querySelector('.timeframe-card.selected');
                if (selectedCard) {
                    currentTimeframe = selectedCard.getAttribute('data-timeframe');
                }
            }
            
            // sourceTimeframe 정보와 함께 드랍
            const coinItem = window.nbCoinDropSystem.dropNBCoin(model.circle.x, model.circle.y, currentTimeframe);
            
            if (coinItem) {
                if (window.logManager) {
                    const timeframeInfo = currentTimeframe ? `, 분봉: ${currentTimeframe}` : '';
                    window.logManager.addLog(`✅ N/B 코인 드랍 완료 - 위치: (${Math.round(coinItem.position.x)}, ${Math.round(coinItem.position.y)})${timeframeInfo}`);
                }
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ N/B 코인 드랍 실패 - 쿨다운 또는 최대 개수 제한`);
                }
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`❌ N/B 코인 드랍 시스템이 초기화되지 않았습니다.`);
            }
        }
        this.nextZoneStep('매수영역');
        return '신호 대기 센터 이동';
    }

    handleSellZoneArrival(model, startX, topY, spacing, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 구역 도착 확인 완료`);
        }
        this.nextZoneStep('매도영역');
        return '매도 수익률 계산';
    }

    handleBuyProfitCalculation(model, currentMajority, config) {
        // 매수 수익률 계산
        const buyProfitRate = this.calculateBuyProfitRate(model, config);
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 수익률 계산 완료: ${buyProfitRate.toFixed(2)}%`);
        }
        
        // 현재 구역에 따라 올바른 구역 단계 진행
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, startX, topY, spacing, config);
        
        // BTC시장탐색구역에서는 BTC시장탐색구역 단계로 진행하여 계속 계산
        if (currentZone === 'BTC시장탐색구역') {
            this.nextZoneStep('BTC시장탐색구역');
            if (window.logManager) {
                window.logManager.addLog(`🔄 BTC 시장 탐색 구역에서 매수 수익률 계산 계속 진행`);
            }
        } else if (currentZone === '매수영역') {
            this.nextZoneStep('매수영역');
        } else {
            this.nextZoneStep(currentZone);
        }
        
        return '매수 의사결정';
    }

    handleSellProfitCalculation(model, currentMajority, config) {
        // 수정된 부분: 올바른 매개변수로 호출
        const sellProfitRate = this.calculateSellProfitRate(model, config);
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 수익률 계산 완료: ${sellProfitRate.toFixed(2)}%`);
        }
        
        // 현재 구역에 따라 올바른 구역 단계 진행
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, startX, topY, spacing, config);
        
        // N/B길드 구역에서는 N/B길드 단계로 진행하여 계속 계산
        if (currentZone === 'N/B길드') {
            this.nextZoneStep('N/B길드');
            if (window.logManager) {
                window.logManager.addLog(`🔄 N/B 길드 구역에서 매도 수익률 계산 계속 진행`);
            }
        } else if (currentZone === '매도영역') {
            this.nextZoneStep('매도영역');
        } else {
            this.nextZoneStep(currentZone);
        }
        
        return '매도 의사결정';
    }

    handleSellDecision(model, sellProfitRate, startX, topY, spacing, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 의사결정 완료: 수익률 ${sellProfitRate?.toFixed(2) || 'N/A'}%`);
        }
        
        // 현재 구역에 따라 올바른 구역 단계 진행
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, startX, topY, spacing, config);
        
        if (currentZone === 'N/B길드') {
            this.nextZoneStep('N/B길드');
        } else if (currentZone === '매도영역') {
            this.nextZoneStep('매도영역');
        } else {
            this.nextZoneStep(currentZone);
        }
        
        return '매도 실행';
    }

    handleSellExecution(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 실행 완료`);
        }
        
        // 현재 구역에 따라 올바른 구역 단계 진행
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;
        const spacing = 120;
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, startX, topY, spacing, config);
        
        if (currentZone === 'N/B길드') {
            this.nextZoneStep('N/B길드');
        } else if (currentZone === '매도영역') {
            this.nextZoneStep('매도영역');
        } else {
            this.nextZoneStep(currentZone);
        }
        
        return '신호 대기 센터 이동';
    }

    handleBTCMarketArrival(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ BTC 시장 도착 확인 완료`);
        }
        this.nextZoneStep('BTC시장탐색구역');
        return '매수 수익률 계산';
    }

    handleMarketAnalysisComplete(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 시장 분석 완료`);
        }
        this.nextZoneStep('BTC시장탐색구역');
        return '신호 대기 센터 이동';
    }

    handleNBGuildArrival(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ N/B 길드 도착 확인 완료`);
        }
        this.nextZoneStep('N/B길드');
        return 'N/B 코인 확인';
    }

    handleNBCoinCheck(model, nbCoins, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ N/B 코인 확인 완료: ${nbCoins}개`);
        }
        this.nextZoneStep('N/B길드');
        return '매도 수익률 계산';
    }

    handleSignalCenterArrival(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 신호 대기 센터 도착 확인 완료`);
        }
        this.nextZoneStep('신호대기센터');
        return '시장 신호 분석';
    }

    handleMarketSignalAnalysis(model, currentMajority, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 시장 신호 분석 완료: ${currentMajority}`);
        }
        this.nextZoneStep('신호대기센터');
        return '다음 목적지 결정';
    }

    handleNextDestinationDecision(model, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 다음 목적지 결정 완료`);
        }
        this.nextZoneStep('신호대기센터');
        return '목적지로 이동';
    }

    handleDestinationMove(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 목적지로 이동 시작`);
        }
        this.nextZoneStep('신호대기센터');
        return '이동 중';
    }

    handleMoveToSignalCenter(model, config) {
        model.targetX = config.width / 2;
        model.targetY = config.height / 2;
        if (window.logManager) {
            window.logManager.addLog(`✅ 신호 대기 센터로 이동 시작`);
        }
        this.resetZoneStep('매수영역');
        this.resetZoneStep('매도영역');
        this.resetZoneStep('BTC시장탐색구역');
        this.resetZoneStep('N/B길드');
        return '신호 대기';
    }

    // 트레이너 의사결정 핸들러 재시작
    restart() {
        //console.log('🔄 트레이너 의사결정 핸들러 재시작 시작...');
        
        try {
            // 기본 상태로 초기화
            this.waitCheckTimer = 0;
            this.waitStartTime = null;
            this.countdownStarted = false;
            this.btcExplorationMode = false;
            this.arrivalLogged = false;
            // 작업 완료 시스템 리셋
            this.resetTaskSystem();
            
            // 수익률 계산 로그 파일 재초기화
            this.initProfitRateLogs();
            
            // 트레이너 모델 찾기 및 재시작
            if (window.aiModels && Array.isArray(window.aiModels)) {
                const trainerModel = window.aiModels.find(model => model.isTrainer);
                if (trainerModel) {
                    this.restartTrainerDecision(trainerModel);
                }
            }
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 트레이너 의사결정 핸들러 재시작 완료`);
            }
            
            //console.log('✅ 트레이너 의사결정 핸들러 재시작 완료');
        } catch (error) {
            console.error('❌ 트레이너 의사결정 핸들러 재시작 실패:', error);
        }
    }

    // 트레이너 의사결정 재시작
    restartTrainerDecision(model) {
        if (!model) {
            //console.log('❌ 트레이너 의사결정 재시작 실패: 모델이 유효하지 않음');
            return;
        }
        
        try {
            // 트레이너 의사결정 상태 초기화
            model.targetAction = '신호 대기';
            model.needsNewDecision = false;
            model.arrivalLogged = false;
            
            // 기본 의사결정 설정
            this.setDefaultTrainerDecision(model);
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 트레이너 의사결정 재시작 완료 - 기본 액션: ${model.targetAction}`);
            }
            
            //console.log('🎯 트레이너 의사결정 재시작 완료');
        } catch (error) {
            console.error('❌ 트레이너 의사결정 재시작 실패:', error);
        }
    }

    // 기본 트레이너 의사결정 설정
    setDefaultTrainerDecision(model) {
        if (!model) return;
        
        try {
            // 현재 시장 상황에 따른 기본 의사결정
            const currentTime = new Date();
            const hour = currentTime.getHours();
            
            // 시간대별 기본 의사결정
            if (hour >= 9 && hour <= 17) {
                // 거래 시간대 - 시장 분석 모드
                model.targetAction = '시장 신호 분석';
            } else if (hour >= 18 && hour <= 23) {
                // 저녁 시간대 - BTC 탐색 모드
                model.targetAction = 'BTC 시장 탐색';
                this.btcExplorationMode = true;
            } else {
                // 새벽 시간대 - 신호 대기 모드
                model.targetAction = '신호 대기';
            }
            
            // 역할 텍스트 업데이트
            if (model.role) {
                model.role.setText(`트레이너 (${model.targetAction})`);
            }
            
            //console.log(`🎯 기본 트레이너 의사결정 설정: ${model.targetAction}`);
        } catch (error) {
            console.error('❌ 기본 트레이너 의사결정 설정 실패:', error);
        }
    }

    // 트레이너 의사결정 상태 확인
    getTrainerDecisionStatus() {
        return {
            waitCheckTimer: this.waitCheckTimer,
            waitStartTime: this.waitStartTime,
            countdownStarted: this.countdownStarted,
            btcExplorationMode: this.btcExplorationMode,
            arrivalLogged: this.arrivalLogged
        };
    }

    // 트레이너 의사결정 상태 설정
    setTrainerDecisionStatus(status) {
        if (status.waitCheckTimer !== undefined) this.waitCheckTimer = status.waitCheckTimer;
        if (status.waitStartTime !== undefined) this.waitStartTime = status.waitStartTime;
        if (status.countdownStarted !== undefined) this.countdownStarted = status.countdownStarted;
        if (status.btcExplorationMode !== undefined) this.btcExplorationMode = status.btcExplorationMode;
        if (status.arrivalLogged !== undefined) this.arrivalLogged = status.arrivalLogged;
    }

    // 작업 완료 기반 의사결정 시스템 메서드들
    
    // 작업 시작
    startZoneTask(zone, model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing) {
        this.taskCompletionSystem.currentZone = zone;
        this.taskCompletionSystem.taskCompleted = false;
        this.taskCompletionSystem.taskStartTime = Date.now();
        
        // 모든 구역 작업 상태 리셋
        Object.keys(this.taskCompletionSystem.zoneTasks).forEach(zoneKey => {
            this.taskCompletionSystem.zoneTasks[zoneKey].completed = false;
        });
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 ${zone} 작업 시작: ${this.taskCompletionSystem.zoneTasks[zone]?.name || '알 수 없는 작업'}`);
        }
        
        // 구역별 작업 실행
        switch (zone) {
            case '매수영역':
                return this.handleBuyAreaTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            case '매도영역':
                return this.handleSellAreaTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            case 'BTC시장':
                return this.handleBTCMarketTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            case 'N/B길드':
                return this.handleNBGuildTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            case '신호대기센터':
                return this.handleSignalCenterTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
            default:
                return this.handleSignalCenterTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate);
        }
    }
    
    // 작업 완료 확인
    checkTaskCompletion(zone) {
        if (!this.taskCompletionSystem.currentZone || this.taskCompletionSystem.currentZone !== zone) {
            return false;
        }
        
        const task = this.taskCompletionSystem.zoneTasks[zone];
        if (!task) {
            return false;
        }
        
        // 타임아웃 확인
        const currentTime = Date.now();
        const elapsedTime = currentTime - this.taskCompletionSystem.taskStartTime;
        
        if (elapsedTime > this.taskCompletionSystem.taskTimeout) {
            if (window.logManager) {
                window.logManager.addLog(`⏰ ${zone} 작업 타임아웃 (${this.taskCompletionSystem.taskTimeout / 1000}초) - 작업 완료로 처리`);
            }
            this.completeZoneTask(zone);
            return true;
        }
        
        return task.completed;
    }
    
    // 작업 완료 처리
    completeZoneTask(zone) {
        const task = this.taskCompletionSystem.zoneTasks[zone];
        if (task) {
            task.completed = true;
            this.taskCompletionSystem.taskCompleted = true;
            
            if (window.logManager) {
                window.logManager.addLog(`✅ ${zone} 작업 완료: ${task.name}`);
            }
            
            // 트레이너 대화창 업데이트
            if (window.trainerDialog) {
                const currentTime = new Date().toLocaleTimeString();
                const dialogText = `✅ 트레이너: ${zone} 작업 완료 (${task.name}) | 시간: ${currentTime}`;
                window.trainerDialog.setText(dialogText);
            }
        }
    }
    
    // 매수 영역 작업 처리
    handleBuyAreaTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (window.logManager) {
            window.logManager.addLog(`💰 매수 영역 작업 시작: 매수 액션 처리`);
        }
        
        // 매수 구역 체크 강화
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, config);
        if (currentZone !== '매수영역') {
            if (window.logManager) {
                window.logManager.addLog(`⛔ 매수 구역에 없음: 현재 구역 ${currentZone} → 매수 액션 취소`);
            }
            return '매수 구역 이동';
        }
        
        // 매수 액션 처리 (실제 매수 로직)
        if (window.trainerStateHandler) {
            // 현재 거래 데이터 가져오기
            const majorityElement = document.getElementById('majority-zone');
            const currentMajority = majorityElement ? majorityElement.textContent.trim() : 'BLUE';
            const currentNbCoins = window.gameInitializer ? window.gameInitializer.gameData.nbCoins : 0;
            
            // 실제 매수 액션 처리
            window.trainerStateHandler.handleBuyAction(model, config, currentMajority, currentNbCoins, buyProfitRate, sellProfitRate);
            
            if (window.logManager) {
                window.logManager.addLog(`💰 매수 액션 처리 완료`);
            }
        }
        
        // 트레이너 색상 변경
        if (window.trainerVisualEffects) {
            window.trainerVisualEffects.changeTrainerColor(model, '매수');
        }
        
        // 작업 완료 처리 (매수 영역은 즉시 완료)
        this.completeZoneTask('매수영역');
        
        return '매수';
    }
    
    // 매도 영역 작업 처리
    handleSellAreaTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (window.logManager) {
            window.logManager.addLog(`💰 매도 영역 작업 시작: 매도 액션 처리`);
        }
        
        // 매도 구역 체크 강화
        const currentZone = this.getCurrentZone(model.circle.x, model.circle.y, config);
        if (currentZone !== '매도영역') {
            if (window.logManager) {
                window.logManager.addLog(`⛔ 매도 구역에 없음: 현재 구역 ${currentZone} → 매도 액션 취소`);
            }
            return '매도 구역 이동';
        }
        
        // 매도 액션 처리 (실제 매도 로직)
        if (window.trainerStateHandler) {
            // 현재 거래 데이터 가져오기
            const majorityElement = document.getElementById('majority-zone');
            const currentMajority = majorityElement ? majorityElement.textContent.trim() : 'ORANGE';
            const currentNbCoins = window.gameInitializer ? window.gameInitializer.gameData.nbCoins : 0;
            
            // 실제 매도 액션 처리
            window.trainerStateHandler.handleSellAction(model, config, currentMajority, currentNbCoins, buyProfitRate, sellProfitRate);
            
            if (window.logManager) {
                window.logManager.addLog(`💰 매도 액션 처리 완료`);
            }
        }
        
        // 트레이너 색상 변경
        if (window.trainerVisualEffects) {
            window.trainerVisualEffects.changeTrainerColor(model, '매도');
        }
        
        // 작업 완료 처리 (매도 영역은 즉시 완료)
        this.completeZoneTask('매도영역');
        
        return '매도';
    }
    
    // BTC 시장 작업 처리 (5초 대기하면서 수익률 계산)
    handleBTCMarketTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        const currentTime = Date.now();
        const elapsedTime = currentTime - this.taskCompletionSystem.taskStartTime;
        const waitTime = 5000; // 5초
        
        if (elapsedTime >= waitTime) {
            // 5초 대기 완료 - 최종 수익률 계산 및 작업 완료
            if (window.logManager) {
                window.logManager.addLog(`📊 BTC 시장 작업 완료: 5초 대기 완료 - 최종 수익률 계산`);
            }
            
            // BTC 시장 학습 핸들러 호출
            if (window.btcMarketLearningHandler) {
                window.btcMarketLearningHandler.handleBTCMarketArrival(model, config, null, currentMajority, null);
            }
            
            // 최종 매수 전 예상 수익률 계산
            const calculatedBuyProfitRate = this.calculateBuyProfitRate(model, config);
            window.buyProfitRate = calculatedBuyProfitRate; // 전역 변수에 저장
            
            if (window.logManager) {
                window.logManager.addLog(`📊 BTC 시장 작업 완료: 매수 전 예상 수익률 ${calculatedBuyProfitRate.toFixed(2)}%`);
            }
            
            // 작업 완료 처리
            this.completeZoneTask('BTC시장');
            
            return 'BTC 시장 방문';
        } else {
            // 5초 대기 중 - 주기적으로 수익률 계산
            const remainingTime = Math.ceil((waitTime - elapsedTime) / 1000);
            
            // 1초마다 수익률 계산 (중간 계산)
            if (Math.floor(elapsedTime / 1000) !== Math.floor((elapsedTime + 16) / 1000)) {
                if (window.logManager) {
                    window.logManager.addLog(`⏰ BTC 시장 대기 중... ${remainingTime}초 남음 - 중간 수익률 계산`);
                }
                
                // 중간 수익률 계산
                const intermediateBuyProfitRate = this.calculateBuyProfitRate(model, config);
                window.buyProfitRate = intermediateBuyProfitRate; // 전역 변수에 저장
                
                if (window.logManager) {
                    window.logManager.addLog(`📊 BTC 시장 중간 계산: 매수 전 예상 수익률 ${intermediateBuyProfitRate.toFixed(2)}%`);
                }
            }
            
            return 'BTC 시장 탐색';
        }
    }
    
    // N/B 길드 작업 처리
    handleNBGuildTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (window.logManager) {
            window.logManager.addLog(`🏛️ N/B 길드 작업 시작: 매도 전 예상 수익률 계산`);
        }
        
        // 매도 전 예상 수익률 계산
        const calculatedSellProfitRate = this.calculateSellProfitRate(model, config);
        
        if (window.logManager) {
            window.logManager.addLog(`🏛️ N/B 길드 작업 완료: 매도 전 예상 수익률 ${calculatedSellProfitRate.toFixed(2)}%`);
        }
        
        // 작업 완료 처리
        this.completeZoneTask('N/B길드');
        
        return 'N/B 길드 방문';
    }
    
    // 신호 대기 센터 작업 처리 (5초 대기)
    handleSignalCenterTask(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate) {
        if (window.logManager) {
            window.logManager.addLog(`🔵 신호 대기 센터 작업 시작: 5초 대기`);
        }
        
        // 신호 대기 센터는 5초 대기 후 완료
        const currentTime = Date.now();
        const elapsedTime = currentTime - this.taskCompletionSystem.taskStartTime;
        const waitTime = 5000; // 5초
        
        if (elapsedTime >= waitTime) {
            if (window.logManager) {
                window.logManager.addLog(`🔵 신호 대기 센터 작업 완료: 5초 대기 완료`);
            }
            
            // 작업 완료 처리
            this.completeZoneTask('신호대기센터');
        } else {
            const remainingTime = Math.ceil((waitTime - elapsedTime) / 1000);
            if (window.logManager && Math.floor(elapsedTime / 1000) !== Math.floor((elapsedTime + 16) / 1000)) {
                window.logManager.addLog(`⏰ 신호 대기 중... ${remainingTime}초 남음`);
            }
        }
        
        return '신호 대기';
    }
    
    // 작업 완료 상태 확인
    isTaskCompleted(zone) {
        const task = this.taskCompletionSystem.zoneTasks[zone];
        return task ? task.completed : false;
    }
    
    // 작업 시스템 리셋
    resetTaskSystem() {
        this.taskCompletionSystem.currentZone = null;
        this.taskCompletionSystem.taskCompleted = false;
        this.taskCompletionSystem.taskStartTime = null;
        
        Object.keys(this.taskCompletionSystem.zoneTasks).forEach(zoneKey => {
            this.taskCompletionSystem.zoneTasks[zoneKey].completed = false;
        });
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 작업 완료 시스템 리셋됨`);
        }
    }
    
    // 작업 시스템 상태 디버그 (전역 함수로 추가)
    debugTaskSystem() {
        //console.log('🔍 작업 완료 시스템 상태:');
        //console.log(`  - 현재 구역: ${this.taskCompletionSystem.currentZone || '없음'}`);
        //console.log(`  - 작업 완료: ${this.taskCompletionSystem.taskCompleted}`);
        //console.log(`  - 작업 시작 시간: ${this.taskCompletionSystem.taskStartTime ? new Date(this.taskCompletionSystem.taskStartTime).toLocaleTimeString() : '없음'}`);
        //console.log('  - 구역별 작업 상태:');
        Object.keys(this.taskCompletionSystem.zoneTasks).forEach(zoneKey => {
            const task = this.taskCompletionSystem.zoneTasks[zoneKey];
            //console.log(`    ${zoneKey}: ${task.completed ? '✅ 완료' : '⏳ 대기'} (${task.name})`);
        });
        
        if (window.logManager) {
            window.logManager.addLog(`🔍 작업 완료 시스템 디버그: 현재구역=${this.taskCompletionSystem.currentZone || '없음'}, 완료=${this.taskCompletionSystem.taskCompleted}`);
        }
    }

    // 현재 구역 확인 함수
    getCurrentZone(x, y, config) {
        const startX = 100;
        const topY = 50;
        const spacing = 120;
        
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
}

// 전역 인스턴스 생성
window.trainerDecisionHandler = new TrainerDecisionHandler();

// cursor ai 1000줄 이상이 되면, 새로운 파일로 분리해서 작업해주세요.