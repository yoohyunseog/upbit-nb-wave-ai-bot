// 액션 프로세스 관리자 모듈
// 트레이너가 목적지에 도착했을 때 실행되는 프로세스를 관리

class ActionProcessManager {
    constructor() {
        this.processStates = {
            BTC_MARKET: 'btc_market_process',
            NB_GUILD: 'nb_guild_process',
            SIGNAL_WAITING: 'signal_waiting_process'
        };
        
        this.currentProcess = null;
        this.processStep = 0;
        this.isProcessing = false;
    }

    // BTC 시장 프로세스 시작
    startBTCMarketProcess(model, config, trainerDialog, currentMajority, buyProfitRateDisplay) {
        console.log('🚀 BTC 시장 프로세스 시작');
        
        this.currentProcess = this.processStates.BTC_MARKET;
        this.processStep = 0;
        this.isProcessing = true;
        
        if (window.logManager) {
            window.logManager.addLog(`🚀 [BTC 시장 프로세스] 트레이너가 BTC 시장에 도착! 프로세스 시작`);
        }
        
        // 프로세스 단계별 실행
        this.executeBTCMarketStep(model, config, trainerDialog, currentMajority, buyProfitRateDisplay);
    }

    // BTC 시장 프로세스 단계 실행
    executeBTCMarketStep(model, config, trainerDialog, currentMajority, buyProfitRateDisplay) {
        if (!this.isProcessing || this.currentProcess !== this.processStates.BTC_MARKET) {
            return;
        }

        switch (this.processStep) {
            case 0:
                // 1단계: 도착 감지 및 시각적 효과
                this.step1_ArrivalDetection(model, config, trainerDialog, currentMajority);
                break;
            case 1:
                // 2단계: 학습 모델 기반 수익률 계산
                this.step2_ProfitCalculation(model, config, trainerDialog, currentMajority, buyProfitRateDisplay);
                break;
            case 2:
                // 3단계: 학습 모델 결과 분석
                this.step3_ModelResultAnalysis(model, config, trainerDialog, currentMajority);
                break;
            case 3:
                // 4단계: 매수 결정 및 실행
                this.step4_BuyDecision(model, config, trainerDialog, currentMajority);
                break;
            case 4:
                // 5단계: 프로세스 완료 및 이동
                this.step5_ProcessCompletion(model, config, trainerDialog);
                break;
            default:
                this.completeProcess(model, config, trainerDialog);
                break;
        }
    }

    // N/B 길드 프로세스 시작
    startNBGuildProcess(model, config, trainerDialog, currentMajority, sellProfitRateDisplay) {
        console.log('🚀 N/B 길드 프로세스 시작');
        
        this.currentProcess = this.processStates.NB_GUILD;
        this.processStep = 0;
        this.isProcessing = true;
        
        if (window.logManager) {
            window.logManager.addLog(`🚀 [N/B 길드 프로세스] 트레이너가 N/B 길드에 도착! 프로세스 시작`);
        }
        
        // 프로세스 단계별 실행
        this.executeNBGuildStep(model, config, trainerDialog, currentMajority, sellProfitRateDisplay);
    }

    // N/B 길드 프로세스 단계 실행
    executeNBGuildStep(model, config, trainerDialog, currentMajority, sellProfitRateDisplay) {
        if (!this.isProcessing || this.currentProcess !== this.processStates.NB_GUILD) {
            return;
        }

        switch (this.processStep) {
            case 0:
                // 1단계: 도착 감지 및 시각적 효과
                this.step1_ArrivalDetection(model, config, trainerDialog, currentMajority);
                break;
            case 1:
                // 2단계: 학습 모델 기반 수익률 계산
                this.step2_SellProfitCalculation(model, config, trainerDialog, currentMajority, sellProfitRateDisplay);
                break;
            case 2:
                // 3단계: 학습 모델 결과 분석
                this.step3_ModelResultAnalysis(model, config, trainerDialog, currentMajority);
                break;
            case 3:
                // 4단계: 매도 결정 및 실행
                this.step4_SellDecision(model, config, trainerDialog, currentMajority);
                break;
            case 4:
                // 5단계: 프로세스 완료 및 이동
                this.step5_ProcessCompletion(model, config, trainerDialog);
                break;
            default:
                this.completeProcess(model, config, trainerDialog);
                break;
        }
    }

    // 1단계: 도착 감지 및 시각적 효과
    step1_ArrivalDetection(model, config, trainerDialog, currentMajority) {
        console.log(`📋 [${this.currentProcess}] 1단계: 도착 감지 및 시각적 효과`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 1단계: 도착 감지 완료 - 시각적 효과 시작`);
        }
        
        // 시각적 효과 실행
        if (this.currentProcess === this.processStates.BTC_MARKET) {
            if (window.btcExplorationManager) {
                window.btcExplorationManager.addBTCMarketVisualEffect();
                window.btcExplorationManager.addTrainerVisualEffect(model);
            }
        } else if (this.currentProcess === this.processStates.NB_GUILD) {
            this.addNBGuildVisualEffect(model);
        }
        
        // 다음 단계로 진행 (1초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 1000);
    }

    // 2단계: 수익률 계산 (BTC 시장)
    step2_ProfitCalculation(model, config, trainerDialog, currentMajority, buyProfitRateDisplay) {
        console.log(`📋 [${this.currentProcess}] 2단계: 학습 모델 기반 수익률 계산`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 2단계: 학습 모델 기반 수익률 계산 시작`);
        }
        
        // BTC 시장 수익률 계산기 호출
        if (window.btcMarketProfitCalculator) {
            const calculationResult = window.btcMarketProfitCalculator.calculateBuyProfitRateAtMarket(
                currentMajority, 
                buyProfitRateDisplay, 
                trainerDialog, 
                model, 
                config, 
                config.width - 100, 
                config.height - 100
            );
            
            if (calculationResult) {
                console.log('✅ BTC 시장 수익률 계산 완료');
                if (window.logManager) {
                    window.logManager.addLog(`✅ [${this.currentProcess}] 2단계: 수익률 계산 완료`);
                }
            } else {
                console.log('⚠️ BTC 시장 수익률 계산 실패');
                if (window.logManager) {
                    window.logManager.addLog(`⚠️ [${this.currentProcess}] 2단계: 수익률 계산 실패`);
                }
            }
        }
        
        // 다음 단계로 진행 (2초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 2000);
    }

    // 2단계: 매도 수익률 계산 (N/B 길드)
    step2_SellProfitCalculation(model, config, trainerDialog, currentMajority, sellProfitRateDisplay) {
        console.log(`📋 [${this.currentProcess}] 2단계: 학습 모델 기반 매도 수익률 계산`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 2단계: 학습 모델 기반 매도 수익률 계산 시작`);
        }
        
        // 매도 수익률 계산 로직 (기존 코드에서 추출)
        if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
            const currentPrice = window.currentPriceManager.getCurrentPrice();
            const buyPrice = model.buyPrice || 0;
            
            if (buyPrice > 0) {
                const sellProfitRate = ((currentPrice - buyPrice) / buyPrice) * 100;
                
                // 매도 수익률 표시 업데이트
                if (sellProfitRateDisplay) {
                    sellProfitRateDisplay.setText(`매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
                }
                
                if (window.logManager) {
                    window.logManager.addLog(`📊 [${this.currentProcess}] 2단계: 매도 전 예상 수익률 계산 완료 - ${sellProfitRate.toFixed(2)}%`);
                }
            }
        }
        
        // 다음 단계로 진행 (2초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 2000);
    }

    // 3단계: 학습 모델 결과 분석
    step3_ModelResultAnalysis(model, config, trainerDialog, currentMajority) {
        console.log(`📋 [${this.currentProcess}] 3단계: 학습 모델 결과 분석`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 3단계: 학습 모델 결과 분석 중...`);
        }
        
        // 현재 신호 분석
        const signalAnalysis = this.analyzeCurrentSignal(currentMajority);
        
        if (window.logManager) {
            window.logManager.addLog(`📊 [${this.currentProcess}] 3단계: 신호 분석 결과 - ${signalAnalysis}`);
        }
        
        // 다음 단계로 진행 (1.5초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 1500);
    }

    // 4단계: 매수 결정 (BTC 시장)
    step4_BuyDecision(model, config, trainerDialog, currentMajority) {
        console.log(`📋 [${this.currentProcess}] 4단계: 매수 결정 및 실행`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 4단계: 매수 결정 분석 중...`);
        }
        
        // BLUE 신호일 때만 매수 결정
        if (currentMajority === 'BLUE') {
            if (window.logManager) {
                window.logManager.addLog(`✅ [${this.currentProcess}] 4단계: BLUE 신호 감지 - 매수 결정 실행`);
            }
            
            // 매수 로직 실행 (기존 코드에서 추출)
            if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                const currentPrice = window.currentPriceManager.getCurrentPrice();
                model.buyPrice = currentPrice;
                
                if (window.logManager) {
                    window.logManager.addLog(`💰 [${this.currentProcess}] 4단계: 매수 완료 - 가격: ${currentPrice.toFixed(2)}`);
                }
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`⏸️ [${this.currentProcess}] 4단계: ${currentMajority} 신호 - 매수 보류`);
            }
        }
        
        // 다음 단계로 진행 (1초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 1000);
    }

    // 4단계: 매도 결정 (N/B 길드)
    step4_SellDecision(model, config, trainerDialog, currentMajority) {
        console.log(`📋 [${this.currentProcess}] 4단계: 매도 결정 및 실행`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 4단계: 매도 결정 분석 중...`);
        }
        
        // ORANGE 신호이고 매수한 적이 있을 때만 매도
        if (currentMajority === 'ORANGE' && model.buyPrice > 0) {
            if (window.logManager) {
                window.logManager.addLog(`✅ [${this.currentProcess}] 4단계: ORANGE 신호 감지 - 매도 결정 실행`);
            }
            
            // 매도 로직 실행
            if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                const currentPrice = window.currentPriceManager.getCurrentPrice();
                const profit = currentPrice - model.buyPrice;
                
                if (window.logManager) {
                    window.logManager.addLog(`💰 [${this.currentProcess}] 4단계: 매도 완료 - 수익: ${profit.toFixed(2)}`);
                }
                
                // 매수 가격 초기화
                model.buyPrice = 0;
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`⏸️ [${this.currentProcess}] 4단계: ${currentMajority} 신호 또는 매수 이력 없음 - 매도 보류`);
            }
        }
        
        // 다음 단계로 진행 (1초 후)
        setTimeout(() => {
            this.processStep++;
            this.executeCurrentStep(model, config, trainerDialog, currentMajority);
        }, 1000);
    }

    // 5단계: 프로세스 완료 및 이동
    step5_ProcessCompletion(model, config, trainerDialog) {
        console.log(`📋 [${this.currentProcess}] 5단계: 프로세스 완료 및 이동`);
        
        if (window.logManager) {
            window.logManager.addLog(`📋 [${this.currentProcess}] 5단계: 프로세스 완료 - 신호 대기 센터로 이동 준비`);
        }
        
        // 신호 대기 센터로 이동
        model.targetAction = '신호 대기';
        model.targetX = config.width / 2;
        model.targetY = config.height / 2;
        model.circle.setFillStyle(0x88ccff);
        
        const dialogMessage = `🔵 [프로세스 완료] 신호 대기 센터로 이동 중...`;
        trainerDialog.setText(dialogMessage);
        
        if (window.logManager) {
            window.logManager.addLog(`🔵 [${this.currentProcess}] 5단계: 신호 대기 센터로 이동 시작`);
        }
        
        // 프로세스 완료
        setTimeout(() => {
            this.completeProcess(model, config, trainerDialog);
        }, 1000);
    }

    // 현재 단계 실행
    executeCurrentStep(model, config, trainerDialog, currentMajority) {
        if (this.currentProcess === this.processStates.BTC_MARKET) {
            this.executeBTCMarketStep(model, config, trainerDialog, currentMajority);
        } else if (this.currentProcess === this.processStates.NB_GUILD) {
            this.executeNBGuildStep(model, config, trainerDialog, currentMajority);
        }
    }

    // 프로세스 완료
    completeProcess(model, config, trainerDialog) {
        console.log(`✅ [${this.currentProcess}] 프로세스 완료`);
        
        this.isProcessing = false;
        this.currentProcess = null;
        this.processStep = 0;
        
        if (window.logManager) {
            window.logManager.addLog(`✅ [프로세스 완료] 모든 단계 완료 - 다음 액션 대기`);
        }
    }

    // 현재 신호 분석
    analyzeCurrentSignal(currentMajority) {
        switch (currentMajority) {
            case 'BLUE':
                return 'BLUE 신호 - 상승 추세, 매수 고려';
            case 'ORANGE':
                return 'ORANGE 신호 - 하락 추세, 매도 고려';
            default:
                return '신호 없음 - 대기';
        }
    }

    // N/B 길드 시각적 효과
    addNBGuildVisualEffect(model) {
        // N/B 길드 다각형 시각적 효과
        if (window.nbGuildPolygon) {
            const originalColor = 0x00ff00;
            let blinkCount = 0;
            const maxBlinks = 6;
            
            const blinkInterval = setInterval(() => {
                if (window.nbGuildPolygon && blinkCount < maxBlinks) {
                    const isBright = blinkCount % 2 === 0;
                    window.nbGuildPolygon.setFillStyle(isBright ? 0x00ff88 : originalColor);
                    blinkCount++;
                } else {
                    clearInterval(blinkInterval);
                    if (window.nbGuildPolygon) {
                        window.nbGuildPolygon.setFillStyle(originalColor);
                    }
                }
            }, 250);
        }
        
        // 트레이너 시각적 효과
        if (model.circle) {
            const originalColor = model.circle.fillColor;
            let blinkCount = 0;
            const maxBlinks = 8;
            
            const circleBlinkInterval = setInterval(() => {
                if (model.circle && blinkCount < maxBlinks) {
                    const isBright = blinkCount % 2 === 0;
                    model.circle.setFillStyle(isBright ? 0xff8800 : originalColor);
                    blinkCount++;
                } else {
                    clearInterval(circleBlinkInterval);
                    if (model.circle) {
                        model.circle.setFillStyle(originalColor);
                    }
                }
            }, 250);
        }
    }

    // 프로세스 상태 확인
    isCurrentlyProcessing() {
        return this.isProcessing;
    }

    // 현재 프로세스 정보
    getCurrentProcessInfo() {
        return {
            process: this.currentProcess,
            step: this.processStep,
            isProcessing: this.isProcessing
        };
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.actionProcessManager = new ActionProcessManager();
}

// 모듈 로딩 완료
