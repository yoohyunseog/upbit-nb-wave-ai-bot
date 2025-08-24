// BTC 시장 학습 기반 처리 모듈
// BTC 시장 도달 시 학습 모델을 사용한 수익률 계산 및 의사결정을 담당

class BTCMarketLearningHandler {
    constructor() {
        this.arrivalThreshold = 60;
        this.profitCalculationExecuted = false;
        this.returnScheduled = false;
        this.learningData = [];
        this.arrivalCount = 0;
    }

    // BTC 시장 도달 처리 (학습 모델 기반)
    handleBTCMarketArrival(model, config, trainerDialog, currentMajority, buyProfitRateDisplay) {
        console.log('🔍 DEBUG: BTC 시장 도달! 학습 모델 기반 처리 시작');
        
        // BTC 시장에 도달했을 때 탐색 모드 해제
        if (model.btcExplorationMode) {
            model.btcExplorationMode = false;
            
            if (window.logManager) {
                window.logManager.addLog(`🎯 트레이너가 BTC 시장에 도달! 학습 모델 기반 처리 시작 (현재 신호: ${currentMajority})`);
            }
        }

        // 학습 데이터 수집
        this.collectLearningData(model, config, currentMajority);
        
        // 학습 모델을 사용한 수익률 계산
        this.calculateProfitRateWithLearning(model, config, trainerDialog, currentMajority, buyProfitRateDisplay);
        
        // 시각적 효과 추가
        this.addVisualEffects(model);
        
        // 학습 기반 의사결정
        this.makeLearningBasedDecision(model, config, trainerDialog, currentMajority);
        
        return true; // 도달 완료
    }

    // 학습 데이터 수집
    collectLearningData(model, config, currentMajority) {
        const learningDataPoint = {
            timestamp: Date.now(),
            arrivalCount: ++this.arrivalCount,
            position: { x: model.circle.x, y: model.circle.y },
            signal: currentMajority,
            btcPrice: window.currentPriceManager ? window.currentPriceManager.getCurrentPrice() : 0,
            marketConditions: this.getMarketConditions(),
            previousActions: this.getPreviousActions(model)
        };
        
        this.learningData.push(learningDataPoint);
        
        if (window.logManager) {
            window.logManager.addLog(`📊 학습 데이터 수집: ${this.arrivalCount}번째 BTC 시장 도달 (신호: ${currentMajority})`);
        }
        
        // 학습 데이터가 너무 많아지면 오래된 것 제거
        if (this.learningData.length > 100) {
            this.learningData = this.learningData.slice(-50);
        }
    }

    // 시장 조건 분석
    getMarketConditions() {
        const conditions = {
            volatility: this.calculateVolatility(),
            trend: this.calculateTrend(),
            volume: this.calculateVolume(),
            momentum: this.calculateMomentum()
        };
        
        return conditions;
    }

    // 변동성 계산
    calculateVolatility() {
        if (this.learningData.length < 5) return 0;
        
        const prices = this.learningData.slice(-10).map(d => d.btcPrice).filter(p => p > 0);
        if (prices.length < 2) return 0;
        
        const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
        const variance = prices.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / prices.length;
        
        return Math.sqrt(variance);
    }

    // 트렌드 계산
    calculateTrend() {
        if (this.learningData.length < 3) return 'neutral';
        
        const recentPrices = this.learningData.slice(-5).map(d => d.btcPrice).filter(p => p > 0);
        if (recentPrices.length < 2) return 'neutral';
        
        const firstPrice = recentPrices[0];
        const lastPrice = recentPrices[recentPrices.length - 1];
        const change = ((lastPrice - firstPrice) / firstPrice) * 100;
        
        if (change > 2) return 'bullish';
        if (change < -2) return 'bearish';
        return 'neutral';
    }

    // 거래량 계산 (시뮬레이션)
    calculateVolume() {
        return Math.random() * 100 + 50; // 50-150 범위의 시뮬레이션 거래량
    }

    // 모멘텀 계산
    calculateMomentum() {
        if (this.learningData.length < 3) return 0;
        
        const recentData = this.learningData.slice(-3);
        const momentum = recentData.reduce((acc, data, index) => {
            if (index === 0) return 0;
            return acc + (data.btcPrice - recentData[index - 1].btcPrice);
        }, 0);
        
        return momentum;
    }

    // 이전 액션들 가져오기
    getPreviousActions(model) {
        return {
            lastAction: model.targetAction,
            actionHistory: model.actionHistory || [],
            explorationMode: model.btcExplorationMode
        };
    }

    // 학습 모델을 사용한 수익률 계산
    calculateProfitRateWithLearning(model, config, trainerDialog, currentMajority, buyProfitRateDisplay) {
        if (this.profitCalculationExecuted) {
            console.log('⚠️ 수익률 계산 이미 완료됨');
            return;
        }

        console.log('🔍 DEBUG: 학습 모델 기반 수익률 계산 시작');
        
        // 기존 수익률 계산기 사용
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
                console.log('✅ 학습 모델 기반 수익률 계산 완료');
                this.profitCalculationExecuted = true;
                
                if (window.logManager) {
                    window.logManager.addLog(`✅ 학습 모델 기반 수익률 계산 완료`);
                }
            }
        } else {
            // 학습 데이터 기반 수익률 계산
            const learningBasedProfitRate = this.calculateLearningBasedProfitRate();
            
            if (window.logManager) {
                window.logManager.addLog(`📊 학습 데이터 기반 수익률: ${learningBasedProfitRate.toFixed(2)}%`);
            }
            
            this.profitCalculationExecuted = true;
        }
    }

    // 학습 데이터 기반 수익률 계산
    calculateLearningBasedProfitRate() {
        if (this.learningData.length < 3) {
            return 5.00; // 기본값
        }
        
        // 최근 10개의 도달 데이터 분석
        const recentData = this.learningData.slice(-10);
        const blueSignals = recentData.filter(d => d.signal === 'BLUE').length;
        const redSignals = recentData.filter(d => d.signal === 'RED').length;
        
        // 시장 조건 분석
        const conditions = this.getMarketConditions();
        
        // 학습 기반 수익률 계산
        let baseRate = 5.00;
        
        // 신호 비율에 따른 조정
        if (blueSignals > redSignals) {
            baseRate += 2.0; // BLUE 신호가 많으면 수익률 증가
        } else if (redSignals > blueSignals) {
            baseRate -= 1.0; // RED 신호가 많으면 수익률 감소
        }
        
        // 변동성에 따른 조정
        if (conditions.volatility > 1000) {
            baseRate += 1.5; // 높은 변동성에서는 수익률 증가
        }
        
        // 트렌드에 따른 조정
        if (conditions.trend === 'bullish') {
            baseRate += 1.0;
        } else if (conditions.trend === 'bearish') {
            baseRate -= 0.5;
        }
        
        // 모멘텀에 따른 조정
        if (conditions.momentum > 0) {
            baseRate += 0.5;
        }
        
        return Math.max(0, Math.min(15, baseRate)); // 0-15% 범위로 제한
    }

    // 시각적 효과 추가
    addVisualEffects(model) {
        // BTC 시장 다각형 깜빡임 효과
        if (window.btcMarketPolygon) {
            const originalColor = 0x0088ff;
            let blinkCount = 0;
            const maxBlinks = 6;
            
            const blinkInterval = setInterval(() => {
                if (window.btcMarketPolygon && blinkCount < maxBlinks) {
                    const isBright = blinkCount % 2 === 0;
                    window.btcMarketPolygon.setFillStyle(isBright ? 0x00ffff : originalColor);
                    blinkCount++;
                } else {
                    clearInterval(blinkInterval);
                    if (window.btcMarketPolygon) {
                        window.btcMarketPolygon.setFillStyle(originalColor);
                    }
                }
            }, 250);
        }
        
        // 트레이너 깜빡임 효과
        if (model.circle) {
            const originalColor = model.circle.fillColor;
            let blinkCount = 0;
            const maxBlinks = 8;
            
            const circleBlinkInterval = setInterval(() => {
                if (model.circle && blinkCount < maxBlinks) {
                    const isBright = blinkCount % 2 === 0;
                    model.circle.setFillStyle(isBright ? 0xffff00 : originalColor);
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

    // 학습 기반 의사결정
    makeLearningBasedDecision(model, config, trainerDialog, currentMajority) {
        // 학습 데이터 기반 대기 시간 결정
        const waitTime = this.calculateOptimalWaitTime();
        
        if (window.logManager) {
            window.logManager.addLog(`🧠 학습 기반 의사결정: ${waitTime}초 대기 후 신호 대기 센터로 복귀`);
        }
        
        // 학습 기반 대기 시간 후 복귀
        if (!this.returnScheduled) {
            this.returnScheduled = true;
            
            setTimeout(() => {
                this.returnToSignalCenter(model, config, trainerDialog);
            }, waitTime * 1000);
        }
    }

    // 최적 대기 시간 계산
    calculateOptimalWaitTime() {
        if (this.learningData.length < 5) {
            return 10; // 기본 10초
        }
        
        // 최근 도달 패턴 분석
        const recentArrivals = this.learningData.slice(-5);
        const avgInterval = this.calculateAverageInterval(recentArrivals);
        
        // 시장 조건에 따른 조정
        const conditions = this.getMarketConditions();
        let optimalTime = 10;
        
        if (conditions.volatility > 1000) {
            optimalTime += 5; // 높은 변동성에서는 더 오래 대기
        }
        
        if (conditions.trend === 'bullish') {
            optimalTime += 3; // 상승 트렌드에서는 더 오래 대기
        }
        
        return Math.max(5, Math.min(20, optimalTime)); // 5-20초 범위
    }

    // 평균 간격 계산
    calculateAverageInterval(arrivals) {
        if (arrivals.length < 2) return 0;
        
        let totalInterval = 0;
        for (let i = 1; i < arrivals.length; i++) {
            totalInterval += arrivals[i].timestamp - arrivals[i-1].timestamp;
        }
        
        return totalInterval / (arrivals.length - 1) / 1000; // 초 단위로 변환
    }

    // 신호 대기 센터로 복귀
    returnToSignalCenter(model, config, trainerDialog) {
        const previousAction = model.targetAction;
        model.targetAction = '신호 대기';
        model.targetX = config.width / 2;
        model.targetY = config.height / 2;
        model.circle.setFillStyle(0x88ccff);
        
        console.log(`🔵 트레이너: 학습 기반 BTC 탐색 완료, 신호 대기 센터로 복귀!`);
        if (window.logManager) {
            window.logManager.addLog(`🔵 학습 기반 BTC 탐색 완료: targetAction 변경 (${previousAction} → 신호 대기)`);
        }
        
        const dialogMessage = `🔵 [학습 기반 탐색 완료] 신호 대기 센터로 복귀 중...`;
        trainerDialog.setText(dialogMessage);
        if (window.logManager) {
            window.logManager.addLog(dialogMessage);
        }
        
        // 상태 초기화
        this.profitCalculationExecuted = false;
        this.returnScheduled = false;
    }

    // 학습 데이터 가져오기
    getLearningData() {
        return this.learningData;
    }

    // 학습 데이터 초기화
    resetLearningData() {
        this.learningData = [];
        this.arrivalCount = 0;
        this.profitCalculationExecuted = false;
        this.returnScheduled = false;
    }

    // 학습 통계 가져오기
    getLearningStats() {
        return {
            totalArrivals: this.arrivalCount,
            dataPoints: this.learningData.length,
            averageProfitRate: this.calculateAverageProfitRate(),
            marketConditions: this.getMarketConditions()
        };
    }

    // 평균 수익률 계산
    calculateAverageProfitRate() {
        if (this.learningData.length === 0) return 0;
        
        const profitRates = this.learningData.map(d => d.profitRate || 5.00);
        return profitRates.reduce((a, b) => a + b, 0) / profitRates.length;
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.btcMarketLearningHandler = new BTCMarketLearningHandler();
}

// 모듈 로딩 완료
