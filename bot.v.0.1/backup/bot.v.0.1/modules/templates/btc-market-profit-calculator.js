// BTC 시장 수익률 계산기 모듈
// BTC 시장에서 매수 전 예상 수익률 계산을 담당

class BTCMarketProfitCalculator {
    constructor() {
        this.isCalculated = false;
        this.lastCalculationTime = 0;
        this.calculationInterval = 5000; // 5초마다 재계산 가능
    }

    // BTC 시장 도달 시 매수 전 예상 수익률 계산
    calculateBuyProfitRateAtMarket(currentMajority, buyProfitRateDisplay, trainerDialog, model, config, startX, topY) {
        try {
            // 중복 계산 방지 (5초 이내 재계산 방지)
            const currentTime = Date.now();
            if (this.isCalculated && (currentTime - this.lastCalculationTime) < this.calculationInterval) {
                console.log('🔍 BTC 시장 수익률 계산 - 중복 계산 방지 (최근 계산됨)');
                return false;
            }

            console.log('🔍 BTC 시장 도달 - CurrentPriceManager 체크:', {
                hasManager: !!window.currentPriceManager,
                isValid: window.currentPriceManager ? window.currentPriceManager.isValidCurrentPrice() : false
            });

            let buyProfitRate = 0;
            let calculationMethod = '';

            // CurrentPriceManager 모듈을 사용한 고급 수익률 계산
            if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                calculationMethod = 'CurrentPriceManager';
                buyProfitRate = this.performAdvancedCalculation(buyProfitRateDisplay);
            } else {
                // 폴백: 기본 수익률 계산
                calculationMethod = '기본 계산';
                buyProfitRate = this.performBasicCalculation(buyProfitRateDisplay);
            }

            // 계산 완료 표시
            this.isCalculated = true;
            this.lastCalculationTime = currentTime;

            // 로그 기록
            this.logCalculationResult(buyProfitRate, calculationMethod, currentMajority);

            // BLUE 신호일 때만 매수 의사결정
            if (currentMajority === 'BLUE') {
                this.handleBuyDecision(buyProfitRate, model, config, startX, topY, trainerDialog);
            } else {
                // BLUE 신호가 아닌 경우 수익률은 계산했지만 매수하지 않고 복귀
                if (window.logManager) {
                    window.logManager.addLog(`🔵 BTC 시장에서 매수 전 예상 수익률 계산 완료 (${buyProfitRate.toFixed(2)}%) - BLUE 신호가 아니므로 매수하지 않음`);
                }
            }

            return true; // 계산 완료

        } catch (error) {
            console.error('❌ BTC 시장 수익률 계산 중 오류 발생:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ BTC 시장 수익률 계산 오류: ${error.message}`);
            }
            return false;
        }
    }

    // CurrentPriceManager를 사용한 고급 수익률 계산
    performAdvancedCalculation(buyProfitRateDisplay) {
        const infoData = window.currentPriceManager.getInfoPanelData();
        console.log('📊 Info Panel 데이터 (BTC 시장 도달):', infoData);
        
        const buyProfitRate = window.currentPriceManager.calculateBuyProfitRate();
        
        // 매수 전 예상 수익률 표시 업데이트
        this.updateProfitRateDisplay(buyProfitRate, buyProfitRateDisplay, infoData);
        
        console.log(`📊 BTC 시장에서 고급 매수 전 예상 수익률 계산 완료: ${buyProfitRate.toFixed(2)}%`);
        
        return buyProfitRate;
    }

    // 기본 수익률 계산 (폴백)
    performBasicCalculation(buyProfitRateDisplay) {
        if (window.logManager) {
            window.logManager.addLog(`⚠️ CurrentPriceManager 모듈 오류 - 기본 수익률 계산으로 대체`);
        }
        
        // 기본 수익률 계산 (5% 고정)
        const buyProfitRate = 5.00;
        
        // 매수 전 예상 수익률 표시 업데이트
        const profitColor = buyProfitRate >= 0 ? '#00ff88' : '#ff0088';
        buyProfitRateDisplay.setFill(profitColor);
        
        let displayText = `매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}% (기본 계산)`;
        buyProfitRateDisplay.setText(displayText);
        
        console.log(`📊 BTC 시장에서 기본 매수 전 예상 수익률 계산: ${buyProfitRate.toFixed(2)}%`);
        
        return buyProfitRate;
    }

    // 수익률 표시 업데이트
    updateProfitRateDisplay(buyProfitRate, buyProfitRateDisplay, infoData) {
        const profitColor = buyProfitRate >= 0 ? '#00ff88' : '#ff0088';
        buyProfitRateDisplay.setFill(profitColor);
        
        let displayText = `매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
        const avgPriceProfitRate = window.currentPriceManager.calculateAvgPriceProfitRate();
        if (avgPriceProfitRate !== 0) {
            displayText += ` (학습모델: ${infoData.predictedProfitRate?.toFixed(2) || '0'}%, 평균단가: ${avgPriceProfitRate.toFixed(2)}%, 기본: 5.00%)`;
        } else {
            displayText += ` (학습모델: ${infoData.predictedProfitRate?.toFixed(2) || '0'}%, 기본: 5.00%)`;
        }
        buyProfitRateDisplay.setText(displayText);
    }

    // 매수 의사결정 처리
    handleBuyDecision(buyProfitRate, model, config, startX, topY, trainerDialog) {
        const currentProfitRate = window.learningSystem ? window.learningSystem.getCurrentProfitRate() : 0;
        
        // BLUE 신호에서는 예상 수익률이 계산되면 매수 (음수여도 매수)
        if (buyProfitRate < currentProfitRate || buyProfitRate > 0) {
            console.log(`📈 BTC 시장에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) → 매수 진행`);
            
            // 매수 액션 설정
            model.targetAction = '매수';
            model.targetX = startX;
            model.targetY = topY;
            model.circle.setFillStyle(0xff0000); // 빨간색 (매수)
            
            if (window.logManager) {
                window.logManager.addLog(`📈 BTC 시장에서 매수 시작: 예상 수익률 ${buyProfitRate.toFixed(2)}%`);
            }
            
            const dialogMessage = `📈 [매수 시작] 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
            trainerDialog.setText(dialogMessage);
            if (window.logManager) {
                window.logManager.addLog(dialogMessage);
            }
            
            // 매수 완료 후 신호 대기 센터로 복귀
            setTimeout(() => {
                model.targetAction = '신호 대기';
                model.targetX = config.width / 2;
                model.targetY = config.height / 2;
                model.circle.setFillStyle(0x88ccff);
                
                if (window.logManager) {
                    window.logManager.addLog(`🔵 매수 완료: 신호 대기 센터로 복귀`);
                }
            }, 3000); // 3초 후 복귀
        } else {
            console.log(`⚠️ BTC 시장에서 매수 조건 불만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) < 현재 수익률(${currentProfitRate.toFixed(2)}%)`);
            
            // 매수 조건이 맞지 않으면 신호 대기 센터로 복귀
            model.targetAction = '신호 대기';
            model.targetX = config.width / 2;
            model.targetY = config.height / 2;
            model.circle.setFillStyle(0x88ccff);
            
            if (window.logManager) {
                window.logManager.addLog(`⚠️ BTC 시장에서 매수 조건 불만족: 신호 대기 센터로 복귀`);
            }
        }
    }

    // 계산 결과 로그 기록
    logCalculationResult(buyProfitRate, calculationMethod, currentMajority) {
        if (window.logManager) {
            if (calculationMethod === 'CurrentPriceManager') {
                window.logManager.addLog(`📊 BTC 시장 도달 - 매수 전 예상 수익률 계산: ${buyProfitRate.toFixed(2)}% (현재 신호: ${currentMajority})`);
            } else {
                window.logManager.addLog(`📊 BTC 시장 도달 - 기본 매수 전 예상 수익률 계산: ${buyProfitRate.toFixed(2)}% (현재 신호: ${currentMajority})`);
            }
        }
    }

    // 계산 상태 초기화 (새로운 탐색 시작 시)
    resetCalculationState() {
        this.isCalculated = false;
        this.lastCalculationTime = 0;
    }

    // 계산 상태 확인
    getCalculationStatus() {
        return {
            isCalculated: this.isCalculated,
            lastCalculationTime: this.lastCalculationTime,
            timeSinceLastCalculation: Date.now() - this.lastCalculationTime
        };
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.btcMarketProfitCalculator = new BTCMarketProfitCalculator();
}

// 모듈 로딩 완료
