// Learning System Module
// AI 트레이너의 학습 및 예측 시스템

class LearningSystem {
    constructor() {
        this.learningData = {};
        this.predictionWeights = {
            learningModel: 0.5,    // 학습 모델 가중치
            avgPrice: 0.3,         // 평균 단가 가중치
            basicCalculation: 0.2  // 기본 계산 가중치
        };
    }

    // Info Panel에서 모든 데이터 추출
    getInfoPanelData() {
        const data = {};
        
        try {
            // 코인 정보
            const coinNameElement = document.getElementById('selected-coin-name');
            data.coinName = coinNameElement ? coinNameElement.textContent : '';
            
            // 코인 잔고
            const coinBalanceElement = document.getElementById('selected-coin-balance');
            data.coinBalance = coinBalanceElement ? parseFloat(coinBalanceElement.textContent) : 0;
            
            // 코인 가치
            const coinValueElement = document.getElementById('selected-coin-value');
            if (coinValueElement) {
                const valueMatch = coinValueElement.textContent.match(/₩([\d,]+\.?\d*)/);
                data.coinValue = valueMatch ? parseFloat(valueMatch[1].replace(/,/g, '')) : 0;
            }
            
            // 현재 가격
            const coinPriceElement = document.getElementById('selected-coin-price');
            if (coinPriceElement) {
                const priceMatch = coinPriceElement.textContent.match(/₩([\d,]+)/);
                data.currentPrice = priceMatch ? parseInt(priceMatch[1].replace(/,/g, '')) : 0;
            }
            
            // 수익률 (학습 모델 예측)
            const pnlElement = document.getElementById('selected-coin-pnl');
            if (pnlElement) {
                const pnlMatch = pnlElement.textContent.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
                data.predictedProfitRate = pnlMatch ? parseFloat(pnlMatch[1]) : 0;
            }
            
            // 평균 단가
            const avgPriceElement = document.getElementById('selected-coin-avg-price');
            if (avgPriceElement) {
                const avgMatch = avgPriceElement.textContent.match(/₩([\d,]+\.?\d*)/);
                data.avgPrice = avgMatch ? parseFloat(avgMatch[1].replace(/,/g, '')) : 0;
            }
            
            // 현재 시간
            const timeElement = document.getElementById('current-time');
            data.currentTime = timeElement ? timeElement.textContent : '';
            
        } catch (error) {
            console.error('Info Panel 데이터 추출 실패:', error);
        }
        
        return data;
    }

    // 학습 모델 기반 고급 매수 전 예상 수익률 계산
    calculateAdvancedBuyProfitRate(currentPriceText) {
        const infoData = this.getInfoPanelData();
        console.log('📊 Info Panel 데이터 (매수 계산):', infoData);
        
        const currentPriceMatch = currentPriceText.match(/₩([\d,]+)/);
        if (!currentPriceMatch) {
            console.warn('현재 가격 정보를 찾을 수 없습니다.');
            return 0;
        }
        
        const currentPrice = parseInt(currentPriceMatch[1].replace(/,/g, ''));
        const predictedProfitRate = infoData.predictedProfitRate || 0;
        
        // 기본 계산: ProfitRateCalculator 모듈 사용
        let basicProfitRate = 0;
        if (window.profitRateCalculator && currentPrice > 0) {
            basicProfitRate = window.profitRateCalculator.calculateBasicProfitRate();
        } else {
            // 폴백: 기본 계산
            const expectedSellPrice = currentPrice * 1.05;
            basicProfitRate = currentPrice > 0 ? ((expectedSellPrice - currentPrice) / currentPrice) * 100 : 0;
        }
        
        // Info Panel 데이터를 활용한 고급 매수 전 예상 수익률 계산
        let advancedBuyProfitRate = predictedProfitRate;
        
        // 평균 단가 대비 현재가 수익률 계산
        if (infoData.avgPrice > 0 && infoData.currentPrice > 0) {
            const avgPriceProfitRate = ((infoData.currentPrice - infoData.avgPrice) / infoData.avgPrice) * 100;
            console.log(`📊 매수 계산 - 평균단가 대비 수익률: ${avgPriceProfitRate.toFixed(2)}% (평균단가: ₩${infoData.avgPrice.toLocaleString()}, 현재가: ₩${infoData.currentPrice.toLocaleString()})`);
            
            // 가중치 조정: 학습모델 50%, 평균단가 30%, 기본계산 20%
            advancedBuyProfitRate = (predictedProfitRate * this.predictionWeights.learningModel) + 
                                   (avgPriceProfitRate * this.predictionWeights.avgPrice) + 
                                   (basicProfitRate * this.predictionWeights.basicCalculation);
        } else {
            // 평균단가 정보가 없으면 기존 방식 사용
            advancedBuyProfitRate = (predictedProfitRate * 0.7) + (basicProfitRate * 0.3);
        }
        
        console.log(`📊 고급 매수 전 예상 수익률 계산 완료: ${advancedBuyProfitRate.toFixed(2)}%`);
        console.log(`   - 학습 모델 예측: ${predictedProfitRate.toFixed(2)}%`);
        console.log(`   - 기본 계산: ${basicProfitRate.toFixed(2)}%`);
        console.log(`   - 최종 매수 전 예상 수익률: ${advancedBuyProfitRate.toFixed(2)}%`);
        
        return {
            advancedProfitRate: advancedBuyProfitRate,
            infoData: infoData,
            predictedProfitRate: predictedProfitRate,
            basicProfitRate: basicProfitRate
        };
    }

    // 학습 모델 기반 고급 매도 전 예상 수익률 계산
    calculateAdvancedSellProfitRate(buyPrice, currentPriceText) {
        const infoData = this.getInfoPanelData();
        console.log('📊 Info Panel 데이터 (매도 계산):', infoData);
        
        // 학습 모델이 예측한 수익률
        const predictedProfitRate = infoData.predictedProfitRate || 0;
        
        // 기존 계산 방식 (매수가 대비 현재가)
        const currentPriceMatch = currentPriceText.match(/₩([\d,]+)/);
        if (!currentPriceMatch) {
            console.warn('현재 가격 정보를 찾을 수 없습니다.');
            return 0;
        }
        
        const currentPrice = parseInt(currentPriceMatch[1].replace(/,/g, ''));
        const basicProfitRate = ((currentPrice - buyPrice) / buyPrice) * 100;
        
        // Info Panel 데이터를 활용한 고급 예상 수익률 계산
        let advancedProfitRate = predictedProfitRate;
        
        // 평균 단가 대비 현재가 수익률 계산
        if (infoData.avgPrice > 0 && infoData.currentPrice > 0) {
            const avgPriceProfitRate = ((infoData.currentPrice - infoData.avgPrice) / infoData.avgPrice) * 100;
            console.log(`📊 평균단가 대비 수익률: ${avgPriceProfitRate.toFixed(2)}% (평균단가: ₩${infoData.avgPrice.toLocaleString()}, 현재가: ₩${infoData.currentPrice.toLocaleString()})`);
            
            // 가중치 조정: 학습모델 50%, 평균단가 30%, 기본계산 20%
            advancedProfitRate = (predictedProfitRate * this.predictionWeights.learningModel) + 
                                (avgPriceProfitRate * this.predictionWeights.avgPrice) + 
                                (basicProfitRate * this.predictionWeights.basicCalculation);
        } else {
            // 평균단가 정보가 없으면 기존 방식 사용
            advancedProfitRate = (predictedProfitRate * 0.7) + (basicProfitRate * 0.3);
        }
        
        console.log(`📊 고급 매도 전 예상 수익률 계산 완료:`);
        console.log(`   - 학습 모델 예측: ${predictedProfitRate.toFixed(2)}%`);
        console.log(`   - 기본 계산: ${basicProfitRate.toFixed(2)}% (매수가: ₩${buyPrice.toLocaleString()}, 현재가: ₩${currentPrice.toLocaleString()})`);
        console.log(`   - 최종 예상 수익률: ${advancedProfitRate.toFixed(2)}%`);
        
        return {
            advancedProfitRate: advancedProfitRate,
            infoData: infoData,
            predictedProfitRate: predictedProfitRate,
            basicProfitRate: basicProfitRate
        };
    }

    // 현재 수익률을 가져오는 함수
    getCurrentProfitRate() {
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

    // 학습 모델 예측 수익률 표시 텍스트 생성
    getPredictedRateDisplayText() {
        const pnlElement = document.getElementById('selected-coin-pnl');
        let predictedRate = '';
        if (pnlElement) {
            const pnlText = pnlElement.textContent;
            const pnlMatch = pnlText.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
            if (pnlMatch) {
                predictedRate = ` (예측: ${parseFloat(pnlMatch[1]).toFixed(2)}%)`;
            }
        }
        return predictedRate;
    }

    // 학습 데이터 저장
    saveLearningData(data) {
        this.learningData = { ...this.learningData, ...data };
        console.log('📚 학습 데이터 저장:', this.learningData);
    }

    // 학습 데이터 가져오기
    getLearningData() {
        return this.learningData;
    }

    // 가중치 조정
    updatePredictionWeights(learningModel, avgPrice, basicCalculation) {
        this.predictionWeights = {
            learningModel: learningModel || 0.5,
            avgPrice: avgPrice || 0.3,
            basicCalculation: basicCalculation || 0.2
        };
        console.log('⚖️ 예측 가중치 업데이트:', this.predictionWeights);
    }
}

// 전역 인스턴스 생성
window.learningSystem = new LearningSystem();
