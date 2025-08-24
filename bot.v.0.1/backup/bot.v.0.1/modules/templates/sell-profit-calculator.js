// 매도 전 예상 수익률 계산 모듈
// predictedProfitRate 변수 정의 문제를 해결하기 위해 분리된 모듈

class SellProfitCalculator {
    constructor() {
        this.predictedProfitRate = 0;
        this.basicProfitRate = 0;
        this.avgPriceProfitRate = 0;
    }

    // Info Panel 데이터에서 예상 수익률 추출
    extractPredictedProfitRate(infoData) {
        if (infoData && infoData.predictedProfitRate !== undefined) {
            this.predictedProfitRate = infoData.predictedProfitRate;
        } else {
            this.predictedProfitRate = 0;
        }
        return this.predictedProfitRate;
    }

    // 평균단가 대비 수익률 계산
    calculateAvgPriceProfitRate(infoData) {
        if (infoData && infoData.avgPrice > 0 && infoData.currentPrice > 0) {
            this.avgPriceProfitRate = ((infoData.currentPrice - infoData.avgPrice) / infoData.avgPrice) * 100;
        } else {
            this.avgPriceProfitRate = 0;
        }
        return this.avgPriceProfitRate;
    }

    // 기본 수익률 계산 (매수가 대비 현재가)
    calculateBasicProfitRate(buyPrice, currentPrice) {
        if (buyPrice > 0 && currentPrice > 0) {
            this.basicProfitRate = ((currentPrice - buyPrice) / buyPrice) * 100;
        } else {
            this.basicProfitRate = 0;
        }
        return this.basicProfitRate;
    }

    // 고급 매도 전 예상 수익률 계산
    calculateAdvancedSellProfitRate(buyPrice, currentPriceText, infoData) {
        // 현재가 추출
        const currentPriceMatch = currentPriceText.match(/₩([\d,]+)/);
        if (!currentPriceMatch) {
            return {
                advancedProfitRate: 0,
                predictedProfitRate: 0,
                basicProfitRate: 0,
                avgPriceProfitRate: 0
            };
        }

        const currentPrice = parseInt(currentPriceMatch[1].replace(/,/g, ''));
        
        // 예상 수익률 추출
        const predictedProfitRate = this.extractPredictedProfitRate(infoData);
        
        // 기본 수익률 계산
        const basicProfitRate = this.calculateBasicProfitRate(buyPrice, currentPrice);
        
        // 평균단가 대비 수익률 계산
        const avgPriceProfitRate = this.calculateAvgPriceProfitRate(infoData);
        
        // 고급 매도 전 예상 수익률 계산 (가중치 적용)
        let advancedProfitRate = 0;
        if (infoData && infoData.avgPrice > 0) {
            // 가중치: 학습모델 50%, 평균단가 30%, 기본계산 20%
            advancedProfitRate = (predictedProfitRate * 0.5) + (avgPriceProfitRate * 0.3) + (basicProfitRate * 0.2);
        } else {
            // 평균단가 정보가 없으면 기존 방식 사용
            advancedProfitRate = (predictedProfitRate * 0.7) + (basicProfitRate * 0.3);
        }

        return {
            advancedProfitRate: advancedProfitRate,
            predictedProfitRate: predictedProfitRate,
            basicProfitRate: basicProfitRate,
            avgPriceProfitRate: avgPriceProfitRate
        };
    }

    // 매도 전 예상 수익률 표시 텍스트 생성
    generateDisplayText(sellProfitRate, infoData) {
        let displayText = `매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
        
        if (infoData && infoData.avgPrice > 0) {
            const avgPriceProfitRate = this.calculateAvgPriceProfitRate(infoData);
            const predictedProfitRate = this.extractPredictedProfitRate(infoData);
            const basicProfitRate = this.basicProfitRate;
            
            displayText += ` (학습모델: ${predictedProfitRate.toFixed(2)}%, 평균단가: ${avgPriceProfitRate.toFixed(2)}%, 기본: ${basicProfitRate.toFixed(2)}%)`;
        } else {
            const predictedProfitRate = this.extractPredictedProfitRate(infoData);
            const basicProfitRate = this.basicProfitRate;
            
            displayText += ` (학습모델: ${predictedProfitRate.toFixed(2)}%, 기본: ${basicProfitRate.toFixed(2)}%)`;
        }
        
        return displayText;
    }

    // 매도 전 예상 수익률 대화창 메시지 생성
    generateDialogMessage(sellProfitRate, infoData, currentPriceText) {
        let dialogMessage = `🎯 [의사결정: 매도 준비] N/B 길드에서 Info Panel 기반 고급 매도 전 예상 수익률 계산 완료! ${sellProfitRate.toFixed(2)}%`;
        
        if (infoData && infoData.avgPrice > 0) {
            const avgPriceProfitRate = this.calculateAvgPriceProfitRate(infoData);
            const predictedProfitRate = this.extractPredictedProfitRate(infoData);
            
            dialogMessage += ` (학습모델: ${predictedProfitRate.toFixed(2)}%, 평균단가: ${avgPriceProfitRate.toFixed(2)}%)`;
        } else {
            const predictedProfitRate = this.extractPredictedProfitRate(infoData);
            
            dialogMessage += ` (학습모델: ${predictedProfitRate.toFixed(2)}%)`;
        }
        
        dialogMessage += ` (${currentPriceText})`;
        
        return dialogMessage;
    }

    // 매도 전 예상 수익률 계산 로그 생성
    generateCalculationLog(infoData, sellProfitRate, buyPrice, currentPrice) {
        const predictedProfitRate = this.extractPredictedProfitRate(infoData);
        const basicProfitRate = this.calculateBasicProfitRate(buyPrice, currentPrice);
        
        console.log(`📊 트레이너: N/B 길드에서 Info Panel 기반 고급 매도 전 예상 수익률 계산 완료:`);
        console.log(`   - Info Panel 데이터:`, infoData);
        console.log(`   - 학습 모델 예측: ${predictedProfitRate.toFixed(2)}%`);
        
        if (infoData && infoData.avgPrice > 0) {
            const avgPriceProfitRate = this.calculateAvgPriceProfitRate(infoData);
            console.log(`   - 평균단가 대비: ${avgPriceProfitRate.toFixed(2)}% (평균단가: ₩${infoData.avgPrice.toLocaleString()}, 현재가: ₩${infoData.currentPrice.toLocaleString()})`);
        }
        
        console.log(`   - 기본 계산: ${basicProfitRate.toFixed(2)}% (매수가: ₩${buyPrice.toLocaleString()}, 현재가: ₩${currentPrice.toLocaleString()})`);
        console.log(`   - 최종 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
    }
}

// 전역 인스턴스 생성
window.sellProfitCalculator = new SellProfitCalculator();
