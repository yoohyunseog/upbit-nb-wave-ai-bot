// 수익률 계산 모듈
// currentPrice와 expectedSellPrice 오류 해결을 위한 전용 모듈

class ProfitRateCalculator {
    constructor() {
        this.currentPrice = 0;
        this.expectedSellPrice = 0;
        this.buyPrice = 0;
        this.basicProfitRate = 0;
        this.predictedProfitRate = 0;
        this.sellProfitRate = 0;
    }

    // 현재 가격 추출 및 설정
    extractCurrentPrice() {
        try {
            const currentPriceElement = document.getElementById('trading-current-price');
            if (currentPriceElement) {
                const currentPriceText = currentPriceElement.textContent;
                const currentPriceMatch = currentPriceText.match(/₩([\d,]+\.?\d*)/);
                this.currentPrice = currentPriceMatch ? parseFloat(currentPriceMatch[1].replace(/,/g, '')) : 0;
            }
            
            // selected-coin-price에서도 시도
            if (this.currentPrice === 0) {
                const coinPriceElement = document.getElementById('selected-coin-price');
                if (coinPriceElement) {
                    const priceMatch = coinPriceElement.textContent.match(/₩([\d,]+\.?\d*)/);
                    this.currentPrice = priceMatch ? parseFloat(priceMatch[1].replace(/,/g, '')) : 0;
                }
            }
            
            return this.currentPrice;
        } catch (error) {
            console.error('현재 가격 추출 오류:', error);
            this.currentPrice = 0;
            return 0;
        }
    }

    // 예상 매도가 계산
    calculateExpectedSellPrice(profitRate = 5.0) {
        try {
            this.extractCurrentPrice();
            this.expectedSellPrice = this.currentPrice * (1 + profitRate / 100);
            return this.expectedSellPrice;
        } catch (error) {
            console.error('예상 매도가 계산 오류:', error);
            this.expectedSellPrice = 0;
            return 0;
        }
    }

    // 기본 수익률 계산 (5% 기준)
    calculateBasicProfitRate() {
        try {
            this.extractCurrentPrice();
            if (this.currentPrice > 0) {
                this.calculateExpectedSellPrice(5.0);
                this.basicProfitRate = ((this.expectedSellPrice - this.currentPrice) / this.currentPrice) * 100;
            } else {
                this.basicProfitRate = 0;
            }
            return this.basicProfitRate;
        } catch (error) {
            console.error('기본 수익률 계산 오류:', error);
            this.basicProfitRate = 0;
            return 0;
        }
    }

    // 매도 수익률 계산
    calculateSellProfitRate(buyPrice) {
        try {
            this.buyPrice = buyPrice || 0;
            this.extractCurrentPrice();
            
            if (this.buyPrice > 0 && this.currentPrice > 0) {
                this.sellProfitRate = ((this.currentPrice - this.buyPrice) / this.buyPrice) * 100;
            } else {
                this.sellProfitRate = 0;
            }
            return this.sellProfitRate;
        } catch (error) {
            console.error('매도 수익률 계산 오류:', error);
            this.sellProfitRate = 0;
            return 0;
        }
    }

    // 안전한 로그 메시지 생성
    generateSafeLogMessage(type = 'basic') {
        try {
            this.extractCurrentPrice();
            
            const currentPriceStr = this.currentPrice > 0 ? this.currentPrice.toLocaleString() : '0';
            const expectedSellPriceStr = this.expectedSellPrice > 0 ? this.expectedSellPrice.toLocaleString() : '0';
            const buyPriceStr = this.buyPrice > 0 ? this.buyPrice.toLocaleString() : '0';
            
            switch (type) {
                case 'basic':
                    return `기본 계산: ${this.basicProfitRate.toFixed(2)}% (현재가: ₩${currentPriceStr}, 예상매도가: ₩${expectedSellPriceStr})`;
                
                case 'sell':
                    return `매도 수익률: ${this.sellProfitRate.toFixed(2)}% (매수가: ₩${buyPriceStr}, 현재가: ₩${currentPriceStr})`;
                
                case 'buy':
                    return `매수 전 예상 수익률: ${this.basicProfitRate.toFixed(2)}% (현재가: ₩${currentPriceStr})`;
                
                default:
                    return `수익률 계산: ${this.basicProfitRate.toFixed(2)}%`;
            }
        } catch (error) {
            console.error('로그 메시지 생성 오류:', error);
            return `수익률 계산 오류: ${error.message}`;
        }
    }

    // 모든 데이터 검증
    isDataValid() {
        return this.currentPrice > 0;
    }

    // 데이터 리셋
    reset() {
        this.currentPrice = 0;
        this.expectedSellPrice = 0;
        this.buyPrice = 0;
        this.basicProfitRate = 0;
        this.predictedProfitRate = 0;
        this.sellProfitRate = 0;
    }

    // 현재 상태 출력
    getStatus() {
        return {
            currentPrice: this.currentPrice,
            expectedSellPrice: this.expectedSellPrice,
            buyPrice: this.buyPrice,
            basicProfitRate: this.basicProfitRate,
            predictedProfitRate: this.predictedProfitRate,
            sellProfitRate: this.sellProfitRate,
            isValid: this.isDataValid()
        };
    }
}

// 전역 인스턴스 생성
window.profitRateCalculator = new ProfitRateCalculator();

console.log('🧮 ProfitRateCalculator 모듈 로드 완료');
