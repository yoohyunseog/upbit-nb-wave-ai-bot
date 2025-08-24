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

    // 현재 가격 추출
    extractCurrentPrice() {
        try {
            const priceElement = document.querySelector('.current-price, .price-display, #current-price');
            if (priceElement) {
                const priceText = priceElement.textContent || priceElement.innerText;
                const priceMatch = priceText.match(/[\d,]+/);
                if (priceMatch) {
                    this.currentPrice = parseFloat(priceMatch[0].replace(/,/g, ''));
                }
            }
        } catch (error) {
            console.error('현재 가격 추출 오류:', error);
            this.currentPrice = 0;
        }
    }

    // 예상 매도가 계산
    calculateExpectedSellPrice(profitPercent = 5.0) {
        try {
            if (this.currentPrice > 0) {
                this.expectedSellPrice = this.currentPrice * (1 + profitPercent / 100);
            } else {
                this.expectedSellPrice = 0;
            }
        } catch (error) {
            console.error('예상 매도가 계산 오류:', error);
            this.expectedSellPrice = 0;
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
            
            // 매수가가 0이면 수익률도 0으로 설정
            if (this.buyPrice <= 0 || this.currentPrice <= 0) {
                this.sellProfitRate = 0;
                console.warn('⚠️ 매수가 또는 현재가가 0이므로 매도 수익률을 0으로 설정');
                return 0;
            }
            
            this.sellProfitRate = ((this.currentPrice - this.buyPrice) / this.buyPrice) * 100;
            
            // 수익률 범위 검증 (-100% ~ 1000%)
            if (this.sellProfitRate < -100 || this.sellProfitRate > 1000) {
                console.warn(`⚠️ 비정상적인 매도 수익률 감지: ${this.sellProfitRate.toFixed(2)}%, 0으로 재설정`);
                this.sellProfitRate = 0;
            }
            
            return this.sellProfitRate;
        } catch (error) {
            console.error('매도 수익률 계산 오류:', error);
            this.sellProfitRate = 0;
            return 0;
        }
    }

    // 매수 전 예상 수익률 계산 (검증 강화)
    calculateBuyProfitRate() {
        try {
            this.extractCurrentPrice();
            
            if (this.currentPrice <= 0) {
                console.warn('⚠️ 현재가가 0이므로 매수 수익률을 0으로 설정');
                return 0;
            }
            
            // 기본 5% 예상 수익률 계산
            this.calculateExpectedSellPrice(5.0);
            let buyProfitRate = ((this.expectedSellPrice - this.currentPrice) / this.currentPrice) * 100;
            
            // 수익률 범위 검증 (-10% ~ 15%)
            if (buyProfitRate < -10 || buyProfitRate > 15) {
                console.warn(`⚠️ 비정상적인 매수 수익률 감지: ${buyProfitRate.toFixed(2)}%, 범위 내로 조정`);
                buyProfitRate = Math.max(-10, Math.min(15, buyProfitRate));
            }
            
            return buyProfitRate;
        } catch (error) {
            console.error('매수 수익률 계산 오류:', error);
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
