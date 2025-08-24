// 현재 가격 관리 모듈
// currentPrice 변수 정의 및 관련 계산 로직을 담당

class CurrentPriceManager {
    constructor() {
        this.currentPrice = 0;
        this.currentPriceText = '';
        this.infoData = {};
    }

    // DOM에서 현재 가격 텍스트 추출
    getCurrentPriceText() {
        const currentPriceElement = document.getElementById('trading-current-price');
        this.currentPriceText = currentPriceElement ? currentPriceElement.textContent : '₩0';
        return this.currentPriceText;
    }

    // 현재 가격을 숫자로 변환
    parseCurrentPrice(priceText = null) {
        const textToParse = priceText || this.currentPriceText;
        const currentPriceMatch = textToParse.match(/₩([\d,]+\.?\d*)/);
        this.currentPrice = currentPriceMatch ? parseFloat(currentPriceMatch[1].replace(/,/g, '')) : 0;
        return this.currentPrice;
    }

    // Info Panel의 모든 데이터 추출
    getInfoPanelData() {
        const data = {};
        
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
            const priceMatch = coinPriceElement.textContent.match(/₩([\d,]+\.?\d*)/);
            data.currentPrice = priceMatch ? parseFloat(priceMatch[1].replace(/,/g, '')) : 0;
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
        
        this.infoData = data;
        return data;
    }

    // 매수 전 예상 수익률 계산
    calculateBuyProfitRate() {
        try {
            const currentPriceText = this.getCurrentPriceText();
            const currentPrice = this.parseCurrentPrice(currentPriceText);
            const infoData = this.getInfoPanelData();
            
            if (currentPrice <= 0) {
                return this.calculateBasicZoneBasedProfitRate();
            }

            // 1. 게임 화면 표시에서 수익률 가져오기
            if (window.buyProfitRateDisplay && window.buyProfitRateDisplay.text) {
                const buyText = window.buyProfitRateDisplay.text;
                const buyMatch = buyText.match(/매수 전 예상 수익률:\s*([+-]?\d+\.?\d*)%/);
                if (buyMatch) {
                    const rate = parseFloat(buyMatch[1]);
                    if (rate !== 0) {
                        return rate;
                    }
                }
            }

            // 2. 학습 모델 예측 수익률 활용
            const predictedProfitRate = infoData.predictedProfitRate || 0;
            
            // 3. 기본 계산: ProfitRateCalculator 모듈 사용
            let basicProfitRate = 0;
            if (window.profitRateCalculator && currentPrice > 0) {
                basicProfitRate = window.profitRateCalculator.calculateBasicProfitRate();
            } else {
                // 폴백: 기본 계산
                const expectedSellPrice = currentPrice * 1.05;
                basicProfitRate = currentPrice > 0 ? ((expectedSellPrice - currentPrice) / currentPrice) * 100 : 0;
            }
            
            // 4. Info Panel 데이터를 활용한 고급 매수 전 예상 수익률 계산
            let advancedBuyProfitRate = predictedProfitRate;
            
            // 평균 단가 대비 현재가 수익률 계산
            if (infoData.avgPrice > 0 && infoData.currentPrice > 0) {
                const avgPriceProfitRate = ((infoData.currentPrice - infoData.avgPrice) / infoData.avgPrice) * 100;
                console.log(`📊 BTC 탐색 - 평균단가 대비 수익률: ${avgPriceProfitRate.toFixed(2)}% (평균단가: ₩${infoData.avgPrice ? infoData.avgPrice.toLocaleString() : '0'}, 현재가: ₩${infoData.currentPrice ? infoData.currentPrice.toLocaleString() : '0'})`);
                
                // 가중치 조정: 학습모델 50%, 평균단가 30%, 기본계산 20%
                advancedBuyProfitRate = (predictedProfitRate * 0.5) + (avgPriceProfitRate * 0.3) + (basicProfitRate * 0.2);
            } else {
                // 평균단가 정보가 없으면 기존 방식 사용
                advancedBuyProfitRate = (predictedProfitRate * 0.7) + (basicProfitRate * 0.3);
            }
            
            // 5. 결과가 0이면 구역 기반 계산 사용
            if (advancedBuyProfitRate === 0) {
                return this.calculateBasicZoneBasedProfitRate();
            }
            
            return advancedBuyProfitRate;
            
        } catch (error) {
            console.warn('매수 수익률 계산 오류:', error);
            return this.calculateBasicZoneBasedProfitRate();
        }
    }

    // 기본 구역 기반 수익률 계산
    calculateBasicZoneBasedProfitRate() {
        try {
            const currentZoneElement = document.getElementById('right-trading-current-zone');
            const zoneStrengthElement = document.getElementById('right-trading-zone-strength');
            
            let currentZone = 'BLUE';
            let zoneStrength = 0;
            
            if (currentZoneElement) {
                const zoneText = currentZoneElement.textContent;
                if (zoneText.includes('ORANGE')) currentZone = 'ORANGE';
                else if (zoneText.includes('BLUE')) currentZone = 'BLUE';
            }
            
            if (zoneStrengthElement) {
                const strengthText = zoneStrengthElement.textContent;
                const strengthMatch = strengthText.match(/-?\d+/);
                if (strengthMatch) {
                    zoneStrength = parseFloat(strengthMatch[0]);
                }
            }
            
            let buyProfitRate = 0;
            
            if (currentZone === 'ORANGE') {
                buyProfitRate = Math.max(0.5, Math.min(3.0, zoneStrength * 0.1));
            } else if (currentZone === 'BLUE') {
                buyProfitRate = Math.max(-2.0, Math.min(1.5, -zoneStrength * 0.05));
            } else {
                buyProfitRate = Math.random() * 2 - 1;
            }
            
            return parseFloat(buyProfitRate.toFixed(2));
            
        } catch (error) {
            console.warn('기본 구역 기반 수익률 계산 오류:', error);
            return 0;
        }
    }

    // 매도 전 예상 수익률 계산
    calculateSellProfitRate(buyPrice) {
        const currentPriceText = this.getCurrentPriceText();
        const currentPrice = this.parseCurrentPrice(currentPriceText);
        
        if (currentPrice <= 0 || buyPrice <= 0) {
            return 0;
        }
        
        return ((currentPrice - buyPrice) / buyPrice) * 100;
    }

    // 평균 단가 대비 수익률 계산
    calculateAvgPriceProfitRate() {
        const infoData = this.getInfoPanelData();
        
        if (infoData.avgPrice > 0 && infoData.currentPrice > 0) {
            return ((infoData.currentPrice - infoData.avgPrice) / infoData.avgPrice) * 100;
        }
        
        return 0;
    }

    // 현재 가격 정보 로그 생성
    generateCurrentPriceLog() {
        const currentPriceText = this.getCurrentPriceText();
        const currentPrice = this.parseCurrentPrice(currentPriceText);
        const infoData = this.getInfoPanelData();
        
        return {
            currentPriceText,
            currentPrice,
            infoData,
            avgPriceProfitRate: this.calculateAvgPriceProfitRate()
        };
    }

    // 현재 가격이 유효한지 확인
    isValidCurrentPrice() {
        const currentPrice = this.parseCurrentPrice();
        return currentPrice > 0;
    }

    // 현재 가격을 포맷된 문자열로 반환
    getFormattedCurrentPrice() {
        const currentPrice = this.parseCurrentPrice();
        return currentPrice > 0 ? currentPrice.toLocaleString() : '0';
    }
}

// 전역 객체로 등록
window.currentPriceManager = new CurrentPriceManager();
