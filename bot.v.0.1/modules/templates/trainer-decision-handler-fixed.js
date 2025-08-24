class TrainerDecisionHandler {
    constructor() {
        this.zoneSteps = {};
        this.currentZone = null;
        this.currentStep = null;
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 의사결정 핸들러 초기화 완료`);
        }
    }

    // 메인 의사결정 처리 메서드
    handleTrainerDecision(model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing) {
        // 현재 구역 확인 (game-initializer로 일원화)
        const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
        
        // 구역 변경 시 단계 리셋
        if (this.currentZone !== currentZone) {
            this.resetZoneStep(this.currentZone);
            this.currentZone = currentZone;
        }
        
        // 현재 단계 가져오기
        const currentStep = this.getCurrentStep(currentZone);
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 의사결정: 구역=${currentZone} | 단계=${currentStep.action}`);
        }
        
        // 단계별 처리
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

    // 현재 구역 확인
    getCurrentZone(x, y, config) {
        const centerX = config.width / 2;
        const centerY = config.height / 2;
        
        // 매수 구역 (좌측)
        if (x < centerX - 200 && y > centerY - 100 && y < centerY + 100) {
            return '매수영역';
        }
        // 매도 구역 (우측)
        else if (x > centerX + 200 && y > centerY - 100 && y < centerY + 100) {
            return '매도영역';
        }
        // BTC 시장 탐색 구역 (우측 하단)
        else if (x > config.width - 200 && y > config.height - 200) {
            return 'BTC시장탐색구역';
        }
        // N/B 길드 (좌측 상단)
        else if (x < 200 && y < 200) {
            return 'N/B길드';
        }
        // 신호 대기 센터 (중앙)
        else if (Math.abs(x - centerX) < 150 && Math.abs(y - centerY) < 150) {
            return '신호대기센터';
        }
        // 이동 중
        else {
            return '이동중';
        }
    }

    // 현재 단계 가져오기
    getCurrentStep(zone) {
        const stepIndex = this.getZoneStep(zone, zone);
        
        const zoneSteps = {
            '매수영역': [
                { action: '매수 구역 도착 확인' },
                { action: '매수 수익률 계산' },
                { action: '매수 의사결정' },
                { action: '매수 실행' },
                { action: 'N/B 코인 드랍' },
                { action: '신호 대기 센터 이동' }
            ],
            '매도영역': [
                { action: '매도 구역 도착 확인' },
                { action: '매도 수익률 계산' },
                { action: '매도 의사결정' },
                { action: '매도 실행' },
                { action: '신호 대기 센터 이동' }
            ],
            'BTC시장탐색구역': [
                { action: 'BTC 시장 도착 확인' },
                { action: '매수 수익률 계산' },
                { action: '시장 분석 완료' },
                { action: '신호 대기 센터 이동' }
            ],
            'N/B길드': [
                { action: 'N/B 길드 도착 확인' },
                { action: 'N/B 코인 확인' },
                { action: '신호 대기 센터 이동' }
            ],
            '신호대기센터': [
                { action: '신호 대기 센터 도착 확인' },
                { action: '시장 신호 분석' },
                { action: '다음 목적지 결정' },
                { action: '목적지로 이동' }
            ],
            '이동중': [
                { action: '목적지로 이동' }
            ]
        };
        
        const steps = zoneSteps[zone] || [{ action: '신호 대기 센터 이동' }];
        return steps[stepIndex] || steps[0];
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

    // 매수 전 예상 수익률 계산 (수정된 버전)
    calculateBuyProfitRate(model, config) {
        // 현재 위치 정보 로그
        if (window.logManager) {
            const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
            const btcMarketPos = `(${Math.round(config.width - 100)}, ${Math.round(config.height - 100)})`;
            const distanceToBTCMarket = Math.sqrt((model.circle.x - (config.width - 100)) ** 2 + (model.circle.y - (config.height - 100)) ** 2);
            window.logManager.addLog(`📊 수익률 계산 시작: 현재위치 ${currentPos} | BTC시장위치 ${btcMarketPos} | BTC시장까지거리 ${Math.round(distanceToBTCMarket)}px`);
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
                window.logManager.addLog(`❌ BTC 시장 데이터를 가져올 수 없음`);
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
        
        // 매수 전 예상 수익률 계산 로직 (개선된 버전)
        let buyProfitRate = 0;
        
        // 1. 현재 수익률 기반 (실제 보유 자산 기준)
        if (btcBalance > 0 && avgPrice > 0) {
            const currentProfitRate = ((currentPrice - avgPrice) / avgPrice) * 100;
            buyProfitRate += currentProfitRate * 0.3; // 현재 수익률의 30% 반영
        }
        
        // 2. 시장 신호에 따른 기본 수익률 (더 현실적인 값)
        if (currentMajority.includes('BLUE')) {
            buyProfitRate += 0.8 + Math.random() * 1.2; // 0.8% ~ 2.0%
        } else if (currentMajority.includes('ORANGE')) {
            buyProfitRate += 0.2 + Math.random() * 0.8; // 0.2% ~ 1.0%
        } else {
            buyProfitRate += -0.3 + Math.random() * 0.6; // -0.3% ~ 0.3%
        }
        
        // 3. 가격 변동률 반영 (더 보수적)
        const priceChangeMatch = priceChangeText.match(/-?[\d.]+/);
        if (priceChangeMatch) {
            const priceChange = parseFloat(priceChangeMatch[0]);
            buyProfitRate += priceChange * 0.3; // 가격 변동률의 30% 반영 (50%에서 감소)
        }
        
        // 4. 구역 강도 반영 (더 세밀한 조정)
        const strength = parseInt(zoneStrength);
        if (strength > 0) {
            buyProfitRate += (strength - 50) * 0.01; // 강도 50 기준으로 ±0.5% 보정 (0.02에서 감소)
        }
        
        // 5. N/B 코인 개수에 따른 보정 (더 현실적)
        if (nbCoins > 0) {
            buyProfitRate += (nbCoins * 0.02); // 코인 1개당 0.02% 추가 (0.05에서 감소)
        }
        
        // 6. 포트폴리오 비율 반영 (더 보수적)
        const totalValue = (btcBalance * currentPrice) + krwBalance;
        if (totalValue > 0) {
            const btcRatio = (btcBalance * currentPrice) / totalValue;
            if (btcRatio > 0.8) {
                buyProfitRate -= 0.3; // BTC 비중이 높으면 매수 신중 (0.5에서 감소)
            } else if (btcRatio < 0.2) {
                buyProfitRate += 0.3; // BTC 비중이 낮으면 매수 적극 (0.5에서 감소)
            }
        }
        
        // 7. 랜덤 요소 추가 (더 현실적)
        buyProfitRate += (Math.random() - 0.5) * 0.5; // ±0.25% 랜덤 요소
        
        // 8. 최종 보정 (과도한 수익률 방지)
        buyProfitRate = Math.max(-5, Math.min(10, buyProfitRate)); // -5% ~ 10% 범위로 제한
        
        if (window.logManager) {
            window.logManager.addLog(`📊 매수 예상 수익률 계산 완료: ${buyProfitRate.toFixed(2)}%`);
        }
        
        return buyProfitRate;
    }

    // 매도 전 예상 수익률 계산
    calculateSellProfitRate(model, config) {
        // 매도 수익률 계산 로직 (매수와 유사하지만 매도 관점)
        let sellProfitRate = 0;
        
        // 실제 거래 데이터 가져오기
        const majorityElement = document.getElementById('majority-zone');
        const currentPriceElement = document.getElementById('right-trading-current-price');
        const btcBalanceElement = document.getElementById('btc-balance');
        const avgPriceElement = document.getElementById('selected-coin-avg-price');
        
        if (!majorityElement || !currentPriceElement || !btcBalanceElement || !avgPriceElement) {
            return 0;
        }
        
        const currentPrice = parseFloat(currentPriceElement.textContent.replace(/[₩,]/g, ''));
        const btcBalance = parseFloat(btcBalanceElement.textContent.match(/[\d.]+/)?.[0] || '0');
        const avgPriceText = avgPriceElement.textContent;
        const avgPriceMatch = avgPriceText.match(/[\d,]+/);
        const avgPrice = avgPriceMatch ? parseFloat(avgPriceMatch[0].replace(/,/g, '')) : currentPrice;
        
        // 현재 수익률 계산
        if (btcBalance > 0 && avgPrice > 0) {
            sellProfitRate = ((currentPrice - avgPrice) / avgPrice) * 100;
        }
        
        // 시장 신호에 따른 보정
        const currentMajority = majorityElement.textContent.trim();
        if (currentMajority.includes('RED')) {
            sellProfitRate += 0.5; // 매도 신호 시 수익률 증가
        } else if (currentMajority.includes('BLUE')) {
            sellProfitRate -= 0.3; // 매수 신호 시 매도 신중
        }
        
        // 랜덤 요소
        sellProfitRate += (Math.random() - 0.5) * 0.3;
        
        if (window.logManager) {
            window.logManager.addLog(`📊 매도 예상 수익률 계산 완료: ${sellProfitRate.toFixed(2)}%`);
        }
        
        return sellProfitRate;
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
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 의사결정 완료: 수익률 ${buyProfitRate?.toFixed(2) || 'N/A'}%`);
        }
        this.nextZoneStep('매수영역');
        return '매수 실행';
    }

    handleBuyExecution(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매수 실행 완료`);
        }
        this.nextZoneStep('매수영역');
        return 'N/B 코인 드랍';
    }

    handleNBCoinDrop(model, config) {
        if (window.nbCoinDropSystem) {
            window.nbCoinDropSystem.dropNBCoin(model.circle.x, model.circle.y);
        }
        if (window.logManager) {
            window.logManager.addLog(`✅ N/B 코인 드랍 완료`);
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

    handleSellProfitCalculation(model, currentMajority, config) {
        // 수정된 부분: 올바른 매개변수로 호출
        const sellProfitRate = this.calculateSellProfitRate(model, config);
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 수익률 계산 완료: ${sellProfitRate.toFixed(2)}%`);
        }
        this.nextZoneStep('매도영역');
        return '매도 의사결정';
    }

    handleSellDecision(model, sellProfitRate, startX, topY, spacing, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 의사결정 완료: 수익률 ${sellProfitRate?.toFixed(2) || 'N/A'}%`);
        }
        this.nextZoneStep('매도영역');
        return '매도 실행';
    }

    handleSellExecution(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 매도 실행 완료`);
        }
        this.nextZoneStep('매도영역');
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
        return '신호 대기 센터 이동';
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
        // 다음 목적지 결정 로직
        let nextDestination = '신호 대기 센터';
        
        if (currentMajority.includes('BLUE') && buyProfitRate > 0.5) {
            nextDestination = '매수영역';
        } else if (currentMajority.includes('RED') && sellProfitRate > 0.3) {
            nextDestination = '매도영역';
        } else if (nbCoins < 3) {
            nextDestination = 'N/B길드';
        } else {
            nextDestination = 'BTC시장탐색구역';
        }
        
        if (window.logManager) {
            window.logManager.addLog(`✅ 다음 목적지 결정: ${nextDestination}`);
        }
        this.nextZoneStep('신호대기센터');
        return '목적지로 이동';
    }

    handleDestinationMove(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 목적지로 이동 중`);
        }
        this.nextZoneStep('신호대기센터');
        return '신호 대기 센터 이동';
    }

    handleMoveToSignalCenter(model, config) {
        if (window.logManager) {
            window.logManager.addLog(`✅ 신호 대기 센터로 이동 중`);
        }
        return '신호 대기 센터 이동';
    }
}

// 전역 객체로 등록
window.TrainerDecisionHandler = TrainerDecisionHandler;
