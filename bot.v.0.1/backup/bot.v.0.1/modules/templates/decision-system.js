// Decision System Module
// AI 트레이너의 의사결정 시스템

class DecisionSystem {
    constructor() {
        this.zones = {
            BUY: '매수영역',
            SELL: '매도영역',
            WAIT: '대기영역',
            SIGNAL_WAIT: '신호대기센터',
            NB_GUILD: 'N/B길드',
            BTC_MARKET: 'BTC시장',
            OTHER: '기타영역'
        };
    }

    // 현재 구역 감지
    getCurrentZone(x, y, startX, topY, spacing, config) {
        // 매수 영역 감지
        if (Math.abs(x - startX) < 50 && Math.abs(y - topY) < 50) {
            return this.zones.BUY;
        }
        // 매도 영역 감지
        else if (Math.abs(x - (startX + spacing)) < 50 && Math.abs(y - topY) < 50) {
            return this.zones.SELL;
        }
        // 대기 영역 감지
        else if (Math.abs(x - (startX + spacing * 2)) < 50 && Math.abs(y - topY) < 50) {
            return this.zones.WAIT;
        }
        // 신호 대기 센터 감지 (화면 중앙)
        else if (Math.abs(x - (config.width / 2)) < 60 && Math.abs(y - (config.height / 2)) < 60) {
            return this.zones.SIGNAL_WAIT;
        }
        // N/B 길드 감지
        else if (Math.abs(x - 100) < 60 && Math.abs(y - 100) < 60) {
            return this.zones.NB_GUILD;
        }
        // BTC 시장 감지
        else if (Math.abs(x - (config.width - 100)) < 60 && Math.abs(y - (config.height - 100)) < 60) {
            return this.zones.BTC_MARKET;
        }
        // 기타 영역
        else {
            return this.zones.OTHER;
        }
    }

    // 구역별 의사 결정
    getZoneDecision(zone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        switch (zone) {
            case this.zones.BUY:
                // 매수 영역에서는 BLUE 신호이고 매수 전 예상 수익률이 계산된 경우
                if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
                    const currentProfitRate = window.learningSystem.getCurrentProfitRate();
                    
                    // 매수 전 예상 수익률이 현재 수익률보다 낮으면 즉시 매수 (더 유리한 가격)
                    if (buyProfitRate < currentProfitRate) {
                        console.log(`📈 매수영역에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) < 현재 수익률(${currentProfitRate.toFixed(2)}%) → 즉시 매수!`);
                        return {
                            action: '매수',
                            targetX: startX,
                            targetY: topY
                        };
                    }
                    // 매수 전 예상 수익률이 양수이고 현재 수익률보다 높아도 매수 (상승 기대)
                    else if (buyProfitRate > 0) {
                        console.log(`📈 매수영역에서 매수 조건 만족: 예상 수익률(${buyProfitRate.toFixed(2)}%) > 0 → 매수 진행`);
                        return {
                            action: '매수',
                            targetX: startX,
                            targetY: topY
                        };
                    }
                }
                return null; // 매수 조건이 맞지 않으면 의사 결정 없음
                
            case this.zones.SELL:
                // 매도 영역에서는 N/B 코인이 있고 매도 전 예상 수익률이 있을 때만 매도
                if (nbCoins > 0 && sellProfitRate !== 0) {
                    return {
                        action: '매도',
                        targetX: startX + spacing,
                        targetY: topY
                    };
                }
                return null; // 매도 조건이 맞지 않으면 의사 결정 없음
                
            case this.zones.NB_GUILD:
                // N/B 길드에서는 N/B 코인이 있고 매도 전 예상 수익률이 계산되지 않았을 때만 계산
                if (nbCoins > 0 && sellProfitRate === 0) {
                    return {
                        action: 'N/B 길드 방문',
                        targetX: 100,
                        targetY: 100
                    };
                }
                return null; // 계산 조건이 맞지 않으면 의사 결정 없음
                
            case this.zones.BTC_MARKET:
                // BTC 시장에서는 BLUE 신호일 때 매수 관련 의사결정
                if (currentMajority === 'BLUE') {
                    // 매수 전 예상 수익률이 계산되지 않았을 때는 정보 수집 완료로 설정
                    if (buyProfitRate === 0) {
                        return {
                            action: '정보 수집 완료',
                            targetX: config.width - 100,
                            targetY: config.height - 100
                        };
                    }
                    // 매수 전 예상 수익률이 이미 계산된 경우 매수 의사결정
                    else if (buyProfitRate !== 0) {
                        const currentProfitRate = window.learningSystem.getCurrentProfitRate();
                        
                        // BLUE 신호에서는 예상 수익률이 계산되면 매수 (음수여도 매수)
                        console.log(`📈 BTC시장에서 매수 조건 만족: BLUE 신호 + 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%) → 매수 진행`);
                        return {
                            action: '매수',
                            targetX: startX,
                            targetY: topY
                        };
                    }
                }
                return null; // BLUE 신호가 아니거나 매수 조건이 맞지 않으면 의사 결정 없음
                
            case this.zones.WAIT:
            case this.zones.SIGNAL_WAIT:
            case this.zones.OTHER:
                // 대기 영역, 신호 대기 센터, 기타 영역에서는 의사 결정 없음
                return null;
                
            default:
                return null;
        }
    }

    // 다음 의사결정 결정
    getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config) {
        // 우선순위: 매도 > 매수 > 대기
        
        // 1. 매도 우선 (N/B 코인이 있고 매도 전 예상 수익률이 계산된 경우)
        if (nbCoins > 0 && sellProfitRate !== 0) {
            return {
                action: '매도',
                targetX: startX + spacing,
                targetY: topY
            };
        }
        
        // 2. 매도 준비 (N/B 코인이 있고 ORANGE 구역이지만 매도 전 예상 수익률이 계산되지 않은 경우)
        if (nbCoins > 0 && sellProfitRate === 0 && currentMajority === 'ORANGE') {
            return {
                action: 'N/B 길드 방문',
                targetX: 100,
                targetY: 100
            };
        }
        
        // 2-1. 매도 준비 시도했지만 BLUE 구역인 경우 → 신호 대기
        if (nbCoins > 0 && sellProfitRate === 0 && currentMajority === 'BLUE') {
            return {
                action: '신호 대기',
                targetX: config.width / 2,
                targetY: config.height / 2
            };
        }
        
        // 3. 매수 (BLUE 신호이고 매수 전 예상 수익률이 계산된 경우)
        if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
            // BLUE 신호에서는 예상 수익률이 계산되면 매수 (음수여도 매수)
            console.log(`📈 매수 조건 만족: BLUE 신호 + 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%) → 매수 진행`);
            return {
                action: '매수',
                targetX: startX,
                targetY: topY
            };
        }
        
        // 4. 매수 준비 (BLUE 신호이지만 매수 전 예상 수익률이 계산되지 않은 경우)
        if (currentMajority === 'BLUE' && buyProfitRate === 0) {
            return {
                action: 'BTC 시장 방문',
                targetX: config.width - 100,
                targetY: config.height - 100
            };
        }
        
        // 5. 대기 (조건이 맞지 않는 경우) - 신호 대기 센터로 이동
        return {
            action: '신호 대기',
            targetX: config.width / 2,
            targetY: config.height / 2
        };
    }

    // 의사결정에 따른 색상 결정
    getActionColor(action) {
        const colorMap = {
            '매수': 0x0088ff,           // 파란색
            '매도': 0xff8800,           // 주황색
            'BTC 시장 방문': 0x0088ff,  // 파란색
            'N/B 길드 방문': 0xff8800,  // 주황색
            '대기': 0xffff00,           // 노란색
            '신호 대기': 0x88ccff,      // 하늘색
            'BTC 시장 탐색': 0x0088ff   // 파란색
        };
        return colorMap[action] || 0x88ccff; // 기본값: 하늘색
    }

    // 의사결정 로그 메시지 생성
    generateDecisionLogMessage(action, currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate) {
        const baseMessage = `🎯 [의사결정: ${action}]`;
        
        switch (action) {
            case '매수':
                return `${baseMessage} 매수 영역으로 이동 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}% (${currentPriceText})`;
            case '매도':
                return `${baseMessage} 매도 영역으로 이동 중... N/B 코인: ${nbCoins}개, 학습모델 기반 예상 수익률: ${sellProfitRate.toFixed(2)}% (${currentPriceText})`;
            case 'BTC 시장 방문':
                return `${baseMessage} BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정 (${currentPriceText})`;
            case 'N/B 길드 방문':
                return `${baseMessage} N/B 길드로 이동 중... 매도 전 예상 수익률 계산 예정 (${currentPriceText})`;
            case '대기':
                return `${baseMessage} 신호 대기 중... N/B 코인: ${nbCoins}개, N/B 미네랄: ${nbMinerals.toFixed(2)}% (드랍 아이템: ${nbCoinItems.length}개) (${currentPriceText})`;
            case '신호 대기':
                return `${baseMessage} 신호 대기 센터에서 대기 중... N/B 코인: ${nbCoins}개, N/B 미네랄: ${nbMinerals.toFixed(2)}% (드랍 아이템: ${nbCoinItems.length}개) (${currentPriceText})`;
            case 'BTC 시장 탐색':
                return `${baseMessage} BTC 시장 탐색에서 매수 전 예상 수익률 계산 중... N/B 코인: ${nbCoins}개, N/B 미네랄: ${nbMinerals.toFixed(2)}% (드랍 아이템: ${nbCoinItems.length}개) (${currentPriceText})`;
            default:
                return `${baseMessage} ${action} 중... (${currentPriceText})`;
        }
    }

    // 이동 중 메시지 생성
    generateMovingMessage(targetAction, distanceToTarget, currentMajority, nbCoins, buyProfitRate, sellProfitRate, currentZone, nextDecision) {
        let movingMessage = '';
        
        if (targetAction === '매수') {
            const currentProfitRate = window.learningSystem.getCurrentProfitRate();
            if (buyProfitRate < currentProfitRate) {
                movingMessage = `🎯 [의사결정: 매수] 매수 영역으로 이동 중... 예상 수익률(${buyProfitRate.toFixed(2)}%) < 현재 수익률(${currentProfitRate.toFixed(2)}%) → 즉시 매수! (${Math.round(distanceToTarget)}px 남음)`;
            } else {
                movingMessage = `🎯 [의사결정: 매수] 매수 영역으로 이동 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}% (현재: ${currentProfitRate.toFixed(2)}%) (${Math.round(distanceToTarget)}px 남음)`;
            }
        } else if (targetAction === '매도') {
            movingMessage = `🎯 [의사결정: 매도] 매도 영역으로 이동 중... N/B 코인: ${nbCoins}개, 학습모델 기반 예상 수익률: ${sellProfitRate.toFixed(2)}% (${Math.round(distanceToTarget)}px 남음)`;
        } else if (targetAction === 'BTC 시장 방문') {
            movingMessage = `🎯 [의사결정: 매수 준비] BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정 (${Math.round(distanceToTarget)}px 남음)`;
        } else if (targetAction === 'N/B 길드 방문') {
            movingMessage = `🎯 [의사결정: 매도 준비] N/B 길드로 이동 중... 매도 전 예상 수익률 계산 예정 (${Math.round(distanceToTarget)}px 남음)`;
        } else if (targetAction === '대기') {
            if (nextDecision && nextDecision.action !== '대기') {
                movingMessage = `🎯 [의사결정: 대기] 현재 구역(${currentZone})에서 의사 결정 없음 → ${nextDecision.action} 준비 중... (${Math.round(distanceToTarget)}px 남음)`;
            } else {
                movingMessage = `🎯 [의사결정: 대기] 대기 영역으로 이동 중... (${Math.round(distanceToTarget)}px 남음)`;
            }
        } else if (targetAction === '신호 대기') {
            if (nextDecision && nextDecision.action !== '신호 대기') {
                movingMessage = `🔵 [의사결정: 신호 대기] 신호 대기 센터로 이동 중... → ${nextDecision.action} 준비 중... (${Math.round(distanceToTarget)}px 남음)`;
            } else {
                movingMessage = `🔵 [의사결정: 신호 대기] 신호 대기 센터로 이동 중... (${Math.round(distanceToTarget)}px 남음)`;
            }
        } else if (targetAction === 'BTC 시장 탐색') {
            movingMessage = `🔵 [BTC 탐색] BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정 (${Math.round(distanceToTarget)}px 남음)`;
        } else {
            movingMessage = `🎯 [의사결정: ${targetAction}] ${currentZone}에서 이동 중... (${Math.round(distanceToTarget)}px 남음)`;
        }
        
        return movingMessage;
    }
}

// 전역 인스턴스 생성
window.decisionSystem = new DecisionSystem();
