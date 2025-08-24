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

    // 좌측 패널의 매도 가능한 코인 확인
    getLeftPanelSellableCoins() {
        try {
            console.log('🔍 좌측 패널 매도 가능 코인 확인 시작...');
            
            // 방법 1: 좌측 패널에서 N/B 코인 개수 확인
            const nbCoinElement = document.getElementById('nb-coin-count') || document.getElementById('nb-coin-display');
            if (nbCoinElement) {
                const text = nbCoinElement.textContent || '';
                const match = text.match(/(\d+)/);
                const result = match ? parseInt(match[1]) : 0;
                console.log(`📊 N/B 코인 요소에서 확인: "${text}" → ${result}개`);
                if (result > 0) {
                    console.log(`✅ N/B 코인 요소에서 매도 가능 코인 ${result}개 발견`);
                    return result;
                }
            }
            
            // 방법 2: N/B 코인 드롭 시스템에서 확인
            if (window.gameInitializer?.gameData?.nbCoins !== undefined) {
                const result = window.gameInitializer.gameData.nbCoins;
                console.log(`📊 게임 데이터에서 확인: ${result}개`);
                if (result > 0) {
                    console.log(`✅ 게임 데이터에서 매도 가능 코인 ${result}개 발견`);
                    return result;
                }
            }
            
            // 방법 3: 좌측 패널의 timeframe 카드에서 확인 (더 정확한 방법)
            const tfCards = document.querySelectorAll('.left-panel .timeframe-card');
            let totalSellableCoins = 0;
            console.log(`📊 좌측 패널 카드 개수: ${tfCards.length}개`);
            
            tfCards.forEach((card, index) => {
                const hasActiveTrade = card.querySelector('.badge.bg-danger') !== null; // 실거래 배지 확인
                const hasBtcBalance = card.querySelector('.coin-balance')?.textContent?.includes('0.00000000') === false;
                const coinBalanceText = card.querySelector('.coin-balance')?.textContent || 'N/A';
                const hasSellButton = card.querySelector('.btn-sell') !== null;
                const isSelected = card.classList.contains('selected');
                
                console.log(`📊 카드 ${index + 1}: 실거래=${hasActiveTrade}, BTC잔고=${hasBtcBalance}(${coinBalanceText}), 매도버튼=${hasSellButton}, 선택됨=${isSelected}`);
                
                if (hasActiveTrade || hasBtcBalance || hasSellButton) {
                    totalSellableCoins++;
                    console.log(`✅ 카드 ${index + 1}에서 매도 가능 조건 발견`);
                }
            });
            
            console.log(`📊 최종 매도 가능 코인: ${totalSellableCoins}개`);
            
            // 방법 4: majority-zone 요소에서 ORANGE 확인
            const majorityElement = document.getElementById('majority-zone');
            if (majorityElement) {
                const majorityText = majorityElement.textContent || '';
                console.log(`📊 Majority 확인: "${majorityText}"`);
                if (majorityText.includes('ORANGE') && totalSellableCoins === 0) {
                    console.log(`⚠️ ORANGE 구역이지만 매도 가능 코인이 0개 - 좌측 패널 확인 필요`);
                }
            }
            
            return totalSellableCoins;
        } catch (e) {
            console.error('❌ 좌측 패널 매도 가능 코인 확인 오류:', e);
            return 0;
        }
    }

    // 현재 구역 감지
    getCurrentZone(x, y, startX, topY, spacing, config) {
        console.log(`🔍 구역 감지: x=${x}, y=${y}, startX=${startX}, topY=${topY}, spacing=${spacing}`);
        
        // 매수 영역 감지
        if (Math.abs(x - startX) < 50 && Math.abs(y - topY) < 50) {
            console.log(`📍 매수 영역 감지됨`);
            return this.zones.BUY;
        }
        // 매도 영역 감지
        else if (Math.abs(x - (startX + spacing)) < 50 && Math.abs(y - topY) < 50) {
            console.log(`📍 매도 영역 감지됨`);
            return this.zones.SELL;
        }
        // 대기 영역 감지
        else if (Math.abs(x - (startX + spacing * 2)) < 50 && Math.abs(y - topY) < 50) {
            console.log(`📍 대기 영역 감지됨`);
            return this.zones.WAIT;
        }
        // 신호 대기 센터 감지 (화면 중앙)
        else if (Math.abs(x - (config.width / 2)) < 60 && Math.abs(y - (config.height / 2)) < 60) {
            console.log(`📍 신호 대기 센터 감지됨`);
            return this.zones.SIGNAL_WAIT;
        }
        // N/B 길드 감지
        else if (Math.abs(x - 100) < 60 && Math.abs(y - 100) < 60) {
            console.log(`📍 N/B 길드 감지됨`);
            return this.zones.NB_GUILD;
        }
        // BTC 시장 감지
        else if (Math.abs(x - (config.width - 100)) < 60 && Math.abs(y - (config.height - 100)) < 60) {
            console.log(`📍 BTC 시장 감지됨`);
            return this.zones.BTC_MARKET;
        }
        // 기타 영역
        else {
            console.log(`📍 기타 영역 (구역 감지 실패)`);
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
        // 좌측 패널의 매도 가능한 코인 확인
        const leftPanelSellableCoins = this.getLeftPanelSellableCoins();
        
        // 현재 구역 확인 - 실제 트레이너 위치를 사용해야 함
        // 트레이너의 현재 위치를 가져오기
        const trainer = window.aiModels?.find(m => m.isTrainer);
        let currentZone = this.zones.OTHER;
        
        if (trainer && trainer.circle) {
            const x = trainer.circle.x;
            const y = trainer.circle.y;
            currentZone = this.getCurrentZone(x, y, startX, topY, spacing, config);
        } else {
            console.log('⚠️ 트레이너 위치를 찾을 수 없음 - 기본값 사용');
        }
        
        // 디버깅 로그 - 모든 조건 출력
        console.log('🔍 의사결정 디버깅:', {
            currentMajority,
            nbCoins,
            buyProfitRate,
            sellProfitRate,
            leftPanelSellableCoins,
            currentZone,
            trainerPosition: trainer ? { x: trainer.circle.x, y: trainer.circle.y } : 'N/A',
            startX,
            topY,
            spacing
        });
        
        // 우선순위: 좌측 패널 매도 > 매도 > 매수 > 대기
        
        // 1. 좌측 패널 매도 우선 (좌측 패널에 매도할 코인이 있고 ORANGE 구역인 경우)
        // 단, 신호 대기 센터에서는 바로 매도하지 않고 매도 구역으로 이동
        if (leftPanelSellableCoins > 0 && currentMajority === 'ORANGE') {
            console.log(`✅ 좌측 패널 매도 조건 만족: leftPanelSellableCoins=${leftPanelSellableCoins}, currentMajority=${currentMajority}`);
            
            // 신호 대기 센터에서는 매도 구역으로 이동
            if (currentZone === this.zones.SIGNAL_WAIT) {
                console.log(`🟠 신호 대기 센터에서 좌측 패널 매도 조건 감지 → 매도 구역으로 이동!`);
                return {
                    action: '매도 구역 이동',
                    targetX: startX + spacing,
                    targetY: topY
                };
            }
            // 매도 구역에서는 바로 매도 실행
            else if (currentZone === this.zones.SELL) {
                console.log(`🟠 매도 구역에서 좌측 패널 매도 조건 만족 → 매도 실행!`);
                return {
                    action: '좌측 패널 매도',
                    targetX: startX + spacing,
                    targetY: topY
                };
            }
            // 다른 구역에서는 매도 구역으로 이동
            else {
                console.log(`🟠 좌측 패널 매도 조건 만족: 매도 가능 코인 ${leftPanelSellableCoins}개 + ORANGE 구역 → 매도 구역으로 이동!`);
                return {
                    action: '매도 구역 이동',
                    targetX: startX + spacing,
                    targetY: topY
                };
            }
        } else {
            console.log(`❌ 좌측 패널 매도 조건 불만족: leftPanelSellableCoins=${leftPanelSellableCoins}, currentMajority=${currentMajority}`);
        }
        
        // 2. 매도 우선 (N/B 코인이 있고 매도 전 예상 수익률이 계산된 경우)
        // 매도는 반드시 매도 구역에서만 실행
        if (nbCoins > 0 && sellProfitRate !== 0) {
            if (currentZone === this.zones.SELL) {
                console.log(`✅ 일반 매도 조건 만족: nbCoins=${nbCoins}, sellProfitRate=${sellProfitRate}, 현재 구역: ${currentZone}`);
                return {
                    action: '매도',
                    targetX: startX + spacing,
                    targetY: topY
                };
            } else {
                console.log(`⚠️ 매도 조건은 만족하지만 매도 구역에 없음: nbCoins=${nbCoins}, sellProfitRate=${sellProfitRate}, 현재 구역: ${currentZone} → 매도 구역으로 이동`);
                return {
                    action: '매도 구역 이동',
                    targetX: startX + spacing,
                    targetY: topY
                };
            }
        } else {
            console.log(`❌ 일반 매도 조건 불만족: nbCoins=${nbCoins}, sellProfitRate=${sellProfitRate}`);
        }
        
        // 3. 매도 준비 (N/B 코인이 있고 ORANGE 구역이지만 매도 전 예상 수익률이 계산되지 않은 경우)
        if (nbCoins > 0 && sellProfitRate === 0 && currentMajority === 'ORANGE') {
            console.log(`✅ 매도 준비 조건 만족: nbCoins=${nbCoins}, sellProfitRate=${sellProfitRate}, currentMajority=${currentMajority}`);
            return {
                action: 'N/B 길드 방문',
                targetX: 100,
                targetY: 100
            };
        } else {
            console.log(`❌ 매도 준비 조건 불만족: nbCoins=${nbCoins}, sellProfitRate=${sellProfitRate}, currentMajority=${currentMajority}`);
        }
        
        // 3-1. 매도 준비 시도했지만 BLUE 구역인 경우 → 신호 대기
        if (nbCoins > 0 && sellProfitRate === 0 && currentMajority === 'BLUE') {
            console.log(`✅ BLUE 구역 매도 준비 조건 만족 → 신호 대기`);
            return {
                action: '신호 대기',
                targetX: config.width / 2,
                targetY: config.height / 2
            };
        }
        
        // 4. 매수 (BLUE 신호이고 매수 전 예상 수익률이 계산된 경우)
        // 매수는 반드시 매수 구역에서만 실행
        if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
            if (currentZone === this.zones.BUY) {
                // BLUE 신호에서는 예상 수익률이 계산되면 매수 (음수여도 매수)
                console.log(`📈 매수 조건 만족: BLUE 신호 + 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%) + 매수 구역 → 매수 진행`);
                return {
                    action: '매수',
                    targetX: startX,
                    targetY: topY
                };
            } else {
                console.log(`⚠️ 매수 조건은 만족하지만 매수 구역에 없음: BLUE 신호 + 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%), 현재 구역: ${currentZone} → 매수 구역으로 이동`);
                return {
                    action: '매수 구역 이동',
                    targetX: startX,
                    targetY: topY
                };
            }
        }
        
        // 5. 매수 준비 (BLUE 신호이지만 매수 전 예상 수익률이 계산되지 않은 경우)
        if (currentMajority === 'BLUE' && buyProfitRate === 0) {
            console.log(`📈 매수 준비 조건 만족 → BTC 시장 방문`);
            return {
                action: 'BTC 시장 방문',
                targetX: config.width - 100,
                targetY: config.height - 100
            };
        }
        
        // 6. 대기 (조건이 맞지 않는 경우) - 신호 대기 센터로 이동
        console.log(`⏳ 모든 조건 불만족 → 신호 대기`);
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
            '좌측 패널 매도': 0xff4400,  // 진한 주황색 (좌측 패널 매도 우선)
            '매도 구역 이동': 0xff6600,  // 중간 주황색 (매도 구역으로 이동)
            '매수 구역 이동': 0x0066ff,  // 진한 파란색 (매수 구역으로 이동)
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
            case '좌측 패널 매도':
                const leftPanelCoins = this.getLeftPanelSellableCoins();
                return `${baseMessage} 좌측 패널 매도 중... 좌측 패널 매도 가능 코인: ${leftPanelCoins}개, N/B 코인: ${nbCoins}개 (${currentPriceText})`;
            case '매도 구역 이동':
                return `${baseMessage} 매도 구역으로 이동 중... N/B 코인: ${nbCoins}개, 학습모델 기반 예상 수익률: ${sellProfitRate.toFixed(2)}% (${currentPriceText})`;
            case '매수 구역 이동':
                return `${baseMessage} 매수 구역으로 이동 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}% (${currentPriceText})`;
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
        } else if (targetAction === '좌측 패널 매도') {
            movingMessage = `🎯 [의사결정: 좌측 패널 매도] 좌측 패널 매도 중... 매도 가능 코인: ${nbCoins}개 (${Math.round(distanceToTarget)}px 남음)`;
        } else if (targetAction === '매도 구역 이동') {
            movingMessage = `🎯 [의사결정: 매도 구역 이동] 매도 구역으로 이동 중... N/B 코인: ${nbCoins}개, 학습모델 기반 예상 수익률: ${sellProfitRate.toFixed(2)}% (${Math.round(distanceToTarget)}px 남음)`;
        } else if (targetAction === '매수 구역 이동') {
            movingMessage = `🎯 [의사결정: 매수 구역 이동] 매수 구역으로 이동 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}% (${Math.round(distanceToTarget)}px 남음)`;
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
