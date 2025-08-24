// Left Panel Comprehensive Logger
// 좌측 패널의 모든 데이터를 포괄적으로 로그에 저장하는 시스템

(function(){
    class LeftPanelComprehensiveLogger {
        constructor() {
            this.logEndpoint = window.LEFT_PANEL_LOG_ENDPOINT || 
                              (window.location.origin + '/api/leftpanel/log');
            this.snapshotEndpoint = window.LEFT_PANEL_SNAPSHOT_ENDPOINT || 
                                   (window.location.origin + '/api/leftpanel/snapshot');
            this.isEnabled = true;
            this.logInterval = 2000; // 2초마다 로그 저장
            this.lastLogTime = 0;
            this.pendingLogs = [];
            
            this.init();
        }

        init() {
            console.log('🔍 좌측 패널 포괄 로거 초기화');
            this.startPeriodicLogging();
            this.setupEventListeners();
        }

        // 모든 좌측 패널 데이터 수집
        collectAllData() {
            const data = {
                timestamp: Date.now(),
                currentTimeframe: this.getCurrentTimeframe(),
                allTimeframes: this.getAllTimeframesData(),
                marketData: this.getMarketData(),
                tradeData: this.getTradeData(),
                systemData: this.getSystemData(),
                uiState: this.getUIState(),
                summary: this.getSummaryData()
            };
            return data;
        }

        // 요약 데이터 생성
        getSummaryData() {
            try {
                const timeframes = this.getAllTimeframesData();
                const summary = {
                    totalTimeframes: Object.keys(timeframes).length,
                    activeTimeframes: Object.values(timeframes).filter(tf => tf.isActive).length,
                    selectedTimeframes: Object.values(timeframes).filter(tf => tf.isSelected).length,
                    totalNbCoins: Object.values(timeframes).reduce((sum, tf) => sum + (tf.nbCoins || 0), 0),
                    totalNbMinerals: Object.values(timeframes).reduce((sum, tf) => sum + (tf.nbMinerals || 0), 0),
                    canSellTimeframes: Object.values(timeframes).filter(tf => tf.canSell).length,
                    averageExpectedReturn: this.calculateAverageExpectedReturn(timeframes),
                    zoneDistribution: this.calculateZoneDistribution(timeframes)
                };
                return summary;
            } catch (e) {
                return {};
            }
        }

        // 평균 예상 수익률 계산
        calculateAverageExpectedReturn(timeframes) {
            try {
                const returns = Object.values(timeframes)
                    .map(tf => tf.expectedReturn)
                    .filter(ret => ret !== null && !isNaN(ret));
                
                if (returns.length === 0) return 0;
                return returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
            } catch (e) {
                return 0;
            }
        }

        // 존 분포 계산
        calculateZoneDistribution(timeframes) {
            try {
                const zones = Object.values(timeframes)
                    .map(tf => tf.zone)
                    .filter(zone => zone !== null);
                
                const distribution = {};
                zones.forEach(zone => {
                    distribution[zone] = (distribution[zone] || 0) + 1;
                });
                
                return distribution;
            } catch (e) {
                return {};
            }
        }

        // 현재 타임프레임 가져오기
        getCurrentTimeframe() {
            try {
                const badge = document.getElementById('left-panel-current-tf');
                if (badge) {
                    const match = badge.textContent.match(/현재 분봉:\s*(\w+)/);
                    return match ? match[1] : null;
                }
                return null;
            } catch (e) {
                return null;
            }
        }

        // 모든 타임프레임 카드 데이터 수집
        getAllTimeframesData() {
            const timeframes = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '1D'];
            const data = {};

            timeframes.forEach(tf => {
                try {
                    const card = document.getElementById(`timeframe-card-${tf}`);
                    if (card) {
                        data[tf] = {
                            zone: this.getZoneFromCard(card),
                            strength: this.getStrengthFromCard(card),
                            price: this.getPriceFromCard(card),
                            priceChange: this.getPriceChangeFromCard(card),
                            mode: this.getModeFromCard(card),
                            nextAction: this.getNextActionFromCard(card),
                            expectedReturn: this.getExpectedReturnFromCard(card),
                            buyExpectedReturn: this.getBuyExpectedReturnFromCard(card),
                            sellExpectedReturn: this.getSellExpectedReturnFromCard(card),
                            learningSummary: this.getLearningSummaryFromCard(card),
                            majority: this.getMajorityFromCard(card),
                            orangeTotal: this.getOrangeTotalFromCard(card),
                            blueTotal: this.getBlueTotalFromCard(card),
                            buyWaitTime: this.getBuyWaitTimeFromCard(card),
                            sellWaitTime: this.getSellWaitTimeFromCard(card),
                            buyElapsed: this.getBuyElapsedFromCard(card),
                            sellElapsed: this.getSellElapsedFromCard(card),
                            nbCoins: this.getNBCoinsFromCard(card),
                            nbMinerals: this.getNBMineralsFromCard(card),
                            isActive: card.classList.contains('active'),
                            isSelected: card.classList.contains('selected'),
                            hasTrade: this.hasActiveTrade(card),
                            canSell: this.canSellFromCard(card)
                        };
                    }
                } catch (e) {
                    console.warn(`타임프레임 ${tf} 데이터 수집 실패:`, e);
                }
            });

            return data;
        }

        // 카드에서 존 정보 추출
        getZoneFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/구역:\s*<b>(\w+)<\/b>/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 강도 정보 추출
        getStrengthFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/강도:\s*(\d+)/);
                return match ? parseInt(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 가격 정보 추출
        getPriceFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/가격:\s*₩([\d,]+)/);
                return match ? parseInt(match[1].replace(/,/g, '')) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 모드 정보 추출
        getModeFromCard(card) {
            try {
                const text = card.textContent;
                if (text.includes('모의전')) return 'paper';
                if (text.includes('실전')) return 'real';
                return null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 다음 액션 추출
        getNextActionFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/액션:\s*<b>(\w+)<\/b>/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 예상 수익률 추출
        getExpectedReturnFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/예상수익:\s*<b>([\d.]+)%/);
                return match ? parseFloat(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 가격 변화 추출
        getPriceChangeFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/\(([-\d.]+)%\)/);
                return match ? parseFloat(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매수전 예상 수익률 추출
        getBuyExpectedReturnFromCard(card) {
            try {
                // 실제 매수가 발생했을 때만 매수 전 예상 수익률 반환
                const hasBuyAction = window.lastBuyAction === true;
                if (!hasBuyAction) {
                    return 0;
                }
                
                const text = card.textContent;
                const match = text.match(/매수전 예상:\s*<b>([\d.]+)%/);
                return match ? parseFloat(match[1]) : 0;
            } catch (e) {
                return 0;
            }
        }

        // 카드에서 매도전 예상 수익률 추출
        getSellExpectedReturnFromCard(card) {
            try {
                // 실제 매도가 발생했을 때만 매도 전 예상 수익률 반환
                const hasSellAction = window.lastSellAction === true;
                if (!hasSellAction) {
                    return 0;
                }
                
                const text = card.textContent;
                const match = text.match(/매도전 예상:\s*<b>([\d.]+)%/);
                return match ? parseFloat(match[1]) : 0;
            } catch (e) {
                return 0;
            }
        }

        // 카드에서 학습 요약 추출
        getLearningSummaryFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/학습요약:\s*O:(\d+)\s*\/\s*B:(\d+)/);
                return match ? { orange: parseInt(match[1]), blue: parseInt(match[2]) } : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 Majority 추출
        getMajorityFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/Majority:\s*<b>(\w+)<\/b>/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 Orange Total 추출
        getOrangeTotalFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/Orange Total:\s*<b>(\d+)<\/b>/);
                return match ? parseInt(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 Blue Total 추출
        getBlueTotalFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/Blue Total:\s*<b>(\d+)<\/b>/);
                return match ? parseInt(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매수 대기 시간 추출
        getBuyWaitTimeFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/매수 대기\s*\((\d+:\d+)\)/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매도 대기 시간 추출
        getSellWaitTimeFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/매도 대기\s*\((\d+:\d+)\)/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매수 경과 시간 추출
        getBuyElapsedFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/매수후\s*(\d+:\d+)/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매도 경과 시간 추출
        getSellElapsedFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/매도후\s*(\d+:\d+)/);
                return match ? match[1] : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 N/B MAX 코인 추출
        getNBCoinsFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/N\/B MAX COIN:\s*(\d+)/);
                return match ? parseInt(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 N/B 미네랄 추출
        getNBMineralsFromCard(card) {
            try {
                const text = card.textContent;
                const match = text.match(/N\/B 미네랄:\s*([\d.]+)%/);
                return match ? parseFloat(match[1]) : null;
            } catch (e) {
                return null;
            }
        }

        // 카드에서 매도 가능 여부 추출
        canSellFromCard(card) {
            try {
                const text = card.textContent;
                return text.includes('매도 가능');
            } catch (e) {
                return false;
            }
        }

        // 활성 거래 여부 확인
        hasActiveTrade(card) {
            try {
                return card.querySelector('.trade-indicator') !== null;
            } catch (e) {
                return false;
            }
        }

        // 시장 데이터 수집
        getMarketData() {
            try {
                return {
                    currentPrice: window.currentPrice || null,
                    priceChange: window.priceChange || null,
                    volume: window.currentVolume || null,
                    marketCap: window.marketCap || null
                };
            } catch (e) {
                return {};
            }
        }

        // 거래 데이터 수집
        getTradeData() {
            try {
                return {
                    activeTrades: window.activeTrades || [],
                    tradeHistory: window.tradeHistory || [],
                    totalPnL: window.totalPnL || 0,
                    todayPnL: window.todayPnL || 0
                };
            } catch (e) {
                return {};
            }
        }

        // 시스템 데이터 수집
        getSystemData() {
            try {
                return {
                    systemStatus: window.systemStatus || 'unknown',
                    lastUpdate: window.lastUpdate || null,
                    connectionStatus: navigator.onLine,
                    memoryUsage: performance.memory ? {
                        used: performance.memory.usedJSHeapSize,
                        total: performance.memory.totalJSHeapSize,
                        limit: performance.memory.jsHeapSizeLimit
                    } : null
                };
            } catch (e) {
                return {};
            }
        }

        // UI 상태 수집
        getUIState() {
            try {
                return {
                    selectedTab: this.getSelectedTab(),
                    visiblePanels: this.getVisiblePanels(),
                    windowSize: {
                        width: window.innerWidth,
                        height: window.innerHeight
                    },
                    scrollPosition: {
                        x: window.scrollX,
                        y: window.scrollY
                    }
                };
            } catch (e) {
                return {};
            }
        }

        // 선택된 탭 가져오기
        getSelectedTab() {
            try {
                const activeTab = document.querySelector('.nav-link.active');
                return activeTab ? activeTab.textContent.trim() : null;
            } catch (e) {
                return null;
            }
        }

        // 보이는 패널들 가져오기
        getVisiblePanels() {
            try {
                const panels = document.querySelectorAll('.panel, .card, .container');
                const visible = [];
                panels.forEach(panel => {
                    if (panel.offsetParent !== null) {
                        visible.push(panel.id || panel.className);
                    }
                });
                return visible;
            } catch (e) {
                return [];
            }
        }

        // 로그 저장
        async saveLog(data, type = 'comprehensive') {
            if (!this.isEnabled) return;

            try {
                const logData = {
                    type: type,
                    timestamp: Date.now(),
                    data: data
                };

                // 스냅샷 로그로 저장 (더 상세한 데이터)
                if (type === 'comprehensive') {
                    await this.saveSnapshot(logData);
                } else {
                    await this.saveStatusLog(logData);
                }

                console.log(`✅ ${type} 로그 저장 완료`);
            } catch (error) {
                console.error(`❌ 로그 저장 실패:`, error);
                // 실패한 로그를 대기열에 추가
                this.pendingLogs.push({ data, type, timestamp: Date.now() });
            }
        }

        // 스냅샷 로그 저장
        async saveSnapshot(logData) {
            const response = await fetch(this.snapshotEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(logData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
        }

        // 상태 로그 저장
        async saveStatusLog(logData) {
            const response = await fetch(this.logEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tf: logData.data.currentTimeframe,
                    text: `포괄 로그: ${logData.type}`,
                    ts: logData.timestamp,
                    mode: 'comprehensive',
                    type: logData.type,
                    data: logData.data
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
        }

        // 주기적 로깅 시작
        startPeriodicLogging() {
            setInterval(() => {
                if (this.isEnabled && Date.now() - this.lastLogTime >= this.logInterval) {
                    this.logAllData();
                    this.lastLogTime = Date.now();
                }
            }, 1000); // 1초마다 체크
        }

        // 모든 데이터 로깅
        logAllData() {
            const allData = this.collectAllData();
            this.saveLog(allData, 'comprehensive');
        }

        // 이벤트 리스너 설정
        setupEventListeners() {
            // 타임프레임 변경 이벤트
            document.addEventListener('timeframeChanged', (e) => {
                this.saveLog({
                    event: 'timeframeChanged',
                    timeframe: e.detail.timeframe,
                    timestamp: Date.now()
                }, 'event');
            });

            // 거래 이벤트
            document.addEventListener('tradeExecuted', (e) => {
                this.saveLog({
                    event: 'tradeExecuted',
                    trade: e.detail,
                    timestamp: Date.now()
                }, 'event');
            });

            // 시장 데이터 업데이트
            document.addEventListener('marketDataUpdated', (e) => {
                this.saveLog({
                    event: 'marketDataUpdated',
                    marketData: e.detail,
                    timestamp: Date.now()
                }, 'event');
            });

            // 페이지 언로드 시 마지막 로그 저장
            window.addEventListener('beforeunload', () => {
                this.logAllData();
            });
        }

        // 로거 활성화/비활성화
        setEnabled(enabled) {
            this.isEnabled = enabled;
            console.log(`🔍 좌측 패널 로거 ${enabled ? '활성화' : '비활성화'}`);
        }

        // 로그 간격 설정
        setLogInterval(interval) {
            this.logInterval = interval;
            console.log(`⏱️ 로그 간격 설정: ${interval}ms`);
        }

        // 대기 중인 로그 재시도
        async retryPendingLogs() {
            if (this.pendingLogs.length === 0) return;

            console.log(`🔄 대기 중인 로그 ${this.pendingLogs.length}개 재시도`);
            
            for (const log of this.pendingLogs) {
                try {
                    await this.saveLog(log.data, log.type);
                } catch (error) {
                    console.error('재시도 실패:', error);
                }
            }
            
            this.pendingLogs = [];
        }

        // 로거 상태 가져오기
        getStatus() {
            return {
                enabled: this.isEnabled,
                logInterval: this.logInterval,
                pendingLogs: this.pendingLogs.length,
                lastLogTime: this.lastLogTime,
                endpoints: {
                    log: this.logEndpoint,
                    snapshot: this.snapshotEndpoint
                }
            };
        }
    }

    // 전역 인스턴스 생성
    window.leftPanelComprehensiveLogger = new LeftPanelComprehensiveLogger();

    // 전역 함수들
    window.enableLeftPanelLogging = (enabled) => {
        window.leftPanelComprehensiveLogger.setEnabled(enabled);
    };

    window.setLeftPanelLogInterval = (interval) => {
        window.leftPanelComprehensiveLogger.setLogInterval(interval);
    };

    window.getLeftPanelLogStatus = () => {
        return window.leftPanelComprehensiveLogger.getStatus();
    };

    window.retryLeftPanelLogs = () => {
        return window.leftPanelComprehensiveLogger.retryPendingLogs();
    };

    console.log('✅ 좌측 패널 포괄 로거 로드 완료');
})();
