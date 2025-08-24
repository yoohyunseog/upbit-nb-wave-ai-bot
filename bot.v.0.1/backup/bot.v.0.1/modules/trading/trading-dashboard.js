// Trading Dashboard Module
class TradingDashboard {
    constructor() {
        this.isInitialized = false;
        this.updateInterval = null;
        this.priceUpdateTimer = null;
    }

    // Trading Dashboard HTML 로드
    loadTradingDashboard() {
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        
        return `
            <div class="trading-dashboard-container">
                <div class="dashboard-header">
                    <h2>Trading Dashboard</h2>
                    <div class="trading-info">
                        <div class="current-price">
                            <span class="label">Current Price:</span>
                            <span id="trading-current-price">₩0</span>
                            <span id="trading-price-change" class="price-change">0.00%</span>
                        </div>
                        <div class="current-zone">
                            <span class="label">Current Zone:</span>
                            <span id="trading-current-zone">Neutral</span>
                            <span id="trading-zone-strength" class="zone-strength">0%</span>
                        </div>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span id="current-timeframe" class="current-timeframe">Current: 1m</span>
                    </div>
                    <canvas id="trading-chart" width="100%" height="400"></canvas>
                </div>

                <div class="wave-count-info">
                    <div class="wave-count-item">
                        <span class="wave-label">Wave Blue:</span>
                        <span id="wave-blue-count">0</span>
                        <span id="wave-blue-last" class="last-indicator">Last: -</span>
                    </div>
                    <div class="wave-count-item">
                        <span class="wave-label">Wave Orange:</span>
                        <span id="wave-orange-count">0</span>
                        <span id="wave-orange-last" class="last-indicator">Last: -</span>
                    </div>
                </div>

                <div class="active-signals">
                    <h3><i class="fas fa-bell"></i> Active Signals</h3>
                    <div class="signals-controls">
                        <button class="btn-signal-refresh" onclick="window.activeSignalsManager.refreshSignals()">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="btn-signal-clear" onclick="window.activeSignalsManager.clearSignals()">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                    <div id="active-signals-container" class="signals-list">
                        <div class="no-signals-message">
                            <i class="fas fa-info-circle"></i>
                            <span>No active signals</span>
                        </div>
                    </div>
                    <div class="signals-stats">
                        <div class="stat-item">
                            <span class="stat-label">Total:</span>
                            <span id="signals-total-count" class="stat-value">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Buy:</span>
                            <span id="signals-buy-count" class="stat-value buy-count">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Sell:</span>
                            <span id="signals-sell-count" class="stat-value sell-count">0</span>
                        </div>
                    </div>
                </div>

                <div class="timeframe-zones">
                    <h3>Timeframe Zones</h3>
                    <div id="timeframe-zones-container" class="timeframe-zones-grid">
                        <!-- Timeframe zones will be populated here -->
                    </div>
                </div>

                <div class="nb-wave-container">
                    <h3>N/B Wave Map</h3>
                    <div id="nb-wave-chart-container">
                        <canvas id="nb-wave-chart" width="100%" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    // Trading Dashboard 초기화
    async initializeTradingCharts() {
        if (this.isInitialized) {
            console.log('🔄 Trading Dashboard already initialized');
            return;
        }

        console.log('🚀 Initializing Trading Dashboard...');
        
        try {
            const selectedCoin = window.selectedKrwCoin || 'BTC';
            
            // API 데이터 가져오기
            const [priceResponse, nbResponse] = await Promise.all([
                fetch(`/api/trading-data?coin=${selectedCoin}`),
                fetch(`http://localhost:5057/api/nb-wave?timeframe=minute1&bars=300&coin=${selectedCoin}`)
            ]);

            const priceData = await priceResponse.json();
            const nbData = await nbResponse.json();

            if (priceData.status === 'success' && priceData.data && Array.isArray(priceData.data) && priceData.data.length > 0) {
                this.updateTradingCurrentPrice(priceData.data);
                this.drawPriceChartFromData(priceData.data, nbData);
            } else {
                console.warn('⚠️ Invalid price data:', priceData);
            }

            if (nbData.status === 'success' && nbData.zones && nbData.zones.length > 0) {
                this.updateTradingCurrentZone(nbData);
                this.updateTimeframeZones(nbData);
                
                // NB Wave Panel 초기화
                if (window.nbWavePanel) {
                    window.nbWavePanel.drawChart(nbData);
                }
            }

            // 실시간 업데이트 시작
            this.startTradingPriceUpdate();
            
            // Active Signals 초기화
            if (window.activeSignalsManager) {
                window.activeSignalsManager.loadActiveSignalsToTradingDashboard();
            }
            
            this.isInitialized = true;
            console.log('✅ Trading Dashboard initialized successfully');

        } catch (error) {
            console.error('❌ Failed to initialize trading charts:', error);
        }
    }

    // 가격 차트 그리기
    drawPriceChartFromData(data, nbData = null) {
        const canvas = document.getElementById('trading-chart');
        if (!canvas) {
            console.warn('⚠️ Trading chart canvas not found');
            return;
        }

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // 캔버스 초기화
        ctx.clearRect(0, 0, width, height);

        if (!data || data.length === 0) {
            console.warn('⚠️ No price data available');
            return;
        }

        // 차트 영역 설정 (80% 너비)
        const chartWidth = width * 0.8;
        const chartX = (width - chartWidth) / 2;
        const chartHeight = height - 40;
        const chartY = 20;

        // 가격 범위 계산
        const prices = data.map(d => typeof d === 'number' ? d : d.close);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;

        // 그리드 그리기
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const y = chartY + (chartHeight / 10) * i;
            ctx.beginPath();
            ctx.moveTo(chartX, y);
            ctx.lineTo(chartX + chartWidth, y);
            ctx.stroke();
        }

        // 가격 라인 그리기
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.beginPath();

        data.forEach((point, index) => {
            const price = typeof point === 'number' ? point : point.close;
            const x = chartX + (chartWidth / (data.length - 1)) * index;
            const y = chartY + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // 점 그리기 및 구역 분석
        this.drawWaveCountInfo(data, nbData);

        // 데이터 저장
        window.sharedMainChartData = {
            prices: data,
            zones: nbData ? nbData.zones : [],
            currentPrice: data[data.length - 1],
            waveAnalysis: this.analyzeWaveCounts(data, nbData)
        };
    }

    // Wave Count 정보 그리기
    drawWaveCountInfo(data, nbData) {
        const canvas = document.getElementById('trading-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const chartWidth = width * 0.8;
        const chartX = (width - chartWidth) / 2;
        const chartHeight = height - 40;
        const chartY = 20;

        const prices = data.map(d => typeof d === 'number' ? d : d.close);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;

        // Wave 분석
        const waveAnalysis = this.analyzeWaveCounts(data, nbData);

        // 점 그리기
        data.forEach((point, index) => {
            const price = typeof point === 'number' ? point : point.close;
            const x = chartX + (chartWidth / (data.length - 1)) * index;
            const y = chartY + chartHeight - ((price - minPrice) / priceRange) * chartHeight;

            // 점 색상 결정
            let color = '#666'; // 기본 회색
            if (index < waveAnalysis.zones.length) {
                const zone = waveAnalysis.zones[index];
                if (zone === 'BLUE') color = '#0066ff';
                else if (zone === 'ORANGE') color = '#ff6600';
            }

            // 점 그리기
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Wave Count 정보 업데이트
        this.updateWaveCountDisplay(waveAnalysis);
    }

    // Wave Count 분석
    analyzeWaveCounts(data, nbData) {
        const zonesToAnalyze = data.slice(-20); // 마지막 20개 점만 분석
        let blueCount = 0;
        let orangeCount = 0;
        let lastZone = null;

        // 각 점의 구역 결정 (간단한 로직)
        const zones = zonesToAnalyze.map((point, index) => {
            if (index === 0) return 'NEUTRAL';
            
            const currentPrice = typeof point === 'number' ? point : point.close;
            const prevPrice = typeof zonesToAnalyze[index - 1] === 'number' ? zonesToAnalyze[index - 1] : zonesToAnalyze[index - 1].close;
            
            const change = currentPrice - prevPrice;
            let zone = 'NEUTRAL';
            
            if (change > 0) {
                zone = 'BLUE';
                blueCount++;
            } else if (change < 0) {
                zone = 'ORANGE';
                orangeCount++;
            }
            
            if (index === zonesToAnalyze.length - 1) {
                lastZone = { zone, change };
            }
            
            return zone;
        });

        // N/B Wave 데이터에서 강도 계산
        let currentZoneStrength = 0;
        if (nbData && nbData.summary) {
            const totalZones = nbData.summary.blue + nbData.summary.orange;
            if (totalZones > 0) {
                currentZoneStrength = nbData.summary.blue / totalZones;
            }
        }

        // 마지막 구역을 전역 변수에 저장
        if (lastZone) {
            window.lastZoneFromChart = lastZone;
        }

        return {
            blueCount,
            orangeCount,
            zones,
            lastZone,
            currentZoneStrength
        };
    }

    // Wave Count 표시 업데이트
    updateWaveCountDisplay(waveAnalysis) {
        const blueCountElement = document.getElementById('wave-blue-count');
        const orangeCountElement = document.getElementById('wave-orange-count');
        const blueLastElement = document.getElementById('wave-blue-last');
        const orangeLastElement = document.getElementById('wave-orange-last');

        if (blueCountElement) blueCountElement.textContent = waveAnalysis.blueCount;
        if (orangeCountElement) orangeCountElement.textContent = waveAnalysis.orangeCount;

        if (blueLastElement) {
            blueLastElement.textContent = waveAnalysis.lastZone && waveAnalysis.lastZone.zone === 'BLUE' ? 'Last: Blue' : '';
        }
        if (orangeLastElement) {
            orangeLastElement.textContent = waveAnalysis.lastZone && waveAnalysis.lastZone.zone === 'ORANGE' ? 'Last: Orange' : '';
        }
    }

    // 현재 가격 업데이트
    updateTradingCurrentPrice(priceData) {
        if (!priceData || priceData.length === 0) {
            console.warn('⚠️ No price data for update');
            return;
        }

        const lastPriceData = priceData[priceData.length - 1];
        let currentPrice, previousPrice;

        if (typeof lastPriceData === 'number') {
            currentPrice = lastPriceData;
            previousPrice = priceData[priceData.length - 2] || currentPrice;
        } else {
            currentPrice = lastPriceData.close;
            previousPrice = priceData[priceData.length - 2]?.close || currentPrice;
        }

        const priceChange = currentPrice - previousPrice;
        const priceChangePercent = (priceChange / previousPrice) * 100;

        // DOM 업데이트
        const currentPriceElement = document.getElementById('trading-current-price');
        const priceChangeElement = document.getElementById('trading-price-change');

        if (currentPriceElement) {
            currentPriceElement.textContent = `₩${currentPrice.toLocaleString()}`;
        }

        if (priceChangeElement) {
            priceChangeElement.textContent = `${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(2)}%`;
            priceChangeElement.className = `price-change ${priceChangePercent >= 0 ? 'positive' : 'negative'}`;
        }

        // 데이터 저장
        window.sharedTradingDashboardData = {
            ...window.sharedTradingDashboardData,
            currentPrice: currentPrice,
            priceChange: priceChange,
            priceChangePercent: priceChangePercent
        };
    }

    // 현재 구역 업데이트
    updateTradingCurrentZone(nbData) {
        if (!nbData || !nbData.zones || nbData.zones.length === 0) {
            console.warn('⚠️ No NB wave data for zone update');
            return;
        }

        // zone-strength-manager 모듈을 사용하여 구역 업데이트
        if (window.zoneStrengthManager) {
            window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbData);
        } else {
            console.warn('⚠️ Zone Strength Manager not loaded yet');
        }

        // 데이터 저장 (zone-strength-manager에서 이미 처리됨)
        console.log('🔄 Trading current zone updated via zone-strength-manager');
    }

    // Timeframe Zones 업데이트
    updateTimeframeZones(nbData) {
        if (!nbData || !nbData.zones || nbData.zones.length === 0) {
            console.warn('⚠️ No NB wave data for timeframe zones');
            return;
        }

        const zones = nbData.zones;
        const totalZones = zones.length;

        // 전체 구역 통계
        const zoneCounts = { BLUE: 0, ORANGE: 0, NEUTRAL: 0 };
        const zoneStrengths = { BLUE: 0, ORANGE: 0, NEUTRAL: 0 };

        zones.forEach(zone => {
            const zoneType = zone.zone || 'NEUTRAL';
            zoneCounts[zoneType]++;
            zoneStrengths[zoneType] += zone.strength || 0;
        });

        // 평균 강도 계산
        Object.keys(zoneStrengths).forEach(zoneType => {
            if (zoneCounts[zoneType] > 0) {
                zoneStrengths[zoneType] = zoneStrengths[zoneType] / zoneCounts[zoneType];
            }
        });

        // 과반수 구역 결정
        const majority = Math.ceil(totalZones / 2);
        let currentZone = 'NEUTRAL';
        let currentZoneCount = 0;

        if (zoneCounts.BLUE >= majority) {
            currentZone = 'BLUE';
            currentZoneCount = zoneCounts.BLUE;
        } else if (zoneCounts.ORANGE >= majority) {
            currentZone = 'ORANGE';
            currentZoneCount = zoneCounts.ORANGE;
        }

        // 각 분봉별 구역 계산 (API 값과 동기화)
        const timeframeData = [
            { name: 'minute1', zones: zones.slice(Math.max(0, zones.length - 60)) },
            { name: 'minute3', zones: zones.slice(Math.max(0, zones.length - 180)) },
            { name: 'minute5', zones: zones.slice(Math.max(0, zones.length - 300)) },
            { name: 'minute10', zones: zones.slice(Math.max(0, zones.length - 300)) },
            { name: 'minute15', zones: zones.slice(Math.max(0, zones.length - 300)) },
            { name: 'minute30', zones: zones.slice(Math.max(0, zones.length - 300)) },
            { name: 'minute60', zones: zones.slice(Math.max(0, zones.length - 300)) },
            { name: 'day', zones }
        ];

        // 각 분봉별로 구역 계산
        window.timeframeResults = timeframeData.map(timeframe => {
            const zoneCounts = {
                BLUE: 0,
                ORANGE: 0,
                NEUTRAL: 0
            };

            timeframe.zones.forEach(zone => {
                const zoneType = zone.zone || 'NEUTRAL';
                zoneCounts[zoneType]++;
            });

            const totalZones = timeframe.zones.length;
            const majority = Math.ceil(totalZones / 2);
            let dominantZone = 'NEUTRAL';

            if (zoneCounts.BLUE >= majority) {
                dominantZone = 'BLUE';
            } else if (zoneCounts.ORANGE >= majority) {
                dominantZone = 'ORANGE';
            }

            return {
                name: timeframe.name,
                dominantZone: dominantZone,
                zoneCounts: zoneCounts,
                totalZones: totalZones
            };
        });

        // 모든 분봉 카드 생성
        const individualZonesHtml = window.timeframeResults.map(result => {
            const isBlue = result.dominantZone === 'BLUE';
            const isOrange = result.dominantZone === 'ORANGE';

            return `
                <div class="timeframe-status-card ${isBlue ? 'blue-zone' : isOrange ? 'orange-zone' : 'neutral-zone'}" 
                     id="timeframe-card-${result.name}" 
                     data-timeframe="${result.name}"
                     onclick="window.tradingDashboard.selectTimeframe('${result.name}', '${result.dominantZone}', ${result.zoneCounts[result.dominantZone]}, ${result.totalZones})">
                    <div class="timeframe-icon">⏱️</div>
                    <div class="timeframe-name">${this.convertTimeframeToDisplay(result.name)}</div>
                    <div class="timeframe-status">Current Zone: ${result.dominantZone}</div>
                    <div class="timeframe-count">${result.zoneCounts[result.dominantZone]}/${result.totalZones}</div>
                </div>
            `;
        }).join('');

        // HTML 생성
        const zonesHtml = `
            <div class="timeframe-zone-summary">
                <div class="current-zone-display ${currentZone.toLowerCase()}-zone" id="current-zone-display" onclick="window.tradingDashboard.selectCurrentZone()">
                    <h4>Current Zone: ${currentZone} (Click to Select)</h4>
                    <div class="zone-stats">
                        <span class="zone-count">${currentZoneCount}/${totalZones} (${((currentZoneCount/totalZones)*100).toFixed(1)}%)</span>
                        <span class="zone-strength">Strength: ${(zoneStrengths[currentZone] * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            
            <div class="timeframe-zone-breakdown">
                <div class="zone-card blue-zone">
                    <div class="zone-header">
                        <span class="zone-name">BLUE ZONE</span>
                        <span class="zone-count">${zoneCounts.BLUE}</span>
                    </div>
                    <div class="zone-details">
                        <span class="zone-percentage">${((zoneCounts.BLUE/totalZones)*100).toFixed(1)}%</span>
                        <span class="zone-strength">${(zoneStrengths.BLUE * 100).toFixed(1)}%</span>
                    </div>
                </div>
                
                <div class="zone-card orange-zone">
                    <div class="zone-header">
                        <span class="zone-name">ORANGE ZONE</span>
                        <span class="zone-count">${zoneCounts.ORANGE}</span>
                    </div>
                    <div class="zone-details">
                        <span class="zone-percentage">${((zoneCounts.ORANGE/totalZones)*100).toFixed(1)}%</span>
                        <span class="zone-strength">${(zoneStrengths.ORANGE * 100).toFixed(1)}%</span>
                    </div>
                </div>
                
                <div class="zone-card neutral-zone">
                    <div class="zone-header">
                        <span class="zone-name">NEUTRAL ZONE</span>
                        <span class="zone-count">${zoneCounts.NEUTRAL}</span>
                    </div>
                    <div class="zone-details">
                        <span class="zone-percentage">${((zoneCounts.NEUTRAL/totalZones)*100).toFixed(1)}%</span>
                        <span class="zone-strength">${(zoneStrengths.NEUTRAL * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            
            <div class="individual-zones-container">
                <div class="timeframe-status-grid">
                    ${individualZonesHtml}
                </div>
            </div>
        `;

        // DOM 업데이트
        const container = document.getElementById('timeframe-zones-container');
        if (container) {
            container.innerHTML = zonesHtml;
        }

        // 전역 변수 저장
        window.currentMajorityZone = currentZone;
        window.currentMajorityZoneCount = currentZoneCount;
        window.currentMajorityZoneStrength = zoneStrengths[currentZone];

        // 자동 순회 시작
        setTimeout(() => {
            this.startTimeframeAutoRotation();
            
            // 초기 선택 (1분봉)
            const defaultTimeframe = 'minute1';
            const defaultResult = window.timeframeResults.find(result => result.name === defaultTimeframe);
            if (defaultResult) {
                this.selectTimeframe(
                    defaultResult.name, 
                    defaultResult.dominantZone, 
                    defaultResult.zoneCounts[defaultResult.dominantZone], 
                    defaultResult.totalZones
                );
            }
        }, 100);

        console.log('💾 Timeframe zones updated:', {
            currentZone,
            currentZoneCount,
            totalZones,
            zoneCounts,
            zoneStrengths
        });
    }

    // 분봉 선택 함수
    selectTimeframe(timeframe, dominantZone, zoneCount, totalZones, isAutoRotation = false) {
        console.log(`🎯 Selected timeframe: ${timeframe}, Zone: ${dominantZone}, Count: ${zoneCount}/${totalZones}, Auto: ${isAutoRotation}`);
        
        // 모든 카드에서 선택 상태 제거
        const allCards = document.querySelectorAll('.timeframe-status-card');
        allCards.forEach(card => {
            card.classList.remove('selected');
        });
        
        // 선택된 카드에 선택 상태 추가
        const selectedCard = document.getElementById(`timeframe-card-${timeframe}`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }
        
        // Current Zone 표시 업데이트
        this.updateCurrentZoneDisplay(timeframe, dominantZone, zoneCount, totalZones);
        
        // 차트의 분봉 표시 업데이트
        const timeframeDisplay = document.getElementById('current-timeframe');
        console.log('🔍 Looking for current-timeframe element:', timeframeDisplay);
        if (timeframeDisplay) {
            const displayTimeframe = this.convertTimeframeToDisplay(timeframe);
            timeframeDisplay.textContent = `Current: ${displayTimeframe}`;
            console.log(`✅ Updated current-timeframe to: Current: ${displayTimeframe}`);
        } else {
            console.warn('⚠️ current-timeframe element not found!');
        }
        
        // 전역 변수에 선택된 분봉 저장
        window.selectedTimeframe = timeframe;
        window.selectedTimeframeZone = dominantZone;
        window.selectedTimeframeCount = zoneCount;
        window.selectedTimeframeTotal = totalZones;
        
        // 수동 선택 시 자동 순회 일시 정지
        if (!isAutoRotation) {
            this.stopTimeframeAutoRotation();
            // 5초 후 자동 순회 재시작
            setTimeout(() => {
                this.startTimeframeAutoRotation();
            }, 5000);
        }
        
        console.log(`✅ Timeframe ${timeframe} selected and connected with zone data`);
    }

    // Current Zone 선택 함수
    selectCurrentZone() {
        console.log(`🎯 Current Zone clicked`);
        
        // 현재 선택된 분봉이 있으면 해당 분봉의 구역으로 설정
        if (window.selectedTimeframe && window.selectedTimeframeZone) {
            this.updateCurrentZoneDisplay(
                window.selectedTimeframe, 
                window.selectedTimeframeZone, 
                window.selectedTimeframeCount, 
                window.selectedTimeframeTotal
            );
            console.log(`✅ Current Zone synchronized with ${window.selectedTimeframe}`);
        } else {
            // 선택된 분봉이 없으면 안내 메시지 표시
            const currentZoneDisplay = document.getElementById('current-zone-display');
            if (currentZoneDisplay) {
                const currentZone = currentZoneDisplay.querySelector('h4');
                if (currentZone) {
                    currentZone.textContent = `Current Zone: ${currentZone} (Select a Timeframe)`;
                }
            }
            console.log(`ℹ️ No timeframe selected - please select a timeframe card`);
        }
    }

    // Current Zone 표시 업데이트 함수
    updateCurrentZoneDisplay(timeframe, zone, count, total) {
        const currentZoneDisplay = document.getElementById('current-zone-display');
        if (currentZoneDisplay) {
            const currentZone = currentZoneDisplay.querySelector('h4');
            const zoneStats = currentZoneDisplay.querySelector('.zone-stats');
            
            if (currentZone) {
                const displayTimeframe = this.convertTimeframeToDisplay(timeframe);
                currentZone.textContent = `Current Zone: ${zone} (${displayTimeframe})`;
            }
            
            if (zoneStats) {
                const zoneCount = zoneStats.querySelector('.zone-count');
                const zoneStrength = zoneStats.querySelector('.zone-strength');
                
                if (zoneCount) {
                    zoneCount.textContent = `${count}/${total} (${((count/total)*100).toFixed(1)}%)`;
                }
                
                if (zoneStrength) {
                    zoneStrength.textContent = `Strength: ${((count/total)*100).toFixed(1)}%`;
                }
            }
        }
    }

    // Timeframe 변환 함수
    convertTimeframeToDisplay(timeframe) {
        const conversions = {
            'minute1': '1m',
            'minute3': '3m', 
            'minute5': '5m',
            'minute10': '10m',
            'minute15': '15m',
            'minute30': '30m',
            'minute60': '1h',
            'minute240': '4h',
            'day': '1D',
            'week': '1W',
            'month': '1M'
        };
        return conversions[timeframe] || timeframe;
    }

    // 자동 순회 분봉 목록 (전역 변수) - API 값과 동기화
    initTimeframeRotation() {
        window.timeframeRotationList = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'day'];
        window.currentRotationIndex = 0; // 1분봉부터 시작
    }

    // 자동 순회 함수
    startTimeframeAutoRotation() {
        // 기존 타이머가 있으면 제거
        if (window.timeframeRotationTimer) {
            clearInterval(window.timeframeRotationTimer);
        }
        
        // 1분봉부터 시작하도록 인덱스 초기화
        window.currentRotationIndex = 0;
        
        // 3초마다 분봉 순회
        window.timeframeRotationTimer = setInterval(() => {
            const timeframeName = window.timeframeRotationList[window.currentRotationIndex];
            const timeframeResult = window.timeframeResults.find(result => result.name === timeframeName);
            
            if (timeframeResult) {
                this.selectTimeframe(
                    timeframeResult.name,
                    timeframeResult.dominantZone,
                    timeframeResult.zoneCounts[timeframeResult.dominantZone],
                    timeframeResult.totalZones,
                    true // 자동 순회 플래그
                );
            }
            
            window.currentRotationIndex = (window.currentRotationIndex + 1) % window.timeframeRotationList.length;
        }, 3000);
        
        console.log('🔄 Timeframe auto-rotation started from 1m');
    }

    // 자동 순회 정지 함수
    stopTimeframeAutoRotation() {
        if (window.timeframeRotationTimer) {
            clearInterval(window.timeframeRotationTimer);
            window.timeframeRotationTimer = null;
            console.log('⏹️ Timeframe auto-rotation stopped');
        }
    }

    // 실시간 가격 업데이트 시작
    startTradingPriceUpdate() {
        // 30초마다 현재가 및 구역 업데이트
        this.priceUpdateTimer = setInterval(async () => {
            const selectedCoin = window.selectedKrwCoin || 'BTC';
            try {
                // 가격 데이터만 업데이트 (N/B Wave는 이미 전역에 저장됨)
                const priceResponse = await fetch(`/api/trading-data?coin=${selectedCoin}`);
                const priceData = await priceResponse.json();
                
                // 전역 저장된 N/B Wave 데이터 사용
                const nbData = window.sharedNbWaveData;
                
                if (priceData.status === 'success' && priceData.data && Array.isArray(priceData.data) && priceData.data.length > 0) {
                    this.updateTradingCurrentPrice(priceData.data);
                } else {
                    console.warn('⚠️ Invalid price data in background update:', priceData);
                }
                
                if (nbData.status === 'success' && nbData.zones && nbData.zones.length > 0) {
                    // 마지막 점의 가격 정보 추가
                    if (priceData.status === 'success' && priceData.data && priceData.data.length > 0) {
                        const lastPrice = priceData.data[priceData.data.length - 1].close;
                        nbData.last_point_price = lastPrice;
                    }
                    this.updateTradingCurrentZone(nbData);
                }
            } catch (error) {
                console.error('Failed to update trading data:', error);
            }
        }, 30000); // 30초마다
    }

    // 정리 함수
    destroy() {
        if (this.priceUpdateTimer) {
            clearInterval(this.priceUpdateTimer);
        }
        if (window.timeframeRotationTimer) {
            clearInterval(window.timeframeRotationTimer);
        }
        this.isInitialized = false;
        console.log('🧹 Trading Dashboard destroyed');
    }
}

// 전역 인스턴스 생성
window.tradingDashboard = new TradingDashboard();
