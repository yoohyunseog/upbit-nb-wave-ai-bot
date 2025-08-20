// ===== 8BIT Trading System - Starcraft Style JavaScript =====

// 전역 변수
let currentModule = null;
let gameState = {
    minerals: 1000,
    gas: 500,
    supply: 8,
    maxSupply: 10
};

// 사운드 재생 함수 (새로운 오디오 시스템 사용)
function playSound(soundType) {
    try {
        if (window.audioSystem && window.audioSystem.isEnabled()) {
            window.audioSystem.play(soundType);
        }
    } catch (e) {
        console.error('Sound play error:', e);
    }
}

// 사운드 초기화 함수
function initializeSound() {
    try {
        console.log('Sound system initialized (using external audio system)');
    } catch (e) {
        console.error('Sound initialization failed:', e);
    }
}

function toggleSound() {
    if (window.audioSystem) {
        window.audioSystem.toggle();
    }
}

// 모든 사운드 테스트 함수
function testAllSounds() {
    if (window.audioSystem) {
        window.audioSystem.testAll();
    }
}

// 연속 사운드 재생 함수
function playSoundSequence() {
    if (window.audioSystem) {
        window.audioSystem.playSoundSequence();
    }
}

// 상태 메시지 업데이트 (타이핑 효과)
function updateStatusMessage(message) {
    const statusElement = document.getElementById('status-message');
    if (statusElement) {
        // 기존 텍스트 클리어
        statusElement.textContent = '';
        
        // 타이핑 효과로 메시지 표시
        let index = 0;
        const typeWriter = () => {
            if (index < message.length) {
                statusElement.textContent += message.charAt(index);
                // 타이핑 효과음 재생 (공백 제외) - Startup 사운드 대신 click 사운드 사용
                if (message.charAt(index) !== ' ') {
                    playSound('click');
                }
                index++;
                setTimeout(typeWriter, 50); // 타이핑 속도
            } else {
                // 타이핑 완료 후 메시지 유지 (Ready 메시지 제거)
            }
        };
        
        // Ready 메시지 제거 - 타이핑 완료 후 빈 상태로 유지
        
        typeWriter();
        statusElement.classList.add('glow');
        setTimeout(() => statusElement.classList.remove('glow'), 2000);
    }
}

// 게임 상태 업데이트
async function updateGameState() {
    try {
        const response = await fetch('/api/game-state');
        const data = await response.json();
        gameState = data;
        
        // UI 업데이트
        document.getElementById('mineral-count').textContent = gameState.minerals;
        document.getElementById('gas-count').textContent = gameState.gas;
        document.getElementById('supply-count').textContent = `${gameState.supply}/${gameState.maxSupply}`;
        
    } catch (error) {
        console.error('Failed to update game state:', error);
    }
}

// 시스템 상태 업데이트
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/system-status');
        const data = await response.json();
        
        // 현재 시간 업데이트
        const currentTime = new Date().toLocaleTimeString();
        document.getElementById('current-time').textContent = currentTime;
        
    } catch (error) {
        console.error('Failed to update system status:', error);
    }
}

// 모듈 로드
async function loadModule(moduleName) {
    // 사운드 재생
    playSound('click');
    
    const contentArea = document.getElementById('content-area');
    currentModule = moduleName;
    
    // 로딩 표시
    contentArea.innerHTML = `
        <div class="loading-screen">
            <div class="loading-spinner">
                <i class="fas fa-cog fa-spin"></i>
            </div>
            <h2>Loading ${moduleName}...</h2>
        </div>
    `;
    
    try {
        let content = '';
        
        switch (moduleName) {
            case 'trading':
                content = await loadTradingDashboard();
                break;
            case 'guild':
                content = await loadGuildSystem();
                break;
            case 'wallet':
                content = await loadWalletModule();
                break;
            default:
                content = '<h2>Module not found</h2>';
        }
        
        contentArea.innerHTML = content;
        contentArea.classList.add('fade-in');

        // 모듈별 초기화 훅
        if (moduleName === 'trading') {
            initializeTradingCharts();
        } else if (moduleName === 'wallet') {
            initializeWalletModule();
        }
        
        updateStatusMessage(`${moduleName} module loaded`);
        playSound('success');
        
    } catch (error) {
        console.error('Failed to load module:', error);
        contentArea.innerHTML = `
            <div class="error-screen">
                <i class="fas fa-exclamation-triangle"></i>
                <h2>Failed to load ${moduleName}</h2>
                <p>${error.message}</p>
            </div>
        `;
        playSound('error');
        updateStatusMessage('Module load failed');
    }
}

// 트레이딩 대시보드 로드
async function loadTradingDashboard() {
    try {
        const response = await fetch('/api/trading-data');
        const data = await response.json();
        
        return `
            <div class="trading-dashboard">
                <div class="dashboard-header">
                    <h2><i class="fas fa-chart-line"></i> Trading Dashboard</h2>
                    <div class="dashboard-stats">
                        <div class="stat-item">
                            <span class="stat-label">Current Price:</span>
                            <span class="stat-value">$${data.current_price.toLocaleString()}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Change:</span>
                            <span class="stat-value ${data.price_change >= 0 ? 'positive' : 'negative'}">
                                ${data.price_change >= 0 ? '+' : ''}${data.price_change}%
                            </span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Volume:</span>
                            <span class="stat-value">${(data.volume / 1000000).toFixed(2)}M</span>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="trading-chart" width="800" height="400"></canvas>
                </div>
                <div class="nb-wave-panel" style="margin-top:14px;">
                    <h3>N/B Wave Map</h3>
                    <canvas id="nb-wave-chart" width="800" height="120"></canvas>
                </div>
                
                <div class="signals-panel">
                    <h3>Active Signals</h3>
                    <div class="signals-grid">
                        ${data.signals.map(signal => `
                            <div class="signal-card ${signal.type}">
                                <div class="signal-type">${signal.type.toUpperCase()}</div>
                                <div class="signal-strength">Strength: ${(signal.strength * 100).toFixed(0)}%</div>
                                <div class="signal-timeframe">${signal.timeframe}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        throw new Error('Failed to load trading data');
    }
}

// 길드 시스템 로드
async function loadGuildSystem() {
    try {
        const response = await fetch('/api/residents');
        const residents = await response.json();
        
        return `
            <div class="guild-system">
                <div class="guild-header">
                    <h2><i class="fas fa-users-cog"></i> Guild System</h2>
                    <div class="guild-stats">
                        <div class="stat-item">
                            <span class="stat-label">Total Members:</span>
                            <span class="stat-value">${Object.keys(residents).length}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Active:</span>
                            <span class="stat-value">${Object.values(residents).filter(r => r.hp > 0).length}</span>
                        </div>
                    </div>
                </div>
                
                <div class="residents-grid">
                    ${Object.entries(residents).map(([id, resident]) => `
                        <div class="resident-card">
                            <div class="resident-header">
                                <h3>${resident.name}</h3>
                                <span class="resident-role">${resident.role}</span>
                            </div>
                            <div class="resident-stats">
                                <div class="stat-bar">
                                    <span>HP:</span>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${(resident.hp / resident.maxHp) * 100}%"></div>
                                    </div>
                                    <span>${resident.hp}/${resident.maxHp}</span>
                                </div>
                                <div class="stat-bar">
                                    <span>Stamina:</span>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: ${(resident.stamina / resident.maxStamina) * 100}%"></div>
                                    </div>
                                    <span>${resident.stamina}/${resident.maxStamina}</span>
                                </div>
                            </div>
                            <div class="resident-info">
                                <p><strong>Location:</strong> ${resident.location}</p>
                                <p><strong>Specialty:</strong> ${resident.specialty}</p>
                                <p><strong>Skill Level:</strong> ${resident.skillLevel}</p>
                            </div>
                            <div class="resident-actions">
                                <button class="action-btn" onclick="manageResident('${id}')">
                                    <i class="fas fa-cog"></i> Manage
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
    } catch (error) {
        throw new Error('Failed to load guild data');
    }
}

// 주민 관리
function manageResident(residentId) {
    playSound('click');
    updateStatusMessage(`Managing ${residentId}...`);
    
    // 여기에 주민 관리 로직 추가
    console.log('Managing resident:', residentId);
}

// 차트 그리기 (간단한 예시)
function drawTradingChart() {
    const canvas = document.getElementById('trading-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // 배경
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(0, 0, width, height);
    
    // 격자
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        const y = (height / 10) * i;
        
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // 샘플 데이터로 선 그리기
    const data = [49000, 49500, 50000, 50500, 51000];
    const stepX = width / (data.length - 1);
    const minPrice = Math.min(...data);
    const maxPrice = Math.max(...data);
    const priceRange = maxPrice - minPrice;
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((price, index) => {
        const x = index * stepX;
        const y = height - ((price - minPrice) / priceRange) * height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
}

// 가격 차트 데이터로 그리기
function drawPriceChartFromData(chartData) {
    const canvas = document.getElementById('trading-chart');
    if (!canvas) {
        console.error('Trading chart canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 배경 클리어
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(0, 0, width, height);

    // 격자 그리기
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)';
    ctx.lineWidth = 1;
    
    // 세로 격자
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    
    // 가로 격자
    for (let i = 0; i <= 8; i++) {
        const y = (height / 8) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    const data = chartData.prices || [];
    if (!data.length) {
        console.warn('No price data available');
        return;
    }

    console.log('Drawing price chart with data:', data);

    const stepX = width / (data.length - 1);
    const minPrice = Math.min(...data);
    const maxPrice = Math.max(...data);
    const priceRange = Math.max(1, maxPrice - minPrice);

    // 가격 라인 그리기
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    data.forEach((price, index) => {
        const x = index * stepX;
        const y = height - ((price - minPrice) / priceRange) * height;
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // 가격 포인트 그리기
    ctx.fillStyle = '#00ff00';
    data.forEach((price, index) => {
        const x = index * stepX;
        const y = height - ((price - minPrice) / priceRange) * height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });

    // 가격 라벨 그리기
    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'left';
    
    // 최소/최대 가격 표시
    ctx.fillText(`Max: ${maxPrice.toLocaleString()}`, 10, 20);
    ctx.fillText(`Min: ${minPrice.toLocaleString()}`, 10, 40);
    ctx.fillText(`Current: ${data[data.length - 1].toLocaleString()}`, 10, 60);
}

// NB Wave 차트 그리기 (점으로 표시 + 거래량 라인)
function drawNbWaveChart(nbData) {
    const canvas = document.getElementById('nb-wave-chart');
    if (!canvas) {
        console.error('NB Wave chart canvas not found');
        return;
    }
    
    if (!nbData) {
        console.warn('No NB wave data available');
        return;
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 배경 클리어
    ctx.fillStyle = 'rgba(0, 0, 0, 0.95)';
    ctx.fillRect(0, 0, width, height);

    const labels = nbData.labels || [];
    const zones = nbData.zones || [];
    
    if (!labels.length || !zones.length) {
        console.warn('No labels or zones data');
        return;
    }

    console.log('Drawing NB wave chart with zones:', zones.length);

    const stepX = width / zones.length;
    const chartHeight = height - 60; // 상단 제목과 하단 정보 공간 확보
    const volumeHeight = 80; // 거래량 차트 높이

    // 격자 그리기
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 40);
        ctx.lineTo(x, height - volumeHeight);
        ctx.stroke();
    }

    // 거래량 데이터 준비
    const volumes = zones.map(z => z.volume || 0);
    const maxVolume = Math.max(...volumes);
    const minVolume = Math.min(...volumes);

    // 거래량 라인 그리기 (하단)
    if (maxVolume > 0) {
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)'; // 노란색
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < volumes.length; i++) {
            const x = i * stepX + stepX / 2;
            const volumeRatio = (volumes[i] - minVolume) / (maxVolume - minVolume);
            const y = height - volumeHeight + (1 - volumeRatio) * (volumeHeight - 20);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    // N/B Wave 점 그리기 및 연결선 - 모든 점 표시
    console.log('Drawing NB wave dots:', zones.length);
    
    for (let i = 0; i < zones.length; i++) {
        const z = zones[i];
        const x = i * stepX + stepX / 2;
        const y = 40 + (chartHeight / 2); // 중앙에 위치
        
        // 점 크기 (강도에 따라) - 더 크게 표시
        const dotSize = 6 + (z.strength || 0.5) * 6;
        
        // 구역별 색상
        if (z.zone === 'ORANGE') {
            ctx.fillStyle = 'rgba(255, 183, 3, 0.9)'; // 주황색
            ctx.strokeStyle = 'rgba(255, 140, 0, 1)';
        } else if (z.zone === 'BLUE') {
            ctx.fillStyle = 'rgba(0, 209, 255, 0.9)'; // 파란색
            ctx.strokeStyle = 'rgba(0, 102, 204, 1)';
        } else {
            ctx.fillStyle = 'rgba(128, 128, 128, 0.7)'; // 회색 (중립)
            ctx.strokeStyle = 'rgba(100, 100, 100, 1)';
        }
        
        // 점 그리기 (항상 표시)
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, dotSize, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // 모든 점을 연결하는 라인 그리기
        if (i > 0) {
            const prevX = (i - 1) * stepX + stepX / 2;
            const prevY = 40 + (chartHeight / 2);
            
            // 구역별 연결선 색상 (현재 점과 이전 점의 구역을 고려)
            const prevZone = zones[i-1].zone;
            let lineColor;
            
            if (z.zone === 'ORANGE' || prevZone === 'ORANGE') {
                lineColor = 'rgba(255, 183, 3, 0.8)'; // 주황색 연결선
            } else if (z.zone === 'BLUE' || prevZone === 'BLUE') {
                lineColor = 'rgba(0, 209, 255, 0.8)'; // 파란색 연결선
            } else {
                lineColor = 'rgba(128, 128, 128, 0.6)'; // 회색 연결선
            }
            
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(x, y);
            ctx.stroke();
        }
        
        // 디버깅용: 점 위치 표시 (개발 중에만)
        if (i % 10 === 0) { // 10개마다 하나씩만 표시
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '8px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`${i}`, x, y - dotSize - 5);
        }
    }

    // 시간 라벨 그리기
    ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
    ctx.font = 'bold 10px Courier New';
    ctx.textAlign = 'center';
    
    // 라벨 개수 제한 (너무 많으면 겹침)
    const labelInterval = Math.max(1, Math.floor(labels.length / 6));
    for (let i = 0; i < labels.length; i += labelInterval) {
        const x = i * stepX + stepX / 2;
        ctx.fillText(labels[i], x, height - 5);
    }

    // 통계 정보 표시
    if (nbData.summary) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 12px Courier New';
        ctx.textAlign = 'left';
        
        // ORANGE 구역 정보
        ctx.fillStyle = 'rgba(255, 183, 3, 0.9)';
        ctx.fillText(`ORANGE: ${nbData.summary.orange}`, 10, height - volumeHeight - 35);
        
        // BLUE 구역 정보
        ctx.fillStyle = 'rgba(0, 209, 255, 0.9)';
        ctx.fillText(`BLUE: ${nbData.summary.blue}`, 10, height - volumeHeight - 20);
        
        // 현재 가격 정보
        if (nbData.summary.current_price) {
            ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
            ctx.fillText(`Price: ${nbData.summary.current_price.toLocaleString()}`, 10, height - volumeHeight - 5);
        }
        
        // 거래량 정보
        if (maxVolume > 0) {
            ctx.fillStyle = 'rgba(255, 255, 0, 0.9)';
            ctx.fillText(`Volume: ${maxVolume.toFixed(2)}`, width - 120, height - volumeHeight - 5);
        }
    }

    // 차트 제목
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 14px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('N/B Wave Analysis (Dots + Volume)', width / 2, 25);
}

// 트레이딩 모듈 초기화(데이터 로드 및 차트 그리기)
async function initializeTradingCharts() {
    try {
        console.log('Initializing trading charts...');
        
        // 설정에서 기본 시간대 가져오기
        const defaultTimeframe = window.settingsManager ? 
            window.settingsManager.getSetting('chart.defaultTimeframe') : 'minute1';
        
        const [tradeRes, nbRes] = await Promise.all([
            fetch(`/api/trading-data?timeframe=${defaultTimeframe}`),
            fetch(`/api/nb-wave?timeframe=${defaultTimeframe}&bars=120`)
        ]);
        
        if (!tradeRes.ok) {
            throw new Error(`Trading data API error: ${tradeRes.status}`);
        }
        if (!nbRes.ok) {
            throw new Error(`NB wave API error: ${nbRes.status}`);
        }
        
        const tradeData = await tradeRes.json();
        const nbData = await nbRes.json();
        
        console.log('Trading data received:', tradeData);
        console.log('NB wave data received:', nbData);

        // auto 모드 처리
        if (defaultTimeframe === 'auto') {
            if (tradeData.mode === 'auto' && nbData.mode === 'auto') {
                // 모든 시간대의 차트를 순차적으로 표시
                await displayAutoCharts(tradeData.timeframes, nbData.timeframes);
            }
        } else {
            // 단일 시간대 차트 그리기
            if (tradeData.chart_data && tradeData.chart_data.prices) {
                drawPriceChartFromData(tradeData.chart_data);
            } else {
                console.error('No chart data in trading response');
            }
            
            if (nbData.zones && nbData.zones.length > 0) {
                drawNbWaveChart(nbData);
            } else {
                console.error('No zones data in NB wave response');
            }
        }
        
        console.log('Trading charts initialized successfully');
        
    } catch (e) {
        console.error('Failed to initialize trading charts:', e);
        
        // 에러 시 fallback 차트 그리기
        drawFallbackCharts();
    }
}

// Auto 모드에서 모든 시간대 차트를 순차적으로 표시
async function displayAutoCharts(tradeTimeframes, nbTimeframes) {
    const timeframes = Object.keys(tradeTimeframes);
    let currentIndex = 0;
    
    // 첫 번째 시간대 표시
    if (timeframes.length > 0) {
        await displayTimeframeCharts(timeframes[currentIndex], tradeTimeframes[timeframes[currentIndex]], nbTimeframes[timeframes[currentIndex]]);
    }
    
    // 5초마다 다음 시간대로 자동 전환
    setInterval(async () => {
        currentIndex = (currentIndex + 1) % timeframes.length;
        const timeframe = timeframes[currentIndex];
        
        if (tradeTimeframes[timeframe] && nbTimeframes[timeframe]) {
            await displayTimeframeCharts(timeframe, tradeTimeframes[timeframe], nbTimeframes[timeframe]);
        }
    }, 5000);
}

// 특정 시간대의 차트 표시
async function displayTimeframeCharts(timeframe, tradeData, nbData) {
    console.log(`Displaying charts for ${timeframe}`);
    
    // 상태 메시지 업데이트
    updateStatusMessage(`Auto Mode: ${timeframe} charts displayed`);
    
    // 가격 차트 그리기
    if (tradeData.chart_data && tradeData.chart_data.prices) {
        drawPriceChartFromData(tradeData.chart_data);
    }
    
    // NB Wave 차트 그리기
    if (nbData.zones && nbData.zones.length > 0) {
        drawNbWaveChart(nbData);
    }
    
    // 현재 시간대 표시 업데이트
    const timeframeDisplay = document.getElementById('current-timeframe');
    if (timeframeDisplay) {
        timeframeDisplay.textContent = `Current: ${timeframe}`;
    }
}

// Fallback 차트 (데이터 없을 때)
function drawFallbackCharts() {
    console.log('Drawing fallback charts...');
    
    // 가격 차트 fallback
    const priceCanvas = document.getElementById('trading-chart');
    if (priceCanvas) {
        const ctx = priceCanvas.getContext('2d');
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, priceCanvas.width, priceCanvas.height);
        
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.font = '16px Courier New';
        ctx.textAlign = 'center';
        ctx.fillText('Price Chart - Data Loading...', priceCanvas.width/2, priceCanvas.height/2);
    }
    
    // NB Wave 차트 fallback
    const nbCanvas = document.getElementById('nb-wave-chart');
    if (nbCanvas) {
        const ctx = nbCanvas.getContext('2d');
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, nbCanvas.width, nbCanvas.height);
        
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.font = '16px Courier New';
        ctx.textAlign = 'center';
        ctx.fillText('NB Wave Chart - Data Loading...', nbCanvas.width/2, nbCanvas.height/2);
    }
}

// 차트 새로고침 함수 (설정에서 호출)
window.refreshCharts = function() {
    if (currentModule === 'trading') {
        initializeTradingCharts();
    }
};

// ===== 스크린샷 기능 =====

// 스크린샷 촬영 (클립보드로 복사)
function takeScreenshot() {
    // 사운드 재생
    playSound('click');
    updateStatusMessage('Taking screenshot...');
    
    try {
        // 현재 모듈에 따라 다른 스크린샷 처리
        if (currentModule === 'trading') {
            takeTradingScreenshot();
        } else if (currentModule === 'guild') {
            takeGuildScreenshot();
        } else {
            takeFullScreenshot();
        }
        
    } catch (error) {
        console.error('Screenshot failed:', error);
        updateStatusMessage('Screenshot failed');
        playSound('error');
    }
}

// 트레이딩 대시보드 스크린샷
function takeTradingScreenshot() {
    const contentArea = document.getElementById('content-area');
    if (!contentArea) return;
    
    // 전체 컨텐츠 영역을 캡처
    html2canvas(contentArea, {
        backgroundColor: '#000000',
        scale: 2, // 고해상도
        useCORS: true,
        allowTaint: true,
        logging: false
    }).then(canvas => {
        // 클립보드로 복사
        copyToClipboard(canvas);
        
        // 성공 메시지만 표시 (팝업 없음)
        updateStatusMessage('Screenshot copied to clipboard!');
        playSound('success');
    });
}

// 길드 시스템 스크린샷
function takeGuildScreenshot() {
    const contentArea = document.getElementById('content-area');
    if (!contentArea) return;
    
    html2canvas(contentArea, {
        backgroundColor: '#000000',
        scale: 2,
        useCORS: true,
        allowTaint: true,
        logging: false
    }).then(canvas => {
        // 클립보드로 복사
        copyToClipboard(canvas);
        
        // 성공 메시지만 표시 (팝업 없음)
        updateStatusMessage('Screenshot copied to clipboard!');
        playSound('success');
    });
}

// 전체 화면 스크린샷
function takeFullScreenshot() {
    html2canvas(document.body, {
        backgroundColor: '#000000',
        scale: 1.5,
        useCORS: true,
        allowTaint: true,
        logging: false
    }).then(canvas => {
        // 클립보드로 복사
        copyToClipboard(canvas);
        
        // 성공 메시지만 표시 (팝업 없음)
        updateStatusMessage('Screenshot copied to clipboard!');
        playSound('success');
    });
}

// 캔버스를 클립보드로 복사
function copyToClipboard(canvas) {
    canvas.toBlob(function(blob) {
        // Clipboard API 사용
        if (navigator.clipboard && window.ClipboardItem) {
            const item = new ClipboardItem({ "image/png": blob });
            navigator.clipboard.write([item]).then(function() {
                console.log('Screenshot copied to clipboard');
            }).catch(function(err) {
                console.error('Failed to copy to clipboard:', err);
                // Fallback: 다운로드
                downloadCanvas(canvas, `screenshot-${Date.now()}.png`);
            });
        } else {
            // Fallback: 다운로드
            downloadCanvas(canvas, `screenshot-${Date.now()}.png`);
        }
    });
}

// 캔버스를 파일로 다운로드 (fallback용)
function downloadCanvas(canvas, filename) {
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// 스크린샷 정보 표시
function showScreenshotInfo(type) {
    const infoDiv = document.createElement('div');
    infoDiv.className = 'screenshot-info';
    infoDiv.innerHTML = `
        <div class="screenshot-popup">
            <h3><i class="fas fa-camera"></i> Screenshot Copied!</h3>
            <p><strong>Type:</strong> ${type}</p>
            <p><strong>Status:</strong> Copied to clipboard</p>
            <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
            <button onclick="this.parentElement.parentElement.remove()">OK</button>
        </div>
    `;
    
    document.body.appendChild(infoDiv);
    
    // 3초 후 자동 제거
    setTimeout(() => {
        if (infoDiv.parentElement) {
            infoDiv.remove();
        }
    }, 3000);
}

// 초기화
function initializeGame() {
    console.log('🎮 Initializing 8BIT Trading System v0.1...');
    
    // 사운드 시스템 초기화
    initializeSound();
    
    // 게임 상태 업데이트
    updateGameState();
    updateSystemStatus();
    
    // 주기적 업데이트
    setInterval(updateGameState, 5000);
    setInterval(updateSystemStatus, 1000);
    
    // 차트는 모듈 로드시 초기화
    
    updateStatusMessage('System initialized');
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 8BIT Trading System v0.1 - Starcraft Style UI Loaded');
    initializeGame();
    // notifyPageLoad(); // 서버 사운드 기능 제거
    // 
    // // 주기적으로 페이지 상태 확인 (30초마다)
    // setInterval(() => {
    //     checkPageStatus();
    // }, 30000);
});

// 키보드 단축키
document.addEventListener('keydown', function(event) {
    switch(event.key) {
        case '1':
            loadModule('trading');
            break;
        case '2':
            loadModule('guild');
            break;
        case '3':
            loadModule('wallet');
            break;
        case 'm':
            toggleSound();
            break;
        case 's':
        case 'S':
            takeScreenshot();
            break;
        case 'Escape':
            // 메인 화면으로 돌아가기
            location.reload();
            break;
    }
});

// 지갑 모듈 로드
async function loadWalletModule() {
    try {
        return `
            <div class="wallet-dashboard">
                <div class="wallet-header">
                    <h2><i class="fas fa-wallet"></i> Wallet Dashboard</h2>
                    <div class="wallet-actions">
                        <button class="wallet-btn" onclick="refreshWalletData()">
                            <i class="fas fa-sync-alt"></i> Refresh
                        </button>
                        <button class="wallet-btn" onclick="exportWalletData()">
                            <i class="fas fa-download"></i> Export
                        </button>
                    </div>
                </div>
                
                <div class="wallet-overview">
                    <div class="balance-card">
                        <h3>Total Balance</h3>
                        <div class="balance-amount" id="total-balance">Loading...</div>
                        <div class="balance-change" id="balance-change">+0.00%</div>
                    </div>
                    
                    <div class="quick-stats">
                        <div class="stat-card">
                            <i class="fas fa-coins"></i>
                            <div class="stat-info">
                                <span class="stat-label">KRW Balance</span>
                                <span class="stat-value" id="krw-balance">Loading...</span>
                            </div>
                        </div>
                        <div class="stat-card">
                            <i class="fas fa-bitcoin"></i>
                            <div class="stat-info">
                                <span class="stat-label">BTC Balance</span>
                                <span class="stat-value" id="btc-balance">Loading...</span>
                            </div>
                        </div>
                        <div class="stat-card">
                            <i class="fas fa-chart-line"></i>
                            <div class="stat-info">
                                <span class="stat-label">Portfolio Value</span>
                                <span class="stat-value" id="portfolio-value">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="wallet-content">
                    <div class="balance-details">
                        <h3>Balance Details</h3>
                        <div class="balance-table" id="balance-table">
                            <div class="loading">Loading balances...</div>
                        </div>
                    </div>
                    
                    <div class="transaction-history">
                        <h3>Recent Transactions</h3>
                        <div class="transaction-list" id="transaction-list">
                            <div class="loading">Loading transactions...</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load wallet module:', error);
        return `
            <div class="error-screen">
                <i class="fas fa-exclamation-triangle"></i>
                <h2>Failed to load Wallet</h2>
                <p>${error.message}</p>
            </div>
        `;
    }
}

// 지갑 모듈 초기화
async function initializeWalletModule() {
    try {
        console.log('Initializing wallet module...');
        await refreshWalletData();
        updateStatusMessage('Wallet module initialized');
    } catch (error) {
        console.error('Failed to initialize wallet module:', error);
        updateStatusMessage('Wallet initialization failed');
    }
}

// 지갑 데이터 새로고침
async function refreshWalletData() {
    try {
        // 설정에서 Upbit API 키 가져오기
        const accessKey = window.settingsManager ? 
            window.settingsManager.getSetting('upbit.accessKey') : '';
        const secretKey = window.settingsManager ? 
            window.settingsManager.getSetting('upbit.secretKey') : '';
        
        if (!accessKey || !secretKey) {
            updateBalanceDisplay([]);
            updateStatusMessage('Please set Upbit API keys in settings');
            return;
        }
        
        // Upbit 잔고 정보 가져오기
        const balanceResponse = await fetch('/api/upbit-balance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                accessKey: accessKey,
                secretKey: secretKey
            })
        });
        
        const balanceData = await balanceResponse.json();
        
        if (balanceData.success) {
            updateBalanceDisplay(balanceData.balances);
        } else {
            console.error('Failed to fetch balance data:', balanceData.error);
            updateBalanceDisplay([]);
            updateStatusMessage('Failed to fetch balance information');
        }
        
        // 거래 내역 가져오기
        updateTransactionHistory();
        
    } catch (error) {
        console.error('Failed to refresh wallet data:', error);
        updateStatusMessage('Failed to refresh wallet data');
    }
}

// 잔고 표시 업데이트
function updateBalanceDisplay(balances) {
    let totalKRW = 0;
    let totalBTC = 0;
    let portfolioValue = 0;
    
    // 잔고 테이블 업데이트
    const balanceTable = document.getElementById('balance-table');
    if (balanceTable) {
        if (balances.length === 0) {
            balanceTable.innerHTML = '<div class="no-data">No balance data available</div>';
        } else {
            let tableHTML = `
                <table class="balance-table-content">
                    <thead>
                        <tr>
                            <th>Currency</th>
                            <th>Balance</th>
                            <th>Locked</th>
                            <th>Avg Price</th>
                            <th>Current</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            balances.forEach(balance => {
                const currency = balance.currency;
                const bal = parseFloat(balance.balance);
                const locked = parseFloat(balance.locked || 0);
                const avgPrice = parseFloat(balance.avg_buy_price || 0);
                const currentPrice = parseFloat(balance.price || 0);
                const assetValue = parseFloat(balance.asset_value || 0);
                
                if (currency === 'KRW') {
                    totalKRW = bal;
                } else if (currency === 'BTC') {
                    totalBTC = bal;
                }
                
                portfolioValue += assetValue;
                
                tableHTML += `
                    <tr>
                        <td>${currency}</td>
                        <td>${bal > 0 ? bal.toFixed(8) : '0.00000000'}</td>
                        <td>${locked > 0 ? locked.toFixed(8) : '0.00000000'}</td>
                        <td>${avgPrice > 0 ? avgPrice.toLocaleString() : '-'}</td>
                        <td>${currentPrice > 0 ? currentPrice.toLocaleString() : '-'}</td>
                        <td>${assetValue > 0 ? assetValue.toLocaleString() : '-'}</td>
                    </tr>
                `;
            });
            
            tableHTML += '</tbody></table>';
            balanceTable.innerHTML = tableHTML;
        }
    }
    
    // 요약 정보 업데이트
    const krwBalance = document.getElementById('krw-balance');
    const btcBalance = document.getElementById('btc-balance');
    const portfolioValueEl = document.getElementById('portfolio-value');
    const totalBalance = document.getElementById('total-balance');
    
    if (krwBalance) krwBalance.textContent = `${totalKRW.toLocaleString()} KRW`;
    if (btcBalance) btcBalance.textContent = `${totalBTC.toFixed(8)} BTC`;
    if (portfolioValueEl) portfolioValueEl.textContent = `${portfolioValue.toLocaleString()} KRW`;
    if (totalBalance) totalBalance.textContent = `${portfolioValue.toLocaleString()} KRW`;
}

    // 거래 내역 업데이트
    async function updateTransactionHistory() {
        const transactionList = document.getElementById('transaction-list');
        if (transactionList) {
            try {
                // 로딩 표시
                transactionList.innerHTML = `
                    <div class="loading">
                        <i class="fas fa-spinner fa-spin" style="color: #f39c12; margin-bottom: 10px; font-size: 24px;"></i>
                        <div style="color: #cccccc;">Loading transaction history...</div>
                    </div>
                `;
                
                // 설정에서 Upbit API 키 가져오기
                const settings = window.settingsManager ? window.settingsManager.getSettings() : {};
                const accessKey = settings.upbitAccessKey || '';
                const secretKey = settings.upbitSecretKey || '';
                
                if (!accessKey || !secretKey) {
                    transactionList.innerHTML = `
                        <div class="no-data">
                            <i class="fas fa-exclamation-triangle" style="color: #e74c3c; margin-bottom: 10px; font-size: 24px;"></i>
                            <div style="color: #cccccc; margin-bottom: 10px;">Transaction History</div>
                            <div style="color: #888888; font-size: 12px;">
                                Upbit API keys not configured.<br>
                                Please set them in Settings to view transaction history.
                            </div>
                        </div>
                    `;
                    return;
                }
                
                // 주문 히스토리 API 호출 (API 키와 함께)
                const response = await fetch('/api/order-history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        accessKey: accessKey,
                        secretKey: secretKey
                    })
                });
                const data = await response.json();
                
                if (data.success && data.orders && data.orders.length > 0) {
                    // 실제 주문 데이터 표시
                    let ordersHtml = '';
                    data.orders.forEach(order => {
                        const orderClass = order.type === 'BUY' ? 'buy' : 'sell';
                        const stateClass = order.state.toLowerCase();
                        
                        ordersHtml += `
                            <div class="transaction-item">
                                <div class="transaction-info">
                                    <div class="transaction-type ${orderClass}">${order.type}</div>
                                    <div class="transaction-details">
                                        <div class="transaction-amount">${order.executed_volume.toFixed(8)} ${order.code.split('-')[1]}</div>
                                        <div class="transaction-price">${order.executed_funds.toLocaleString()} KRW</div>
                                        <div class="transaction-time">${order.time}</div>
                                        <div class="transaction-state ${stateClass}">${order.state}</div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    transactionList.innerHTML = ordersHtml;
                } else {
                    // 주문 데이터가 없거나 WebSocket이 연결되지 않은 경우
                                                        transactionList.innerHTML = `
                                        <div class="no-data">
                                            <i class="fas fa-info-circle" style="color: #f39c12; margin-bottom: 10px; font-size: 24px;"></i>
                                            <div style="color: #cccccc; margin-bottom: 10px;">Transaction History</div>
                                            <div style="color: #888888; font-size: 12px;">
                                                No recent transactions found.<br>
                                                <strong>Note:</strong> This shows completed orders from Upbit API.<br>
                                                Transactions will appear here when you have trading history.
                                            </div>
                                            <button onclick="updateTransactionHistory()" class="btn btn-primary" style="margin-top: 10px;">
                                                <i class="fas fa-sync-alt"></i> Refresh History
                                            </button>
                                        </div>
                                    `;
                }
                
            } catch (error) {
                console.error('Transaction history error:', error);
                transactionList.innerHTML = `
                    <div class="no-data">
                        <i class="fas fa-exclamation-triangle" style="color: #e74c3c; margin-bottom: 10px; font-size: 24px;"></i>
                        <div style="color: #cccccc; margin-bottom: 10px;">Transaction History</div>
                        <div style="color: #888888; font-size: 12px;">
                            Failed to load transaction history.<br>
                            Please check your connection and try again.
                        </div>
                    </div>
                `;
            }
        }
    }
    
    // WebSocket 연결 시작
    async function startWebSocketConnection() {
        try {
            const settings = window.settingsManager.getSettings();
            const accessKey = settings.upbitAccessKey || '';
            const secretKey = settings.upbitSecretKey || '';
            
            if (!accessKey || !secretKey) {
                updateStatusMessage('❌ Upbit API keys not configured. Please set them in Settings.');
                return;
            }
            
            updateStatusMessage('🔗 Starting WebSocket connection...');
            
            const response = await fetch('/api/start-websocket', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    accessKey: accessKey,
                    secretKey: secretKey
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                updateStatusMessage('✅ Real-time order events connected! New orders will appear here when you trade.');
                // 잠시 후 거래 내역 새로고침
                setTimeout(() => {
                    updateTransactionHistory();
                }, 2000);
            } else {
                updateStatusMessage(`❌ Real-time events connection failed: ${data.error}`);
            }
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            updateStatusMessage('❌ Failed to start WebSocket connection.');
        }
    }

// 지갑 데이터 내보내기
function exportWalletData() {
    try {
        const data = {
            timestamp: new Date().toISOString(),
            balances: document.getElementById('balance-table').innerHTML,
            transactions: document.getElementById('transaction-list').innerHTML
        };
        
        const dataStr = JSON.stringify(data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `wallet-data-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        URL.revokeObjectURL(url);
        
        updateStatusMessage('Wallet data exported successfully');
        playSound('success');
    } catch (error) {
        console.error('Failed to export wallet data:', error);
        updateStatusMessage('Failed to export wallet data');
        playSound('error');
    }
}

// 추가 CSS 스타일 (동적으로 추가)
const additionalStyles = `
    .loading-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #00ff00;
    }
    
    .loading-spinner {
        font-size: 48px;
        margin-bottom: 20px;
        color: #00ff00;
    }
    
    .error-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #ff6b6b;
    }
    
    .error-screen i {
        font-size: 48px;
        margin-bottom: 20px;
    }
    
    .dashboard-header, .guild-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #00ff00;
    }
    
    .dashboard-stats, .guild-stats {
        display: flex;
        gap: 20px;
    }
    
    .stat-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .stat-label {
        font-size: 12px;
        color: #cccccc;
    }
    
    .stat-value {
        font-size: 18px;
        font-weight: bold;
        color: #00ff00;
    }
    
    .positive { color: #00ff00; }
    .negative { color: #ff6b6b; }
    
    .chart-container {
        margin: 20px 0;
        border: 1px solid #00ff00;
        border-radius: 5px;
        padding: 10px;
    }
    
    .signals-panel {
        margin-top: 20px;
    }
    
    .signals-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    
    .signal-card {
        padding: 15px;
        border: 1px solid #00ff00;
        border-radius: 5px;
        background: rgba(0, 255, 0, 0.1);
    }
    
    .signal-card.buy {
        border-color: #00ff00;
        background: rgba(0, 255, 0, 0.1);
    }
    
    .signal-card.sell {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
    }
    
    .signal-type {
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .residents-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .resident-card {
        border: 1px solid #00ff00;
        border-radius: 5px;
        padding: 15px;
        background: rgba(0, 255, 0, 0.1);
    }
    
    .resident-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .resident-role {
        font-size: 12px;
        color: #cccccc;
        padding: 2px 8px;
        border: 1px solid #00ff00;
        border-radius: 10px;
    }
    
    .resident-stats {
        margin-bottom: 15px;
    }
    
    .stat-bar {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    
    .progress-bar {
        flex: 1;
        height: 8px;
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid #00ff00;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff00, #00cc00);
        transition: width 0.3s ease;
    }
    
    .resident-info p {
        margin-bottom: 5px;
        font-size: 14px;
    }
    
    .resident-actions {
        margin-top: 15px;
    }
    
    .action-btn {
        background: none;
        border: 1px solid #00ff00;
        color: #00ff00;
        padding: 8px 16px;
        cursor: pointer;
        border-radius: 3px;
        transition: all 0.3s ease;
    }
    
    .action-btn:hover {
        background: #00ff00;
        color: #000;
    }
    
    /* Wallet Dashboard Styles */
    .wallet-dashboard {
        padding: 20px;
    }
    
    .wallet-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid #f39c12;
    }
    
    .wallet-header h2 {
        color: #f39c12;
        margin: 0;
        text-shadow: 0 0 10px #f39c12;
    }
    
    .wallet-actions {
        display: flex;
        gap: 10px;
    }
    
    .wallet-actions .wallet-btn {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        border: 1px solid #f39c12;
        color: #000;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    .wallet-actions .wallet-btn:hover {
        background: linear-gradient(135deg, #e67e22, #d35400);
        box-shadow: 0 0 10px rgba(243, 156, 18, 0.5);
    }
    
    .wallet-overview {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .balance-card {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        border: 2px solid #f39c12;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #000;
    }
    
    .balance-card h3 {
        margin: 0 0 15px 0;
        font-size: 18px;
    }
    
    .balance-amount {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .balance-change {
        font-size: 14px;
        opacity: 0.8;
    }
    
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    
    .stat-card {
        background: rgba(243, 156, 18, 0.1);
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 15px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .stat-card i {
        font-size: 24px;
        color: #f39c12;
    }
    
    .stat-info {
        display: flex;
        flex-direction: column;
    }
    
    .stat-label {
        font-size: 12px;
        color: #cccccc;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 16px;
        font-weight: bold;
        color: #f39c12;
    }
    
    .wallet-content {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    
    .balance-details, .transaction-history {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 20px;
        max-height: 400px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #f39c12 rgba(243, 156, 18, 0.2);
    }
    
    .balance-details::-webkit-scrollbar,
    .transaction-history::-webkit-scrollbar {
        width: 8px;
    }
    
    .balance-details::-webkit-scrollbar-track,
    .transaction-history::-webkit-scrollbar-track {
        background: rgba(243, 156, 18, 0.1);
        border-radius: 4px;
    }
    
    .balance-details::-webkit-scrollbar-thumb,
    .transaction-history::-webkit-scrollbar-thumb {
        background: #f39c12;
        border-radius: 4px;
    }
    
    .balance-details::-webkit-scrollbar-thumb:hover,
    .transaction-history::-webkit-scrollbar-thumb:hover {
        background: #e67e22;
    }
    
    .balance-details h3, .transaction-history h3 {
        color: #f39c12;
        margin: 0 0 20px 0;
        text-shadow: 0 0 10px #f39c12;
    }
    
    .balance-table-content {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        table-layout: fixed;
    }
    
    .balance-table-content th,
    .balance-table-content td {
        padding: 6px 4px;
        text-align: left;
        border-bottom: 1px solid #f39c12;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .balance-table-content th:nth-child(1),
    .balance-table-content td:nth-child(1) {
        width: 12%;
    }
    
    .balance-table-content th:nth-child(2),
    .balance-table-content td:nth-child(2) {
        width: 18%;
    }
    
    .balance-table-content th:nth-child(3),
    .balance-table-content td:nth-child(3) {
        width: 18%;
    }
    
    .balance-table-content th:nth-child(4),
    .balance-table-content td:nth-child(4) {
        width: 18%;
    }
    
    .balance-table-content th:nth-child(5),
    .balance-table-content td:nth-child(5) {
        width: 18%;
    }
    
    .balance-table-content th:nth-child(6),
    .balance-table-content td:nth-child(6) {
        width: 16%;
    }
    
    .balance-table-content th {
        background: rgba(243, 156, 18, 0.2);
        color: #f39c12;
        font-weight: bold;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    .balance-table-content td {
        color: #ffffff;
    }
    
    .transaction-item {
        padding: 8px;
        border-bottom: 1px solid #2c3e50;
        display: flex;
        align-items: center;
    }
    
    .transaction-info {
        display: flex;
        align-items: center;
        width: 100%;
    }
    
    .transaction-type {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 10px;
        min-width: 40px;
        text-align: center;
    }
    
    .transaction-type.buy {
        background-color: #27ae60;
        color: white;
    }
    
    .transaction-type.sell {
        background-color: #e74c3c;
        color: white;
    }
    
    .transaction-details {
        flex: 1;
        font-size: 11px;
    }
    
    .transaction-amount {
        color: #ecf0f1;
        font-weight: bold;
    }
    
    .transaction-price {
        color: #bdc3c7;
    }
    
    .transaction-time {
        color: #95a5a6;
        font-size: 10px;
    }
    
    .transaction-state {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 9px;
        margin-top: 2px;
    }
    
    .transaction-state.pending {
        background-color: #f39c12;
        color: white;
    }
    
    .transaction-state.trading {
        background-color: #3498db;
        color: white;
    }
    
    .transaction-state.completed {
        background-color: #27ae60;
        color: white;
    }
    
    .transaction-state.cancelled {
        background-color: #e74c3c;
        color: white;
    }
    
    .transaction-state.prevented {
        background-color: #9b59b6;
        color: white;
    }
    
    .loading {
        text-align: center;
        padding: 20px;
    }
    
    .no-data {
        text-align: center;
        padding: 20px;
    }
    
    .btn-primary {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
    }
    
    .btn-primary:hover {
        background-color: #2980b9;
    }
        border: 1px solid #f39c12;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .transaction-info {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .transaction-type {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        min-width: 50px;
        text-align: center;
    }
    
    .transaction-type.buy {
        background: #00ff00;
        color: #000;
    }
    
    .transaction-type.sell {
        background: #ff6b6b;
        color: #000;
    }
    
    .transaction-details {
        flex: 1;
    }
    
    .transaction-amount {
        font-weight: bold;
        color: #f39c12;
        margin-bottom: 5px;
    }
    
    .transaction-price {
        color: #ffffff;
        margin-bottom: 3px;
    }
    
    .transaction-time {
        font-size: 12px;
        color: #cccccc;
    }
    
    .loading, .no-data {
        text-align: center;
        color: #cccccc;
        padding: 20px;
    }
`;

// 스타일 추가
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);
