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
                // 타이핑 완료 후 3초 후 "Ready"로 복원
                setTimeout(() => {
                    if (statusElement.textContent === message) {
                        typeWriterReady();
                    }
                }, 3000);
            }
        };
        
        const typeWriterReady = () => {
            const readyMessage = 'Ready';
            statusElement.textContent = '';
            index = 0;
            const typeReady = () => {
                if (index < readyMessage.length) {
                    statusElement.textContent += readyMessage.charAt(index);
                    // 타이핑 효과음 재생 (공백 제외) - Startup 사운드 대신 click 사운드 사용
                    if (readyMessage.charAt(index) !== ' ') {
                        playSound('click');
                    }
                    index++;
                    setTimeout(typeReady, 30);
                }
            };
            typeReady();
        };
        
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
            default:
                content = '<h2>Module not found</h2>';
        }
        
        contentArea.innerHTML = content;
        contentArea.classList.add('fade-in');

        // 모듈별 초기화 훅
        if (moduleName === 'trading') {
            initializeTradingCharts();
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

// NB Wave 차트 그리기 (기존 bot 시스템 기반)
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

    // 격자 그리기
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    // 구역 그리기 (기존 bot 시스템 스타일)
    for (let i = 0; i < zones.length; i++) {
        const z = zones[i];
        
        // 기존 bot 시스템의 색상 사용
        if (z.zone === 'ORANGE') {
            ctx.fillStyle = 'rgba(255, 183, 3, 0.7)'; // 주황색 영역
            ctx.strokeStyle = 'rgba(255, 140, 0, 0.9)'; // 주황색 선
        } else {
            ctx.fillStyle = 'rgba(0, 209, 255, 0.7)'; // 파란색 영역
            ctx.strokeStyle = 'rgba(0, 102, 204, 0.9)'; // 파란색 선
        }
        
        const x = i * stepX;
        const barWidth = Math.ceil(stepX) + 1;
        
        // 구역 배경
        ctx.fillRect(x, 0, barWidth, height);
        
        // 구역 경계선
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    // 구역 경계선 강조
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    for (let i = 1; i < zones.length; i++) {
        if (zones[i].zone !== zones[i-1].zone) {
            const x = i * stepX;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
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
        ctx.fillText(labels[i], x, 18);
    }

    // 통계 정보 표시 (기존 bot 시스템 스타일)
    if (nbData.summary) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 12px Courier New';
        ctx.textAlign = 'left';
        
        // ORANGE 구역 정보
        ctx.fillStyle = 'rgba(255, 183, 3, 0.9)';
        ctx.fillText(`ORANGE: ${nbData.summary.orange}`, 10, height - 35);
        
        // BLUE 구역 정보
        ctx.fillStyle = 'rgba(0, 209, 255, 0.9)';
        ctx.fillText(`BLUE: ${nbData.summary.blue}`, 10, height - 20);
        
        // 현재 가격 정보
        if (nbData.summary.current_price) {
            ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
            ctx.fillText(`Price: ${nbData.summary.current_price.toLocaleString()}`, 10, height - 5);
        }
    }

    // 차트 제목
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 14px Courier New';
    ctx.textAlign = 'center';
    ctx.fillText('N/B Wave Analysis', width / 2, 25);
}

// 트레이딩 모듈 초기화(데이터 로드 및 차트 그리기)
async function initializeTradingCharts() {
    try {
        console.log('Initializing trading charts...');
        
        const [tradeRes, nbRes] = await Promise.all([
            fetch('/api/trading-data'),
            fetch('/api/nb-wave?timeframe=minute1&bars=120')
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

        // 차트 그리기
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
        
        console.log('Trading charts initialized successfully');
        
    } catch (e) {
        console.error('Failed to initialize trading charts:', e);
        
        // 에러 시 fallback 차트 그리기
        drawFallbackCharts();
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
`;

// 스타일 추가
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);
