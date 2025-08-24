// ===== 8BIT Trading System - Starcraft Style JavaScript =====

// 전역 변수
let currentModule = null;
let selectedKrwCoin = 'BTC'; // 선택된 KRW 코인
let gameState = {
    minerals: 0,
    gas: 0,
    supply: 0,
    maxSupply: 100
};

// 현재 구역 강도 전역 변수
window.currentZoneStrength = 0;

// Asset Display Module 로드
// 자산 표시 관련 기능들은 modules/asset/asset_display.js에서 처리됩니다.

// Asset Display 관련 함수들은 modules/asset/asset_display.js에서 처리됩니다.
// 기존 함수들은 호환성을 위해 래퍼로 유지합니다.

// 초기 자산 데이터 설정 (래퍼 함수)
function initializeAssetData() {
    if (window.assetDisplayManager) {
        window.assetDisplayManager.initializeAssetData();
    }
}

// balance-item과 asset-display 동기화 함수 (래퍼 함수)
function syncBalanceWithAssetDisplay() {
    if (window.assetDisplayManager) {
        window.assetDisplayManager.syncBalanceWithAssetDisplay();
    }
}

// 백그라운드 지갑 데이터 시스템 (Header Manager로 대체됨)
// 이제 modules/header/header_manager.js에서 처리됩니다.

// 페이지 로드 시 CSS 스타일 추가
document.addEventListener('DOMContentLoaded', () => {
    addHeaderUpdateStyles();
    addWalletStyles();
    
    // 헤더 관리는 modules/header/header_manager.js에서 처리됩니다.
    
    // 자산 데이터 초기화
    setTimeout(() => {
        initializeAssetData();
    }, 1000);
    
    // 모든 모듈 백그라운드 초기화 시작
    initializeAllModulesInBackground();
    
    // 우측 패널 Trading Dashboard Info 초기 업데이트
    showRightPanelLoading();
    updateRightPanelTradingInfo();
    
    // 우측 패널 주기적 업데이트 (30초마다)
    window.rightPanelTimer = setInterval(() => {
        updateRightPanelTradingInfo();
    }, 30000);
    
    // 우측 패널 현재가가 0일 때 감지하여 재호출
    window.priceDetectionTimer = setInterval(() => {
        const rightCurrentPriceElement = document.getElementById('right-trading-current-price');
        if (rightCurrentPriceElement) {
            const currentPriceText = rightCurrentPriceElement.textContent;
            if (currentPriceText === '₩0' || currentPriceText === '₩0') {
                //console.log('⚠️ Right panel current price is 0, triggering immediate update...');
                updateRightPanelTradingInfo();
            }
        }
    }, 5000); // 5초마다 체크
});

// 모든 모듈 백그라운드 초기화 함수
async function initializeAllModulesInBackground() {
    //console.log('🚀 Starting background initialization of all modules...');
    
    try {
        // 1. Central System 백그라운드 초기화
        //console.log('🏛️ Initializing Central System in background...');
        await initializeCentralSystemInBackground();
        
        // 2. Trading Dashboard 백그라운드 초기화
        //console.log('📊 Initializing Trading Dashboard in background...');
        await initializeTradingDashboardInBackground();
        
        // 3. Guild System 백그라운드 초기화
        //console.log('⚔️ Initializing Guild System in background...');
        await initializeGuildSystemInBackground();
        
        // 4. Wallet 백그라운드 초기화
        //console.log('💰 Initializing Wallet in background...');
        await initializeWalletInBackground();
        
        //console.log('✅ All modules initialized in background successfully!');
        
    } catch (error) {
        console.error('❌ Failed to initialize modules in background:', error);
    }
}

// Central System 백그라운드 초기화
async function initializeCentralSystemInBackground() {
    try {
        // Central System 관련 데이터 미리 로드
        //console.log('🏛️ Central System background data loaded');
    } catch (error) {
        console.error('❌ Central System background init failed:', error);
    }
}

// Trading Dashboard 백그라운드 초기화
async function initializeTradingDashboardInBackground() {
    try {
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        
        // Trading Dashboard 데이터 미리 로드
        const [tradeRes, nbRes] = await Promise.all([
            fetch(`/api/trading-data?coin=${selectedCoin}`),
            fetch(`/api/nb-wave?coin=${selectedCoin}`)
        ]);
        
        if (tradeRes.ok && nbRes.ok) {
            const tradeData = await tradeRes.json();
            const nbData = await nbRes.json();
            
            // 전역 변수에 데이터 저장
            window.backgroundTradingData = tradeData;
            window.backgroundNbData = nbData;
            
            //console.log('📊 Trading Dashboard background data loaded');
        }
        
        // Trading Dashboard 백그라운드 타이머 시작
        startTradingPriceUpdate();
        
    } catch (error) {
        console.error('❌ Trading Dashboard background init failed:', error);
    }
}

// Guild System 백그라운드 초기화
async function initializeGuildSystemInBackground() {
    try {
        // Guild System 데이터 미리 로드
        const response = await fetch('/api/residents');
        if (response.ok) {
            const residents = await response.json();
            window.backgroundGuildData = residents;
            //console.log('⚔️ Guild System background data loaded');
        }
    } catch (error) {
        console.error('❌ Guild System background init failed:', error);
    }
}

// Wallet 백그라운드 초기화
async function initializeWalletInBackground() {
    try {
        // Wallet 데이터는 Header Manager에서 처리되므로 추가 초기화만
        //console.log('💰 Wallet background initialization completed');
    } catch (error) {
        console.error('❌ Wallet background init failed:', error);
    }
}

// 사운드 재생 함수 (새로운 오디오 시스템 사용)
function playSound(soundType) {
    try {
        if (window.audioSystem && window.audioSystem.soundEnabled) {
            window.audioSystem.play(soundType);
        }
    } catch (e) {
        console.error('Sound play error:', e);
    }
}

// 사운드 초기화 함수
function initializeSound() {
    try {
        //console.log('Sound system initialized (using external audio system)');
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
                // 타이핑 효과음 재생 (공백 제외)
                if (message.charAt(index) !== ' ') {
                    // 사운드 시스템이 준비되었는지 확인
                    if (window.audioSystem && window.audioSystem.soundEnabled) {
                        try {
                            window.audioSystem.play('click');
                        } catch (e) {
                            console.debug('타이핑 사운드 재생 실패:', e.message);
                        }
                    }
                }
                index++;
                setTimeout(typeWriter, 50); // 타이핑 속도
            } else {
                // 타이핑 완료 후 메시지 유지
            }
        };
        
        typeWriter();
        statusElement.classList.add('glow');
        setTimeout(() => statusElement.classList.remove('glow'), 2000);
    }
}

// 게임 상태 업데이트 (Header Manager로 대체됨)
async function updateGameState() {
    try {
        const response = await fetch('/api/game-state');
        const data = await response.json();
        gameState = data;
        
        // UI 업데이트는 Header Manager에서 처리하므로 여기서는 제거
        // document.getElementById('mineral-count').textContent = gameState.minerals;
        // document.getElementById('gas-count').textContent = gameState.gas;
        // document.getElementById('supply-count').textContent = `${gameState.supply}/${gameState.maxSupply}`;
        
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
// 모듈 로드 함수
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
            case 'central':
                content = await loadCentralSystem();
                break;
            case 'trading':
                content = await loadTradingDashboard();
                break;
            case 'guild':
                content = await loadGuildSystem();
                break;
            case 'wallet':
                content = await loadWalletModule();
                break;
            case 'settings':
                content = await loadSettings();
                break;
            default:
                content = '<h2>Module not found</h2>';
        }
        
        contentArea.innerHTML = content;
        contentArea.classList.add('fade-in');

        // 모듈별 초기화 훅
        if (moduleName === 'central') {
            initializeCentralSystem();
        } else if (moduleName === 'trading') {
            initializeTradingCharts();
        } else if (moduleName === 'wallet') {
            initializeWalletModule();
        } else if (moduleName === 'settings') {
            initializeSettings();
        }
        
        updateStatusMessage(`${moduleName} module loaded`);
        playSound('success');

        // 우측 패널은 어떤 메뉴에서도 최신 상태 유지 - 즉시 갱신
        //console.log(`🔄 Menu switched to ${moduleName}, updating right panel...`);
        
        // 우측 패널 갱신을 약간 지연시켜 DOM이 완전히 로드된 후 실행
        setTimeout(() => {
            if (typeof updateRightPanelTradingInfo === 'function') {
                try { 
                    updateRightPanelTradingInfo(); 
                    //console.log(`✅ Right panel updated for ${moduleName}`);
                } catch (e) { 
                    console.error('❌ Right panel refresh failed:', e); 
                }
            } else {
                console.warn('⚠️ updateRightPanelTradingInfo function not found');
            }
        }, 100);
        
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

// Active Signals 템플릿 로드
async function loadActiveSignalsTemplate(data) {
    try {
        if (window.templateLoader) {
            const template = await window.templateLoader.loadActiveSignalsTemplate();
            return window.templateLoader.bindTemplate(template, data);
        } else {
            // 템플릿 로더가 없으면 기본 HTML 반환
            return `
                <div class="signals-panel">
                    <h3>Active Signals</h3>
                    <div class="signals-grid">
                        <div class="no-signals-message">Template loader not available</div>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load Active Signals template:', error);
        return '<div class="signals-panel"><h3>Active Signals</h3><div class="signals-grid">Template load failed</div></div>';
    }
}

// 트레이딩 대시보드 로드
async function loadTradingDashboard() {
    try {
        // 선택된 코인 가져오기
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        //console.log('🪙 Loading Trading Dashboard for coin:', selectedCoin);
        
        const response = await fetch(`/api/trading-data?coin=${selectedCoin}`);
        const data = await response.json();
        
        return `
            <div class="trading-dashboard">
                <div class="dashboard-header">
                    <h2><i class="fas fa-chart-line"></i> Trading Dashboard - ${selectedCoin}/KRW</h2>
                    <div class="current-price-display">
                        <div class="price-main">
                            <span class="price-label">현재가:</span>
                            <span id="trading-current-price" class="current-price">₩${data.current_price ? data.current_price.toLocaleString() : '0'}</span>
                        </div>
                        <div class="price-change-info">
                            <span id="trading-price-change" class="price-change ${data.price_change >= 0 ? 'positive' : 'negative'}">
                                ${data.price_change >= 0 ? '+' : ''}${data.price_change ? data.price_change.toFixed(2) : '0.00'}%
                            </span>
                        </div>
                    </div>
                    <div class="current-zone-display">
                        <div class="zone-main">
                            <span class="zone-label">현재 구역:</span>
                            <span id="trading-current-zone" class="current-zone">Loading...</span>
                        </div>
                        <div class="zone-info">
                            <span id="trading-zone-strength" class="zone-strength">강도: 0%</span>
                        </div>
                    </div>
                    <div class="dashboard-stats">
                        <div class="stat-item">
                            <span class="stat-label">Volume:</span>
                            <span class="stat-value">${(data.volume / 1000000).toFixed(2)}M</span>
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <span id="current-timeframe" class="current-timeframe">Current: ${convertTimeframeToDisplay(data.timeframe || 'day')}</span>
                    </div>
                    <canvas id="trading-chart" width="100%" height="400"></canvas>
                </div>
                <div id="nb-wave-container"></div>
                <div id="timeframe-cards-container"></div>
                
                ${await this.loadActiveSignalsTemplate(data)}
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

// Central System 초기화 (통합 뷰)
async function initializeCentralSystem() {
    try {
        //console.log('🔄 Initializing Central System with integrated view...');
        
        // 1. Trading Dashboard 로드 (기존 페이지 그대로)
        //console.log('📊 Loading Trading Dashboard in Central Hub...');
        const tradingContent = await loadTradingDashboard();
        const tradingContainer = document.getElementById('trading-dashboard-container');
        if (tradingContainer) {
            tradingContainer.innerHTML = tradingContent;
            await initializeTradingCharts();
            
            // 분봉선택 카드들 초기화 (Central Hub에서도 활성화)
            await initializeTimeframeCards();
        }
        
        // 2. Wallet 로드 (기존 페이지 그대로)
        //console.log('💰 Loading Wallet in Central Hub...');
        const walletContent = await loadWalletModule();
        const walletContainer = document.getElementById('wallet-dashboard-container');
        if (walletContainer) {
            walletContainer.innerHTML = walletContent;
            await initializeWalletModule();
        }
        
        //console.log('✅ Central System initialized successfully');
        
    } catch (error) {
        console.error('❌ Failed to initialize Central System:', error);
    }
}



// 길드 시스템 초기화 (대시보드용)
async function initializeGuildSystem() {
    try {
        //console.log('🔄 Initializing Guild System...');
        
        const response = await fetch('/api/residents');
        const residents = await response.json();
        
        // 길드 멤버 수 업데이트
        const memberCountElement = document.getElementById('guild-member-count');
        if (memberCountElement) {
            memberCountElement.textContent = Object.keys(residents).length;
        }
        
        // 길드 멤버 목록 업데이트
        const membersListElement = document.getElementById('guild-members-list');
        if (membersListElement) {
            if (Object.keys(residents).length > 0) {
                membersListElement.innerHTML = Object.entries(residents).map(([id, resident]) => `
                    <div class="resident-card" style="background: rgba(0,255,0,0.1); border: 1px solid #00ff00; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <div class="resident-header">
                            <h4 style="color: #00ff00; margin: 0 0 5px 0;">${resident.name}</h4>
                            <span style="color: #cccccc; font-size: 12px;">${resident.role}</span>
                        </div>
                        <div class="resident-stats" style="margin: 10px 0;">
                            <div style="margin-bottom: 5px;">
                                <span style="color: #00ff00;">HP: ${resident.hp}/${resident.maxHp}</span>
                            </div>
                            <div>
                                <span style="color: #00ff00;">Stamina: ${resident.stamina}/${resident.maxStamina}</span>
                            </div>
                        </div>
                        <div style="font-size: 12px; color: #cccccc;">
                            <div>Location: ${resident.location}</div>
                            <div>Specialty: ${resident.specialty}</div>
                            <div>Skill: ${resident.skillLevel}</div>
                        </div>
                    </div>
                `).join('');
            } else {
                membersListElement.innerHTML = '<div style="color: #cccccc; text-align: center;">No guild members found</div>';
            }
        }
        
        //console.log('✅ Guild System initialized successfully');
        
    } catch (error) {
        console.error('Failed to initialize guild system:', error);
        const membersListElement = document.getElementById('guild-members-list');
        if (membersListElement) {
            membersListElement.innerHTML = '<div style="color: #ff6b6b; text-align: center;">Failed to load guild data</div>';
        }
    }
}

// 주민 관리
function manageResident(residentId) {
    playSound('click');
    updateStatusMessage(`Managing ${residentId}...`);
    
    // 여기에 주민 관리 로직 추가
    //console.log('Managing resident:', residentId);
}

// 설정 모듈 로드
async function loadSettings() {
    return `
        <div class="settings-module">
            <div class="settings-header">
                <h2><i class="fas fa-cog"></i> Settings</h2>
                <p>System Configuration</p>
            </div>
            <div class="settings-content">
                <div class="settings-section">
                    <h3>General Settings</h3>
                    <div class="setting-item">
                        <label>Theme:</label>
                        <select id="theme-select">
                            <option value="dark">Dark Theme</option>
                            <option value="light">Light Theme</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Language:</label>
                        <select id="language-select">
                            <option value="ko">한국어</option>
                            <option value="en">English</option>
                        </select>
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>Upbit API Settings</h3>
                    <div class="setting-item">
                        <label>Access Key:</label>
                        <input type="password" id="upbit-access-key" placeholder="Enter Upbit Access Key">
                        <button onclick="togglePasswordVisibility('upbit-access-key')" class="eye-btn">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                    <div class="setting-item">
                        <label>Secret Key:</label>
                        <input type="password" id="upbit-secret-key" placeholder="Enter Upbit Secret Key">
                        <button onclick="togglePasswordVisibility('upbit-secret-key')" class="eye-btn">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                    <div class="setting-item">
                        <label>API Test:</label>
                        <button onclick="testUpbitAPI()" class="test-btn">Test API Connection</button>
                        <span id="api-test-result"></span>
                    </div>
                    <div class="setting-item">
                        <label>Default KRW Coin:</label>
                        <select id="default-krw-coin">
                            <option value="BTC">Bitcoin (BTC)</option>
                            <option value="ETH">Ethereum (ETH)</option>
                            <option value="XRP">Ripple (XRP)</option>
                            <option value="ADA">Cardano (ADA)</option>
                            <option value="DOT">Polkadot (DOT)</option>
                            <option value="LINK">Chainlink (LINK)</option>
                            <option value="LTC">Litecoin (LTC)</option>
                            <option value="BCH">Bitcoin Cash (BCH)</option>
                            <option value="XLM">Stellar (XLM)</option>
                            <option value="VET">VeChain (VET)</option>
                            <option value="TRX">TRON (TRX)</option>
                            <option value="FIL">Filecoin (FIL)</option>
                            <option value="ATOM">Cosmos (ATOM)</option>
                            <option value="ALGO">Algorand (ALGO)</option>
                            <option value="NEAR">NEAR Protocol (NEAR)</option>
                            <option value="FTM">Fantom (FTM)</option>
                            <option value="AVAX">Avalanche (AVAX)</option>
                            <option value="SOL">Solana (SOL)</option>
                            <option value="MATIC">Polygon (MATIC)</option>
                            <option value="UNI">Uniswap (UNI)</option>
                            <option value="AAVE">Aave (AAVE)</option>
                            <option value="SUSHI">SushiSwap (SUSHI)</option>
                            <option value="CAKE">PancakeSwap (CAKE)</option>
                            <option value="DOGE">Dogecoin (DOGE)</option>
                            <option value="SHIB">Shiba Inu (SHIB)</option>
                        </select>
                        <span class="setting-description">Select your preferred KRW trading pair</span>
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>Sound Settings</h3>
                    <div class="setting-item">
                        <label>Master Volume:</label>
                        <input type="range" id="master-volume" min="0" max="100" value="50">
                        <span id="volume-value">50%</span>
                    </div>
                    <div class="setting-item">
                        <label>Sound Effects:</label>
                        <input type="checkbox" id="sound-effects" checked>
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>Game Settings</h3>
                    <div class="setting-item">
                        <label>Phaser Ball Size:</label>
                        <input type="range" id="ball-size" min="1" max="3" step="0.1" value="1.5">
                        <span id="ball-size-value">1.5x</span>
                    </div>
                </div>
                
                <div class="settings-section">
                    <h3>Trading Settings</h3>
                    <div class="setting-item">
                        <label>Default Timeframe:</label>
                        <select id="default-timeframe">
                            <option value="minute1">1 Minute</option>
                            <option value="minute3">3 Minutes</option>
                            <option value="minute5">5 Minutes</option>
                            <option value="minute10" selected>10 Minutes</option>
                            <option value="minute15">15 Minutes</option>
                            <option value="minute30">30 Minutes</option>
                            <option value="minute60">1 Hour</option>
                            <option value="day">1 Day</option>
                        </select>
                    </div>
                    <div class="setting-item">
                        <label>Auto Rotate Timeframe Cards:</label>
                        <input type="checkbox" id="timeframe-auto-rotate" checked>
                        <span class="setting-description">Timeframe cards will rotate automatically</span>
                    </div>
                    <div class="setting-item">
                        <label>Auto Rotate Interval (seconds):</label>
                        <input type="range" id="auto-rotate-interval" min="3" max="10" value="5">
                        <span id="auto-rotate-interval-value">5 seconds</span>
                    </div>
                </div>
                
                <div class="settings-actions">
                    <button onclick="saveSettings()" class="save-btn">Save Settings</button>
                    <button onclick="resetSettings()" class="reset-btn">Reset to Default</button>
                </div>
            </div>
        </div>
    `;
}

// 설정 저장
async function saveSettings() {
    const settings = {
        theme: document.getElementById('theme-select').value,
        language: document.getElementById('language-select').value,
        upbitAccessKey: document.getElementById('upbit-access-key').value,
        upbitSecretKey: document.getElementById('upbit-secret-key').value,
        masterVolume: document.getElementById('master-volume').value,
        soundEffects: document.getElementById('sound-effects').checked,
        ballSize: document.getElementById('ball-size').value,
        defaultTimeframe: document.getElementById('default-timeframe').value,
        timeframe_auto_rotate: document.getElementById('timeframe-auto-rotate').checked,
        auto_rotate_interval: parseInt(document.getElementById('auto-rotate-interval').value)
    };
    
    // localStorage에 저장
    localStorage.setItem('8bit-settings', JSON.stringify(settings));
    
    // 백엔드에 API 키 저장
    try {
        await fetch('/api/settings/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                section: 'upbit',
                key: 'upbitAccessKey',
                value: settings.upbitAccessKey
            })
        });
        
                            await fetch('/api/settings/update', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            section: 'upbit',
                            key: 'upbitSecretKey',
                            value: settings.upbitSecretKey
                        })
                    });
                    
                    // 기본 KRW 코인 설정 저장
                    await fetch('/api/settings/update', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            section: 'upbit',
                            key: 'defaultKrwCoin',
                            value: settings.defaultKrwCoin
                        })
                    });
                    
                    //console.log('✅ API 키와 코인 설정이 백엔드에 저장되었습니다');
    } catch (error) {
        console.error('❌ API 키 저장 실패:', error);
    }
    
    // 실시간으로 모든 설정 적용
    applyAllSettings();
    
    // 분봉 카드 설정 업데이트
    updateTimeframeCardsSettings(settings);
    
    updateStatusMessage('설정이 저장되고 적용되었습니다');
    playSound('success');
}

// 분봉 카드 설정 업데이트
function updateTimeframeCardsSettings(settings) {
    if (window.timeframeCards) {
        // 자동 순회 설정 업데이트
        if (settings.timeframe_auto_rotate) {
            const interval = (settings.auto_rotate_interval || 5) * 1000;
            window.timeframeCards.startAutoRotate(interval);
        } else {
            window.timeframeCards.stopAutoRotate();
        }
        
        // 기본 분봉 설정 업데이트
        if (settings.defaultTimeframe && settings.defaultTimeframe !== window.timeframeCards.getCurrentTimeframe()) {
            window.timeframeCards.selectTimeframe(settings.defaultTimeframe);
        }
    }
}

// 설정 리셋
function resetSettings() {
    if (confirm('Are you sure you want to reset all settings to default?')) {
        localStorage.removeItem('8bit-settings');
        location.reload();
    }
}

// 비밀번호 표시/숨김 토글
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    const button = input.nextElementSibling;
    const icon = button.querySelector('i');
    
    if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fas fa-eye-slash';
    } else {
        input.type = 'password';
        icon.className = 'fas fa-eye';
    }
}

// Upbit API 연결 테스트
async function testUpbitAPI() {
    const accessKey = document.getElementById('upbit-access-key').value;
    const secretKey = document.getElementById('upbit-secret-key').value;
    const resultSpan = document.getElementById('api-test-result');
    
    if (!accessKey || !secretKey) {
        resultSpan.innerHTML = '<span style="color: #e74c3c;">⚠️ Please enter API keys</span>';
        return;
    }
    
    resultSpan.innerHTML = '<span style="color: #f39c12;">🔄 Testing...</span>';
    
    try {
        const response = await fetch('/api/test-upbit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                accessKey: accessKey,
                secretKey: secretKey
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                resultSpan.innerHTML = '<span style="color: #27ae60;">✅ API connection successful</span>';
            } else {
                resultSpan.innerHTML = `<span style="color: #e74c3c;">❌ ${data.error}</span>`;
            }
        } else {
            resultSpan.innerHTML = '<span style="color: #e74c3c;">❌ Server error</span>';
        }
    } catch (error) {
        resultSpan.innerHTML = '<span style="color: #e74c3c;">❌ Network error</span>';
        console.error('Upbit API test failed:', error);
    }
}

// 설정 초기화
function initializeSettings() {
    try {
        const savedSettings = localStorage.getItem('8bit-settings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            // 설정값 적용 (안전하게)
            setTimeout(() => {
                if (document.getElementById('theme-select')) {
                    document.getElementById('theme-select').value = settings.theme || 'dark';
                }
                if (document.getElementById('language-select')) {
                    document.getElementById('language-select').value = settings.language || 'ko';
                }
                if (document.getElementById('upbit-access-key')) {
                    document.getElementById('upbit-access-key').value = settings.upbitAccessKey || '';
                }
                if (document.getElementById('upbit-secret-key')) {
                    document.getElementById('upbit-secret-key').value = settings.upbitSecretKey || '';
                }
                if (document.getElementById('default-krw-coin')) {
                    document.getElementById('default-krw-coin').value = settings.defaultKrwCoin || 'BTC';
                    // 전역 변수도 업데이트
                    selectedKrwCoin = settings.defaultKrwCoin || 'BTC';
                    window.selectedKrwCoin = selectedKrwCoin;
                }
                if (document.getElementById('master-volume')) {
                    document.getElementById('master-volume').value = settings.masterVolume || 50;
                    const volumeValue = document.getElementById('volume-value');
                    if (volumeValue) {
                        volumeValue.textContent = settings.masterVolume + '%';
                    }
                }
                if (document.getElementById('sound-effects')) {
                    document.getElementById('sound-effects').checked = settings.soundEffects !== false;
                }
                if (document.getElementById('ball-size')) {
                    document.getElementById('ball-size').value = settings.ballSize || 1.5;
                    const ballSizeValue = document.getElementById('ball-size-value');
                    if (ballSizeValue) {
                        ballSizeValue.textContent = settings.ballSize + 'x';
                    }
                }
                
                // 분봉 카드 설정
                if (document.getElementById('default-timeframe')) {
                    document.getElementById('default-timeframe').value = settings.defaultTimeframe || 'minute10';
                }
                if (document.getElementById('timeframe-auto-rotate')) {
                    document.getElementById('timeframe-auto-rotate').checked = settings.timeframe_auto_rotate !== false;
                }
                if (document.getElementById('auto-rotate-interval')) {
                    document.getElementById('auto-rotate-interval').value = settings.auto_rotate_interval || 5;
                    const intervalValue = document.getElementById('auto-rotate-interval-value');
                    if (intervalValue) {
                        intervalValue.textContent = (settings.auto_rotate_interval || 5) + ' seconds';
                    }
                }
            }, 100); // DOM 로드 대기
        }
        
        // 이벤트 리스너 추가 (안전하게)
        setTimeout(() => {
            if (document.getElementById('master-volume')) {
                document.getElementById('master-volume').addEventListener('input', function() {
                    const volume = this.value;
                    const volumeValue = document.getElementById('volume-value');
                    if (volumeValue) {
                        volumeValue.textContent = volume + '%';
                    }
                    // 실시간 볼륨 적용
                    applyVolumeSettings(volume);
                });
            }
            
            if (document.getElementById('ball-size')) {
                document.getElementById('ball-size').addEventListener('input', function() {
                    const size = this.value;
                    const ballSizeValue = document.getElementById('ball-size-value');
                    if (ballSizeValue) {
                        ballSizeValue.textContent = size + 'x';
                    }
                    // 실시간 Phaser Ball 크기 적용
                    applyBallSizeSettings(size);
                });
            }
            
            if (document.getElementById('theme-select')) {
                document.getElementById('theme-select').addEventListener('change', function() {
                    const theme = this.value;
                    // 실시간 테마 적용
                    applyThemeSettings(theme);
                });
            }
            
            if (document.getElementById('sound-effects')) {
                document.getElementById('sound-effects').addEventListener('change', function() {
                    const enabled = this.checked;
                    // 실시간 사운드 설정 적용
                    applySoundSettings(enabled);
                });
            }
            
            // 분봉 카드 설정 이벤트 리스너
            if (document.getElementById('timeframe-auto-rotate')) {
                document.getElementById('timeframe-auto-rotate').addEventListener('change', function() {
                    const enabled = this.checked;
                    // 실시간 분봉 카드 자동 순회 설정 적용
                    if (window.timeframeCards) {
                        if (enabled) {
                            const interval = parseInt(document.getElementById('auto-rotate-interval').value || 5) * 1000;
                            window.timeframeCards.startAutoRotate(interval);
                        } else {
                            window.timeframeCards.stopAutoRotate();
                        }
                    }
                });
            }
            
            if (document.getElementById('auto-rotate-interval')) {
                document.getElementById('auto-rotate-interval').addEventListener('input', function() {
                    const interval = this.value;
                    const intervalValue = document.getElementById('auto-rotate-interval-value');
                    if (intervalValue) {
                        intervalValue.textContent = interval + ' seconds';
                    }
                    // 실시간 자동 순회 간격 적용
                    if (window.timeframeCards && window.timeframeCards.isAutoRotating) {
                        window.timeframeCards.stopAutoRotate();
                        window.timeframeCards.startAutoRotate(interval * 1000);
                    }
                });
            }
            
            // 코인 선택 이벤트 리스너
            if (document.getElementById('default-krw-coin')) {
                document.getElementById('default-krw-coin').addEventListener('change', function() {
                    updateSelectedCoin();
                });
            }
        }, 200); // 이벤트 리스너 추가 대기
        
        // 초기 설정값 적용
        applyAllSettings();
        
        // 설정 모듈 CSS 추가
        addSettingsStyles();
        
        //console.log('✅ Settings initialized successfully');
        
    } catch (error) {
        console.error('❌ Settings initialization failed:', error);
    }
}

// 모든 설정값 적용
function applyAllSettings() {
    const savedSettings = localStorage.getItem('8bit-settings');
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        // 테마 적용
        if (settings.theme) {
            applyThemeSettings(settings.theme);
        }
        
        // 볼륨 적용
        if (settings.masterVolume) {
            applyVolumeSettings(settings.masterVolume);
        }
        
        // 사운드 설정 적용
        if (settings.soundEffects !== undefined) {
            applySoundSettings(settings.soundEffects);
        }
        
        // Phaser Ball 크기 적용
        if (settings.ballSize) {
            applyBallSizeSettings(settings.ballSize);
        }
        
        // 분봉 카드 설정 적용
        if (window.timeframeCards) {
            if (settings.timeframe_auto_rotate !== undefined) {
                if (settings.timeframe_auto_rotate) {
                    const interval = (settings.auto_rotate_interval || 5) * 1000;
                    window.timeframeCards.startAutoRotate(interval);
                } else {
                    window.timeframeCards.stopAutoRotate();
                }
            }
            
            if (settings.defaultTimeframe) {
                window.timeframeCards.selectTimeframe(settings.defaultTimeframe);
            }
        }
    }
}

// 테마 설정 적용
function applyThemeSettings(theme) {
    const body = document.body;
    const gameContainer = document.querySelector('.game-container');
    
    if (theme === 'light') {
        body.classList.add('light-theme');
        body.classList.remove('dark-theme');
        if (gameContainer) {
            gameContainer.classList.add('light-theme');
            gameContainer.classList.remove('dark-theme');
        }
    } else {
        body.classList.add('dark-theme');
        body.classList.remove('light-theme');
        if (gameContainer) {
            gameContainer.classList.add('dark-theme');
            gameContainer.classList.remove('light-theme');
        }
    }
    
    // 테마별 CSS 변수 설정
    if (theme === 'light') {
        document.documentElement.style.setProperty('--bg-color', '#f5f5f5');
        document.documentElement.style.setProperty('--text-color', '#333333');
        document.documentElement.style.setProperty('--border-color', '#cccccc');
        document.documentElement.style.setProperty('--accent-color', '#3498db');
    } else {
        document.documentElement.style.setProperty('--bg-color', '#1a1a1a');
        document.documentElement.style.setProperty('--text-color', '#ecf0f1');
        document.documentElement.style.setProperty('--border-color', '#34495e');
        document.documentElement.style.setProperty('--accent-color', '#00ff00');
    }
}

// 볼륨 설정 적용
function applyVolumeSettings(volume) {
    try {
        // 오디오 시스템에 볼륨 적용
        if (window.audioSystem && typeof window.audioSystem.setVolume === 'function') {
            window.audioSystem.setVolume(volume / 100);
        }
        
        // HTML5 오디오 요소들에 볼륨 적용
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            try {
                audio.volume = volume / 100;
            } catch (e) {
                console.debug('Audio volume setting failed:', e.message);
            }
        });
        
        // Web Audio API 컨텍스트에 볼륨 적용
        if (window.audioContext && window.audioContext.gainNode) {
            try {
                window.audioContext.gainNode.gain.value = volume / 100;
            } catch (e) {
                console.debug('Web Audio API volume setting failed:', e.message);
            }
        }
        
        //console.log(`🔊 Volume applied: ${volume}%`);
        
    } catch (error) {
        console.error('❌ Volume setting failed:', error);
    }
}

// 사운드 설정 적용
function applySoundSettings(enabled) {
    try {
        // 오디오 시스템 활성화/비활성화
        if (window.audioSystem) {
            if (enabled) {
                if (typeof window.audioSystem.enable === 'function') {
                    window.audioSystem.enable();
                }
            } else {
                if (typeof window.audioSystem.disable === 'function') {
                    window.audioSystem.disable();
                }
            }
        }
        
        // 전역 사운드 플래그 설정
        window.soundEnabled = enabled;
        
        //console.log(`🔊 Sound ${enabled ? 'enabled' : 'disabled'}`);
        
    } catch (error) {
        console.error('❌ Sound setting failed:', error);
    }
}

// Phaser Ball 크기 설정 적용
function applyBallSizeSettings(size) {
    // Central Hub의 Phaser Ball 크기 변경 (물리 엔진 고려)
    if (window.ball && window.game) {
        const newScale = parseFloat(size) * 0.5; // 50% 더 작게
        window.ball.setScale(newScale);
        
        // 물리 바디 크기도 함께 조정
        if (window.ball.body) {
            window.ball.body.setSize(window.ball.width * newScale, window.ball.height * newScale);
        }
    }
    
    // 전역 Ball 크기 설정 저장
    window.currentBallSize = parseFloat(size);
}

// 선택된 코인 업데이트
function updateSelectedCoin() {
    const coinSelect = document.getElementById('default-krw-coin');
    if (coinSelect) {
        selectedKrwCoin = coinSelect.value;
        //console.log('🪙 Selected coin updated to:', selectedKrwCoin);
        
        // 전역 변수로 저장 (다른 모듈에서 사용 가능)
        window.selectedKrwCoin = selectedKrwCoin;
        
        // 오른쪽 패널 업데이트
        if (window.headerManager) {
            window.headerManager.setDefaultRightPanelValues();
        }
        
        // Trading Dashboard가 활성화되어 있다면 차트 새로고침
        if (currentModule === 'trading') {
            //console.log('🔄 Refreshing trading charts for new coin:', selectedKrwCoin);
            initializeTradingCharts();
        }
        
        // 우측 패널 Trading Dashboard Info 업데이트
        updateRightPanelTradingInfo();
    }
}

// 설정 저장 확인 메시지 표시
function showSettingsSaveMessage(message) {
    // 기존 메시지 제거
    const existingMessage = document.getElementById('settings-save-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // 새 메시지 생성
    const messageDiv = document.createElement('div');
    messageDiv.id = 'settings-save-message';
    messageDiv.className = 'settings-save-message';
    messageDiv.innerHTML = `
        <div class="save-message-content">
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    // 스타일 추가
    if (!document.getElementById('settings-save-message-style')) {
        const style = document.createElement('style');
        style.id = 'settings-save-message-style';
        style.textContent = `
            .settings-save-message {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #27ae60;
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                z-index: 10000;
                animation: slideInRight 0.5s ease-out;
                font-family: 'Courier New', monospace;
                font-weight: bold;
            }
            
            .save-message-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .save-message-content i {
                font-size: 18px;
                color: #fff;
            }
            
            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            .settings-save-message.fade-out {
                animation: slideOutRight 0.5s ease-in forwards;
            }
            
            @keyframes slideOutRight {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // 메시지 표시
    document.body.appendChild(messageDiv);
    
    // 3초 후 자동 제거
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.classList.add('fade-out');
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.remove();
                }
            }, 500);
        }
    }, 3000);
}

// 설정 저장 (실시간 적용 포함)
async function saveSettings() {
    const settings = {
        theme: document.getElementById('theme-select').value,
        language: document.getElementById('language-select').value,
        upbitAccessKey: document.getElementById('upbit-access-key').value,
        upbitSecretKey: document.getElementById('upbit-secret-key').value,
                    defaultKrwCoin: document.getElementById('default-krw-coin').value,
        masterVolume: document.getElementById('master-volume').value,
        soundEffects: document.getElementById('sound-effects').checked,
        ballSize: document.getElementById('ball-size').value
    };
    
    localStorage.setItem('8bit-settings', JSON.stringify(settings));
    
    // 백엔드에 API 키 저장
    try {
        await fetch('/api/settings/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                section: 'upbit',
                key: 'upbitAccessKey',
                value: settings.upbitAccessKey
            })
        });
        
        await fetch('/api/settings/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                section: 'upbit',
                key: 'upbitSecretKey',
                value: settings.upbitSecretKey
            })
        });
        
        //console.log('✅ API keys saved to backend successfully');
    } catch (error) {
        console.error('❌ Failed to save API keys:', error);
    }
    
    // 실시간으로 모든 설정 적용
    applyAllSettings();
    
    // 저장 확인 메시지 표시
    showSettingsSaveMessage('Settings saved successfully! ✅');
    updateStatusMessage('Settings saved and applied successfully');
    playSound('success');
}

// 페이지 로드 시 설정 적용
function applySettingsOnLoad() {
    const savedSettings = localStorage.getItem('8bit-settings');
    if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        
        // 기본 테마 적용
        if (settings.theme) {
            applyThemeSettings(settings.theme);
        }
        
        // 기본 볼륨 적용
        if (settings.masterVolume) {
            applyVolumeSettings(settings.masterVolume);
        }
        
        // 기본 사운드 설정 적용
        if (settings.soundEffects !== undefined) {
            applySoundSettings(settings.soundEffects);
        }
    }
}

// 설정 모듈 CSS 추가
function addSettingsStyles() {
    if (!document.getElementById('settings-styles')) {
        const styleElement = document.createElement('style');
        styleElement.id = 'settings-styles';
        style.textContent = `
            .settings-module {
                padding: 20px;
                background: rgba(44, 62, 80, 0.95);
                border-radius: 10px;
                margin: 20px 0;
            }
            
            .settings-header {
                margin-bottom: 30px;
                text-align: center;
                color: #ecf0f1;
            }
            
            .settings-header h2 {
                margin-bottom: 10px;
                font-size: 1.8em;
                text-shadow: 0 0 10px rgba(236, 240, 241, 0.5);
            }
            
            .settings-content {
                max-width: 600px;
                margin: 0 auto;
            }
            
            .settings-section {
                background: rgba(52, 73, 94, 0.8);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #34495e;
            }
            
            .settings-section h3 {
                color: #3498db;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            
            .setting-item {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
                padding: 10px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
            }
            
            .setting-item label {
                color: #ecf0f1;
                min-width: 120px;
                font-weight: bold;
            }
            
            .setting-item select,
            .setting-item input[type="range"],
            .setting-item input[type="password"],
            .setting-item input[type="text"] {
                flex: 1;
                margin: 0 10px;
                background: #2c3e50;
                border: 1px solid #34495e;
                color: #ecf0f1;
                padding: 5px;
                border-radius: 3px;
            }
            
            .setting-item input[type="password"],
            .setting-item input[type="text"] {
                font-family: monospace;
                font-size: 12px;
            }
            
            .eye-btn {
                background: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 5px 8px;
                border-radius: 3px;
                cursor: pointer;
                margin-left: 5px;
            }
            
            .eye-btn:hover {
                background: #2c3e50;
            }
            
            .test-btn {
                background: #f39c12;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .test-btn:hover {
                background: #e67e22;
            }
            
            .setting-item input[type="checkbox"] {
                margin-left: 10px;
                transform: scale(1.2);
            }
            
            .setting-item span {
                color: #95a5a6;
                min-width: 50px;
                text-align: right;
            }
            
            .settings-actions {
                text-align: center;
                margin-top: 30px;
            }
            
            .save-btn, .reset-btn {
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                margin: 0 10px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s ease;
            }
            
            .save-btn:hover {
                background: #2980b9;
            }
            
            .reset-btn {
                background: #e74c3c;
            }
            
            .reset-btn:hover {
                background: #c0392b;
            }
        `;
        document.head.appendChild(styleElement);
    }
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
    
    // 샘플 데이터로 선 그리기 (비활성화됨 - 중복 라인 제거)
    // const data = [49000, 49500, 50000, 50500, 51000];
    // const stepX = width / (data.length - 1);
    // const minPrice = Math.min(...data);
    // const maxPrice = Math.max(...data);
    // const priceRange = maxPrice - minPrice;
    
    // ctx.strokeStyle = '#00ff00';
    // ctx.lineWidth = 2;
    // ctx.beginPath();
    
    // data.forEach((price, index) => {
    //     const x = index * stepX;
    //     const y = height - ((price - minPrice) / priceRange) * height;
        
    //     if (index === 0) {
    //         ctx.moveTo(x, y);
    //     } else {
    //         ctx.lineTo(x, y);
    //     }
    // });
    
    // ctx.stroke();
}

// 가격 차트 데이터로 그리기 (N/B Wave 오버레이 포함)
function drawPriceChartFromData(chartData, nbData = null) {
    const canvas = document.getElementById('trading-chart');
    if (!canvas) {
        console.debug('Trading chart canvas not found - module may not be loaded yet');
        return;
    }

    // 캔버스 크기를 컨테이너에 맞게 동적으로 설정
    const container = canvas.parentElement;
    const containerWidth = container.clientWidth;
    canvas.width = containerWidth;
    canvas.height = 400;

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

    //console.log('Drawing price chart with data:', data);

    // 차트 영역을 캔버스의 80%로 제한 (좌우 10%씩 패딩)
    const chartPadding = width * 0.1; // 좌우 각각 10% 패딩
    const chartWidth = width - (chartPadding * 2); // 실제 차트 영역은 80%
    
    const stepX = chartWidth / (data.length - 1);
    const minPrice = Math.min(...data);
    const maxPrice = Math.max(...data);
    const priceRange = Math.max(1, maxPrice - minPrice);
    const dataLength = data.length;

    // N/B Wave 구역 배경 그리기 제거됨 (깔끔한 차트를 위해)

    // 가격 라인 그리기
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    data.forEach((price, index) => {
        const x = chartPadding + (index * stepX); // 좌측 패딩 추가
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
        const x = chartPadding + (index * stepX); // 좌측 패딩 추가
        const y = height - ((price - minPrice) / priceRange) * height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });

    // N/B Wave 구역 표시 제거됨 (깔끔한 차트를 위해)

    // N/B Wave 라인 그리기 (실제 가격과 스캘핑)
    if (nbData && nbData.zones && nbData.zones.length > 0) {
        //console.log('Drawing N/B Wave lines with nbData:', nbData);
        //console.log('Zones count:', nbData.zones.length);
        //console.log('First zone:', nbData.zones[0]);
        
        // r_value 확인
        //console.log('First 5 zones r_value:', nbData.zones.slice(0, 5).map(z => z.r_value));
        /** console.log('r_value range:', {
            min: Math.min(...nbData.zones.map(z => z.r_value || 0)),
            max: Math.max(...nbData.zones.map(z => z.r_value || 0))
        });
		*/
        
        // Blue/Orange Wave 카운트 및 마지막 상태 분석
        //console.log('🔍 drawPriceChartFromData - nbData.zones:', nbData.zones);
        //console.log('🔍 drawPriceChartFromData - dataLength:', dataLength);
        //console.log('🔍 drawPriceChartFromData - zones length:', nbData.zones ? nbData.zones.length : 0);
        
        const waveAnalysis = analyzeWaveCounts(nbData.zones, dataLength, nbData);
        
        drawNbWaveScalpingLine(ctx, nbData, width, height, stepX, dataLength, minPrice, maxPrice, priceRange, chartPadding);
        // N/B Wave 값을 가격과 동일한 스케일로 스캘핑하는 라인 추가
        drawNbWavePriceScalpingLine(ctx, nbData, width, height, stepX, dataLength, minPrice, maxPrice, priceRange, chartPadding);
        
        // Wave 카운트 및 마지막 상태 표시
        drawWaveCountInfo(ctx, waveAnalysis, width, height);
        
        // 메인 차트의 20개 zone 데이터를 전역으로 저장
        window.sharedMainChartData = {
            prices: data,
            zones: nbData.zones.slice(0, dataLength), // 메인 차트에 표시되는 20개 zone
            waveAnalysis: waveAnalysis,
            currentPrice: data[data.length - 1],
            last_update: new Date().toISOString(),
            timestamp: new Date().getTime()
        };
        
        //console.log('💾 Main chart data saved to global storage:', window.sharedMainChartData);
        
        // 마지막 점의 가격 정보를 nbData에 추가
        if (data.length > 0) {
            const lastPrice = data[data.length - 1];
            nbData.last_point_price = lastPrice;
            
            //console.log('📈 Last price from chart data:', lastPrice);
            /** console.log('📊 nbData structure:', {
                zones: nbData.zones ? nbData.zones.length : 0,
                last_point_price: nbData.last_point_price,
                last_zone: nbData.zones && nbData.zones.length > 0 ? nbData.zones[nbData.zones.length - 1].zone : 'N/A'
            });
            */

            // 전역 변수가 업데이트된 후 현재 구역 업데이트 (차트 마지막 구역과 동기화)
            if (window.zoneStrengthManager) {
                window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbData);
            } else {
                console.warn('⚠️ Zone Strength Manager not loaded yet');
            }
        }
    } else {
        //console.log('No nbData or zones available for N/B Wave lines');
    }

    // 가격 라벨 그리기 (우측에 표시)
    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'right';
    
    // 최소/최대 가격 표시 (우측 상단)
    ctx.fillText(`Max: ${maxPrice.toLocaleString()}`, width - 10, 20);
    ctx.fillText(`Min: ${minPrice.toLocaleString()}`, width - 10, 40);
    ctx.fillText(`Current: ${data[data.length - 1].toLocaleString()}`, width - 10, 60);

    // N/B Wave 정보 표시 (우측 하단에 표시)
    if (nbData && nbData.summary) {
        drawNbWaveInfo(ctx, nbData, width, height);
    }
}

// N/B Wave 배경 그리기 (비활성화됨 - 깔끔한 차트를 위해)
function drawNbWaveBackground(ctx, nbData, width, height, stepX, dataLength) {
    // 배경색 제거됨 - 깔끔한 차트 표시
    return;
}

// N/B Wave 구역 표시 (비활성화됨 - 깔끔한 차트를 위해)
function drawNbWaveZones(ctx, nbData, width, height, stepX, dataLength) {
    // 구역선 및 라벨 제거됨 - 깔끔한 차트 표시
    return;
}

// N/B Wave 스캘핑 라인 그리기 (실제 가격과 비교)
function drawNbWaveScalpingLine(ctx, nbData, width, height, stepX, dataLength, minPrice, maxPrice, priceRange, chartPadding) {
    const zones = nbData.zones;
    if (!zones || zones.length === 0) return;
    
    // N/B Wave change 값을 가격 범위에 맞게 스케일링 (가격과 비슷한 스케일)
    const changeValues = zones.map(z => z.change || 0.0);
    const minChange = Math.min(...changeValues);
    const maxChange = Math.max(...changeValues);
    const changeRange = Math.max(1e-9, maxChange - minChange);
    
    //console.log('N/B Wave Scalping - changeValues range:', { minChange, maxChange, changeRange });
    //console.log('First 5 changeValues:', changeValues.slice(0, 5));
    
    // 가격과 N/B Wave를 비슷한 스케일로 맞추기
    const priceScale = priceRange / 100; // 가격을 100 단위로 스케일링
    const nbScale = changeRange / 10; // N/B Wave를 10 단위로 스케일링
    const scaleRatio = priceScale / nbScale; // 스케일 비율
    
    // N/B Wave 라인 그리기 (파란색, 더 두껍게)
    ctx.strokeStyle = 'rgba(0, 209, 255, 1.0)'; // 파란색, 완전 불투명
    ctx.lineWidth = 6; // 더 두껍게
    ctx.setLineDash([10, 6]); // 더 큰 점선
    ctx.beginPath();
    
            zones.forEach((zone, index) => {
                // 가격 데이터 길이에 맞춰서 제한
            if (index < dataLength) {
                    const x = chartPadding + (index * stepX); // 좌측 패딩 추가
                // change 값을 가격 범위에 매핑 (가격과 비슷한 스케일)
                const changeValue = zone.change || 0.0;
                const scaledChange = minPrice + ((changeValue - minChange) * scaleRatio);
                const y = height - ((scaledChange - minPrice) / priceRange) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    });
    ctx.stroke();
    ctx.setLineDash([]); // 점선 해제
    
    // N/B Wave 포인트 그리기 (구역별 색상)
            zones.forEach((zone, index) => {
                // 가격 데이터 길이에 맞춰서 제한
            if (index < dataLength) {
                    const x = chartPadding + (index * stepX); // 좌측 패딩 추가
                const changeValue = zone.change || 0.0;
                const scaledChange = minPrice + ((changeValue - minChange) * scaleRatio);
                const y = height - ((scaledChange - minPrice) / priceRange) * height;
            
            // 구역별 색상
            if (zone.zone === 'BLUE') {
                ctx.fillStyle = 'rgba(0, 209, 255, 1.0)'; // 파란색, 완전 불투명
            } else if (zone.zone === 'ORANGE') {
                ctx.fillStyle = 'rgba(255, 183, 3, 1.0)'; // 주황색, 완전 불투명
            } else {
                ctx.fillStyle = 'rgba(128, 128, 128, 1.0)'; // 회색, 완전 불투명
            }
            
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI); // 포인트 크기 증가
            ctx.fill();
        }
    });
    
    // N/B Wave 정보 표시
    ctx.fillStyle = 'rgba(0, 209, 255, 0.9)';
    ctx.font = '12px Courier New';
    ctx.textAlign = 'right';
    ctx.fillText(`N/B Wave (${zones.length} zones)`, width - 10, 80);
    
    // NB Window 값 표시 (항상 랜덤 값을 사용하되, API 요청에 사용한 값을 표시)
    const nbWindow = (typeof window.currentNbWindow === 'number') ? window.currentNbWindow : (nbData.window || 50);
    const randomBitUsed = (typeof nbData.random_bit_used === 'number')
        ? nbData.random_bit_used
        : (5.5 + (nbWindow % 95) * 0.5);
    ctx.fillText(`NB Window: ${nbWindow} (${randomBitUsed.toFixed(1)})`, width - 10, 100);
    
    // 현재 N/B 값 표시
    if (zones.length > 0) {
        const currentZone = zones[zones.length - 1];
        const changeValue = currentZone.change || 0.0;
        const zone = currentZone.zone || 'UNKNOWN';
        ctx.fillText(`Current: ${changeValue.toFixed(3)} (${zone})`, width - 10, 120);
    }
}

// N/B Wave 값을 가격과 동일한 스케일로 스캘핑하는 라인 그리기
function drawNbWavePriceScalpingLine(ctx, nbData, width, height, stepX, dataLength, minPrice, maxPrice, priceRange, chartPadding) {
    const zones = nbData.zones;
    if (!zones || zones.length === 0) {
        //console.log('No zones data for N/B Wave Price Scalping Line');
        return;
    }
    
    //console.log('Drawing N/B Wave Price Scalping Line with', zones.length, 'zones');
    
    // N/B Wave change 값을 가격 범위에 직접 매핑 (가격과 동일한 스케일)
    const changeValues = zones.map(z => z.change || 0.0);
    const minChange = Math.min(...changeValues);
    const maxChange = Math.max(...changeValues);
    const changeRange = Math.max(1e-9, maxChange - minChange);
    
    //console.log('N/B Wave Price Scalping - changeValues range:', { minChange, maxChange, changeRange });
    //console.log('First 5 changeValues:', changeValues.slice(0, 5));
    
    // N/B Wave 값을 가격 범위에 직접 매핑
    const scaleRatio = priceRange / changeRange;
    
    // N/B Wave 가격 스캘핑 라인 그리기 (주황색, 점선)
    ctx.strokeStyle = 'rgba(255, 165, 0, 1.0)'; // 주황색, 완전 불투명
    ctx.lineWidth = 3; // 더 두꺼운 라인
    ctx.setLineDash([8, 4]); // 더 큰 점선
    ctx.beginPath();
    
            zones.forEach((zone, index) => {
                // 가격 데이터 길이에 맞춰서 제한
            if (index < dataLength) {
                    const x = chartPadding + (index * stepX); // 좌측 패딩 추가
                // change 값을 가격 범위에 직접 매핑
                const changeValue = zone.change || 0.0;
                const scaledPrice = minPrice + ((changeValue - minChange) * scaleRatio);
                const y = height - ((scaledPrice - minPrice) / priceRange) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    });
    ctx.stroke();
    ctx.setLineDash([]); // 점선 해제
    //console.log('N/B Wave Price Scalping Line drawn');
    
    // N/B Wave 가격 스캘핑 포인트 그리기
            zones.forEach((zone, index) => {
                // 가격 데이터 길이에 맞춰서 제한
            if (index < dataLength) {
                    const x = chartPadding + (index * stepX); // 좌측 패딩 추가
                const changeValue = zone.change || 0.0;
                const scaledPrice = minPrice + ((changeValue - minChange) * scaleRatio);
                const y = height - ((scaledPrice - minPrice) / priceRange) * height;
            
            // 구역별 색상으로 포인트 표시
            if (zone.zone === 'BLUE') {
                ctx.fillStyle = 'rgba(0, 209, 255, 1.0)'; // 파란색, 완전 불투명
            } else if (zone.zone === 'ORANGE') {
                ctx.fillStyle = 'rgba(255, 183, 3, 1.0)'; // 주황색, 완전 불투명
            } else {
                ctx.fillStyle = 'rgba(128, 128, 128, 1.0)'; // 회색, 완전 불투명
            }
            
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI); // 더 큰 포인트
            ctx.fill();
        }
    });
    
    // N/B Wave 가격 스캘핑 정보 표시
    ctx.fillStyle = 'rgba(255, 165, 0, 0.8)';
    ctx.font = '11px Courier New';
    ctx.textAlign = 'right';
    ctx.fillText(`N/B Price Scale`, width - 10, 140);
}

// N/B Wave 라인 그리기 (비활성화됨 - 모든 라인 제거)
function drawNbWaveLines(ctx, nbData, width, height, stepX, dataLength, minPrice, maxPrice, priceRange) {
    // 모든 N/B Wave 라인 제거됨
    return;
}

// N/B Wave 정보 표시
function drawNbWaveInfo(ctx, nbData, width, height) {
    const summary = nbData.summary;
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = 'bold 12px Courier New';
    ctx.textAlign = 'right';
    
    // 우측 하단에 N/B Wave 정보 표시
    let yPos = height - 80;
    
    if (summary.orange !== undefined) {
        ctx.fillStyle = 'rgba(255, 183, 3, 0.9)';
        ctx.fillText(`ORANGE: ${summary.orange}`, width - 10, yPos);
        yPos += 15;
    }
    
    if (summary.blue !== undefined) {
        ctx.fillStyle = 'rgba(0, 209, 255, 0.9)';
        ctx.fillText(`BLUE: ${summary.blue}`, width - 10, yPos);
        yPos += 15;
    }
    
    if (summary.current_price) {
        ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
        ctx.fillText(`NB Price: ${summary.current_price.toLocaleString()}`, width - 10, yPos);
    }
}

// 분봉 카드 초기화
async function initializeTimeframeCards() {
    try {
        // HTML 생성 및 삽입
        const container = document.getElementById('timeframe-cards-container');
        if (container && window.timeframeCards) {
            // 좌측 패널 버튼과 ID 충돌을 피하기 위해 내부 자동 순회 버튼은 숨김
            container.innerHTML = window.timeframeCards.constructor.generateHTML('timeframe-cards-container', false);
            window.timeframeCards.init('timeframe-cards-container', {
                defaultTimeframe: 'minute10',
                onTimeframeChange: (timeframe) => {
                    //console.log('Timeframe changed to:', timeframe);
                    // 분봉 변경 시 차트 업데이트
                    updateChartsForTimeframe(timeframe);
                }
            });
        } else {
            console.error('Timeframe Cards container or module not found');
        }
    } catch (error) {
        console.error('Failed to initialize Timeframe Cards:', error);
    }
}

// 분봉 변경 시 차트 업데이트
async function updateChartsForTimeframe(timeframe) {
    try {
        //console.log('Updating charts for timeframe:', timeframe);
        
        // 랜덤 NB Window 값 생성 (5.5 ~ 100 범위) - 항상 랜덤 사용, 표시와 동기화 위해 전역 보관
        const randomNbWindow = Math.floor(5.5 + Math.random() * 94.5);
        window.currentNbWindow = randomNbWindow;
        
        // 트레이딩 차트와 NB Wave 데이터 동시 가져오기
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        const [tradeRes, nbRes] = await Promise.all([
            fetch(`/api/trading-data?timeframe=${timeframe}&coin=${selectedCoin}`),
            fetch(`http://localhost:5057/api/nb-wave?timeframe=${timeframe}&bars=300&window=${randomNbWindow}&coin=${selectedCoin}`)
        ]);
        
        if (tradeRes.ok && nbRes.ok) {
            const tradeData = await tradeRes.json();
            const nbData = await nbRes.json();
            
            // API에서 받은 timeframe으로 current-timeframe 업데이트
            if (nbData.timeframe) {
                updateCurrentTimeframeFromAPI(nbData.timeframe);
            }
            
            // 가격 차트 업데이트 (N/B Wave 오버레이 포함)
            if (tradeData.chart_data && tradeData.chart_data.prices) {
                drawPriceChartFromData(tradeData.chart_data, nbData);
            }
            
            // 별도 NB Wave 차트 업데이트 (이미 전역 데이터 사용 중)
            if (nbData.zones && nbData.zones.length > 0 && window.nbWavePanel) {
                window.nbWavePanel.drawChart(nbData);
            }
            
            // 분봉별 구역 업데이트
            if (typeof updateTimeframeZones === 'function') {
                // 전역 데이터에 nbData 저장 후 업데이트
                window.sharedNbWaveData = nbData;
                updateTimeframeZones();
            }
            
            // 차트 마지막 구역과 현재 구역 동기화
            if (window.zoneStrengthManager) {
                window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbData);
            } else {
                console.warn('⚠️ Zone Strength Manager not loaded yet');
            }
        }
        
    } catch (error) {
        console.error('Failed to update charts for timeframe:', error);
    }
}

// NB Wave Panel 초기화
async function initializeNbWavePanel() {
    try {
        // NB Wave Panel 모듈이 로드되지 않았다면 로드
        if (!window.nbWavePanel) {
            //console.log('Loading NB Wave Panel module...');
            const NbWavePanel = (await import('./modules/trading/nb-wave-panel.js')).default;
            window.nbWavePanel = new NbWavePanel();
        }
        
        // HTML 생성 및 삽입
        const container = document.getElementById('nb-wave-container');
        if (container && window.nbWavePanel) {
            container.innerHTML = window.nbWavePanel.constructor.generateHTML();
            window.nbWavePanel.init();
            //console.log('✅ NB Wave Panel initialized successfully');
        } else {
            console.error('❌ NB Wave Panel container or module not found');
        }
    } catch (error) {
        console.error('Failed to initialize NB Wave Panel:', error);
    }
}

// 트레이딩 모듈 초기화(데이터 로드 및 차트 그리기)
async function initializeTradingCharts() {
    try {
        //console.log('Initializing trading charts...');
        
        // 분봉 카드 초기화 (Central Hub에서도 활성화)
        await initializeTimeframeCards();
        
        // NB Wave Panel 초기화
        await initializeNbWavePanel();
        
        // 설정에서 기본 시간대 가져오기
        const defaultTimeframe = window.settingsManager && typeof window.settingsManager.getSetting === 'function' ?
            window.settingsManager.getSetting('chart.defaultTimeframe') : 'minute1';
        
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        const [tradeRes, nbRes] = await Promise.all([
            fetch(`/api/trading-data?timeframe=${defaultTimeframe}&coin=${selectedCoin}`),
            fetch(`http://localhost:5057/api/nb-wave?timeframe=${defaultTimeframe}&bars=300&coin=${selectedCoin}`)
        ]);
        
        if (!tradeRes.ok) {
            throw new Error(`Trading data API error: ${tradeRes.status}`);
        }
        if (!nbRes.ok) {
            throw new Error(`NB wave API error: ${nbRes.status}`);
        }
        
        const tradeData = await tradeRes.json();
        const nbData = await nbRes.json();
        
        //console.log('Trading data received:', tradeData);
        //console.log('NB wave data received:', nbData);

        // 전역 데이터에 nbData 저장
        window.sharedNbWaveData = nbData;

        // API에서 받은 timeframe으로 current-timeframe 업데이트
        if (nbData.timeframe) {
            updateCurrentTimeframeFromAPI(nbData.timeframe);
        }

        // auto 모드 처리
        if (defaultTimeframe === 'auto') {
            if (tradeData.mode === 'auto' && nbData.mode === 'auto') {
                // 모든 시간대의 차트를 순차적으로 표시
                await displayAutoCharts(tradeData.timeframes, nbData.timeframes);
            }
        } else {
            // 단일 시간대 차트 그리기 (N/B Wave 오버레이 포함)
            if (tradeData.chart_data && tradeData.chart_data.prices && Array.isArray(tradeData.chart_data.prices) && tradeData.chart_data.prices.length > 0) {
                drawPriceChartFromData(tradeData.chart_data, nbData);
                
                // 현재가 업데이트 (안전한 검증 후)
                updateTradingCurrentPrice(tradeData.chart_data.prices);
                

            } else {
                console.error('No valid chart data in trading response:', tradeData.chart_data);
            }
            
            // 별도 NB Wave 차트는 선택적으로 표시
            if (nbData.zones && nbData.zones.length > 0) {
                if (window.nbWavePanel) {
                    window.nbWavePanel.drawChart(nbData);
                } else {
                    //console.log('NB Wave Panel not initialized');
                }
                
                // 현재 구역 업데이트
                updateTradingCurrentZone(nbData);
            } else {
                console.error('No zones data in NB wave response');
                setDefaultZoneDisplay();
            }
        }
        
        //console.log('Trading charts initialized successfully');
        
        // 실시간 현재가 업데이트 시작 (30초마다)
        startTradingPriceUpdate();
        
    } catch (e) {
        console.error('Failed to initialize trading charts:', e);
        
        // 에러 시 fallback 차트 그리기
        drawFallbackCharts();
    }
}

// 실시간 현재가 업데이트 시작
function startTradingPriceUpdate() {
    // 기존 타이머가 있다면 제거
    if (window.tradingPriceTimer) {
        clearInterval(window.tradingPriceTimer);
    }
    
    // 30초마다 현재가 및 구역 업데이트
    window.tradingPriceTimer = setInterval(async () => {
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        try {
            // 가격 데이터만 업데이트 (N/B Wave는 이미 전역에 저장됨)
            const priceResponse = await fetch(`/api/trading-data?coin=${selectedCoin}`);
            const priceData = await priceResponse.json();
            
            // 전역 저장된 N/B Wave 데이터 사용
            const nbData = window.sharedNbWaveData;
            
            if (priceData.status === 'success' && priceData.data && Array.isArray(priceData.data) && priceData.data.length > 0) {
                updateTradingCurrentPrice(priceData.data);
            } else {
                console.warn('⚠️ Invalid price data in background update:', priceData);
            }
            
            if (nbData.status === 'success' && nbData.zones && nbData.zones.length > 0) {
                // 마지막 점의 가격 정보 추가
                if (priceData.status === 'success' && priceData.data && priceData.data.length > 0) {
                    const lastPrice = priceData.data[priceData.data.length - 1].close;
                    nbData.last_point_price = lastPrice;
                }
                updateTradingCurrentZone(nbData);
            }
        } catch (error) {
            console.error('Failed to update trading data:', error);
        }
    }, 30000); // 30초마다
    
    //console.log('🔄 Trading price update timer started (30s interval)');
}

// Trading Dashboard 현재가 업데이트
function updateTradingCurrentPrice(priceData) {
    // 데이터 유효성 검사
    if (!priceData || !Array.isArray(priceData) || priceData.length === 0) {
        console.warn('⚠️ Invalid priceData:', priceData);
        return;
    }
    
    // priceData가 숫자 배열인지 객체 배열인지 확인
    const lastPriceData = priceData[priceData.length - 1];
    let currentPrice, previousPrice;
    
    if (typeof lastPriceData === 'number') {
        // 숫자 배열인 경우
        currentPrice = lastPriceData;
        previousPrice = priceData.length > 1 ? priceData[priceData.length - 2] : currentPrice;
    } else if (lastPriceData && typeof lastPriceData.close === 'number') {
        // 객체 배열인 경우
        currentPrice = lastPriceData.close;
        previousPrice = priceData.length > 1 && priceData[priceData.length - 2] && typeof priceData[priceData.length - 2].close === 'number' 
            ? priceData[priceData.length - 2].close 
            : currentPrice;
    } else {
        console.warn('⚠️ Invalid last price data:', lastPriceData);
        return;
    }
    
    const currentPriceElement = document.getElementById('trading-current-price');
    const priceChangeElement = document.getElementById('trading-price-change');
    const rightCurrentPriceElement = document.getElementById('right-trading-current-price');
    const rightPriceChangeElement = document.getElementById('right-trading-price-change');
    
    if (currentPriceElement && priceChangeElement) {
        const priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;
        
        // 현재가 업데이트
        currentPriceElement.textContent = `₩${currentPrice.toLocaleString()}`;
        
        // 가격 변화율 업데이트
        const changeText = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`;
        priceChangeElement.textContent = changeText;
        
        // 색상 업데이트
        priceChangeElement.className = `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        
        // 우측 패널 업데이트
        if (rightCurrentPriceElement) {
            rightCurrentPriceElement.textContent = `₩${currentPrice.toLocaleString()}`;
        }
        if (rightPriceChangeElement) {
            rightPriceChangeElement.textContent = changeText;
            rightPriceChangeElement.className = `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        }
        
        //console.log('💰 Trading current price updated:', currentPrice, 'Change:', priceChange.toFixed(2) + '%');
        
        // Trading Dashboard 출력 데이터를 전역으로 저장
        window.sharedTradingDashboardData = {
            currentPrice: currentPrice,
            priceChange: priceChange,
            priceChangePercent: priceChange,
            last_update: new Date().toISOString(),
            timestamp: new Date().getTime()
        };
        
        //console.log('💾 Trading Dashboard data saved to global storage:', window.sharedTradingDashboardData);
    }
}

// ===== Zone Strength Management =====
// 이 함수들은 modules/trading/zone-strength-manager.js에서 관리됩니다.
// zone-strength-manager 모듈이 로드되면 자동으로 전역 함수로 등록됩니다.

// Trading Dashboard 현재 구역 업데이트 (차트 마지막 구역과 동기화)
// 이 함수는 modules/trading/zone-strength-manager.js의 updateTradingCurrentZoneFromChart()로 대체됨
function updateTradingCurrentZoneFromChart(nbData) {
    if (window.zoneStrengthManager) {
        window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbData);
    } else {
        console.warn('⚠️ Zone Strength Manager not loaded yet');
    }
}

// Trading Dashboard 현재 구역 업데이트 (기존 함수 - 호환성 유지)
// 이 함수는 modules/trading/zone-strength-manager.js의 updateTradingCurrentZone()로 대체됨
function updateTradingCurrentZone(nbData) {
    if (window.zoneStrengthManager) {
        window.zoneStrengthManager.updateTradingCurrentZone(nbData);
    } else {
        console.warn('⚠️ Zone Strength Manager not loaded yet');
    }
}

// 기본 구역 표시 설정
// 이 함수는 modules/trading/zone-strength-manager.js의 setDefaultZoneDisplay()로 대체됨
function setDefaultZoneDisplay() {
    if (window.zoneStrengthManager) {
        window.zoneStrengthManager.setDefaultZoneDisplay();
    } else {
        console.warn('⚠️ Zone Strength Manager not loaded yet');
    }
}

// 우측 패널 로딩 상태 제어 함수
function showRightPanelLoading() {
    const loadingElement = document.getElementById('right-panel-loading');
    const contentElement = document.getElementById('right-panel-content');
    
    if (loadingElement) {
        loadingElement.style.display = 'block';
    }
    if (contentElement) {
        contentElement.style.display = 'none';
    }
}

function hideRightPanelLoading() {
    const loadingElement = document.getElementById('right-panel-loading');
    const contentElement = document.getElementById('right-panel-content');
    
    if (loadingElement) {
        loadingElement.style.display = 'none';
    }
    if (contentElement) {
        contentElement.style.display = 'block';
        contentElement.classList.add('right-panel-content');
    }
}

// 우측 패널 Trading Dashboard Info 독립 업데이트 함수 (저장된 데이터 사용)
async function updateRightPanelTradingInfo() {
    try {
        // 로딩 시작
        showRightPanelLoading();
        
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        
        // 저장된 Trading Dashboard 데이터 사용 (API 호출 대신)
        const tradingData = window.sharedTradingDashboardData;
        const mainChartData = window.sharedMainChartData;
        const nbWaveData = window.sharedNbWaveData;
        
        if (!tradingData || !mainChartData || !nbWaveData) {
            //console.log('⚠️ No shared trading data available');
            hideRightPanelLoading();
            return;
        }
        
        const currentPrice = tradingData.currentPrice || 0;
        const priceChange = tradingData.priceChangePercent || 0;
        const currentZone = tradingData.currentZone || 'Neutral Zone';
        const zoneStrength = tradingData.zoneStrength || 0;
        
        //console.log('📊 Using shared trading data:', { currentPrice, priceChange, currentZone, zoneStrength });

        const rightCurrentPriceElement = document.getElementById('right-trading-current-price');
        const rightPriceChangeElement = document.getElementById('right-trading-price-change');

        if (currentPrice <= 0 || isNaN(currentPrice)) {
            //console.log('⚠️ No/invalid price data from shared data');
            hideRightPanelLoading();
            return;
        }
        
        if (rightCurrentPriceElement) {
            rightCurrentPriceElement.textContent = `₩${currentPrice.toLocaleString()}`;
        }
        if (rightPriceChangeElement) {
            const changeText = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%`;
            rightPriceChangeElement.textContent = changeText;
            rightPriceChangeElement.className = `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        }
        
        // 현재 구역 및 강도 업데이트 (저장된 데이터 사용)
        const rightCurrentZoneElement = document.getElementById('right-trading-current-zone');
        const rightZoneStrengthElement = document.getElementById('right-trading-zone-strength');
        
        // zone-strength-manager를 사용하여 우측 패널 구역 업데이트
        if (window.zoneStrengthManager && nbWaveData) {
            // zone-strength-manager가 우측 패널도 함께 업데이트함
            window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbWaveData);
        } else {
            // fallback: 직접 업데이트
            if (rightCurrentZoneElement) {
                rightCurrentZoneElement.textContent = currentZone;
                rightCurrentZoneElement.className = 'current-zone';
                if (currentZone === 'Blue Zone') {
                    rightCurrentZoneElement.classList.add('zone-blue');
                } else if (currentZone === 'Orange Zone') {
                    rightCurrentZoneElement.classList.add('zone-orange');
                } else {
                    rightCurrentZoneElement.classList.add('zone-neutral');
                }
            }
            
            if (rightZoneStrengthElement) {
                rightZoneStrengthElement.textContent = `강도: ${zoneStrength}`;
            }
        }
        
        //console.log('🔄 Right panel trading info updated using shared data:', { currentZone, zoneStrength });
        
        // 로딩 완료
        hideRightPanelLoading();
        
    } catch (error) {
        console.error('❌ Error updating right panel trading info:', error);
        hideRightPanelLoading();
    }
}

// Selected Coin Status 업데이트 - asset_display.js로 이동됨
// 이 함수는 modules/asset/asset_display.js에서 관리됩니다



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
    //console.log(`Displaying charts for ${timeframe}`);
    
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
    

}

// Fallback 차트 (데이터 없을 때)
function drawFallbackCharts() {
    //console.log('Drawing fallback charts...');
    
    // 가격 차트 fallback
    const priceCanvas = document.getElementById('trading-chart');
    if (priceCanvas) {
        const ctx = priceCanvas.getContext('2d');
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, priceCanvas.width, priceCanvas.height);
        
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.font = '16px Courier New';
        ctx.textAlign = 'center';
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        ctx.fillText(`${selectedCoin}/KRW Price Chart - Data Loading...`, priceCanvas.width/2, priceCanvas.height/2);
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
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        ctx.fillText(`${selectedCoin}/KRW NB Wave Chart - Data Loading...`, nbCanvas.width/2, nbCanvas.height/2);
    }
}

// 분봉별 구역 계산 및 표시 함수
function updateTimeframeZones() {
    const container = document.getElementById('timeframe-zones-container');
    if (!container) {
        //console.log('⚠️ Timeframe zones container not found');
        return;
    }
    
    // 저장된 N/B Wave 데이터 사용
    const nbWaveData = window.sharedNbWaveData;
    if (!nbWaveData || !nbWaveData.zones) {
        //console.log('⚠️ No N/B Wave data available for timeframe zones');
        return;
    }
    
    const zones = nbWaveData.zones;
    const labels = nbWaveData.labels || [];
    const totalZones = zones.length;
    
    // 구역별 카운트 계산 (과반수 계산용)
    const zoneCounts = {
        BLUE: 0,
        ORANGE: 0,
        NEUTRAL: 0
    };
    
    zones.forEach(zone => {
        const zoneType = zone.zone || 'NEUTRAL';
        zoneCounts[zoneType]++;
    });
    
    // 과반수 계산
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
    
    // 구역별 강도 계산
    const zoneStrengths = {
        BLUE: 0,
        ORANGE: 0,
        NEUTRAL: 0
    };
    
    zones.forEach(zone => {
        const zoneType = zone.zone || 'NEUTRAL';
        zoneStrengths[zoneType] += zone.strength || 0;
    });
    
    // 평균 강도 계산
    Object.keys(zoneStrengths).forEach(zoneType => {
        if (zoneCounts[zoneType] > 0) {
            zoneStrengths[zoneType] = zoneStrengths[zoneType] / zoneCounts[zoneType];
        }
    });
    
    // 실제 차트의 strength 값을 전역 변수로 저장 (zone-strength-manager에서 사용)
    // 차트에서 직접 계산된 strength 값 사용
    const chartActualStrength = window.currentZoneStrength || 0;
    
    window.chartZoneStrengths = {
        BLUE: chartActualStrength,
        ORANGE: chartActualStrength,
        NEUTRAL: chartActualStrength
    };
    window.chartCurrentZone = currentZone;
    
    /**console.log('📊 Chart zone strengths saved to global:', {
        chartCurrentZone: window.chartCurrentZone,
        chartZoneStrengths: window.chartZoneStrengths
    });
     */
    // 각 분봉별 구역 계산
    const timeframeData = [
        { name: '1m', zones: zones.slice(Math.max(0, zones.length - 60)) },
        { name: '3m', zones: zones.slice(Math.max(0, zones.length - 180)) },
        { name: '5m', zones: zones.slice(Math.max(0, zones.length - 300)) },
        { name: '10m', zones: zones.slice(Math.max(0, zones.length - 300)) },
        { name: '15m', zones: zones.slice(Math.max(0, zones.length - 300)) },
        { name: '30m', zones: zones.slice(Math.max(0, zones.length - 300)) },
        { name: '1h', zones: zones.slice(Math.max(0, zones.length - 300)) },
        { name: '1D', zones }
    ];
    
    // 각 분봉별로 구역 계산 (강도 정보 포함)
    window.timeframeResults = timeframeData.map(timeframe => {
        const zoneCounts = {
            BLUE: 0,
            ORANGE: 0,
            NEUTRAL: 0
        };
        
        const zoneStrengths = {
            BLUE: 0,
            ORANGE: 0,
            NEUTRAL: 0
        };
        
        timeframe.zones.forEach(zone => {
            const zoneType = zone.zone || 'NEUTRAL';
            zoneCounts[zoneType]++;
            zoneStrengths[zoneType] += zone.strength || 0;
        });
        
        const totalZones = timeframe.zones.length;
        const majority = Math.ceil(totalZones / 2);
        let dominantZone = 'NEUTRAL';
        
        if (zoneCounts.BLUE >= majority) {
            dominantZone = 'BLUE';
        } else if (zoneCounts.ORANGE >= majority) {
            dominantZone = 'ORANGE';
        }
        
        // 평균 강도 계산
        const avgStrengths = {};
        Object.keys(zoneStrengths).forEach(zoneType => {
            if (zoneCounts[zoneType] > 0) {
                avgStrengths[zoneType] = zoneStrengths[zoneType] / zoneCounts[zoneType];
            } else {
                avgStrengths[zoneType] = 0;
            }
        });
        
        // 마지막 구역 정보
        const lastZone = timeframe.zones.length > 0 ? timeframe.zones[timeframe.zones.length - 1] : null;
        const lastZoneType = lastZone ? lastZone.zone : 'NEUTRAL';
        const lastZoneStrength = lastZone ? (lastZone.strength || 0) * 100 : 0;
        
        return {
            name: timeframe.name,
            dominantZone: dominantZone,
            zoneCounts: zoneCounts,
            totalZones: totalZones,
            avgStrengths: avgStrengths,
            lastZone: lastZoneType,
            lastZoneStrength: Math.round(lastZoneStrength),
            dominantZoneStrength: Math.round(avgStrengths[dominantZone] * 100)
        };
    });
    
    // API timeframe 정보 가져오기
    const apiTimeframeDisplay = window.currentDisplayTimeframe || '1D';
    
    // 각 분봉별 개별 카드 생성 - 상세 정보 포함
    const individualZonesHtml = window.timeframeResults.map(result => {
        const isBlue = result.dominantZone === 'BLUE';
        const isOrange = result.dominantZone === 'ORANGE';
        
        // 각 분봉의 구역 정보 표시
        const blueCount = result.zoneCounts.BLUE;
        const orangeCount = result.zoneCounts.ORANGE;
        const neutralCount = result.zoneCounts.NEUTRAL;
        const blueStrength = Math.round(result.avgStrengths.BLUE * 100);
        const orangeStrength = Math.round(result.avgStrengths.ORANGE * 100);
        const neutralStrength = Math.round(result.avgStrengths.NEUTRAL * 100);
        
        // 마지막 구역 정보
        const lastZoneInfo = result.lastZone === 'BLUE' ? '🔵' : result.lastZone === 'ORANGE' ? '🟠' : '⚪';
        const lastZoneStrength = result.lastZoneStrength;
        
        return `
            <div class="timeframe-status-card ${isBlue ? 'blue-zone' : isOrange ? 'orange-zone' : 'neutral-zone'}" 
                 id="timeframe-card-${result.name}" 
                 data-timeframe="${result.name}"
                 onclick="selectTimeframe('${result.name}', '${result.dominantZone}', ${result.zoneCounts[result.dominantZone]}, ${result.totalZones})">
                <div class="timeframe-header">
                    <span class="timeframe-name">${result.name}</span>
                    <span class="timeframe-zone">${lastZoneInfo} ${result.lastZone}</span>
                </div>
                <div class="timeframe-content">
                    <span class="timeframe-strength">${lastZoneStrength}%</span>
                    <div class="timeframe-breakdown">
                        <span class="blue-count">🔵 ${blueCount}</span>
                        <span class="orange-count">🟠 ${orangeCount}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    // 차트의 마지막 구역 정보 가져오기
    let chartLastZone = 'NEUTRAL';
    let chartLastZoneStrengthValue = 0;
    let chartLastZoneTimeframe = apiTimeframeDisplay;
    
    if (nbWaveData && nbWaveData.zones && nbWaveData.zones.length > 0) {
        const lastZone = nbWaveData.zones[nbWaveData.zones.length - 1];
        if (lastZone && lastZone.zone) {
            chartLastZone = lastZone.zone;
            chartLastZoneStrengthValue = lastZone.strength ? Math.round(lastZone.strength * 100) : 0;
        }
    }
    
    // 차트의 strength 값을 전역 변수로 저장 (zone-strength-manager에서 사용)
    window.chartLastZoneStrength = chartLastZoneStrengthValue;
    window.chartLastZone = chartLastZone;
    
    /** console.log('📊 Chart strength values saved to global:', {
        chartLastZone: window.chartLastZone,
        chartLastZoneStrength: window.chartLastZoneStrength
    });
	*/
    
    // HTML 생성 - 차트 마지막 구역과 동기화
    const zonesHtml = `
        <div class="timeframe-zone-summary">
            <div class="current-zone-display ${chartLastZone.toLowerCase()}-zone" id="current-zone-display" onclick="selectCurrentZone()">
                <h4>Current Zone: ${chartLastZone} (${chartLastZoneTimeframe})</h4>
                <div class="zone-stats">
                    <span class="zone-count">Chart Last Zone</span>
                    <span class="zone-strength">Strength: ${chartLastZoneStrengthValue}%</span>
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
        
        <div class="timeframe-status-grid">
            ${individualZonesHtml}
        </div>
    `;
    
    // 기존 카드들 제거
    const existingZoneBreakdown = document.querySelector('.timeframe-zone-breakdown');
    if (existingZoneBreakdown) {
        existingZoneBreakdown.remove();
    }
    
    const existingTimeframeGrid = document.querySelector('.timeframe-status-grid');
    if (existingTimeframeGrid) {
        existingTimeframeGrid.remove();
    }
    
    // timeframe-cards-container도 미니 카드 스타일로 업데이트
    const timeframeCardsContainer = document.getElementById('timeframe-cards-container');
    if (timeframeCardsContainer && window.timeframeResults) {
        // 기존 헤더와 컨트롤은 유지하고 카드들만 업데이트
        const existingHeader = timeframeCardsContainer.querySelector('.timeframe-header');
        const existingControls = timeframeCardsContainer.querySelector('.timeframe-controls');
        
        const timeframeCardsHtml = window.timeframeResults.map(result => {
            const isBlue = result.lastZone === 'BLUE';
            const isOrange = result.lastZone === 'ORANGE';
            const lastZoneInfo = result.lastZone === 'BLUE' ? '🔵' : result.lastZone === 'ORANGE' ? '🟠' : '⚪';
            const lastZoneStrength = result.lastZoneStrength;
            const blueCount = result.zoneCounts.BLUE;
            const orangeCount = result.zoneCounts.ORANGE;
            
            return `
                <div class="timeframe-status-card ${isBlue ? 'blue-zone' : isOrange ? 'orange-zone' : 'neutral-zone'}" 
                     id="timeframe-card-${result.name}" 
                     data-timeframe="${result.name}"
                     onclick="selectTimeframe('${result.name}', '${result.dominantZone}', ${result.zoneCounts[result.dominantZone]}, ${result.totalZones})">
                    <div class="timeframe-header">
                        <span class="timeframe-name">${result.name}</span>
                        <span class="timeframe-zone">${lastZoneInfo} ${result.lastZone}</span>
                    </div>
                    <div class="timeframe-content">
                        <span class="timeframe-strength">${lastZoneStrength}%</span>
                        <div class="timeframe-breakdown">
                            <span class="blue-count">🔵 ${blueCount}</span>
                            <span class="orange-count">🟠 ${orangeCount}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        // Bootstrap 스타일로 카드들 교체
        timeframeCardsContainer.innerHTML = `
            <div class="card border-secondary rounded-3 p-3">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="text-white mb-0">
                        <i class="fas fa-clock me-2"></i>분봉 선택
                    </h5>
                    <div class="d-flex align-items-center gap-2">
                        <span id="currentTimeframe" class="badge bg-primary fs-6">${window.currentDisplayTimeframe || '1h'}</span>
                        <button id="btnAutoRotate" class="btn btn-sm btn-outline-warning">
                            <i class="fas fa-pause me-1"></i>순회 중지
                        </button>
                    </div>
                </div>
                <div class="row g-2" style="display:none">
                    ${window.timeframeResults.map(result => {
                        return `
                            <div class="col-4">
                                <div class="timeframe-status-card h-100" 
                                     id="timeframe-card-${result.name}" 
                                     data-timeframe="${result.name}"
                                     onclick="selectTimeframe('${result.name}', '${result.dominantZone}', ${result.zoneCounts[result.dominantZone]}, ${result.totalZones})">
                                    <div class="d-flex flex-column justify-content-center h-100">
                                        <div class="text-center">
                                            <div class="timeframe-name fw-bold">${result.name}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        // 렌더 후 배지 동기화 (혹시 window.currentDisplayTimeframe가 늦게 설정된 경우 대비)
        const badgeEl = document.getElementById('currentTimeframe');
        if (badgeEl) {
            badgeEl.textContent = window.currentDisplayTimeframe || badgeEl.textContent || '1h';
        }
    }
    
    // 전역 변수 업데이트
    window.currentMajorityZone = currentZone;
    window.currentMajorityZoneCount = currentZoneCount;
    window.currentMajorityZoneStrength = zoneStrengths[currentZone];
    
    /** console.log('💾 Timeframe zones updated:', {
        currentZone,
        currentZoneCount,
        totalZones,
        zoneCounts,
        zoneStrengths
    });
	*/
    
    // 우측 패널 현재 분봉 표시 업데이트
    if (typeof updateCurrentTimeframeDisplay === 'function') {
        updateCurrentTimeframeDisplay(window.timeframeResults);
    }
}



// 분봉 선택 함수
window.selectTimeframe = function(timeframe, dominantZone, zoneCount, totalZones, isAutoRotation = false) {
    //console.log(`🎯 Selected timeframe: ${timeframe}, Zone: ${dominantZone}, Count: ${zoneCount}/${totalZones}, Auto: ${isAutoRotation}`);
    
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
    updateCurrentZoneDisplay(timeframe, dominantZone, zoneCount, totalZones);
    
    // 차트의 분봉 표시 업데이트 (API timeframe 우선 사용)
    const timeframeDisplay = document.getElementById('current-timeframe');
    if (timeframeDisplay) {
        // API에서 받은 timeframe이 있으면 우선 사용, 없으면 선택된 timeframe 사용
        const apiTimeframe = window.currentApiTimeframe || timeframe;
        const displayTimeframe = convertTimeframeToDisplay(apiTimeframe);
        timeframeDisplay.textContent = `Current: ${displayTimeframe}`;
        //console.log(`🔄 Updated current-timeframe in selectTimeframe: ${displayTimeframe} (API: ${apiTimeframe}, Selected: ${timeframe})`);
    }
    
    // 전역 변수에 선택된 분봉 저장
    window.selectedTimeframe = timeframe;
    window.selectedTimeframeZone = dominantZone;
    window.selectedTimeframeCount = zoneCount;
    window.selectedTimeframeTotal = totalZones;
    
    // 새로운 히스토리 카드 시스템 사용
    if (typeof window.addHistoryItem === 'function') {
        const currentResult = window.timeframeResults?.find(result => result.name === timeframe);
        const strength = currentResult ? currentResult.lastZoneStrength : 0;
        window.addHistoryItem(timeframe, dominantZone, strength);
    }
    
    // 현재 timeframe display와 연동하여 히스토리에 추가
    if (typeof window.addCurrentTimeframeToHistory === 'function') {
        // 약간의 지연 후 현재 display 값을 히스토리에 추가
        setTimeout(() => {
            window.addCurrentTimeframeToHistory();
        }, 100);
    }
    
    //console.log(`✅ Timeframe ${timeframe} selected and connected with zone data`);
};

// Current Zone 선택 함수
window.selectCurrentZone = function() {
    //console.log(`🎯 Current Zone clicked`);
    
    // 현재 선택된 분봉이 있으면 해당 분봉의 구역으로 설정
    if (window.selectedTimeframe && window.selectedTimeframeZone) {
        updateCurrentZoneDisplay(
            window.selectedTimeframe, 
            window.selectedTimeframeZone, 
            window.selectedTimeframeCount, 
            window.selectedTimeframeTotal
        );
        //console.log(`✅ Current Zone synchronized with ${window.selectedTimeframe}`);
    } else {
        // 선택된 분봉이 없으면 안내 메시지 표시
        const currentZoneDisplay = document.getElementById('current-zone-display');
        if (currentZoneDisplay) {
            const currentZone = currentZoneDisplay.querySelector('h4');
            if (currentZone) {
                currentZone.textContent = `Current Zone: ${currentZone}`;
            }
        }
        //console.log(`ℹ️ No timeframe selected - please select a timeframe card`);
    }
};

// Timeframe 변환 함수
function convertTimeframeToDisplay(timeframe) {
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

// API에서 받은 timeframe으로 current-timeframe 업데이트 함수
function updateCurrentTimeframeFromAPI(apiTimeframe) {
    //console.log('🔄 Updating current-timeframe from API:', apiTimeframe);
    
    const timeframeDisplay = document.getElementById('current-timeframe');
    const timeframeBadge = document.getElementById('currentTimeframe');
    if (timeframeDisplay) {
        const displayTimeframe = convertTimeframeToDisplay(apiTimeframe);
        timeframeDisplay.textContent = `Current: ${displayTimeframe}`;
        //console.log(`✅ Updated current-timeframe to: Current: ${displayTimeframe} (from API: ${apiTimeframe})`);
        
        // 전역 변수에도 저장
        window.currentApiTimeframe = apiTimeframe;
        window.currentDisplayTimeframe = displayTimeframe;
    } else {
        console.warn('⚠️ current-timeframe element not found!');
    }

    // timeframe-cards-container 상단 배지(id="currentTimeframe") 동기화
    if (timeframeBadge) {
        const displayTimeframe = convertTimeframeToDisplay(apiTimeframe);
        timeframeBadge.textContent = displayTimeframe;
        // 자동화 모드에서도 좌측 패널 로거 분봉을 즉시 동기화
        try { window.leftPanelTradeLogger && window.leftPanelTradeLogger.setCurrentTimeframe(displayTimeframe); } catch(_) { }
    }
}

// 자동 순회 분봉 목록 (전역 변수)
window.timeframeRotationList = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '1D'];
window.currentRotationIndex = 0;

// 자동 순회 표시 관리 함수들
function clearAutoRotationIndicators() {
    const allCards = document.querySelectorAll('.timeframe-status-card');
    allCards.forEach(card => {
        card.classList.remove('auto-rotating');
        const autoIndicator = card.querySelector('.auto-rotation-indicator');
        if (autoIndicator) {
            autoIndicator.remove();
        }
    });
}

function setAutoRotationIndicator(timeframe) {
    // 모든 자동 순회 표시 제거
    clearAutoRotationIndicators();
    
    // 현재 순회 중인 분봉에 표시 추가
    const currentCard = document.getElementById(`timeframe-card-${timeframe}`);
    if (currentCard) {
        currentCard.classList.add('auto-rotating');
        
        const autoIndicator = document.createElement('div');
        autoIndicator.className = 'auto-rotation-indicator';
        autoIndicator.textContent = 'AUTO';
        currentCard.appendChild(autoIndicator);
        
        //console.log('🔄 Auto-rotation indicator set for:', timeframe);
    }
}

// 자동 순회 함수
function startTimeframeAutoRotation() {
    // 기존 타이머가 있으면 제거
    if (window.timeframeRotationTimer) {
        clearInterval(window.timeframeRotationTimer);
    }
    
    // 자동 순회 시작 시 히스토리 초기화
    //console.log('🔄 Starting new auto-rotation cycle, clearing history...');
    if (typeof window.clearHistory === 'function') {
        window.clearHistory();
    }
    
    // 3초마다 분봉 순회
    window.timeframeRotationTimer = setInterval(() => {
        const timeframeName = window.timeframeRotationList[window.currentRotationIndex];
        const timeframeResult = window.timeframeResults.find(result => result.name === timeframeName);
        
        if (timeframeResult) {
            // 자동 순회 표시 업데이트
            setAutoRotationIndicator(timeframeName);
            
            selectTimeframe(
                timeframeResult.name,
                timeframeResult.dominantZone,
                timeframeResult.zoneCounts[timeframeResult.dominantZone],
                timeframeResult.totalZones,
                true // 자동 순회 플래그
            );
        }
        
        // 다음 인덱스로 이동
        window.currentRotationIndex = (window.currentRotationIndex + 1) % window.timeframeRotationList.length;
        
        // 한 사이클이 완료되면 히스토리 초기화
        if (window.currentRotationIndex === 0) {
            //console.log('🔄 Auto-rotation cycle completed, resetting history...');
            if (typeof window.clearHistory === 'function') {
                window.clearHistory();
            }
        }
    }, 3000);
    
    // 버튼 상태 업데이트
    updateAutoRotationButtonState(true);
    
    //console.log('🔄 Timeframe auto-rotation started');
}

// 자동 순회 정지 함수
function stopTimeframeAutoRotation() {
    if (window.timeframeRotationTimer) {
        clearInterval(window.timeframeRotationTimer);
        window.timeframeRotationTimer = null;
        
        // 자동 순회 표시 제거
        clearAutoRotationIndicators();
        
        // 버튼 상태 업데이트
        updateAutoRotationButtonState(false);
        
        //console.log('⏹️ Timeframe auto-rotation stopped');
    }
}

// 모든 타이머 종료 함수
function stopAllTimers() {
    //console.log('🛑 Stopping all timers...');
    
    // 분봉 순환 타이머 종료
    if (window.timeframeRotationTimer) {
        clearInterval(window.timeframeRotationTimer);
        window.timeframeRotationTimer = null;
        //console.log('⏹️ Timeframe rotation timer stopped');
    }
    
    // 트레이딩 가격 업데이트 타이머 종료
    if (window.tradingPriceTimer) {
        clearInterval(window.tradingPriceTimer);
        window.tradingPriceTimer = null;
        //console.log('⏹️ Trading price timer stopped');
    }
    
    // 게임 상태 업데이트 타이머 종료
    if (window.gameStateTimer) {
        clearInterval(window.gameStateTimer);
        window.gameStateTimer = null;
        //console.log('⏹️ Game state timer stopped');
    }
    
    // 시스템 상태 업데이트 타이머 종료
    if (window.systemStatusTimer) {
        clearInterval(window.systemStatusTimer);
        window.systemStatusTimer = null;
        //console.log('⏹️ System status timer stopped');
    }
    
    // 우측 패널 업데이트 타이머 종료
    if (window.rightPanelTimer) {
        clearInterval(window.rightPanelTimer);
        window.rightPanelTimer = null;
        //console.log('⏹️ Right panel timer stopped');
    }
    
    // 현재가 감지 타이머 종료
    if (window.priceDetectionTimer) {
        clearInterval(window.priceDetectionTimer);
        window.priceDetectionTimer = null;
        //console.log('⏹️ Price detection timer stopped');
    }
    
    //console.log('✅ All timers stopped successfully');
}

// Current Zone 표시 업데이트 함수 - 차트 마지막 구역과 동기화
function updateCurrentZoneDisplay(timeframe, zone, count, total) {
    const currentZoneDisplay = document.getElementById('current-zone-display');
    if (currentZoneDisplay) {
        const currentZone = currentZoneDisplay.querySelector('h4');
        const zoneStats = currentZoneDisplay.querySelector('.zone-stats');
        
        // 차트의 마지막 구역 정보 가져오기
        let chartLastZone = zone;
        let chartLastZoneStrengthValue2 = 0;
        let chartLastZoneTimeframe = window.currentDisplayTimeframe || convertTimeframeToDisplay(timeframe);
        
        // 전역 저장된 차트 데이터에서 마지막 구역 정보 가져오기
        if (window.sharedMainChartData && window.sharedMainChartData.waveAnalysis) {
            const waveAnalysis = window.sharedMainChartData.waveAnalysis;
            chartLastZone = waveAnalysis.lastZone || zone;
            chartLastZoneStrengthValue2 = waveAnalysis.currentZoneStrength || 0;
        }
        
        if (currentZone) {
            currentZone.textContent = `Current Zone: ${chartLastZone} (${chartLastZoneTimeframe})`;
            //console.log(`🔄 Updated Current Zone display from chart: ${chartLastZone} (${chartLastZoneTimeframe})`);
        }
        
        if (zoneStats) {
            const zoneCountSpan = zoneStats.querySelector('.zone-count');
            const zoneStrengthSpan = zoneStats.querySelector('.zone-strength');
            
            if (zoneCountSpan) {
                zoneCountSpan.textContent = `Chart Last Zone`;
            }
            if (zoneStrengthSpan) {
                zoneStrengthSpan.textContent = `Strength: ${chartLastZoneStrengthValue2}%`;
            }
        }
        
        // Current Zone 배경색도 업데이트 (차트 마지막 구역 기준)
        currentZoneDisplay.className = `current-zone-display ${chartLastZone.toLowerCase()}-zone`;
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
                //console.log('Screenshot copied to clipboard');
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
    //console.log('🎮 Initializing 8BIT Trading System v0.1...');
    
    // settingsManager 안전 초기화
    if (typeof SettingsManager !== 'undefined' && !window.settingsManager) {
        try {
            window.settingsManager = new SettingsManager();
            //console.log('✅ SettingsManager initialized');
        } catch (error) {
            console.error('❌ SettingsManager initialization failed:', error);
        }
    }
    
    // 페이지 로드 시 설정 적용
    applySettingsOnLoad();
    
    // 사운드 시스템 초기화
    initializeSound();
    
    // 게임 상태 업데이트
    updateGameState();
    updateSystemStatus();
    
    // 주기적 업데이트
    window.gameStateTimer = setInterval(updateGameState, 5000);
    window.systemStatusTimer = setInterval(updateSystemStatus, 1000);
    
    // 차트는 모듈 로드시 초기화
    
    updateStatusMessage('System initialized with settings applied');
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    //console.log('🚀 8BIT Trading System v0.1 - Starcraft Style UI Loaded');
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
            <div id="wallet-module-container">
                <!-- 새로운 Wallet Frontend가 여기에 동적으로 생성됩니다 -->
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
        //console.log('Initializing wallet module...');
        
        // Wallet Frontend 인스턴스 생성 (없으면 생성)
        if (!window.walletFrontend) {
            const WalletFrontend = (await import('./modules/wallet/wallet_frontend.js')).default;
            window.walletFrontend = new WalletFrontend();
        }
        
        // 새로운 Wallet Frontend 초기화
        const container = document.getElementById('wallet-module-container');
        if (container && window.walletFrontend) {
            window.walletFrontend.initialize(container);
            //console.log('✅ Wallet Frontend initialized successfully');
        } else {
            console.error('❌ Wallet container or frontend not found');
            if (typeof updateStatusMessage === 'function') {
                updateStatusMessage('Wallet initialization failed');
            }
        }
        
        if (typeof updateStatusMessage === 'function') {
        updateStatusMessage('Wallet module initialized');
        }
    } catch (error) {
        console.error('Failed to initialize wallet module:', error);
        if (typeof updateStatusMessage === 'function') {
        updateStatusMessage('Wallet initialization failed');
        }
    }
}

// 지갑 데이터 새로고침
async function refreshWalletData() {
    try {
        // 설정에서 Upbit API 키 가져오기
                const accessKey = window.settingsManager && typeof window.settingsManager.getSetting === 'function' ?
            window.settingsManager.getSetting('upbit.accessKey') : '';
        const secretKey = window.settingsManager && typeof window.settingsManager.getSetting === 'function' ?
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
                const settings = window.settingsManager && typeof window.settingsManager.getSettings === 'function' ? 
            window.settingsManager.getSettings() : {};
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
            const settings = window.settingsManager && typeof window.settingsManager.getSettings === 'function' ? 
            window.settingsManager.getSettings() : {};
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

// Central Nervous System은 별도 파일로 분리됨
// central-nervous-system.js 파일을 참조하세요

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
    
    .current-price-display {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
        padding: 15px 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .price-main {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .price-label {
        font-size: 14px;
        color: #888;
        font-weight: 500;
    }
    
    .current-price {
        font-size: 24px;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    .price-change-info {
        display: flex;
        align-items: center;
    }
    
    .price-change {
        font-size: 16px;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .price-change.positive {
        color: #00ff88;
        background: rgba(0, 255, 136, 0.2);
    }
    
    .price-change.negative {
        color: #ff6b6b;
        background: rgba(255, 107, 107, 0.2);
    }
    
    .current-zone-display {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
        padding: 15px 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-left: 15px;
    }
    
    .zone-main {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .zone-label {
        font-size: 14px;
        color: #888;
        font-weight: 500;
    }
    
    .current-zone {
        font-size: 20px;
        font-weight: bold;
        color: #ffd700;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    .zone-info {
        display: flex;
        align-items: center;
    }
    
    .zone-strength {
        font-size: 14px;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        color: #00ffff;
    }
    
    .zone-blue {
        color: #00ffff !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5) !important;
    }
    
    .zone-orange {
        color: #ff8c00 !important;
        text-shadow: 0 0 10px rgba(255, 140, 0, 0.5) !important;
    }
    
    .zone-neutral {
        color: #888888 !important;
        text-shadow: none !important;
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
        width: 100%;
        max-width: 100%;
    }
    
    .chart-container canvas {
        width: 100% !important;
        height: auto;
    }
    
    .nb-wave-panel canvas {
        width: 100% !important;
        height: auto;
    }
    
    .signals-panel {
        margin-top: 20px;
    }
    
    /* Active Signals CSS는 active-signals.css에서 관리됨 */
    
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
        display: flex;
        flex-direction: column;
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
    
    .timeframe-status-card {
        background: rgba(0, 0, 0, 0.95);
        border: 2px solid;
        border-radius: 8px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        position: relative;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        text-align: center;
    }
    
    .timeframe-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 12px;
        margin-top: 15px;
        justify-items: center;
        align-items: start;
    }
    
    .timeframe-status-card .timeframe-name {
        font-size: 14px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        margin-bottom: 4px;
        text-align: center;
    }
    
    .timeframe-status-card .timeframe-zone {
        font-size: 12px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        margin-bottom: 4px;
        text-align: center;
    }
    
    .timeframe-status-card .timeframe-strength {
        font-size: 16px;
        font-weight: bold;
        color: #00ff00;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        margin: 4px 0;
        text-align: center;
    }
    
    .timeframe-status-card .timeframe-breakdown {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 6px;
        margin-top: 4px;
        width: 100%;
    }
    
    .timeframe-status-card .blue-count {
        color: #00ffff;
        font-weight: bold;
        font-size: 12px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        text-align: center;
        flex: 1;
    }
    
    .timeframe-status-card .orange-count {
        color: #ff8c00;
        font-weight: bold;
        font-size: 12px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        text-align: center;
        flex: 1;
    }
`;

// 스타일 추가
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// 전역 함수로 노출
window.loadSettings = loadSettings;
window.saveSettings = saveSettings;
window.resetSettings = resetSettings;
window.initializeSettings = initializeSettings;
window.togglePasswordVisibility = togglePasswordVisibility;
window.testUpbitAPI = testUpbitAPI;

// Wave 카운트 분석 함수 (N/B WAVE MAP 값으로 강도 계산)
function analyzeWaveCounts(zones, dataLength, nbData) {
    let blueCount = 0;
    let orangeCount = 0;
    let lastZone = null;
    
    // 가격 데이터 길이에 맞춰서 분석 (메인 차트용)
    const zonesToAnalyze = zones.slice(0, dataLength);
    
    zonesToAnalyze.forEach(zone => {
        if (zone.zone === 'BLUE') {
            blueCount++;
        } else if (zone.zone === 'ORANGE') {
            orangeCount++;
        }
    });
    
    // 마지막 zone 찾기 (가격 데이터 길이 내에서)
    if (zonesToAnalyze.length > 0) {
        lastZone = zonesToAnalyze[zonesToAnalyze.length - 1];
    }
    
    // N/B WAVE MAP 값으로 강도 계산
    let currentZoneStrength = 0;
    if (lastZone && lastZone.zone === 'BLUE') {
        // N/B WAVE MAP의 BLUE - ORANGE 값 사용
        const nbBlue = nbData && nbData.summary ? (nbData.summary.blue || 0) : 0;
        const nbOrange = nbData && nbData.summary ? (nbData.summary.orange || 0) : 0;
        currentZoneStrength = nbBlue - nbOrange;
        //console.log('🔵 Blue Zone - N/B WAVE MAP 강도 계산:', { nbBlue, nbOrange, currentZoneStrength });
    } else if (lastZone && lastZone.zone === 'ORANGE') {
        // N/B WAVE MAP의 ORANGE - BLUE 값 사용
        const nbBlue = nbData && nbData.summary ? (nbData.summary.blue || 0) : 0;
        const nbOrange = nbData && nbData.summary ? (nbData.summary.orange || 0) : 0;
        currentZoneStrength = nbOrange - nbBlue;
        //console.log('🟠 Orange Zone - N/B WAVE MAP 강도 계산:', { nbBlue, nbOrange, currentZoneStrength });
    }
    
    // 전역 변수에 현재 구역 강도 저장
    window.currentZoneStrength = currentZoneStrength;
    /**console.log('💪 Current zone strength calculated:', {
        blueCount,
        orangeCount,
        lastZone: lastZone ? lastZone.zone : 'UNKNOWN',
        lastZoneObject: lastZone,
        currentZoneStrength
    });
     */
    // lastZone이 객체인지 확인하고 zone 속성 추출
    const lastZoneString = lastZone && lastZone.zone ? lastZone.zone : 'UNKNOWN';
    const lastChangeValue = lastZone && lastZone.change ? lastZone.change : 0;
    
    //console.log('🔍 analyzeWaveCounts - lastZone object:', lastZone);
    //console.log('🔍 analyzeWaveCounts - lastZoneString:', lastZoneString);
    
    return {
        blueCount,
        orangeCount,
        totalCount: zonesToAnalyze.length,
        lastZone: lastZoneString,
        lastChange: lastChangeValue,
        currentZoneStrength: currentZoneStrength
    };
}

// Wave 카운트 정보 표시 함수
function drawWaveCountInfo(ctx, waveAnalysis, width, height) {
    const { blueCount, orangeCount, totalCount, lastZone, lastChange, currentZoneStrength } = waveAnalysis;
    
    //console.log('🎨 drawWaveCountInfo - waveAnalysis:', waveAnalysis);
    //console.log('🎨 drawWaveCountInfo - lastZone:', lastZone);
    
    // 텍스트 스타일 설정
    ctx.font = 'bold 14px Courier New';
    ctx.textAlign = 'left';
    
    // Wave Blue 카운트
    ctx.fillStyle = 'rgba(0, 209, 255, 1.0)'; // 파란색
    ctx.fillText(`Wave Blue: ${blueCount}`, 20, 30);
    
    // Wave Orange 카운트
    ctx.fillStyle = 'rgba(255, 183, 3, 1.0)'; // 주황색
    ctx.fillText(`Wave Orange: ${orangeCount}`, 20, 50);
    
    // 총 카운트
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`Total: ${totalCount}`, 20, 70);
    
    // 현재 구역 강도 표시
    ctx.font = 'bold 14px Courier New';
    if (currentZoneStrength > 0) {
        ctx.fillStyle = lastZone === 'BLUE' ? 'rgba(0, 209, 255, 1.0)' : 'rgba(255, 183, 3, 1.0)';
    } else if (currentZoneStrength < 0) {
        ctx.fillStyle = '#ff6b6b'; // 빨간색 (약세)
    } else {
        ctx.fillStyle = '#ffffff'; // 흰색 (중립)
    }
    ctx.fillText(`Strength: ${currentZoneStrength}`, 20, 90);
    
    // 마지막 상태를 전역 변수로 저장 (문자열로 저장)
    const lastZoneString = lastZone || 'UNKNOWN';
    window.lastZoneFromChart = lastZoneString;
    //console.log('📊 Last zone saved to global variable:', lastZoneString);
    
    // 마지막 상태 표시
    ctx.font = 'bold 16px Courier New';
    if (lastZoneString === 'BLUE') {
        ctx.fillStyle = 'rgba(0, 209, 255, 1.0)'; // 파란색
    } else if (lastZoneString === 'ORANGE') {
        ctx.fillStyle = 'rgba(255, 183, 3, 1.0)'; // 주황색
    } else {
        ctx.fillStyle = '#ffffff';
    }
    ctx.fillText(`Last: ${lastZoneString}`, 20, 115);
    
    // 마지막 change 값 표시
    ctx.font = '12px Courier New';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`Change: ${lastChange.toFixed(3)}`, 20, 130);
}

// 전역 헤더 업데이트 함수 (Header Manager로 대체됨)
// 이제 modules/header/header_manager.js에서 처리됩니다.

// 디버깅용 강제 리셋 함수
window.forceResetHeader = function() {
    //console.log('🔄 Force resetting header...');
    if (window.headerManager) {
        window.headerManager.setDefaultValues();
    }
};

// 헤더 업데이트 CSS 스타일 추가
function addHeaderUpdateStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .value-updated {
            animation: valueUpdate 1s ease-in-out;
        }
        
        @keyframes valueUpdate {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); color: #00ff00; text-shadow: 0 0 10px #00ff00; }
            100% { transform: scale(1); }
        }
        
        .mineral-counter, .gas-counter, .supply-counter {
            transition: all 0.3s ease;
        }
        
        .mineral-counter:hover, .gas-counter:hover, .supply-counter:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 255, 0, 0.3);
        }
    `;
    document.head.appendChild(style);
}

// WALLET 페이지 CSS 스타일 추가
function addWalletStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .wallet-container {
            padding: 20px;
            font-family: 'Courier New', monospace;
        }
        
        .wallet-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #00ff00;
        }
        
        .wallet-header h2 {
            color: #00ff00;
            margin: 0;
            text-shadow: 0 0 10px #00ff00;
        }
        
        .wallet-actions {
            display: flex;
            gap: 10px;
        }
        
        .wallet-btn {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #00ff00;
            color: #00ff00;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Courier New', monospace;
        }
        
        .wallet-btn:hover {
            background: #00ff00;
            color: #000;
            box-shadow: 0 0 10px #00ff00;
        }
        
        .wallet-content {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .wallet-section {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid #00ff00;
            border-radius: 8px;
            padding: 20px;
        }
        
        .wallet-section h3 {
            color: #00ff00;
            margin-top: 0;
            margin-bottom: 15px;
            text-shadow: 0 0 5px #00ff00;
        }
        
        .balance-container, .transaction-container {
            min-height: 200px;
        }
        
        .loading {
            color: #888;
            text-align: center;
            padding: 20px;
            font-style: italic;
        }
        
        @media (max-width: 768px) {
            .wallet-content {
                flex-direction: column;
            }
            
            .wallet-header {
                flex-direction: column;
                gap: 15px;
            }
        }
    `;
    document.head.appendChild(style);
}

// 전역 함수 등록 (함수 정의 후에 등록)
window.updateGameHeader = updateGameHeader;

// 강제 초기화 함수 (개발자 도구에서 호출 가능)
window.forceResetHeader = function() {
    //console.log('🔄 Force resetting header values...');
    setDefaultHeaderValues();
    //console.log('✅ Header values reset complete');
};

// 기본 헤더 값 설정 함수 (백그라운드 시스템이 실패할 때 사용)
// 기본 헤더 값 설정 함수 (Header Manager로 대체됨)
// 이제 modules/header/header_manager.js에서 처리됩니다.

// 현재 분봉 표시 업데이트 함수 (헤더와 우측 패널 모두 업데이트)
function updateCurrentTimeframeDisplay(timeframeResults = []) {
    try {
        // 헤더와 우측 패널의 현재 분봉 표시 요소들
        const headerTimeframeDisplay = document.getElementById('current-timeframe-display');
        const headerTimeframeZone = document.getElementById('current-timeframe-zone');
        const headerTimeframeStrength = document.getElementById('current-timeframe-strength');
        
        // 현재 선택된 분봉 찾기
        const selectedTimeframe = window.selectedTimeframe || '1h';
        const currentResult = timeframeResults.find(result => result.name === selectedTimeframe);
        
        // 분봉 이름 업데이트 (헤더와 우측 패널 모두)
        if (headerTimeframeDisplay) {
            headerTimeframeDisplay.textContent = selectedTimeframe;
        }
        
        if (currentResult) {
            // 구역 정보 업데이트
            if (headerTimeframeZone) {
                headerTimeframeZone.textContent = currentResult.lastZone;
                
                // 구역에 따른 색상 변경
                if (currentResult.lastZone === 'BLUE') {
                    headerTimeframeZone.style.color = '#00d1ff';
                } else if (currentResult.lastZone === 'ORANGE') {
                    headerTimeframeZone.style.color = '#ffb703';
                } else {
                    headerTimeframeZone.style.color = '#cccccc';
                }
            }
            
            // 강도 정보 업데이트
            if (headerTimeframeStrength) {
                headerTimeframeStrength.textContent = `${currentResult.lastZoneStrength}%`;
            }
        } else {
            // 기본값 설정
            if (headerTimeframeZone) {
                headerTimeframeZone.textContent = 'NEUTRAL';
                headerTimeframeZone.style.color = '#cccccc';
            }
            if (headerTimeframeStrength) {
                headerTimeframeStrength.textContent = '0%';
            }
        }
        
        //console.log('⏰ Current timeframe display updated (header):', selectedTimeframe);
    } catch (error) {
        console.error('❌ Error updating current timeframe display:', error);
    }
}

// 전역 함수 등록
// window.updateAssetDisplay은 modules/asset/asset_display.js에서 처리됩니다.
window.updateCurrentTimeframeDisplay = updateCurrentTimeframeDisplay;

// 전역 함수 등록 (함수 정의 후에 등록)
window.updateGameHeader = updateGameHeader;

// 기본 선택된 분봉 초기화 함수
function initializeDefaultTimeframe() {
    // 기본값으로 1h 선택
    const defaultTimeframe = '1h';
    window.selectedTimeframe = defaultTimeframe;
    
    const defaultCard = document.getElementById(`timeframe-card-${defaultTimeframe}`);
    if (defaultCard) {
        defaultCard.classList.add('selected');
        
        //console.log('✅ Default timeframe initialized:', defaultTimeframe);
    }
}

// 페이지 로드 시 기본 분봉 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 약간의 지연 후 초기화 (DOM이 완전히 로드된 후)
    setTimeout(() => {
        initializeDefaultTimeframe();
    }, 1000);
});

// 자동 순회 버튼 상태 업데이트 함수
function updateAutoRotationButtonState(isRunning = false) {
    const autoRotateBtn = document.getElementById('btnAutoRotate');
    if (autoRotateBtn) {
        if (isRunning) {
            autoRotateBtn.innerHTML = '<i class="fas fa-pause me-1"></i>순회 중지';
            autoRotateBtn.className = 'btn btn-sm btn-outline-danger';
            autoRotateBtn.title = '자동 순회를 중지합니다';
        } else {
            autoRotateBtn.innerHTML = '<i class="fas fa-play me-1"></i>순회 시작';
            autoRotateBtn.className = 'btn btn-sm btn-outline-warning';
            autoRotateBtn.title = '자동 순회를 시작합니다';
        }
    }
}

// 자동 순회 표시 관리 함수들

// 자동 순회 버튼 클릭 이벤트
window.toggleAutoRotation = function() {
    if (window.timeframeRotationTimer) {
        // 자동 순회 중이면 정지
        stopTimeframeAutoRotation();
    } else {
        // 자동 순회 중이 아니면 시작
        startTimeframeAutoRotation();
    }
};

// 페이지 로드 시 자동 순회 버튼 이벤트 연결
document.addEventListener('DOMContentLoaded', () => {
    const autoRotateBtn = document.getElementById('btnAutoRotate');
    if (autoRotateBtn) {
        autoRotateBtn.addEventListener('click', window.toggleAutoRotation);
        //console.log('✅ Auto-rotation button event connected');
    }
    
    // 약간의 지연 후 초기화 (DOM이 완전히 로드된 후)
    setTimeout(() => {
        initializeDefaultTimeframe();
    }, 1000);
});

// 히스토리 시각화 업데이트 함수
function updateHistoryVisualization(timeframeHistory = []) {
    try {
        const historyDots = document.getElementById('history-dots');
        const historyBars = document.getElementById('history-bars');
        const historyLine = document.getElementById('history-line');
        
        if (!historyDots || !historyBars || !historyLine) return;
        
        // 기존 요소들 제거
        historyDots.innerHTML = '';
        historyBars.innerHTML = '';
        historyLine.innerHTML = '';
        
        // 히스토리 데이터로 시각화 요소 생성
        timeframeHistory.forEach((history, index) => {
            const zoneClass = history.zone.toLowerCase();
            const tooltipText = `${history.timeframe} - ${history.zone} (${history.strength}%)`;
            
            // 점(Dot) 생성
            const dot = document.createElement('div');
            dot.className = `history-dot ${zoneClass}`;
            dot.title = tooltipText;
            historyDots.appendChild(dot);
            
            // 바(Bar) 생성 (강도에 따라 높이 조정)
            const bar = document.createElement('div');
            bar.className = `history-bar ${zoneClass}`;
            bar.title = tooltipText;
            bar.style.height = `${Math.max(4, history.strength / 10)}px`;
            historyBars.appendChild(bar);
            
            // 라인 세그먼트 생성
            const lineSegment = document.createElement('div');
            lineSegment.className = `history-line-segment ${zoneClass}`;
            lineSegment.title = tooltipText;
            lineSegment.style.width = `${Math.max(6, history.strength / 5)}px`;
            historyLine.appendChild(lineSegment);
        });
        
        // 통계도 함께 업데이트
        updateHistoryStatistics(timeframeHistory);
        
        //console.log('🎨 History visualization updated with', timeframeHistory.length, 'items');
    } catch (error) {
        console.error('❌ Error updating history visualization:', error);
    }
}

// 헤더 히스토리 카드 업데이트 함수
function updateHeaderHistoryCard(timeframeHistory = []) {
    try {
        // 최근 5개의 히스토리만 표시
        const recentHistory = timeframeHistory.slice(-5);
        
        // 각 히스토리 아이템 업데이트
        for (let i = 0; i < 5; i++) {
            const historyItem = document.getElementById(`history-item-${i + 1}`);
            if (historyItem) {
                if (i < recentHistory.length) {
                    const history = recentHistory[i];
                    const timeframeSpan = historyItem.querySelector('.history-timeframe');
                    const zoneSpan = historyItem.querySelector('.history-zone');
                    const strengthSpan = historyItem.querySelector('.history-strength');
                    
                    if (timeframeSpan) timeframeSpan.textContent = history.timeframe;
                    if (strengthSpan) strengthSpan.textContent = `${history.strength}%`;
                    
                    if (zoneSpan) {
                        zoneSpan.textContent = history.zone;
                        // 기존 클래스 제거
                        zoneSpan.classList.remove('blue-zone', 'orange-zone', 'neutral-zone');
                        
                        // 구역에 따른 클래스 추가
                        if (history.zone === 'BLUE') {
                            zoneSpan.classList.add('blue-zone');
                        } else if (history.zone === 'ORANGE') {
                            zoneSpan.classList.add('orange-zone');
                        } else {
                            zoneSpan.classList.add('neutral-zone');
                        }
                    }
                    
                    historyItem.style.display = 'flex';
                } else {
                    // 히스토리가 없으면 숨김
                    historyItem.style.display = 'none';
                }
            }
        }
        
        // 시각화도 함께 업데이트
        updateHistoryVisualization(timeframeHistory);
        
        //console.log('📜 Header history card updated with', recentHistory.length, 'items');
    } catch (error) {
        console.error('❌ Error updating header history card:', error);
    }
}

// TimeframeHistory 객체 정의
const TimeframeHistory = {
    history: [],
    maxHistory: 50, // 최대 50개까지 저장
    
    add: function(timeframe, zone, strength, timestamp = new Date().toISOString()) {
        const historyItem = {
            timeframe: timeframe,
            zone: zone,
            strength: strength,
            timestamp: timestamp
        };
        
        this.history.push(historyItem);
        
        // 최대 개수 제한
        if (this.history.length > this.maxHistory) {
            this.history = this.history.slice(-this.maxHistory);
        }
        
        // 로컬 스토리지에 저장
        this.saveToStorage();
        
        // 헤더 히스토리 카드 업데이트
        if (typeof updateHeaderHistoryCard === 'function') {
            updateHeaderHistoryCard(this.history);
        }
        
        //console.log(`📝 Added to history: ${timeframe} - ${zone} (${strength}%)`);
    },
    
    saveToStorage: function() {
        try {
            localStorage.setItem('timeframeHistory', JSON.stringify(this.history));
        } catch (error) {
            console.error('❌ Error saving timeframe history to storage:', error);
        }
    },
    
    loadFromStorage: function() {
        try {
            const saved = localStorage.getItem('timeframeHistory');
            if (saved) {
                this.history = JSON.parse(saved);
                //console.log(`📚 Loaded ${this.history.length} history items from storage`);
            }
        } catch (error) {
            console.error('❌ Error loading timeframe history from storage:', error);
        }
    },
    
    clear: function() {
        this.history = [];
        localStorage.removeItem('timeframeHistory');
        //console.log('🗑️ Timeframe history cleared');
    },
    
    getRecent: function(count = 5) {
        return this.history.slice(-count);
    }
};

// 전역 함수 등록
window.updateHeaderHistoryCard = updateHeaderHistoryCard;
window.updateHistoryVisualization = updateHistoryVisualization;
window.updateHistoryStatistics = updateHistoryStatistics;
window.TimeframeHistory = TimeframeHistory;

// 페이지 로드 시 히스토리 로드
document.addEventListener('DOMContentLoaded', () => {
    TimeframeHistory.loadFromStorage();
});

// 히스토리 통계 계산 및 업데이트 함수
function updateHistoryStatistics(timeframeHistory = []) {
    try {
        let orangeCount = 0, blueCount = 0, orangeSum = 0, blueSum = 0;
        
        // 구역별 카운트와 강도 합계 계산
        timeframeHistory.forEach(history => {
            if (history.zone === 'ORANGE') {
                orangeCount++;
                orangeSum += history.strength;
            } else if (history.zone === 'BLUE') {
                blueCount++;
                blueSum += history.strength;
            }
        });
        
        // 다수 구역 결정
        let majority = '-';
        let majorityClass = '';
        
        if (orangeCount > blueCount) {
            majority = 'ORANGE';
            majorityClass = 'orange';
        } else if (blueCount > orangeCount) {
            majority = 'BLUE';
            majorityClass = 'blue';
        } else if (orangeCount === blueCount && orangeCount > 0) {
            majority = 'TIE';
            majorityClass = 'tie';
        }
        
        // jQuery로 통계 업데이트 (새로운 카드)
        $('#majority-zone').text(majority).removeClass('orange blue tie').addClass(majorityClass);
        $('#orange-sum').text(orangeSum);
        $('#blue-sum').text(blueSum);
        
        // 기존 통계도 업데이트
        $('#majority-zone-old').text(majority).removeClass('orange blue tie').addClass(majorityClass);
        $('#orange-sum-old').text(orangeSum);
        $('#blue-sum-old').text(blueSum);
        
        //console.log('📊 History statistics updated:', { majority, orangeSum, blueSum });
    } catch (error) {
        console.error('❌ Error updating history statistics:', error);
    }
}

// 히스토리 시각화 업데이트 함수
function updateHistoryVisualization(timeframeHistory = []) {
    try {
        const historyDots = document.getElementById('history-dots');
        const historyBars = document.getElementById('history-bars');
        const historyLine = document.getElementById('history-line');
        
        if (!historyDots || !historyBars || !historyLine) return;
        
        // 기존 요소들 제거
        historyDots.innerHTML = '';
        historyBars.innerHTML = '';
        historyLine.innerHTML = '';
        
        // 히스토리 데이터로 시각화 요소 생성
        timeframeHistory.forEach((history, index) => {
            const zoneClass = history.zone.toLowerCase();
            const tooltipText = `${history.timeframe} - ${history.zone} (${history.strength}%)`;
            
            // 점(Dot) 생성 - 새로운 스타일
            const dot = document.createElement('div');
            dot.className = `history-dot ${zoneClass}`;
            dot.title = tooltipText;
            dot.style.cssText = `
                width: ${8 + history.strength * 0.3}px;
                height: ${8 + history.strength * 0.3}px;
                border-radius: 50%;
                margin: 2px;
                background: ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'};
                box-shadow: 0 0 6px ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'};
                transition: all 0.3s ease;
                cursor: pointer;
            `;
            historyDots.appendChild(dot);
            
            // 바(Bar) 생성 - 새로운 스타일
            const bar = document.createElement('div');
            bar.className = `history-bar ${zoneClass}`;
            bar.title = tooltipText;
            bar.style.cssText = `
                width: 12px;
                height: ${20 + history.strength * 1.8}px;
                background: ${history.zone === 'BLUE' ? 'linear-gradient(to top, #00d1ff, #0099cc)' : 
                             history.zone === 'ORANGE' ? 'linear-gradient(to top, #ffb703, #ff8c00)' : 
                             'linear-gradient(to top, #cccccc, #999999)'};
                border-radius: 2px;
                margin: 0 1px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                cursor: pointer;
                flex-shrink: 0;
            `;
            historyBars.appendChild(bar);
            
            // 라인 세그먼트 생성 - 새로운 스타일
            const lineSegment = document.createElement('div');
            lineSegment.className = `history-line-segment ${zoneClass}`;
            lineSegment.title = tooltipText;
            lineSegment.style.cssText = `
                width: ${6 + history.strength * 0.2}px;
                height: 4px;
                border-radius: 2px;
                background: ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'};
                box-shadow: 0 0 4px ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'};
                transition: all 0.3s ease;
                cursor: pointer;
                margin: 0 1px;
            `;
            historyLine.appendChild(lineSegment);
            
            // 호버 효과 추가
            [dot, bar, lineSegment].forEach(element => {
                element.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.1)';
                    this.style.boxShadow = `0 0 12px ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'}`;
                });
                
                element.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = `0 0 6px ${history.zone === 'BLUE' ? '#00d1ff' : history.zone === 'ORANGE' ? '#ffb703' : '#cccccc'}`;
                });
            });
        });
        
        // 통계도 함께 업데이트
        updateHistoryStatistics(timeframeHistory);
        
        //console.log('🎨 History visualization updated with', timeframeHistory.length, 'items');
    } catch (error) {
        console.error('❌ Error updating history visualization:', error);
    }
}

// 테스트용 히스토리 데이터 추가 함수
function addTestHistoryData() {
    const testData = [
        { timeframe: '1h', zone: 'BLUE', strength: 75 },
        { timeframe: '30m', zone: 'ORANGE', strength: 60 },
        { timeframe: '15m', zone: 'BLUE', strength: 80 },
        { timeframe: '5m', zone: 'NEUTRAL', strength: 45 },
        { timeframe: '1m', zone: 'ORANGE', strength: 55 },
        { timeframe: '1h', zone: 'BLUE', strength: 70 },
        { timeframe: '30m', zone: 'ORANGE', strength: 65 },
        { timeframe: '15m', zone: 'BLUE', strength: 85 },
        { timeframe: '5m', zone: 'ORANGE', strength: 50 },
        { timeframe: '1m', zone: 'BLUE', strength: 40 }
    ];
    
    testData.forEach(data => {
        TimeframeHistory.add(data.timeframe, data.zone, data.strength);
    });
    
    //console.log('🧪 Test history data added');
}

// 전역 함수 등록
window.updateHeaderHistoryCard = updateHeaderHistoryCard;
window.updateHistoryVisualization = updateHistoryVisualization;
window.updateHistoryStatistics = updateHistoryStatistics;
window.TimeframeHistory = TimeframeHistory;
window.addTestHistoryData = addTestHistoryData;

// Asset Display 관련 함수들은 modules/asset/asset_display.js에서 처리됩니다.