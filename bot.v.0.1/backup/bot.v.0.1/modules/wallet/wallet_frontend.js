// ===== Wallet Frontend - JavaScript =====

class WalletFrontend {
    constructor() {
        this.container = null;
        this.updateInterval = null;
        this.lastUpdateTime = null;
        this.updateCount = 0;
        this.maxUpdatesPerMinute = 1; // 1분에 1번
        this.updateIntervalMs = 60000; // 1분마다 업데이트
        this.isBackgroundUpdateActive = false;
        
        // DOM 요소들
        this.statusBar = null;
        this.balanceContainer = null;
        this.transactionsContainer = null;
        this.refreshButton = null;
        this.settingsButton = null;
        this.toggleButton = null;
    }
    
    initialize(container) {
        this.container = container;
        this.createWalletInterface();
        this.registerEventListeners();
        this.startBackgroundUpdate();
        this.loadWalletData();
        
        console.log('💰 Wallet Frontend initialized');
    }
    
    createWalletInterface() {
        this.container.innerHTML = `
            <div class="wallet-container">
                <!-- 실시간 상태바 -->
                <div class="wallet-status-bar" id="wallet-status-bar">
                    <div class="status-indicator">
                        <i class="fas fa-circle status-icon" id="status-icon"></i>
                        <span id="status-text">연결 중...</span>
                    </div>
                    <div class="status-info">
                        <span id="total-balance">총 자산: ₩0</span>
                        <span id="last-update">마지막 업데이트: -</span>
                        <span id="update-count">업데이트: 0/분</span>
                    </div>
                    <div class="status-actions">
                        <button id="wallet-refresh-btn" class="btn btn-sm btn-primary">
                            <i class="fas fa-sync-alt"></i> 새로고침
                        </button>
                        <button id="wallet-settings-btn" class="btn btn-sm btn-secondary">
                            <i class="fas fa-cog"></i> 설정
                        </button>
                        <button id="toggle-background-update" class="btn btn-sm btn-warning">
                            <i class="fas fa-pause"></i> 일시정지
                        </button>
                    </div>
                </div>
                
                <!-- 에러 메시지 영역 -->
                <div class="wallet-error-message" id="wallet-error-message" style="display: none;">
                    <div class="error-content">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span id="error-text"></span>
                        <div class="error-actions">
                            <button id="error-settings-btn" class="btn btn-sm btn-primary">설정 열기</button>
                            <button id="error-retry-btn" class="btn btn-sm btn-secondary">다시 시도</button>
                        </div>
                    </div>
                </div>
                
                <!-- 잔고 개요 -->
                <div class="wallet-section">
                    <h3><i class="fas fa-wallet"></i> 잔고 개요</h3>
                    <div class="balance-overview" id="balance-overview">
                        <div class="loading-message">
                            <i class="fas fa-spinner fa-spin"></i>
                            잔고 데이터를 불러오는 중...
                        </div>
                    </div>
                </div>
                
                <!-- 거래 내역 -->
                <div class="wallet-section">
                    <h3><i class="fas fa-history"></i> 거래 내역</h3>
                    <div class="transactions-container" id="transactions-container">
                        <div class="loading-message">
                            <i class="fas fa-spinner fa-spin"></i>
                            거래 내역을 불러오는 중...
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // DOM 요소 참조 저장
        this.statusBar = document.getElementById('wallet-status-bar');
        this.balanceContainer = document.getElementById('balance-overview');
        this.transactionsContainer = document.getElementById('transactions-container');
        this.refreshButton = document.getElementById('wallet-refresh-btn');
        this.settingsButton = document.getElementById('wallet-settings-btn');
        this.toggleButton = document.getElementById('toggle-background-update');
        
        this.addWalletStyles();
    }
    
    registerEventListeners() {
        // 새로고침 버튼
        this.refreshButton.addEventListener('click', () => {
            this.loadWalletData();
        });
        
        // 설정 버튼
        this.settingsButton.addEventListener('click', () => {
            this.openSettings();
        });
        
        // 백그라운드 업데이트 토글
        this.toggleButton.addEventListener('click', () => {
            this.toggleBackgroundUpdate();
        });
        
        // 에러 메시지 버튼들
        document.getElementById('error-settings-btn').addEventListener('click', () => {
            this.openSettings();
        });
        
        document.getElementById('error-retry-btn').addEventListener('click', () => {
                this.loadWalletData();
            });
        
        // Settings 변경 이벤트 리스너
        document.addEventListener('settingsChanged', (event) => {
            if (event.detail.section === 'upbit') {
                console.log('🔧 Upbit settings changed, reloading wallet data...');
                setTimeout(() => {
                    this.loadWalletData();
                }, 1000);
            }
        });
    }
    
    async loadWalletData() {
        try {
            this.updateStatusIndicator('loading');
            
            // 잔고 데이터 로드
            const balanceResponse = await fetch('/api/wallet/balance');
            const balanceData = await balanceResponse.json();
            
            // 거래 내역 로드
            const transactionsResponse = await fetch('/api/wallet/transactions');
            const transactionsData = await transactionsResponse.json();
            
            // 결과 처리
            this.handleBalanceResponse(balanceData);
            this.handleTransactionsResponse(transactionsData);
            
        } catch (error) {
            console.error('❌ Failed to load wallet data:', error);
            this.showError('네트워크 오류가 발생했습니다.');
            this.updateStatusIndicator('error');
        }
    }
    
    handleBalanceResponse(data) {
        if (data.status === 'success') {
            this.updateBalanceDisplay(data.data);
            this.updateStatusIndicator('success');
            this.hideError();
            
            // 게임 헤더 업데이트
            if (window.updateGameHeader) {
                window.updateGameHeader(data.data);
            }
            } else {
            this.showError(data.message || '잔고 조회에 실패했습니다.');
            this.updateStatusIndicator('error');
            
            // 게임 헤더 초기화
            if (window.updateGameHeader) {
                window.updateGameHeader({
                    total_value: 0,
                    total_krw: 0,
                    total_btc_value: 0
                });
            }
        }
    }
    
    handleTransactionsResponse(data) {
        if (data.status === 'success') {
            this.updateTransactionsDisplay(data.data);
        } else {
            if (this.transactionsContainer) {
                this.transactionsContainer.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle"></i>
                        거래 내역을 불러올 수 없습니다: ${data.message}
                    </div>
                `;
            } else {
                console.warn('⚠️ Transactions container not found');
            }
        }
    }
    
    updateBalanceDisplay(balanceData) {
        const { total_value, total_krw, total_btc_value, balances, last_update } = balanceData;
        
        // 전역으로 지갑 데이터 저장 (다른 컴포넌트에서 사용하기 위해)
        window.sharedWalletData = {
            total_value,
            total_krw, 
            total_btc_value,
            balances,
            last_update,
            timestamp: new Date().getTime()
        };
        
        console.log('💾 Wallet data saved to global storage:', window.sharedWalletData);
        
        this.renderBalanceDisplay(balanceData);
    }
    
    renderBalanceDisplay(balanceData) {
        const { total_value, total_krw, total_btc_value, balances, last_update } = balanceData;
        
        // 상태바 업데이트 - DOM 요소 존재 확인
        const totalBalanceEl = document.getElementById('total-balance');
        const lastUpdateEl = document.getElementById('last-update');
        const updateCountEl = document.getElementById('update-count');
        
        if (totalBalanceEl) totalBalanceEl.textContent = `총 자산: ₩${total_value.toLocaleString()}`;
        if (lastUpdateEl) lastUpdateEl.textContent = `마지막 업데이트: ${this.formatTime(last_update)}`;
        if (updateCountEl) updateCountEl.textContent = `업데이트: ${this.updateCount}/${this.maxUpdatesPerMinute}/분`;
        
        // 잔고 개요 업데이트
        let balanceHtml = `
            <div class="balance-summary">
                <div class="balance-card total">
                    <div class="balance-label">총 자산</div>
                    <div class="balance-value">₩${total_value.toLocaleString()}</div>
                </div>
                <div class="balance-card krw">
                    <div class="balance-label">KRW</div>
                    <div class="balance-value">₩${total_krw.toLocaleString()}</div>
                    </div>
                <div class="balance-card btc">
                    <div class="balance-label">BTC 가치</div>
                    <div class="balance-value">₩${total_btc_value.toLocaleString()}</div>
                    </div>
                </div>
        `;
            
        if (balances.length > 0) {
            balanceHtml += `
            <div class="balance-details">
                    <h4>자산 상세</h4>
                    <div class="balance-list">
            `;
            
                        // Settings에서 선택한 코인과 KRW만 필터링
            const selectedCoin = window.selectedKrwCoin || 'BTC';
            let filteredBalances = balances.filter(balance => 
                balance.currency === 'KRW' || balance.currency === selectedCoin
            );
            
            // KRW가 없으면 0으로 추가
            const hasKRW = filteredBalances.some(balance => balance.currency === 'KRW');
            if (!hasKRW) {
                filteredBalances.push({
                    currency: 'KRW',
                    balance: '0',
                    current_price: 1,
                    asset_value: 0
                });
            }
            
            // 선택된 코인이 없으면 0으로 추가
            const hasSelectedCoin = filteredBalances.some(balance => balance.currency === selectedCoin);
            if (!hasSelectedCoin) {
                filteredBalances.push({
                    currency: selectedCoin,
                    balance: '0',
                    current_price: 0,
                    asset_value: 0
                });
            }
            
            // KRW를 먼저, 선택된 코인을 나중에 정렬
            filteredBalances.sort((a, b) => {
                if (a.currency === 'KRW') return -1;
                if (b.currency === 'KRW') return 1;
                return 0;
            });
            
            filteredBalances.forEach(balance => {
                const { currency, balance: amount, current_price, asset_value, avg_buy_price } = balance;
                
                // API에서 제공하는 평균 매수가 사용
                const avgPrice = currency === 'KRW' ? 1 : parseFloat(avg_buy_price || 0);
                
                balanceHtml += `
                    <div class="balance-item">
                        <div class="currency-header">
                            <div class="currency-info">
                                <span class="currency-name">${currency}</span>
                            </div>
                    </div>
                        <div class="currency-details">
                            <div class="balance-row">
                                <span class="label">보유 수량:</span>
                                <span class="value">${currency === 'KRW' ? parseFloat(amount).toLocaleString() : parseFloat(amount).toFixed(8)}</span>
                    </div>
                            <div class="balance-row">
                                <span class="label">평균 매수가:</span>
                                <span class="value">${avgPrice > 0 ? `₩${avgPrice.toLocaleString()}` : (currency === 'KRW' ? '₩1' : '거래 내역 없음')}</span>
                    </div>
                </div>
            </div>
                `;
            });
            
            balanceHtml += `
                </div>
            </div>
        `;
    }
    
        if (this.balanceContainer) {
            this.balanceContainer.innerHTML = balanceHtml;
        } else {
            console.warn('⚠️ Balance container not found');
        }
    }
    
    updateTransactionsDisplay(transactions) {
        if (!this.transactionsContainer) {
            console.warn('⚠️ Transactions container not found');
            return;
        }
        
        if (transactions.length === 0) {
            this.transactionsContainer.innerHTML = `
                <div class="no-data">
                    <i class="fas fa-inbox"></i>
                    거래 내역이 없습니다.
            </div>
        `;
            return;
        }
        
        let transactionsHtml = `
            <div class="transactions-list">
        `;
        
        transactions.forEach(tx => {
            const isBuy = tx.side === 'bid';
            const sideText = isBuy ? '매수' : '매도';
            const sideClass = isBuy ? 'buy' : 'sell';
            
            transactionsHtml += `
                <div class="transaction-item ${sideClass}">
                    <div class="transaction-header">
                        <span class="transaction-side">${sideText}</span>
                        <span class="transaction-market">${tx.market}</span>
                        <span class="transaction-time">${this.formatTime(tx.created_at)}</span>
                </div>
                    <div class="transaction-details">
                        <div class="transaction-price">₩${tx.price.toLocaleString()}</div>
                        <div class="transaction-volume">${parseFloat(tx.executed_volume).toFixed(8)}</div>
                        <div class="transaction-funds">₩${tx.executed_funds.toLocaleString()}</div>
                </div>
            </div>
        `;
        });
        
        transactionsHtml += `</div>`;
        if (this.transactionsContainer) {
            this.transactionsContainer.innerHTML = transactionsHtml;
        }
    }
    
    startBackgroundUpdate() {
        this.isBackgroundUpdateActive = true;
        
        // 분당 업데이트 카운터 리셋
        setInterval(() => {
            this.updateCount = 0;
        }, 60000);
        
        // 주기적 업데이트
        this.updateInterval = setInterval(() => {
            if (this.updateCount < this.maxUpdatesPerMinute) {
                this.updateRealTimeData();
            }
        }, this.updateIntervalMs);
        
        console.log('🔄 Background update started');
    }
    
    async updateRealTimeData() {
        try {
            const response = await fetch('/api/wallet/balance?light=true');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateBalanceDisplay(data.data);
                this.updateStatusIndicator('success');
                this.hideError();
                this.updateCount++;
                
                // 게임 헤더 업데이트
                if (window.headerManager) {
                    window.headerManager.updateHeader(data.data);
                } else if (window.updateGameHeader) {
                    window.updateGameHeader(data.data); // Fallback for compatibility
                }
                
                // 자산 표시 업데이트 (Asset Display)
                if (window.assetDisplayManager) {
                    window.assetDisplayManager.syncBalanceWithAssetDisplay();
                }
                } else {
                this.updateStatusIndicator('error');
                this.showError(data.message);
                
                // 게임 헤더 초기화
                if (window.headerManager) {
                    window.headerManager.updateHeader({
                        total_value: 0,
                        total_krw: 0,
                        total_btc_value: 0
                    });
                } else if (window.updateGameHeader) {
                    window.updateGameHeader({ // Fallback for compatibility
                        total_value: 0,
                        total_krw: 0,
                        total_btc_value: 0
                    });
                }
                
                // 자산 표시 초기화 (Asset Display)
                if (window.assetDisplayManager) {
                    window.assetDisplayManager.updateAssetDisplay(0, 0, 0);
                }
                }
            } catch (error) {
            console.error('❌ Real-time update failed:', error);
            this.updateStatusIndicator('error');
            this.showError('실시간 업데이트에 실패했습니다.');
        }
    }
    
    stopBackgroundUpdate() {
        this.isBackgroundUpdateActive = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        console.log('⏹️ Background update stopped');
    }
    
    toggleBackgroundUpdate() {
        if (this.isBackgroundUpdateActive) {
            this.stopBackgroundUpdate();
            this.toggleButton.innerHTML = '<i class="fas fa-play"></i> 재시작';
            this.toggleButton.className = 'btn btn-sm btn-success';
        } else {
            this.startBackgroundUpdate();
            this.toggleButton.innerHTML = '<i class="fas fa-pause"></i> 일시정지';
            this.toggleButton.className = 'btn btn-sm btn-warning';
        }
    }
    
    updateStatusIndicator(status) {
        const icon = document.getElementById('status-icon');
        const text = document.getElementById('status-text');
        
        // DOM 요소가 존재하는지 확인
        if (!icon || !text) {
            console.warn('⚠️ Status indicator elements not found');
                return;
            }
            
        icon.className = 'fas fa-circle status-icon';
        
        switch (status) {
            case 'success':
                icon.classList.add('status-success');
                text.textContent = '연결됨';
                break;
            case 'error':
                icon.classList.add('status-error');
                text.textContent = '연결 오류';
                break;
            case 'loading':
                icon.classList.add('status-loading');
                text.textContent = '연결 중...';
                break;
            default:
                icon.classList.add('status-warning');
                text.textContent = '대기 중';
        }
    }
    
    showError(message) {
        const errorContainer = document.getElementById('wallet-error-message');
        const errorText = document.getElementById('error-text');
        
        // DOM 요소가 존재하는지 확인
        if (!errorContainer || !errorText) {
            console.warn('⚠️ Error message elements not found');
            return;
        }
        
        errorText.textContent = message;
        errorContainer.style.display = 'block';
    }
    
    hideError() {
        const errorContainer = document.getElementById('wallet-error-message');
        
        // DOM 요소가 존재하는지 확인
        if (!errorContainer) {
            console.warn('⚠️ Error container not found');
            return;
        }
        
        errorContainer.style.display = 'none';
    }
    
    openSettings() {
        if (window.settingsManager && window.settingsManager.toggleSettings) {
            window.settingsManager.toggleSettings();
        } else {
            // Settings 모듈이 없으면 기본 설정 페이지로 이동
            loadModule('settings');
        }
    }
    
    formatTime(timeString) {
        if (!timeString) return '-';
        
        try {
            const date = new Date(timeString);
            return date.toLocaleString('ko-KR');
        } catch (error) {
            return timeString;
        }
    }
    
    addWalletStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .wallet-container {
                padding: 20px;
                background: #1a1a1a;
                border-radius: 8px;
                margin: 10px;
            }
            
            .wallet-status-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #2a2a2a;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
                border: 1px solid #444;
            }
            
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .status-icon {
                font-size: 12px;
            }
            
            .status-success { color: #27ae60; }
            .status-error { color: #e74c3c; }
            .status-loading { color: #f39c12; animation: pulse 1s infinite; }
            .status-warning { color: #f39c12; }
            
            .status-info {
                display: flex;
                gap: 20px;
                font-size: 14px;
                color: #ccc;
            }
            
            .status-actions {
                display: flex;
                gap: 10px;
            }
            
            .wallet-error-message {
                background: #e74c3c;
                color: white;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
            }
            
            .error-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .error-actions {
                margin-left: auto;
                display: flex;
                gap: 10px;
            }
            
            .wallet-section {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 20px;
                border: 1px solid #444;
            }
            
            .wallet-section h3 {
                margin: 0 0 15px 0;
                color: #fff;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .balance-summary {
                display: flex;
                flex-direction: column;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .balance-card {
                background: #333;
                padding: 20px;
                border-radius: 6px;
                text-align: center;
                border: 1px solid #555;
            }
            
            .balance-card.total { border-color: #27ae60; }
            .balance-card.krw { border-color: #3498db; }
            .balance-card.btc { border-color: #f39c12; }
            
            .balance-label {
                font-size: 14px;
                color: #ccc;
                margin-bottom: 8px;
            }
            
            .balance-value {
                font-size: 24px;
                font-weight: bold;
                color: #fff;
            }
            
            .balance-details h4 {
                margin: 0 0 15px 0;
                color: #fff;
            }
            
            .balance-list {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .balance-item {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid #555;
                border-radius: 8px;
                padding: 15px;
                transition: all 0.3s ease;
            }
            
            .balance-item:hover {
                border-color: #00ff00;
                box-shadow: 0 0 10px rgba(0, 255, 0, 0.2);
            }
            
            .currency-header {
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #444;
            }
            
            .currency-name {
                font-size: 18px;
                font-weight: bold;
                color: #00ff00;
                text-shadow: 0 0 5px #00ff00;
            }
            
            .currency-details {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .balance-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 0;
            }
            
            .balance-row .label {
                font-size: 14px;
                color: #aaa;
            }
            
            .balance-row .value {
                font-size: 16px;
                font-weight: bold;
                color: #fff;
            }
            
            .balance-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                background: #333;
                border-radius: 4px;
                border: 1px solid #555;
            }
            
            .currency-info {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            
            .currency-name {
                font-weight: bold;
                color: #fff;
            }
            
            .currency-amount {
                font-size: 12px;
                color: #ccc;
            }
            
            .currency-value {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 4px;
            }
            
            .current-price {
                font-size: 14px;
                color: #fff;
            }
            
            .asset-value {
                font-size: 12px;
                color: #27ae60;
            }
            
            .transactions-list {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .transaction-item {
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #555;
            }
            
            .transaction-item.buy {
                background: rgba(39, 174, 96, 0.1);
                border-color: #27ae60;
            }
            
            .transaction-item.sell {
                background: rgba(231, 76, 60, 0.1);
                border-color: #e74c3c;
            }
            
            .transaction-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            
            .transaction-side {
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            
            .transaction-item.buy .transaction-side {
                background: #27ae60;
                color: white;
            }
            
            .transaction-item.sell .transaction-side {
                background: #e74c3c;
                color: white;
            }
            
            .transaction-market {
                color: #ccc;
                font-size: 14px;
            }
            
            .transaction-time {
                color: #999;
                font-size: 12px;
            }
            
            .transaction-details {
                display: flex;
                flex-direction: column;
                gap: 10px;
                text-align: center;
            }
            
            .transaction-price,
            .transaction-volume,
            .transaction-funds {
                font-size: 14px;
                color: #fff;
            }
            
            .loading-message,
            .no-data,
            .error-message {
                text-align: center;
                padding: 40px;
                color: #ccc;
                font-size: 16px;
            }
            
            .loading-message i,
            .no-data i,
            .error-message i {
                font-size: 24px;
                margin-bottom: 10px;
                display: block;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .btn {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                transition: all 0.3s ease;
            }
            
            .btn-sm {
                padding: 6px 12px;
                font-size: 12px;
            }
            
            .btn-primary {
                background: #3498db;
                color: white;
            }
            
            .btn-secondary {
                background: #95a5a6;
                color: white;
            }
            
            .btn-success {
                background: #27ae60;
                color: white;
            }
            
            .btn-warning {
                background: #f39c12;
                color: white;
            }
            
            .btn:hover {
                opacity: 0.8;
                transform: translateY(-1px);
            }
        `;
        document.head.appendChild(style);
    }
    
    destroy() {
        this.stopBackgroundUpdate();
        if (this.container) {
            this.container.innerHTML = '';
        }
        console.log('💰 Wallet Frontend destroyed');
    }
}

// 전역 인스턴스
window.walletFrontend = new WalletFrontend();
