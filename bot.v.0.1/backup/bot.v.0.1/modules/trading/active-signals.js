// ===== Active Signals Module =====
// 실시간 트레이딩 신호를 관리하는 모듈

class ActiveSignalsManager {
    constructor() {
        this.signals = [];
        this.container = null;
        this.updateInterval = null;
        this.isInitialized = false;
        
        console.log('🚦 Active Signals Manager initialized');
    }
    
    // Active Signals HTML 생성
    generateActiveSignalsHTML() {
        return `
            <div class="active-signals-panel">
                <div class="signals-header">
                    <h3><i class="fas fa-bell"></i> Active Signals</h3>
                    <div class="signals-controls">
                        <button class="btn-signal-refresh" onclick="window.activeSignalsManager.refreshSignals()">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="btn-signal-clear" onclick="window.activeSignalsManager.clearSignals()">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                
                <div class="signals-container">
                    <div id="active-signals-list" class="signals-list">
                        <div class="no-signals-message">
                            <i class="fas fa-info-circle"></i>
                            <span>No active signals</span>
                        </div>
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
        `;
    }
    
    // 신호 카드 HTML 생성
    generateSignalCard(signal) {
        const timestamp = new Date(signal.timestamp || Date.now()).toLocaleTimeString('ko-KR');
        const strengthColor = signal.strength >= 80 ? '#00ff00' : signal.strength >= 60 ? '#ffff00' : '#ff6b6b';
        
        return `
            <div class="signal-card ${signal.type.toLowerCase()}" data-signal-id="${signal.id}">
                <div class="signal-header">
                    <div class="signal-type ${signal.type.toLowerCase()}">
                        <i class="fas fa-${signal.type.toLowerCase() === 'buy' ? 'arrow-up' : 'arrow-down'}"></i>
                        <span>${signal.type.toUpperCase()}</span>
                    </div>
                    <div class="signal-time">${timestamp}</div>
                </div>
                
                <div class="signal-body">
                    <div class="signal-strength">
                        <span class="strength-label">Strength:</span>
                        <span class="strength-value" style="color: ${strengthColor}">
                            ${(signal.strength * 100).toFixed(0)}%
                        </span>
                    </div>
                    
                    <div class="signal-timeframe">
                        <span class="timeframe-label">Timeframe:</span>
                        <span class="timeframe-value">${signal.timeframe}</span>
                    </div>
                    
                    ${signal.price ? `
                        <div class="signal-price">
                            <span class="price-label">Price:</span>
                            <span class="price-value">₩${signal.price.toLocaleString()}</span>
                        </div>
                    ` : ''}
                    
                    ${signal.reason ? `
                        <div class="signal-reason">
                            <span class="reason-label">Reason:</span>
                            <span class="reason-value">${signal.reason}</span>
                        </div>
                    ` : ''}
                </div>
                
                <div class="signal-actions">
                    <button class="btn-signal-action" onclick="window.activeSignalsManager.executeSignal('${signal.id}')">
                        <i class="fas fa-play"></i> Execute
                    </button>
                    <button class="btn-signal-dismiss" onclick="window.activeSignalsManager.dismissSignal('${signal.id}')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `;
    }
    
    // 신호 추가
    addSignal(signal) {
        const signalWithId = {
            ...signal,
            id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now()
        };
        
        this.signals.unshift(signalWithId); // 최신 신호를 맨 앞에 추가
        
        // 최대 10개까지만 유지
        if (this.signals.length > 10) {
            this.signals = this.signals.slice(0, 10);
        }
        
        this.updateSignalsDisplay();
        this.updateSignalsStats();
        
        console.log('🚦 New signal added:', signalWithId);
        
        // 신호 알림 (옵션)
        this.showSignalNotification(signalWithId);
    }
    
    // 신호 제거
    removeSignal(signalId) {
        this.signals = this.signals.filter(signal => signal.id !== signalId);
        this.updateSignalsDisplay();
        this.updateSignalsStats();
        
        console.log('🚦 Signal removed:', signalId);
    }
    
    // 신호 무시
    dismissSignal(signalId) {
        this.removeSignal(signalId);
    }
    
    // 신호 실행
    executeSignal(signalId) {
        const signal = this.signals.find(s => s.id === signalId);
        if (!signal) {
            console.warn('⚠️ Signal not found:', signalId);
            return;
        }
        
        console.log('🚀 Executing signal:', signal);
        
        // 여기에 실제 거래 실행 로직 추가
        // 예: window.tradingManager.executeOrder(signal)
        
        // 실행된 신호는 제거
        this.removeSignal(signalId);
    }
    
    // 신호 표시 업데이트
    updateSignalsDisplay() {
        const container = document.getElementById('active-signals-list');
        if (!container) {
            console.warn('⚠️ Active signals container not found');
            return;
        }
        
        if (this.signals.length === 0) {
            container.innerHTML = `
                <div class="no-signals-message">
                    <i class="fas fa-info-circle"></i>
                    <span>No active signals</span>
                </div>
            `;
        } else {
            container.innerHTML = this.signals.map(signal => this.generateSignalCard(signal)).join('');
        }
    }
    
    // 신호 통계 업데이트
    updateSignalsStats() {
        const totalCount = this.signals.length;
        const buyCount = this.signals.filter(s => s.type.toLowerCase() === 'buy').length;
        const sellCount = this.signals.filter(s => s.type.toLowerCase() === 'sell').length;
        
        const totalElement = document.getElementById('signals-total-count');
        const buyElement = document.getElementById('signals-buy-count');
        const sellElement = document.getElementById('signals-sell-count');
        
        if (totalElement) totalElement.textContent = totalCount;
        if (buyElement) buyElement.textContent = buyCount;
        if (sellElement) sellElement.textContent = sellCount;
    }
    
    // 신호 새로고침
    refreshSignals() {
        console.log('🔄 Refreshing signals...');
        this.fetchLatestSignals();
    }
    
    // 신호 초기화
    clearSignals() {
        this.signals = [];
        this.updateSignalsDisplay();
        this.updateSignalsStats();
        console.log('🗑️ All signals cleared');
    }
    
    // 최신 신호 가져오기 (API 호출)
    async fetchLatestSignals() {
        try {
            const selectedCoin = window.selectedKrwCoin || 'BTC';
            const response = await fetch(`/api/trading-signals?coin=${selectedCoin}`);
            const data = await response.json();
            
            if (data.status === 'success' && data.signals) {
                // 기존 신호와 새로운 신호 병합
                const newSignals = data.signals.filter(newSignal => 
                    !this.signals.some(existingSignal => 
                        existingSignal.type === newSignal.type && 
                        existingSignal.timeframe === newSignal.timeframe &&
                        Math.abs(existingSignal.timestamp - newSignal.timestamp) < 60000 // 1분 이내
                    )
                );
                
                newSignals.forEach(signal => this.addSignal(signal));
            }
        } catch (error) {
            console.error('❌ Failed to fetch signals:', error);
        }
    }
    
    // 신호 알림 표시
    showSignalNotification(signal) {
        // 브라우저 알림 (사용자 권한 필요)
        if (Notification.permission === 'granted') {
            new Notification(`Trading Signal: ${signal.type.toUpperCase()}`, {
                body: `${signal.timeframe} - Strength: ${(signal.strength * 100).toFixed(0)}%`,
                icon: '/favicon.ico'
            });
        }
        
        // 화면 내 알림
        this.showToastNotification(signal);
    }
    
    // 토스트 알림 표시
    showToastNotification(signal) {
        const toast = document.createElement('div');
        toast.className = `signal-toast ${signal.type.toLowerCase()}`;
        toast.innerHTML = `
            <div class="toast-header">
                <i class="fas fa-${signal.type.toLowerCase() === 'buy' ? 'arrow-up' : 'arrow-down'}"></i>
                <span>${signal.type.toUpperCase()} Signal</span>
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="toast-body">
                <div>Timeframe: ${signal.timeframe}</div>
                <div>Strength: ${(signal.strength * 100).toFixed(0)}%</div>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // 5초 후 자동 제거
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
    
    // 초기화
    async initialize() {
        if (this.isInitialized) {
            console.log('🔄 Active Signals already initialized');
            return;
        }
        
        console.log('🚀 Initializing Active Signals...');
        
        try {
            // 초기 신호 로드
            await this.fetchLatestSignals();
            
            // 실시간 업데이트 시작 (30초마다)
            this.updateInterval = setInterval(() => {
                this.fetchLatestSignals();
            }, 30000);
            
            this.isInitialized = true;
            console.log('✅ Active Signals initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Active Signals:', error);
        }
    }
    
    // Trading Dashboard 내부에 Active Signals 로드
    loadActiveSignalsToTradingDashboard() {
        const container = document.getElementById('active-signals-container');
        if (!container) {
            console.warn('⚠️ Active signals container not found in Trading Dashboard');
            return;
        }
        
        // 컨테이너가 이미 Trading Dashboard 내부에 있으므로 HTML만 업데이트
        this.updateSignalsDisplay();
        this.updateSignalsStats();
        
        console.log('🚦 Active Signals loaded to Trading Dashboard');
    }
    
    // 정리
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        this.signals = [];
        this.isInitialized = false;
        
        console.log('🧹 Active Signals destroyed');
    }
}

// 전역 인스턴스 생성
window.activeSignalsManager = new ActiveSignalsManager();

// 전역 함수들
window.addTradingSignal = (signal) => {
    window.activeSignalsManager.addSignal(signal);
};

window.clearAllSignals = () => {
    window.activeSignalsManager.clearSignals();
};

window.refreshSignals = () => {
    window.activeSignalsManager.refreshSignals();
};

// 테스트용 신호 추가 함수
window.addTestSignal = (type = 'buy') => {
    const testSignal = {
        type: type,
        strength: Math.random() * 0.4 + 0.6, // 60-100%
        timeframe: ['1m', '5m', '15m', '1h', '4h'][Math.floor(Math.random() * 5)],
        price: Math.floor(Math.random() * 50000000) + 150000000, // 150M-200M
        reason: `Test ${type.toUpperCase()} signal - ${Math.random().toString(36).substr(2, 8)}`
    };
    
    window.activeSignalsManager.addSignal(testSignal);
    console.log('🧪 Test signal added:', testSignal);
};

    // 페이지 로드 시 초기화
    document.addEventListener('DOMContentLoaded', () => {
        // 약간의 지연 후 초기화 (다른 모듈들이 로드된 후)
        setTimeout(() => {
            window.activeSignalsManager.initialize();
            // Trading Dashboard가 로드되면 Active Signals도 함께 로드됨
        }, 1000);
    });

console.log('✅ Active Signals Module loaded');
