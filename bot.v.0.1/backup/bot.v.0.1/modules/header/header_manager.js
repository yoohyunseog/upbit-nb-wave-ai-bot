// ===== Header Manager - JavaScript =====

class HeaderManager {
    constructor() {
        this.isInitialized = false;
        this.updateInterval = null;
        this.lastUpdateTime = null;
        this.updateCount = 0;
        this.maxUpdatesPerMinute = 1; // 1분에 1번
        this.updateIntervalMs = 60000; // 1분마다 업데이트
    }
    
    initialize() {
        if (this.isInitialized) return;
        
        console.log('🔄 Initializing Header Manager...');
        
        // 즉시 기본값 설정 (강제로 0으로 설정)
        this.forceSetDefaultValues();
        
        // 백그라운드 업데이트 시작
        this.startBackgroundUpdate();
        
        // 5초 후 초기 데이터 확인
        setTimeout(() => {
            this.checkInitialData();
        }, 5000);
        
        this.isInitialized = true;
        console.log('✅ Header Manager initialized');
    }
    
    setDefaultValues() {
        console.log('🔄 Setting default header values...');
        
        const mineralElement = document.getElementById('mineral-count');
        const gasElement = document.getElementById('gas-count');
        const supplyElement = document.getElementById('supply-count');
        
        if (mineralElement) {
            mineralElement.textContent = '0';
            mineralElement.title = '총 자산: ₩0';
            console.log('✅ Mineral count set to 0');
        }
        
        if (gasElement) {
            gasElement.textContent = '0';
            gasElement.style.color = '#888888'; // 기본 회색
            gasElement.title = '매수 가능한 KRW: ₩0';
            console.log('✅ Gas count (available balance) set to 0');
        }
        
        if (supplyElement) {
            supplyElement.textContent = '0/100';
            supplyElement.title = 'BTC 자산 비율: 0%';
            console.log('✅ Supply count set to 0/100');
        }
        
        // 오른쪽 패널 기본값 설정
        this.setDefaultRightPanelValues();
        
        // Asset Display 기본값 설정
        if (window.assetDisplayManager) {
            window.assetDisplayManager.updateAssetDisplay(0, 0, 0);
        }
    }
    
    // 오른쪽 패널 기본값 설정
    setDefaultRightPanelValues() {
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        const coinNameElement = document.getElementById('selected-coin-name');
        const coinBalanceElement = document.getElementById('selected-coin-balance');
        const coinValueElement = document.getElementById('selected-coin-value');
        const coinPriceElement = document.getElementById('selected-coin-price');
        const coinPnlElement = document.getElementById('selected-coin-pnl');
        const coinAvgPriceElement = document.getElementById('selected-coin-avg-price');
        
        if (coinNameElement) {
            coinNameElement.textContent = `${selectedCoin}/KRW`;
        }
        
        if (coinBalanceElement) {
            coinBalanceElement.textContent = '0.00000000';
        }
        
        if (coinValueElement) {
            coinValueElement.textContent = '₩0';
        }
        
        if (coinPriceElement) {
            coinPriceElement.textContent = '₩0';
        }
        
        if (coinPnlElement) {
            coinPnlElement.textContent = '수익율: 0.00%';
            coinPnlElement.style.color = '#888888';
        }
        
        if (coinAvgPriceElement) {
            coinAvgPriceElement.textContent = '평균단가: ₩0';
        }
    }
    
    // 강제로 기본값 설정 (다른 시스템의 간섭 방지)
    forceSetDefaultValues() {
        console.log('🔄 Force setting default header values...');
        
        const mineralElement = document.getElementById('mineral-count');
        const gasElement = document.getElementById('gas-count');
        const supplyElement = document.getElementById('supply-count');
        
        if (mineralElement) {
            mineralElement.textContent = '0';
            mineralElement.title = '총 자산: ₩0';
            console.log('✅ Mineral count force set to 0');
        }
        
        if (gasElement) {
            gasElement.textContent = '0';
            gasElement.style.color = '#888888'; // 기본 회색
            gasElement.title = '매수 가능한 KRW: ₩0';
            console.log('✅ Gas count force set to 0');
        }
        
        if (supplyElement) {
            supplyElement.textContent = '0/100';
            supplyElement.title = 'BTC 자산 비율: 0%';
            console.log('✅ Supply count force set to 0/100');
        }
        
        // 다른 시스템이 업데이트하지 못하도록 잠시 대기
        setTimeout(() => {
            this.setDefaultValues();
        }, 100);
        
        // Asset Display 강제 기본값 설정
        if (window.assetDisplayManager) {
            window.assetDisplayManager.updateAssetDisplay(0, 0, 0);
        }
    }
    
    updateHeader(walletData = null) {
        console.log('🔄 Header update called with:', walletData);
        
        // walletData가 없으면 저장된 데이터 사용
        if (!walletData) {
            walletData = window.sharedWalletData;
        }
        
        if (!walletData) {
            console.log('⚠️ No wallet data available for header update');
            this.setDefaultValues();
            return;
        }
        
        // 헤더 업데이트 상태 표시
        this.showUpdateStatus('🔄 Updating wallet data...');
        
        // 데이터 구조 확인 및 정규화
        let summary = this.normalizeWalletData(walletData);
        
        if (!summary) {
            console.log('❌ Invalid wallet data structure');
            this.showUpdateStatus('❌ Invalid wallet data');
            return;
        }
        
        console.log('📊 Summary data:', summary);
        
        // 각 카운터 업데이트
        this.updateMineralCounter(summary);
        this.updateGasCounter(summary);
        this.updateSupplyCounter(summary);
        
        // 오른쪽 패널 업데이트 - asset_display.js로 이동됨
        // 이 기능은 modules/asset/asset_display.js에서 관리됩니다
        
        // Asset Display 업데이트
        this.updateAssetDisplay(summary);
        
        // 업데이트 완료 상태 메시지
        this.showUpdateStatus('✅ Wallet data updated');
        
        // 전역 데이터 저장
        window.globalWalletData = walletData;
    }
    
    normalizeWalletData(walletData) {
        if (!walletData) {
            console.log('❌ No wallet data provided');
            return null;
        }
        
        // 다양한 데이터 구조 지원
        if (walletData.summary) {
            return walletData.summary;
        } else if (walletData.data) {
            return walletData.data;
        } else if (walletData.total_value !== undefined) {
            // 직접 데이터 객체인 경우
            return walletData;
        } else {
            console.log('❌ Unknown wallet data structure');
            console.log('Expected: {summary: {...}} or {data: {...}} or direct data object');
            console.log('Received:', walletData);
            return null;
        }
    }
    
    updateMineralCounter(summary) {
        const mineralElement = document.getElementById('mineral-count');
        console.log('💰 Mineral element found:', mineralElement);
        
        if (mineralElement) {
            const totalValue = summary.total_value || 0;
            console.log('💎 Total value:', totalValue);
            mineralElement.textContent = totalValue.toLocaleString();
            
            // 툴팁 업데이트
            mineralElement.title = `총 자산: ₩${totalValue.toLocaleString()}`;
            
            // 값이 변경되었을 때 애니메이션 효과
            this.addUpdateAnimation(mineralElement);
        } else {
            console.log('❌ Mineral element not found');
        }
    }
    
    updateGasCounter(summary) {
        const gasElement = document.getElementById('gas-count');
        if (gasElement) {
            const krwBalance = summary.total_krw || 0;
            const availableBalance = Math.max(0, krwBalance); // 매수 가능한 금액 (음수 방지)
            
            // 매수 가능한 금액 표시
            gasElement.textContent = availableBalance.toLocaleString();
            
            // 툴팁 추가 (매수 가능한 금액임을 명시)
            gasElement.title = `매수 가능한 KRW: ₩${availableBalance.toLocaleString()}`;
            
            // 매수 가능한 금액에 따른 색상 변경
            this.updateGasColor(gasElement, availableBalance);
            
            // 값이 변경되었을 때 애니메이션 효과
            this.addUpdateAnimation(gasElement);
        }
    }
    
    updateGasColor(gasElement, availableBalance) {
        if (availableBalance > 100000) { // 10만원 이상
            gasElement.style.color = '#00ff00'; // 녹색
        } else if (availableBalance > 10000) { // 1만원 이상
            gasElement.style.color = '#ffff00'; // 노란색
        } else if (availableBalance > 0) { // 0원 초과
            gasElement.style.color = '#ff6b6b'; // 빨간색
        } else { // 0원
            gasElement.style.color = '#888888'; // 회색
        }
    }
    
    updateSupplyCounter(summary) {
        const supplyElement = document.getElementById('supply-count');
        if (supplyElement) {
            const btcBalance = summary.total_btc || 0;
            const btcValue = summary.total_btc_value || 0;
            const maxSupply = 1000000; // 최대 서플라이 (예: 100만원)
            
            const currentSupply = Math.min(btcValue, maxSupply);
            const supplyRatio = Math.floor((currentSupply / maxSupply) * 100);
            
            supplyElement.textContent = `${supplyRatio}/100`;
            
            // 툴팁 업데이트
            supplyElement.title = `BTC 자산 비율: ${supplyRatio}%`;
            
            // 서플라이가 높을 때 경고 색상
            this.updateSupplyColor(supplyElement, supplyRatio);
        }
        
        // 상단 우측 자산 표시 업데이트
        this.updateAssetDisplay(summary);
    }
    
    updateSupplyColor(supplyElement, supplyRatio) {
        if (supplyRatio > 80) {
            supplyElement.style.color = '#ff6b6b';
        } else if (supplyRatio > 50) {
            supplyElement.style.color = '#ffd700';
        } else {
            supplyElement.style.color = '#00ff00';
        }
    }
    
    // 오른쪽 패널 업데이트 - asset_display.js로 이동됨
    // 이 기능은 modules/asset/asset_display.js에서 관리됩니다
    
    addUpdateAnimation(element) {
        element.classList.add('value-updated');
        setTimeout(() => {
            element.classList.remove('value-updated');
        }, 1000);
    }
    
    showUpdateStatus(message) {
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            statusMessage.textContent = message;
            if (message.includes('✅')) {
                setTimeout(() => {
                    statusMessage.textContent = '';
                }, 2000);
            }
        }
    }
    
    startBackgroundUpdate() {
        console.log('🔄 Starting background header update...');
        
        // 즉시 첫 업데이트 실행
        this.performBackgroundUpdate();
        
        // 주기적 업데이트 시작
        this.updateInterval = setInterval(() => {
            this.performBackgroundUpdate();
        }, this.updateIntervalMs);
        
        // 1분마다 카운트 리셋
        setInterval(() => {
            this.updateCount = 0;
        }, 60000);
    }
    
    async performBackgroundUpdate() {
        try {
            console.log('🔄 Background header update using shared data...');
            
            // 저장된 데이터 사용 (API 호출 대신)
            if (window.sharedWalletData) {
                this.lastUpdateTime = new Date();
                
                // 헤더 업데이트
                this.updateHeader();
                
                console.log('✅ Background header update successful using shared data');
            } else {
                console.log('⚠️ No shared wallet data available');
                // 데이터가 없으면 기본값 설정
                this.setDefaultValues();
            }
        } catch (error) {
            console.error('❌ Background header update error:', error);
            // 에러 시 기본값 설정
            this.setDefaultValues();
        }
    }
    
    checkInitialData() {
        console.log('🔄 Initial header update check...');
        if (window.globalWalletData) {
            this.updateHeader(window.globalWalletData);
        } else {
            console.log('⚠️ No global wallet data available, using defaults');
            this.setDefaultValues();
        }
    }
    
    stopBackgroundUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        console.log('⏹️ Background header update stopped');
    }
    
    // 상단 우측 자산 표시 업데이트
    updateAssetDisplay(summary) {
        try {
            const btcBalance = summary.total_btc || 0;
            const krwBalance = summary.total_krw || 0;
            const totalValue = summary.total_value || 0;
            
            // 포트폴리오 비율 계산 (BTC 가치 / 총 자산)
            let portfolioRatio = 0;
            if (totalValue > 0) {
                const btcValue = summary.total_btc_value || 0;
                portfolioRatio = (btcValue / totalValue) * 100;
            }
            
            // 전역 함수 호출
            if (typeof window.updateAssetDisplay === 'function') {
                window.updateAssetDisplay(btcBalance, krwBalance, portfolioRatio);
            }
            
            console.log('💰 Asset display updated:', { btcBalance, krwBalance, portfolioRatio });
        } catch (error) {
            console.error('❌ Error updating asset display:', error);
        }
    }
    
    destroy() {
        this.stopBackgroundUpdate();
        this.isInitialized = false;
        console.log('🗑️ Header Manager destroyed');
    }
}

// 전역 인스턴스 생성
window.headerManager = new HeaderManager();

// 전역 함수로 등록 (기존 코드와의 호환성)
window.updateGameHeader = function(walletData) {
    if (window.headerManager) {
        window.headerManager.updateHeader(walletData);
    }
};

window.setDefaultHeaderValues = function() {
    if (window.headerManager) {
        window.headerManager.setDefaultValues();
    }
};

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 즉시 기본값을 0으로 설정
    const mineralElement = document.getElementById('mineral-count');
    const gasElement = document.getElementById('gas-count');
    const supplyElement = document.getElementById('supply-count');
    
    if (mineralElement) mineralElement.textContent = '0';
    if (gasElement) gasElement.textContent = '0';
    if (supplyElement) supplyElement.textContent = '0/100';
    
    // 오른쪽 패널 기본값 설정
    const selectedCoin = window.selectedKrwCoin || 'BTC';
    const coinNameElement = document.getElementById('selected-coin-name');
    const coinBalanceElement = document.getElementById('selected-coin-balance');
    const coinValueElement = document.getElementById('selected-coin-value');
    const coinPriceElement = document.getElementById('selected-coin-price');
    const coinPnlElement = document.getElementById('selected-coin-pnl');
    const coinAvgPriceElement = document.getElementById('selected-coin-avg-price');
    
    if (coinNameElement) coinNameElement.textContent = `${selectedCoin}/KRW`;
    if (coinBalanceElement) coinBalanceElement.textContent = '0.00000000';
    if (coinValueElement) coinValueElement.textContent = '₩0';
    if (coinPriceElement) coinPriceElement.textContent = '₩0';
    if (coinPnlElement) coinPnlElement.textContent = '수익율: 0.00%';
    if (coinAvgPriceElement) coinAvgPriceElement.textContent = '평균단가: ₩0';
    
    // Header Manager 초기화
    if (window.headerManager) {
        window.headerManager.initialize();
    }
});

// 페이지 언로드 시 정리
window.addEventListener('beforeunload', () => {
    if (window.headerManager) {
        window.headerManager.destroy();
    }
});
