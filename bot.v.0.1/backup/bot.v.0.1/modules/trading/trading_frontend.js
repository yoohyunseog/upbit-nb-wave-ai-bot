// ===== Trading Module - JavaScript Frontend =====

class TradingFrontend {
    constructor() {
        this.currentTimeframe = 'minute1';
        this.isAutoRotation = false;
        this.autoRotationInterval = null;
        this.chartCanvas = null;
        this.nbWaveCanvas = null;
        this.isInitialized = false;
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('📊 Initializing Trading Frontend...');
        
        // 캔버스 초기화
        this.initializeCanvases();
        
        // 초기 데이터 로드
        await this.loadTradingData();
        
        // 이벤트 리스너 등록
        this.registerEventListeners();
        
        this.isInitialized = true;
        console.log('✅ Trading Frontend initialized');
    }
    
    initializeCanvases() {
        // 차트 캔버스
        this.chartCanvas = document.getElementById('price-chart');
        if (this.chartCanvas) {
            this.chartCtx = this.chartCanvas.getContext('2d');
        }
        
        // N/B Wave 캔버스
        this.nbWaveCanvas = document.getElementById('nb-wave-chart');
        if (this.nbWaveCanvas) {
            this.nbWaveCtx = this.nbWaveCanvas.getContext('2d');
        }
    }
    
    registerEventListeners() {
        // 타임프레임 버튼 이벤트
        const timeframeButtons = document.querySelectorAll('.timeframe-btn');
        timeframeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const timeframe = e.target.dataset.timeframe;
                this.setTimeframe(timeframe);
            });
        });
        
        // 자동 순환 토글
        const autoRotationBtn = document.getElementById('auto-rotation-btn');
        if (autoRotationBtn) {
            autoRotationBtn.addEventListener('click', () => {
                this.toggleAutoRotation();
            });
        }
    }
    
    async loadTradingData() {
        try {
            const response = await fetch(`/api/trading/data/${this.currentTimeframe}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateCharts(data.data);
            } else {
                console.error('Failed to load trading data:', data.message);
            }
        } catch (error) {
            console.error('Error loading trading data:', error);
        }
    }
    
    updateCharts(data) {
        // 가격 차트 업데이트
        this.updatePriceChart(data);
        
        // N/B Wave 차트 업데이트
        this.updateNbWaveChart(data);
        
        // 요약 정보 업데이트
        this.updateSummary(data.summary);
    }
    
    updatePriceChart(data) {
        if (!this.chartCtx) return;
        
        const canvas = this.chartCanvas;
        const ctx = this.chartCtx;
        
        // 캔버스 클리어
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!data.prices || data.prices.length === 0) return;
        
        const prices = data.prices;
        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;
        
        // 가격 범위 계산
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;
        
        // 스케일 계산
        const xScale = (width - 2 * padding) / (prices.length - 1);
        const yScale = (height - 2 * padding) / priceRange;
        
        // 차트 그리기
        ctx.beginPath();
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        
        prices.forEach((price, index) => {
            const x = padding + index * xScale;
            const y = height - padding - (price - minPrice) * yScale;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // 축 그리기
        this.drawAxes(ctx, width, height, padding, prices, minPrice, maxPrice);
    }
    
    updateNbWaveChart(data) {
        if (!this.nbWaveCtx) return;
        
        const canvas = this.nbWaveCanvas;
        const ctx = this.nbWaveCtx;
        
        // 캔버스 클리어
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (!data.zones || data.zones.length === 0) return;
        
        const zones = data.zones;
        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;
        
        // 점 그리기
        const xScale = (width - 2 * padding) / (zones.length - 1);
        const yScale = (height - 2 * padding) / 100; // 0-100% 범위
        
        zones.forEach((zone, index) => {
            const x = padding + index * xScale;
            const y = height - padding - (zone.strength * 100) * yScale;
            
            // 색상 설정
            if (zone.zone === 'ORANGE') {
                ctx.fillStyle = '#ff6600';
            } else if (zone.zone === 'BLUE') {
                ctx.fillStyle = '#0066ff';
            } else {
                ctx.fillStyle = '#666666';
            }
            
            // 점 그리기
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // 선 연결 (이전 점과)
            if (index > 0) {
                const prevZone = zones[index - 1];
                const prevX = padding + (index - 1) * xScale;
                const prevY = height - padding - (prevZone.strength * 100) * yScale;
                
                ctx.beginPath();
                ctx.strokeStyle = zone.zone === 'ORANGE' ? '#ff6600' : 
                                 zone.zone === 'BLUE' ? '#0066ff' : '#666666';
                ctx.lineWidth = 1;
                ctx.moveTo(prevX, prevY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        });
        
        // 축 그리기
        this.drawNbWaveAxes(ctx, width, height, padding, zones);
    }
    
    drawAxes(ctx, width, height, padding, prices, minPrice, maxPrice) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        
        // Y축
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // X축
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Y축 라벨
        const steps = 5;
        for (let i = 0; i <= steps; i++) {
            const y = padding + (height - 2 * padding) * i / steps;
            const price = maxPrice - (maxPrice - minPrice) * i / steps;
            ctx.fillText(price.toLocaleString(), 5, y + 4);
        }
    }
    
    drawNbWaveAxes(ctx, width, height, padding, zones) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        
        // Y축
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // X축
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();
        
        // Y축 라벨 (0-100%)
        for (let i = 0; i <= 5; i++) {
            const y = padding + (height - 2 * padding) * i / 5;
            const value = 100 - (100 * i / 5);
            ctx.fillText(`${value}%`, 5, y + 4);
        }
    }
    
    updateSummary(summary) {
        const summaryElement = document.getElementById('trading-summary');
        if (!summaryElement) return;
        
        summaryElement.innerHTML = `
            <div class="summary-item">
                <span class="label">Current Price:</span>
                <span class="value">₩${summary.current_price?.toLocaleString() || 'N/A'}</span>
            </div>
            <div class="summary-item">
                <span class="label">24h Change:</span>
                <span class="value ${summary.price_change_24h >= 0 ? 'positive' : 'negative'}">
                    ${summary.price_change_24h || 0}%
                </span>
            </div>
            <div class="summary-item">
                <span class="label">Orange Zones:</span>
                <span class="value orange">${summary.orange || 0}</span>
            </div>
            <div class="summary-item">
                <span class="label">Blue Zones:</span>
                <span class="value blue">${summary.blue || 0}</span>
            </div>
        `;
    }
    
    async setTimeframe(timeframe) {
        this.currentTimeframe = timeframe;
        
        // UI 업데이트
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeBtn = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
        
        // 데이터 새로고침
        await this.loadTradingData();
    }
    
    toggleAutoRotation() {
        this.isAutoRotation = !this.isAutoRotation;
        
        if (this.isAutoRotation) {
            this.startAutoRotation();
        } else {
            this.stopAutoRotation();
        }
        
        // UI 업데이트
        const btn = document.getElementById('auto-rotation-btn');
        if (btn) {
            btn.textContent = this.isAutoRotation ? 'Stop Auto' : 'Start Auto';
            btn.classList.toggle('active', this.isAutoRotation);
        }
    }
    
    startAutoRotation() {
        const timeframes = ['minute1', 'minute3', 'minute5', 'minute15', 'minute30', 'hour1', 'hour4', 'day'];
        let currentIndex = 0;
        
        this.autoRotationInterval = setInterval(() => {
            const timeframe = timeframes[currentIndex];
            this.setTimeframe(timeframe);
            
            currentIndex = (currentIndex + 1) % timeframes.length;
        }, 5000); // 5초마다 변경
    }
    
    stopAutoRotation() {
        if (this.autoRotationInterval) {
            clearInterval(this.autoRotationInterval);
            this.autoRotationInterval = null;
        }
    }
    
    destroy() {
        this.stopAutoRotation();
        this.isInitialized = false;
    }
}

// 전역 인스턴스
window.tradingFrontend = new TradingFrontend();
