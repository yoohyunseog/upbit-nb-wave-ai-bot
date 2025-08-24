// ===== Timeframe Cards Module =====
/**
 * Timeframe Cards Module
 * 
 * - 분봉 카드 표시
 * - 자동 순회 기능
 * - 클릭 이벤트 처리
 * - 시각적 상태 표시
 */

class TimeframeCards {
    constructor() {
        this.cards = [];
        this.currentTimeframe = 'minute10';
        this.autoRotateInterval = null;
        this.isAutoRotating = false;
        this.container = null;
        this.onTimeframeChange = null;
    }

    // 초기화
    async init(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Timeframe cards container not found: ${containerId}`);
            return false;
        }

        this.onTimeframeChange = options.onTimeframeChange || null;
        this.currentTimeframe = options.defaultTimeframe || 'minute10';
        
        this.render();
        this.bindEvents();
        
        // Settings에서 자동 순회 설정 확인
        const autoRotateEnabled = await this.getSettingFromStorage('timeframe_auto_rotate', true);
        if (autoRotateEnabled) {
            const interval = await this.getSettingFromStorage('auto_rotate_interval', 5);
            this.startAutoRotate(interval * 1000);
        }
        
        console.log('Timeframe Cards initialized with settings integration');
        return true;
    }
    
    // Settings에서 설정값 가져오기
    async getSettingFromStorage(key, defaultValue) {
        try {
            const settings = localStorage.getItem('8bit-settings');
            if (settings) {
                const parsedSettings = JSON.parse(settings);
                return parsedSettings[key] !== undefined ? parsedSettings[key] : defaultValue;
            }
        } catch (error) {
            console.error('Error reading setting from storage:', error);
        }
        return defaultValue;
    }

    // HTML 생성
    static generateHTML(containerId = 'timeframe-cards-container', showAutoRotate = true) {
        const timeframes = [
            { value: 'minute1', label: '1m', icon: '⏱️' },
            { value: 'minute3', label: '3m', icon: '⏱️' },
            { value: 'minute5', label: '5m', icon: '⏱️' },
            { value: 'minute10', label: '10m', icon: '⏱️' },
            { value: 'minute15', label: '15m', icon: '⏱️' },
            { value: 'minute30', label: '30m', icon: '⏱️' },
            { value: 'minute60', label: '1h', icon: '⏱️' },
            { value: 'day', label: '1D', icon: '⏱️' }
        ];

        const cardsHTML = timeframes.map(tf => `
            <div class="timeframe-card" data-timeframe="${tf.value}">
                <div class="timeframe-icon">${tf.icon}</div>
                <div class="timeframe-name">${tf.label}</div>
                <div class="timeframe-status">대기</div>
            </div>
        `).join('');

        const autoRotateButton = showAutoRotate ? `
            <button id="btnAutoRotate" class="btn btn-sm btn-outline-primary">
                <i class="fas fa-play"></i> 자동 순회
            </button>
        ` : '';

        return `
            <div class="timeframe-cards-section">
                <div class="timeframe-header">
                    <h4>분봉 선택</h4>
                    <div class="timeframe-controls">
                        <span id="currentTimeframe" class="badge bg-primary">10m</span>
                        ${autoRotateButton}
                    </div>
                </div>
                <div id="${containerId}" class="timeframe-cards-container">
                    ${cardsHTML}
                </div>
            </div>
        `;
    }

    // 렌더링
    render() {
        if (!this.container) return;

        // 카드 요소들 가져오기
        this.cards = Array.from(this.container.querySelectorAll('.timeframe-card'));
        
        // 초기 상태 설정
        this.updateActiveCard();
        this.updateCurrentTimeframeBadge();
    }

    // 이벤트 바인딩
    bindEvents() {
        // 카드 클릭 이벤트
        this.cards.forEach(card => {
            card.addEventListener('click', () => {
                const timeframe = card.getAttribute('data-timeframe');
                this.selectTimeframe(timeframe);
            });
        });

        // 자동 순회 버튼 이벤트
        const autoRotateBtn = document.getElementById('btnAutoRotate');
        if (autoRotateBtn) {
            autoRotateBtn.addEventListener('click', () => {
                this.toggleAutoRotate();
            });
        }
    }

    // 분봉 선택
    selectTimeframe(timeframe) {
        if (this.currentTimeframe === timeframe) return;

        this.currentTimeframe = timeframe;
        this.updateActiveCard();
        this.updateCurrentTimeframeBadge();

        // 콜백 함수 호출
        if (this.onTimeframeChange && typeof this.onTimeframeChange === 'function') {
            this.onTimeframeChange(timeframe);
        }

        // 이벤트 발생
        this.dispatchTimeframeChangeEvent(timeframe);
    }

    // 활성 카드 업데이트
    updateActiveCard() {
        this.cards.forEach(card => {
            const timeframe = card.getAttribute('data-timeframe');
            const statusElement = card.querySelector('.timeframe-status');
            
            if (timeframe === this.currentTimeframe) {
                card.classList.add('active');
                if (statusElement) {
                    statusElement.textContent = '활성';
                    statusElement.className = 'timeframe-status active';
                }
            } else {
                card.classList.remove('active');
                if (statusElement) {
                    statusElement.textContent = '대기';
                    statusElement.className = 'timeframe-status';
                }
            }
        });
    }

    // 현재 분봉 배지 업데이트
    updateCurrentTimeframeBadge() {
        const badge = document.getElementById('currentTimeframe');
        if (badge) {
            const label = this.getTimeframeLabel(this.currentTimeframe);
            badge.textContent = label;
        }
    }

    // 분봉 라벨 가져오기
    getTimeframeLabel(timeframe) {
        const labels = {
            'minute1': '1m',
            'minute3': '3m',
            'minute5': '5m',
            'minute10': '10m',
            'minute15': '15m',
            'minute30': '30m',
            'minute60': '1h',
            'day': '1D'
        };
        return labels[timeframe] || timeframe;
    }

    // 자동 순회 토글
    toggleAutoRotate() {
        if (this.isAutoRotating) {
            this.stopAutoRotate();
        } else {
            this.startAutoRotate();
        }
    }

    // 자동 순회 시작
    startAutoRotate(interval = 5000) {
        if (this.isAutoRotating) return;

        this.isAutoRotating = true;
        this.autoRotateInterval = setInterval(() => {
            this.rotateToNextTimeframe();
        }, interval);

        this.updateAutoRotateButton();
        console.log('Auto rotate started');
    }

    // 자동 순회 중지
    stopAutoRotate() {
        if (!this.isAutoRotating) return;

        this.isAutoRotating = false;
        if (this.autoRotateInterval) {
            clearInterval(this.autoRotateInterval);
            this.autoRotateInterval = null;
        }

        this.updateAutoRotateButton();
        console.log('Auto rotate stopped');
    }

    // 다음 분봉으로 순회
    rotateToNextTimeframe() {
        const timeframes = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'day'];
        const currentIndex = timeframes.indexOf(this.currentTimeframe);
        const nextIndex = (currentIndex + 1) % timeframes.length;
        const nextTimeframe = timeframes[nextIndex];
        
        this.selectTimeframe(nextTimeframe);
    }

    // 자동 순회 버튼 업데이트
    updateAutoRotateButton() {
        const btn = document.getElementById('btnAutoRotate');
        if (!btn) return;

        if (this.isAutoRotating) {
            btn.innerHTML = '<i class="fas fa-pause"></i> 순회 중지';
            btn.className = 'btn btn-sm btn-outline-warning';
        } else {
            btn.innerHTML = '<i class="fas fa-play"></i> 자동 순회';
            btn.className = 'btn btn-sm btn-outline-primary';
        }
    }

    // 이벤트 발생
    dispatchTimeframeChangeEvent(timeframe) {
        const event = new CustomEvent('timeframeChanged', {
            detail: { timeframe, label: this.getTimeframeLabel(timeframe) }
        });
        document.dispatchEvent(event);
    }

    // 현재 분봉 가져오기
    getCurrentTimeframe() {
        return this.currentTimeframe;
    }

    // 분봉 변경 콜백 설정
    setTimeframeChangeCallback(callback) {
        this.onTimeframeChange = callback;
    }

    // 정리
    destroy() {
        this.stopAutoRotate();
        this.cards = [];
        this.container = null;
        this.onTimeframeChange = null;
    }
}

// 전역 인스턴스 생성
window.timeframeCards = new TimeframeCards();

// 모듈 내보내기
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TimeframeCards;
}
