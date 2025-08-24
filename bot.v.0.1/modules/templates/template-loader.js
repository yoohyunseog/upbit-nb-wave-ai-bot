// Template Loader Module
// 스크립트 로더 역할만 담당 - 모든 함수는 분리된 모듈에서 로드

// 모듈 로드 확인 및 초기화
function initializeTrainerModules() {
    //console.log('🔧 트레이너 모듈 초기화 중...');
    
    // 모듈 로드 상태 확인
    const modules = {
        'trainerMovementController': window.trainerMovementController,
        'trainerVisualEffects': window.trainerVisualEffects,
        'trainerDecisionHandler': window.trainerDecisionHandler,
        'gameStateManager': window.gameStateManager,
        'logManager': window.logManager,
        'decisionSystem': window.decisionSystem,
        'learningSystem': window.learningSystem,
        'currentPriceManager': window.currentPriceManager,
        'sellProfitCalculator': window.sellProfitCalculator,
        'btcMarketCalculator': window.btcMarketCalculator,
        'btcMarketLearningHandler': window.btcMarketLearningHandler,
        'trainerActivityLogger': window.trainerActivityLogger
    };
    
    let allModulesLoaded = true;
    
    Object.entries(modules).forEach(([name, module]) => {
        if (module) {
            //console.log(`✅ ${name} 모듈 로드됨`);
            if (window.logManager) {
                window.logManager.addLog(`✅ ${name} 모듈 로드됨`);
            }
        } else {
            console.warn(`⚠️ ${name} 모듈이 로드되지 않음`);
            if (window.logManager) {
                window.logManager.addLog(`⚠️ ${name} 모듈이 로드되지 않음`);
            }
            allModulesLoaded = false;
        }
    });
    
    if (allModulesLoaded) {
        //console.log('🔧 모든 트레이너 모듈이 성공적으로 로드됨');
        if (window.logManager) {
            window.logManager.addLog('🔧 모든 트레이너 모듈이 성공적으로 로드됨');
        }
    } else {
        console.warn('⚠️ 일부 트레이너 모듈이 로드되지 않음 - 폴백 모드로 동작');
        if (window.logManager) {
            window.logManager.addLog('⚠️ 일부 트레이너 모듈이 로드되지 않음 - 폴백 모드로 동작');
        }
    }
    
    //console.log('🔧 트레이너 모듈 초기화 완료');
}

class TemplateLoader {
    constructor() {
        this.templates = new Map();
    }

    // HTML 템플릿 파일 로드
    async loadTemplate(templatePath) {
        if (this.templates.has(templatePath)) {
            return this.templates.get(templatePath);
        }

        try {
            const response = await fetch(templatePath);
            if (!response.ok) {
                throw new Error(`Failed to load template: ${templatePath}`);
            }
            const html = await response.text();
            this.templates.set(templatePath, html);
            return html;
        } catch (error) {
            console.error('Template loading error:', error);
            return '';
        }
    }

    // Active Signals 템플릿 로드
    async loadActiveSignalsTemplate() {
        return await this.loadTemplate('./templates/active-signals-template.html');
    }

    // 메시지 출력 함수
    showMessage(message, duration = 3000) {
        const messageElement = document.getElementById('message-text');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.style.color = '#00ff00';
            
            if (duration > 0) {
                setTimeout(() => {
                    messageElement.textContent = '시스템 대기 중...';
                    messageElement.style.color = '#00ff00';
                }, duration);
            }
        }
    }

    // 경고 메시지 출력
    showWarning(message, duration = 3000) {
        const messageElement = document.getElementById('message-text');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.style.color = '#ff8800';
            
            if (duration > 0) {
                setTimeout(() => {
                    messageElement.textContent = '시스템 대기 중...';
                    messageElement.style.color = '#00ff00';
                }, duration);
            }
        }
    }

    // 오류 메시지 출력
    showError(message, duration = 3000) {
        const messageElement = document.getElementById('message-text');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.style.color = '#ff0088';
            
            if (duration > 0) {
                setTimeout(() => {
                    messageElement.textContent = '시스템 대기 중...';
                    messageElement.style.color = '#00ff00';
                }, duration);
            }
        }
    }

    // 템플릿에 데이터 바인딩
    bindTemplate(template, data) {
        let boundTemplate = template;
        
        if (data && data.signals) {
            const signalsHtml = data.signals.map(signal => `
                <div class="signal-card ${signal.type}">
                    <div class="signal-type">${signal.type.toUpperCase()}</div>
                    <div class="signal-strength">Strength: ${(signal.strength * 100).toFixed(0)}%</div>
                    <div class="signal-timeframe">${signal.timeframe}</div>
                </div>
            `).join('');

            boundTemplate = boundTemplate.replace(
                '<div class="signals-grid">',
                `<div class="signals-grid">${signalsHtml}`
            );
        }
        
        // 템플릿이 DOM에 추가된 후 게임 초기화
        setTimeout(() => {
            //console.log('🎮 Attempting to initialize AI models game...');
            
            if (typeof Phaser === 'undefined') {
                console.error('❌ Phaser library not loaded');
                return;
            }
            
            const container = document.getElementById('floating-ball-game');
            if (container) {
                // 게임 초기화는 별도 모듈에서 처리
                if (window.gameInitializer && typeof window.gameInitializer.initializeGame === 'function') {
                    window.gameInitializer.initializeGame(container);
                } else {
                    console.warn('⚠️ GameInitializer 모듈이 로드되지 않음');
                }
            } else {
                console.error('❌ Floating ball game container not found');
            }
        }, 1000);
        
        return boundTemplate;
    }
}

// 전역 인스턴스 생성
window.templateLoader = new TemplateLoader();
