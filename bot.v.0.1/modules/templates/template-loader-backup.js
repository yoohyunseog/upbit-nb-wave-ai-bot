// Template Loader Module

// 분리된 모듈들 사용
// - learning-system.js: 학습 및 예측 시스템
// - decision-system.js: 의사결정 시스템  
// - game-state-manager.js: 게임 상태 관리
// - log-manager.js: 로그 관리
// - trainer-movement-controller.js: 트레이너 이동 컨트롤러
// - trainer-visual-effects.js: 트레이너 시각적 효과
// - trainer-decision-handler.js: 트레이너 의사결정 핸들러

// 모듈 로드 확인 및 초기화
function initializeTrainerModules() {
    console.log('🔧 트레이너 모듈 초기화 중...');
    
    // 모듈 로드 상태 확인
    const modules = {
        'trainerMovementController': window.trainerMovementController,
        'trainerVisualEffects': window.trainerVisualEffects,
        'trainerDecisionHandler': window.trainerDecisionHandler
    };
    
    let allModulesLoaded = true;
    
    Object.entries(modules).forEach(([name, module]) => {
        if (module) {
            console.log(`✅ ${name} 모듈 로드됨`);
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
        console.log('🔧 모든 트레이너 모듈이 성공적으로 로드됨');
        if (window.logManager) {
            window.logManager.addLog('🔧 모든 트레이너 모듈이 성공적으로 로드됨');
        }
    } else {
        console.warn('⚠️ 일부 트레이너 모듈이 로드되지 않음 - 폴백 모드로 동작');
        if (window.logManager) {
            window.logManager.addLog('⚠️ 일부 트레이너 모듈이 로드되지 않음 - 폴백 모드로 동작');
        }
    }
    
    console.log('🔧 트레이너 모듈 초기화 완료');
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
            console.log('🎮 Attempting to initialize AI models game...');
            
            if (typeof Phaser === 'undefined') {
                console.error('❌ Phaser library not loaded');
                return;
            }
            
            const container = document.getElementById('floating-ball-game');
            if (container) {
                const config = {
                    type: Phaser.AUTO,
                    parent: 'floating-ball-game',
                    width: container.offsetWidth || 400,
                    height: 500,
                    backgroundColor: '#000011',
                    disableContextMenu: true,
                    input: {
                        mouse: {
                            preventDefaultWheel: false
                        }
                    },
                    physics: {
                        default: 'arcade',
                        arcade: {
                            gravity: { y: 0 },
                            debug: false
                        }
                    },
                    scene: {
                        create: function() {
                            console.log('🎮 Creating AI models system...');
                            
                            // 2D 맵 그리드 생성
                            const gridSize = 20;
                            const cols = Math.floor(config.width / gridSize);
                            const rows = Math.floor(config.height / gridSize);
                            
                            const graphics = this.add.graphics();
                            graphics.lineStyle(1, 0x00ff00, 0.3);
                            
                            for (let i = 0; i <= cols; i++) {
                                graphics.moveTo(i * gridSize, 0);
                                graphics.lineTo(i * gridSize, config.height);
                            }
                            
                            for (let i = 0; i <= rows; i++) {
                                graphics.moveTo(0, i * gridSize);
                                graphics.lineTo(config.width, i * gridSize);
                            }
                            
                            // N/B 길드 다각형 (전역 변수로 저장)
                            const guildPolygon = this.add.polygon(100, 100, [
                                0, -30, 22, -15, 22, 15, 0, 30, -22, 15, -22, -15
                            ], 0x00ff00);
                            guildPolygon.setOrigin(0.5, 0.5);
                            // 전역 변수로 저장하여 충돌 검사에 사용
                            window.nbGuildPolygon = guildPolygon;
                            
                            // N/B 길드 내부 구역 표시 원
                            const guildZoneIndicator = this.add.circle(100, 100, 15, 0x00d1ff); // 기본 파란색
                            guildZoneIndicator.setOrigin(0.5, 0.5);
                            
                            // N/B 길드 내부 구역 텍스트
                            const guildZoneText = this.add.text(100, 100, 'BLUE', {
                                fontSize: '8px',
                                fill: '#ffffff',
                                fontStyle: 'bold'
                            }).setOrigin(0.5);
                            
                            // N/B 길드 구역 표시 애니메이션
                            this.tweens.add({
                                targets: [guildZoneIndicator, guildZoneText],
                                scaleX: 1.1,
                                scaleY: 1.1,
                                duration: 2000,
                                yoyo: true,
                                repeat: -1,
                                ease: 'Sine.easeInOut'
                            });
                            
                            // BTC 시장 다각형 (전역 변수로 저장)
                            const marketPolygon = this.add.polygon(config.width - 100, config.height - 100, [
                                0, -35, 25, -17, 25, 17, 0, 35, -25, 17, -25, -17
                            ], 0x0088ff);
                            marketPolygon.setOrigin(0.5, 0.5);
                            // 전역 변수로 저장하여 충돌 검사에 사용
                            window.btcMarketPolygon = marketPolygon;
                            
                            // 다각형 라벨
                            this.add.text(100, 140, 'N/B 길드', {
                                fontSize: '12px',
                                fill: '#00ff00'
                            }).setOrigin(0.5);
                            
                            this.add.text(config.width - 100, config.height - 140, 'BTC 시장', {
                                fontSize: '12px',
                                fill: '#0088ff'
                            }).setOrigin(0.5);
                            
                            // 상단 매수/매도/대기 4각형들
                            const spacing = 120;
                            const startX = (config.width - (spacing * 2)) / 2;
                            const topY = 60;
                            
                            const buyPolygon = this.add.polygon(startX, topY, [
                                -20, -18, 20, -18, 20, 18, -20, 18
                            ], 0x00ff00);
                            buyPolygon.setOrigin(0.5, 0.5);
                            
                            const sellPolygon = this.add.polygon(startX + spacing, topY, [
                                -20, -18, 20, -18, 20, 18, -20, 18
                            ], 0xff0000);
                            sellPolygon.setOrigin(0.5, 0.5);
                            
                            const waitPolygon = this.add.polygon(startX + spacing * 2, topY, [
                                -20, -18, 20, -18, 20, 18, -20, 18
                            ], 0xffff00);
                            waitPolygon.setOrigin(0.5, 0.5);
                            
                            // 4각형 라벨
                            this.add.text(startX, topY + 30, '매수', {
                                fontSize: '10px',
                                fill: '#00ff00'
                            }).setOrigin(0.5);
                            
                            this.add.text(startX + spacing, topY + 30, '매도', {
                                fontSize: '10px',
                                fill: '#ff0000'
                            }).setOrigin(0.5);
                            
                            this.add.text(startX + spacing * 2, topY + 30, '대기', {
                                fontSize: '10px',
                                fill: '#ffff00'
                            }).setOrigin(0.5);
                            
                            // 신호 대기 센터 (화면 중앙)
                            const centerX = config.width / 2;
                            const centerY = config.height / 2;
                            
                            const signalWaitCenter = this.add.circle(centerX, centerY, 40, 0x88ccff);
                            signalWaitCenter.setStrokeStyle(3, 0xffffff);
                            
                            this.add.text(centerX, centerY - 10, '신호 대기', {
                                fontSize: '12px',
                                fill: '#ffffff',
                                fontStyle: 'bold'
                            }).setOrigin(0.5);
                            
                            this.add.text(centerX, centerY + 10, '센터', {
                                fontSize: '12px',
                                fill: '#ffffff',
                                fontStyle: 'bold'
                            }).setOrigin(0.5);
                            
                            // 5개의 AI 모델 생성 (4개 탐색 + 1개 트레이너)
                            const aiModels = [];
                            const modelColors = [0xff8800, 0x00ff88, 0x8800ff, 0xff0088, 0xffff00];
                            const modelNames = ['Explorer-1', 'Explorer-2', 'Explorer-3', 'Explorer-4', 'Trainer'];
                            const modelRoles = ['탐색', '탐색', '탐색', '탐색', '트레이너'];
                            
                            const initialPositions = [
                                { x: config.width / 2, y: config.height / 2 },
                                { x: config.width / 4, y: config.height / 4 },
                                { x: config.width * 3/4, y: config.height / 4 },
                                { x: config.width / 4, y: config.height * 3/4 },
                                { x: config.width * 3/4, y: config.height * 3/4 }
                            ];
                            
                            for (let i = 0; i < 5; i++) {
                                const circleRadius = i === 4 ? 20 : 10; // 트레이너는 20, 탐색자는 10
                                const fontSize = i === 4 ? '8px' : '6px'; // 트레이너는 8px, 탐색자는 6px
                                const roleFontSize = i === 4 ? '6px' : '5px'; // 트레이너는 6px, 탐색자는 5px
                                
                                const model = {
                                    circle: this.add.circle(initialPositions[i].x, initialPositions[i].y, circleRadius, modelColors[i]),
                                    name: this.add.text(initialPositions[i].x, initialPositions[i].y - (i === 4 ? 6 : 4), modelNames[i], {
                                        fontSize: fontSize,
                                        fill: '#ffffff',
                                        fontStyle: 'bold'
                                    }).setOrigin(0.5),
                                    role: this.add.text(initialPositions[i].x, initialPositions[i].y + (i === 4 ? 6 : 4), modelRoles[i], {
                                        fontSize: roleFontSize,
                                        fill: '#ffffff'
                                    }).setOrigin(0.5),
                                    targetX: initialPositions[i].x,
                                    targetY: initialPositions[i].y,
                                    targetAction: i === 4 ? '신호 대기' : '', // 트레이너 초기 액션 설정
                                    isExplorer: i < 4,
                                    isTrainer: i === 4,
                                    discoveredCoords: [],
                                    memoryIndex: 0,
                                    explorationTimer: 0
                                };
                                
                                model.circle.setOrigin(0.5, 0.5);
                                aiModels.push(model);
                            }
                            
                            // 학습 상태 표시
                            const learningStatus = this.add.text(config.width / 2, config.height - 30, 'AI 모델 시스템 시작', {
                                fontSize: '12px',
                                fill: '#00ff00'
                            }).setOrigin(0.5);
                            
                            const progressBar = this.add.graphics();
                            
                            // 연결선을 그리기 위한 그래픽 객체
                            const connectionLines = this.add.graphics();
                            
                            // N/B 길드와 트레이너 연결선
                            const guildTrainerConnection = this.add.graphics();
                             
                             // 트레이너 활동 대화창
                             const trainerDialog = this.add.text(10, config.height - 60, '🎯 트레이너: AI 시스템 시작 중...', {
                                 fontSize: '10px',
                                 fill: '#ffff00',
                                 backgroundColor: '#000000',
                                 padding: { x: 5, y: 2 }
                             }).setOrigin(0, 0.5);
                             
                                                         // N/B 코인 아이템 관리 시스템
                            let nbCoins = 0;
                            let nbMinerals = 0.0; // N/B 코인 미네랄 누적 수익률
                            let nbCoinItems = [];
                            let lastBuyAction = false;
                            let lastSellAction = false;
                            
                            // N/B 코인 아이템 생성 함수
                            const createNBCoinItem = () => {
                                const item = {
                                      polygon: this.add.polygon(
                                          Math.random() * (config.width - 100) + 50,
                                          Math.random() * (config.height - 100) + 50,
                                          [0, -8, 6, -4, 6, 4, 0, 8, -6, 4, -6, -4],
                                          0xffaa00
                                      ),
                                      collected: false,
                                      connectionLines: [] // 탐색자와의 연결선들을 저장
                                  };
                                  item.polygon.setOrigin(0.5, 0.5);
                                  
                                  // 회전 애니메이션
                                  this.tweens.add({
                                      targets: item.polygon,
                                      rotation: Math.PI * 2,
                                      duration: 3000,
                                      repeat: -1,
                                      ease: 'Linear'
                                  });
                                  
                                  nbCoinItems.push(item);
                                  console.log(`🪙 N/B 코인 아이템 생성: 탐색자 연결 대기 중...`);
                              };
                             
                             // N/B 코인 개수 표시
                             const nbCoinDisplay = this.add.text(config.width - 10, 10, 'N/B 코인: 0개', {
                                 fontSize: '12px',
                                 fill: '#ffaa00',
                                 backgroundColor: '#000000',
                                 padding: { x: 5, y: 2 }
                             }).setOrigin(1, 0);
                             
                             // N/B 코인 미네랄 누적 수익률 표시
                             const nbMineralDisplay = this.add.text(config.width - 10, 30, 'N/B 미네랄: 0.00%', {
                                 fontSize: '12px',
                                 fill: '#00ffaa',
                                 backgroundColor: '#000000',
                                 padding: { x: 5, y: 2 }
                             }).setOrigin(1, 0);
                             
                             // 매수 전 예상 수익률 표시
                             const buyProfitRateDisplay = this.add.text(10, 10, '매수 전 예상 수익률: 0.00%', {
                                 fontSize: '12px',
                                 fill: '#00ff88',
                                 backgroundColor: '#000000',
                                 padding: { x: 5, y: 2 }
                             }).setOrigin(0, 0);
                             
                             // 매도 전 예상 수익률 표시
                             const sellProfitRateDisplay = this.add.text(10, 30, '매도 전 예상 수익률: 0.00%', {
                                 fontSize: '12px',
                                 fill: '#ff0088',
                                 backgroundColor: '#000000',
                                 padding: { x: 5, y: 2 }
                             }).setOrigin(0, 0);
                             
                             // HTML 초기화 버튼 이벤트 리스너 추가
                             const resetGame = () => {
                                 console.log('🔄 초기화 버튼 클릭됨');
                                 
                                 // 초기화 확인 메시지
                                 const dialogMessage = '🔄 초기화 중... 모든 데이터를 리셋합니다.';
                                 trainerDialog.setText(dialogMessage);
                                 if (window.logManager) {
                                     window.logManager.addLog(dialogMessage);
                                 }
                                 
                                 // 모든 변수 초기화
                                 nbCoins = 0;
                                 nbMinerals = 0.0; // N/B 미네랄 누적 수익률 초기화
                                 buyPrice = 0;
                                 buyProfitRate = 0;
                                 sellProfitRate = 0;
                                 lastBuyAction = false;
                                 lastSellAction = false;
                                 
                                 // N/B 코인 아이템들 제거
                                 nbCoinItems.forEach(item => {
                                     if (item.polygon && item.polygon.destroy) {
                                         item.polygon.destroy();
                                     }
                                     item.connectionLines.forEach(line => {
                                         if (line && line.destroy) {
                                             line.destroy();
                                         }
                                     });
                                 });
                                 nbCoinItems = [];
                                 
                                 // AI 모델들 초기 위치로 리셋
                                 const initialPositions = [
                                     { x: config.width / 2, y: config.height / 2 },
                                     { x: config.width / 4, y: config.height / 4 },
                                     { x: config.width * 3/4, y: config.height / 4 },
                                     { x: config.width / 4, y: config.height * 3/4 },
                                     { x: config.width * 3/4, y: config.height * 3/4 }
                                 ];
                                 
                                 aiModels.forEach((model, index) => {
                                     model.circle.x = initialPositions[index].x;
                                     model.circle.y = initialPositions[index].y;
                                     model.name.x = initialPositions[index].x;
                                     model.name.y = initialPositions[index].y - (index === 4 ? 6 : 4);
                                     model.role.x = initialPositions[index].x;
                                     model.role.y = initialPositions[index].y + (index === 4 ? 6 : 4);
                                     model.targetX = initialPositions[index].x;
                                     model.targetY = initialPositions[index].y;
                                     model.discoveredCoords = [];
                                     model.memoryIndex = 0;
                                     model.explorationTimer = 0;
                                     
                                     // 트레이너 기본 색상으로 복원
                                     if (model.isTrainer) {
                                         model.circle.setFillStyle(0xffff00);
                                     }
                                 });
                                 
                                 // UI 업데이트
                                 nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                 nbMineralDisplay.setText(`N/B 미네랄: ${nbMinerals.toFixed(2)}%`);
                                 buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
                                 sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
                                 
                                 // 로컬 스토리지에서 게임 상태 삭제
                                 localStorage.removeItem('aiTradingGameState');
                                 
                                 // 초기화 완료 메시지
                                 setTimeout(() => {
                                     const dialogMessage = '✅ 초기화 완료! 모든 데이터가 리셋되었습니다.';
                                 trainerDialog.setText(dialogMessage);
                                 if (window.logManager) {
                                     window.logManager.addLog(dialogMessage);
                                 }
                                     console.log('✅ 게임 초기화 완료');
                                 }, 1000);
                             };
                             
                             // HTML 버튼에 이벤트 리스너 추가
                             setTimeout(() => {
                                 const resetButton = document.getElementById('game-reset-button');
                                 if (resetButton) {
                                     resetButton.addEventListener('click', resetGame);
                                     console.log('✅ HTML 초기화 버튼 이벤트 리스너 추가 완료');
                                 }
                                 
                                 // N/B 코인 +1 버튼 이벤트 리스너
                                 const nbCoinPlusButton = document.getElementById('nb-coin-plus-button');
                                 if (nbCoinPlusButton) {
                                     nbCoinPlusButton.addEventListener('click', () => {
                                         nbCoins++;
                                         nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                         const dialogMessage = `🎯 [수동 조작] N/B 코인 +1 추가! 현재: ${nbCoins}개`;
                                         trainerDialog.setText(dialogMessage);
                                         if (window.logManager) {
                                             window.logManager.addLog(dialogMessage);
                                             
                                             // 수동 조작 후 상태 정보 로그
                                             const pnlElement = document.getElementById('selected-coin-pnl');
                                             const currentPnl = pnlElement ? pnlElement.textContent : '수익율: 0%';
                                             const currentPriceElement = document.getElementById('trading-current-price');
                                             const currentPriceText = currentPriceElement ? currentPriceElement.textContent : '₩0';
                                             const majorityElement = document.getElementById('majority-zone');
                                             const currentMajority = majorityElement ? majorityElement.textContent.trim() : 'UNKNOWN';
                                             const statusLog = `📊 수동 조작 후 상태: N/B 코인 ${nbCoins}개, 드랍 아이템 ${nbCoinItems.length}개, N/B 미네랄 ${nbMinerals.toFixed(2)}%, 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}%, 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}%, ${currentPnl}, BTC 현재가 ${currentPriceText}, 현재 구역 ${currentMajority}`;
                                             window.logManager.addLog(statusLog);
                                         }
                                         console.log(`➕ N/B 코인 +1 추가됨. 현재: ${nbCoins}개`);
                                         saveGameState();
                                     });
                                     console.log('✅ N/B 코인 +1 버튼 이벤트 리스너 추가 완료');
                                 }
                                 
                                 // N/B 코인 -1 버튼 이벤트 리스너
                                 const nbCoinMinusButton = document.getElementById('nb-coin-minus-button');
                                 if (nbCoinMinusButton) {
                                     nbCoinMinusButton.addEventListener('click', () => {
                                         if (nbCoins > 0) {
                                             nbCoins--;
                                             nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                             const dialogMessage = `🎯 [수동 조작] N/B 코인 -1 감소! 현재: ${nbCoins}개`;
                                             trainerDialog.setText(dialogMessage);
                                             if (window.logManager) {
                                                 window.logManager.addLog(dialogMessage);
                                             }
                                             console.log(`➖ N/B 코인 -1 감소됨. 현재: ${nbCoins}개`);
                                             saveGameState();
                                         } else {
                                             const dialogMessage = `🎯 [수동 조작] N/B 코인이 0개라서 감소할 수 없습니다!`;
                                             trainerDialog.setText(dialogMessage);
                                             if (window.logManager) {
                                                 window.logManager.addLog(dialogMessage);
                                                 
                                                 // 수동 조작 후 상태 정보 로그
                                                 const pnlElement = document.getElementById('selected-coin-pnl');
                                                 const currentPnl = pnlElement ? pnlElement.textContent : '수익율: 0%';
                                                 const currentPriceElement = document.getElementById('trading-current-price');
                                                 const currentPriceText = currentPriceElement ? currentPriceElement.textContent : '₩0';
                                                 const majorityElement = document.getElementById('majority-zone');
                                                 const currentMajority = majorityElement ? majorityElement.textContent.trim() : 'UNKNOWN';
                                                 const statusLog = `📊 수동 조작 후 상태: N/B 코인 ${nbCoins}개, 드랍 아이템 ${nbCoinItems.length}개, N/B 미네랄 ${nbMinerals.toFixed(2)}%, 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}%, 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}%, ${currentPnl}, BTC 현재가 ${currentPriceText}, 현재 구역 ${currentMajority}`;
                                                 window.logManager.addLog(statusLog);
                                             }
                                             console.log(`❌ N/B 코인이 0개라서 감소할 수 없음`);
                                         }
                                     });
                                     console.log('✅ N/B 코인 -1 버튼 이벤트 리스너 추가 완료');
                                 }
                             }, 1000);
                             
                            // 수익률 계산 변수
                            let buyPrice = 0;
                            let buyProfitRate = 0;  // 매수 전 예상 수익률
                            let sellProfitRate = 0;  // 매도 전 예상 수익률
                              
                              // 게임 상태 관리 함수들 (분리된 모듈 사용)
                            const saveGameState = () => {
                                const gameData = {
                                    nbCoins: nbCoins,
                                    nbMinerals: nbMinerals,
                                    buyPrice: buyPrice,
                                    buyProfitRate: buyProfitRate,
                                    sellProfitRate: sellProfitRate,
                                    lastBuyAction: lastBuyAction,
                                    lastSellAction: lastSellAction,
                                    aiModels: window.gameStateManager && typeof window.gameStateManager.convertAiModelsForStorage === 'function'
                                        ? window.gameStateManager.convertAiModelsForStorage(aiModels)
                                        : [],
                                    nbCoinItems: window.gameStateManager && typeof window.gameStateManager.convertNBCoinItemsForStorage === 'function'
                                        ? window.gameStateManager.convertNBCoinItemsForStorage(nbCoinItems)
                                        : []
                                };
                                
                                if (window.gameStateManager && typeof window.gameStateManager.saveGameState === 'function') {
                                    window.gameStateManager.saveGameState(gameData);
                                }
                            };
                            
                            const loadGameState = () => {
                                let gameState = null;
                                if (window.gameStateManager && typeof window.gameStateManager.loadGameState === 'function') {
                                    gameState = window.gameStateManager.loadGameState();
                                }
                                if (gameState) {
                                    // 기본 변수들 복원
                                    nbCoins = gameState.nbCoins || 0;
                                    nbMinerals = gameState.nbMinerals || 0.0;
                                    buyPrice = gameState.buyPrice || 0;
                                    buyProfitRate = gameState.buyProfitRate || 0;
                                    sellProfitRate = gameState.sellProfitRate || 0;
                                    lastBuyAction = gameState.lastBuyAction || false;
                                    lastSellAction = gameState.lastSellAction || false;
                                    
                                    // AI 모델들 위치 복원
                                    if (window.gameStateManager && typeof window.gameStateManager.restoreAiModelsFromStorage === 'function') {
                                        window.gameStateManager.restoreAiModelsFromStorage(gameState.aiModels, aiModels);
                                    }
                                    
                                    // N/B 코인 아이템들 복원
                                    if (window.gameStateManager && typeof window.gameStateManager.restoreNBCoinItemsFromStorage === 'function') {
                                        window.gameStateManager.restoreNBCoinItemsFromStorage(gameState.nbCoinItems, this, nbCoinItems);
                                    }
                                    
                                    // UI 업데이트
                                    nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                    nbMineralDisplay.setText(`N/B 미네랄: ${nbMinerals.toFixed(2)}%`);
                                    buyProfitRateDisplay.setText(`매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`);
                                    sellProfitRateDisplay.setText(`매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
                                    
                                    const dialogMessage = `🎯 트레이너: 이전 상태 복원 완료! N/B 코인: ${nbCoins}개, AI 시스템 시작 중...`;
                                    trainerDialog.setText(dialogMessage);
                                    if (window.logManager) {
                                        window.logManager.addLog(dialogMessage);
                                    }
                                }
                            };
                            
                            // AI 모델들의 분업 시스템
                            const aiSystemAlgorithm = () => {
                                // 실제 트레이딩 데이터 가져오기
                                const majorityElement = document.getElementById('majority-zone');
                                const orangeSumElement = document.getElementById('orange-sum');
                                const blueSumElement = document.getElementById('blue-sum');
                                const currentPriceElement = document.getElementById('trading-current-price');
                                
                                // 디버깅: 데이터 요소 확인 (첫 번째 실행 시에만)
                                if (!window.aiSystemStarted) {
                                    console.log('🔍 AI System Algorithm 첫 실행...');
                                    console.log('   - majorityElement:', majorityElement);
                                    console.log('   - orangeSumElement:', orangeSumElement);
                                    console.log('   - blueSumElement:', blueSumElement);
                                    console.log('   - currentPriceElement:', currentPriceElement);
                                    window.aiSystemStarted = true;
                                }
                                
                                if (!majorityElement || !orangeSumElement || !blueSumElement) {
                                    learningStatus.setText('데이터 로딩 중...');
                                    const dialogMessage = `🎯 트레이너: 데이터 로딩 중... 필수 요소를 찾을 수 없음`;
                                    trainerDialog.setText(dialogMessage);
                                    if (window.logManager) {
                                        window.logManager.addLog(dialogMessage);
                                    }
                                    console.log('❌ 필수 데이터 요소를 찾을 수 없음');
                                    return;
                                }
                                
                                const currentMajority = majorityElement.textContent.trim();
                                
                                // AI 시스템이 정상적으로 실행되고 있음을 표시
                                if (trainerDialog) {
                                    const currentText = trainerDialog.text;
                                    if (currentText.includes('AI 시스템 시작 중...') || currentText.includes('데이터 분석 시작...')) {
                                        const dialogMessage = `🎯 트레이너: AI 시스템 정상 작동 중... 현재 구역: ${currentMajority}, N/B 코인: ${nbCoins}개`;
                                        trainerDialog.setText(dialogMessage);
                                        if (window.logManager) {
                                            window.logManager.addLog(dialogMessage);
                                    }
                                }
                                }
                                
                                const orangeSum = parseInt(orangeSumElement.textContent) || 0;
                                const blueSum = parseInt(blueSumElement.textContent) || 0;
                                const currentPriceText = currentPriceElement ? currentPriceElement.textContent : '₩0';
                                
                                // N/B 길드 내부 구역 표시 업데이트
                                if (guildZoneIndicator && guildZoneText) {
                                    if (currentMajority === 'BLUE') {
                                        guildZoneIndicator.setFillStyle(0x00d1ff); // 파란색
                                        guildZoneText.setText('BLUE');
                                        guildZoneText.setFill('#ffffff');
                                        console.log('🔵 N/B 길드 구역 표시: BLUE');
                                    } else if (currentMajority === 'ORANGE') {
                                        guildZoneIndicator.setFillStyle(0xffb703); // 주황색
                                        guildZoneText.setText('ORANGE');
                                        guildZoneText.setFill('#ffffff');
                                        console.log('🟠 N/B 길드 구역 표시: ORANGE');
                                    } else {
                                        guildZoneIndicator.setFillStyle(0xffffff); // 흰색 (중립)
                                        guildZoneText.setText('NEUTRAL');
                                        guildZoneText.setFill('#000000');
                                        console.log('⚪ N/B 길드 구역 표시: NEUTRAL');
                                    }
                                }
                                
                                // 각 AI 모델의 독립적인 행동
                                aiModels.forEach((model, modelIndex) => {
                                    // 모든 상태 정보를 주기적으로 로그에 저장 (5초마다, 중복 방지)
                                    const currentSecond = Math.floor(Date.now() / 1000);
                                    if (window.logManager && currentSecond % 5 === 0 && !model.statusLoggedThisSecond) {
                                        const pnlElement = document.getElementById('selected-coin-pnl');
                                        const currentPnl = pnlElement ? pnlElement.textContent : '수익율: 0%';
                                        
                                        const statusLog = `📊 상태 정보: N/B 코인 ${nbCoins}개, 드랍 아이템 ${nbCoinItems.length}개, N/B 미네랄 ${nbMinerals.toFixed(2)}%, 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}%, 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}%, ${currentPnl}, BTC 현재가 ${currentPriceText}, 현재 구역 ${currentMajority}`;
                                        window.logManager.addLog(statusLog);
                                        model.statusLoggedThisSecond = true;
                                    } else if (currentSecond % 5 !== 0) {
                                        model.statusLoggedThisSecond = false;
                                    }
                                    
                                    // 트레이너 상태 디버깅 로그 (5초마다)
                                    if (currentSecond % 5 === 0 && !model.debugLoggedThisSecond) {
                                        // 현재 위치 기반 활동 상태 판단
                                        let currentActivity = '이동 중';
                                        let currentLocation = '알 수 없음';
                                        
                                        // BTC 시장 근처 (우하단)
                                        if (Math.abs(model.circle.x - (config.width - 100)) < 60 && Math.abs(model.circle.y - (config.height - 100)) < 60) {
                                            currentLocation = 'BTC 시장';
                                            if (model.targetAction === '매수') {
                                                currentActivity = '매수 시도 중';
                                            } else if (model.targetAction === 'BTC 시장 탐색') {
                                                currentActivity = '매수 전 예상 수익률 계산 중';
                                            } else {
                                                currentActivity = 'BTC 시장에서 대기 중';
                                            }
                                        }
                                        // N/B 길드 근처 (좌상단)
                                        else if (Math.abs(model.circle.x - 100) < 60 && Math.abs(model.circle.y - 100) < 60) {
                                            currentLocation = 'N/B 길드';
                                            if (model.targetAction === '매도') {
                                                currentActivity = '매도 시도 중';
                                            } else if (model.targetAction === 'N/B 길드 방문') {
                                                currentActivity = '매도 전 예상 수익률 계산 중';
                                            } else {
                                                currentActivity = 'N/B 길드에서 대기 중';
                                            }
                                        }
                                        // 신호 대기 센터 근처 (중앙)
                                        else if (Math.abs(model.circle.x - config.width / 2) < 60 && Math.abs(model.circle.y - config.height / 2) < 60) {
                                            currentLocation = '신호 대기 센터';
                                            if (model.targetAction === '신호 대기') {
                                                currentActivity = '신호 대기 중';
                                            } else {
                                                currentActivity = '신호 대기 센터에서 대기 중';
                                            }
                                        }
                                        // 이동 중인 경우
                                        else {
                                            if (model.targetAction === 'BTC 시장 방문') {
                                                currentLocation = 'BTC 시장으로 이동 중';
                                                currentActivity = 'BTC 시장으로 이동 중';
                                            } else if (model.targetAction === 'N/B 길드 방문') {
                                                currentLocation = 'N/B 길드로 이동 중';
                                                currentActivity = 'N/B 길드로 이동 중';
                                            } else if (model.targetAction === '신호 대기') {
                                                currentLocation = '신호 대기 센터로 이동 중';
                                                currentActivity = '신호 대기 센터로 이동 중';
                                            } else {
                                                currentLocation = '이동 중';
                                                currentActivity = '목표로 이동 중';
                                            }
                                        }
                                        
                                        const debugLog = `🔍 트레이너 상태: ${currentLocation}에서 ${currentActivity} | targetAction=${model.targetAction} | 위치=(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)}) | 목표=(${Math.round(model.targetX)}, ${Math.round(model.targetY)}) | buyProfitRate=${buyProfitRate.toFixed(2)}% | sellProfitRate=${sellProfitRate.toFixed(2)}%`;
                                        if (window.logManager) {
                                            window.logManager.addLog(debugLog);
                                        }
                                        model.debugLoggedThisSecond = true;
                                    } else if (currentSecond % 5 !== 0) {
                                        model.debugLoggedThisSecond = false;
                                    }
                                    const modelX = model.circle.x;
                                    const modelY = model.circle.y;
                                    
                                                                            if (model.isExplorer) {
                                            // 탐색 모델: 랜덤 탐험 및 좌표 발견
                                            const distanceToTarget = Math.sqrt((model.targetX - modelX) ** 2 + (model.targetY - modelY) ** 2);
                                            
                                                                                         // 특별한 위치 감지 (목표에 도달하기 전에 먼저 체크)
                                             let specialLocation = '';
                                             let locationColor = 0xffffff;
                                             
                                             // 매수 영역 감지 (더 넓은 범위)
                                             if (Math.abs(modelX - startX) < 50 && Math.abs(modelY - topY) < 50) {
                                                 specialLocation = '매수 영역';
                                                 locationColor = 0x00ff00;
                                             }
                                             // 매도 영역 감지
                                             else if (Math.abs(modelX - (startX + spacing)) < 50 && Math.abs(modelY - topY) < 50) {
                                                 specialLocation = '매도 영역';
                                                 locationColor = 0xff0000;
                                             }
                                             // 대기 영역 감지
                                             else if (Math.abs(modelX - (startX + spacing * 2)) < 50 && Math.abs(modelY - topY) < 50) {
                                                 specialLocation = '대기 영역';
                                                 locationColor = 0xffff00;
                                             }
                                             // N/B 길드 감지
                                             else if (Math.abs(modelX - 100) < 60 && Math.abs(modelY - 100) < 60) {
                                                 specialLocation = 'N/B 길드';
                                                 locationColor = 0x00ff00;
                                             }
                                             // BTC 시장 감지
                                             else if (Math.abs(modelX - (config.width - 100)) < 60 && Math.abs(modelY - (config.height - 100)) < 60) {
                                                 specialLocation = 'BTC 시장';
                                                 locationColor = 0x0088ff;
                                             }
                                             
                                             // 특별한 위치 발견 시 출력
                                             if (specialLocation) {
                                                 console.log(`🎯 Explorer-${modelIndex + 1} 특별 위치 발견: ${specialLocation} (${Math.round(modelX)}, ${Math.round(modelY)})`);
                                                 
                                                 // 화면에 메시지 출력
                                                 const messageElement = document.getElementById('message-text');
                                                 if (messageElement) {
                                                     messageElement.textContent = `Explorer-${modelIndex + 1} 발견: ${specialLocation}`;
                                                     messageElement.style.color = `#${locationColor.toString(16).padStart(6, '0')}`;
                                                     
                                                     // 3초 후 원래 메시지로 복원
                                                     setTimeout(() => {
                                                         messageElement.textContent = 'AI 시스템 작동 중...';
                                                         messageElement.style.color = '#00ff00';
                                                     }, 3000);
                                                 }
                                                 
                                                 // 모델 색상 일시적 변경
                                                 const originalColor = model.circle.fillColor;
                                                 model.circle.setFillStyle(locationColor);
                                                 setTimeout(() => {
                                                     model.circle.setFillStyle(originalColor);
                                                 }, 2000);
                                             }
                                             
                                                                                          if (distanceToTarget < 25) {
                                                 // 새로운 좌표 발견
                                                 const currentCoord = { x: Math.round(modelX), y: Math.round(modelY) };
                                                 
                                                 // 중복 체크
                                                 const isDuplicate = model.discoveredCoords.some(coord => 
                                                     Math.abs(coord.x - currentCoord.x) < 15 && 
                                                     Math.abs(coord.y - currentCoord.y) < 15
                                                 );
                                                 
                                                 if (!isDuplicate) {
                                                     model.discoveredCoords.push(currentCoord);
                                                     
                                                     if (model.discoveredCoords.length > 8) {
                                                         model.discoveredCoords.shift();
                                                     }
                                                     
                                                     console.log(`🔍 Explorer-${modelIndex + 1} 좌표 발견: (${currentCoord.x}, ${currentCoord.y})`);
                                                 }
                                                 
                                                 // N/B 코인 아이템이 있는지 확인하고 우선순위 설정
                                                 const nearestItem = nbCoinItems.find(item => !item.collected);
                                                 if (nearestItem) {
                                                     // 가장 가까운 N/B 코인 아이템을 목표로 설정
                                                     model.targetX = nearestItem.polygon.x;
                                                     model.targetY = nearestItem.polygon.y;
                                                     model.role.setText(`N/B 코인 수집 중 (${model.discoveredCoords.length}/8)`);
                                                     console.log(`🎯 Explorer-${modelIndex + 1} N/B 코인 아이템으로 이동 중...`);
                                                 } else {
                                                     // N/B 코인 아이템이 없으면 랜덤 탐색
                                                     model.targetX = Math.random() * (config.width - 80) + 40;
                                                     model.targetY = Math.random() * (config.height - 80) + 40;
                                                     model.role.setText(`탐색 (${model.discoveredCoords.length}/8)`);
                                                 }
                                             }
                                        
                                        // 탐색 모델 이동
                                        const dx = model.targetX - modelX;
                                        const dy = model.targetY - modelY;
                                        
                                        if (Math.abs(dx) > 1) {
                                            model.circle.x += dx * 0.03;
                                            model.name.x = model.circle.x;
                                            model.role.x = model.circle.x;
                                        }
                                        
                                                                                 if (Math.abs(dy) > 1) {
                                             model.circle.y += dy * 0.03;
                                             model.name.y = model.circle.y - 6;
                                             model.role.y = model.circle.y + 6;
                                         }
                                         
                                                                                   // N/B 코인 아이템 수집 체크
                                          nbCoinItems.forEach((item, itemIndex) => {
                                              if (!item.collected) {
                                                  const distanceToItem = Math.sqrt((modelX - item.polygon.x) ** 2 + (modelY - item.polygon.y) ** 2);
                                                  if (distanceToItem < 25) {
                                                      // N/B 코인 수집 시 +1 증가
                                                      nbCoins++;
                                                      nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length - 1}개)`);
                                                      
                                                                                                                                                              // 상태 즉시 저장 (분리된 모듈 사용)
                                                 const gameData = {
                                                     nbCoins: nbCoins,
                                                     nbMinerals: nbMinerals,
                                                     buyPrice: buyPrice,
                                                     buyProfitRate: buyProfitRate,
                                                     sellProfitRate: sellProfitRate,
                                                     lastBuyAction: lastBuyAction,
                                                     lastSellAction: lastSellAction,
                                                     aiModels: window.gameStateManager && typeof window.gameStateManager.convertAiModelsForStorage === 'function'
                                                         ? window.gameStateManager.convertAiModelsForStorage(aiModels)
                                                         : [],
                                                     nbCoinItems: window.gameStateManager && typeof window.gameStateManager.convertNBCoinItemsForStorage === 'function'
                                                         ? window.gameStateManager.convertNBCoinItemsForStorage(nbCoinItems)
                                                         : []
                                                 };
                                                 if (window.gameStateManager && typeof window.gameStateManager.saveGameState === 'function') {
                                                     window.gameStateManager.saveGameState(gameData);
                                                 }
                                                       
                                                       // 연결선들 제거
                                                      item.connectionLines.forEach(line => {
                                                          if (line && line.destroy) {
                                                              line.destroy();
                                                          }
                                                      });
                                                      
                                                      item.collected = true;
                                                      item.polygon.destroy();
                                                      nbCoinItems.splice(itemIndex, 1);
                                                      
                                                      // 탐색자 역할 텍스트 업데이트
                                                      model.role.setText(`수집 완료!`);
                                                      
                                                      // N/B 코인 수집 시 매도 전 수익률 초기화
                                                      sellProfitRate = 0;
                                                      sellProfitRateDisplay.setFill('#ff0088');
                                                      sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
                                                      
                                                      console.log(`🎯 Explorer-${modelIndex + 1} N/B 코인 수집 완료! N/B 코인: ${nbCoins}개, 수익률 초기화됨`);
                                                      
                                                      // N/B 길드로 이동하여 손실 계산
                                                      model.targetX = 100;
                                                      model.targetY = 100;
                                                      
                                                      console.log(`🎯 Explorer-${modelIndex + 1} N/B 길드로 이동 중...`);
                                                      
                                                      // N/B 길드 도달 시 매도 전 예상 수익률 계산 (CurrentPriceManager 모듈 사용)
                                                      setTimeout(() => {
                                                          if (buyPrice > 0 && window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                                                              sellProfitRate = window.currentPriceManager.calculateSellProfitRate(buyPrice);
                                                              
                                                              // 매도 전 예상 수익률 표시 업데이트
                                                              const profitColor = sellProfitRate >= 0 ? '#00ff88' : '#ff0088';
                                                              sellProfitRateDisplay.setFill(profitColor);
                                                              sellProfitRateDisplay.setText(`매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
                                                              
                                                              // 상태 즉시 저장
                                                              saveGameState();
                                                              
                                                              // 탐색자 역할 텍스트 업데이트
                                                              model.role.setText(`수익률: ${sellProfitRate.toFixed(2)}%`);
                                                              
                                                              const currentPriceLog = window.currentPriceManager.generateCurrentPriceLog();
                                                              console.log(`📊 Explorer-${modelIndex + 1} 매도 전 예상 수익률 계산 완료: ${sellProfitRate.toFixed(2)}% (매수가: ₩${buyPrice.toLocaleString()}, 현재가: ₩${currentPriceLog.currentPrice.toLocaleString()})`);
                                                              
                                                              // 트레이너 대화창 업데이트
                                                              const dialogMessage = `N/B 길드: 매도 전 예상 수익률 계산 완료! ${sellProfitRate.toFixed(2)}% (${currentPriceLog.currentPriceText})`;
                                                              trainerDialog.setText(dialogMessage);
                                                              if (window.logManager) {
                                                                  window.logManager.addLog(dialogMessage);
                                                              }
                                                          }
                                                      }, 1000);
                                                      
                                                      // 5초 후 다시 탐색 모드로 (N/B 코인 아이템 우선순위 확인)
                                                      setTimeout(() => {
                                                          const remainingItem = nbCoinItems.find(item => !item.collected);
                                                          if (remainingItem) {
                                                              // 남은 N/B 코인 아이템이 있으면 그것을 목표로 설정
                                                              model.targetX = remainingItem.polygon.x;
                                                              model.targetY = remainingItem.polygon.y;
                                                              model.role.setText(`N/B 코인 수집 중 (${model.discoveredCoords.length}/8)`);
                                                              console.log(`🎯 Explorer-${modelIndex + 1} 남은 N/B 코인 아이템으로 이동 중...`);
                                                          } else {
                                                              // N/B 코인 아이템이 없으면 랜덤 탐색
                                                              model.role.setText(`탐색 (${model.discoveredCoords.length}/8)`);
                                                              model.targetX = Math.random() * (config.width - 80) + 40;
                                                              model.targetY = Math.random() * (config.height - 80) + 40;
                                                          }
                                                      }, 5000);
                                                  }
                                              }
                                          });
                                        
                                                                             } else if (model.isTrainer) {
                                             // 트레이너 모델: 의사결정 및 학습
                                             // targetAction이 undefined이면 기본값 설정
                                             if (typeof model.targetAction === 'undefined' || model.targetAction === '') {
                                                 model.targetAction = '신호 대기';
                                                 targetAction = '신호 대기';
                                                 model.targetX = config.width / 2;
                                                 model.targetY = config.height / 2;
                                                 model.circle.setFillStyle(0x88ccff); // 하늘색 (신호 대기)
                                                 
                                                 if (window.logManager) {
                                                     window.logManager.addLog(`🔵 트레이너: targetAction 초기화 → 신호 대기 센터로 이동`);
                                                 }
                                             }
                                             let targetAction = model.targetAction;
                                             
                                             // 트레이너 활동 로그 (TrainerActivityLogger 모듈 사용) - 강제 호출
                                             if (window.trainerActivityLogger && typeof window.trainerActivityLogger.logTrainerActivity === 'function') {
                                                 window.trainerActivityLogger.logTrainerActivity(
                                                     model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate
                                                 );
                                                 console.log('📝 트레이너 활동 로그 호출됨');
                                             } else {
                                                 console.warn('⚠️ TrainerActivityLogger 모듈이 로드되지 않았습니다.');
                                             }
                                             
                                                                                           // BTC 시장에서 매수 전 예상 수익률 계산 (BTCMarketCalculator 모듈 사용)
                                              if (window.btcMarketCalculator && typeof window.btcMarketCalculator.calculateBuyProfitRateAtMarket === 'function') {
                                                  const calculationResult = window.btcMarketCalculator.calculateBuyProfitRateAtMarket(
                                                      modelX, modelY, config, currentMajority, buyProfitRateDisplay, trainerDialog
                                                  );
                                                  
                                                  if (calculationResult) {
                                                      // 계산이 완료되었으면 buyProfitRate 업데이트
                                                      if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                                                          buyProfitRate = window.currentPriceManager.calculateBuyProfitRate();
                                                      }
                                                      
                                                      // 정보 수집 목적으로 방문한 경우 신호 대기 센터로 복귀 준비
                                                      if (model.infoCollectionMode) {
                                                          model.infoCollectionMode = false;
                                                          setTimeout(() => {
                                                              targetAction = '정보 수집 완료';
                                                          }, 2000); // 2초 후 복귀
                                                      }
                                                  }
                                              } else {
                                                  console.warn('⚠️ BTCMarketCalculator 모듈이 로드되지 않았습니다.');
                                              }
                                              
                                                                                                                                      // 트레이너 의사결정 처리 (분리된 모듈 사용)
                                             if (window.trainerDecisionHandler) {
                                                 targetAction = window.trainerDecisionHandler.handleTrainerDecision(
                                                     model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing
                                                 );
                                             } else {
                                                 // 폴백: 기존 의사결정 로직
                                                 const currentZone = window.decisionSystem && typeof window.decisionSystem.getCurrentZone === 'function' 
                                                     ? window.decisionSystem.getCurrentZone(modelX, modelY, startX, topY, spacing, config)
                                                     : '기타영역';
                                                 const zoneDecision = window.decisionSystem && typeof window.decisionSystem.getZoneDecision === 'function'
                                                     ? window.decisionSystem.getZoneDecision(currentZone, currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
                                                     : null;
                                                 
                                                 if (!zoneDecision && !model.countdownStarted && !model.btcExplorationMode) {
                                                     if (targetAction !== '신호 대기') {
                                                         const previousAction = targetAction;
                                                         targetAction = '신호 대기';
                                                         model.targetAction = targetAction;
                                                         model.targetX = config.width / 2;
                                                         model.targetY = config.height / 2;
                                                         model.circle.setFillStyle(0x88ccff);
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`🔵 트레이너: 현재 구역(${currentZone})에서 의사 결정 없음 → 신호 대기 센터로 이동 (이전 액션: ${previousAction})`);
                                                         }
                                                     }
                                                 } else if (zoneDecision && !model.countdownStarted) {
                                                     targetAction = zoneDecision.action;
                                                     model.targetAction = targetAction;
                                                     model.targetX = zoneDecision.targetX;
                                                     model.targetY = zoneDecision.targetY;
                                                     
                                                     // 디버깅: 목표 위치 설정 확인
                                                     if (window.logManager) {
                                                         window.logManager.addLog(`🎯 의사결정: ${targetAction} → 목표 위치 (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
                                                     }
                                                     
                                                     // 색상 변경 (분리된 모듈 사용)
                                                     if (window.trainerVisualEffects) {
                                                         window.trainerVisualEffects.changeTrainerColor(model, targetAction);
                                                     } else {
                                                         // 폴백: 기존 색상 변경 로직
                                                         if (targetAction === '매도') {
                                                             model.circle.setFillStyle(0xff8800);
                                                         } else if (targetAction === '매수') {
                                                             model.circle.setFillStyle(0x0088ff);
                                                         } else if (targetAction === 'BTC 시장 방문') {
                                                             model.circle.setFillStyle(0x0088ff);
                                                         } else if (targetAction === 'N/B 길드 방문') {
                                                             model.circle.setFillStyle(0xff8800);
                                                         } else if (targetAction === '대기') {
                                                             model.circle.setFillStyle(0xffff00);
                                                         } else if (targetAction === '신호 대기') {
                                                             model.circle.setFillStyle(0x88ccff);
                                                         }
                                                     }
                                                 } else if (model.countdownStarted) {
                                                     targetAction = '신호 대기';
                                                     model.targetAction = targetAction;
                                                     model.targetX = config.width / 2;
                                                     model.targetY = config.height / 2;
                                                     model.circle.setFillStyle(0x88ccff);
                                                 }
                                             }
                                             
                                             // 강제로 목표 위치가 설정되지 않은 경우 기본값 설정
                                             if (typeof model.targetX === 'undefined' || typeof model.targetY === 'undefined') {
                                                 model.targetX = config.width / 2;
                                                 model.targetY = config.height / 2;
                                                 model.targetAction = '신호 대기';
                                                 if (window.logManager) {
                                                     window.logManager.addLog(`🔧 트레이너 목표 위치 강제 설정: (${model.targetX}, ${model.targetY})`);
                                                 }
                                             }
                                             
                                             // 신호 대기 상태에서 주기적으로 조건 확인 및 액션 실행
                                             if (targetAction === '신호 대기') {
                                                 // 신호 대기 센터에 도달했을 때만 조건 확인
                                                 const distanceToCenter = Math.sqrt((modelX - (config.width / 2)) ** 2 + (modelY - (config.height / 2)) ** 2);
                                                 
                                                // 디버깅: 거리 확인 (10초마다, 0px일 때는 출력하지 않음)
                                                if (model.isTrainer && Math.floor(Date.now() / 1000) % 10 === 0 && distanceToCenter > 0) {
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔍 신호 대기 센터 거리: ${Math.round(distanceToCenter)}px, 조건: ${distanceToCenter < 30}`);
                                                    }
                                                }
                                                 
                                                 if (distanceToCenter < 30) {
                                                     // 신호 대기 센터에 도달했을 때 주기적으로 조건 확인
                                                     if (!model.waitCheckTimer) {
                                                         model.waitCheckTimer = 0;
                                                         model.waitStartTime = Date.now(); // 시작 시간 기록
                                                     }
                                                     model.waitCheckTimer++;
                                                     
                                                     // 카운트다운이 시작되었음을 표시
                                                     model.countdownStarted = true;
                                                     
                                                     // 경과 시간 계산
                                                     const elapsedSeconds = (Date.now() - model.waitStartTime) / 1000;
                                                     
                                                // 디버깅: 센터 도달 확인
                                                if (model.isTrainer && model.waitCheckTimer === 1) {
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🎯 트레이너가 신호 대기 센터에 도달! 카운트다운 시작`);
                                                    }
                                                }
                                                     
                                                // 디버깅: 타이머 값 확인 (1초마다)
                                                if (model.isTrainer && Math.floor(elapsedSeconds) % 1 === 0 && elapsedSeconds > 0 && elapsedSeconds < 5) {
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔍 타이머 디버깅: elapsedSeconds = ${elapsedSeconds.toFixed(1)}, countdownStarted = ${model.countdownStarted}, remainingSeconds = ${Math.ceil(5 - elapsedSeconds)}`);
                                                    }
                                                }
                                                     
                                                     // 카운트다운이 시작되면 조건 확인 없이 무조건 5초 후 이동
                                                     
                                                     // 5초 이상 머무르면 BTC 시장으로 이동 (정확한 시간 기준)
                                                     if (elapsedSeconds >= 5) {
                                                    targetAction = 'BTC 시장 탐색';
                                                    model.targetAction = targetAction; // model에 저장
                                                    model.targetX = config.width - 100;
                                                    model.targetY = config.height - 100;
                                                    model.circle.setFillStyle(0x0088ff);
                                                    model.btcExplorationMode = true; // BTC 탐색 모드 설정
                                                    model.countdownStarted = false; // 카운트다운 플래그 리셋
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔵 트레이너: 신호 대기 센터에서 5초 이상 대기 → BTC 시장 탐색으로 이동! targetAction: ${targetAction}, targetX: ${model.targetX}, targetY: ${model.targetY}`);
                                                    }
                                                    const dialogMessage = `🔵 [장시간 대기] BTC 시장 탐색으로 이동 중... 매수 전 예상 수익률 계산 예정`;
                                                    trainerDialog.setText(dialogMessage);
                                                    if (window.logManager) {
                                                        window.logManager.addLog(dialogMessage);
                                                    }
                                              } else {
                                                         // 카운트다운 표시 (5초에서 0초까지)
                                                         const remainingSeconds = Math.ceil(5 - elapsedSeconds);
                                                         
                                                         if (remainingSeconds <= 0) {
                                                             const dialogMessage = `🔵 [신호 대기] BTC 시장 탐색으로 이동 중... N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
                                                             trainerDialog.setText(dialogMessage);
                                                             if (window.logManager) {
                                                                 window.logManager.addLog(dialogMessage);
                                                             }
                                                    } else {
                                                        const dialogMessage = `🔵 [신호 대기] 센터에서 대기 중... (${remainingSeconds}초 후 BTC 시장 탐색으로 이동) N/B 코인: ${nbCoins}개, 신호: ${currentMajority}`;
                                                        trainerDialog.setText(dialogMessage);
                                                        if (window.logManager) {
                                                            window.logManager.addLog(dialogMessage);
                                                        }
                                                    }
                                                     }
                                                 } else {
                                                     // 신호 대기 센터에서 멀리 떨어져 있을 때만 타이머 리셋 (50px 이상)
                                                     // 단, 카운트다운이 시작되지 않았을 때만 리셋
                                                     if (distanceToCenter > 50 && !model.countdownStarted) {
                                                         model.waitCheckTimer = 0;
                                                     }
                                                     
                                                // 디버깅: 센터에서 멀리 떨어져 있을 때
                                                if (model.isTrainer && Math.floor(Date.now() / 1000) % 3 === 0) {
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔍 트레이너가 신호 대기 센터에서 멀리 떨어짐: ${Math.round(distanceToCenter)}px, countdownStarted: ${model.countdownStarted}`);
                                                    }
                                                }
                                                 }
                                             }
                                             
                                             // BTC 시장 탐색 모드에서의 행동 로직 (학습 모델 기반)
                                             if (targetAction === 'BTC 시장 탐색') {
                                                 // BTC 시장까지의 거리 계산
                                                 const distanceToBTCMarket = Math.sqrt((modelX - (config.width - 100)) ** 2 + (modelY - (config.height - 100)) ** 2);
                                                 
                                                 // BTC 시장 충돌 검사
                                                 let isCollidingWithBTCMarket = false;
                                                 if (window.btcMarketPolygon && model.circle) {
                                                     const circleBounds = model.circle.getBounds();
                                                     const polygonBounds = window.btcMarketPolygon.getBounds();
                                                     isCollidingWithBTCMarket = Phaser.Geom.Rectangle.Overlaps(circleBounds, polygonBounds);
                                                     
                                                     // 더 정확한 검사를 위해 거리도 확인
                                                     if (isCollidingWithBTCMarket) {
                                                         const centerDistance = Math.sqrt(
                                                             (model.circle.x - window.btcMarketPolygon.x) ** 2 + 
                                                             (model.circle.y - window.btcMarketPolygon.y) ** 2
                                                         );
                                                         if (centerDistance > 50) {
                                                             isCollidingWithBTCMarket = false;
                                                         }
                                                     }
                                                 }
                                                 
                                                 // BTC 시장에 도달했는지 확인
                                                 if (isCollidingWithBTCMarket || distanceToBTCMarket < 60) {
                                                     // BTC 시장에 도달했을 때 학습 모델 기반 처리
                                                     if (model.btcExplorationMode) {
                                                         model.btcExplorationMode = false;
                                                         
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`🎯 트레이너가 BTC 시장에 도달! 학습 모델 기반 처리 시작`);
                                                         }
                                                     }
                                                     
                                                     // 학습 모델을 사용한 BTC 시장 도달 처리
                                                     if (window.btcMarketLearningHandler) {
                                                         window.btcMarketLearningHandler.handleBTCMarketArrival(model, config, trainerDialog, currentMajority, buyProfitRateDisplay);
                                                     } else {
                                                         console.warn('⚠️ BTC 시장 학습 핸들러가 로드되지 않음');
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`⚠️ BTC 시장 학습 핸들러 오류 - 기본 처리 실행`);
                                                         }
                                                         
                                                         // 기본 처리: 즉시 신호 대기 센터로 복귀
                                                         model.targetAction = '신호 대기';
                                                         model.targetX = config.width / 2;
                                                         model.targetY = config.height / 2;
                                                         model.circle.setFillStyle(0x88ccff);
                                                     }
                                                 } else {
                                                     // BTC 시장 탐색으로 이동 중 상태 표시
                                                     const statusText = `🔵 [BTC 탐색] BTC 시장 탐색으로 이동 중... (${Math.round(distanceToBTCMarket)}px 남음)`;
                                                     trainerDialog.setText(statusText);
                                                     if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
                                                         window.logManager.addLog(`🔵 BTC 시장 탐색 진행 중: 거리 ${Math.round(distanceToBTCMarket)}px, 현재 신호: ${currentMajority}`);
                                                     }
                                                 }
                                             }
                                             
                                             // BTC 탐색 관리자 모듈이 없을 때의 기본 처리
                                             if (!window.btcExplorationManager) {
                                                 // BTC 탐색 완료 후 신호 대기 센터로 복귀 (한 번만 실행)
                                                 if (model.btcExplorationMode && !model.btcExplorationCompleted) {
                                                     model.btcExplorationCompleted = true;
                                                     
                                                     if (window.logManager) {
                                                         window.logManager.addLog(`🔵 BTC 탐색 완료: 2초 후 신호 대기 센터로 복귀 (현재 신호: ${currentMajority})`);
                                                     }
                                                     
                                                     setTimeout(() => {
                                                         const previousAction = targetAction;
                                                         targetAction = '신호 대기';
                                                         model.targetAction = targetAction; // model에 저장
                                                         model.targetX = config.width / 2;
                                                         model.targetY = config.height / 2;
                                                         model.circle.setFillStyle(0x88ccff);
                                                         // 타이머 및 상태 초기화
                                                         model.waitCheckTimer = 0;
                                                         model.countdownStarted = false;
                                                         model.arrivalLogged = false; // 도착 로그 플래그 리셋
                                                         model.btcExplorationMode = false;
                                                         model.btcExplorationCompleted = false;
                                                         
                                                         console.log(`🔵 트레이너: BTC 탐색 완료, 신호 대기 센터로 복귀!`);
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`🔵 BTC 탐색 완료: targetAction 변경 (${previousAction} → 신호 대기)`);
                                                         }
                                                         const dialogMessage = `🔵 [탐색 완료] 신호 대기 센터로 복귀 중... 매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
                                                         trainerDialog.setText(dialogMessage);
                                                         if (window.logManager) {
                                                             window.logManager.addLog(dialogMessage);
                                                         }
                                                     }, 2000); // 2초 후 복귀
                                                 }
                                             } else {
                                                 // BTC 시장 탐색으로 이동 중 상태 표시
                                                 const distanceToBTCMarket = Math.sqrt((modelX - (config.width - 100)) ** 2 + (modelY - (config.height - 100)) ** 2);
                                                 const statusText = `🔵 [BTC 탐색] BTC 시장 탐색으로 이동 중... (${Math.round(distanceToBTCMarket)}px 남음)`;
                                                 trainerDialog.setText(statusText);
                                                 if (window.logManager) {
                                                     window.logManager.addLog(statusText);
                                                 }
                                                 
                                                 // 로그에도 상태 업데이트
                                                 if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
                                                     window.logManager.addLog(`🔵 BTC 시장 탐색 진행 중: 거리 ${Math.round(distanceToBTCMarket)}px, 현재 신호: ${currentMajority}`);
                                                 }
                                             }
                                             
                                             
                                        
                                             const distanceToTarget = Math.sqrt((model.targetX - modelX) ** 2 + (model.targetY - modelY) ** 2);
                                        
                                             if (distanceToTarget < 30) {
                                                 // 목표 도달 로그 (중복 방지)
                                                 if (!model.arrivalLogged) {
                                                let arrivalMessage = '';
                                                if (targetAction === 'BTC 시장 방문') {
                                                    arrivalMessage = `🎯 트레이너: BTC 시장에 도착! 매수 전 예상 수익률 계산 시작...`;
                                                } else if (targetAction === 'N/B 길드 방문') {
                                                    arrivalMessage = `🎯 트레이너: N/B 길드에 도착! 매도 전 예상 수익률 계산 시작...`;
                                                } else if (targetAction === '신호 대기') {
                                                    arrivalMessage = `🎯 트레이너: 신호 대기 센터에 도착! 신호 대기 시작...`;
                                                }
                                                
                                                if (arrivalMessage && window.logManager) {
                                                    window.logManager.addLog(arrivalMessage);
                                                }
                                                model.arrivalLogged = true;
                                            }
                                            
                                            model.role.setText(`트레이너 (${targetAction})`);
                                              
                                            // N/B 길드 도달 시 구역 체크 및 학습 모델 기반 매도 전 예상 수익률 계산
                                            if (targetAction === 'N/B 길드 방문') {
                                                  // 구역 불일치 체크: BLUE 구역에서는 매도 준비 불가
                                                  if (currentMajority === 'BLUE') {
                                                      // BLUE 구역에서는 신호 대기 센터로 이동
                                                      targetAction = '신호 대기';
                                                      model.targetAction = targetAction;
                                                      model.targetX = config.width / 2;
                                                      model.targetY = config.height / 2;
                                                      model.circle.setFillStyle(0x88ccff); // 하늘색 (신호 대기)
                                                      
                                                      if (window.logManager) {
                                                          window.logManager.addLog(`🔵 N/B 길드에서 매도 준비 중이지만 BLUE 구역임 → 신호 대기 센터로 이동`);
                                                      }
                                                      return; // 현재 처리 중단
                                                  }
                                                  
                                                  // 매수한 적이 있는 경우에만 매도 전 예상 수익률 계산
                                                  if (buyPrice > 0) {
                                                                                                     // N/B 길드 시각적 효과 (분리된 모듈 사용)
                                                  if (window.trainerVisualEffects) {
                                                      window.trainerVisualEffects.createNBGuildEffects(model);
                                                  } else {
                                                      // 폴백: 기존 시각적 효과 로직
                                                      if (window.logManager) {
                                                          window.logManager.addLog(`📳 N/B 길드 트레이너 원형 진동 효과 시작: 1~3초간 지속`);
                                                      }
                                                      
                                                      // N/B 길드 다각형 깜빡임 효과
                                                      if (window.nbGuildPolygon) {
                                                          const originalColor = 0x00ff00;
                                                          let blinkCount = 0;
                                                          const maxBlinks = 6;
                                                          
                                                          const blinkInterval = setInterval(() => {
                                                              if (window.nbGuildPolygon && blinkCount < maxBlinks) {
                                                                  const isBright = blinkCount % 2 === 0;
                                                                  window.nbGuildPolygon.setFillStyle(isBright ? 0x00ff88 : originalColor);
                                                                  blinkCount++;
                                                              } else {
                                                                  clearInterval(blinkInterval);
                                                                  if (window.nbGuildPolygon) {
                                                                      window.nbGuildPolygon.setFillStyle(originalColor);
                                                                  }
                                                              }
                                                          }, 250);
                                                      }
                                                      
                                                      // 트레이너 원형 깜빡임 및 진동 효과
                                                      if (model.circle) {
                                                          const originalColor = model.circle.fillColor;
                                                          let blinkCount = 0;
                                                          const maxBlinks = 8;
                                                          
                                                          const circleBlinkInterval = setInterval(() => {
                                                              if (model.circle && blinkCount < maxBlinks) {
                                                                  const isBright = blinkCount % 2 === 0;
                                                                  model.circle.setFillStyle(isBright ? 0xff8800 : originalColor);
                                                                  blinkCount++;
                                                              } else {
                                                                  clearInterval(circleBlinkInterval);
                                                                  if (model.circle) {
                                                                      model.circle.setFillStyle(originalColor);
                                                                  }
                                                              }
                                                          }, 250);
                                                          
                                                          // 진동 효과
                                                          const originalX = model.circle.x;
                                                          const originalY = model.circle.y;
                                                          let shakeCount = 0;
                                                          const maxShakes = 10;
                                                          
                                                          const shakeInterval = setInterval(() => {
                                                              if (model.circle && shakeCount < maxShakes) {
                                                                  const angle = Math.random() * 2 * Math.PI;
                                                                  const distance = Math.random() * 3;
                                                                  const shakeX = originalX + Math.cos(angle) * distance;
                                                                  const shakeY = originalY + Math.sin(angle) * distance;
                                                                  
                                                                  model.circle.x = shakeX;
                                                                  model.circle.y = shakeY;
                                                                  
                                                                  if (model.name) {
                                                                      model.name.x = shakeX;
                                                                      model.name.y = shakeY - 6;
                                                                  }
                                                                  if (model.role) {
                                                                      model.role.x = shakeX;
                                                                      model.role.y = shakeY + 6;
                                                                  }
                                                                  
                                                                  shakeCount++;
                                                              } else {
                                                                  clearInterval(shakeInterval);
                                                                  if (model.circle) {
                                                                      model.circle.x = originalX;
                                                                      model.circle.y = originalY;
                                                                      if (model.name) {
                                                                          model.name.x = originalX;
                                                                          model.name.y = originalY - 6;
                                                                      }
                                                                      if (model.role) {
                                                                          model.role.x = originalX;
                                                                          model.role.y = originalY + 6;
                                                                      }
                                                                  }
                                                              }
                                                          }, 1);
                                                      }
                                                  }
                                                  // Info Panel의 모든 데이터 추출
                                                  // Info Panel 데이터 추출 (분리된 모듈 사용)
                                                  const infoData = window.learningSystem && typeof window.learningSystem.getInfoPanelData === 'function'
                                                      ? window.learningSystem.getInfoPanelData()
                                                      : {};
                                                  console.log('📊 Info Panel 데이터:', infoData);
                                                  
                                                  // 학습 모델 기반 고급 매도 전 예상 수익률 계산 (CurrentPriceManager 모듈 사용)
                                                  if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                                                      sellProfitRate = window.currentPriceManager.calculateSellProfitRate(buyPrice);
                                                  } else {
                                                      // 폴백: 기본 계산
                                                      const currentPriceMatch = currentPriceText.match(/₩([\d,]+)/);
                                                      if (currentPriceMatch && buyPrice > 0) {
                                                          const currentPrice = currentPriceMatch ? parseInt(currentPriceMatch[1].replace(/,/g, '')) : 0;
                                                          sellProfitRate = ((currentPrice - buyPrice) / buyPrice) * 100;
                                                      }
                                                  }
                                                      
                                                      // 매도 전 예상 수익률 표시 업데이트 (분리된 모듈 사용)
                                                      const profitColor = sellProfitRate >= 0 ? '#00ff88' : '#ff0088';
                                                      sellProfitRateDisplay.setFill(profitColor);
                                                      
                                                      let displayText = '';
                                                      if (window.sellProfitCalculator && typeof window.sellProfitCalculator.generateDisplayText === 'function') {
                                                          displayText = window.sellProfitCalculator.generateDisplayText(sellProfitRate, infoData);
                                                      } else {
                                                          // 폴백: 기본 표시 텍스트
                                                          displayText = `매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
                                                      }
                                                      sellProfitRateDisplay.setText(displayText);
                                                      
                                                      // 상태 즉시 저장
                                                      saveGameState();
                                                      
                                                      // 매도 전 예상 수익률 계산 로그 (CurrentPriceManager 모듈 사용)
                                                      if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                                                          const currentPriceLog = window.currentPriceManager.generateCurrentPriceLog();
                                                          console.log(`📊 트레이너: N/B 길드에서 매도 전 예상 수익률 계산 완료: ${sellProfitRate.toFixed(2)}% (현재가: ₩${currentPriceLog.currentPrice.toLocaleString()})`);
                                                      } else {
                                                          // 폴백: 기본 로그
                                                          console.log(`📊 트레이너: N/B 길드에서 매도 전 예상 수익률 계산 완료: ${sellProfitRate.toFixed(2)}%`);
                                                      }
                                                     
                                                     // 트레이너 대화창 업데이트 (분리된 모듈 사용)
                                                     let dialogMessage = '';
                                                     if (window.sellProfitCalculator && typeof window.sellProfitCalculator.generateDialogMessage === 'function') {
                                                         dialogMessage = window.sellProfitCalculator.generateDialogMessage(sellProfitRate, infoData, currentPriceText);
                                                     } else {
                                                         // 폴백: 기본 메시지
                                                         dialogMessage = `🎯 [의사결정: 매도 준비] N/B 길드에서 매도 전 예상 수익률 계산 완료! ${sellProfitRate.toFixed(2)}% (${currentPriceText})`;
                                                     }
                                                     trainerDialog.setText(dialogMessage);
                                                     
                                                     // 매도 전 예상 수익률이 계산되면 즉시 매도 영역으로 이동
                                                     model.targetX = startX + spacing;
                                                     model.targetY = topY;
                                                     console.log(`🎯 트레이너: 매도 영역으로 즉시 이동 시작!`);
                                                     
                                                     // 정보 수집 목적으로 방문한 경우 신호 대기 센터로 복귀 준비
                                                     if (model.infoCollectionMode) {
                                                         model.infoCollectionMode = false;
                                                         setTimeout(() => {
                                                             targetAction = '정보 수집 완료';
                                                         }, 2000); // 2초 후 복귀
                                                     }
                                                 }
                                             }
                                             
                                            // 매수 액션 처리 (BLUE 구역에서만)
                                            if (targetAction === '매수' && !lastBuyAction && currentMajority === 'BLUE') {
                                                 lastBuyAction = true;
                                                 lastSellAction = false;
                                                 // 매수 시작 시 플래그 리셋
                                                 model.postBuyDecisionMade = false;
                                                 model.postSellDecisionMade = false;
                                                 // 색상 변경 로그 플래그 리셋
                                                 model.buyColorLogged = false;
                                                 model.sellColorLogged = false;
                                                 model.nbGuildColorLogged = false;
                                                 // nbCoins는 탐색자가 수집할 때 증가하도록 변경
                                                 
                                                 // 매수가격 저장 (CurrentPriceManager 모듈 사용)
                                                 if (window.currentPriceManager && window.currentPriceManager.isValidCurrentPrice()) {
                                                     buyPrice = window.currentPriceManager.parseCurrentPrice();
                                                     console.log(`💰 매수가격 저장: ₩${buyPrice.toLocaleString()}`);
                                                 } else {
                                                     // 폴백: 기존 방식
                                                     const priceMatch = currentPriceText.match(/₩([\d,]+)/);
                                                     if (priceMatch) {
                                                         buyPrice = parseInt(priceMatch[1].replace(/,/g, ''));
                                                         console.log(`💰 매수가격 저장: ₩${buyPrice.toLocaleString()}`);
                                                     }
                                                 }
                                                 
                                                 // 매수 완료 후 매수 전 예상 수익률 리셋
                                                 buyProfitRate = 0;
                                                 buyProfitRateDisplay.setFill('#00ff88');
                                                 buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
                                                 
                                                 createNBCoinItem();
                                                 nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                                // 매수 완료 후 대화창 업데이트 (분리된 모듈 사용)
                                                let dialogMessage = '';
                                                if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
                                                    dialogMessage = window.decisionSystem.generateDecisionLogMessage('매수 완료', currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate);
                                                } else {
                                                    // 폴백: 기본 매수 완료 메시지
                                                    dialogMessage = `💰 [매수 완료] N/B 코인: ${nbCoins}개, 미네랄: ${nbMinerals.toFixed(2)}%, 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
                                                }
                                                trainerDialog.setText(dialogMessage);
                                                if (window.logManager) {
                                                    window.logManager.addLog(dialogMessage);
                                                    // 매수 완료 상세 로그
                                                    const detailedLog = `💰 매수 완료 상세: 매수가=${currentPriceText}, 매수 전 예상 수익률=${buyProfitRate.toFixed(2)}%, 현재 구역=${currentMajority}, 트레이너 위치=(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
                                                    window.logManager.addLog(detailedLog);
                                                }
                                                 console.log(`💰 매수 완료! N/B 코인 드랍 아이템 생성, 매수가: ${currentPriceText}`);
                                                 
                                                 // 매수 완료 후 상태 정보 로그
                                                 if (window.logManager) {
                                                     const pnlElement = document.getElementById('selected-coin-pnl');
                                                     const currentPnl = pnlElement ? pnlElement.textContent : '수익율: 0%';
                                                     const statusLog = `💰 매수 완료 후 상태: N/B 코인 ${nbCoins}개, 드랍 아이템 ${nbCoinItems.length}개, N/B 미네랄 ${nbMinerals.toFixed(2)}%, 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}%, 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}%, ${currentPnl}, BTC 현재가 ${currentPriceText}, 현재 구역 ${currentMajority}`;
                                                     window.logManager.addLog(statusLog);
                                                 }
                                                 
                                                 // 매수 완료 후 다음 의사결정 실행 (한 번만)
                                                 if (!model.postBuyDecisionMade) {
                                                     model.postBuyDecisionMade = true; // 플래그 설정
                                                     const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                                                         ? window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
                                                         : null;
                                                     if (nextDecision) {
                                                         targetAction = nextDecision.action;
                                                         model.targetAction = targetAction; // model에 저장
                                                         model.targetX = nextDecision.targetX;
                                                         model.targetY = nextDecision.targetY;
                                                         
                                                         // 의사결정에 따른 색상 변경 (분리된 모듈 사용)
                                                         let actionColor = 0x88ccff; // 기본 색상
                                                         if (window.decisionSystem && typeof window.decisionSystem.getActionColor === 'function') {
                                                             actionColor = window.decisionSystem.getActionColor(targetAction);
                                                         } else {
                                                             // 폴백: 기본 색상 매핑
                                                             switch (targetAction) {
                                                                 case '매수': actionColor = 0x0088ff; break;
                                                                 case '매도': actionColor = 0xff8800; break;
                                                                 case '신호 대기': actionColor = 0x88ccff; break;
                                                                 case 'BTC 시장 탐색': actionColor = 0x88ff88; break;
                                                                 default: actionColor = 0x88ccff;
                                                             }
                                                         }
                                                         model.circle.setFillStyle(actionColor);
                                                         console.log(`🎨 매수 완료 후: ${targetAction} 의사결정! 색상 변경 (${actionColor.toString(16)})`);
                                                         
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`🔄 매수 완료 후 다음 의사결정: ${targetAction}`);
                                                         }
                                                     }
                                                 }
                                                 
                                                 // 상태 즉시 저장 (분리된 모듈 사용)
                                                 const gameData = {
                                                     nbCoins: nbCoins,
                                                     nbMinerals: nbMinerals,
                                                     buyPrice: buyPrice,
                                                     buyProfitRate: buyProfitRate,
                                                     sellProfitRate: sellProfitRate,
                                                     lastBuyAction: lastBuyAction,
                                                     lastSellAction: lastSellAction,
                                                     aiModels: window.gameStateManager && typeof window.gameStateManager.convertAiModelsForStorage === 'function'
                                                         ? window.gameStateManager.convertAiModelsForStorage(aiModels)
                                                         : [],
                                                     nbCoinItems: window.gameStateManager && typeof window.gameStateManager.convertNBCoinItemsForStorage === 'function'
                                                         ? window.gameStateManager.convertNBCoinItemsForStorage(nbCoinItems)
                                                         : []
                                                 };
                                                 if (window.gameStateManager && typeof window.gameStateManager.saveGameState === 'function') {
                                                     window.gameStateManager.saveGameState(gameData);
                                                 }
                                             }
                                             // 매수 액션 처리 (BLUE 구역이 아닐 때)
                                             else if (targetAction === '매수' && !lastBuyAction && currentMajority !== 'BLUE') {
                                                 // BLUE 구역이 아니면 신호 대기 센터로 이동
                                                 targetAction = '신호 대기';
                                                 model.targetAction = targetAction;
                                                 model.targetX = config.width / 2;
                                                 model.targetY = config.height / 2;
                                                 model.circle.setFillStyle(0x88ccff); // 하늘색 (신호 대기)
                                                 
                                                 if (window.logManager) {
                                                     window.logManager.addLog(`🔵 매수 시도 중이지만 BLUE 구역이 아님 (현재: ${currentMajority}) → 신호 대기 센터로 이동`);
                                                 }
                                             }
                                             // 매도 액션 처리 (ORANGE 구역에서만)
                                             else if (targetAction === '매도' && !lastSellAction && nbCoins > 0 && currentMajority === 'ORANGE') {
                                                 lastSellAction = true;
                                                 lastBuyAction = false;
                                                 // 매도 시작 시 플래그 리셋
                                                 model.postBuyDecisionMade = false;
                                                 model.postSellDecisionMade = false;
                                                 // 색상 변경 로그 플래그 리셋
                                                 model.buyColorLogged = false;
                                                 model.sellColorLogged = false;
                                                 model.nbGuildColorLogged = false;
                                                 nbCoins--;
                                                 
                                                 // 매도 완료 시 현재 수익률을 N/B 미네랄에 누적 (분리된 모듈 사용)
                                                 let currentPnl = 0;
                                                 if (window.learningSystem && typeof window.learningSystem.getCurrentProfitRate === 'function') {
                                                     currentPnl = window.learningSystem.getCurrentProfitRate();
                                                 } else {
                                                     // 폴백: DOM에서 직접 수익률 가져오기
                                                     const pnlElement = document.getElementById('selected-coin-pnl');
                                                     if (pnlElement) {
                                                         const pnlMatch = pnlElement.textContent.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
                                                         if (pnlMatch) {
                                                             currentPnl = parseFloat(pnlMatch[1]);
                                                         }
                                                     }
                                                 }
                                                 nbMinerals += currentPnl;
                                                 console.log(`📊 현재 수익률: ${currentPnl}%, 누적 미네랄: ${nbMinerals.toFixed(2)}%`);
                                                 
                                                 // 매도 시 수익률 리셋
                                                 buyPrice = 0;
                                                 buyProfitRate = 0;
                                                 sellProfitRate = 0;
                                                 buyProfitRateDisplay.setFill('#00ff88');
                                                 buyProfitRateDisplay.setText(`매수 전 예상 수익률: 0.00%`);
                                                 sellProfitRateDisplay.setFill('#ff0088');
                                                 sellProfitRateDisplay.setText(`매도 전 예상 수익률: 0.00%`);
                                                 
                                                 nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                                 nbMineralDisplay.setText(`N/B 미네랄: ${nbMinerals.toFixed(2)}%`);
                                                // 매도 완료 후 대화창 업데이트 (분리된 모듈 사용)
                                                let dialogMessage = '';
                                                if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
                                                    dialogMessage = window.decisionSystem.generateDecisionLogMessage('매도 완료', currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate);
                                                } else {
                                                    // 폴백: 기본 매도 완료 메시지
                                                    dialogMessage = `💸 [매도 완료] N/B 코인: ${nbCoins}개, 미네랄: ${nbMinerals.toFixed(2)}%, 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
                                                }
                                                trainerDialog.setText(dialogMessage);
                                                if (window.logManager) {
                                                    window.logManager.addLog(dialogMessage);
                                                    // 매도 완료 상세 로그
                                                    const detailedLog = `💸 매도 완료 상세: 매도가=${currentPriceText}, 매도 전 예상 수익률=${sellProfitRate.toFixed(2)}%, 현재 구역=${currentMajority}, 트레이너 위치=(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)}), 누적 미네랄=${nbMinerals.toFixed(2)}%`;
                                                    window.logManager.addLog(detailedLog);
                                                }
                                                 console.log(`💸 매도 완료! N/B 코인: ${nbCoins}개, N/B 미네랄 누적: ${nbMinerals.toFixed(2)}%, 수익률 리셋`);
                                                 
                                                 // 매도 완료 후 상태 정보 로그 (분리된 모듈 사용)
                                                 if (window.logManager) {
                                                     let currentPnl = 0;
                                                     if (window.learningSystem && typeof window.learningSystem.getCurrentProfitRate === 'function') {
                                                         currentPnl = window.learningSystem.getCurrentProfitRate();
                                                     } else {
                                                         // 폴백: DOM에서 직접 수익률 가져오기
                                                         const pnlElement = document.getElementById('selected-coin-pnl');
                                                         if (pnlElement) {
                                                             const pnlMatch = pnlElement.textContent.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
                                                             if (pnlMatch) {
                                                                 currentPnl = parseFloat(pnlMatch[1]);
                                                             }
                                                         }
                                                     }
                                                     const statusLog = `💸 매도 완료 후 상태: N/B 코인 ${nbCoins}개, 드랍 아이템 ${nbCoinItems.length}개, N/B 미네랄 ${nbMinerals.toFixed(2)}%, 매수 전 예상 수익률 ${buyProfitRate.toFixed(2)}%, 매도 전 예상 수익률 ${sellProfitRate.toFixed(2)}%, 수익율: ${currentPnl.toFixed(2)}%, BTC 현재가 ${currentPriceText}, 현재 구역 ${currentMajority}`;
                                                     window.logManager.addLog(statusLog);
                                                 }
                                                 
                                                 // 매도 완료 후 다음 의사결정 실행 (한 번만)
                                                 if (!model.postSellDecisionMade) {
                                                     model.postSellDecisionMade = true; // 플래그 설정
                                                     const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                                                         ? window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
                                                         : null;
                                                     if (nextDecision) {
                                                         targetAction = nextDecision.action;
                                                         model.targetAction = targetAction; // model에 저장
                                                         model.targetX = nextDecision.targetX;
                                                         model.targetY = nextDecision.targetY;
                                                         
                                                         // 의사결정에 따른 색상 변경 (분리된 모듈 사용)
                                                         let actionColor = 0x88ccff; // 기본 색상
                                                         if (window.decisionSystem && typeof window.decisionSystem.getActionColor === 'function') {
                                                             actionColor = window.decisionSystem.getActionColor(targetAction);
                                                         } else {
                                                             // 폴백: 기본 색상 매핑
                                                             switch (targetAction) {
                                                                 case '매수': actionColor = 0x0088ff; break;
                                                                 case '매도': actionColor = 0xff8800; break;
                                                                 case '신호 대기': actionColor = 0x88ccff; break;
                                                                 case 'BTC 시장 탐색': actionColor = 0x88ff88; break;
                                                                 default: actionColor = 0x88ccff;
                                                             }
                                                         }
                                                         model.circle.setFillStyle(actionColor);
                                                         console.log(`🎨 매도 완료 후: ${targetAction} 의사결정! 색상 변경 (${actionColor.toString(16)})`);
                                                         
                                                         if (window.logManager) {
                                                             window.logManager.addLog(`🔄 매도 완료 후 다음 의사결정: ${targetAction}`);
                                                         }
                                                     }
                                                 }
                                                 
                                                 // 상태 즉시 저장 (분리된 모듈 사용)
                                                 const gameData = {
                                                     nbCoins: nbCoins,
                                                     nbMinerals: nbMinerals,
                                                     buyPrice: buyPrice,
                                                     buyProfitRate: buyProfitRate,
                                                     sellProfitRate: sellProfitRate,
                                                     lastBuyAction: lastBuyAction,
                                                     lastSellAction: lastSellAction,
                                                     aiModels: window.gameStateManager && typeof window.gameStateManager.convertAiModelsForStorage === 'function'
                                                         ? window.gameStateManager.convertAiModelsForStorage(aiModels)
                                                         : [],
                                                     nbCoinItems: window.gameStateManager && typeof window.gameStateManager.convertNBCoinItemsForStorage === 'function'
                                                         ? window.gameStateManager.convertNBCoinItemsForStorage(nbCoinItems)
                                                         : []
                                                 };
                                                 if (window.gameStateManager && typeof window.gameStateManager.saveGameState === 'function') {
                                                     window.gameStateManager.saveGameState(gameData);
                                                 }
                                             }
                                             // 매도 액션 처리 (ORANGE 구역이 아닐 때)
                                             else if (targetAction === '매도' && !lastSellAction && nbCoins > 0 && currentMajority !== 'ORANGE') {
                                                 // ORANGE 구역이 아니면 신호 대기 센터로 이동
                                                 targetAction = '신호 대기';
                                                 model.targetAction = targetAction;
                                                 model.targetX = config.width / 2;
                                                 model.targetY = config.height / 2;
                                                 model.circle.setFillStyle(0x88ccff); // 하늘색 (신호 대기)
                                                 
                                                 if (window.logManager) {
                                                     window.logManager.addLog(`🔵 매도 시도 중이지만 ORANGE 구역이 아님 (현재: ${currentMajority}) → 신호 대기 센터로 이동`);
                                                 }
                                             }
                                             // 대기 상태
                                             else if (targetAction === '대기') {
                                                 lastBuyAction = false;
                                                 lastSellAction = false;
                                                // 대기 상태 대화창 업데이트 (분리된 모듈 사용)
                                                let dialogMessage = '';
                                                if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
                                                    dialogMessage = window.decisionSystem.generateDecisionLogMessage('대기', currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate);
                                                } else {
                                                    // 폴백: 기본 대기 메시지
                                                    dialogMessage = `⏳ [대기] N/B 코인: ${nbCoins}개, 미네랄: ${nbMinerals.toFixed(2)}%, 현재 구역: ${currentMajority}`;
                                                }
                                                trainerDialog.setText(dialogMessage);
                                                if (window.logManager) {
                                                    window.logManager.addLog(dialogMessage);
                                                }
                                             }
                                             // 신호 대기 상태
                                             else if (targetAction === '신호 대기') {
                                                 lastBuyAction = false;
                                                 lastSellAction = false;
                                                 
                                                 // 신호 대기 센터에 도달했을 때 카운트다운 표시
                                                 const distanceToCenter = Math.sqrt((modelX - (config.width / 2)) ** 2 + (modelY - (config.height / 2)) ** 2);
                                                 
                                                 // 신호 대기 센터에서의 상태 표시 (타이머 로직은 1227번 라인에서 처리)
                                                // 신호 대기 상태 대화창 업데이트 (분리된 모듈 사용)
                                                let dialogMessage = '';
                                                if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
                                                    dialogMessage = window.decisionSystem.generateDecisionLogMessage('신호 대기', currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate);
                                                } else {
                                                    // 폴백: 기본 신호 대기 메시지
                                                    dialogMessage = `🔵 [신호 대기] N/B 코인: ${nbCoins}개, 미네랄: ${nbMinerals.toFixed(2)}%, 현재 구역: ${currentMajority}`;
                                                }
                                                trainerDialog.setText(dialogMessage);
                                                if (window.logManager) {
                                                    window.logManager.addLog(dialogMessage);
                                                }
                                             }
                                             // BTC 시장 탐색 상태
                                             else if (targetAction === 'BTC 시장 탐색') {
                                                 lastBuyAction = false;
                                                 lastSellAction = false;
                                                // BTC 시장 탐색 상태 대화창 업데이트 (분리된 모듈 사용)
                                                let dialogMessage = '';
                                                if (window.decisionSystem && typeof window.decisionSystem.generateDecisionLogMessage === 'function') {
                                                    dialogMessage = window.decisionSystem.generateDecisionLogMessage('BTC 시장 탐색', currentMajority, nbCoins, nbMinerals, nbCoinItems, currentPriceText, buyProfitRate, sellProfitRate);
                                                } else {
                                                    // 폴백: 기본 BTC 시장 탐색 메시지
                                                    dialogMessage = `🔍 [BTC 시장 탐색] N/B 코인: ${nbCoins}개, 미네랄: ${nbMinerals.toFixed(2)}%, 현재 구역: ${currentMajority}`;
                                                }
                                                trainerDialog.setText(dialogMessage);
                                                if (window.logManager) {
                                                    window.logManager.addLog(dialogMessage);
                                                }
                                             }
                                             // 정보 수집 후 신호 대기 센터로 복귀
                                             else if (targetAction === '정보 수집 완료') {
                                                 // 예상 수익률이 계산되었고 BLUE 신호라면 즉시 매수
                                                 if (currentMajority === 'BLUE' && buyProfitRate !== 0) {
                                                     targetAction = '매수';
                                                     model.targetAction = targetAction; // model에 저장
                                                     model.targetX = startX;
                                                     model.targetY = topY;
                                                     model.circle.setFillStyle(0x0088ff); // 파란색 (매수)
                                                     console.log(`📈 트레이너: 정보 수집 완료 후 즉시 매수! 예상 수익률: ${buyProfitRate.toFixed(2)}%`);
                                                     const dialogMessage = `📈 [정보 수집 완료] 예상 수익률 계산됨(${buyProfitRate.toFixed(2)}%) → 즉시 매수!`;
                                                     trainerDialog.setText(dialogMessage);
                                                     if (window.logManager) {
                                                         window.logManager.addLog(dialogMessage);
                                                     }
                                                 } else {
                                                     // 예상 수익률이 계산되지 않았거나 BLUE 신호가 아니면 신호 대기 센터로 복귀
                                                 targetAction = '신호 대기';
                                                     model.targetAction = targetAction; // model에 저장
                                                 model.targetX = config.width / 2;
                                                 model.targetY = config.height / 2;
                                                 model.circle.setFillStyle(0x88ccff);
                                                 model.waitCheckTimer = 0; // 타이머 리셋
                                                 model.countdownStarted = false; // 카운트다운 플래그도 리셋
                                                 console.log(`🔵 트레이너: 정보 수집 완료, 신호 대기 센터로 복귀!`);
                                                     const dialogMessage = `🔵 [정보 수집 완료] 신호 대기 센터로 복귀 중...`;
                                                     trainerDialog.setText(dialogMessage);
                                                     if (window.logManager) {
                                                         window.logManager.addLog(dialogMessage);
                                                     }
                                                 }
                                             }
                                         }
                                        
                                        // 트레이너 이동 (분리된 모듈 사용)
                                        if (window.trainerMovementController) {
                                            // 디버깅: 모듈 로드 확인 및 목표 위치 설정
                                            if (model.isTrainer) {
                                                // 목표 위치가 설정되지 않았으면 기본값 설정
                                                if (typeof model.targetX === 'undefined' || typeof model.targetY === 'undefined') {
                                                    model.targetX = config.width / 2;
                                                    model.targetY = config.height / 2;
                                                    model.targetAction = '신호 대기';
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔧 트레이너 목표 위치 초기화: (${model.targetX}, ${model.targetY})`);
                                                    }
                                                }
                                                
                                                // 이동 처리
                                                window.trainerMovementController.updateTrainerMovement(model, config);
                                                
                                                // 디버깅: 트레이너 상태 확인 (5초마다)
                                                if (Math.floor(Date.now() / 1000) % 5 === 0) {
                                                    const distanceToTarget = window.trainerMovementController.calculateDistanceToTarget(model);
                                                    if (window.logManager) {
                                                        window.logManager.addLog(`🔍 트레이너 상태: targetAction=${targetAction}, 현재위치=(${Math.round(modelX)}, ${Math.round(modelY)}), 목표위치=(${Math.round(model.targetX)}, ${Math.round(model.targetY)}), 거리=${Math.round(distanceToTarget)}px`);
                                                    }
                                                }
                                            }
                                        } else {
                                            // 폴백: 기존 이동 로직
                                            const dx = model.targetX - modelX;
                                            const dy = model.targetY - modelY;
                                            
                                            if (Math.abs(dx) <= 2 && Math.abs(dy) <= 2) {
                                                model.circle.x = model.targetX;
                                                model.circle.y = model.targetY;
                                                model.name.x = model.circle.x;
                                                model.name.y = model.circle.y - 6;
                                                model.role.x = model.circle.x;
                                                model.role.y = model.circle.y + 6;
                                            } else {
                                                if (Math.abs(dx) > 0.5) {
                                                    model.circle.x += dx * 0.1;
                                                    model.name.x = model.circle.x;
                                                    model.role.x = model.circle.x;
                                                }
                                                
                                                if (Math.abs(dy) > 0.5) {
                                                    model.circle.y += dy * 0.1;
                                                    model.name.y = model.circle.y - 6;
                                                    model.role.y = model.circle.y + 6;
                                                }
                                            }
                                        }
                                         
                                            // 트레이너 이동 중 대화창 업데이트 (분리된 모듈 사용)
                                            if (window.trainerMovementController) {
                                                const distanceToTarget = window.trainerMovementController.calculateDistanceToTarget(model);
                                                if (distanceToTarget > 30) {
                                                    // 목표에서 벗어나면 도착 로그 플래그 리셋
                                                    model.arrivalLogged = false;
                                                    
                                                    const currentZone = window.decisionSystem && typeof window.decisionSystem.getCurrentZone === 'function'
                                                        ? window.decisionSystem.getCurrentZone(modelX, modelY, startX, topY, spacing, config)
                                                        : '기타영역';
                                                    const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                                                        ? window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
                                                        : null;
                                                    
                                                    let movingMessage = '';
                                                    if (window.decisionSystem && typeof window.decisionSystem.generateMovingMessage === 'function') {
                                                        movingMessage = window.decisionSystem.generateMovingMessage(targetAction, distanceToTarget, currentMajority, nbCoins, buyProfitRate, sellProfitRate, currentZone, nextDecision);
                                                    } else {
                                                        // 폴백: 기본 이동 메시지
                                                        movingMessage = `🎯 [의사결정: ${targetAction}] 이동 중... (${Math.round(distanceToTarget)}px 남음)`;
                                                    }
                                                    
                                                    trainerDialog.setText(movingMessage);
                                                    if (window.logManager && typeof window.logManager.addLog === 'function') {
                                                        window.logManager.addLog(movingMessage);
                                                    }
                                                }
                                            } else {
                                                // 폴백: 기존 이동 중 대화창 업데이트 로직
                                                const distanceToTarget = Math.sqrt((model.targetX - modelX) ** 2 + (model.targetY - modelY) ** 2);
                                                if (distanceToTarget > 30) {
                                                    model.arrivalLogged = false;
                                                    
                                                    const currentZone = window.decisionSystem && typeof window.decisionSystem.getCurrentZone === 'function'
                                                        ? window.decisionSystem.getCurrentZone(modelX, modelY, startX, topY, spacing, config)
                                                        : '기타영역';
                                                    const nextDecision = window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function'
                                                        ? window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config)
                                                        : null;
                                                    
                                                    let movingMessage = '';
                                                    if (window.decisionSystem && typeof window.decisionSystem.generateMovingMessage === 'function') {
                                                        movingMessage = window.decisionSystem.generateMovingMessage(targetAction, distanceToTarget, currentMajority, nbCoins, buyProfitRate, sellProfitRate, currentZone, nextDecision);
                                                    } else {
                                                        movingMessage = `🎯 [의사결정: ${targetAction}] 이동 중... (${Math.round(distanceToTarget)}px 남음)`;
                                                    }
                                                    
                                                    trainerDialog.setText(movingMessage);
                                                    if (window.logManager && typeof window.logManager.addLog === 'function') {
                                                        window.logManager.addLog(movingMessage);
                                                    }
                                                }
                                            }
                                    
                                }
                            });
                                
                            // 전체 상태 업데이트
                            const totalDiscovered = aiModels.reduce((sum, model) => sum + model.discoveredCoords.length, 0);
                            learningStatus.setText(`AI 시스템 작동 중 - 총 발견 좌표: ${totalDiscovered}`);
                                 
                            // 실시간 데이터 표시 업데이트
                            aiModels[4].name.setText(`Trainer (${currentMajority})`);
                            
                            // 트레이너 역할 텍스트를 의사결정 상태로 업데이트
                            const trainerModel = aiModels.find(model => model.isTrainer);
                                 if (trainerModel) {
                                     // 트레이너의 현재 의사결정 상태를 결정 (구역 기반)
                                     let currentAction = '신호 대기';
                                     
                                     // 트레이너의 현재 의사결정 상태를 결정 (분리된 모듈 사용)
                                     if (window.decisionSystem && typeof window.decisionSystem.getNextDecision === 'function') {
                                         const nextDecision = window.decisionSystem.getNextDecision(currentMajority, nbCoins, buyProfitRate, sellProfitRate, startX, topY, spacing, config);
                                         currentAction = nextDecision ? nextDecision.action : '신호 대기';
                                     }
                                     
                                     // 학습 모델 예측 수익률 포함하여 트레이너 역할 텍스트 업데이트 (분리된 모듈 사용)
                                     const predictedRate = window.learningSystem && typeof window.learningSystem.getPredictedRateDisplayText === 'function'
                                         ? window.learningSystem.getPredictedRateDisplayText()
                                         : '';
                                     trainerModel.role.setText(`${currentAction}${predictedRate} (${currentPriceText})`);
                                 }
                                 
                            // 디버깅: 첫 번째 탐색자의 위치 정보 표시
                            if (aiModels[0] && aiModels[0].isExplorer) {
                                     const explorer1 = aiModels[0];
                                     const distanceToBuy = Math.sqrt((explorer1.circle.x - startX) ** 2 + (explorer1.circle.y - topY) ** 2);
                                     const distanceToGuild = Math.sqrt((explorer1.circle.x - 100) ** 2 + (explorer1.circle.y - 100) ** 2);
                                     const distanceToMarket = Math.sqrt((explorer1.circle.x - (config.width - 100)) ** 2 + (explorer1.circle.y - (config.height - 100)) ** 2);
                                     
                                     console.log(`🔍 Explorer-1 위치: (${Math.round(explorer1.circle.x)}, ${Math.round(explorer1.circle.y)})`);
                                     console.log(`📏 거리 - 매수: ${Math.round(distanceToBuy)}, 길드: ${Math.round(distanceToGuild)}, 시장: ${Math.round(distanceToMarket)}`);
                                 }
                                 
                            // 트레이너와 탐색자들을 연결하는 선 그리기
                            connectionLines.clear();
                                  
                            // N/B 길드와 트레이너 연결선 그리기
                            guildTrainerConnection.clear();
                                  const trainer = aiModels.find(model => model.isTrainer);
                                  if (trainer) {
                                      // N/B 코인이 1개 이상일 때 연결선 색상 변경
                                      const connectionColor = nbCoins > 0 ? 0xffaa00 : 0x00ff00; // 주황색 또는 초록색
                                      const connectionAlpha = nbCoins > 0 ? 0.8 : 0.6; // 더 진하게 또는 반투명
                                      
                                      // N/B 길드와 트레이너 연결선 (항시 연결)
                                      guildTrainerConnection.lineStyle(2, connectionColor, connectionAlpha);
                                      guildTrainerConnection.beginPath();
                                      guildTrainerConnection.moveTo(100, 100); // N/B 길드 위치
                                      guildTrainerConnection.lineTo(trainer.circle.x, trainer.circle.y);
                                      guildTrainerConnection.strokePath();
                                      
                                      // 연결선 중간에 데이터 전송 표시 (더 큰 원)
                                      const midX = (100 + trainer.circle.x) / 2;
                                      const midY = (100 + trainer.circle.y) / 2;
                                      guildTrainerConnection.fillStyle(connectionColor, connectionAlpha + 0.2);
                                      guildTrainerConnection.fillCircle(midX, midY, 3);
                                      
                                      // 각 탐색자와 트레이너를 연결
                                      aiModels.forEach((model, index) => {
                                          if (model.isExplorer) {
                                              // 연결선 그리기
                                              connectionLines.lineStyle(1, 0x00ff88, 0.4); // 반투명 초록색
                                              connectionLines.beginPath();
                                              connectionLines.moveTo(trainer.circle.x, trainer.circle.y);
                                              connectionLines.lineTo(model.circle.x, model.circle.y);
                                              connectionLines.strokePath();
                                              
                                              // 연결선 중간에 작은 원 그리기 (데이터 전송 표시)
                                              const midX = (trainer.circle.x + model.circle.x) / 2;
                                              const midY = (trainer.circle.y + model.circle.y) / 2;
                                              connectionLines.fillStyle(0x00ff88, 0.6);
                                              connectionLines.fillCircle(midX, midY, 2);
                                          }
                                      });
                                      
                                      // N/B 코인 아이템과 탐색자들을 연결하는 선 그리기
                                      nbCoinItems.forEach((item, itemIndex) => {
                                          if (!item.collected) {
                                              // 기존 연결선들 제거
                                              item.connectionLines.forEach(line => {
                                                  if (line && line.destroy) {
                                                      line.destroy();
                                                  }
                                              });
                                              item.connectionLines = [];
                                              
                                              // 각 탐색자와 N/B 코인 아이템을 연결
                                              aiModels.forEach((model, modelIndex) => {
                                                  if (model.isExplorer) {
                                                      const connectionLine = this.add.graphics();
                                                      connectionLine.lineStyle(1, 0xffaa00, 0.5); // 주황색, 반투명
                                                      connectionLine.beginPath();
                                                      connectionLine.moveTo(item.polygon.x, item.polygon.y);
                                                      connectionLine.lineTo(model.circle.x, model.circle.y);
                                                      connectionLine.strokePath();
                                                      
                                                      // 연결선 중간에 작은 원 그리기 (수집 가능 표시)
                                                      const midX = (item.polygon.x + model.circle.x) / 2;
                                                      const midY = (item.polygon.y + model.circle.y) / 2;
                                                      connectionLine.fillStyle(0xffaa00, 0.7);
                                                      connectionLine.fillCircle(midX, midY, 2);
                                                      
                                                      item.connectionLines.push(connectionLine);
                                                  }
                                              });
                                          }
                                      });
                                  }
                            };
                            
                            // 게임 시작 시 이전 상태 로드 (분리된 모듈 사용)
                            if (window.gameStateManager && typeof window.gameStateManager.loadGameState === 'function') {
                                const gameState = window.gameStateManager.loadGameState();
                                if (gameState) {
                                    // 기본 변수들 복원
                                    nbCoins = gameState.nbCoins || 0;
                                    nbMinerals = gameState.nbMinerals || 0.0;
                                    buyPrice = gameState.buyPrice || 0;
                                    buyProfitRate = gameState.buyProfitRate || 0;
                                    sellProfitRate = gameState.sellProfitRate || 0;
                                    lastBuyAction = gameState.lastBuyAction || false;
                                    lastSellAction = gameState.lastSellAction || false;
                                    
                                    // AI 모델들 위치 복원
                                    if (typeof window.gameStateManager.restoreAiModelsFromStorage === 'function') {
                                        window.gameStateManager.restoreAiModelsFromStorage(gameState.aiModels, aiModels);
                                    }
                                    
                                    // N/B 코인 아이템들 복원
                                    if (typeof window.gameStateManager.restoreNBCoinItemsFromStorage === 'function') {
                                        window.gameStateManager.restoreNBCoinItemsFromStorage(gameState.nbCoinItems, this, nbCoinItems);
                                    }
                                    
                                    // UI 업데이트
                                    nbCoinDisplay.setText(`N/B 코인: ${nbCoins}개 (드랍 아이템: ${nbCoinItems.length}개)`);
                                    nbMineralDisplay.setText(`N/B 미네랄: ${nbMinerals.toFixed(2)}%`);
                                    buyProfitRateDisplay.setText(`매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`);
                                    sellProfitRateDisplay.setText(`매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`);
                                    
                                    const dialogMessage = `🎯 트레이너: 이전 상태 복원 완료! N/B 코인: ${nbCoins}개, AI 시스템 시작 중...`;
                                    trainerDialog.setText(dialogMessage);
                                    if (window.logManager && typeof window.logManager.addLog === 'function') {
                                        window.logManager.addLog(dialogMessage);
                                    }
                                }
                            } else {
                                console.warn('⚠️ GameStateManager 모듈이 로드되지 않았습니다. 기본 상태로 시작합니다.');
                            }
                                 
                            // 게임 시작 후 N/B 코인 아이템이 있으면 탐색자들의 목표를 설정
                            setTimeout(() => {
                                     const availableItems = nbCoinItems.filter(item => !item.collected);
                                     if (availableItems.length > 0) {
                                         aiModels.forEach((model, index) => {
                                             if (model.isExplorer) {
                                                 // 각 탐색자에게 가장 가까운 N/B 코인 아이템을 목표로 설정
                                                 const nearestItem = availableItems.reduce((nearest, item) => {
                                                     const distanceToNearest = Math.sqrt((model.circle.x - nearest.polygon.x) ** 2 + (model.circle.y - nearest.polygon.y) ** 2);
                                                     const distanceToItem = Math.sqrt((model.circle.x - item.polygon.x) ** 2 + (model.circle.y - item.polygon.y) ** 2);
                                                     return distanceToItem < distanceToNearest ? item : nearest;
                                                 });
                                                 
                                                 model.targetX = nearestItem.polygon.x;
                                                 model.targetY = nearestItem.polygon.y;
                                                 model.role.setText(`N/B 코인 수집 중 (${model.discoveredCoords.length}/8)`);
                                                 console.log(`🎯 Explorer-${index + 1} 게임 시작 시 N/B 코인 아이템 목표 설정`);
                                             }
                                         });
                                     }
                                 }, 2000);
                             
                            // AI 시스템 루프 시작
                            this.time.addEvent({
                                 delay: 100,
                                 callback: aiSystemAlgorithm,
                                 loop: true
                             });
                             
                            // 상태 저장 루프 (5초마다 자동 저장) - 분리된 모듈 사용
                            this.time.addEvent({
                                delay: 5000,
                                callback: () => {
                                    if (window.gameStateManager && typeof window.gameStateManager.checkAutoSave === 'function') {
                                        const gameData = {
                                            nbCoins: nbCoins,
                                            nbMinerals: nbMinerals,
                                            buyPrice: buyPrice,
                                            buyProfitRate: buyProfitRate,
                                            sellProfitRate: sellProfitRate,
                                            lastBuyAction: lastBuyAction,
                                            lastSellAction: lastSellAction,
                                            aiModels: window.gameStateManager.convertAiModelsForStorage ? window.gameStateManager.convertAiModelsForStorage(aiModels) : [],
                                            nbCoinItems: window.gameStateManager.convertNBCoinItemsForStorage ? window.gameStateManager.convertNBCoinItemsForStorage(nbCoinItems) : []
                                        };
                                        window.gameStateManager.checkAutoSave(gameData);
                                    }
                                },
                                loop: true
                            });
                            
                            // 다각형 회전 애니메이션
                            this.tweens.add({
                                targets: guildPolygon,
                                rotation: Math.PI * 2,
                                duration: 8000,
                                repeat: -1,
                                ease: 'Linear'
                            });
                            
                            this.tweens.add({
                                targets: marketPolygon,
                                rotation: -Math.PI * 2,
                                duration: 6000,
                                repeat: -1,
                                ease: 'Linear'
                            });
                            
                            this.tweens.add({
                                targets: [buyPolygon, sellPolygon, waitPolygon],
                                rotation: Math.PI * 2,
                                duration: 10000,
                                repeat: -1,
                                ease: 'Linear'
                            });
                            
                            // AI 모델들의 크기 변화 애니메이션
                            aiModels.forEach(model => {
                                this.tweens.add({
                                    targets: [model.circle, model.name, model.role],
                                    scaleX: 1.1,
                                    scaleY: 1.1,
                                    duration: 2000 + Math.random() * 1000,
                                    yoyo: true,
                                    repeat: -1
                                });
                            });
                            
                            console.log('🎮 AI models system created successfully');
                            
                            // 트레이너 모듈 초기화
                            initializeTrainerModules();
                             
                            // 게임 시작 완료 메시지
                            setTimeout(() => {
                                if (trainerDialog) {
                                    const dialogMessage = '🎯 트레이너: AI 시스템 준비 완료! 데이터 분석 시작...';
                                    trainerDialog.setText(dialogMessage);
                                    if (window.logManager) {
                                        window.logManager.addLog(dialogMessage);
                                    }
                                }
                            }, 3000);
                        }
                    }
                };
                
                if (window.floatingBallGame) {
                    window.floatingBallGame.destroy(true);
                }
                
                window.floatingBallGame = new Phaser.Game(config);
                console.log('🎮 AI models game creation successful');
            } else {
                console.error('❌ Floating ball game container not found');
            }
        }, 1000);
        
        return boundTemplate;
    }
}

// 전역 인스턴스 생성
window.templateLoader = new TemplateLoader();
