// Game Initializer Module
// Phaser 게임 초기화 및 기본 화면 구현을 담당

class GameInitializer {
    constructor() {
        this.game = null;
        this.scene = null;
        this.aiModels = [];
        this.nbCoinItems = [];
        this.gameData = {
            nbCoins: 0,
            // N/B 미네랄 평균 표시를 위해 합계/개수 동시 관리
            nbMinerals: 0.0,          // 표시용 평균 값
            nbMineralsSum: 0.0,       // 누적 합계
            nbMineralsCount: 0,       // 누적 개수
            buyPrice: 0,
            buyProfitRate: 0,
            sellProfitRate: 0,
            buyThresholdPercent: 0.5,
            sellThresholdPercent: -5.0, // 매도 임계치를 -5.0%로 설정하여 손실 허용 (모의전)
            lastBuyAction: false,
            lastSellAction: false,
            dropItemsCount: 0, // 현재 필드에 있는 드랍 아이템 개수
            dropItemsCollected: 0 // 누적 수집된 드랍 아이템 개수 (통계용)
        };
        // 학습 기반 임계치 결정기
        this.buyThresholdLearner = null;
        this.sellThresholdLearner = null;
        
        // 자동 저장 관련 변수 (game-state-manager.js에서 처리)
        this.autoSaveInterval = null;
        this.lastSaveTime = Date.now();
        this.saveIntervalMs = 30000; // 30초마다 자동 저장
        
        // cardStorageSystem 초기화 확인
        this.checkCardStorageSystem();
    }

    // cardStorageSystem 초기화 확인
    checkCardStorageSystem() {
        if (!window.cardStorageSystem) {
            console.log('⚠️ cardStorageSystem이 아직 초기화되지 않았습니다. 1초 후 다시 확인합니다.');
            setTimeout(() => {
                this.checkCardStorageSystem();
            }, 1000);
            return;
        }
        
        if (typeof window.cardStorageSystem.addNBCoin !== 'function' || 
            typeof window.cardStorageSystem.removeNBCoin !== 'function') {
            console.log('⚠️ cardStorageSystem의 필수 함수들이 아직 로드되지 않았습니다. 1초 후 다시 확인합니다.');
            setTimeout(() => {
                this.checkCardStorageSystem();
            }, 1000);
            return;
        }
        
        console.log('✅ cardStorageSystem이 정상적으로 초기화되었습니다.');
        
        // N/B MIN 코인과 N/B MAX COIN 분리 확인
        console.log(`🔍 초기화 완료 - N/B MIN 코인: ${this.gameData.nbCoins}개`);
        
        // N/B MAX COIN 상태 확인
        const totalMaxCoins = this.checkNBMaxCoins();
        console.log(`🔍 초기화 완료 - N/B MAX COIN 총합: ${totalMaxCoins}개`);
        
        console.log('✅ N/B MIN 코인과 N/B MAX COIN이 완전히 분리되었습니다.');
    }
    
    // N/B MIN 코인 리셋
    resetNBMINCoins() {
        try {
            console.log('🔄 N/B MIN 코인 리셋 시작');
            const oldValue = this.gameData.nbCoins;
            this.gameData.nbCoins = 0;
            
            // UI 업데이트
            if (window.nbCoinDisplay) {
                const coinText = `N/B MIN 코인: ${this.gameData.nbCoins}개`;
                window.nbCoinDisplay.setText(coinText);
            }
            
            console.log(`🔄 N/B MIN 코인 리셋 완료: ${oldValue} → ${this.gameData.nbCoins}개`);
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 N/B MIN 코인 리셋: ${oldValue}개 → ${this.gameData.nbCoins}개`);
            }
        } catch (error) {
            console.error('❌ N/B MIN 코인 리셋 중 오류:', error);
        }
    }

    // N/B MAX COIN 상태 확인 (N/B MIN 코인과 별개)
    checkNBMaxCoins() {
        try {
            if (window.cardStorageSystem) {
                // 모든 타임프레임의 N/B MAX COIN 개수 합계 계산
                const timeframes = ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '1D'];
                let totalMaxCoins = 0;
                
                console.log('🔍 N/B MAX COIN 상태 확인...');
                
                for (const timeframe of timeframes) {
                    const storage = window.cardStorageSystem.getCardStorage(timeframe);
                    if (storage) {
                        console.log(`🔍 ${timeframe} 타임프레임: N/B MAX COIN ${storage.nbCoins}개`);
                        if (storage.nbCoins > 0) {
                            totalMaxCoins += storage.nbCoins;
                        }
                    }
                }
                
                console.log(`🔍 N/B MAX COIN 총합: ${totalMaxCoins}개`);
                return totalMaxCoins;
            } else {
                console.log('⚠️ cardStorageSystem이 초기화되지 않았습니다.');
                return 0;
            }
        } catch (error) {
            console.error('❌ N/B MAX COIN 확인 중 오류:', error);
            return 0;
        }
    }

    // gameData와 cardStorageSystem 동기화 (더 이상 사용하지 않음 - N/B MIN 코인과 N/B MAX COIN 분리)
    syncGameDataWithCardStorage() {
        console.log('⚠️ syncGameDataWithCardStorage는 더 이상 사용하지 않습니다. N/B MIN 코인과 N/B MAX COIN이 분리되었습니다.');
    }

    // 게임 초기화
    initializeGame(container) {
        //console.log('🎮 Game Initializer: 게임 초기화 시작');
        
        if (typeof Phaser === 'undefined') {
            console.error('❌ Phaser library not loaded');
            return;
        }

        // 기존 게임이 있으면 제거
        if (window.floatingBallGame) {
            //console.log('🧹 기존 게임 인스턴스 정리 중...');
            window.floatingBallGame.destroy(true);
            window.floatingBallGame = null;
        }
        
        // 기존 AI 모델들도 정리
        if (this.aiModels && this.aiModels.length > 0) {
            //console.log(`🧹 기존 AI 모델 ${this.aiModels.length}개 정리 중...`);
            this.aiModels.forEach(model => {
                if (model.circle) model.circle.destroy();
                if (model.name) model.name.destroy();
                if (model.role) model.role.destroy();
            });
            this.aiModels = [];
        }

        // 게임 씬 클래스 정의
        class GameScene extends Phaser.Scene {
            constructor() {
                super({ key: 'GameScene' });
            }

            create() {
                //console.log('🎮 Scene created, initializing game scene...');
                this.gameInitializer.scene = this;
                this.gameInitializer.createGameScene();
            }

            update() {
                if (this.gameInitializer) {
                    this.gameInitializer.updateGame();
                }
            }
        }

        const config = {
            type: Phaser.AUTO,
            parent: 'floating-ball-game',
            width: container.offsetWidth || 1086,
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
            scene: GameScene
        };

        // 씬에 게임 초기화자 참조 전달
        GameScene.prototype.gameInitializer = this;

        this.game = new Phaser.Game(config);
        window.floatingBallGame = this.game;
        
        //console.log('🎮 Game Initializer: 게임 초기화 완료');
    }

    // 게임 씬 생성
    createGameScene() {
        //console.log('🎮 Game Initializer: 게임 씬 생성 시작');
        
        if (!this.scene) {
            console.error('❌ Scene is not initialized');
            return;
        }
        
        try {
            // N/B 코인 드랍 시스템 초기화
            if (window.nbCoinDropSystem) {
                window.nbCoinDropSystem.initialize(this.scene, this.game.config);
                //console.log('✅ N/B 코인 드랍 시스템 초기화 완료');
            }

            // 임계치 학습기 초기화
            if (window.ThresholdLearner) {
                this.buyThresholdLearner = new window.ThresholdLearner('buy');
                this.sellThresholdLearner = new window.ThresholdLearner('sell');
                //console.log('✅ 임계치 학습기 초기화 완료');
            }
            
            // majority-zone DOM 변화 감지 → N/B 길드 인디케이터 및 트레이너 원 동기화
            const majorityEl = document.getElementById('majority-zone');
            if (majorityEl) {
                const updateGuildIndicator = () => {
                    const val = (majorityEl.textContent || '').trim();
                    const color = this.getMajorityColor(val);
                    if (window.guildZoneIndicator && typeof window.guildZoneIndicator.setFillStyle === 'function') {
                        window.guildZoneIndicator.setFillStyle(color);
                    }
                    if (window.guildZoneText && typeof window.guildZoneText.setText === 'function') {
                        window.guildZoneText.setText(val);
                    }
                    const trainer = this.aiModels?.find(m => m.isTrainer);
                    if (trainer && trainer.circle && typeof trainer.circle.setFillStyle === 'function') {
                        trainer.circle.setFillStyle(color);
                    }
                };
                updateGuildIndicator();
                try {
                    const observer = new MutationObserver(updateGuildIndicator);
                    observer.observe(majorityEl, { characterData: true, subtree: true, childList: true });
                    window._majorityZoneObserver = observer;
                } catch (e) {
                    // MutationObserver 사용 불가 시 주기적 폴링
                    setInterval(updateGuildIndicator, 500);
                }
            }

            // (중복 제거) 기존 동기화 블록 유지
            
            // 주민 수집 시스템 초기화
            if (window.residentCollectionSystem) {
                window.residentCollectionSystem.initialize(this.scene, this.game.config);
                //console.log('✅ 주민 수집 시스템 초기화 완료');
            }
            
            // 탐색원 이동 시스템 초기화
            if (window.explorerMovementSystem) {
                window.explorerMovementSystem.initialize(this.scene, this.game.config);
                //console.log('✅ 탐색원 이동 시스템 초기화 완료');
            }
            
            // 주민 지속성 관리자 초기화
            if (window.residentPersistenceManager) {
                window.residentPersistenceManager.initialize();
                //console.log('✅ 주민 지속성 관리자 초기화 완료');
            }
            
            // 2D 맵 그리드 생성
            this.createGrid();
            
            // 게임 영역 생성
            this.createGameAreas();
            
            // AI 모델들 생성
            this.createAIModels();
            
            // UI 요소들 생성
            this.createUIElements();
            
            // 이벤트 리스너 설정
            this.setupEventListeners();
            
            // 트레이너 이동 속도 로드
            this.loadTrainerSpeed();
            
            // 이동 속도 표시 초기화
            this.updateTrainerSpeedDisplay();
            
            // 게임 루프 시작
            this.startGameLoop();
            
            // 트레이너 모듈 초기화
            if (typeof initializeTrainerModules === 'function') {
                initializeTrainerModules();
            }
            
            // Paper Trading Simulator 초기화 (자동 시작)
            if (window.PaperTradingSimulator && !window.paperTrading) {
                window.paperTrading = new window.PaperTradingSimulator(this);
                //console.log('✅ PaperTradingSimulator 초기화 완료');
                if (window.logManager) window.logManager.addLog('✅ PaperTradingSimulator 초기화 완료');
                
                // 모의전 즉시 시작 (사용자 요청: "모의전에서는 매도 구역에서는 그냥 매도가 되야 함")
                if (window.paperTrading && !window.paperTrading.isRunning) {
                    window.paperTrading.start();
                    //console.log('🚀 PaperTradingSimulator 즉시 시작됨');
                    if (window.logManager) window.logManager.addLog('🚀 PaperTradingSimulator 즉시 시작됨');
                }
            }

            // 게임 상태 관리자 초기화
            if (window.gameStateManager) {
                window.gameStateManager.initializeGameState();
                if (window.logManager) {
                    window.logManager.addLog(`💾 게임 상태 관리자 초기화 완료 - 자동 저장 시스템 활성화`);
                }
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`❌ 게임 상태 관리자를 찾을 수 없음 - 자동 저장 비활성화`);
                }
            }
            
            // 카드 저장소 시스템 초기화
            if (window.cardStorageSystem && typeof window.cardStorageSystem.initialize === 'function') {
                window.cardStorageSystem.initialize();
                if (window.logManager) {
                    window.logManager.addLog(`📦 카드 저장소 시스템 초기화 완료`);
                }
            } else {
                if (window.logManager) {
                    window.logManager.addLog(`❌ 카드 저장소 시스템을 찾을 수 없음`);
                }
            }
            
            // N/B 코인 디스플레이 초기화
            if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.updateNBCoinDisplay === 'function') {
                window.nbCoinDropSystem.updateNBCoinDisplay();
            }
            
            //console.log('🎮 Game Initializer: 게임 씬 생성 완료');
        } catch (error) {
            console.error('❌ Error creating game scene:', error);
        }
    }

    // 2D 맵 그리드 생성
    createGrid() {
        if (!this.scene || !this.game) {
            console.error('❌ Scene or game not initialized for grid creation');
            return;
        }
        
        const gridSize = 20;
        const cols = Math.floor(this.game.config.width / gridSize);
        const rows = Math.floor(this.game.config.height / gridSize);
        
        try {
            const graphics = this.scene.add.graphics();
            graphics.lineStyle(1, 0x00ff00, 0.3);
            
            for (let i = 0; i <= cols; i++) {
                graphics.moveTo(i * gridSize, 0);
                graphics.lineTo(i * gridSize, this.game.config.height);
            }
            
            for (let i = 0; i <= rows; i++) {
                graphics.moveTo(0, i * gridSize);
                graphics.lineTo(this.game.config.width, i * gridSize);
            }
            
            //console.log('✅ Grid created successfully');
        } catch (error) {
            console.error('❌ Error creating grid:', error);
        }
    }

    // 게임 영역 생성
    createGameAreas() {
        if (!this.scene || !this.game) {
            console.error('❌ Scene or game not initialized for game areas creation');
            return;
        }
        
        const config = this.game.config;
        
        // N/B 길드 다각형 (좌상단)
        const guildPolygon = this.scene.add.polygon(100, 100, [
            0, -30, 22, -15, 22, 15, 0, 30, -22, 15, -22, -15
        ], 0x00ff00);
        guildPolygon.setOrigin(0.5, 0.5);
        window.nbGuildPolygon = guildPolygon;
        
        // N/B 길드 내부 구역 표시 원 (majority-zone과 동기화)
        const guildZoneIndicator = this.scene.add.circle(100, 100, 15, 0x00d1ff);
        guildZoneIndicator.setOrigin(0.5, 0.5);
        window.guildZoneIndicator = guildZoneIndicator;
        
        // N/B 길드 내부 구역 텍스트 (majority 표시)
        const guildZoneText = this.scene.add.text(100, 100, 'BLUE', {
            fontSize: '8px',
            fill: '#ffffff',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        window.guildZoneText = guildZoneText;

        // 초기 majority-zone 기반으로 N/B 길드 인디케이터 및 트레이너 원 동기화
        try {
            const majorityEl = document.getElementById('majority-zone');
            const majorityVal = majorityEl ? (majorityEl.textContent || '').trim() : '';
            const color = this.getMajorityColor(majorityVal);
            guildZoneIndicator.setFillStyle(color);
            guildZoneText.setText(majorityVal || '');
            const trainer = this.aiModels?.find(m => m.isTrainer);
            if (trainer && trainer.circle && typeof trainer.circle.setFillStyle === 'function') {
                trainer.circle.setFillStyle(color);
            }
        } catch (e) {
            // DOM 미존재 시 무시
        }
        
        // BTC 시장 탐색 구역 (우하단) - Yellow 원형, 기존 대비 1.2배 (반지름 35 → 42)
        const marketCircle = this.scene.add.circle(config.width - 100, config.height - 100, 42, 0xffff00);
        marketCircle.setOrigin(0.5, 0.5);
        window.btcMarketPolygon = marketCircle;
        
        // 상단 매수/매도/대기 4각형들
        const spacing = 120;
        const startX = (config.width - (spacing * 2)) / 2;
        const topY = 60;
        
        const buyPolygon = this.scene.add.polygon(startX, topY, [
            -20, -18, 20, -18, 20, 18, -20, 18
        ], 0x00ff00);
        buyPolygon.setOrigin(0.5, 0.5);
        window.buyPolygon = buyPolygon;
        
        const sellPolygon = this.scene.add.polygon(startX + spacing, topY, [
            -20, -18, 20, -18, 20, 18, -20, 18
        ], 0xff0000);
        sellPolygon.setOrigin(0.5, 0.5);
        window.sellPolygon = sellPolygon;
        
        const waitPolygon = this.scene.add.polygon(startX + spacing * 2, topY, [
            -20, -18, 20, -18, 20, 18, -20, 18
        ], 0xffff00);
        waitPolygon.setOrigin(0.5, 0.5);
        
        // 신호 대기 센터 (화면 중앙)
        const centerX = config.width / 2;
        const centerY = config.height / 2;
        
        const signalWaitCenter = this.scene.add.circle(centerX, centerY, 40, 0x88ccff);
        signalWaitCenter.setStrokeStyle(3, 0xffffff);
        window.signalWaitCenter = signalWaitCenter;
        
        // 신호 대기 센터 하단 슬라이드형 로딩 바 (매수 전 프로세스 진행률)
        // 신호 대기 센터 하단 슬라이드형 로딩 바 (매수 전 프로세스 진행률)
        const processBarWidth = 160;
        const processBarHeight = 8;
        // 한 줄 정렬을 위해 y를 듀얼 프로세스와 동일선으로 이동
        const unifiedY = centerY + 100;
        const processBarBg = this.scene.add.rectangle(centerX, unifiedY, processBarWidth, processBarHeight, 0x222222).setOrigin(0.5, 0.5);
        // 중앙 프로세스: 두 줄(상/하) 모두 좌->우 진행
        const leftX = centerX - processBarWidth / 2;
        const processBarBgTop = this.scene.add.rectangle(centerX, unifiedY - 5, processBarWidth, processBarHeight, 0x222222).setOrigin(0.5, 0.5);
        const processBarBgBottom = this.scene.add.rectangle(centerX, unifiedY + 5, processBarWidth, processBarHeight, 0x222222).setOrigin(0.5, 0.5);
        // 중앙 기준 양방향처럼 보이되, 항상 좌->우 진행: 중심 정렬 후 displayWidth 변경
        const processBarFillTop = this.scene.add.rectangle(centerX, unifiedY - 5, 1, processBarHeight, 0x00ccff).setOrigin(0.5, 0.5);
        const processBarFillBottom = this.scene.add.rectangle(centerX, unifiedY + 5, 1, processBarHeight, 0x00ccff).setOrigin(0.5, 0.5);
        const processBarText = this.scene.add.text(centerX, unifiedY + 16, '매수 전 예상 수익률 - 프로세스 0%', {
            fontSize: '11px',
            fill: '#00ccff',
            backgroundColor: '#000000',
            padding: { x: 4, y: 2 }
        }).setOrigin(0.5, 0.5);
        // 겹침 시각화를 위해 투명도/깊이 조정
        processBarBg.setDepth(1).setAlpha(0.0); // 숨김 (상/하 배경 사용)
        processBarBgTop.setDepth(2).setStrokeStyle(1, 0x00ccff);
        processBarBgBottom.setDepth(2).setStrokeStyle(1, 0x00ccff);
        processBarFillTop.setDepth(5).setAlpha(0.85);
        processBarFillBottom.setDepth(5).setAlpha(0.85);
        processBarText.setDepth(2);

        // 중앙 프로세스 전용 좌/우 경계선 (시각적 가이드)
        const centerProcLeftEdge = this.scene.add.rectangle(centerX - processBarWidth / 2, unifiedY, 2, processBarHeight + 16, 0x00ccff).setOrigin(0.5, 0.5);
        const centerProcRightEdge = this.scene.add.rectangle(centerX + processBarWidth / 2, unifiedY, 2, processBarHeight + 16, 0x00ccff).setOrigin(0.5, 0.5);
        centerProcLeftEdge.setDepth(3).setAlpha(0.9);
        centerProcRightEdge.setDepth(3).setAlpha(0.9);
        
        window.buyProcessBar = {
            bgTop: processBarBgTop,
            bgBottom: processBarBgBottom,
            fillTop: processBarFillTop,
            fillBottom: processBarFillBottom,
            baseWidth: 1,
            text: processBarText,
            width: processBarWidth
        };

        // 신호 대기 센터 프로세스(중앙 바) - 학습 모델 제어 가능 API
        const bar = window.buyProcessBar; // 중앙 듀얼 바를 재사용
        const clamp01 = (v) => Math.max(0, Math.min(1, v));
        const postDecision = (payload) => {
            try {
                fetch('/api/trainer/decision-log', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(Object.assign({ level: 'info', ts: Date.now() }, payload))
                });
            } catch(_) { }
        };
        const composeFiveW1H = (action, trainer, zone, majority, why, how) => ({
            who: 'TrainerModel',
            what: action,
            when: Date.now(),
            where: { zone, x: Math.round(trainer?.circle?.x || 0), y: Math.round(trainer?.circle?.y || 0) },
            why: why || '',
            how: how || ''
        });
        const getTrainer = () => (window.gameInitializer?.aiModels?.find(m => m.isTrainer));
        const getZoneForTrainer = () => {
            const t = getTrainer();
            return window.gameInitializer?.getCurrentZoneName(t?.circle?.x, t?.circle?.y) || '기타영역';
        };
        const isZoneAllowed = () => {
            const z = getZoneForTrainer();
            return z === '매수영역' || z === '매도영역' || z === '신호대기센터';
        };
        const logBlocked = (act) => {
            try {
                const z = getZoneForTrainer();
                postDecision({ action: 'SIGNAL_CENTER_BLOCKED', blockedAction: act, zone: z, level: 'info' });
            } catch(_) { }
        };
        const setDisplay = (progress, mode) => {
            if (!bar) return;
            const width = Math.max(1, bar.width * clamp01(progress));
            if (bar.fillTop) bar.fillTop.displayWidth = width;
            if (bar.fillBottom) bar.fillBottom.displayWidth = width;
            const color = mode === 'sell' ? (progress < 0.5 ? 0xff8800 : (progress < 0.8 ? 0xffbb33 : 0xffdd55))
                                          : (progress < 0.5 ? 0x00ccff : (progress < 0.8 ? 0x00ffaa : 0x66ff33));
            if (bar.fillTop) bar.fillTop.fillColor = color;
            if (bar.fillBottom) bar.fillBottom.fillColor = color;
            if (bar.text) {
                const label = mode === 'sell' ? '매도' : (mode === 'idle' ? '신호 대기' : '매수');
                bar.text.setText(`${label} 프로세스 ${Math.round(clamp01(progress) * 100)}%`);
            }
        };

        const controller = {
            enabled: true,
            takeover: false,
            mode: 'buy',        // 'buy' | 'sell' | 'idle'
            progress: 0,        // 0..1
            velocity: 0,        // per second
            _lastLogAt: 0,
            _lastLogValue: 0,
            followProfitRate: false,
            _lastAutoLogAt: 0,
            _lastAutoProgress: 0,
            _autoThrottleMs: 5000,
            _autoMinDelta: 0.05,
            // 하트비트 로깅 상태
            _lastMaj: '',
            _lastBuyRate: null,
            _lastSellRate: null,
            _lastThreshold: null,
            _heartbeatId: null,
            _heartbeatMs: 5000,
            _syncHeartbeat(){
                if (this._heartbeatId) { clearInterval(this._heartbeatId); this._heartbeatId = null; }
                if (!this.enabled || !this.takeover || !this.followProfitRate) return;
                this._heartbeatId = setInterval(() => {
                    try {
                            const trainer = getTrainer();
                            const zone = window.gameInitializer?.getCurrentZoneName(trainer?.circle?.x, trainer?.circle?.y) || '기타영역';
                            if (!(zone === '매수영역' || zone === '매도영역' || zone === '신호대기센터' || zone === 'BTC시장탐색구역')) return;

                            // 좌측 패널 분봉 수집 + 폴백 로직
                            const domNodes = Array.from(document.querySelectorAll('#timeframe-cards-container .timeframe-card[data-timeframe], .left-panel .timeframe-card[data-timeframe]'));
                            let tfList = domNodes.map(node => node.getAttribute('data-timeframe')).filter(Boolean);
                            if (tfList.length === 0 && window.timeframeCards && Array.isArray(window.timeframeCards.cards)) {
                                tfList = window.timeframeCards.cards.map(c => c.getAttribute && c.getAttribute('data-timeframe')).filter(Boolean);
                            }
                            if (tfList.length === 0 && window.nbCoinStatus && typeof window.nbCoinStatus === 'object') {
                                tfList = Object.keys(window.nbCoinStatus);
                            }
                            if (tfList.length === 0) {
                                tfList = ['minute1','minute3','minute5','minute10','minute15','minute30','minute60','day'];
                            }
                            const seen = new Set();
                            tfList = tfList.filter(tf => { if (!tf || seen.has(tf)) return false; seen.add(tf); return true; });
                            const thr = (typeof this._lastThreshold === 'number' && !isNaN(this._lastThreshold)) ? this._lastThreshold : 0.5;
                            const rate = (this._lastMaj === 'BLUE') ? (typeof this._lastBuyRate === 'number' ? this._lastBuyRate : 0) : (typeof this._lastSellRate === 'number' ? this._lastSellRate : 0);
                            const timeframes = tfList.map(tf => {
                                const held = (window.nbCoinStatus && typeof window.nbCoinStatus[tf] !== 'undefined') ? (window.nbCoinStatus[tf] ? 1 : 0) : 0;
                                const buyable = (this._lastMaj === 'BLUE' && held === 0 && rate >= thr) ? 1 : 0;
                                return { timeframe: tf, held, buyable };
                            });
                            const hasBuyable = timeframes.some(t => t.buyable === 1);
                            const leftHeldCount = timeframes.reduce((acc, t) => acc + (t.held ? 1 : 0), 0);
                            const trainerNbCoins = (window.gameInitializer?.gameData?.nbCoins ?? 0);
                            // 좌측 패널 N/B 코인 수 파싱(로그용)
                            let leftPanelNbCoins = 0;
                            try {
                                const nbCoinEl = document.getElementById('nb-coin-count');
                                if (nbCoinEl) {
                                    const txt = nbCoinEl.textContent || '';
                                    const m = txt.match(/(\d+)/);
                                    if (m) leftPanelNbCoins = parseInt(m[1]);
                                }
                                // fallback: gameData에서 직접 가져오기
                                if (leftPanelNbCoins === 0) {
                                    leftPanelNbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
                                }
                            } catch(_) {}
                            const why = (this._lastMaj === 'ORANGE')
                                ? `majority=ORANGE, sellRate=${(this._lastSellRate??0).toFixed(2)}%, hasBuyable=${hasBuyable} → progress=${Math.round(this.progress*100)}%`
                                : `majority=BLUE, buyRate=${(this._lastBuyRate??0).toFixed(2)}%, threshold=${((this._lastThreshold??0.5)).toFixed(2)}%, hasBuyable=${hasBuyable} → progress=${Math.round(this.progress*100)}%`;
                            postDecision(Object.assign({ action:'SIGNAL_CENTER_AUTO_DECISION', mode:this.mode, progress:this.progress, leftPanelTimeframes: timeframes, leftPanelHasBuyable: hasBuyable, leftPanelHeldCount: leftHeldCount, leftPanelTimeframesCount: timeframes.length, leftPanelNbCoins: leftPanelNbCoins, trainerNbCoins: trainerNbCoins, nbCoins: trainerNbCoins }, composeFiveW1H('SIGNAL_CENTER_AUTO_DECISION', trainer, zone, this._lastMaj || 'BLUE', why, 'learningAdapter')));
                        } catch(_) { }
                    }, this._heartbeatMs);
                },
                setEnabled(v){ this.enabled = !!v; postDecision({ action: 'SIGNAL_CENTER_ENABLED', enabled: this.enabled }); },
                setTakeover(v){ this.takeover = !!v; postDecision({ action: 'SIGNAL_CENTER_TAKEOVER', enabled: this.takeover }); this._syncHeartbeat(); },
                setMode(m){
                    if (this.takeover && !isZoneAllowed()) { logBlocked('setMode'); return; }
                    const newMode = (m === 'sell' ? 'sell' : (m === 'idle' ? 'idle' : 'buy'));
                    if (newMode === this.mode) return; // 변경 없으면 로그 생략
                    this.mode = newMode;
                    setDisplay(this.progress, this.mode);
                    postDecision({ action: 'SIGNAL_CENTER_MODE', mode: this.mode });
                },
                setProgress(p){ if (this.takeover && !isZoneAllowed()) { logBlocked('setProgress'); return; } this.progress = clamp01(Number(p)||0); setDisplay(this.progress, this.mode); postDecision({ action: 'SIGNAL_CENTER_PROGRESS', progress: this.progress, mode: this.mode }); },
                step(dp){ if (this.takeover && !isZoneAllowed()) { logBlocked('step'); return; } const next = this.progress + Number(dp||0); this.setProgress(next); },
                setVelocity(v){ if (this.takeover && !isZoneAllowed()) { logBlocked('setVelocity'); return; } this.velocity = Number(v)||0; postDecision({ action: 'SIGNAL_CENTER_VELOCITY', velocity: this.velocity }); },
                reset(){ if (this.takeover && !isZoneAllowed()) { logBlocked('reset'); return; } this.progress = 0; setDisplay(this.progress, this.mode); postDecision({ action: 'SIGNAL_CENTER_RESET', mode: this.mode }); },
                setFollowFromRates(v){ this.followProfitRate = !!v; postDecision({ action: 'SIGNAL_CENTER_FOLLOW_FROM_RATES', enabled: this.followProfitRate }); this._syncHeartbeat(); },
                applyFromRates(buyRate, sellRate, majority, trainer, threshold){
                    if (!this.enabled || !this.takeover || !this.followProfitRate) return;
                    const zone = window.gameInitializer?.getCurrentZoneName(trainer?.circle?.x, trainer?.circle?.y) || '기타영역';
                    const maj = (majority||'').toUpperCase();
                    // 최근 입력값 스냅샷 저장(구역 무관, 하트비트 로그용)
                    const thrSnapshot = (typeof threshold === 'number' && !isNaN(threshold)) ? threshold : 0.5;
                    this._lastThreshold = thrSnapshot;
                    if (maj.includes('BLUE') && typeof buyRate === 'number') { this._lastMaj = 'BLUE'; this._lastBuyRate = buyRate; }
                    else if (maj.includes('ORANGE') && typeof sellRate === 'number') { this._lastMaj = 'ORANGE'; this._lastSellRate = sellRate; }
                    if (!(zone === '매수영역' || zone === '매도영역' || zone === '신호대기센터')) return;
                    if (maj.includes('BLUE') && typeof buyRate === 'number'){
                        this._lastMaj = 'BLUE';
                        // 좌측 패널 모든 분봉 정보 수집 (보유/매수가능 플래그 포함)
                        const tfNodes = Array.from(document.querySelectorAll('#timeframe-cards-container .timeframe-card[data-timeframe], .left-panel .timeframe-card[data-timeframe]'));
                        const seen = new Set();
                        const thr = (typeof threshold === 'number' && !isNaN(threshold)) ? threshold : 0.5;
                        this._lastThreshold = thr;
                        this._lastBuyRate = buyRate;
                        const timeframes = tfNodes.map(node => node.getAttribute('data-timeframe')).filter(tf => {
                            if (!tf || seen.has(tf)) return false; seen.add(tf); return true;
                        }).map(tf => {
                            const held = (window.nbCoinStatus && typeof window.nbCoinStatus[tf] !== 'undefined') ? (window.nbCoinStatus[tf] ? 1 : 0) : 0;
                            const buyable = (held === 0 && typeof buyRate === 'number' && buyRate >= thr) ? 1 : 0;
                            return { timeframe: tf, held, buyable };
                        });
                        const hasBuyable = timeframes.some(t => t.buyable === 1);
                        const leftHeldCount = timeframes.reduce((acc, t) => acc + (t.held ? 1 : 0), 0);
                        const trainerNbCoins = (window.gameInitializer?.gameData?.nbCoins ?? 0);
                        this.setMode('buy');
                        let progress;
                        if (buyRate <= 0) progress = 0;
                        else if (buyRate < thr) progress = Math.max(0.05, buyRate / thr * 0.7);
                        else progress = Math.min(1, 0.7 + (buyRate - thr) / 5 * 0.3);
                        this.progress = clamp01(progress);
                        setDisplay(this.progress, this.mode);
                        // 로깅은 하트비트 타이머가 담당
                        return;
                    }
                    if (maj.includes('ORANGE') && typeof sellRate === 'number'){
                        this._lastMaj = 'ORANGE';
                        this._lastSellRate = sellRate;
                        this._lastThreshold = (typeof threshold === 'number' && !isNaN(threshold)) ? threshold : 0.5;
                        // 좌측 패널 모든 분봉 정보 수집 (보유/매수가능 플래그 포함)
                        const tfNodes = Array.from(document.querySelectorAll('#timeframe-cards-container .timeframe-card[data-timeframe], .left-panel .timeframe-card[data-timeframe]'));
                        const seen = new Set();
                        const thr = (typeof threshold === 'number' && !isNaN(threshold)) ? threshold : 0.5;
                        const timeframes = tfNodes.map(node => node.getAttribute('data-timeframe')).filter(tf => {
                            if (!tf || seen.has(tf)) return false; seen.add(tf); return true;
                        }).map(tf => {
                            const held = (window.nbCoinStatus && typeof window.nbCoinStatus[tf] !== 'undefined') ? (window.nbCoinStatus[tf] ? 1 : 0) : 0;
                            const buyable = (held === 0 && typeof buyRate === 'number' && buyRate >= thr) ? 1 : 0;
                            return { timeframe: tf, held, buyable };
                        });
                        const hasBuyable = timeframes.some(t => t.buyable === 1);
                        const leftHeldCount = timeframes.reduce((acc, t) => acc + (t.held ? 1 : 0), 0);
                        const trainerNbCoins = (window.gameInitializer?.gameData?.nbCoins ?? 0);
                        this.setMode('sell');
                        // -5%~+5% → 0~1 매핑
                        const normalized = Math.max(0, Math.min(1, (sellRate + 5) / 10));
                        this.progress = normalized;
                        setDisplay(this.progress, this.mode);
                        // 로깅은 하트비트 타이머가 담당
                        return;
                    }
                },
                _tick(dt){
                    if (!this.enabled || !this.velocity) return;
                    if (!this.takeover) return; // 외부 제어 중일 때만 자동 진행
                    this.progress = clamp01(this.progress + this.velocity * dt);
                    setDisplay(this.progress, this.mode);
                    // Tick 로깅(스로틀/변화량 기준)
                    const now = Date.now();
                    const delta = Math.abs(this.progress - (this._lastLogValue || 0));
                    if ((now - this._lastLogAt) > 1000 || delta >= 0.1) {
                        this._lastLogAt = now;
                        this._lastLogValue = this.progress;
                        postDecision({ action: 'SIGNAL_CENTER_PROGRESS_TICK', progress: this.progress, mode: this.mode });
                    }
                }
            };
            window.signalCenterProcess = controller;
            // 초기 표시 및 자동 활성화(새로고침 후 바로 작동)
            try {
                setDisplay(controller.progress, controller.mode);
                controller.setTakeover(true);
                controller.setFollowFromRates(true);
            } catch(_){ }
        
        window.signalCenterProcess = controller;
        
        // ORANGE / BLUE TOTAL 듀얼 프로세스 바 (LEFT 50%, RIGHT 50%) - 제거됨
        /*
        const totalBarHalfWidth = 160;
        const totalBarHeight = 8;
        // 좌측: ORANGE (좌->우)
        const orangeBg = this.scene.add.rectangle(centerX - totalBarHalfWidth / 2, unifiedY, totalBarHalfWidth, totalBarHeight, 0x222222).setOrigin(0.5, 0.5);
        orangeBg.setStrokeStyle(1, 0xffaa00);
        // ORANGE: 중앙 기준 양방향(부호에 따라 선택) → 좌측/우측용 채움 둘 다 생성
        const orangeFillLeft = this.scene.add.rectangle(centerX, unifiedY, 0, totalBarHeight, 0xff8800).setOrigin(1, 0.5);   // 좌측으로 확장
        const orangeFillRight = this.scene.add.rectangle(centerX, unifiedY, 0, totalBarHeight, 0xff8800).setOrigin(0, 0.5);  // 우측으로 확장 (음수일 때 사용)
        const orangeLabel = this.scene.add.text(centerX - totalBarHalfWidth - 8, centerY + 110, 'ORANGE TOTAL →', {
            fontSize: '10px', fill: '#ffaa00', backgroundColor: '#000000', padding: { x: 3, y: 1 }
        }).setOrigin(1, 0.5);
        // 우측: BLUE (우->좌)
        const blueBg = this.scene.add.rectangle(centerX + totalBarHalfWidth / 2, unifiedY, totalBarHalfWidth, totalBarHeight, 0x222222).setOrigin(0.5, 0.5);
        blueBg.setStrokeStyle(1, 0x00aaff);
        // BLUE: 우측 끝 → 좌측으로 채움 (우측 하프 바의 오른쪽 끝에 정렬)
        const blueFill = this.scene.add.rectangle(centerX + totalBarHalfWidth, unifiedY, 0, totalBarHeight, 0x0088ff).setOrigin(1, 0.5);
        // 듀얼 바는 배경/채움 깊이를 낮춰 중앙 바가 위에 보이도록 함
        orangeBg.setDepth(1);
        if (orangeFillLeft) orangeFillLeft.setDepth(1);
        if (orangeFillRight) orangeFillRight.setDepth(1);
        blueBg.setDepth(1); blueFill.setDepth(1);

        // 경계선 가시화: 좌측 끝/중앙/우측 끝 가이드
        const leftEdge = this.scene.add.rectangle(centerX - totalBarHalfWidth, unifiedY, 2, totalBarHeight + 4, 0xffffff).setOrigin(0.5, 0.5);
        const centerEdge = this.scene.add.rectangle(centerX, unifiedY, 2, totalBarHeight + 6, 0xffffff).setOrigin(0.5, 0.5);
        const rightEdge = this.scene.add.rectangle(centerX + totalBarHalfWidth, unifiedY, 2, totalBarHeight + 4, 0xffffff).setOrigin(0.5, 0.5);
        leftEdge.setDepth(3).setAlpha(0.6);
        centerEdge.setDepth(3).setAlpha(0.8);
        rightEdge.setDepth(3).setAlpha(0.6);
        const blueLabel = this.scene.add.text(centerX + totalBarHalfWidth + 8, centerY + 110, '← BLUE TOTAL', {
            fontSize: '10px', fill: '#00aaff', backgroundColor: '#000000', padding: { x: 3, y: 1 }
        }).setOrigin(0, 0.5);
        
        window.orangeProcessBar = { bg: orangeBg, fillLeft: orangeFillLeft, fillRight: orangeFillRight, width: totalBarHalfWidth, label: orangeLabel };
        window.blueProcessBar = { bg: blueBg, fill: blueFill, width: totalBarHalfWidth, label: blueLabel };

        // 중앙 프로세스 라벨: 양방향 표시
        const processCenterLabel = this.scene.add.text(centerX, centerY + 88, '<-> PROCESS <->', {
            fontSize: '10px', fill: '#ffffff', backgroundColor: '#000000', padding: { x: 4, y: 2 }
        }).setOrigin(0.5, 0.5);
        window.dualProcessCenterLabel = processCenterLabel;
        */
        
        // 라벨들 추가
        this.scene.add.text(100, 140, 'N/B 길드', {
            fontSize: '12px',
            fill: '#00ff00'
        }).setOrigin(0.5);
        
        this.scene.add.text(config.width - 100, config.height - 140, 'BTC 시장 탐색 구역', {
            fontSize: '12px',
            fill: '#ffff00'
        }).setOrigin(0.5);
        
        const buyLabel = this.scene.add.text(startX, topY + 30, '매수', {
            fontSize: '10px',
            fill: '#00ff00'
        }).setOrigin(0.5);
        window.buyLabel = buyLabel;
        
        const sellLabel = this.scene.add.text(startX + spacing, topY + 30, '매도', {
            fontSize: '10px',
            fill: '#ff0000'
        }).setOrigin(0.5);
        window.sellLabel = sellLabel;
        
        this.scene.add.text(startX + spacing * 2, topY + 30, '대기', {
            fontSize: '10px',
            fill: '#ffff00'
        }).setOrigin(0.5);
        
        this.scene.add.text(centerX, centerY - 10, '신호 대기', {
            fontSize: '12px',
            fill: '#ffffff',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        
        this.scene.add.text(centerX, centerY + 10, '센터', {
            fontSize: '12px',
            fill: '#ffffff',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        
        // 애니메이션 추가
        this.scene.tweens.add({
            targets: guildPolygon,
            rotation: Math.PI * 2,
            duration: 8000,
            repeat: -1,
            ease: 'Linear'
        });
        
        // BTC 시장 탐색 구역은 정지 상태로 유지 (회전 애니메이션 제거)
        
        this.scene.tweens.add({
            targets: [buyPolygon, sellPolygon, waitPolygon],
            rotation: Math.PI * 2,
            duration: 10000,
            repeat: -1,
            ease: 'Linear'
        });
    }

    // AI 모델들 생성
    createAIModels() {
        if (!this.scene || !this.game) {
            console.error('❌ Scene or game not initialized for AI models creation');
            return;
        }
        
        // 기존 AI 모델들 정리 (중복 생성 방지)
        if (this.aiModels && this.aiModels.length > 0) {
            //console.log(`🧹 기존 AI 모델 ${this.aiModels.length}개 정리 중...`);
            this.aiModels.forEach(model => {
                if (model.circle) model.circle.destroy();
                if (model.name) model.name.destroy();
                if (model.role) model.role.destroy();
            });
            this.aiModels = [];
        }
        
        // 저장된 상태에서 AI 모델 복원 시도
        if (window.gameStateManager && this.scene) {
            const savedState = window.gameStateManager.loadGameState();
            if (savedState && savedState.aiModels && savedState.aiModels.length > 0) {
                try {
                    this.aiModels = window.gameStateManager.convertAiModelsFromStorage(savedState.aiModels, this.scene);
                    
                    // 복원된 모델들에 애니메이션 추가
                    this.aiModels.forEach(model => {
                        this.scene.tweens.add({
                            targets: [model.circle, model.name, model.role],
                            scaleX: 1.1,
                            scaleY: 1.1,
                            duration: 2000 + Math.random() * 1000,
                            yoyo: true,
                            repeat: -1
                        });
                    });
                    
                    //console.log(`🔄 AI 모델들 저장된 상태에서 복원됨 (총 ${this.aiModels.length}개)`);
                    return;
                } catch (error) {
                    console.error('❌ AI 모델 복원 중 오류:', error);
                    // 오류 발생 시 새로 생성하도록 계속 진행
                }
            }
        }
        
        // 저장된 상태가 없으면 새로 생성
        const config = this.game.config;
        const modelColors = [0xff8800, 0x00ff88, 0x8800ff, 0xff0088, 0xffff00];
        const modelNames = ['Explorer-1', 'Explorer-2', 'Explorer-3', 'Explorer-4', 'Trainer'];
        const modelRoles = ['탐색', '탐색', '탐색', '탐색', '트레이너'];
        
        const initialPositions = [
            { x: config.width / 2, y: config.height / 2 },
            { x: config.width / 4, y: config.height / 4 },
            { x: config.width * 3/4, y: config.height / 4 },
            { x: config.width / 4, y: config.height * 3/4 },
            { x: 150, y: 150 }  // 트레이너를 N/B 길드에서 시작
        ];
        
        for (let i = 0; i < 5; i++) {
            const circleRadius = i === 4 ? 20 : 10; // 트레이너는 20, 탐색자는 10
            const fontSize = i === 4 ? '8px' : '6px';
            const roleFontSize = i === 4 ? '6px' : '5px';
            
            const model = {
                circle: this.scene.add.circle(initialPositions[i].x, initialPositions[i].y, circleRadius, modelColors[i]),
                name: this.scene.add.text(initialPositions[i].x, initialPositions[i].y - (i === 4 ? 6 : 4), modelNames[i], {
                    fontSize: fontSize,
                    fill: '#ffffff',
                    fontStyle: 'bold'
                }).setOrigin(0.5),
                role: this.scene.add.text(initialPositions[i].x, initialPositions[i].y + (i === 4 ? 6 : 4), modelRoles[i], {
                    fontSize: roleFontSize,
                    fill: '#ffffff'
                }).setOrigin(0.5),
                // 트레이너인 경우에만 속도 표시 텍스트 추가
                speedDisplay: i === 4 ? this.scene.add.text(initialPositions[i].x, initialPositions[i].y + 20, '속도: 0.2 (실제: 0.0)', {
                    fontSize: '6px',
                    fill: '#8800ff',
                    fontStyle: 'bold',
                    backgroundColor: '#000000',
                    padding: { x: 2, y: 1 }
                }).setOrigin(0.5) : null,
                targetX: initialPositions[i].x,
                targetY: initialPositions[i].y,
                targetAction: i === 4 ? 'N/B 코인 확인' : '',
                isExplorer: i < 4,
                isTrainer: i === 4,
                discoveredCoords: [],
                memoryIndex: 0,
                explorationTimer: 0,
                needsNewDecision: i === 4 ? false : false,
                arrivalLogged: i === 4 ? false : false
            };
            
            model.circle.setOrigin(0.5, 0.5);
            this.aiModels.push(model);
            
            // 크기 변화 애니메이션
            this.scene.tweens.add({
                targets: i === 4 ? [model.circle, model.name, model.role, model.speedDisplay] : [model.circle, model.name, model.role],
                scaleX: 1.1,
                scaleY: 1.1,
                duration: 2000 + Math.random() * 1000,
                yoyo: true,
                repeat: -1
            });
        }
        
        //console.log(`🆕 AI 모델들 새로 생성됨 (탐색자 ${this.aiModels.filter(m => m.isExplorer).length}명, 트레이너 ${this.aiModels.filter(m => m.isTrainer).length}명, 총 ${this.aiModels.length}개)`);
        
        // 탐색자들을 주민 수집 시스템에 등록
        if (window.residentCollectionSystem) {
            const explorers = this.aiModels.filter(m => m.isExplorer);
            window.residentCollectionSystem.registerResidents(explorers);
        }
        
        // 탐색자들을 탐색원 이동 시스템에 등록
        if (window.explorerMovementSystem) {
            const explorers = this.aiModels.filter(m => m.isExplorer);
            window.explorerMovementSystem.registerExplorers(explorers);
        }
    }

    // UI 요소들 생성
    createUIElements() {
        if (!this.scene || !this.game) {
            console.error('❌ Scene or game not initialized for UI elements creation');
            return;
        }
        
        const config = this.game.config;
        
        // 학습 상태 표시
        const learningStatus = this.scene.add.text(config.width / 2, config.height - 30, 'AI 모델 시스템 시작', {
            fontSize: '12px',
            fill: '#00ff00'
        }).setOrigin(0.5);
        
        // 트레이너 활동 대화창
        const trainerDialog = this.scene.add.text(10, config.height - 60, '🎯 트레이너: AI 시스템 시작 중...', {
            fontSize: '10px',
            fill: '#ffff00',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(0, 0.5);
        
        // 트레이너 위치 정보 표시
        const trainerPositionInfo = this.scene.add.text(10, config.height - 80, '📍 위치: (0, 0) | 목표: (0, 0)', {
            fontSize: '10px',
            fill: '#00ffff',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(0, 0.5);
        
        // N/B MIN 코인 개수 표시
        const nbCoinDisplay = this.scene.add.text(config.width - 10, 10, 'N/B MIN 코인: 0개', {
            fontSize: '12px',
            fill: '#ffaa00',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(1, 0);
        
        // N/B 코인 미네랄 누적 수익률 표시
        const nbMineralDisplay = this.scene.add.text(config.width - 10, 30, 'N/B 미네랄: 0.00%', {
            fontSize: '12px',
            fill: '#00ffaa',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(1, 0);
        
        // 매수 전 예상 수익률 표시
        const buyProfitRateDisplay = this.scene.add.text(10, 10, '매수 전 예상 수익률: 0.00%', {
            fontSize: '12px',
            fill: '#00ff88',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(0, 0);
        
        // 매도 전 예상 수익률 표시
        const sellProfitRateDisplay = this.scene.add.text(10, 30, '매도 전 예상 수익률: 0.00%', {
            fontSize: '12px',
            fill: '#ff0088',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(0, 0);
        
        // 매수 전 적중률 표시
        const buyAccuracyDisplay = this.scene.add.text(10, 50, '매수 전 적중률: 0.00%', {
            fontSize: '12px',
            fill: '#88ccff',
            backgroundColor: '#000000',
            padding: { x: 5, y: 2 }
        }).setOrigin(0, 0);
        
        // 연결선을 그리기 위한 그래픽 객체
        const connectionLines = this.scene.add.graphics();
        const guildTrainerConnection = this.scene.add.graphics();
        const marketBuyConnection = this.scene.add.graphics();
        const guildSellZoneConnection = this.scene.add.graphics();
        
        // 연결선 초기 설정
        connectionLines.setDepth(1); // 다른 객체들 위에 그리기
        guildTrainerConnection.setDepth(1);
        marketBuyConnection.setDepth(1);
        guildSellZoneConnection.setDepth(1);
        
        // 전역 변수로 저장
        window.learningStatus = learningStatus;
        window.trainerDialog = trainerDialog;
        window.trainerPositionInfo = trainerPositionInfo;
        window.nbCoinDisplay = nbCoinDisplay;
        window.nbMineralDisplay = nbMineralDisplay;
        window.buyProfitRateDisplay = buyProfitRateDisplay;
        window.sellProfitRateDisplay = sellProfitRateDisplay;
        window.buyAccuracyDisplay = buyAccuracyDisplay;
        
        // 수익률 표시 요소들에 기본값 설정
        if (window.buyProfitRateDisplay) {
            window.buyProfitRateDisplay.setText('매수 전 예상 수익률: 0.00%');
            window.buyProfitRateDisplay.setFill('#00ff88');
        }
        if (window.sellProfitRateDisplay) {
            window.sellProfitRateDisplay.setText('매도 전 예상 수익률: 0.00%');
            window.sellProfitRateDisplay.setFill('#ff0088');
        }
        
        // 기본 수익률 계산 및 설정
        this.initializeDefaultProfitRates();
        
        if (window.buyAccuracyDisplay) {
            window.buyAccuracyDisplay.setVisible(false); // 텍스트 방식 숨김
        }
        window.connectionLines = connectionLines;
        window.guildTrainerConnection = guildTrainerConnection;
        window.marketBuyConnection = marketBuyConnection;
        window.guildSellZoneConnection = guildSellZoneConnection;
        window.signalWaitCenter = signalWaitCenter;
        
        //console.log('🔗 연결선 그래픽 객체 초기화 완료');
        
        // N/B 코인 디스플레이 초기 업데이트
        if (window.residentCollectionSystem && typeof window.residentCollectionSystem.updateNBCoinDisplay === 'function') {
            window.residentCollectionSystem.updateNBCoinDisplay();
        }
        
        // 전역 함수로 연결선 진단 추가
        window.diagnoseConnectionLines = () => {
            if (window.gameInitializer) {
                window.gameInitializer.diagnoseConnectionLines();
            } else {
                //console.log('❌ 게임 초기화기가 없어서 연결선 진단을 할 수 없습니다.');
            }
        };
    }

    // 이벤트 리스너 설정
    setupEventListeners() {
        setTimeout(() => {
            // HTML 초기화 버튼 이벤트 리스너
            const resetButton = document.getElementById('game-reset-button');
            if (resetButton) {
                resetButton.addEventListener('click', () => this.resetGame());
            }
            
            // N/B 코인 +1 버튼 이벤트 리스너
            const nbCoinPlusButton = document.getElementById('nb-coin-plus-button');
            if (nbCoinPlusButton) {
                nbCoinPlusButton.addEventListener('click', () => this.addNBCoin());
            }
            
            // N/B 코인 -1 버튼 이벤트 리스너
            const nbCoinMinusButton = document.getElementById('nb-coin-minus-button');
            if (nbCoinMinusButton) {
                nbCoinMinusButton.addEventListener('click', () => this.removeNBCoin());
            }
            
            // 드랍 아이템 -1 버튼 이벤트 리스너
            const dropItemMinusButton = document.getElementById('drop-item-minus-button');
            if (dropItemMinusButton) {
                dropItemMinusButton.addEventListener('click', () => this.removeDropItem());
            }
            

            
            // 트레이너 이동 속도 +1 버튼 이벤트 리스너
            const trainerSpeedPlusButton = document.getElementById('trainer-speed-plus-button');
            if (trainerSpeedPlusButton) {
                trainerSpeedPlusButton.addEventListener('click', () => this.increaseTrainerSpeed());
            }
            
            // 트레이너 이동 속도 -1 버튼 이벤트 리스너
            const trainerSpeedMinusButton = document.getElementById('trainer-speed-minus-button');
            if (trainerSpeedMinusButton) {
                trainerSpeedMinusButton.addEventListener('click', () => this.decreaseTrainerSpeed());
            }
        }, 1000);
        
        // 페이지 언로드 시 데이터 저장
        window.addEventListener('beforeunload', () => {
            if (window.residentCollectionSystem && window.residentPersistenceManager) {
                window.residentPersistenceManager.saveResidentData(window.residentCollectionSystem);
                //console.log('💾 페이지 언로드 시 주민 데이터 저장 완료');
            }
        });
        
        // 페이지 숨김 시 데이터 저장
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && window.residentCollectionSystem && window.residentPersistenceManager) {
                window.residentPersistenceManager.saveResidentData(window.residentCollectionSystem);
                //console.log('💾 페이지 숨김 시 주민 데이터 저장 완료');
            }
        });
    }

    // 게임 루프 시작
    startGameLoop() {
        if (!this.scene) {
            console.error('❌ Scene not initialized for game loop');
            return;
        }
        
        try {
            this.scene.time.addEvent({
                delay: 100,
                callback: () => this.aiSystemAlgorithm(),
                loop: true
            });
            
            // 자동 저장은 게임 상태 관리자가 담당
            //console.log('💾 자동 저장: 게임 상태 관리자가 담당');
            
            //console.log('✅ Game loop started successfully');
        } catch (error) {
            console.error('❌ Error starting game loop:', error);
        }
    }

    // AI 시스템 알고리즘
    aiSystemAlgorithm() {
        // 실제 트레이딩 데이터 가져오기
        const majorityElement = document.getElementById('majority-zone');
        const currentPriceElement = document.getElementById('trading-current-price');
        
        if (!majorityElement || !currentPriceElement) {
            if (window.learningStatus) {
                window.learningStatus.setText('데이터 로딩 중...');
            }
            return;
        }
        
        const currentMajority = majorityElement.textContent.trim();
        const currentPriceText = currentPriceElement.textContent;
        
        // 주민 수집 시스템 업데이트
        if (window.residentCollectionSystem) {
            window.residentCollectionSystem.update();
        }
        
        // AI 모델들의 독립적인 행동
        this.aiModels.forEach((model, modelIndex) => {
            if (model.isExplorer) {
                this.handleExplorer(model, modelIndex);
            } else if (model.isTrainer) {
                this.handleTrainer(model, currentMajority, currentPriceText);
            }
        });
        
        // 충돌 감지 및 수익률 계산
        this.checkCollisionsAndCalculateProfit();
        
        // 연결선 업데이트
        this.updateConnectionLines();
        
        // 연결선 진단 (30초마다)
        if (Math.floor(Date.now() / 1000) % 30 === 0) {
            this.diagnoseConnectionLines();
        }

        // 신호 대기 센터 프로세스: 수익률 기반 자동 제어 어댑터
        try {
            const trainer = this.aiModels.find(m => m.isTrainer);
            const majorityEl = document.getElementById('majority-zone');
            const majority = majorityEl ? (majorityEl.textContent || '').trim() : '';
            
            // 신호대기센터에서도 수익률 계산 (충돌 없이도 업데이트)
            if (trainer && window.trainerDecisionHandler) {
                const trainerZone = this.getCurrentZoneName(trainer.circle.x, trainer.circle.y);
                
                // BLUE 신호일 때 매수 수익률 계산
                if (majority.includes('BLUE') && typeof window.trainerDecisionHandler.calculateBuyProfitRate === 'function') {
                    const buyRate = window.trainerDecisionHandler.calculateBuyProfitRate(trainer, this.game.config) || 0;
                    if (this.gameData) this.gameData.buyProfitRate = buyRate;
                }
                
                // ORANGE 신호일 때 매도 수익률 계산
                if (majority.includes('ORANGE') && typeof window.trainerDecisionHandler.calculateSellProfitRate === 'function') {
                    const sellRate = window.trainerDecisionHandler.calculateSellProfitRate(trainer, this.game.config);
                    const safeSell = (typeof sellRate === 'number' && !isNaN(sellRate)) ? sellRate : 0;
                    if (this.gameData) this.gameData.sellProfitRate = safeSell;
                }
            }
            
            const buyRate = this.gameData?.buyProfitRate;
            const sellRate = this.gameData?.sellProfitRate;
            const thr = (this.gameData && typeof this.gameData.buyThresholdPercent === 'number') ? this.gameData.buyThresholdPercent : 0.5;
            if (window.signalCenterProcess && typeof window.signalCenterProcess.applyFromRates === 'function') {
                window.signalCenterProcess.applyFromRates(buyRate, sellRate, majority, trainer, thr);
            }
        } catch(_) { }
        
        // 전체 상태 업데이트
        const totalDiscovered = this.aiModels.reduce((sum, model) => sum + model.discoveredCoords.length, 0);
        if (window.learningStatus) {
            const learningText = `AI 시스템 작동 중 - 총 발견 좌표: ${totalDiscovered}`;
            window.learningStatus.setText(learningText);
            
            // 화면 출력 내용을 로그에 저장
            if (window.logManager) {
                window.logManager.addLog(`📺 화면출력(학습상태): ${learningText}`);
            }
        }
    }

    // 탐색자 처리 (새로운 탐색원 이동 시스템 사용)
    handleExplorer(model, modelIndex) {
        // 디버깅: 시스템 상태 확인 (10초마다)
        if (Math.floor(Date.now() / 1000) % 10 === 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔍 탐색자 ${modelIndex} 처리: 탐색원시스템=${!!window.explorerMovementSystem?.isInitialized}, 주민시스템=${!!window.residentCollectionSystem?.isInitialized}`);
            }
        }
        
        // 탐색원 이동 시스템이 활성화되어 있으면 해당 시스템 사용
        if (window.explorerMovementSystem && window.explorerMovementSystem.isInitialized) {
            // 개별 탐색원 업데이트 호출
            window.explorerMovementSystem.updateExplorerByIndex(modelIndex);
            return;
        }
        
        // 탐색원 이동 시스템이 없을 때는 주민 수집 시스템 사용 (fallback)
        if (window.residentCollectionSystem && window.residentCollectionSystem.isInitialized) {
            // 개별 주민 업데이트 호출
            window.residentCollectionSystem.updateResidentByIndex(modelIndex);
            return;
        }
        
        // 모든 시스템이 없을 때는 기본 탐색 로직 사용 (최후 fallback)
        if (Math.floor(Date.now() / 1000) % 10 === 0) {
            if (window.logManager) {
                window.logManager.addLog(`🔍 탐색자 ${modelIndex}: 기본 탐색 로직 사용`);
            }
        }
        
        const modelX = model.circle.x;
        const modelY = model.circle.y;

        // 드랍 아이템이 있으면 탐색 일시 중지하고 가장 가까운 드랍 아이템으로 이동
        const items = (window.getNBCoinItems && typeof window.getNBCoinItems === 'function') ? window.getNBCoinItems() : (window.nbCoinDropSystem ? window.nbCoinDropSystem.nbCoinItems : []);
        let overrideTarget = null;
        if (items && items.length > 0) {
            let minDist = Number.POSITIVE_INFINITY;
            items.forEach(item => {
                if (!item.collected) {
                    const d = Math.sqrt((item.position.x - modelX) ** 2 + (item.position.y - modelY) ** 2);
                    if (d < minDist) {
                        minDist = d;
                        overrideTarget = item;
                    }
                }
            });
        }

        // 목표 결정: 드랍 아이템이 있으면 그쪽으로, 없으면 기존 목표 유지
        let targetX = model.targetX;
        let targetY = model.targetY;
        if (overrideTarget) {
            targetX = overrideTarget.position.x;
            targetY = overrideTarget.position.y;
            model.role.setText(`드랍아이템 수집중`);
        }

        const distanceToTarget = Math.sqrt((targetX - modelX) ** 2 + (targetY - modelY) ** 2);
        
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
            }
            
            // 드랍 아이템 수집 처리
            if (overrideTarget && window.nbCoinDropSystem && typeof window.nbCoinDropSystem.collectNBCoinItem === 'function') {
                window.nbCoinDropSystem.collectNBCoinItem(overrideTarget);
                model.role.setText(`습득 완료`);
                
                // 1초 후 탐색 상태로 복귀
                setTimeout(() => {
                    model.role.setText(`탐색 (${model.discoveredCoords.length}/8)`);
                }, 1000);
            }
            
            // 새로운 목표 설정 (아이템이 없을 때만 랜덤 탐색 갱신)
            if (!overrideTarget) {
                model.targetX = Math.random() * (this.game.config.width - 80) + 40;
                model.targetY = Math.random() * (this.game.config.height - 80) + 40;
                model.role.setText(`탐색 (${model.discoveredCoords.length}/8)`);
            }
        }
        
        // 탐색자 이동
        const dx = targetX - modelX;
        const dy = targetY - modelY;
        
        if (Math.abs(dx) > 1) {
            model.circle.x += dx * 0.05; // 이동 속도
            model.name.x = model.circle.x;
            model.role.x = model.circle.x;
        }
        
        if (Math.abs(dy) > 1) {
            model.circle.y += dy * 0.05; // 이동 속도
            model.name.y = model.circle.y - 6;
            model.role.y = model.circle.y + 6;
        }
    }

    // 구역 감지 시 수익률 계산 (충돌 대신 감지 기반)
    checkCollisionsAndCalculateProfit() {
        const trainer = this.aiModels.find(model => model.isTrainer);
        if (!trainer) return;

        // 트레이너 현재 구역 감지로 전환
        const trainerZone = this.getCurrentZoneName(trainer.circle.x, trainer.circle.y);

        // 감지 대상 구역: N/B 길드, BTC 시장 탐색 구역
        if (trainerZone === 'N/B길드' || trainerZone === 'N/B 길드' || trainerZone === 'BTC시장탐색구역') {
            this.handleCollisionProfitCalculation(trainer, null, trainerZone, trainerZone);
        }
    }
    
    // 감지 시 수익률 계산 처리 (구역별로 분기)
    handleCollisionProfitCalculation(trainer, explorer, trainerZone, explorerZone) {
        if (!window.trainerDecisionHandler) return;

        const nowStr = new Date().toLocaleTimeString();

        // BTC 시장 탐색 구역: 매수 전 예상 수익률만 계산
        if (trainerZone === 'BTC시장탐색구역') {
            if (typeof window.trainerDecisionHandler.calculateBuyProfitRate === 'function') {
                const buyRate = window.trainerDecisionHandler.calculateBuyProfitRate(trainer, this.game.config) || 0;
                if (this.gameData) this.gameData.buyProfitRate = buyRate;

                if (window.trainerDialog && typeof window.trainerDialog.setText === 'function') {
                    const dialogText = `🔔 감지됨! (BTC 시장) 매수 전 예상 수익률: ${buyRate.toFixed(2)}% | 시간: ${nowStr}`;
                    window.trainerDialog.setText(dialogText);
                    if (window.logManager) window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }

                if (window.logManager) {
                    window.logManager.addLog(`🔔 감지됨! 구역(${trainerZone}) - 매수 전 예상 수익률만 계산: ${buyRate.toFixed(2)}%`);
                }
            }
            return;
        }

        // N/B 길드: 매도 전 예상 수익률만 계산
        if (trainerZone === 'N/B길드' || trainerZone === 'N/B 길드') {
            if (typeof window.trainerDecisionHandler.calculateSellProfitRate === 'function') {
                const sellRate = window.trainerDecisionHandler.calculateSellProfitRate(trainer, this.game.config);
                const safeSell = (typeof sellRate === 'number' && !isNaN(sellRate)) ? sellRate : 0;
                if (this.gameData) this.gameData.sellProfitRate = safeSell;

                if (window.trainerDialog && typeof window.trainerDialog.setText === 'function') {
                    const dialogText = `🔔 감지됨! (N/B 길드) 매도 전 예상 수익률: ${safeSell.toFixed(2)}% | 시간: ${nowStr}`;
                    window.trainerDialog.setText(dialogText);
                    if (window.logManager) window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }

                if (window.logManager) {
                    window.logManager.addLog(`🔔 감지됨! 구역(${trainerZone}) - 매도 전 예상 수익률만 계산: ${safeSell.toFixed(2)}%`);
                }
            }
            return;
        }
    }

    // 트레이너 처리 (분리된 모듈 사용)
    handleTrainer(model, currentMajority, currentPriceText) {
        // 최신 신호 읽기
        const majorityEl = document.getElementById('majority-zone');
        const liveMajority = majorityEl ? (majorityEl.textContent || '').trim() : currentMajority;

        // 우선순위 강제 로직: 코인 보유/풀매수 시 신호 대기 센터 유지
        if (window.trainerStateHandler) {
            try {
                window.trainerStateHandler.processDecisionSystem(
                    model,
                    this.game.config,
                    liveMajority,
                    this.gameData.nbCoins,
                    this.gameData.buyProfitRate,
                    this.gameData.sellProfitRate
                );
                window.trainerStateHandler.handleSignalWaiting(
                    model,
                    this.game.config,
                    liveMajority,
                    this.gameData.nbCoins,
                    this.gameData.buyProfitRate,
                    this.gameData.sellProfitRate,
                    window.trainerDialog || null
                );
            } catch(_) { }
        }

        // 트레이너 의사결정 처리 (강제 조건이 신호 대기인 경우에는 덮어쓰지 않음)
        if (window.trainerDecisionHandler && model.targetAction !== '신호 대기') {
            const targetAction = window.trainerDecisionHandler.handleTrainerDecision(
                model, this.game.config, liveMajority, this.gameData.nbCoins,
                this.gameData.buyProfitRate, this.gameData.sellProfitRate,
                (this.game.config.width - 240) / 2, 60, 120
            );
            model.targetAction = targetAction;
        }
        // 진단 로그 (간헐적)
        if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
            window.logManager.addLog(`🧭 현재 신호: ${liveMajority} → 의사결정: ${model.targetAction}`);
        }
        
        // 트레이너 이동 처리 (의사결정 시스템 전용 경로)
        if (window.trainerStateHandler && typeof window.trainerStateHandler.updateTrainerMovement === 'function') {
            window.trainerStateHandler.updateTrainerMovement(model, this.game.config);
        }

        // 트레이너 액션 실행 처리 (매수/매도 실행)
        if (window.trainerStateHandler && typeof window.trainerStateHandler.handleTrainerActions === 'function') {
            const majorityEl2 = document.getElementById('majority-zone');
            const liveMajority2 = majorityEl2 ? (majorityEl2.textContent || '').trim() : currentMajority;
            window.trainerStateHandler.handleTrainerActions(
                model,
                this.game.config,
                liveMajority2,
                this.gameData.nbCoins,
                this.gameData.buyProfitRate,
                this.gameData.sellProfitRate,
                window.trainerDialog || null
            );
        }
        
        // 트레이너 역할 텍스트 업데이트
        model.role.setText(`트레이너 (${model.targetAction})`);
        
        // 트레이너 속도 표시 업데이트
        if (model.speedDisplay && window.trainerStateHandler) {
            const currentSpeed = window.trainerStateHandler.movementSpeed || 0.2;
            
            // 실제 이동 속도 계산 (목표까지의 거리와 설정된 속도 기반)
            const dx = model.targetX - model.circle.x;
            const dy = model.targetY - model.circle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const actualSpeed = distance > 0 ? Math.min(currentSpeed, distance) : 0;
            
            // 속도 표시 텍스트 업데이트
            const speedText = `속도: ${currentSpeed.toFixed(1)} (실제: ${actualSpeed.toFixed(1)})`;
            model.speedDisplay.setText(speedText);
            model.speedDisplay.x = model.circle.x;
            model.speedDisplay.y = model.circle.y + 20;
        }
        
        // 트레이너 대화창 및 UI 업데이트
        if (window.trainerDialogSystem) {
            window.trainerDialogSystem.updateAllUI(model, this.game.config, this);
        }
    }

    // 연결선 업데이트
    updateConnectionLines() {
        const trainer = this.aiModels.find(model => model.isTrainer);
        if (!trainer) {
            if (window.logManager && Math.floor(Date.now() / 1000) % 10 === 0) {
                window.logManager.addLog(`❌ 연결선 업데이트 실패: 트레이너를 찾을 수 없음`);
            }
            return;
        }
        
        // 연결선 초기화
        if (window.connectionLines) {
            window.connectionLines.clear();
        }
        if (window.guildTrainerConnection) {
            window.guildTrainerConnection.clear();
        }
        if (window.marketBuyConnection) {
            window.marketBuyConnection.clear();
        }
        if (window.guildSellZoneConnection) {
            window.guildSellZoneConnection.clear();
        }
        
        // N/B 길드와 트레이너 연결선 (항시 표시)
        const connectionColor = this.gameData.nbCoins > 0 ? 0xffaa00 : 0x00ff00;
        const connectionAlpha = this.gameData.nbCoins > 0 ? 0.8 : 0.6;
        
        if (window.guildTrainerConnection) {
            window.guildTrainerConnection.lineStyle(2, connectionColor, connectionAlpha);
            window.guildTrainerConnection.beginPath();
            window.guildTrainerConnection.moveTo(100, 100);
            window.guildTrainerConnection.lineTo(trainer.circle.x, trainer.circle.y);
            window.guildTrainerConnection.strokePath();
        }
        
        // BLUE: 트레이너가 매수 구역에 있을 때 BTC 시장 ↔ 매수 구역 연결선 표시
        const trainerZone = this.getCurrentZoneName(trainer.circle.x, trainer.circle.y);
        const majorityEl = document.getElementById('majority-zone');
        const majorityVal = majorityEl ? (majorityEl.textContent || '').trim().toUpperCase() : '';
        const isBlue = majorityVal.includes('BLUE');
        const isOrange = majorityVal.includes('ORANGE');
        const config = this.game.config;
        const startX = (config.width - (120 * 2)) / 2;
        const topY = 60;

        if (isBlue && trainerZone === '매수영역' && window.marketBuyConnection) {
            window.marketBuyConnection.lineStyle(2, 0xffff00, 0.9);
            window.marketBuyConnection.beginPath();
            window.marketBuyConnection.moveTo(config.width - 100, config.height - 100); // BTC 시장
            window.marketBuyConnection.lineTo(startX, topY); // 매수 구역
            window.marketBuyConnection.strokePath();

            // 연결선이 활성화된 동안 매수 전 예상 수익률을 항시 계산
            if (window.trainerDecisionHandler && typeof window.trainerDecisionHandler.calculateBuyProfitRate === 'function') {
                const newBuyRate = window.trainerDecisionHandler.calculateBuyProfitRate(trainer, this.game.config) || 0;
                if (this.gameData) this.gameData.buyProfitRate = newBuyRate;
                this.updateBuyAccuracy(newBuyRate);
                this.updateBuyProcessBar(newBuyRate);
            }
        }

        // ORANGE: 트레이너가 매도 구역에 있을 때 N/B 길드 ↔ 매도 구역 연결선 표시
        if (isOrange && trainerZone === '매도영역' && window.guildSellZoneConnection) {
            window.guildSellZoneConnection.lineStyle(2, 0xff8800, 0.9);
            window.guildSellZoneConnection.beginPath();
            window.guildSellZoneConnection.moveTo(100, 100); // N/B 길드
            window.guildSellZoneConnection.lineTo(startX + 120, topY); // 매도 구역 (startX + spacing)
            window.guildSellZoneConnection.strokePath();

            // 연결선이 활성화된 동안 매도 전 예상 수익률을 항시 계산
            if (window.trainerDecisionHandler && typeof window.trainerDecisionHandler.calculateSellProfitRate === 'function') {
                const newSellRate = window.trainerDecisionHandler.calculateSellProfitRate(trainer, this.game.config);
                const safeSell = (typeof newSellRate === 'number' && !isNaN(newSellRate)) ? newSellRate : 0;
                if (this.gameData) this.gameData.sellProfitRate = safeSell;
                this.updateSellProcessBar(safeSell);
            }
        }

        // 각 탐색자와 트레이너를 연결
        let explorerCount = 0;
        this.aiModels.forEach((model) => {
            if (model.isExplorer && model.circle) {
                if (window.connectionLines) {
                    window.connectionLines.lineStyle(1, 0x00ff88, 0.4);
                    window.connectionLines.beginPath();
                    window.connectionLines.moveTo(trainer.circle.x, trainer.circle.y);
                    window.connectionLines.lineTo(model.circle.x, model.circle.y);
                    window.connectionLines.strokePath();
                    
                    // 연결선 중간에 작은 원 그리기 (데이터 전송 표시)
                    const midX = (trainer.circle.x + model.circle.x) / 2;
                    const midY = (trainer.circle.y + model.circle.y) / 2;
                    window.connectionLines.fillStyle(0x00ff88, 0.6);
                    window.connectionLines.fillCircle(midX, midY, 2);
                }
                explorerCount++;
            }
        });
        
        // 연결선 상태 로그 (10초마다)
        if (window.logManager && Math.floor(Date.now() / 1000) % 10 === 0) {
            window.logManager.addLog(`🔗 연결선 업데이트: 트레이너-${explorerCount}개 탐색원 연결됨`);
        }
    }

    // 연결선 진단
    diagnoseConnectionLines() {
        if (window.logManager) {
            window.logManager.addLog(`🔍 연결선 진단 시작`);
        }
        
        // 1. 그래픽 객체 존재 확인
        const hasConnectionLines = !!window.connectionLines;
        const hasGuildTrainerConnection = !!window.guildTrainerConnection;
        
        if (window.logManager) {
            window.logManager.addLog(`🔍 연결선 그래픽 객체: connectionLines=${hasConnectionLines}, guildTrainerConnection=${hasGuildTrainerConnection}`);
        }
        
        // 2. AI 모델 상태 확인
        const trainer = this.aiModels.find(model => model.isTrainer);
        const explorers = this.aiModels.filter(model => model.isExplorer);
        
        if (window.logManager) {
            window.logManager.addLog(`🔍 AI 모델 상태: 트레이너=${!!trainer}, 탐색원=${explorers.length}개`);
        }
        
        // 3. 탐색원 위치 정보 확인
        explorers.forEach((explorer, index) => {
            const hasCircle = !!explorer.circle;
            const hasValidPosition = hasCircle && typeof explorer.circle.x === 'number' && typeof explorer.circle.y === 'number';
            
            if (window.logManager) {
                window.logManager.addLog(`🔍 탐색원${index}: circle=${hasCircle}, 위치유효=${hasValidPosition}`);
            }
        });
        
        // 4. 트레이너 위치 정보 확인
        if (trainer) {
            const hasCircle = !!trainer.circle;
            const hasValidPosition = hasCircle && typeof trainer.circle.x === 'number' && typeof trainer.circle.y === 'number';
            
            if (window.logManager) {
                window.logManager.addLog(`🔍 트레이너: circle=${hasCircle}, 위치유효=${hasValidPosition}`);
            }
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🔍 연결선 진단 완료`);
        }
    }

    // 매수 전 적중률 업데이트 (지수이동평균 기반)
    updateBuyAccuracy(latestBuyProfitRate) {
        if (!this.gameData) return;
        if (typeof this.gameData.buyAccuracy !== 'number') {
            this.gameData.buyAccuracy = 0; // 0~100 (%)
            this.gameData.buyAccuracyInitialized = false;
        }
        // 유효한 수익률만 반영
        if (typeof latestBuyProfitRate !== 'number' || isNaN(latestBuyProfitRate)) return;

        // 동적 임계치 계산: 가격 상승 시 임계치 하향, 하락 시 임계치 상향
        const priceChangeEl = document.getElementById('right-trading-price-change');
        const priceChangeText = priceChangeEl ? priceChangeEl.textContent || '' : '';
        const match = priceChangeText.match(/-?[\d.]+/);
        const change = match ? parseFloat(match[0]) : 0; // % 값
        // 기준 임계치 0.5%에서 시작
        // 상승(+): 임계치 = base - change * 0.5 (예: +1% → 0.5 - 0.5 = 0.0 → 최소값 적용)
        // 하락(-): 임계치 = base + |change| * 0.5 (예: -1% → 0.5 + 0.5 = 1.0)
        const base = 0.5;
        let dynamicThreshold = change >= 0 ? (base - change * 0.5) : (base + (-change) * 0.5);
        // 하한만 적용 (0.1% 이상), 상한 없음 → 가격 하락 지속 시 임계치도 무제한 증가
        dynamicThreshold = Math.max(0.1, dynamicThreshold);

        // 학습기 예측 적용 (있으면 동적 임계치와 병합)
        let finalThreshold = dynamicThreshold;
        if (this.buyThresholdLearner && this.buyThresholdLearner.enabled) {
            const context = { nbCoins: this.gameData.nbCoins };
            const p = this.buyThresholdLearner.predict(context); // 0..1
            const learned = this.buyThresholdLearner.mapProbToThreshold(p);
            // 더 엄격한 기준을 택함 (max)
            finalThreshold = Math.max(dynamicThreshold, learned);
        }
        this.gameData.buyThresholdPercent = finalThreshold; // 전역 저장

        // 적중 판단: 최신 수익률이 임계치 이상이면 100, 아니면 0
        const hit = latestBuyProfitRate >= dynamicThreshold ? 100 : 0;
        const alpha = 0.1; // EMA 계수

        if (!this.gameData.buyAccuracyInitialized) {
            this.gameData.buyAccuracy = hit;
            this.gameData.buyAccuracyInitialized = true;
        } else {
            this.gameData.buyAccuracy = (1 - alpha) * this.gameData.buyAccuracy + alpha * hit;
        }

        // 화면 표시
        if (window.buyAccuracyDisplay) {
            window.buyAccuracyDisplay.setText(`매수 전 적중률: ${this.gameData.buyAccuracy.toFixed(2)}%`);
        }

        // 10초마다 로그
        if (window.logManager && Math.floor(Date.now() / 1000) % 10 === 0) {
            window.logManager.addLog(`🎯 매수 전 적중률(EMA): ${this.gameData.buyAccuracy.toFixed(2)}% | 임계치: ${dynamicThreshold.toFixed(2)}%`);
        }
    }

    // 매수 전 프로세스 바 업데이트
    updateBuyProcessBar(latestBuyProfitRate) {
        if (!window.buyProcessBar) return;
        const bar = window.buyProcessBar;
        const rate = (typeof latestBuyProfitRate === 'number' && !isNaN(latestBuyProfitRate)) ? latestBuyProfitRate : 0;
        const threshold = (this.gameData && typeof this.gameData.buyThresholdPercent === 'number') ? this.gameData.buyThresholdPercent : 0.5;
        // 0~threshold%를 0~70%, threshold%~(threshold+5)%를 70%~100%로 맵핑
        let progress;
        if (rate <= 0) progress = 0;
        else if (rate < threshold) progress = Math.max(0.05, rate / threshold * 0.7);
        else progress = Math.min(1, 0.7 + (rate - threshold) / 5 * 0.3);
        // ORANGE TOTAL 영향 반영: 매수 프로세스는 ORANGE TOTAL 비중에 비례
        try {
            const orangeText = (document.getElementById('orange-sum')?.textContent || '0').replace(/[^0-9]/g,'');
            const blueText = (document.getElementById('blue-sum')?.textContent || '0').replace(/[^0-9]/g,'');
            const orangeVal = Math.max(0, parseInt(orangeText || '0', 10) || 0);
            const blueVal = Math.max(0, parseInt(blueText || '0', 10) || 0);
            const total = Math.max(1, orangeVal + blueVal);
            const orangeRatio = orangeVal / total; // 0..1
            progress = Math.max(0, Math.min(1, progress * orangeRatio));
        } catch(_) { /* ignore */ }
        // 중앙 기준 좌/우 50%씩 분배
        const half = Math.min(1, progress);
        // 두 줄 모두 좌->우 진행: leftX에서 width 증가
        if (bar.fillTop) bar.fillTop.displayWidth = Math.max(1, bar.width * half);
        if (bar.fillBottom) bar.fillBottom.displayWidth = Math.max(1, bar.width * half);
        bar.text.setText(`매수 전 예상 수익률 - 프로세스 ${Math.round(progress * 100)}% (임계치 ${threshold.toFixed(2)}%)`);
        // 컬러 변화: 저→고 (파랑→청록→라임)
        const color = progress < 0.5 ? 0x00ccff : (progress < 0.8 ? 0x00ffaa : 0x66ff33);
        if (bar.fillTop) bar.fillTop.fillColor = color;
        if (bar.fillBottom) bar.fillBottom.fillColor = color;

        // 100% 도달 시 N/B 드랍 아이템 1개 드랍 (중복 드랍 방지 쿨다운)
        if (progress >= 1) {
            const now = Date.now();
            const cooldownMs = 2000; // 2초 쿨다운
            if (!this.gameData.lastBuyProcessDropAt || (now - this.gameData.lastBuyProcessDropAt) > cooldownMs) {
                this.gameData.lastBuyProcessDropAt = now;
                const trainer = this.aiModels.find(m => m.isTrainer);
                if (trainer && window.nbCoinDropSystem && typeof window.nbCoinDropSystem.dropNBCoin === 'function') {
                    // 무작위 위치로 드랍 (화면 가장자리 20px 여백)
                    const gw = this.game?.config?.width || 1200;
                    const gh = this.game?.config?.height || 700;
                    const rx = 20 + Math.random() * Math.max(0, gw - 40);
                    const ry = 20 + Math.random() * Math.max(0, gh - 40);
                    window.nbCoinDropSystem.dropNBCoin(rx, ry);
                    if (window.logManager) {
                        window.logManager.addLog(`🪙 매수 프로세스 100% 달성 → N/B 드랍 아이템 1개 생성 (랜덤 위치: ${Math.round(rx)}, ${Math.round(ry)})`);
                    }
                }

                // 트레이너를 N/B 길드로 복귀시키고 매도 수익률 계산 유도
                if (trainer) {
                    trainer.targetAction = 'N/B 길드 방문';
                    trainer.targetX = 100;
                    trainer.targetY = 100;
                    if (window.logManager) {
                        window.logManager.addLog(`🏁 프로세스 완료 → 트레이너 목표: N/B 길드 복귀 (100, 100)`);
                    }
                }
            }
        }
    }

    // 매도 전 프로세스 바 업데이트 (확장 대비용)
    updateSellProcessBar(latestSellProfitRate) {
        if (!window.buyProcessBar) return;
        const bar = window.buyProcessBar;
        const rate = (typeof latestSellProfitRate === 'number' && !isNaN(latestSellProfitRate)) ? latestSellProfitRate : 0;
        // 매도 쪽도 동일한 호흡형 진행도 규칙 사용 (임계치와 무관하게 시각화만)
        const normalized = Math.max(0, Math.min(1, (rate + 5) / 10)); // -5%~+5% → 0~1 매핑
        let progress = normalized; 
        // BLUE TOTAL 영향 반영: 매도 프로세스는 BLUE TOTAL 비중에 비례
        try {
            const orangeText = (document.getElementById('orange-sum')?.textContent || '0').replace(/[^0-9]/g,'');
            const blueText = (document.getElementById('blue-sum')?.textContent || '0').replace(/[^0-9]/g,'');
            const orangeVal = Math.max(0, parseInt(orangeText || '0', 10) || 0);
            const blueVal = Math.max(0, parseInt(blueText || '0', 10) || 0);
            const total = Math.max(1, orangeVal + blueVal);
            const blueRatio = blueVal / total; // 0..1
            progress = Math.max(0, Math.min(1, progress * blueRatio));
        } catch(_) { /* ignore */ }
        const half = Math.min(1, progress);
        if (bar.fillTop) bar.fillTop.displayWidth = Math.max(1, bar.width * half);
        if (bar.fillBottom) bar.fillBottom.displayWidth = Math.max(1, bar.width * half);
        const color = progress < 0.5 ? 0xff8800 : (progress < 0.8 ? 0xffbb33 : 0xffdd55);
        if (bar.fillTop) bar.fillTop.fillColor = color;
        if (bar.fillBottom) bar.fillBottom.fillColor = color;
        if (bar.text) {
            bar.text.setText(`매도 전 예상 수익률 - 프로세스 ${Math.round(progress * 100)}%`);
        }
    }

    // 게임 업데이트
    updateGame() {
        // 매 프레임 트레이너 원 색상을 majority-zone과 동기화하여 외부 변경을 무효화
        this.enforceTrainerColorSync();

        // ORANGE/BLUE TOTAL 듀얼 프로세스 바 업데이트 제거됨
        /*
        if (window.dualProcessBarUpdater) {
            console.log('🎮 updateGame에서 dualProcessBarUpdater 호출 시도');
            
            // 디버깅: 프로세스바 업데이트 호출 확인
            if (!window._lastUpdateGameLog || (Date.now() - window._lastUpdateGameLog) > 2000) {
                console.log('🎮 updateGame 호출:', {
                    dualProcessBarUpdater: !!window.dualProcessBarUpdater,
                    orangeProcessBar: !!window.orangeProcessBar,
                    blueProcessBar: !!window.blueProcessBar,
                    signalCenterProcess: !!window.signalCenterProcess
                });
                window._lastUpdateGameLog = Date.now();
            }
            
            try {
                window.dualProcessBarUpdater.update(window.orangeProcessBar, window.blueProcessBar, {
                    orangeId: 'orange-sum',
                    blueId: 'blue-sum'
                });
                console.log('✅ dualProcessBarUpdater.update 호출 성공');
            } catch (error) {
                console.error('❌ dualProcessBarUpdater.update 호출 실패:', error);
            }
        } else {
            console.warn('⚠️ dualProcessBarUpdater가 없습니다');
        }
        */
    }

    // 게임 리셋
    resetGame() {
        //console.log('🔄 게임 완전 초기화 시작 - N/B 길드 위치로 리셋');
        
        // 게임 데이터 초기화 (N/B 코인과 미네랄은 보존)
        const preservedNbCoins = this.gameData?.nbCoins || 0;
        const preservedNbMineralsSum = this.gameData?.nbMineralsSum || 0.0;
        const preservedNbMineralsCount = this.gameData?.nbMineralsCount || 0;
        const preservedNbMinerals = (preservedNbMineralsCount > 0)
            ? (preservedNbMineralsSum / preservedNbMineralsCount)
            : (this.gameData?.nbMinerals || 0.0);
        const preservedDropItemsCollected = this.gameData?.dropItemsCollected || 0;
        
        this.gameData = {
            nbCoins: preservedNbCoins, // N/B 코인 보존
            nbMinerals: preservedNbMinerals, // N/B 미네랄 평균 보존
            nbMineralsSum: preservedNbMineralsSum, // 합계 보존
            nbMineralsCount: preservedNbMineralsCount, // 개수 보존
            dropItemsCount: 0, // 드랍 아이템 개수 0으로 리셋 (새로고침 시)
            dropItemsCollected: preservedDropItemsCollected, // 누적 수집 개수 보존
            buyPrice: 0,
            buyProfitRate: 0,
            sellProfitRate: 0,
            lastBuyAction: false,
            lastSellAction: false
        };
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 게임 리셋: N/B 코인 ${preservedNbCoins}개, 미네랄 ${preservedNbMinerals.toFixed(2)}% 보존`);
        }
        
        // AI 모델들을 모두 N/B 길드 위치 (100, 100)로 리셋
        if (this.aiModels) {
            this.aiModels.forEach((model, index) => {
                if (model.circle) {
                    model.circle.x = 100;
                    model.circle.y = 100;
                }
                if (model.name) {
                    model.name.x = 100;
                    model.name.y = 100;
                }
                if (model.role) {
                    model.role.x = 100;
                    model.role.y = 100;
                }
                model.targetX = 100;
                model.targetY = 100;
                model.discoveredCoords = [];
                model.collectedCoins = 0;
                model.isCarryingCoin = false;
                model.deliveryTarget = null;
                model.state = 'exploring';
                model.collectionTimer = 0;
                model.deliveryTimer = 0;
            });
        }
        
        // N/B 코인 드롭 시스템 리셋
        if (window.nbCoinDropSystem) {
            window.nbCoinDropSystem.clearNBCoinItems();
        }
        
        // 주민 수집 시스템 리셋
        if (window.residentCollectionSystem) {
            window.residentCollectionSystem.reset();
        }
        
        // N/B 코인 디스플레이 업데이트
        if (window.nbCoinDropSystem) {
            window.nbCoinDropSystem.updateNBCoinDisplay();
        }
        
        // 게임 데이터 저장
        this.saveGameData();
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 게임 리셋 완료 - 모든 AI 모델 N/B 길드 위치로 이동`);
        }
    }

    // 완전 초기화 (N/B 코인과 좌측 패널 모두 0으로 리셋)
    completeReset() {
        console.log('🔄 완전 초기화 시작 - N/B 코인과 좌측 패널 모두 0으로 리셋');
        
        // 게임 데이터 완전 초기화 (N/B 코인도 0으로)
        this.gameData = {
            nbCoins: 0, // N/B 코인 0으로 리셋
            nbMinerals: 0.0, // N/B 미네랄 0으로 리셋
            nbMineralsSum: 0.0, // 합계 0으로 리셋
            nbMineralsCount: 0, // 개수 0으로 리셋
            dropItemsCount: 0, // 드랍 아이템 개수 0으로 리셋
            dropItemsCollected: 0, // 누적 수집 개수 0으로 리셋
            buyPrice: 0,
            buyProfitRate: 0,
            sellProfitRate: 0,
            lastBuyAction: false,
            lastSellAction: false
        };
        
        // 좌측 패널 N/B 코인 표시 0으로 리셋
        this.resetLeftPanelNBCoins();
        
        // AI 모델들을 모두 N/B 길드 위치 (100, 100)로 리셋
        if (this.aiModels) {
            this.aiModels.forEach((model, index) => {
                if (model.circle) {
                    model.circle.x = 100;
                    model.circle.y = 100;
                }
                if (model.name) {
                    model.name.x = 100;
                    model.name.y = 100;
                }
                if (model.role) {
                    model.role.x = 100;
                    model.role.y = 100;
                }
                model.targetX = 100;
                model.targetY = 100;
                model.discoveredCoords = [];
                model.collectedCoins = 0;
                model.isCarryingCoin = false;
                model.deliveryTarget = null;
                model.state = 'exploring';
                model.collectionTimer = 0;
                model.deliveryTimer = 0;
            });
        }
        
        // N/B 코인 드롭 시스템 완전 리셋
        if (window.nbCoinDropSystem) {
            window.nbCoinDropSystem.clearNBCoinItems();
        }
        
        // 주민 수집 시스템 리셋
        if (window.residentCollectionSystem) {
            window.residentCollectionSystem.reset();
        }
        
        // N/B 코인 디스플레이 업데이트
        if (window.nbCoinDropSystem) {
            window.nbCoinDropSystem.updateNBCoinDisplay();
        }
        
        // localStorage에서 모든 관련 데이터 삭제
        try {
            const keysToRemove = [
                'paperTradingData',
                'nbCoinData',
                'leftPanelTradeState',
                'left_panel_trade_logger_v1',
                'aiGameState',
                'gameState',
                'selectedCoinStatus',
                'tradeSystemState_v1',
                'historicalData',
                'auto_trade_toggle_states'
            ];
            
            keysToRemove.forEach(key => {
                localStorage.removeItem(key);
                console.log(`🗑️ localStorage에서 ${key} 삭제 완료`);
            });
            
            // 추가로 N/B 코인 관련 키들도 삭제
            Object.keys(localStorage).forEach(key => {
                if (key.includes('nb') || key.includes('coin') || key.includes('trade')) {
                    localStorage.removeItem(key);
                    console.log(`🗑️ localStorage에서 ${key} 삭제 완료`);
                }
            });
            
            console.log('🗑️ localStorage에서 모든 N/B 코인 관련 데이터 삭제 완료');
        } catch (e) {
            console.error('❌ localStorage 데이터 삭제 오류:', e);
        }
        
        // 게임 상태 매니저 리셋
        if (window.gameStateManager) {
            window.gameStateManager.resetGameState();
        }
        
        // 게임 데이터 저장
        this.saveGameData();
        
        // UI 업데이트 강제 실행
        setTimeout(() => {
            // N/B 코인 디스플레이 다시 업데이트
            if (window.nbCoinDropSystem && window.nbCoinDropSystem.updateNBCoinDisplay) {
                window.nbCoinDropSystem.updateNBCoinDisplay();
            }
            
            // 좌측 패널 강제 리셋
            this.resetLeftPanelNBCoins();
            
            console.log('✅ UI 업데이트 완료');
        }, 100);
        
        if (window.logManager) {
            window.logManager.addLog(`🔄 완전 초기화 완료 - N/B 코인 0개, 좌측 패널 0개, 모든 AI 모델 N/B 길드 위치로 이동`);
        }
        
        console.log('✅ 완전 초기화 완료');
    }

    // 좌측 패널 N/B 코인 표시 리셋
    resetLeftPanelNBCoins() {
        try {
            // 방법 1: N/B 코인 카운터 요소 찾아서 0으로 설정
            const nbCoinElements = [
                document.getElementById('nb-coin-count'),
                document.getElementById('nb-coin-display'),
                document.querySelector('.nb-coin-counter'),
                document.querySelector('[data-nb-coin]')
            ];
            
            nbCoinElements.forEach(element => {
                if (element) {
                    const originalText = element.textContent || '';
                    // 숫자를 0으로 교체
                    const newText = originalText.replace(/\d+/, '0');
                    element.textContent = newText;
                    console.log(`📊 좌측 패널 N/B 코인 리셋: "${originalText}" → "${newText}"`);
                }
            });
            
            // 방법 2: 좌측 패널의 timeframe 카드에서 BTC 잔고 0으로 설정
            const tfCards = document.querySelectorAll('.left-panel .timeframe-card');
            tfCards.forEach((card, index) => {
                const coinBalanceElement = card.querySelector('.coin-balance');
                if (coinBalanceElement) {
                    const originalBalance = coinBalanceElement.textContent || '';
                    coinBalanceElement.textContent = '0.00000000';
                    console.log(`📊 카드 ${index + 1} BTC 잔고 리셋: "${originalBalance}" → "0.00000000"`);
                }
                
                // 매도 버튼 비활성화
                const sellButton = card.querySelector('.btn-sell');
                if (sellButton) {
                    sellButton.disabled = true;
                    sellButton.classList.add('disabled');
                    console.log(`📊 카드 ${index + 1} 매도 버튼 비활성화`);
                }
            });
            
            // 방법 3: 전역 N/B 코인 상태 초기화
            if (window.nbCoinStatus) {
                Object.keys(window.nbCoinStatus).forEach(key => {
                    window.nbCoinStatus[key] = 0;
                });
                console.log('📊 전역 N/B 코인 상태 0으로 초기화');
            }
            
            console.log('✅ 좌측 패널 N/B 코인 리셋 완료');
        } catch (e) {
            console.error('❌ 좌측 패널 N/B 코인 리셋 오류:', e);
        }
    }

        // N/B MIN 코인 추가 (N/B MAX COIN과 별개)
        addNBCoin() {
            try {
                // 현재 상태 디버깅
                console.log(`🔍 현재 N/B MIN 코인: ${this.gameData.nbCoins}개`);
                
                // N/B MIN 코인 직접 추가 (cardStorageSystem과 별개)
                this.gameData.nbCoins += 1;
                
                // UI 업데이트
                if (window.nbCoinDisplay) {
                    const coinText = `N/B MIN 코인: ${this.gameData.nbCoins}개`;
                    window.nbCoinDisplay.setText(coinText);
                    console.log(`📺 N/B MIN 코인 UI 업데이트: ${coinText}`);
                }
                
                // 로그 기록
                if (window.logManager) {
                    window.logManager.addLog(`🪙 N/B MIN 코인 +1 추가됨. 현재: ${this.gameData.nbCoins}개`);
                }
                
                // 자동 저장
                this.saveGameData();
                
                console.log(`✅ N/B MIN 코인 1개 추가됨. 총 개수: ${this.gameData.nbCoins}개`);
                
            } catch (error) {
                console.error('N/B MIN 코인 추가 중 오류 발생:', error);
                if (window.logManager) {
                    window.logManager.addLog(`❌ N/B MIN 코인 추가 실패: ${error.message}`, 'error');
                }
            }
        }

        // N/B MIN 코인 제거 (N/B MAX COIN과 별개)
        removeNBCoin() {
            try {
                // 현재 상태 디버깅
                console.log(`🔍 현재 N/B MIN 코인: ${this.gameData.nbCoins}개`);
                
                // N/B MIN 코인이 0보다 큰지 확인
                if (this.gameData.nbCoins > 0) {
                    // N/B MIN 코인 직접 제거 (cardStorageSystem과 별개)
                    this.gameData.nbCoins -= 1;
                    
                    // 좌측 패널의 맨 위 카드에서도 N/B MAX COIN 제거
                    this.removeNBCoinFromTopTimeframeCard();
                    
                    // UI 업데이트
                    if (window.nbCoinDisplay) {
                        const coinText = `N/B MIN 코인: ${this.gameData.nbCoins}개`;
                        window.nbCoinDisplay.setText(coinText);
                        console.log(`📺 N/B MIN 코인 UI 업데이트: ${coinText}`);
                    }
                    
                    // 로그 기록
                    if (window.logManager) {
                        window.logManager.addLog(`🪙 N/B MIN 코인 -1 감소됨. 현재: ${this.gameData.nbCoins}개`);
                    }
                    
                    // 자동 저장
                    this.saveGameData();
                    
                    console.log(`✅ N/B MIN 코인 1개 제거됨. 총 개수: ${this.gameData.nbCoins}개`);
                } else {
                    console.log('⚠️ 제거할 N/B MIN 코인이 없습니다.');
                    if (window.logManager) {
                        window.logManager.addLog('⚠️ 제거할 N/B MIN 코인이 없습니다.', 'warning');
                    }
                }
                
            } catch (error) {
                console.error('N/B MIN 코인 제거 중 오류 발생:', error);
                if (window.logManager) {
                    window.logManager.addLog(`❌ N/B MIN 코인 제거 실패: ${error.message}`, 'error');
                }
            }
        }

        // 좌측 패널의 맨 위 분봉 카드에서 N/B 코인 증가
        addNBCoinToTopTimeframeCard() {
            try {
                // 좌측 패널의 분봉 카드들 찾기
                const leftPanelCards = document.querySelectorAll('.left-panel .card');
                if (leftPanelCards.length === 0) {
                    console.log('⚠️ 좌측 패널 분봉 카드를 찾을 수 없음');
                    return;
                }
                
                // 맨 위 카드 (첫 번째 카드) 선택
                const topCard = leftPanelCards[0];
                const timeframe = topCard.getAttribute('data-timeframe') || '1m'; // 기본값 설정
                
                // 해당 분봉의 현재 N/B MAX COIN 상태 확인
                let currentNbCoins = 0;
                if (window.cardStorageSystem) {
                    const storage = window.cardStorageSystem.getCardStorage(timeframe);
                    currentNbCoins = storage.nbCoins || 0;
                }
                
                // 카드 저장소 시스템에 N/B 코인 추가
                if (window.cardStorageSystem) {
                    const newCount = window.cardStorageSystem.addNBCoin(timeframe, 1);
                    console.log(`✅ 카드 저장소 ${timeframe}에 N/B 코인 +1 추가 → 총 ${newCount}개`);
                } else {
                    console.log('⚠️ cardStorageSystem이 로드되지 않았습니다.');
                }
                
                // N/B 코인 배지 찾기 (여러 방법 시도)
                let nbCoinBadge = null;
                
                // 방법 1: N/B MAX COIN 텍스트가 포함된 배지 찾기
                const allBadges = topCard.querySelectorAll('.badge');
                for (const badge of allBadges) {
                    if (badge.textContent && badge.textContent.includes('N/B MAX COIN')) {
                        nbCoinBadge = badge;
                        break;
                    }
                }
                
                // 방법 2: 성공/실패 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('.badge.bg-success, .badge.bg-secondary');
                }
                
                // 방법 3: 일반 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('[class*="badge"]');
                }
                
                // 방법 4: data-nb-coin 속성 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('[data-nb-coin]');
                }
                
                if (nbCoinBadge) {
                    // 카드 저장소에서 최신 값 가져오기
                    let updatedNbCoins = 0;
                    if (window.cardStorageSystem) {
                        const storage = window.cardStorageSystem.getCardStorage(timeframe);
                        updatedNbCoins = storage.nbCoins;
                    } else {
                        // 폴백: 현재 배지에서 값 읽기
                        const currentText = nbCoinBadge.textContent || '';
                        const match = currentText.match(/(\d+)/);
                        if (match) {
                            updatedNbCoins = parseInt(match[1]);
                        }
                        updatedNbCoins = updatedNbCoins + 1; // 1개씩 증가
                    }
                    
                    // 배지 텍스트와 클래스 업데이트 (안전한 값으로 제한)
                    const safeNbCoins = Math.max(0, updatedNbCoins); // 음수 방지
                    nbCoinBadge.textContent = `N/B MAX COIN: ${safeNbCoins}`;
                    nbCoinBadge.className = safeNbCoins > 0 ? 'badge bg-success' : 'badge bg-secondary';
                    nbCoinBadge.setAttribute('data-nb-coin', safeNbCoins);
                    
                    // 전역 상태 업데이트
                    if (window.nbCoinStatus && timeframe) {
                        window.nbCoinStatus[timeframe] = updatedNbCoins;
                    }
                    
                    console.log(`✅ 좌측 패널 맨 위 카드(${timeframe}) N/B MAX 코인 +1 증가: ${updatedNbCoins}개`);
                } else {
                    // 배지가 없으면 새로 생성
                    const cardBody = topCard.querySelector('.card-body');
                    if (cardBody) {
                        const newBadge = document.createElement('span');
                        newBadge.className = 'badge bg-success';
                        newBadge.textContent = 'N/B MAX COIN: 1';
                        newBadge.setAttribute('data-nb-coin', '1');
                        newBadge.style.marginLeft = '10px';
                        cardBody.appendChild(newBadge);
                        
                        // 전역 상태 업데이트
                        if (window.nbCoinStatus && timeframe) {
                            window.nbCoinStatus[timeframe] = 1;
                        }
                        
                        console.log(`✅ 좌측 패널 맨 위 카드(${timeframe})에 N/B MAX 코인 배지 생성: 1개`);
                    } else {
                        console.log(`⚠️ 좌측 패널 맨 위 카드(${timeframe})에서 card-body를 찾을 수 없음`);
                    }
                }
                
            } catch (error) {
                console.error('❌ 좌측 패널 N/B 코인 증가 중 오류:', error);
            }
        }

        // 좌측 패널의 맨 위 분봉 카드에서 N/B 코인 감소
        removeNBCoinFromTopTimeframeCard() {
            try {
                // 좌측 패널의 분봉 카드들 찾기
                const leftPanelCards = document.querySelectorAll('.left-panel .card');
                if (leftPanelCards.length === 0) {
                    console.log('⚠️ 좌측 패널 분봉 카드를 찾을 수 없음');
                    return;
                }
                
                // 맨 위 카드 (첫 번째 카드) 선택
                const topCard = leftPanelCards[0];
                const timeframe = topCard.getAttribute('data-timeframe') || '1m'; // 기본값 설정
                
                // 해당 분봉의 현재 N/B MAX COIN 상태 확인
                let currentNbCoins = 0;
                if (window.cardStorageSystem) {
                    const storage = window.cardStorageSystem.getCardStorage(timeframe);
                    currentNbCoins = storage.nbCoins || 0;
                }
                
                // N/B MAX COIN이 이미 0이면 제거하지 않음
                if (currentNbCoins <= 0) {
                    console.log(`⚠️ ${timeframe} 분봉의 N/B MAX COIN이 이미 ${currentNbCoins}개입니다. 제거하지 않습니다.`);
                    return;
                }
                
                // 카드 저장소 시스템에서 N/B 코인 제거 (1 이상일 때만)
                if (window.cardStorageSystem) {
                    const newCount = window.cardStorageSystem.removeNBCoin(timeframe, 1);
                    console.log(`✅ 카드 저장소 ${timeframe}에서 N/B 코인 -1 제거 → 총 ${newCount}개`);
                } else {
                    console.log('⚠️ cardStorageSystem이 로드되지 않았습니다.');
                }
                
                // N/B 코인 배지 찾기 (여러 방법 시도)
                let nbCoinBadge = null;
                
                // 방법 1: N/B MAX COIN 텍스트가 포함된 배지 찾기
                const allBadges = topCard.querySelectorAll('.badge');
                for (const badge of allBadges) {
                    if (badge.textContent && badge.textContent.includes('N/B MAX COIN')) {
                        nbCoinBadge = badge;
                        break;
                    }
                }
                
                // 방법 2: 성공/실패 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('.badge.bg-success, .badge.bg-secondary');
                }
                
                // 방법 3: 일반 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('[class*="badge"]');
                }
                
                // 방법 4: data-nb-coin 속성 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = topCard.querySelector('[data-nb-coin]');
                }
                
                if (nbCoinBadge) {
                    // 카드 저장소에서 최신 값 가져오기
                    let updatedNbCoins = 0;
                    if (window.cardStorageSystem) {
                        const storage = window.cardStorageSystem.getCardStorage(timeframe);
                        updatedNbCoins = storage.nbCoins;
                    } else {
                        // 폴백: 현재 배지에서 값 읽기
                        const currentText = nbCoinBadge.textContent || '';
                        const match = currentText.match(/(\d+)/);
                        if (match) {
                            updatedNbCoins = parseInt(match[1]);
                        }
                        updatedNbCoins = Math.max(updatedNbCoins - 1, 0); // 최소 0개까지만
                    }
                    
                    // 배지 텍스트와 클래스 업데이트 (안전한 값으로 제한)
                    const safeNbCoins = Math.max(0, updatedNbCoins); // 음수 방지
                    nbCoinBadge.textContent = `N/B MAX COIN: ${safeNbCoins}`;
                    nbCoinBadge.className = safeNbCoins > 0 ? 'badge bg-success' : 'badge bg-secondary';
                    nbCoinBadge.setAttribute('data-nb-coin', safeNbCoins);
                    
                    // 전역 상태 업데이트
                    if (window.nbCoinStatus && timeframe) {
                        window.nbCoinStatus[timeframe] = updatedNbCoins;
                    }
                    
                    console.log(`✅ 좌측 패널 맨 위 카드(${timeframe}) N/B MAX 코인 -1 감소: ${updatedNbCoins}개`);
                } else {
                    console.log(`⚠️ 좌측 패널 맨 위 카드(${timeframe})에서 N/B MAX 코인 배지를 찾을 수 없음`);
                }
                
                // N/B MAX COIN은 전역 N/B 코인 카운트와 별개이므로 gameData.nbCoins는 수정하지 않음
                // this.gameData.nbCoins는 N/B MIN 코인용이고, N/B MAX COIN은 별도 시스템
                
            } catch (error) {
                console.error('❌ 좌측 패널 N/B 코인 감소 중 오류:', error);
            }
        }

        // 드랍 아이템 제거
        removeDropItem() {
            try {
                // 현재 드랍 아이템 개수 확인
                const currentDropItems = this.gameData?.dropItemsCount || 0;
                
                if (currentDropItems <= 0) {
                    console.log('⚠️ 드랍 아이템이 없습니다.');
                    return;
                }
                
                // 드랍 아이템 개수 감소
                this.gameData.dropItemsCount = Math.max(0, currentDropItems - 1);
                
                // UI 업데이트
                if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.updateNBCoinDisplay === 'function') {
                    window.nbCoinDropSystem.updateNBCoinDisplay();
                }
                
                console.log(`✅ 드랍 아이템 -1 제거됨. 현재: ${this.gameData.dropItemsCount}개`);
                
                // 로그 추가
                if (window.logManager) {
                    window.logManager.addLog(`📦 드랍 아이템 -1 제거됨. 현재: ${this.gameData.dropItemsCount}개`);
                }
                
                // 상태 저장
                this.saveGameData();
            } catch (error) {
                console.error('❌ 드랍 아이템 제거 중 오류:', error);
            }
        }

        // N/B 미네랄 추가


        // 트레이너 이동 속도 증가
        increaseTrainerSpeed() {
            //console.log('🔍 increaseTrainerSpeed() 호출됨');
            
            // 모든 트레이너 관련 모듈의 이동 속도 증가 (최대 100)
            if (window.trainerStateHandler) {
                const oldSpeed = window.trainerStateHandler.movementSpeed;
                window.trainerStateHandler.movementSpeed = Math.min(100, window.trainerStateHandler.movementSpeed + 1);
                //console.log(`⚡ 트레이너 이동 속도 +1 증가됨. ${oldSpeed} → ${window.trainerStateHandler.movementSpeed}`);
                
                // 최대값 도달 시 알림
                if (window.trainerStateHandler.movementSpeed >= 100) {
                    //console.log(`🚨 트레이너 이동 속도 최대값(100)에 도달했습니다!`);
                    if (window.logManager) {
                        window.logManager.addLog(`🚨 트레이너 이동 속도 최대값(100)에 도달했습니다!`);
                    }
                }
            } else {
                console.warn('⚠️ window.trainerStateHandler가 없습니다');
            }
            
            if (window.trainerManager) {
                const oldSpeed = window.trainerManager.movementSpeed;
                window.trainerManager.movementSpeed = Math.min(100, window.trainerManager.movementSpeed + 1);
                //console.log(`⚡ 트레이너 매니저 이동 속도 +1 증가됨. ${oldSpeed} → ${window.trainerManager.movementSpeed}`);
            } else {
                console.warn('⚠️ window.trainerManager가 없습니다');
            }
            
            if (window.trainerMovementController) {
                const oldSpeed = window.trainerMovementController.movementSpeed;
                window.trainerMovementController.movementSpeed = Math.min(100, window.trainerMovementController.movementSpeed + 1);
                //console.log(`⚡ 트레이너 이동 컨트롤러 속도 +1 증가됨. ${oldSpeed} → ${window.trainerMovementController.movementSpeed}`);
            } else {
                console.warn('⚠️ window.trainerMovementController가 없습니다');
            }
            
            // UI 업데이트
            this.updateTrainerSpeedDisplay();
            
            // 로그 기록
            if (window.logManager) {
                const currentSpeed = window.trainerStateHandler?.movementSpeed || 'N/A';
                window.logManager.addLog(`⚡ 트레이너 이동 속도 +1 증가됨. 현재 속도: ${currentSpeed}`);
            }
            
            // 트레이너 이동 속도 자동 저장
            this.saveTrainerSpeed();
        }

        // 트레이너 이동 속도 감소
        decreaseTrainerSpeed() {
            //console.log('🔍 decreaseTrainerSpeed() 호출됨');
            
            // 모든 트레이너 관련 모듈의 이동 속도 감소 (최소 0.1)
            if (window.trainerStateHandler) {
                const oldSpeed = window.trainerStateHandler.movementSpeed;
                window.trainerStateHandler.movementSpeed = Math.max(0.1, window.trainerStateHandler.movementSpeed - 1);
                //console.log(`🐌 트레이너 이동 속도 -1 감소됨. ${oldSpeed} → ${window.trainerStateHandler.movementSpeed}`);
            } else {
                console.warn('⚠️ window.trainerStateHandler가 없습니다');
            }
            
            if (window.trainerManager) {
                const oldSpeed = window.trainerManager.movementSpeed;
                window.trainerManager.movementSpeed = Math.max(0.1, window.trainerManager.movementSpeed - 1);
                //console.log(`🐌 트레이너 매니저 이동 속도 -1 감소됨. ${oldSpeed} → ${window.trainerManager.movementSpeed}`);
            } else {
                console.warn('⚠️ window.trainerManager가 없습니다');
            }
            
            if (window.trainerMovementController) {
                const oldSpeed = window.trainerMovementController.movementSpeed;
                window.trainerMovementController.movementSpeed = Math.max(0.1, window.trainerMovementController.movementSpeed - 1);
                //console.log(`🐌 트레이너 이동 컨트롤러 속도 -1 감소됨. ${oldSpeed} → ${window.trainerMovementController.movementSpeed}`);
            } else {
                console.warn('⚠️ window.trainerMovementController가 없습니다');
            }
            
            // 로그 기록
            if (window.logManager) {
                const currentSpeed = window.trainerStateHandler?.movementSpeed || 'N/A';
                window.logManager.addLog(`🐌 트레이너 이동 속도 -1 감소됨. 현재 속도: ${currentSpeed}`);
            }
            
            // 트레이너 이동 속도 자동 저장
            this.saveTrainerSpeed();
        }

        // 트레이너 이동 속도 저장
        saveTrainerSpeed() {
            try {
                const trainerSpeed = window.trainerStateHandler?.movementSpeed || 0.2;
                localStorage.setItem('trainerMovementSpeed', trainerSpeed.toString());
                //console.log(`💾 트레이너 이동 속도 저장됨: ${trainerSpeed}`);
                
                if (window.logManager) {
                    window.logManager.addLog(`💾 트레이너 이동 속도 자동 저장: ${trainerSpeed}`);
                }
            } catch (error) {
                console.error(`❌ 트레이너 이동 속도 저장 중 오류: ${error.message}`);
            }
        }

        // 트레이너 이동 속도 로드
        loadTrainerSpeed() {
            try {
                const savedSpeed = localStorage.getItem('trainerMovementSpeed');
                if (savedSpeed) {
                    const speed = parseFloat(savedSpeed);
                    if (!isNaN(speed) && speed > 0) {
                        // 모든 트레이너 관련 모듈의 이동 속도 설정
                        if (window.trainerStateHandler) {
                            window.trainerStateHandler.movementSpeed = speed;
                        }
                        if (window.trainerManager) {
                            window.trainerManager.movementSpeed = speed;
                        }
                        if (window.trainerMovementController) {
                            window.trainerMovementController.movementSpeed = speed;
                        }
                        
                        //console.log(`📂 트레이너 이동 속도 로드됨: ${speed}`);
                        
                        if (window.logManager) {
                            window.logManager.addLog(`📂 트레이너 이동 속도 로드됨: ${speed}`);
                        }
                    }
                } else {
                                         // 저장된 속도가 없으면 기본값 0.2 설정
                     const defaultSpeed = 0.2;
                    if (window.trainerStateHandler) {
                        window.trainerStateHandler.movementSpeed = defaultSpeed;
                    }
                    if (window.trainerManager) {
                        window.trainerManager.movementSpeed = defaultSpeed;
                    }
                    if (window.trainerMovementController) {
                        window.trainerMovementController.movementSpeed = defaultSpeed;
                    }
                    
                    //console.log(`📂 트레이너 이동 속도 기본값 설정: ${defaultSpeed}`);
                    
                    if (window.logManager) {
                        window.logManager.addLog(`📂 트레이너 이동 속도 기본값 설정: ${defaultSpeed}`);
                    }
                }
            } catch (error) {
                console.error(`❌ 트레이너 이동 속도 로드 중 오류: ${error.message}`);
            }
        }

    // 현재 위치의 장소 이름 반환
    getCurrentZoneName(x, y) {
        const config = this.game.config;
        const spacing = 120;
        const startX = (config.width - (spacing * 2)) / 2;
        const topY = 60;
        
        // 매수 영역 (감지 범위 확대)
        if (Math.abs(x - startX) < 100 && Math.abs(y - topY) < 100) {
            return '매수영역';
        }
        // 매도 영역
        else if (Math.abs(x - (startX + spacing)) < 100 && Math.abs(y - topY) < 100) {
            return '매도영역';
        }
        // 대기 영역
        else if (Math.abs(x - (startX + spacing * 2)) < 100 && Math.abs(y - topY) < 100) {
            return '대기영역';
        }
        // N/B 길드
        else if (Math.abs(x - 100) < 120 && Math.abs(y - 100) < 120) {
            return 'N/B길드';
        }
        // BTC 시장 탐색 구역
        else if (Math.abs(x - (config.width - 100)) < 120 && Math.abs(y - (config.height - 100)) < 120) {
            return 'BTC시장탐색구역';
        }
        // 신호 대기 센터
        else if (Math.abs(x - config.width / 2) < 100 && Math.abs(y - config.height / 2) < 100) {
            return '신호대기센터';
        }
        
        return '기타영역';
    }

    // 게임 데이터 저장 (game-state-manager.js의 localStorage 사용)
    saveGameData() {
        try {
            // game-state-manager.js를 통해 localStorage에 저장
            if (window.gameStateManager) {
                const gameData = {
                    ...this.gameData,
                    aiModels: this.aiModels || []
                };
                window.gameStateManager.saveGameState(gameData);
                this.lastSaveTime = Date.now();
                
                //console.log(`💾 N/B 코인/미네랄 localStorage 저장: N/B코인 ${this.gameData.nbCoins}개, N/B미네랄 ${this.gameData.nbMinerals.toFixed(2)}%`);
                
                if (window.logManager) {
                    window.logManager.addLog(`💾 N/B 코인/미네랄 localStorage 저장: N/B코인 ${this.gameData.nbCoins}개, N/B미네랄 ${this.gameData.nbMinerals.toFixed(2)}%`);
                }
            }
        } catch (error) {
            console.error(`❌ 게임 데이터 저장 중 오류: ${error.message}`);
        }
    }

    // 트레이너 이동 속도 표시 업데이트
    updateTrainerSpeedDisplay() {
        const speedDisplay = document.getElementById('trainer-speed-display');
        if (speedDisplay) {
            const speed = window.trainerStateHandler?.movementSpeed || 0.2;
            speedDisplay.textContent = speed.toFixed(1);
            //console.log(`📊 이동 속도 표시 업데이트: ${speed.toFixed(1)}`);
        }
        
        // 트레이너 원의 속도 표시 업데이트
        const trainer = this.aiModels.find(model => model.isTrainer);
        if (trainer && trainer.speedDisplay && window.trainerStateHandler) {
            const speed = window.trainerStateHandler.movementSpeed || 0.2;
            
            // 실제 이동 속도 계산 (목표까지의 거리와 설정된 속도 기반)
            const dx = trainer.targetX - trainer.circle.x;
            const dy = trainer.targetY - trainer.circle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const actualSpeed = distance > 0 ? Math.min(speed, distance) : 0;
            
            // 속도 표시 텍스트 업데이트
            const speedText = `속도: ${speed.toFixed(1)} (실제: ${actualSpeed.toFixed(1)})`;
            trainer.speedDisplay.setText(speedText);
            //console.log(`📊 트레이너 원 속도 표시 업데이트: ${speedText}`);
        }
        
        // 기존 위치 정보 업데이트도 유지
        if (window.trainerPositionInfo) {
            const trainer = this.aiModels.find(model => model.isTrainer);
            if (trainer) {
                const speed = window.trainerStateHandler?.movementSpeed || 0.2;
                
                // 구역 정보 가져오기
                const currentZone = this.getCurrentZoneName(trainer.circle.x, trainer.circle.y);
                const targetZone = this.getCurrentZoneName(trainer.targetX, trainer.targetY);
                
                const speedText = `📍 위치: (${Math.round(trainer.circle.x)}, ${Math.round(trainer.circle.y)}) (${currentZone}) | 목표: (${Math.round(trainer.targetX)}, ${Math.round(trainer.targetY)}) (${targetZone}) | 속도: ${speed.toFixed(1)}`;
                window.trainerPositionInfo.setText(speedText);
                if (window.logManager) {
                    window.logManager.addLog(`📺 화면출력(위치정보): ${speedText}`);
                }
            }
        }
    }

    // 트레이너 원 색상을 majority-zone과 동기화 (매 프레임 호출)
    enforceTrainerColorSync() {
        try {
            const majorityEl = document.getElementById('majority-zone');
            const val = majorityEl ? (majorityEl.textContent || '').trim() : '';
            const color = this.getMajorityColor(val);
            const trainer = this.aiModels?.find(m => m.isTrainer);
            if (trainer && trainer.circle && typeof trainer.circle.setFillStyle === 'function') {
                trainer.circle.setFillStyle(color);
            }

            // BLUE일 때 매도 구역 숨김, ORANGE일 때 매수 구역 숨김
            // 항상 표시 (숨김 모드 제거)
            if (window.sellPolygon) window.sellPolygon.setVisible(true);
            if (window.sellLabel) window.sellLabel.setVisible(true);
            if (window.buyPolygon) window.buyPolygon.setVisible(true);
            if (window.buyLabel) window.buyLabel.setVisible(true);
        } catch (e) {
            // ignore
        }
    }

    // majority 문자열을 색상으로 변환
    getMajorityColor(value) {
        const v = (value || '').toUpperCase();
        switch (true) {
            case v.includes('BLUE'): return 0x0088ff;
            case v.includes('ORANGE'): return 0xff8800;
            case v.includes('GREEN'): return 0x00ff88;
            case v.includes('RED'): return 0xff0000;
            default: return 0x00d1ff;
        }
    }

    // 기본 수익률 초기화
    initializeDefaultProfitRates() {
        try {
            // 현재 시장 정보 가져오기
            const currentPriceElement = document.getElementById('right-trading-current-price');
            const currentZoneElement = document.getElementById('right-trading-current-zone');
            const zoneStrengthElement = document.getElementById('right-trading-zone-strength');
            
            let currentPrice = 160000000; // 기본값
            let currentZone = 'BLUE';
            let zoneStrength = 0;
            
            if (currentPriceElement) {
                const priceText = currentPriceElement.textContent;
                const priceMatch = priceText.match(/[\d,]+/);
                if (priceMatch) {
                    currentPrice = parseFloat(priceMatch[0].replace(/,/g, ''));
                }
            }
            
            if (currentZoneElement) {
                const zoneText = currentZoneElement.textContent;
                if (zoneText.includes('ORANGE')) currentZone = 'ORANGE';
                else if (zoneText.includes('BLUE')) currentZone = 'BLUE';
            }
            
            if (zoneStrengthElement) {
                const strengthText = zoneStrengthElement.textContent;
                const strengthMatch = strengthText.match(/-?\d+/);
                if (strengthMatch) {
                    zoneStrength = parseFloat(strengthMatch[0]);
                }
            }
            
            // 기본 수익률 계산
            let buyProfitRate = 0;
            let sellProfitRate = 0;
            
            if (currentZone === 'ORANGE') {
                buyProfitRate = Math.max(0.5, Math.min(3.0, zoneStrength * 0.1));
            } else if (currentZone === 'BLUE') {
                buyProfitRate = Math.max(-2.0, Math.min(1.5, -zoneStrength * 0.05));
            } else {
                buyProfitRate = Math.random() * 2 - 1;
            }
            
            sellProfitRate = -buyProfitRate * 0.5;
            
            // 수익률 표시 업데이트
            if (window.buyProfitRateDisplay) {
                const buyText = `매수 전 예상 수익률: ${buyProfitRate.toFixed(2)}%`;
                window.buyProfitRateDisplay.setText(buyText);
                window.buyProfitRateDisplay.setFill(buyProfitRate >= 0 ? '#00ff88' : '#ff0088');
            }
            
            if (window.sellProfitRateDisplay) {
                const sellText = `매도 전 예상 수익률: ${sellProfitRate.toFixed(2)}%`;
                window.sellProfitRateDisplay.setText(sellText);
                window.sellProfitRateDisplay.setFill(sellProfitRate >= 0 ? '#00ff88' : '#ff0088');
            }
            
            //console.log(`🎯 기본 수익률 초기화 완료 - 매수: ${buyProfitRate.toFixed(2)}%, 매도: ${sellProfitRate.toFixed(2)}%`);
            
        } catch (error) {
            console.warn('기본 수익률 초기화 오류:', error);
        }
    }

    // 강제 완전 초기화 (모든 데이터 삭제 후 페이지 새로고침)
    forceCompleteReset() {
        console.log('🔄 강제 완전 초기화 시작');
        
        try {
            // 1. 모든 localStorage 데이터 삭제
            const allKeys = Object.keys(localStorage);
            allKeys.forEach(key => {
                localStorage.removeItem(key);
                console.log(`🗑️ localStorage에서 ${key} 삭제 완료`);
            });
            
            // 2. 모든 sessionStorage 데이터 삭제
            const allSessionKeys = Object.keys(sessionStorage);
            allSessionKeys.forEach(key => {
                sessionStorage.removeItem(key);
                console.log(`🗑️ sessionStorage에서 ${key} 삭제 완료`);
            });
            
            // 3. 게임 데이터 완전 초기화
            this.gameData = {
                nbCoins: 0,
                nbMinerals: 0.0,
                nbMineralsSum: 0.0,
                nbMineralsCount: 0,
                dropItemsCount: 0,
                dropItemsCollected: 0,
                buyPrice: 0,
                buyProfitRate: 0,
                sellProfitRate: 0,
                lastBuyAction: false,
                lastSellAction: false
            };
            
            // 4. 전역 변수들 초기화
            if (window.nbCoinStatus) {
                Object.keys(window.nbCoinStatus).forEach(key => {
                    window.nbCoinStatus[key] = 0;
                });
            }
            
            if (window.leftPanelTradeState) {
                window.leftPanelTradeState = { lastBuyTs: 0, lastSellTs: 0 };
            }
            
            // 5. AI 모델들 초기화
            if (this.aiModels) {
                this.aiModels.forEach(model => {
                    if (model.circle) {
                        model.circle.x = 100;
                        model.circle.y = 100;
                    }
                    if (model.name) {
                        model.name.x = 100;
                        model.name.y = 100;
                    }
                    if (model.role) {
                        model.role.x = 100;
                        model.role.y = 100;
                    }
                    model.targetX = 100;
                    model.targetY = 100;
                    model.discoveredCoords = [];
                    model.collectedCoins = 0;
                    model.isCarryingCoin = false;
                    model.deliveryTarget = null;
                    model.state = 'exploring';
                    model.collectionTimer = 0;
                    model.deliveryTimer = 0;
                });
            }
            
            // 6. 시스템들 리셋
            if (window.nbCoinDropSystem) {
                window.nbCoinDropSystem.clearNBCoinItems();
            }
            
            if (window.residentCollectionSystem) {
                window.residentCollectionSystem.reset();
            }
            
            if (window.gameStateManager) {
                window.gameStateManager.resetGameState();
            }
            
            // 7. 로그 기록
            if (window.logManager) {
                window.logManager.addLog('🔄 강제 완전 초기화 완료 - 모든 데이터 삭제됨');
            }
            
            console.log('✅ 강제 완전 초기화 완료');
            
            // 8. 즉시 페이지 새로고침
            setTimeout(() => {
                console.log('🔄 페이지 새로고침으로 완전 리셋');
                window.location.reload(true); // 강제 새로고침
            }, 500);
            
        } catch (error) {
            console.error('❌ 강제 초기화 중 오류:', error);
            // 오류 발생 시에도 페이지 새로고침
            setTimeout(() => {
                window.location.reload(true);
            }, 1000);
        }
    }
}

// 전역 인스턴스 생성
window.gameInitializer = new GameInitializer();

