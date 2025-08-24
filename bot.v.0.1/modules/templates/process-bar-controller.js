// 프로세스 바 컨트롤러 (신호 대기 센터 하단 슬라이드)
class ProcessBarController {
    constructor() {
        this.scene = null;
        this.config = null;
    }

    init(scene, config) {
        this.scene = scene;
        this.config = config;

        const centerX = config.width / 2;
        const centerY = config.height / 2;

        const processBarWidth = 160;
        const processBarHeight = 8;
        const processBarBg = scene.add.rectangle(centerX, centerY + 62, processBarWidth, processBarHeight, 0x222222).setOrigin(0.5, 0.5);
        const processBarFill = scene.add.rectangle(centerX - processBarWidth / 2, centerY + 62, 0, processBarHeight, 0x00ccff).setOrigin(0, 0.5);
        const processBarText = scene.add.text(centerX, centerY + 78, '매수 전 예상 수익률 - 프로세스 0%', {
            fontSize: '11px',
            fill: '#00ccff',
            backgroundColor: '#000000',
            padding: { x: 4, y: 2 }
        }).setOrigin(0.5, 0.5);

        window.buyProcessBar = { bg: processBarBg, fill: processBarFill, text: processBarText, width: processBarWidth };
    }

    updateBuy(latestBuyProfitRate, thresholdPercent) {
        if (!window.buyProcessBar || !window.gameInitializer) return;
        const bar = window.buyProcessBar;
        const rate = (typeof latestBuyProfitRate === 'number' && !isNaN(latestBuyProfitRate)) ? latestBuyProfitRate : 0;
        const threshold = (typeof thresholdPercent === 'number' && !isNaN(thresholdPercent)) ? thresholdPercent : 0.5;

        // 0~threshold%를 0~70%, threshold%~(threshold+5)%를 70%~100%로 맵핑
        let progress;
        if (rate <= 0) progress = 0;
        else if (rate < threshold) progress = Math.max(0.05, rate / threshold * 0.7);
        else progress = Math.min(1, 0.7 + (rate - threshold) / 5 * 0.3);

        bar.fill.width = bar.width * progress;
        bar.text.setText(`매수 전 예상 수익률 - 프로세스 ${Math.round(progress * 100)}% (임계치 ${threshold.toFixed(2)}%)`);

        // 컬러 변화: 저→고 (파랑→청록→라임)
        const color = progress < 0.5 ? 0x00ccff : (progress < 0.8 ? 0x00ffaa : 0x66ff33);
        bar.fill.fillColor = color;

        // 100% 도달 시 N/B 드랍 아이템 1개 드랍 (중복 드랍 방지 쿨다운)
        if (progress >= 1) {
            const gi = window.gameInitializer;
            const now = Date.now();
            const cooldownMs = 2000;
            if (!gi.gameData.lastBuyProcessDropAt || (now - gi.gameData.lastBuyProcessDropAt) > cooldownMs) {
                gi.gameData.lastBuyProcessDropAt = now;

                const trainer = gi.aiModels.find(m => m.isTrainer);
                if (trainer && window.nbCoinDropSystem && typeof window.nbCoinDropSystem.dropNBCoin === 'function') {
                    // 매수 구역 좌표 계산 (매수 구역에서만 드랍)
                    const startX = 100; // 매수 구역 X 좌표
                    const topY = 50;    // 매수 구역 Y 좌표
                    const buyAreaRadius = 30; // 매수 구역 반지름
                    
                    // 매수 구역 내에서 랜덤 위치 생성
                    const angle = Math.random() * Math.PI * 2;
                    const distance = Math.random() * buyAreaRadius;
                    const rx = startX + Math.cos(angle) * distance;
                    const ry = topY + Math.sin(angle) * distance;
                    
                    // 현재 선택된 분봉 확인
                    let currentTimeframe = null;
                    
                    // 방법 1: 활성화된 분봉 카드에서 확인
                    const activeCard = document.querySelector('.timeframe-card.active');
                    if (activeCard) {
                        currentTimeframe = activeCard.getAttribute('data-timeframe');
                    }
                    
                    // 방법 2: 선택된 분봉 카드에서 확인
                    if (!currentTimeframe) {
                        const selectedCard = document.querySelector('.timeframe-card.selected');
                        if (selectedCard) {
                            currentTimeframe = selectedCard.getAttribute('data-timeframe');
                        }
                    }
                    
                    // 방법 3: 기본값 설정
                    if (!currentTimeframe) {
                        currentTimeframe = '1m'; // 기본값
                    }
                    
                    window.nbCoinDropSystem.dropNBCoin(rx, ry, currentTimeframe);
                    if (window.logManager) {
                        window.logManager.addLog(`🪙 매수 프로세스 100% 달성 → N/B 드랍 아이템 1개 생성 (매수 구역 내 위치: ${Math.round(rx)}, ${Math.round(ry)}, 분봉: ${currentTimeframe})`);
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
}

window.ProcessBarController = ProcessBarController;

