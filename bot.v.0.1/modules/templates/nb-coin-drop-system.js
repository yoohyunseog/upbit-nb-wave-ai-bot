// N/B MIN 코인 드랍 시스템 모듈
// 매수 구역에서 N/B MIN 코인을 랜덤 위치로 드랍하는 기능을 담당

class NBCoinDropSystem {
    constructor() {
        this.nbCoinItems = [];
        this.dropCount = 0;
        this.maxDrops = 10; // 최대 드랍 개수 제한
        this.dropCooldown = 5000; // 드랍 쿨다운 (5초)
        this.lastDropTime = 0;
        this.isInitialized = false;
    }

    // 시스템 초기화
    initialize(scene, config) {
        if (this.isInitialized) {
            console.log('⚠️ N/B 코인 드랍 시스템이 이미 초기화되었습니다.');
            return;
        }

        this.scene = scene;
        this.config = config;
        this.isInitialized = true;

        // 새로고침 시 드랍 아이템 초기화
        this.clearNBCoinItems();
        
        // gameData의 드랍 아이템 카운터도 0으로 초기화
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.dropItemsCount = 0;
        }

        // 전역 함수로 등록
        window.createNBCoinItem = () => this.createNBCoinItem();
        window.dropNBCoin = (x, y) => this.dropNBCoin(x, y);
        window.removeNBCoinItem = (item) => this.removeNBCoinItem(item);
        window.getNBCoinItems = () => this.nbCoinItems;
        window.clearNBCoinItems = () => this.clearNBCoinItems();

        console.log('✅ N/B MIN 코인 드랍 시스템 초기화 완료 (드랍 아이템 초기화됨)');
    }

    // N/B MIN 코인 드랍 (매수 구역에서만 드랍)
    dropNBCoin(x, y, sourceTimeframe = null) {
        if (!this.isInitialized) {
            console.error('❌ N/B MIN 코인 드랍 시스템이 초기화되지 않았습니다.');
            return null;
        }

        // 드랍 개수 제한 확인
        if (this.nbCoinItems.length >= this.maxDrops) {
            console.log(`⚠️ 최대 드랍 개수(${this.maxDrops}개)에 도달했습니다.`);
            return null;
        }

        // 쿨다운 확인
        const currentTime = Date.now();
        if (currentTime - this.lastDropTime < this.dropCooldown) {
            console.log(`⏳ 드랍 쿨다운 중... (${Math.ceil((this.dropCooldown - (currentTime - this.lastDropTime)) / 1000)}초 남음)`);
            return null;
        }

        // 씬 준비 보장
        if (!this.scene && window.gameInitializer?.scene) {
            this.scene = window.gameInitializer.scene;
        }
        if (!this.scene || !this.scene.add) {
            console.error('❌ Scene not ready: dropNBCoin 호출 시 this.scene이 유효하지 않습니다.');
            return null;
        }

        // 매수 구역 좌표 계산 (매수 구역에서만 드랍)
        const startX = 100; // 매수 구역 X 좌표
        const topY = 50;    // 매수 구역 Y 좌표
        const buyAreaRadius = 30; // 매수 구역 반지름
        
        // 매수 구역 내에서 랜덤 위치 생성
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.random() * buyAreaRadius;
        const finalX = startX + Math.cos(angle) * distance;
        const finalY = topY + Math.sin(angle) * distance;

        // N/B MIN 코인 아이템 생성
        const item = {
            id: `nb-coin-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            polygon: this.scene.add.polygon(finalX, finalY, [
                0, -8, 6, -4, 6, 4, 0, 8, -6, 4, -6, -4
            ], 0xffaa00),
            collected: false,
            connectionLines: [],
            dropTime: currentTime,
            position: { x: finalX, y: finalY },
            sourceTimeframe: sourceTimeframe // 드랍된 분봉 추적
        };

        item.polygon.setOrigin(0.5, 0.5);
        item.polygon.setInteractive();

        // 회전 애니메이션
        if (this.scene.tweens) {
            this.scene.tweens.add({
                targets: item.polygon,
                angle: 360,
                duration: 2000,
                ease: 'Linear',
                repeat: -1
            });
        }

        // 아이템을 배열에 추가
        this.nbCoinItems.push(item);
        this.dropCount++;
        this.lastDropTime = currentTime;

        // 드랍 아이템 카운터 증가
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.dropItemsCount = (window.gameInitializer.gameData.dropItemsCount || 0) + 1;
        }

        console.log(`🪙 N/B MIN 코인 드랍 완료: 매수 구역 내 위치 (${Math.round(finalX)}, ${Math.round(finalY)})${sourceTimeframe ? `, 분봉: ${sourceTimeframe}` : ''}`);
        
        if (window.logManager) {
            window.logManager.addLog(`🪙 N/B MIN 코인 드랍 완료: 매수 구역 내 위치 (${Math.round(finalX)}, ${Math.round(finalY)})${sourceTimeframe ? `, 분봉: ${sourceTimeframe}` : ''}`);
        }
        
        return item;
    }

    // N/B MIN 코인 아이템 생성 (매수 구역에서만 호출)
    createNBCoinItem() {
        if (!this.isInitialized) {
            console.error('❌ N/B MIN 코인 드랍 시스템이 초기화되지 않았습니다.');
            return null;
        }

        // 씬 준비 보장
        if (!this.scene && window.gameInitializer?.scene) {
            this.scene = window.gameInitializer.scene;
        }
        if (!this.scene || !this.scene.add) {
            console.error('❌ Scene not ready: createNBCoinItem 호출 시 this.scene이 유효하지 않습니다.');
            return null;
        }

        // 드랍 개수 제한 확인
        if (this.nbCoinItems.length >= this.maxDrops) {
            console.log(`⚠️ 최대 드랍 개수(${this.maxDrops}개)에 도달했습니다.`);
            return null;
        }

        // 쿨다운 확인
        const currentTime = Date.now();
        if (currentTime - this.lastDropTime < this.dropCooldown) {
            console.log(`⏳ 드랍 쿨다운 중... (${Math.ceil((this.dropCooldown - (currentTime - this.lastDropTime)) / 1000)}초 남음)`);
            return null;
        }

        // 매수 구역 좌표 계산 (매수 구역에서만 드랍)
        const startX = 100; // 매수 구역 X 좌표
        const topY = 50;    // 매수 구역 Y 좌표
        const buyAreaRadius = 30; // 매수 구역 반지름
        
        // 매수 구역 내에서 랜덤 위치 생성
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.random() * buyAreaRadius;
        const x = startX + Math.cos(angle) * distance;
        const y = topY + Math.sin(angle) * distance;

        // N/B MIN 코인 아이템 생성
        const item = {
            id: `nb-coin-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            polygon: this.scene.add.polygon(x, y, [
                0, -8, 6, -4, 6, 4, 0, 8, -6, 4, -6, -4
            ], 0xffaa00),
            collected: false,
            connectionLines: [],
            dropTime: currentTime,
            position: { x, y }
        };

        item.polygon.setOrigin(0.5, 0.5);
        item.polygon.setInteractive();

        // 회전 애니메이션 (tweens가 준비된 경우에만)
        if (this.scene.tweens) {
            this.scene.tweens.add({
                targets: item.polygon,
                rotation: Math.PI * 2,
                duration: 3000,
                repeat: -1,
                ease: 'Linear'
            });
        }

        // 클릭 이벤트 추가
        item.polygon.on('pointerdown', () => {
            this.collectNBCoinItem(item);
        });

        // 호버 효과
        item.polygon.on('pointerover', () => {
            item.polygon.setScale(1.2);
        });

        item.polygon.on('pointerout', () => {
            item.polygon.setScale(1.0);
        });

        this.nbCoinItems.push(item);
        this.dropCount++;
        this.lastDropTime = currentTime;

        // 드랍 아이템 카운터 증가
        if (window.gameInitializer && window.gameInitializer.gameData) {
            window.gameInitializer.gameData.dropItemsCount = (window.gameInitializer.gameData.dropItemsCount || 0) + 1;
        }

        // 로그 기록
        if (window.logManager) {
            const dropItemsCount = window.gameInitializer?.gameData?.dropItemsCount || 0;
            window.logManager.addLog(`🪙 N/B MIN 코인 드랍: 매수 구역 내 위치 (${Math.round(x)}, ${Math.round(y)}) → 드랍 아이템 ${dropItemsCount}개`);
        }

        console.log(`🪙 N/B MIN 코인 아이템 생성: 매수 구역 내 위치 (${Math.round(x)}, ${Math.round(y)}) | 총 ${this.nbCoinItems.length}개`);

        // UI 업데이트
        this.updateNBCoinDisplay();

        return item;
    }

    // N/B MIN 코인 아이템 수집
    collectNBCoinItem(item) {
        if (item.collected) {
            return;
        }

        item.collected = true;

        // 수집 애니메이션 (빠른 수집)
        this.scene.tweens.add({
            targets: item.polygon,
            scaleX: 0,
            scaleY: 0,
            alpha: 0,
            duration: 200, // 수집 애니메이션 속도 증가 (500ms → 200ms)
            ease: 'Power2',
            onComplete: () => {
                this.removeNBCoinItem(item);
            }
        });

        // 수집 효과음 (옵션)
        if (window.soundManager) {
            window.soundManager.playCollectSound();
        }

        // N/B MIN 코인 개수 증가 및 드랍 아이템 카운터 감소
        if (window.gameInitializer && window.gameInitializer.gameData) {
            const previousCoins = window.gameInitializer.gameData.nbCoins || 0;
            const previousDropItems = window.gameInitializer.gameData.dropItemsCount || 0;
            
            // N/B MIN 코인 증가
            window.gameInitializer.gameData.nbCoins = previousCoins + 1;
            
            // 드랍 아이템 카운터 감소 (수집되었으므로)
            window.gameInitializer.gameData.dropItemsCount = Math.max(0, previousDropItems - 1);
            
            // 특정 분봉의 N/B MAX 코인 증가 (sourceTimeframe이 있는 경우)
            if (item.sourceTimeframe && window.nbCoinStatus) {
                // 해당 분봉의 현재 N/B MAX COIN 상태 확인
                let currentNbCoins = 0;
                if (window.cardStorageSystem) {
                    const storage = window.cardStorageSystem.getCardStorage(item.sourceTimeframe);
                    currentNbCoins = storage.nbCoins || 0;
                }
                
                // N/B MAX COIN이 이미 1 이상이면 추가하지 않음
                if (currentNbCoins >= 1) {
                    console.log(`⚠️ ${item.sourceTimeframe} 분봉의 N/B MAX COIN이 이미 ${currentNbCoins}개입니다. 추가하지 않습니다.`);
                } else {
                    // N/B MAX COIN이 0일 때만 추가
                    window.nbCoinStatus[item.sourceTimeframe] = 1;
                    
                    // 카드 저장소 시스템에도 추가
                    if (window.cardStorageSystem && typeof window.cardStorageSystem.addNBCoin === 'function') {
                        window.cardStorageSystem.addNBCoin(item.sourceTimeframe, 1);
                    }
                    
                    // N/B 미네랄도 추가 (수집 시 1.0% 추가)
                    if (window.cardStorageSystem && typeof window.cardStorageSystem.addNBMineral === 'function') {
                        window.cardStorageSystem.addNBMineral(item.sourceTimeframe, 1.0);
                    }
                    
                    // 해당 분봉 카드의 N/B MAX 코인 배지 업데이트
                    this.updateTimeframeCardNBCoin(item.sourceTimeframe, 1);
                    
                    console.log(`💰 N/B MIN 코인 수집 완료: 분봉 ${item.sourceTimeframe}의 N/B MAX 코인 1개 증가, N/B 미네랄 1.00% 증가`);
                }
            }
            
            // 로그 기록 (N/B MIN 코인 증가 없음)
            if (window.logManager) {
                const timeframeInfo = item.sourceTimeframe ? `, 분봉 ${item.sourceTimeframe} N/B MAX 코인 증가` : '';
                window.logManager.addLog(`💰 N/B MIN 코인 수집 완료: 위치 (${Math.round(item.position.x)}, ${Math.round(item.position.y)}) → N/B MIN 코인 ${window.gameInitializer.gameData.nbCoins}개 (+1), 드랍 아이템 ${window.gameInitializer.gameData.dropItemsCount}개 (-1)${timeframeInfo}`);
            }
            
            console.log(`💰 N/B MIN 코인 수집 완료: 위치 (${Math.round(item.position.x)}, ${Math.round(item.position.y)}) → N/B MIN 코인 ${window.gameInitializer.gameData.nbCoins}개 (+1), 드랍 아이템 ${window.gameInitializer.gameData.dropItemsCount}개 (-1)${item.sourceTimeframe ? `, 분봉 ${item.sourceTimeframe}` : ''}`);
            
            // N/B MIN 코인 디스플레이 업데이트 (기존 값 유지)
            if (window.nbCoinDisplay && typeof window.nbCoinDisplay.setText === 'function') {
                const nbCoins = window.gameInitializer.gameData.nbCoins;
                const dropItems = this.nbCoinItems.length;
                window.nbCoinDisplay.setText(`N/B MIN 코인: ${nbCoins}개 (드랍 아이템: ${dropItems}개)`);
            }
            
            // 자동 저장
            window.gameInitializer.saveGameData();
        }
    }

    // 특정 분봉 카드의 N/B 코인 배지 업데이트
    updateTimeframeCardNBCoin(timeframe, nbCoins) {
        try {
            // 해당 분봉 카드 찾기
            const card = document.querySelector(`[data-timeframe="${timeframe}"]`);
            if (card) {
                // N/B 코인 배지 찾기 (여러 방법 시도)
                let nbCoinBadge = null;
                
                // 방법 1: N/B COIN 텍스트가 포함된 배지 찾기
                const allBadges = card.querySelectorAll('.badge');
                for (const badge of allBadges) {
                    if (badge.textContent && badge.textContent.includes('N/B COIN')) {
                        nbCoinBadge = badge;
                        break;
                    }
                }
                
                // 방법 2: 성공/실패 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = card.querySelector('.badge.bg-success, .badge.bg-secondary');
                }
                
                // 방법 3: 일반 배지 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = card.querySelector('[class*="badge"]');
                }
                
                // 방법 4: data-nb-coin 속성 찾기
                if (!nbCoinBadge) {
                    nbCoinBadge = card.querySelector('[data-nb-coin]');
                }
                
                if (nbCoinBadge) {
                    // 배지 텍스트와 클래스 업데이트
                    nbCoinBadge.textContent = `N/B COIN: ${nbCoins}`;
                    nbCoinBadge.className = nbCoins > 0 ? 'badge bg-success' : 'badge bg-secondary';
                    nbCoinBadge.setAttribute('data-nb-coin', nbCoins);
                    
                    console.log(`✅ 분봉 ${timeframe} 카드 N/B 코인 배지 업데이트: ${nbCoins}개`);
                } else {
                    console.log(`⚠️ 분봉 ${timeframe} 카드에서 N/B 코인 배지를 찾을 수 없음`);
                }
            } else {
                console.log(`⚠️ 분봉 ${timeframe} 카드를 찾을 수 없음`);
            }
        } catch (error) {
            console.error('❌ 분봉 카드 N/B 코인 업데이트 중 오류:', error);
        }
    }

    // N/B 코인 아이템 제거
    removeNBCoinItem(item) {
        const index = this.nbCoinItems.indexOf(item);
        if (index > -1) {
            this.nbCoinItems.splice(index, 1);
        }

        // 연결선들 제거
        item.connectionLines.forEach(line => {
            if (line && line.destroy) {
                line.destroy();
            }
        });

        // 폴리곤 제거
        if (item.polygon && item.polygon.destroy) {
            item.polygon.destroy();
        }

        // UI 업데이트
        this.updateNBCoinDisplay();
    }

    // 모든 N/B 코인 아이템 제거
    clearNBCoinItems() {
        this.nbCoinItems.forEach(item => {
            this.removeNBCoinItem(item);
        });
        this.nbCoinItems = [];
        this.dropCount = 0;
        this.lastDropTime = 0;

        console.log('🗑️ 모든 N/B 코인 아이템 제거 완료');
    }

    // N/B MIN 코인 디스플레이 업데이트
    updateNBCoinDisplay() {
        // Phaser 텍스트 객체 업데이트
        if (window.nbCoinDisplay && typeof window.nbCoinDisplay.setText === 'function') {
            const nbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
            const dropItemsCount = window.gameInitializer?.gameData?.dropItemsCount || 0;
            window.nbCoinDisplay.setText(`N/B MIN 코인: ${nbCoins}개 (드랍 아이템: ${dropItemsCount}개)`);
        }
        
        // HTML 요소 업데이트
        const nbCoinElement = document.getElementById('nb-coin-count');
        if (nbCoinElement) {
            const nbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
            nbCoinElement.textContent = `N/B MIN 코인: ${nbCoins}개`;
        }
    }

    // 매수 구역 도착 시 자동 드랍
    handleBuyAreaArrival() {
        if (!this.isInitialized) {
            return;
        }

        // 자동 드랍 실행
        const droppedItem = this.createNBCoinItem();
        
        if (droppedItem) {
            // 트레이너 대화창 업데이트
            if (window.trainerDialog) {
                const currentTime = new Date().toLocaleTimeString();
                const dialogText = `🪙 매수 구역 도착 → N/B MIN 코인 드랍 완료 | 위치: (${Math.round(droppedItem.position.x)}, ${Math.round(droppedItem.position.y)}) | 시간: ${currentTime}`;
                if (window.trainerDialog && typeof window.trainerDialog.setText === 'function') {
                    window.trainerDialog.setText(dialogText);
                }
                
                if (window.logManager) {
                    window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }
            }
        }
    }

    // 시스템 상태 확인
    getSystemStatus() {
        return {
            isInitialized: this.isInitialized,
            totalItems: this.nbCoinItems.length,
            dropCount: this.dropCount,
            maxDrops: this.maxDrops,
            cooldownRemaining: Math.max(0, this.dropCooldown - (Date.now() - this.lastDropTime))
        };
    }

    // 설정 업데이트
    updateSettings(settings) {
        if (settings.maxDrops !== undefined) {
            this.maxDrops = settings.maxDrops;
        }
        if (settings.dropCooldown !== undefined) {
            this.dropCooldown = settings.dropCooldown;
        }
        
        console.log('⚙️ N/B MIN 코인 드랍 시스템 설정 업데이트:', settings);
    }

    // 디버그 정보 출력
    debugInfo() {
        console.log('🔍 N/B MIN 코인 드랍 시스템 디버그 정보:');
        console.log('- 초기화 상태:', this.isInitialized);
        console.log('- 총 아이템 수:', this.nbCoinItems.length);
        console.log('- 드랍 횟수:', this.dropCount);
        console.log('- 최대 드랍 개수:', this.maxDrops);
        console.log('- 쿨다운 남은 시간:', Math.max(0, this.dropCooldown - (Date.now() - this.lastDropTime)), 'ms');
        
        this.nbCoinItems.forEach((item, index) => {
            console.log(`- 아이템 ${index + 1}: ID=${item.id}, 위치=(${Math.round(item.position.x)}, ${Math.round(item.position.y)}), 수집됨=${item.collected}`);
        });
    }
}

// 전역 인스턴스 생성
window.nbCoinDropSystem = new NBCoinDropSystem();

// 전역 함수들
window.createNBCoinItem = () => window.nbCoinDropSystem.createNBCoinItem();
window.dropNBCoin = (x, y) => window.nbCoinDropSystem.dropNBCoin(x, y);
window.removeNBCoinItem = (item) => window.nbCoinDropSystem.removeNBCoinItem(item);
window.getNBCoinItems = () => window.nbCoinDropSystem.nbCoinItems;
window.clearNBCoinItems = () => window.nbCoinDropSystem.clearNBCoinItems();
window.handleBuyAreaArrival = () => window.nbCoinDropSystem.handleBuyAreaArrival();

// 전역 디버깅 함수들
window.debugNBCoinSystem = () => {
    if (window.nbCoinDropSystem) {
        window.nbCoinDropSystem.debugInfo();
        return window.nbCoinDropSystem.getSystemStatus();
    } else {
        console.log('❌ N/B MIN 코인 드랍 시스템이 초기화되지 않았습니다.');
        return null;
    }
};

window.getNBCoinSystemStatus = () => {
    if (window.nbCoinDropSystem) {
        return window.nbCoinDropSystem.getSystemStatus();
    } else {
        console.log('❌ N/B 코인 드랍 시스템이 초기화되지 않았습니다.');
        return null;
    }
};

window.forceDropNBCoin = (x, y) => {
    if (window.nbCoinDropSystem) {
        return window.nbCoinDropSystem.dropNBCoin(x, y);
    } else {
        console.log('❌ N/B 코인 드랍 시스템이 초기화되지 않았습니다.');
        return null;
    }
};
