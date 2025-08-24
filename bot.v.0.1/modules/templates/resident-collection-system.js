// 주민 수집 시스템 모듈
// 주민들이 맵을 탐색하면서 드랍된 N/B 코인을 수집하고 트레이너의 창고에 저장하는 기능을 담당

class ResidentCollectionSystem {
    constructor() {
        this.residents = []; // 주민 객체들
        this.collectedCoins = 0; // 수집된 코인 수
        this.warehouseCapacity = 100; // 창고 용량
        this.collectionRange = 30; // 수집 범위
        this.deliveryRange = 50; // 트레이너에게 전달하는 범위
        this.isInitialized = false;
    }

    // 시스템 초기화
    initialize(scene, config) {
        if (this.isInitialized) return;
        
        this.scene = scene;
        this.config = config;
        this.isInitialized = true;
        
        // 지속성 관리자 초기화
        if (window.residentPersistenceManager) {
            window.residentPersistenceManager.initialize();
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🏘️ 주민 수집 시스템 초기화 완료 (scene: ${!!scene}, config: ${!!config})`);
        }
        
        console.log('🏘️ 주민 수집 시스템 초기화 완료');
    }

    // 주민 등록 (기존 탐색자들을 주민으로 등록)
    registerResidents(explorers) {
        if (!explorers || !Array.isArray(explorers)) {
            console.warn('주민 등록 실패: 탐색자 데이터가 없습니다.');
            return;
        }

        this.residents = explorers.map((explorer, index) => ({
            id: index,
            circle: explorer.circle,
            name: explorer.name,
            role: explorer.role,
            targetX: explorer.targetX,
            targetY: explorer.targetY,
            discoveredCoords: explorer.discoveredCoords || [],
            collectedCoins: 0, // 개인 수집 코인 수
            isCarryingCoin: false, // 코인을 들고 있는지
            deliveryTarget: null, // 전달할 트레이너 위치
            state: 'exploring', // exploring, collecting, delivering
            collectionTimer: 0,
            deliveryTimer: 0
        }));

        if (window.logManager) {
            window.logManager.addLog(`🏘️ ${this.residents.length}명의 주민이 수집 시스템에 등록되었습니다.`);
            
            // 각 주민의 초기 상태 로그
            this.residents.forEach((resident, index) => {
                window.logManager.addLog(`🏘️ 주민 ${index}: ${resident.name.text} - 초기위치 (${Math.round(resident.circle.x)}, ${Math.round(resident.circle.y)})`);
            });
        }
        
        console.log(`🏘️ ${this.residents.length}명의 주민이 수집 시스템에 등록되었습니다.`);
        
        // 저장된 데이터가 있으면 복원
        this.restoreSavedData();
    }

    // 주민 업데이트 (매 프레임 호출)
    update() {
        if (!this.isInitialized || !this.residents.length) return;

        this.residents.forEach((resident, index) => {
            this.updateResident(resident, index);
        });
    }

    // 개별 주민 업데이트 (외부에서 호출)
    updateResidentByIndex(index) {
        if (!this.isInitialized || !this.residents[index]) return;
        
        // 디버깅: 주민 업데이트 상태 확인 (10초마다)
        if (Math.floor(Date.now() / 1000) % 10 === 0) {
            const resident = this.residents[index];
            if (window.logManager) {
                window.logManager.addLog(`🔍 주민 ${index} 업데이트: 상태=${resident.state}, 위치=(${Math.round(resident.circle.x)}, ${Math.round(resident.circle.y)})`);
            }
        }
        
        this.updateResident(this.residents[index], index);
    }

    // 개별 주민 업데이트
    updateResident(resident, index) {
        switch (resident.state) {
            case 'exploring':
                this.handleExploring(resident, index);
                break;
            case 'collecting':
                this.handleCollecting(resident, index);
                break;
            case 'delivering':
                this.handleDelivering(resident, index);
                break;
        }
    }

    // 탐색 상태 처리
    handleExploring(resident, index) {
        // N/B 코인 아이템이 있는지 확인
        const nearbyCoins = this.findNearbyCoins(resident);
        
        if (nearbyCoins.length > 0 && !resident.isCarryingCoin) {
            // 코인 발견 - 수집 모드로 전환
            resident.state = 'collecting';
            resident.targetCoin = nearbyCoins[0];
            resident.collectionTimer = 0;
            
            if (window.logManager) {
                window.logManager.addLog(`🔍 주민 ${resident.name.text}: N/B 코인 발견! 수집 시작...`);
            }
            
            // 역할 텍스트 업데이트
            if (resident.role && typeof resident.role.setText === 'function') {
                resident.role.setText('수집 중');
            }
        } else {
            // 기존 탐색 로직 유지
            this.continueExploration(resident);
        }
    }

    // 수집 상태 처리
    handleCollecting(resident, index) {
        if (!resident.targetCoin) {
            resident.state = 'exploring';
            return;
        }

        // 코인까지 이동
        const dx = resident.targetCoin.x - resident.circle.x;
        const dy = resident.targetCoin.y - resident.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < this.collectionRange) {
            // 코인 즉시 수집 (타이머 제거)
            this.collectCoin(resident, resident.targetCoin);
            resident.state = 'delivering';
            resident.isCarryingCoin = true;
            resident.collectedCoins++;
            resident.collectionTimer = 0;
            
            // 트레이너 위치를 목표로 설정
            this.setDeliveryTarget(resident);
            
            if (window.logManager) {
                window.logManager.addLog(`💰 주민 ${resident.name.text}: N/B 코인 즉시 수집 완료! 트레이너에게 전달 중...`);
            }
            
            // 역할 텍스트 업데이트
            if (resident.role && typeof resident.role.setText === 'function') {
                resident.role.setText('전달 중');
            }
        } else {
            // 코인으로 이동
            resident.circle.x += dx * 0.05;
            resident.circle.y += dy * 0.05;
            resident.name.x = resident.circle.x;
            resident.name.y = resident.circle.y - 6;
            resident.role.x = resident.circle.x;
            resident.role.y = resident.circle.y + 6;
        }
    }

    // 전달 상태 처리
    handleDelivering(resident, index) {
        if (!resident.deliveryTarget) {
            resident.state = 'exploring';
            return;
        }

        // 트레이너까지 이동
        const dx = resident.deliveryTarget.x - resident.circle.x;
        const dy = resident.deliveryTarget.y - resident.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < this.deliveryRange) {
            // 코인 전달
            resident.deliveryTimer++;
            
            if (resident.deliveryTimer >= 30) { // 0.5초 후 전달 완료
                this.deliverCoin(resident);
                resident.state = 'exploring';
                resident.isCarryingCoin = false;
                resident.deliveryTimer = 0;
                resident.deliveryTarget = null;
                
                if (window.logManager) {
                    window.logManager.addLog(`📦 주민 ${resident.name.text}: N/B 코인을 트레이너 창고에 전달 완료!`);
                }
                
                // 역할 텍스트 업데이트
                if (resident.role && typeof resident.role.setText === 'function') {
                    resident.role.setText(`탐색 (${resident.discoveredCoords.length}/8)`);
                }
            }
        } else {
            // 트레이너로 이동
            resident.circle.x += dx * 0.05;
            resident.circle.y += dy * 0.05;
            resident.name.x = resident.circle.x;
            resident.name.y = resident.circle.y - 6;
            resident.role.x = resident.circle.x;
            resident.role.y = resident.circle.y + 6;
        }
    }

    // 탐색 계속
    continueExploration(resident) {
        const modelX = resident.circle.x;
        const modelY = resident.circle.y;
        const distanceToTarget = Math.sqrt((resident.targetX - modelX) ** 2 + (resident.targetY - modelY) ** 2);
        
        if (distanceToTarget < 25) {
            // 새로운 좌표 발견
            const currentCoord = { x: Math.round(modelX), y: Math.round(modelY) };
            
            // 중복 체크
            const isDuplicate = resident.discoveredCoords.some(coord => 
                Math.abs(coord.x - currentCoord.x) < 15 && 
                Math.abs(coord.y - currentCoord.y) < 15
            );
            
            if (!isDuplicate) {
                resident.discoveredCoords.push(currentCoord);
                
                if (resident.discoveredCoords.length > 8) {
                    resident.discoveredCoords.shift();
                }
            }
            
            // 새로운 목표 설정
            const config = window.gameInitializer?.game?.config || { width: 1086, height: 500 };
            resident.targetX = Math.random() * (config.width - 80) + 40;
            resident.targetY = Math.random() * (config.height - 80) + 40;
            
            if (resident.role && typeof resident.role.setText === 'function') {
                resident.role.setText(`탐색 (${resident.discoveredCoords.length}/8)`);
            }
        }
        
        // 탐색자 이동
        const dx = resident.targetX - modelX;
        const dy = resident.targetY - modelY;
        
        if (Math.abs(dx) > 1) {
            resident.circle.x += dx * 0.05; // 이동 속도 증가
            resident.name.x = resident.circle.x;
            resident.role.x = resident.circle.x;
        }
        
        if (Math.abs(dy) > 1) {
            resident.circle.y += dy * 0.05; // 이동 속도 증가
            resident.name.y = resident.circle.y - 6;
            resident.role.y = resident.circle.y + 6;
        }
    }

    // 근처의 N/B 코인 찾기
    findNearbyCoins(resident) {
        if (!window.nbCoinDropSystem || !window.nbCoinDropSystem.nbCoinItems) {
            return [];
        }

        const nearbyCoins = [];
        const residentX = resident.circle.x;
        const residentY = resident.circle.y;

        window.nbCoinDropSystem.nbCoinItems.forEach(coin => {
            if (coin && coin.sprite) {
                const distance = Math.sqrt(
                    (coin.sprite.x - residentX) ** 2 + 
                    (coin.sprite.y - residentY) ** 2
                );
                
                if (distance < 150) { // 탐지 범위 확장 (100 → 150)
                    nearbyCoins.push(coin.sprite);
                }
            }
        });

        return nearbyCoins;
    }

    // 코인 수집
    collectCoin(resident, coinSprite) {
        if (!window.nbCoinDropSystem) return;

        // N/B 코인 드롭 시스템에서 코인 제거
        const coinIndex = window.nbCoinDropSystem.nbCoinItems.findIndex(coin => 
            coin && coin.sprite === coinSprite
        );
        
        if (coinIndex !== -1) {
            const collectedCoin = window.nbCoinDropSystem.nbCoinItems.splice(coinIndex, 1)[0];
            
            // 스프라이트 제거
            if (collectedCoin.sprite) {
                collectedCoin.sprite.destroy();
            }
            
            // N/B 코인 디스플레이 업데이트
            window.nbCoinDropSystem.updateNBCoinDisplay();
            
            if (window.logManager) {
                window.logManager.addLog(`💰 주민 ${resident.name.text}: N/B 코인 수집! (총 수집: ${this.collectedCoins + 1}개)`);
            }
        }
    }

    // 전달 목표 설정
    setDeliveryTarget(resident) {
        // 트레이너 위치 찾기
        if (window.gameInitializer && window.gameInitializer.trainer) {
            const trainer = window.gameInitializer.trainer;
            resident.deliveryTarget = {
                x: trainer.circle.x,
                y: trainer.circle.y
            };
        } else {
            // 트레이너가 없으면 중앙으로 이동
            const config = window.gameInitializer?.game?.config || { width: 1086, height: 500 };
            resident.deliveryTarget = {
                x: config.width / 2,
                y: config.height / 2
            };
        }
    }

    // 코인 전달
    deliverCoin(resident) {
        // 주민이 N/B MAX COIN을 전달하는 것이므로 N/B MIN 코인은 증가시키지 않음
        if (window.gameInitializer && window.gameInitializer.gameData) {
            // N/B 코인 디스플레이 업데이트 (기존 값 유지)
            if (window.nbCoinDisplay && typeof window.nbCoinDisplay.setText === 'function') {
                const nbCoins = window.gameInitializer.gameData.nbCoins;
                const dropItems = window.nbCoinDropSystem ? window.nbCoinDropSystem.nbCoinItems.length : 0;
                window.nbCoinDisplay.setText(`N/B MIN 코인: ${nbCoins}개 (드랍 아이템: ${dropItems}개)`);
            }
            
            // 트레이너 다이얼로그 업데이트
            if (window.trainerDialog && typeof window.trainerDialog.setText === 'function') {
                window.trainerDialog.setText(`📦 주민 ${resident.name.text}이(가) N/B MAX COIN을 창고에 전달했습니다!`);
            }
            
            if (window.logManager) {
                window.logManager.addLog(`📦 창고 업데이트: N/B MIN 코인 ${window.gameInitializer.gameData.nbCoins}개 유지 (주민 전달)`);
            }
            
            // 코인 전달 후 즉시 저장
            if (window.residentPersistenceManager) {
                window.residentPersistenceManager.saveResidentData(this);
            }
        }
    }

    // 주민 상태 정보 가져오기
    getResidentStatus() {
        return {
            totalResidents: this.residents.length,
            collectedCoins: this.collectedCoins,
            warehouseCapacity: this.warehouseCapacity,
            residents: this.residents.map(resident => ({
                name: resident.name.text,
                state: resident.state,
                collectedCoins: resident.collectedCoins,
                isCarryingCoin: resident.isCarryingCoin,
                discoveredCoords: resident.discoveredCoords.length
            }))
        };
    }

    // 저장된 데이터 복원
    restoreSavedData() {
        if (!window.residentPersistenceManager) {
            console.warn('지속성 관리자가 없어서 저장된 데이터를 복원할 수 없습니다.');
            return;
        }

        const savedData = window.residentPersistenceManager.loadResidentData();
        if (savedData) {
            const restored = window.residentPersistenceManager.applySavedData(this, savedData);
            if (restored) {
                // UI가 준비된 경우에만 N/B 코인 디스플레이 업데이트
                if (window.nbCoinDisplay) {
                    this.updateNBCoinDisplay();
                } else {
                    console.log('⚠️ UI가 아직 준비되지 않아 N/B 코인 디스플레이 업데이트를 건너뜀');
                }
                
                if (window.logManager) {
                    window.logManager.addLog(`🔄 주민 수집 시스템 저장된 데이터 복원 완료`);
                }
            }
        }
    }

    // N/B MIN 코인 디스플레이 업데이트
    updateNBCoinDisplay() {
        // UI 요소가 아직 생성되지 않은 경우 스킵
        if (!window.nbCoinDisplay) {
            console.log('⚠️ N/B MIN 코인 디스플레이가 아직 생성되지 않음 - 업데이트 스킵');
            return;
        }
        
        // setText 메서드가 있는지 확인
        if (typeof window.nbCoinDisplay.setText !== 'function') {
            console.warn('⚠️ N/B MIN 코인 디스플레이의 setText 메서드가 유효하지 않음');
            return;
        }
        
        try {
            const nbCoins = window.gameInitializer?.gameData?.nbCoins || 0;
            const dropItems = window.nbCoinDropSystem ? window.nbCoinDropSystem.nbCoinItems.length : 0;
            window.nbCoinDisplay.setText(`N/B MIN 코인: ${nbCoins}개 (드랍 아이템: ${dropItems}개)`);
        } catch (error) {
            console.error('❌ N/B MIN 코인 디스플레이 업데이트 중 오류:', error);
        }
    }

    // 시스템 상태 리셋
    reset() {
        // 모든 주민을 N/B 길드 위치 (100, 100)로 리셋
        this.residents.forEach((resident, index) => {
            if (resident.circle) {
                resident.circle.x = 100;
                resident.circle.y = 100;
            }
            if (resident.name) {
                resident.name.x = 100;
                resident.name.y = 100;
            }
            if (resident.role) {
                resident.role.x = 100;
                resident.role.y = 100;
            }
            resident.targetX = 100;
            resident.targetY = 100;
            resident.discoveredCoords = [];
            resident.collectedCoins = 0;
            resident.isCarryingCoin = false;
            resident.deliveryTarget = null;
            resident.state = 'exploring';
            resident.collectionTimer = 0;
            resident.deliveryTimer = 0;
        });
        
        this.collectedCoins = 0;
        this.isInitialized = false;
        
        // 저장된 데이터도 삭제
        if (window.residentPersistenceManager) {
            window.residentPersistenceManager.clearSavedData();
        }
        
        if (window.logManager) {
            window.logManager.addLog(`🏘️ 주민 수집 시스템 완전 리셋 - 모든 위치 N/B 길드 (100,100), 모든 데이터 0으로 초기화`);
        }
    }

    // 주민 수집 시스템 재시작
    restart() {
        console.log('🔄 주민 수집 시스템 재시작 시작...');
        
        try {
            // 시스템 상태 초기화
            this.collectedCoins = 0;
            this.isInitialized = false;
            
            // 주민들 재시작
            if (this.residents && Array.isArray(this.residents)) {
                this.residents.forEach((resident, index) => {
                    this.restartResident(resident, index);
                });
            }
            
            // 시스템 재초기화
            this.isInitialized = true;
            
            // 지속성 관리자 재시작
            if (window.residentPersistenceManager) {
                window.residentPersistenceManager.restart();
            }
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 주민 수집 시스템 재시작 완료 - ${this.residents.length}명의 주민 재활성화`);
            }
            
            console.log('✅ 주민 수집 시스템 재시작 완료');
        } catch (error) {
            console.error('❌ 주민 수집 시스템 재시작 실패:', error);
        }
    }

    // 개별 주민 재시작
    restartResident(resident, index) {
        if (!resident || !resident.circle) {
            console.log(`❌ 주민 ${index} 재시작 실패: 주민 데이터가 유효하지 않음`);
            return;
        }
        
        try {
            // 주민 상태 초기화
            resident.collectedCoins = 0;
            resident.isCarryingCoin = false;
            resident.deliveryTarget = null;
            resident.state = 'exploring';
            resident.collectionTimer = 0;
            resident.deliveryTimer = 0;
            
            // 새로운 탐색 목표 설정
            this.setNewExplorationTarget(resident, index);
            
            // 주민 위치 초기화 (기본 위치로)
            this.initializeResidentPosition(resident, index);
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 주민 ${index} 재시작 완료 - 상태: ${resident.state}, 위치: (${Math.round(resident.circle.x)}, ${Math.round(resident.circle.y)})`);
            }
            
            console.log(`🏘️ 주민 ${index} 재시작 완료`);
        } catch (error) {
            console.error(`❌ 주민 ${index} 재시작 실패:`, error);
        }
    }

    // 주민 위치 초기화
    initializeResidentPosition(resident, index) {
        if (!resident.circle) return;
        
        try {
            // 기본 위치 설정 (화면의 다른 구역에 분산 배치)
            const config = window.gameInitializer?.game?.config || { width: 1086, height: 500 };
            const margin = 50;
            
            let defaultX, defaultY;
            
            switch (index) {
                case 0: // 첫 번째 주민 - 좌측 상단
                    defaultX = config.width * 0.25;
                    defaultY = config.height * 0.25;
                    break;
                case 1: // 두 번째 주민 - 우측 상단
                    defaultX = config.width * 0.75;
                    defaultY = config.height * 0.25;
                    break;
                case 2: // 세 번째 주민 - 좌측 하단
                    defaultX = config.width * 0.25;
                    defaultY = config.height * 0.75;
                    break;
                case 3: // 네 번째 주민 - 우측 하단
                    defaultX = config.width * 0.75;
                    defaultY = config.height * 0.75;
                    break;
                default: // 기본 - 중앙
                    defaultX = config.width * 0.5;
                    defaultY = config.height * 0.5;
            }
            
            // 화면 경계 내로 제한
            defaultX = Math.max(margin, Math.min(config.width - margin, defaultX));
            defaultY = Math.max(margin, Math.min(config.height - margin, defaultY));
            
            // 위치 설정
            resident.circle.x = defaultX;
            resident.circle.y = defaultY;
            resident.targetX = defaultX;
            resident.targetY = defaultY;
            
            // 텍스트 위치 업데이트
            if (resident.name) {
                resident.name.x = defaultX;
                resident.name.y = defaultY - 4;
            }
            if (resident.role) {
                resident.role.x = defaultX;
                resident.role.y = defaultY + 4;
            }
            
            console.log(`📍 주민 ${index} 위치 초기화: (${Math.round(defaultX)}, ${Math.round(defaultY)})`);
        } catch (error) {
            console.error(`❌ 주민 ${index} 위치 초기화 실패:`, error);
        }
    }

    // 새로운 탐색 목표 설정
    setNewExplorationTarget(resident, index) {
        if (!resident.circle) return;
        
        try {
            // 현재 위치를 기준으로 새로운 목표 설정
            const currentX = resident.circle.x;
            const currentY = resident.circle.y;
            
            // 랜덤한 방향으로 새로운 목표 설정
            const angle = Math.random() * 2 * Math.PI;
            const distance = 80 + Math.random() * 120; // 80~200px 거리
            
            const newTargetX = currentX + Math.cos(angle) * distance;
            const newTargetY = currentY + Math.sin(angle) * distance;
            
            // 화면 경계 내로 제한
            const config = window.gameInitializer?.game?.config || { width: 1086, height: 500 };
            const margin = 50;
            
            resident.targetX = Math.max(margin, Math.min(config.width - margin, newTargetX));
            resident.targetY = Math.max(margin, Math.min(config.height - margin, newTargetY));
            
            console.log(`🎯 주민 ${index} 새로운 탐색 목표 설정: (${Math.round(resident.targetX)}, ${Math.round(resident.targetY)})`);
        } catch (error) {
            console.error(`❌ 주민 ${index} 새로운 탐색 목표 설정 실패:`, error);
        }
    }

    // 주민 수집 시스템 상태 확인
    getSystemStatus() {
        return {
            totalResidents: this.residents ? this.residents.length : 0,
            collectedCoins: this.collectedCoins,
            warehouseCapacity: this.warehouseCapacity,
            isInitialized: this.isInitialized,
            activeResidents: this.residents ? this.residents.filter(r => r.state !== 'idle').length : 0
        };
    }

    // 주민 수집 시스템 상태 설정
    setSystemStatus(status) {
        if (status.collectedCoins !== undefined) this.collectedCoins = status.collectedCoins;
        if (status.warehouseCapacity !== undefined) this.warehouseCapacity = status.warehouseCapacity;
        if (status.isInitialized !== undefined) this.isInitialized = status.isInitialized;
    }
}

// 전역 인스턴스 생성
window.residentCollectionSystem = new ResidentCollectionSystem();
