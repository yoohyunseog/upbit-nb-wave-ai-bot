// 탐색원 이동 시스템 모듈
// 탐색원들의 기본적인 탐색 이동을 담당하는 독립적인 시스템

class ExplorerMovementSystem {
    constructor() {
        this.explorers = []; // 탐색원 객체들
        this.isInitialized = false;
        this.movementSpeed = 0.2; // 이동 속도 대폭 증가 (0.1 → 0.2)
        this.arrivalThreshold = 15; // 도착 임계값 감소 (25px → 15px)
        this.explorationRange = 15; // 탐색 범위
        this.logMessages = []; // 로그 메시지 배열
        this.maxLogMessages = 100; // 최대 로그 메시지 수
        this.lastSaveTime = 0; // 마지막 저장 시간
        this.saveInterval = 3000; // 저장 간격 (3초)
    }

    // 로그 메시지 추가
    addLogMessage(message) {
        const timestamp = new Date().toLocaleString('ko-KR');
        const logEntry = `[${timestamp}] ${message}`;
        
        this.logMessages.push(logEntry);
        
        // 최대 100개까지만 유지
        if (this.logMessages.length > this.maxLogMessages) {
            this.logMessages.shift();
        }
        
        // 3초마다 한 번씩만 저장
        const currentTime = Date.now();
        if (currentTime - this.lastSaveTime >= this.saveInterval) {
            this.saveLogToFile();
            this.lastSaveTime = currentTime;
        }
    }

    // 로그 파일에 저장
    saveLogToFile() {
        try {
            const logContent = this.logMessages.join('\n');
            
            // 파이썬 서버에 로그 저장 요청
            fetch('/api/explorer-log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    logData: logContent,
                    fileName: 'explorer_movement.log'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`💾 탐색원 로그 저장 완료: ${data.file_path}`);
                } else {
                    console.error(`❌ 탐색원 로그 저장 실패: ${data.error}`);
                    // 폴백: localStorage에 저장
                    localStorage.setItem('explorer_movement_log', logContent);
                }
            })
            .catch(error => {
                console.error(`❌ 탐색원 로그 저장 요청 실패: ${error.message}`);
                // 폴백: localStorage에 저장
                localStorage.setItem('explorer_movement_log', logContent);
            });
        } catch (error) {
            console.error('로그 파일 저장 실패:', error);
            // 폴백: localStorage에 저장
            localStorage.setItem('explorer_movement_log', this.logMessages.join('\n'));
        }
    }

    // 로그 파일 읽기
    loadLogFromFile() {
        try {
            // 파이썬 서버에서 로그 읽기 요청
            fetch('/api/explorer-log?fileName=explorer_movement.log')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.logMessages = data.content.split('\n').filter(line => line.trim());
                    console.log(`📖 탐색원 로그 파일 읽기 완료: ${this.logMessages.length}개 메시지`);
                } else {
                    console.log(`📄 탐색원 로그 파일이 존재하지 않습니다: ${data.error}`);
                    // 폴백: localStorage에서 읽기
                    const content = localStorage.getItem('explorer_movement_log');
                    if (content) {
                        this.logMessages = content.split('\n').filter(line => line.trim());
                    }
                }
            })
            .catch(error => {
                console.error(`❌ 탐색원 로그 파일 읽기 실패: ${error.message}`);
                // 폴백: localStorage에서 읽기
                const content = localStorage.getItem('explorer_movement_log');
                if (content) {
                    this.logMessages = content.split('\n').filter(line => line.trim());
                }
            });
        } catch (error) {
            console.error('로그 파일 읽기 실패:', error);
            // 폴백: localStorage에서 읽기
            const content = localStorage.getItem('explorer_movement_log');
            if (content) {
                this.logMessages = content.split('\n').filter(line => line.trim());
            }
        }
    }

    // 로그 파일 다운로드 (브라우저 환경)
    downloadLogFile() {
        try {
            const logContent = this.logMessages.join('\n');
            const blob = new Blob([logContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'explorer_movement.log';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.addLogMessage(`📥 로그 파일 다운로드 완료`);
        } catch (error) {
            console.error('로그 파일 다운로드 실패:', error);
            this.addLogMessage(`❌ 로그 파일 다운로드 실패: ${error.message}`);
        }
    }

    // 로그 내용 가져오기
    getLogContent() {
        return this.logMessages.join('\n');
    }

    // 로그 초기화
    clearLog() {
        this.logMessages = [];
        this.saveLogToFile();
        this.addLogMessage(`🧹 로그 초기화 완료`);
    }

    // 시스템 초기화
    initialize(scene, config) {
        if (this.isInitialized) {
            this.addLogMessage(`🔍 탐색원 이동 시스템: 이미 초기화됨`);
            return;
        }
        
        this.scene = scene;
        this.config = config;
        this.isInitialized = true;
        
        this.addLogMessage(`🔍 탐색원 이동 시스템 초기화 완료 (scene: ${!!scene}, config: ${!!config})`);
        
        // 기본 탐색원 자동 생성 비활성화
        // if (this.explorers.length === 0) {
        //     this.createDefaultExplorers();
        // }
        
        console.log('🔍 탐색원 이동 시스템 초기화 완료');
    }

    // 기본 탐색원 생성
    createDefaultExplorer(index) {
        if (!this.config) {
            this.addLogMessage(`❌ 탐색원 ${index} 생성 실패: 설정이 없음`);
            return;
        }
        
        // 기본 탐색원 객체 생성
        const defaultExplorer = {
            id: index,
            circle: {
                x: Math.random() * (this.config.width - 80) + 40,
                y: Math.random() * (this.config.height - 80) + 40
            },
            name: { text: `탐색원${index}` },
            role: { text: '탐색자' },
            targetX: Math.random() * (this.config.width - 80) + 40,
            targetY: Math.random() * (this.config.height - 80) + 40,
            discoveredCoords: [],
            isMoving: true,
            lastUpdateTime: Date.now()
        };
        
        // 탐색원 배열에 추가
        this.explorers[index] = defaultExplorer;
        this.addLogMessage(`✅ 탐색원 ${index} 자동 생성 완료: ${defaultExplorer.name.text} - 위치(${Math.round(defaultExplorer.circle.x)}, ${Math.round(defaultExplorer.circle.y)})`);
    }

    // 기본 탐색원들 일괄 생성
    createDefaultExplorers() {
        this.addLogMessage(`🔧 기본 탐색원들 자동 생성 시작`);
        
        // 9개의 기본 탐색원 생성 (0~8)
        for (let i = 0; i < 9; i++) {
            this.createDefaultExplorer(i);
        }
        
        this.addLogMessage(`✅ ${this.explorers.length}개의 기본 탐색원 생성 완료`);
    }

    // 탐색원 등록
    registerExplorers(explorerModels) {
        if (!explorerModels || !Array.isArray(explorerModels)) {
            console.warn('탐색원 등록 실패: 탐색자 데이터가 없습니다.');
            this.addLogMessage(`❌ 탐색원 등록 실패: 탐색자 데이터가 없습니다. (explorerModels: ${!!explorerModels}, isArray: ${Array.isArray(explorerModels)})`);
            // 기본 탐색원들을 자동으로 생성
            this.createDefaultExplorers();
            return;
        }

        this.addLogMessage(`🔍 탐색원 등록 시작: ${explorerModels.length}개의 탐색자 모델`);

        this.explorers = explorerModels.map((explorer, index) => {
            const explorerData = {
                id: index,
                circle: explorer.circle,
                name: explorer.name,
                role: explorer.role,
                targetX: explorer.targetX || Math.random() * (this.config.width - 80) + 40,
                targetY: explorer.targetY || Math.random() * (this.config.height - 80) + 40,
                discoveredCoords: explorer.discoveredCoords || [],
                isMoving: true,
                lastUpdateTime: Date.now()
            };
            
            this.addLogMessage(`🔍 탐색원 ${index} 등록: ${explorer.name?.text || 'Unknown'} - 위치(${Math.round(explorerData.targetX)}, ${Math.round(explorerData.targetY)})`);
            
            return explorerData;
        });

        this.addLogMessage(`🔍 ${this.explorers.length}명의 탐색원이 이동 시스템에 등록되었습니다.`);
        
        // 각 탐색원의 초기 상태 로그
        this.explorers.forEach((explorer, index) => {
            this.addLogMessage(`🔍 탐색원 ${index}: ${explorer.name.text} - 초기위치 (${Math.round(explorer.circle.x)}, ${Math.round(explorer.circle.y)}) | 목표위치 (${Math.round(explorer.targetX)}, ${Math.round(explorer.targetY)})`);
        });
        
        console.log(`🔍 ${this.explorers.length}명의 탐색원이 이동 시스템에 등록되었습니다.`);
    }

    // 개별 탐색원 업데이트
    updateExplorerByIndex(index) {
        // 디버깅: 시스템 상태 확인
        if (!this.isInitialized) {
            this.addLogMessage(`❌ 탐색원 ${index}: 시스템이 초기화되지 않음`);
            return;
        }
        
        if (!this.explorers[index]) {
            // 탐색원이 없으면 자동으로 생성 시도
            this.addLogMessage(`⚠️ 탐색원 ${index}: 탐색원 데이터가 없음 - 자동 생성 시도`);
            this.createDefaultExplorer(index);
            return;
        }
        
        // 디버깅: 탐색원 업데이트 상태 확인 (5초마다)
        if (Math.floor(Date.now() / 1000) % 5 === 0) {
            const explorer = this.explorers[index];
            this.addLogMessage(`🔍 탐색원 ${index} 업데이트: 상태=${explorer.isMoving ? '이동중' : '정지'}, 위치=(${Math.round(explorer.circle.x)}, ${Math.round(explorer.circle.y)})`);
        }
        
        const explorer = this.explorers[index];
        this.updateExplorerMovement(explorer, index);
    }

    // 탐색원 이동 업데이트
    updateExplorerMovement(explorer, index) {
        // 디버깅: 이동 상태 확인
        if (!explorer.isMoving) {
            if (Math.floor(Date.now() / 1000) % 10 === 0) { // 10초마다 로그
                this.addLogMessage(`⏸️ 탐색원 ${index}: 이동이 일시정지됨`);
            }
            return;
        }

        const currentX = explorer.circle.x;
        const currentY = explorer.circle.y;
        const targetX = explorer.targetX;
        const targetY = explorer.targetY;

        // 목표까지의 거리 계산
        const distanceToTarget = Math.sqrt((targetX - currentX) ** 2 + (targetY - currentY) ** 2);

        // 목표에 도달했는지 확인
        if (distanceToTarget < this.arrivalThreshold) {
            this.handleTargetArrival(explorer, index);
        } else {
            // 목표로 이동
            this.moveTowardsTarget(explorer, index);
        }

        // UI 요소들 위치 동기화
        this.syncUIElements(explorer);
    }

    // 목표로 이동
    moveTowardsTarget(explorer, index) {
        const currentX = explorer.circle.x;
        const currentY = explorer.circle.y;
        const targetX = explorer.targetX;
        const targetY = explorer.targetY;

        // 방향 벡터 계산
        const dx = targetX - currentX;
        const dy = targetY - currentY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > 0) {
            // 정규화된 방향 벡터
            const normalizedDx = dx / distance;
            const normalizedDy = dy / distance;

            // 이동 적용 (더 빠른 속도)
            const moveX = normalizedDx * this.movementSpeed * 3; // 속도 3배 증가
            const moveY = normalizedDy * this.movementSpeed * 3;

            explorer.circle.x += moveX;
            explorer.circle.y += moveY;
        }

        // 이동 중에도 N/B 코인 아이템 자동 수집 체크 (50% 확률로 체크 - 속도 증가)
        if (Math.random() < 0.5) {
            this.checkAndCollectNBCoins(explorer, index);
        }

        // 이동 로그 (더 자주 출력하도록 수정)
        if (Math.random() < 0.01) { // 1% 확률로 로그 출력
            const currentPos = `(${Math.round(explorer.circle.x)}, ${Math.round(explorer.circle.y)})`;
            const remainingDistance = Math.round(distance);
            this.addLogMessage(`🚶 탐색원 ${index} 이동 중: ${currentPos} | 남은거리: ${remainingDistance}px`);
        }
    }

    // 목표 도달 처리
    handleTargetArrival(explorer, index) {
        // 새로운 좌표 발견
        const currentCoord = { 
            x: Math.round(explorer.circle.x), 
            y: Math.round(explorer.circle.y) 
        };

        // 중복 체크
        const isDuplicate = explorer.discoveredCoords.some(coord => 
            Math.abs(coord.x - currentCoord.x) < this.explorationRange && 
            Math.abs(coord.y - currentCoord.y) < this.explorationRange
        );

        if (!isDuplicate) {
            explorer.discoveredCoords.push(currentCoord);
            
            if (explorer.discoveredCoords.length > 8) {
                explorer.discoveredCoords.shift();
            }
        }

        // N/B 코인 아이템 자동 수집 체크
        this.checkAndCollectNBCoins(explorer, index);

        // 새로운 목표 설정
        this.setNewTarget(explorer, index);

        // 역할 텍스트 업데이트
        if (explorer.role && typeof explorer.role.setText === 'function') {
            explorer.role.setText(`탐색 (${explorer.discoveredCoords.length}/8)`);
        }

        this.addLogMessage(`🎯 탐색원 ${index}: 새로운 좌표 발견! (${currentCoord.x}, ${currentCoord.y}) - 총 발견: ${explorer.discoveredCoords.length}개`);
    }

    // 새로운 목표 설정
    setNewTarget(explorer, index) {
        // 화면 경계 내에서 랜덤 목표 설정
        const margin = 50;
        
        // config 안전하게 가져오기
        let maxX = 1086; // 기본값
        let maxY = 500;  // 기본값
        
        if (this.scene && this.scene.config) {
            maxX = this.scene.config.width - margin;
            maxY = this.scene.config.height - margin;
        } else if (this.config) {
            maxX = this.config.width - margin;
            maxY = this.config.height - margin;
        } else if (window.gameInitializer && window.gameInitializer.game && window.gameInitializer.game.config) {
            maxX = window.gameInitializer.game.config.width - margin;
            maxY = window.gameInitializer.game.config.height - margin;
        }
        
        explorer.targetX = margin + Math.random() * (maxX - margin);
        explorer.targetY = margin + Math.random() * (maxY - margin);
        
        // 탐색 상태 업데이트
        if (explorer.role && typeof explorer.role.setText === 'function') {
            explorer.role.setText(`탐색 (${explorer.discoveredCoords.length}/8)`);
        }
        
        // 드랍 아이템이 있으면 우선적으로 이동
        if (window.nbCoinDropSystem && window.nbCoinDropSystem.nbCoinItems) {
            const availableCoins = window.nbCoinDropSystem.nbCoinItems.filter(item => !item.collected);
            if (availableCoins.length > 0) {
                // 가장 가까운 드랍 아이템으로 이동
                let nearestCoin = availableCoins[0];
                let nearestDistance = Infinity;
                
                for (const coin of availableCoins) {
                    const distance = Math.sqrt(
                        (explorer.circle.x - coin.position.x) ** 2 + 
                        (explorer.circle.y - coin.position.y) ** 2
                    );
                    
                    if (distance < nearestDistance) {
                        nearestDistance = distance;
                        nearestCoin = coin;
                    }
                }
                
                // 가장 가까운 드랍 아이템으로 목표 설정
                explorer.targetX = nearestCoin.position.x;
                explorer.targetY = nearestCoin.position.y;
                
                if (explorer.role && typeof explorer.role.setText === 'function') {
                    explorer.role.setText(`드랍 아이템 탐색`);
                }
            }
        }
    }

    // UI 요소들 위치 동기화
    syncUIElements(explorer) {
        if (!explorer || !explorer.circle) {
            return;
        }
        
        if (explorer.name && typeof explorer.name.x !== 'undefined' && typeof explorer.name.y !== 'undefined') {
            explorer.name.x = explorer.circle.x;
            explorer.name.y = explorer.circle.y - 6;
        }
        
        if (explorer.role && typeof explorer.role.x !== 'undefined' && typeof explorer.role.y !== 'undefined') {
            explorer.role.x = explorer.circle.x;
            explorer.role.y = explorer.circle.y + 6;
        }
    }

    // N/B 코인 아이템 자동 수집 체크
    checkAndCollectNBCoins(explorer, index) {
        if (!window.nbCoinDropSystem || !window.nbCoinDropSystem.nbCoinItems) {
            return;
        }

        const collectionRange = 80; // 수집 범위 대폭 증가 (50px → 80px)
        const explorerX = explorer.circle.x;
        const explorerY = explorer.circle.y;

        // 수집되지 않은 N/B 코인 아이템들 확인
        const availableCoins = window.nbCoinDropSystem.nbCoinItems.filter(item => !item.collected);

        for (const coin of availableCoins) {
            const distance = Math.sqrt(
                (explorerX - coin.position.x) ** 2 + 
                (explorerY - coin.position.y) ** 2
            );

            if (distance <= collectionRange) {
                // 자동 수집 실행 (누적 수집 제거)
                this.collectNBCoin(coin, explorer, index);
                // 한 번에 여러 개 수집 가능하도록 break 제거
            }
        }
    }

    // N/B 코인 수집 처리 (누적 수집 제거)
    collectNBCoin(coin, explorer, index) {
        if (coin.collected) {
            return;
        }

        coin.collected = true;

        // 수집 애니메이션 (빠른 수집) - scene이 있을 때만 실행
        if (this.scene && this.scene.tweens) {
            this.scene.tweens.add({
                targets: coin.polygon,
                scaleX: 0,
                scaleY: 0,
                alpha: 0,
                duration: 100, // 수집 애니메이션 속도 대폭 증가 (200ms → 100ms)
                ease: 'Power2',
                onComplete: () => {
                    if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.removeNBCoinItem === 'function') {
                        window.nbCoinDropSystem.removeNBCoinItem(coin);
                    }
                }
            });
        } else {
            // scene이 없으면 즉시 제거
            if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.removeNBCoinItem === 'function') {
                window.nbCoinDropSystem.removeNBCoinItem(coin);
            }
        }

        // 수집 효과음 (옵션)
        if (window.soundManager) {
            window.soundManager.playCollectSound();
        }

        // N/B MIN 코인 개수 증가 및 드랍 아이템 카운터 감소
        if (window.gameInitializer && window.gameInitializer.gameData) {
            const previousCoins = window.gameInitializer.gameData.nbCoins || 0;
            
            // N/B MIN 코인 증가
            window.gameInitializer.gameData.nbCoins = previousCoins + 1;
            
            // 드랍 아이템 카운터 감소
            window.gameInitializer.gameData.dropItemsCount = Math.max(0, (window.gameInitializer.gameData.dropItemsCount || 0) - 1);
            
            // 특정 분봉의 N/B 코인 증가 (sourceTimeframe이 있는 경우)
            if (coin.sourceTimeframe && window.nbCoinStatus) {
                // 해당 분봉의 현재 N/B MAX COIN 상태 확인
                let currentNbCoins = 0;
                if (window.cardStorageSystem) {
                    const storage = window.cardStorageSystem.getCardStorage(coin.sourceTimeframe);
                    currentNbCoins = storage.nbCoins || 0;
                }
                
                // N/B MAX COIN이 이미 1 이상이면 추가하지 않음
                if (currentNbCoins >= 1) {
                    console.log(`⚠️ ${coin.sourceTimeframe} 분봉의 N/B MAX COIN이 이미 ${currentNbCoins}개입니다. 추가하지 않습니다.`);
                } else {
                    // N/B MAX COIN이 0일 때만 추가
                    window.nbCoinStatus[coin.sourceTimeframe] = 1;
                    
                    // 카드 저장소 시스템에도 추가
                    if (window.cardStorageSystem && typeof window.cardStorageSystem.addNBCoin === 'function') {
                        window.cardStorageSystem.addNBCoin(coin.sourceTimeframe, 1);
                    }
                    
                    // N/B 미네랄도 추가 (수집 시 1.0% 추가)
                    if (window.cardStorageSystem && typeof window.cardStorageSystem.addNBMineral === 'function') {
                        window.cardStorageSystem.addNBMineral(coin.sourceTimeframe, 1.0);
                    }
                    
                    // 해당 분봉 카드의 N/B 코인 배지 업데이트
                    if (window.nbCoinDropSystem && typeof window.nbCoinDropSystem.updateTimeframeCardNBCoin === 'function') {
                        window.nbCoinDropSystem.updateTimeframeCardNBCoin(coin.sourceTimeframe, 1);
                    }
                    
                    console.log(`💰 탐색원 ${explorer.name ? explorer.name.text : `탐색원${index}`}이(가) N/B 코인 수집 완료: 분봉 ${coin.sourceTimeframe}의 N/B 코인 1개 증가, N/B 미네랄 1.00% 증가`);
                }
            }
            
            // 로그 기록
            if (window.logManager) {
                const timeframeInfo = coin.sourceTimeframe ? `, 분봉 ${coin.sourceTimeframe} N/B MAX COIN 증가` : '';
                window.logManager.addLog(`💰 탐색원 ${explorer.name ? explorer.name.text : `탐색원${index}`}이(가) N/B MIN 코인 수집 완료: 위치 (${Math.round(coin.position.x)}, ${Math.round(coin.position.y)}) → N/B MIN 코인 ${window.gameInitializer.gameData.nbCoins}개 (+1)${timeframeInfo}`);
            }
            
            console.log(`💰 탐색원 ${explorer.name ? explorer.name.text : `탐색원${index}`}이(가) N/B MIN 코인 수집 완료: 위치 (${Math.round(coin.position.x)}, ${Math.round(coin.position.y)}) → N/B MIN 코인 ${window.gameInitializer.gameData.nbCoins}개 (+1)${coin.sourceTimeframe ? `, 분봉 ${coin.sourceTimeframe}` : ''}`);
            
            // N/B MIN 코인 디스플레이 업데이트 (기존 값 유지)
            if (window.nbCoinDisplay && typeof window.nbCoinDisplay.setText === 'function') {
                const nbCoins = window.gameInitializer.gameData.nbCoins;
                const dropItems = window.nbCoinDropSystem ? window.nbCoinDropSystem.nbCoinItems.length : 0;
                window.nbCoinDisplay.setText(`N/B MIN 코인: ${nbCoins}개 (드랍 아이템: ${dropItems}개)`);
            }
            
            // 자동 저장
            if (window.gameInitializer.saveGameData) {
                window.gameInitializer.saveGameData();
            }
        }

        // 탐색원 상태 업데이트
        if (explorer.role && typeof explorer.role.setText === 'function') {
            explorer.role.setText(`수집 완료`);
        }
        
        // 1초 후 탐색 상태로 복귀
        setTimeout(() => {
            if (explorer.role && typeof explorer.role.setText === 'function') {
                explorer.role.setText(`탐색 (${explorer.discoveredCoords.length}/8)`);
            }
        }, 1000);
    }

    // 탐색원 상태 정보 가져오기
    getExplorerStatus() {
        return {
            totalExplorers: this.explorers.length,
            explorers: this.explorers.map(explorer => ({
                name: explorer.name.text,
                position: { x: Math.round(explorer.circle.x), y: Math.round(explorer.circle.y) },
                target: { x: Math.round(explorer.targetX), y: Math.round(explorer.targetY) },
                discoveredCoords: explorer.discoveredCoords.length,
                isMoving: explorer.isMoving
            }))
        };
    }

    // 탐색원 이동 일시정지/재개
    toggleExplorerMovement(index) {
        if (this.explorers[index]) {
            this.explorers[index].isMoving = !this.explorers[index].isMoving;
            const status = this.explorers[index].isMoving ? '재개' : '일시정지';
            
            this.addLogMessage(`⏸️ 탐색원 ${index} 이동 ${status}`);
        }
    }

    // 모든 탐색원 이동 일시정지/재개
    toggleAllExplorerMovement() {
        const allMoving = this.explorers.every(explorer => explorer.isMoving);
        const newStatus = !allMoving;
        
        this.explorers.forEach(explorer => {
            explorer.isMoving = newStatus;
        });

        const status = newStatus ? '재개' : '일시정지';
        this.addLogMessage(`⏸️ 모든 탐색원 이동 ${status}`);
    }

    // 시스템 리셋
    reset() {
        this.explorers = [];
        this.isInitialized = false;
        
        // 모든 탐색원을 N/B 길드 위치 (100, 100)로 리셋
        this.explorers.forEach((explorer, index) => {
            if (explorer.circle) {
                explorer.circle.x = 100;
                explorer.circle.y = 100;
            }
            if (explorer.name) {
                explorer.name.x = 100;
                explorer.name.y = 100;
            }
            if (explorer.role) {
                explorer.role.x = 100;
                explorer.role.y = 100;
            }
            explorer.targetX = 100;
            explorer.targetY = 100;
            explorer.discoveredCoords = [];
            explorer.isMoving = false;
            explorer.lastUpdateTime = Date.now();
        });
        
        this.addLogMessage(`🔍 탐색원 이동 시스템 완전 리셋 - 모든 위치 N/B 길드 (100,100)`);
    }

    // 탐색원 문제 진단
    diagnoseExplorerIssues() {
        this.addLogMessage(`🔍 탐색원 문제 진단 시작`);
        
        // 시스템 초기화 상태 확인
        this.addLogMessage(`🔍 시스템 초기화 상태: ${this.isInitialized}`);
        
        // 탐색원 등록 상태 확인
        this.addLogMessage(`🔍 등록된 탐색원 수: ${this.explorers.length}`);
        
        // 각 탐색원의 상태 확인
        this.explorers.forEach((explorer, index) => {
            this.addLogMessage(`🔍 탐색원 ${index} 상태:`);
            this.addLogMessage(`  - 이름: ${explorer.name?.text || 'Unknown'}`);
            this.addLogMessage(`  - 이동 상태: ${explorer.isMoving ? '이동중' : '정지'}`);
            this.addLogMessage(`  - 현재 위치: (${Math.round(explorer.circle?.x || 0)}, ${Math.round(explorer.circle?.y || 0)})`);
            this.addLogMessage(`  - 목표 위치: (${Math.round(explorer.targetX || 0)}, ${Math.round(explorer.targetY || 0)})`);
            this.addLogMessage(`  - 원 객체 존재: ${!!explorer.circle}`);
            this.addLogMessage(`  - 이름 객체 존재: ${!!explorer.name}`);
            this.addLogMessage(`  - 역할 객체 존재: ${!!explorer.role}`);
        });
        
        // 시스템 설정 확인
        this.addLogMessage(`🔍 시스템 설정:`);
        this.addLogMessage(`  - 이동 속도: ${this.movementSpeed}`);
        this.addLogMessage(`  - 도착 임계값: ${this.arrivalThreshold}`);
        this.addLogMessage(`  - 탐색 범위: ${this.explorationRange}`);
        
        this.addLogMessage(`🔍 탐색원 문제 진단 완료`);
    }

    // 강제 이동 테스트
    forceMoveTest() {
        this.addLogMessage(`🧪 강제 이동 테스트 시작`);
        
        this.explorers.forEach((explorer, index) => {
            if (explorer.circle) {
                // 현재 위치에서 50픽셀 이동
                const newX = explorer.circle.x + 50;
                const newY = explorer.circle.y + 50;
                
                explorer.circle.x = newX;
                explorer.circle.y = newY;
                
                this.addLogMessage(`🧪 탐색원 ${index} 강제 이동: (${Math.round(newX)}, ${Math.round(newY)})`);
            }
        });
        
        this.addLogMessage(`🧪 강제 이동 테스트 완료`);
    }

    // 탐색원 재시작 (개별)
    restartExplorer(model, index) {
        if (!model || !model.circle) {
            console.log(`❌ 탐색원 ${index + 1} 재시작 실패: 모델이 유효하지 않음`);
            return;
        }
        
        try {
            // 탐색원 상태 초기화
            model.explorationTimer = 0;
            model.arrivalLogged = false;
            model.needsNewDecision = false;
            
            // 새로운 목표 위치 설정
            this.setNewTargetForExplorer(model, index);
            
            // 로그 메시지 추가
            this.addLogMessage(`🔍 탐색원 ${index + 1} 재시작 완료 - 새로운 목표: (${model.targetX}, ${model.targetY})`);
            
            console.log(`🔍 탐색원 ${index + 1} 재시작 완료`);
        } catch (error) {
            console.error(`❌ 탐색원 ${index + 1} 재시작 실패:`, error);
        }
    }

    // 전체 탐색 시스템 재시작
    restart() {
        console.log('🔄 탐색 시스템 재시작 시작...');
        
        try {
            // 모든 탐색원 재시작
            if (window.aiModels && Array.isArray(window.aiModels)) {
                window.aiModels.forEach((model, index) => {
                    if (model.isExplorer) {
                        this.restartExplorer(model, index);
                    }
                });
            }
            
            // 시스템 상태 초기화
            this.isInitialized = true;
            
            // 로그 메시지 추가
            this.addLogMessage('🔄 탐색 시스템 전체 재시작 완료');
            
            console.log('✅ 탐색 시스템 재시작 완료');
        } catch (error) {
            console.error('❌ 탐색 시스템 재시작 실패:', error);
        }
    }

    // 탐색원 업데이트 (게임 루프용)
    updateExplorer(model, index) {
        if (!model || !model.circle) return;
        
        try {
            // 이동 처리
            this.moveExplorer(model, index);
            
            // 도착 확인
            this.checkArrival(model, index);
            
            // 새로운 의사결정 필요 여부 확인
            this.checkNewDecisionNeeded(model, index);
        } catch (error) {
            console.error(`❌ 탐색원 ${index + 1} 업데이트 실패:`, error);
        }
    }

    // 탐색원 이동 처리
    moveExplorer(explorer, index) {
        if (!explorer || !explorer.circle) {
            return;
        }

        const currentX = explorer.circle.x;
        const currentY = explorer.circle.y;
        const targetX = explorer.targetX || currentX;
        const targetY = explorer.targetY || currentY;

        // 목표 지점까지의 거리 계산
        const distance = Math.sqrt((targetX - currentX) ** 2 + (targetY - currentY) ** 2);

        // 목표 지점에 도달했는지 확인 (도달 범위 증가)
        if (distance < 40) { // 도달 범위 증가 (25px → 40px)
            // 새로운 좌표 발견
            const currentCoord = { x: Math.round(currentX), y: Math.round(currentY) };
            
            // 중복 체크 (중복 허용 범위 증가)
            const isDuplicate = explorer.discoveredCoords.some(coord => 
                Math.abs(coord.x - currentCoord.x) < 25 && 
                Math.abs(coord.y - currentCoord.y) < 25
            );
            
            if (!isDuplicate) {
                explorer.discoveredCoords.push(currentCoord);
                
                if (explorer.discoveredCoords.length > 8) {
                    explorer.discoveredCoords.shift();
                }
            }
            
            // 새로운 목표 설정
            this.setNewTarget(explorer, index);
        }

        // 이동 속도 증가
        const moveSpeed = 0.08; // 이동 속도 증가 (0.05 → 0.08)
        
        // X축 이동
        if (Math.abs(targetX - currentX) > 1) {
            const dx = (targetX - currentX) * moveSpeed;
            explorer.circle.x += dx;
            
            if (explorer.name && typeof explorer.name.x !== 'undefined') {
                explorer.name.x = explorer.circle.x;
            }
            
            if (explorer.role && typeof explorer.role.x !== 'undefined') {
                explorer.role.x = explorer.circle.x;
            }
        }
        
        // Y축 이동
        if (Math.abs(targetY - currentY) > 1) {
            const dy = (targetY - currentY) * moveSpeed;
            explorer.circle.y += dy;
            
            if (explorer.name && typeof explorer.name.y !== 'undefined') {
                explorer.name.y = explorer.circle.y - 6;
            }
            
            if (explorer.role && typeof explorer.role.y !== 'undefined') {
                explorer.role.y = explorer.circle.y + 6;
            }
        }

        // N/B 코인 아이템 자동 수집 체크
        this.checkAndCollectNBCoins(explorer, index);
    }

    // 도착 확인
    checkArrival(model, index) {
        if (!model.circle) return;
        
        const dx = model.targetX - model.circle.x;
        const dy = model.targetY - model.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance <= this.arrivalThreshold && !model.arrivalLogged) {
            model.arrivalLogged = true;
            model.needsNewDecision = true;
            
            // 도착 로그
            this.addLogMessage(`📍 탐색원 ${index + 1} 목표 도착: (${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`);
            
            // 새로운 목표 설정
            setTimeout(() => {
                this.setNewTargetForExplorer(model, index);
            }, 1000);
        }
    }

    // 새로운 의사결정 필요 여부 확인
    checkNewDecisionNeeded(model, index) {
        if (model.needsNewDecision) {
            // 의사결정 시스템에 새로운 목표 요청
            if (window.explorerDecisionSystem) {
                window.explorerDecisionSystem.makeDecision(model, index);
            }
            model.needsNewDecision = false;
        }
    }

    // 새로운 목표 설정
    setNewTargetForExplorer(model, index) {
        if (!model || !model.circle) return;
        
        try {
            // 현재 위치를 기준으로 새로운 목표 설정
            const currentX = model.circle.x;
            const currentY = model.circle.y;
            
            // 랜덤한 방향으로 새로운 목표 설정
            const angle = Math.random() * 2 * Math.PI;
            const distance = 50 + Math.random() * 100; // 50~150px 거리
            
            const newTargetX = currentX + Math.cos(angle) * distance;
            const newTargetY = currentY + Math.sin(angle) * distance;
            
            // 화면 경계 내로 제한
            let config = { width: 1086, height: 500 }; // 기본값
            
            if (window.gameInitializer && window.gameInitializer.game && window.gameInitializer.game.config) {
                config = window.gameInitializer.game.config;
            } else if (this.config) {
                config = this.config;
            } else if (this.scene && this.scene.config) {
                config = this.scene.config;
            }
            
            const margin = 50;
            
            model.targetX = Math.max(margin, Math.min(config.width - margin, newTargetX));
            model.targetY = Math.max(margin, Math.min(config.height - margin, newTargetY));
            
            // 도착 상태 초기화
            model.arrivalLogged = false;
            
            console.log(`🎯 탐색원 ${index + 1} 새로운 목표 설정: (${Math.round(model.targetX)}, ${Math.round(model.targetY)})`);
        } catch (error) {
            console.error(`❌ 탐색원 ${index + 1} 새로운 목표 설정 실패:`, error);
        }
    }
}

// 전역 인스턴스 생성
window.explorerMovementSystem = new ExplorerMovementSystem();

// 전역 디버깅 함수들
window.debugExplorerSystem = function() {
    if (window.explorerMovementSystem) {
        const status = window.explorerMovementSystem.getExplorerStatus();
        console.log('🔍 탐색원 시스템 상태:', status);
        window.explorerMovementSystem.addLogMessage(`🔍 탐색원 시스템 디버그: ${JSON.stringify(status)}`);
        return status;
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
        return null;
    }
};

window.toggleExplorerMovement = function(index) {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.toggleExplorerMovement(index);
    }
};

window.toggleAllExplorerMovement = function() {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.toggleAllExplorerMovement();
    }
};

// 로그 파일 관리 전역 함수들
window.downloadExplorerLog = function() {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.downloadLogFile();
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
    }
};

window.getExplorerLogContent = function() {
    if (window.explorerMovementSystem) {
        return window.explorerMovementSystem.getLogContent();
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
        return '';
    }
};

window.clearExplorerLog = function() {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.clearLog();
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
    }
};

window.showExplorerLog = function() {
    if (window.explorerMovementSystem) {
        const logContent = window.explorerMovementSystem.getLogContent();
        console.log('🔍 탐색원 로그 내용:');
        console.log(logContent);
        return logContent;
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
        return '';
    }
};

// 로그 통계 정보 가져오기
window.getExplorerLogStats = function() {
    return new Promise((resolve, reject) => {
        fetch('/api/explorer-log-stats?fileName=explorer_movement.log')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('📊 탐색원 로그 통계:', data.stats);
                resolve(data.stats);
            } else {
                console.error('❌ 탐색원 로그 통계 조회 실패:', data.error);
                reject(data.error);
            }
        })
        .catch(error => {
            console.error('❌ 탐색원 로그 통계 요청 실패:', error);
            reject(error);
        });
    });
};

// 진단 및 테스트 전역 함수들
window.diagnoseExplorerIssues = function() {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.diagnoseExplorerIssues();
        console.log('🔍 탐색원 문제 진단 완료 - 로그를 확인하세요');
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
    }
};

window.forceMoveTest = function() {
    if (window.explorerMovementSystem) {
        window.explorerMovementSystem.forceMoveTest();
        console.log('🧪 강제 이동 테스트 완료');
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
    }
};

window.testExplorerMovement = function() {
    if (window.explorerMovementSystem) {
        // 모든 탐색원의 이동을 재개
        window.explorerMovementSystem.explorers.forEach(explorer => {
            explorer.isMoving = true;
        });
        
        // 새로운 목표 설정
        window.explorerMovementSystem.explorers.forEach((explorer, index) => {
            const margin = 40;
            const newTargetX = Math.random() * (window.explorerMovementSystem.config.width - 2 * margin) + margin;
            const newTargetY = Math.random() * (window.explorerMovementSystem.config.height - 2 * margin) + margin;
            
            explorer.targetX = newTargetX;
            explorer.targetY = newTargetY;
            
            window.explorerMovementSystem.addLogMessage(`🧪 탐색원 ${index} 테스트 목표 설정: (${Math.round(newTargetX)}, ${Math.round(newTargetY)})`);
        });
        
        console.log('🧪 탐색원 이동 테스트 시작');
    } else {
        console.log('❌ 탐색원 이동 시스템이 초기화되지 않았습니다.');
    }
};
