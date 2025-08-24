// 주민 지속성 관리자 모듈
// 주민 수집 시스템의 상태를 localStorage에 저장하고 새로 고침 시에도 유지되도록 하는 기능을 담당

class ResidentPersistenceManager {
    constructor() {
        this.storageKey = 'resident_collection_system_data';
        this.autoSaveInterval = null;
        this.saveIntervalMs = 10000; // 10초마다 자동 저장
        this.lastSaveTime = Date.now();
    }

    // 시스템 초기화
    initialize() {
        //console.log('💾 주민 지속성 관리자 초기화 완료');
        
        // 자동 저장 시작
        this.startAutoSave();
        
        if (window.logManager) {
            window.logManager.addLog(`💾 주민 지속성 관리자 초기화 완료 (자동 저장: ${this.saveIntervalMs/1000}초 간격)`);
        }
    }

    // 주민 수집 시스템 데이터 저장
    saveResidentData(residentCollectionSystem) {
        if (!residentCollectionSystem || !residentCollectionSystem.residents) {
            console.warn('주민 데이터 저장 실패: 시스템이 초기화되지 않았습니다.');
            return;
        }

        try {
            const saveData = {
                timestamp: Date.now(),
                collectedCoins: residentCollectionSystem.collectedCoins,
                warehouseCapacity: residentCollectionSystem.warehouseCapacity,
                residents: residentCollectionSystem.residents.map(resident => ({
                    id: resident.id,
                    name: resident.name ? resident.name.text : `Resident-${resident.id}`,
                    targetX: resident.targetX,
                    targetY: resident.targetY,
                    discoveredCoords: resident.discoveredCoords || [],
                    collectedCoins: resident.collectedCoins,
                    isCarryingCoin: resident.isCarryingCoin,
                    state: resident.state,
                    collectionTimer: resident.collectionTimer,
                    deliveryTimer: resident.deliveryTimer,
                    deliveryTarget: resident.deliveryTarget
                }))
            };

            localStorage.setItem(this.storageKey, JSON.stringify(saveData));
            this.lastSaveTime = Date.now();

            if (window.logManager) {
                window.logManager.addLog(`💾 주민 수집 시스템 데이터 저장 완료 (수집된 코인: ${saveData.collectedCoins}개, 주민 수: ${saveData.residents.length}명)`);
            }

            //console.log(`💾 주민 수집 시스템 데이터 저장 완료 (수집된 코인: ${saveData.collectedCoins}개, 주민 수: ${saveData.residents.length}명)`);
        } catch (error) {
            console.error('❌ 주민 데이터 저장 중 오류:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 주민 데이터 저장 중 오류: ${error.message}`);
            }
        }
    }

    // 주민 수집 시스템 데이터 로드
    loadResidentData() {
        try {
            const savedData = localStorage.getItem(this.storageKey);
            if (!savedData) {
                //console.log('📂 저장된 주민 데이터가 없습니다. 새로 시작합니다.');
                return null;
            }

            const parsedData = JSON.parse(savedData);
            
            // 데이터 유효성 검사
            if (!parsedData || !parsedData.residents || !Array.isArray(parsedData.residents)) {
                console.warn('📂 저장된 주민 데이터가 유효하지 않습니다.');
                return null;
            }

            if (window.logManager) {
                window.logManager.addLog(`📂 주민 수집 시스템 데이터 로드 완료 (수집된 코인: ${parsedData.collectedCoins}개, 주민 수: ${parsedData.residents.length}명)`);
            }

            //console.log(`📂 주민 수집 시스템 데이터 로드 완료 (수집된 코인: ${parsedData.collectedCoins}개, 주민 수: ${parsedData.residents.length}명)`);
            return parsedData;
        } catch (error) {
            console.error('❌ 주민 데이터 로드 중 오류:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 주민 데이터 로드 중 오류: ${error.message}`);
            }
            return null;
        }
    }

    // 주민 수집 시스템에 저장된 데이터 적용
    applySavedData(residentCollectionSystem, savedData) {
        if (!residentCollectionSystem || !savedData) {
            return false;
        }

        try {
            // 기본 데이터 복원
            residentCollectionSystem.collectedCoins = savedData.collectedCoins || 0;
            residentCollectionSystem.warehouseCapacity = savedData.warehouseCapacity || 100;

            // 주민 데이터 복원
            if (savedData.residents && Array.isArray(savedData.residents)) {
                savedData.residents.forEach(savedResident => {
                    const existingResident = residentCollectionSystem.residents.find(r => r.id === savedResident.id);
                    if (existingResident) {
                        // 위치 및 상태 복원
                        existingResident.targetX = savedResident.targetX || existingResident.targetX;
                        existingResident.targetY = savedResident.targetY || existingResident.targetY;
                        existingResident.discoveredCoords = savedResident.discoveredCoords || [];
                        existingResident.collectedCoins = savedResident.collectedCoins || 0;
                        existingResident.isCarryingCoin = savedResident.isCarryingCoin || false;
                        existingResident.state = savedResident.state || 'exploring';
                        existingResident.collectionTimer = savedResident.collectionTimer || 0;
                        existingResident.deliveryTimer = savedResident.deliveryTimer || 0;
                        existingResident.deliveryTarget = savedResident.deliveryTarget || null;

                        // UI 텍스트 업데이트
                        if (existingResident.role && typeof existingResident.role.setText === 'function') {
                            if (existingResident.state === 'collecting') {
                                existingResident.role.setText('수집 중');
                            } else if (existingResident.state === 'delivering') {
                                existingResident.role.setText('전달 중');
                            } else {
                                existingResident.role.setText(`탐색 (${existingResident.discoveredCoords.length}/8)`);
                            }
                        }
                    }
                });
            }

            if (window.logManager) {
                window.logManager.addLog(`🔄 주민 수집 시스템 데이터 복원 완료 (수집된 코인: ${residentCollectionSystem.collectedCoins}개)`);
            }

            //console.log(`🔄 주민 수집 시스템 데이터 복원 완료 (수집된 코인: ${residentCollectionSystem.collectedCoins}개)`);
            return true;
        } catch (error) {
            console.error('❌ 주민 데이터 복원 중 오류:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 주민 데이터 복원 중 오류: ${error.message}`);
            }
            return false;
        }
    }

    // 자동 저장 시작
    startAutoSave() {
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
        }

        this.autoSaveInterval = setInterval(() => {
            if (window.residentCollectionSystem && window.residentCollectionSystem.isInitialized) {
                this.saveResidentData(window.residentCollectionSystem);
            }
        }, this.saveIntervalMs);

        //console.log(`💾 주민 자동 저장 시작 (${this.saveIntervalMs/1000}초 간격)`);
    }

    // 자동 저장 중지
    stopAutoSave() {
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = null;
            //console.log('💾 주민 자동 저장 중지');
        }
    }

    // 저장된 데이터 삭제
    clearSavedData() {
        try {
            localStorage.removeItem(this.storageKey);
            //console.log('🗑️ 주민 수집 시스템 저장 데이터 삭제 완료');
            
            if (window.logManager) {
                window.logManager.addLog('🗑️ 주민 수집 시스템 저장 데이터 삭제 완료');
            }
        } catch (error) {
            console.error('❌ 주민 데이터 삭제 중 오류:', error);
        }
    }

    // 저장된 데이터 정보 가져오기
    getSavedDataInfo() {
        try {
            const savedData = localStorage.getItem(this.storageKey);
            if (!savedData) {
                return {
                    exists: false,
                    message: '저장된 데이터가 없습니다.'
                };
            }

            const parsedData = JSON.parse(savedData);
            const saveTime = new Date(parsedData.timestamp);
            
            return {
                exists: true,
                timestamp: parsedData.timestamp,
                saveTime: saveTime.toLocaleString(),
                collectedCoins: parsedData.collectedCoins || 0,
                residentCount: parsedData.residents ? parsedData.residents.length : 0,
                message: `${saveTime.toLocaleString()}에 저장됨 (수집된 코인: ${parsedData.collectedCoins || 0}개, 주민 수: ${parsedData.residents ? parsedData.residents.length : 0}명)`
            };
        } catch (error) {
            return {
                exists: false,
                message: `데이터 읽기 오류: ${error.message}`
            };
        }
    }

    // 시스템 정리
    cleanup() {
        this.stopAutoSave();
        //console.log('🧹 주민 지속성 관리자 정리 완료');
    }

    // 시스템 재시작
    restart() {
        try {
            //console.log('🔄 주민 지속성 관리자 재시작 시작...');
            
            // 자동 저장 중지
            this.stopAutoSave();
            
            // 자동 저장 재시작
            this.startAutoSave();
            
            // 마지막 저장 시간 초기화
            this.lastSaveTime = Date.now();
            
            if (window.logManager) {
                window.logManager.addLog('🔄 주민 지속성 관리자 재시작 완료');
            }
            
            //console.log('✅ 주민 지속성 관리자 재시작 완료');
        } catch (error) {
            console.error('❌ 주민 지속성 관리자 재시작 실패:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 주민 지속성 관리자 재시작 실패: ${error.message}`);
            }
        }
    }
}

// 전역 인스턴스 생성
window.residentPersistenceManager = new ResidentPersistenceManager();
