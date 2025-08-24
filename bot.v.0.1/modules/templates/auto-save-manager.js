// 자동 저장 관리자 모듈
// 화면의 모든 정보를 주기적으로 자동 저장하는 기능을 담당

class AutoSaveManager {
    constructor() {
        this.autoSaveInterval = 30000; // 30초마다 자동 저장
        this.lastSaveTime = 0;
        this.isAutoSaveEnabled = true;
        this.saveHistory = [];
        this.maxSaveHistory = 100; // 최대 저장 히스토리 개수
        this.autoSaveTimer = null;
        
        // 페이지 로드 시 자동 복원
        this.initializeAutoRestore();
        this.initializeAutoSave();
    }

    // 자동 복원 초기화 (페이지 로드 시 실행)
    initializeAutoRestore() {
        // DOM이 로드된 후 복원 실행
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => this.restoreGameScreenData(), 1000); // 1초 후 복원
            });
        } else {
            setTimeout(() => this.restoreGameScreenData(), 1000); // 1초 후 복원
        }
        console.log('🔄 자동 복원 시스템 초기화 완료');
    }

    // 자동 저장 초기화
    initializeAutoSave() {
        if (this.isAutoSaveEnabled) {
            this.startAutoSave();
            console.log('🔄 자동 저장 시스템 초기화 완료 (30초 간격)');
        }
    }

    // 자동 저장 시작
    startAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
        }
        
        this.autoSaveTimer = setInterval(() => {
            this.performAutoSave();
        }, this.autoSaveInterval);
        
        console.log('✅ 자동 저장 시작');
    }

    // 자동 저장 중지
    stopAutoSave() {
        if (this.autoSaveTimer) {
            clearInterval(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
        console.log('⏹️ 자동 저장 중지');
    }

    // 자동 저장 실행
    async performAutoSave() {
        try {
            const currentTime = Date.now();
            const saveData = this.collectAllData();
            
            // 저장 실행
            const saveResult = await this.saveDataToFile(saveData);
            
            if (saveResult.success) {
                this.lastSaveTime = currentTime;
                this.addToSaveHistory({
                    timestamp: currentTime,
                    type: 'auto_save',
                    success: true,
                    dataSize: JSON.stringify(saveData).length
                });
                
                console.log('💾 자동 저장 완료:', {
                    timestamp: new Date(currentTime).toLocaleString(),
                    dataSize: saveResult.dataSize,
                    filePath: saveResult.filePath
                });
                
                if (window.logManager) {
                    window.logManager.addLog(`💾 자동 저장 완료: ${saveResult.dataSize} bytes`);
                }
            } else {
                console.error('❌ 자동 저장 실패:', saveResult.error);
                if (window.logManager) {
                    window.logManager.addLog(`❌ 자동 저장 실패: ${saveResult.error}`);
                }
            }
            
        } catch (error) {
            console.error('❌ 자동 저장 중 오류 발생:', error);
            if (window.logManager) {
                window.logManager.addLog(`❌ 자동 저장 오류: ${error.message}`);
            }
        }
    }

    // 게임 화면 데이터만 수집 (Active Signals 패널 중심)
    collectAllData() {
        const currentTime = new Date().toISOString();
        
        return {
            timestamp: currentTime,
            version: '1.0.0',
            gameScreenData: this.collectGameScreenData()
        };
    }

    // 게임 화면 데이터만 수집 (Active Signals 패널 중심)
    collectGameScreenData() {
        try {
            const gameScreenData = {
                // Active Signals 패널 정보
                signalsPanel: {
                    title: 'Active Signals',
                    hasFloatingBallGame: !!document.getElementById('floating-ball-game'),
                    gameCanvas: {
                        width: document.querySelector('#floating-ball-game canvas')?.width || 0,
                        height: document.querySelector('#floating-ball-game canvas')?.height || 0
                    },
                    signalsGrid: {
                        elementCount: document.querySelector('.signals-grid')?.children?.length || 0,
                        signals: this.collectSignalsData()
                    },
                    timeframeZones: {
                        elementCount: document.querySelector('#timeframe-zones-container')?.children?.length || 0,
                        zones: this.collectTimeframeZonesData()
                    },
                    messageDisplay: {
                        text: document.getElementById('message-text')?.textContent || 'AI 시스템 작동 중...',
                        isVisible: document.getElementById('message-display')?.style.display !== 'none'
                    }
                },
                
                // 게임 컨트롤 버튼 상태
                gameControls: {
                    resetButton: {
                        exists: !!document.getElementById('game-reset-button'),
                        text: document.getElementById('game-reset-button')?.textContent || '🔄 초기화'
                    },
                    nbCoinPlusButton: {
                        exists: !!document.getElementById('nb-coin-plus-button'),
                        text: document.getElementById('nb-coin-plus-button')?.textContent || '➕ N/B MIN 코인 +1'
                    },
                    nbCoinMinusButton: {
                        exists: !!document.getElementById('nb-coin-minus-button'),
                        text: document.getElementById('nb-coin-minus-button')?.textContent || '➖ N/B MIN 코인 -1'
                    }
                },
                
                // 기본 게임 정보
                currentMajority: document.getElementById('majority-zone')?.textContent || '-',
                orangeSum: document.getElementById('orange-sum')?.textContent || '0',
                blueSum: document.getElementById('blue-sum')?.textContent || '0'
            };

            // Phaser 게임 상태 (있는 경우)
            if (window.game && window.game.scene) {
                const scene = window.game.scene.getScene('GameScene');
                if (scene) {
                    gameScreenData.phaserState = {
                        hasScene: true,
                        sceneName: scene.scene.key,
                        isActive: scene.scene.isActive()
                    };
                }
            }

            return gameScreenData;
        } catch (error) {
            console.warn('⚠️ 게임 화면 데이터 수집 중 오류:', error);
            return { error: error.message };
        }
    }

    // 신호 데이터 수집
    collectSignalsData() {
        try {
            const signals = [];
            const signalElements = document.querySelectorAll('.signals-grid .signal-card');
            
            signalElements.forEach((element, index) => {
                signals.push({
                    index: index,
                    text: element.textContent || '',
                    className: element.className || '',
                    style: element.style.cssText || ''
                });
            });
            
            return signals;
        } catch (error) {
            console.warn('⚠️ 신호 데이터 수집 중 오류:', error);
            return [];
        }
    }

    // 타임프레임 구역 데이터 수집
    collectTimeframeZonesData() {
        try {
            const zones = [];
            const zoneElements = document.querySelectorAll('#timeframe-zones-container .timeframe-zone');
            
            zoneElements.forEach((element, index) => {
                zones.push({
                    index: index,
                    text: element.textContent || '',
                    className: element.className || '',
                    style: element.style.cssText || ''
                });
            });
            
            return zones;
        } catch (error) {
            console.warn('⚠️ 타임프레임 구역 데이터 수집 중 오류:', error);
            return [];
        }
    }

    // 게임 화면 데이터 복원 (새로 고침 시 자동 실행)
    restoreGameScreenData() {
        try {
            const savedData = localStorage.getItem('gameScreenAutoSave');
            if (!savedData) {
                console.log('📂 저장된 게임 화면 데이터가 없습니다.');
                return false;
            }

            const data = JSON.parse(savedData);
            console.log('📂 저장된 게임 화면 데이터 복원 중...', data);

            // 메시지 디스플레이 복원
            if (data.gameScreenData?.signalsPanel?.messageDisplay) {
                const messageText = document.getElementById('message-text');
                if (messageText) {
                    messageText.textContent = data.gameScreenData.signalsPanel.messageDisplay.text;
                }
            }

            // 신호 그리드 복원
            if (data.gameScreenData?.signalsPanel?.signalsGrid?.signals) {
                this.restoreSignalsGrid(data.gameScreenData.signalsPanel.signalsGrid.signals);
            }

            // 타임프레임 구역 복원
            if (data.gameScreenData?.signalsPanel?.timeframeZones?.zones) {
                this.restoreTimeframeZones(data.gameScreenData.signalsPanel.timeframeZones.zones);
            }

            console.log('✅ 게임 화면 데이터 복원 완료');
            return true;

        } catch (error) {
            console.error('❌ 게임 화면 데이터 복원 실패:', error);
            return false;
        }
    }

    // 신호 그리드 복원
    restoreSignalsGrid(signals) {
        try {
            const signalsGrid = document.querySelector('.signals-grid');
            if (!signalsGrid) return;

            // 기존 신호들 제거
            signalsGrid.innerHTML = '';

            // 저장된 신호들 복원
            signals.forEach(signal => {
                const signalElement = document.createElement('div');
                signalElement.className = signal.className || 'signal-card';
                signalElement.textContent = signal.text;
                signalElement.style.cssText = signal.style;
                signalsGrid.appendChild(signalElement);
            });

            console.log(`✅ ${signals.length}개의 신호 복원 완료`);
        } catch (error) {
            console.warn('⚠️ 신호 그리드 복원 중 오류:', error);
        }
    }

    // 타임프레임 구역 복원
    restoreTimeframeZones(zones) {
        try {
            const zonesContainer = document.getElementById('timeframe-zones-container');
            if (!zonesContainer) return;

            // 기존 구역들 제거
            zonesContainer.innerHTML = '';

            // 저장된 구역들 복원
            zones.forEach(zone => {
                const zoneElement = document.createElement('div');
                zoneElement.className = zone.className || 'timeframe-zone';
                zoneElement.textContent = zone.text;
                zoneElement.style.cssText = zone.style;
                zonesContainer.appendChild(zoneElement);
            });

            console.log(`✅ ${zones.length}개의 타임프레임 구역 복원 완료`);
        } catch (error) {
            console.warn('⚠️ 타임프레임 구역 복원 중 오류:', error);
        }
    }

    // 데이터를 localStorage에 저장 (새로 고침 시 복원용)
    async saveDataToFile(data) {
        try {
            const timestamp = new Date().toISOString();
            const dataString = JSON.stringify(data, null, 2);
            
            // localStorage에 저장
            localStorage.setItem('gameScreenAutoSave', dataString);
            localStorage.setItem('gameScreenAutoSaveTimestamp', timestamp);
            
            return {
                success: true,
                filePath: 'localStorage',
                dataSize: dataString.length,
                timestamp: timestamp
            };
            
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    // 저장 히스토리에 추가
    addToSaveHistory(saveInfo) {
        this.saveHistory.push(saveInfo);
        
        // 최대 히스토리 개수 제한
        if (this.saveHistory.length > this.maxSaveHistory) {
            this.saveHistory.shift();
        }
    }

    // 수동 저장 실행
    async manualSave() {
        console.log('💾 수동 저장 시작...');
        await this.performAutoSave();
    }

    // 자동 저장 설정 변경
    setAutoSaveInterval(intervalMs) {
        this.autoSaveInterval = intervalMs;
        if (this.isAutoSaveEnabled) {
            this.stopAutoSave();
            this.startAutoSave();
        }
        console.log(`🔄 자동 저장 간격 변경: ${intervalMs / 1000}초`);
    }

    // 자동 저장 활성화/비활성화
    setAutoSaveEnabled(enabled) {
        this.isAutoSaveEnabled = enabled;
        if (enabled) {
            this.startAutoSave();
        } else {
            this.stopAutoSave();
        }
        console.log(`🔄 자동 저장 ${enabled ? '활성화' : '비활성화'}`);
    }

    // 저장 히스토리 조회
    getSaveHistory() {
        return this.saveHistory;
    }

    // 마지막 저장 시간 조회
    getLastSaveTime() {
        return this.lastSaveTime;
    }

    // 자동 저장 상태 조회
    getAutoSaveStatus() {
        return {
            isEnabled: this.isAutoSaveEnabled,
            interval: this.autoSaveInterval,
            lastSaveTime: this.lastSaveTime,
            saveHistoryCount: this.saveHistory.length,
            nextSaveTime: this.lastSaveTime + this.autoSaveInterval
        };
    }

    // 저장된 데이터 복원 (필요시)
    async restoreFromFile(file) {
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            console.log('📂 저장된 데이터 복원:', data);
            return { success: true, data: data };
        } catch (error) {
            console.error('❌ 데이터 복원 실패:', error);
            return { success: false, error: error.message };
        }
    }

    // localStorage에서 저장된 데이터 삭제
    clearSavedData() {
        try {
            localStorage.removeItem('gameScreenAutoSave');
            localStorage.removeItem('gameScreenAutoSaveTimestamp');
            console.log('🗑️ 저장된 게임 화면 데이터 삭제 완료');
            return { success: true };
        } catch (error) {
            console.error('❌ 저장된 데이터 삭제 실패:', error);
            return { success: false, error: error.message };
        }
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.autoSaveManager = new AutoSaveManager();
}

// 모듈 로딩 완료
