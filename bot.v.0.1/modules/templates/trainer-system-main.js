// Trainer System Main Module
// 모든 트레이너 관련 모듈을 초기화하고 조정하는 메인 시스템

class TrainerSystemMain {
    constructor() {
        this.isInitialized = false;
        this.modules = {
            decisionHandler: null,
            movementController: null,
            dialogSystem: null
        };
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 시스템 메인 초기화 시작`);
        }
    }

    // 트레이너 시스템 초기화
    initialize() {
        try {
            // 1. 트레이너 의사결정 핸들러 초기화
            if (window.TrainerDecisionHandler) {
                this.modules.decisionHandler = new window.TrainerDecisionHandler();
                window.trainerDecisionHandler = this.modules.decisionHandler;
                
                if (window.logManager) {
                    window.logManager.addLog(`✅ 트레이너 의사결정 핸들러 초기화 완료`);
                }
            } else {
                throw new Error('TrainerDecisionHandler 클래스를 찾을 수 없습니다');
            }

            // 2. 트레이너 이동 컨트롤러 초기화
            if (window.TrainerMovementController) {
                this.modules.movementController = new window.TrainerMovementController();
                window.trainerMovementController = this.modules.movementController;
                
                if (window.logManager) {
                    window.logManager.addLog(`✅ 트레이너 이동 컨트롤러 초기화 완료`);
                }
            } else {
                throw new Error('TrainerMovementController 클래스를 찾을 수 없습니다');
            }

            // 3. 트레이너 대화 시스템 초기화
            if (window.TrainerDialogSystem) {
                this.modules.dialogSystem = new window.TrainerDialogSystem();
                window.trainerDialogSystem = this.modules.dialogSystem;
                
                // 대화창 초기화
                this.modules.dialogSystem.initializeDialog();
                
                if (window.logManager) {
                    window.logManager.addLog(`✅ 트레이너 대화 시스템 초기화 완료`);
                }
            } else {
                throw new Error('TrainerDialogSystem 클래스를 찾을 수 없습니다');
            }

            this.isInitialized = true;
            
            if (window.logManager) {
                window.logManager.addLog(`🎯 트레이너 시스템 메인 초기화 완료 - 모든 모듈 준비됨`);
            }
            
            return true;
            
        } catch (error) {
            if (window.logManager) {
                window.logManager.addLog(`❌ 트레이너 시스템 초기화 실패: ${error.message}`);
            }
            console.error('Trainer System Initialization Error:', error);
            return false;
        }
    }

    // 트레이너 시스템 상태 확인
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            modules: {
                decisionHandler: !!this.modules.decisionHandler,
                movementController: !!this.modules.movementController,
                dialogSystem: !!this.modules.dialogSystem
            },
            globalObjects: {
                trainerDecisionHandler: !!window.trainerDecisionHandler,
                trainerMovementController: !!window.trainerMovementController,
                trainerDialogSystem: !!window.trainerDialogSystem
            }
        };
    }

    // 트레이너 시스템 재시작
    restart() {
        if (window.logManager) {
            window.logManager.addLog(`🔄 트레이너 시스템 재시작 중...`);
        }
        
        // 기존 모듈 정리
        this.cleanup();
        
        // 재초기화
        const success = this.initialize();
        
        if (success) {
            if (window.logManager) {
                window.logManager.addLog(`✅ 트레이너 시스템 재시작 완료`);
            }
        } else {
            if (window.logManager) {
                window.logManager.addLog(`❌ 트레이너 시스템 재시작 실패`);
            }
        }
        
        return success;
    }

    // 트레이너 시스템 정리
    cleanup() {
        // 전역 객체 정리
        if (window.trainerDecisionHandler) {
            delete window.trainerDecisionHandler;
        }
        if (window.trainerMovementController) {
            delete window.trainerMovementController;
        }
        if (window.trainerDialogSystem) {
            delete window.trainerDialogSystem;
        }
        
        // 모듈 참조 정리
        this.modules = {
            decisionHandler: null,
            movementController: null,
            dialogSystem: null
        };
        
        this.isInitialized = false;
        
        if (window.logManager) {
            window.logManager.addLog(`🧹 트레이너 시스템 정리 완료`);
        }
    }

    // 트레이너 시스템 설정 업데이트
    updateSettings(settings) {
        if (!this.isInitialized) {
            if (window.logManager) {
                window.logManager.addLog(`⚠️ 트레이너 시스템이 초기화되지 않아 설정을 업데이트할 수 없습니다`);
            }
            return false;
        }
        
        try {
            // 이동 속도 설정
            if (settings.movementSpeed && this.modules.movementController) {
                this.modules.movementController.setMovementSpeed(settings.movementSpeed);
            }
            
            // 도착 임계값 설정
            if (settings.arrivalThreshold && this.modules.movementController) {
                this.modules.movementController.setArrivalThreshold(settings.arrivalThreshold);
            }
            
            // 대화 업데이트 간격 설정
            if (settings.dialogUpdateInterval && this.modules.dialogSystem) {
                this.modules.dialogSystem.setUpdateInterval(settings.dialogUpdateInterval);
            }
            
            if (window.logManager) {
                window.logManager.addLog(`⚙️ 트레이너 시스템 설정 업데이트 완료`);
            }
            
            return true;
            
        } catch (error) {
            if (window.logManager) {
                window.logManager.addLog(`❌ 트레이너 시스템 설정 업데이트 실패: ${error.message}`);
            }
            return false;
        }
    }

    // 트레이너 시스템 디버그 정보 반환
    getDebugInfo() {
        const status = this.getStatus();
        const debugInfo = {
            systemStatus: status,
            moduleDetails: {}
        };
        
        // 각 모듈의 상세 정보 수집
        if (this.modules.decisionHandler) {
            debugInfo.moduleDetails.decisionHandler = {
                zoneSteps: this.modules.decisionHandler.zoneSteps,
                currentZone: this.modules.decisionHandler.currentZone
            };
        }
        
        if (this.modules.movementController) {
            debugInfo.moduleDetails.movementController = {
                movementSpeed: this.modules.movementController.movementSpeed,
                arrivalThreshold: this.modules.movementController.arrivalThreshold,
                targetX: this.modules.movementController.targetX,
                targetY: this.modules.movementController.targetY
            };
        }
        
        if (this.modules.dialogSystem) {
            debugInfo.moduleDetails.dialogSystem = this.modules.dialogSystem.getDialogStatus();
        }
        
        return debugInfo;
    }

    // 트레이너 시스템 테스트
    test() {
        if (!this.isInitialized) {
            return { success: false, message: '시스템이 초기화되지 않았습니다' };
        }
        
        const testResults = {
            success: true,
            tests: {}
        };
        
        try {
            // 의사결정 핸들러 테스트
            if (this.modules.decisionHandler) {
                testResults.tests.decisionHandler = 'OK';
            } else {
                testResults.tests.decisionHandler = 'FAIL';
                testResults.success = false;
            }
            
            // 이동 컨트롤러 테스트
            if (this.modules.movementController) {
                testResults.tests.movementController = 'OK';
            } else {
                testResults.tests.movementController = 'FAIL';
                testResults.success = false;
            }
            
            // 대화 시스템 테스트
            if (this.modules.dialogSystem) {
                testResults.tests.dialogSystem = 'OK';
            } else {
                testResults.tests.dialogSystem = 'FAIL';
                testResults.success = false;
            }
            
            if (window.logManager) {
                window.logManager.addLog(`🧪 트레이너 시스템 테스트 완료: ${testResults.success ? '성공' : '실패'}`);
            }
            
        } catch (error) {
            testResults.success = false;
            testResults.error = error.message;
            
            if (window.logManager) {
                window.logManager.addLog(`❌ 트레이너 시스템 테스트 실패: ${error.message}`);
            }
        }
        
        return testResults;
    }
}

// 전역 객체로 등록
window.TrainerSystemMain = TrainerSystemMain;

// 자동 초기화 (페이지 로드 시)
document.addEventListener('DOMContentLoaded', function() {
    if (window.TrainerSystemMain) {
        window.trainerSystemMain = new window.TrainerSystemMain();
        window.trainerSystemMain.initialize();
    }
});
