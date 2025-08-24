class TrainerDialogSystem {
    constructor() {
        this.dialogText = '';
        this.positionText = '';
        this.learningText = '';
        this.updateInterval = 1000; // 1초마다 업데이트
        this.lastUpdate = 0;
        
        if (window.logManager) {
            window.logManager.addLog(`💬 트레이너 대화 시스템 초기화 완료`);
        }
    }

    // 트레이너 대화창 업데이트
    updateTrainerDialog(model) {
        const currentTime = new Date().toLocaleTimeString();
        const position = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
        const dialogText = `🎯 트레이너: ${model.targetAction} | 위치: ${position} | 시간: ${currentTime}`;
        
        if (window.trainerDialog) {
            window.trainerDialog.setText(dialogText);
        }
        
        // 화면 출력 내용을 로그에 저장
        if (window.logManager) {
            window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
        }
        
        this.dialogText = dialogText;
    }

    // 트레이너 위치 정보 업데이트
    updateTrainerPositionInfo(model, config) {
        const currentPosition = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
        const targetPosition = `(${Math.round(model.targetX || 0)}, ${Math.round(model.targetY || 0)})`;
        const distance = Math.sqrt((model.targetX - model.circle.x) ** 2 + (model.targetY - model.circle.y) ** 2);
        
        // 현재 위치의 장소 확인 (game-initializer로 일원화)
        const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
        const targetZone = window.gameInitializer?.getCurrentZoneName(model.targetX || 0, model.targetY || 0) || '기타영역';
        
        const positionText = `📍 위치: ${currentPosition} (${currentZone}) | 목표: ${targetPosition} (${targetZone}) | 거리: ${Math.round(distance)}px`;
        
        if (window.trainerPositionInfo) {
            window.trainerPositionInfo.setText(positionText);
        }
        
        // 화면 출력 내용을 로그에 저장
        if (window.logManager) {
            window.logManager.addLog(`📺 화면출력(트레이너위치정보): ${positionText}`);
        }
        
        this.positionText = positionText;
    }

    // 학습 상태 업데이트
    updateLearningStatus(model, gameInitializer) {
        const totalDiscovered = gameInitializer.aiModels.reduce((sum, m) => sum + m.discoveredCoords.length, 0);
        const learningText = `AI 시스템 작동 중 - 트레이너: ${model.targetAction} | 발견 좌표: ${totalDiscovered}`;
        
        if (window.learningStatus) {
            window.learningStatus.setText(learningText);
        }
        
        // 화면 출력 내용을 로그에 저장
        if (window.logManager) {
            window.logManager.addLog(`📺 화면출력(학습상태): ${learningText}`);
        }
        
        this.learningText = learningText;
    }

    // 구역 판정은 game-initializer에서만 수행

    // 모든 UI 업데이트
    updateAllUI(model, config, gameInitializer) {
        const now = Date.now();
        
        // 업데이트 간격 제한
        if (now - this.lastUpdate < this.updateInterval) {
            return;
        }
        
        this.lastUpdate = now;
        
        // 각 UI 요소 업데이트
        this.updateTrainerDialog(model);
        this.updateTrainerPositionInfo(model, config);
        this.updateLearningStatus(model, gameInitializer);
    }

    // 특별한 메시지 표시
    showSpecialMessage(message, duration = 5000) {
        const specialDialogText = `🚨 ${message}`;
        
        if (window.trainerDialog) {
            window.trainerDialog.setText(specialDialogText);
        }
        
        if (window.logManager) {
            window.logManager.addLog(`📺 화면출력(특별메시지): ${specialDialogText}`);
        }
        
        // 지정된 시간 후 원래 메시지로 복원
        setTimeout(() => {
            if (this.dialogText && window.trainerDialog) {
                window.trainerDialog.setText(this.dialogText);
            }
        }, duration);
    }

    // 업데이트 간격 조정
    setUpdateInterval(interval) {
        this.updateInterval = Math.max(100, Math.min(10000, interval));
        if (window.logManager) {
            window.logManager.addLog(`⚙️ 트레이너 대화 업데이트 간격 조정: ${this.updateInterval}ms`);
        }
    }

    // 현재 대화 상태 정보 반환
    getDialogStatus() {
        return {
            dialogText: this.dialogText,
            positionText: this.positionText,
            learningText: this.learningText,
            updateInterval: this.updateInterval,
            lastUpdate: this.lastUpdate
        };
    }

    // 대화창 초기화
    initializeDialog() {
        if (window.trainerDialog) {
            window.trainerDialog.setText('🎯 트레이너 시스템 초기화 중...');
        }
        
        if (window.trainerPositionInfo) {
            window.trainerPositionInfo.setText('📍 위치 정보 로딩 중...');
        }
        
        if (window.learningStatus) {
            window.learningStatus.setText('🧠 AI 시스템 초기화 중...');
        }
        
        if (window.logManager) {
            window.logManager.addLog(`💬 트레이너 대화창 초기화 완료`);
        }
    }
}

// 전역 객체로 등록
window.TrainerDialogSystem = TrainerDialogSystem;
