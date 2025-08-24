// 트레이너 이동 관리자 모듈
// 트레이너의 이동, 상태 관리, 위치 계산을 담당

class TrainerMovementManager {
    constructor() {
        this.movementSpeed = 2;
        this.arrivalThreshold = 30;
    }

    // 트레이너 이동 처리
    updateTrainerMovement(model, config) {
        // TrainerStateHandler가 이미 처리하고 있으므로 이 함수는 비활성화
        // 중복 이동 처리를 방지하기 위해 아무것도 하지 않음
        if (window.logManager && Math.floor(Date.now() / 1000) % 10 === 0) {
            window.logManager.addLog(`⚠️ TrainerMovementManager.updateTrainerMovement 비활성화: TrainerStateHandler가 처리 중`);
        }
        
        return false; // 이동 처리하지 않음
    }

    // 목표 도달 처리
    handleArrival(model, config) {
        // 목표 도달 로그 (중복 방지)
        if (!model.arrivalLogged) {
            let arrivalMessage = '';
            if (model.targetAction === 'BTC 시장 방문') {
                arrivalMessage = `🎯 트레이너: BTC 시장에 도착! 매수 전 예상 수익률 계산 시작...`;
            } else if (model.targetAction === 'N/B 길드 방문') {
                arrivalMessage = `🎯 트레이너: N/B 길드에 도착! 매도 전 예상 수익률 계산 시작...`;
            } else if (model.targetAction === '신호 대기') {
                arrivalMessage = `🎯 트레이너: 신호 대기 센터에 도착! 신호 대기 시작...`;
            }
            
            if (arrivalMessage && window.logManager) {
                window.logManager.addLog(arrivalMessage);
            }
            model.arrivalLogged = true;
        }
        
        // 역할 텍스트 업데이트
        model.role.setText(`트레이너 (${model.targetAction})`);
    }

    // 거리 계산 유틸리티
    calculateDistance(x1, y1, x2, y2) {
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    }

    // 신호 대기 센터까지의 거리 계산
    getDistanceToSignalCenter(model, config) {
        return this.calculateDistance(
            model.circle.x, 
            model.circle.y, 
            config.width / 2, 
            config.height / 2
        );
    }

    // BTC 시장까지의 거리 계산
    getDistanceToBTCMarket(model, config) {
        return this.calculateDistance(
            model.circle.x, 
            model.circle.y, 
            config.width - 100, 
            config.height - 100
        );
    }

    // N/B 길드까지의 거리 계산
    getDistanceToNBGuild(model, config) {
        return this.calculateDistance(
            model.circle.x, 
            model.circle.y, 
            100, 
            config.height - 100
        );
    }

    // 충돌 검사
    checkCollision(model, targetObject) {
        if (!targetObject || !model.circle) return false;
        
        const circleBounds = model.circle.getBounds();
        const targetBounds = targetObject.getBounds();
        
        return Phaser.Geom.Rectangle.Overlaps(circleBounds, targetBounds);
    }

    // BTC 시장 충돌 검사
    checkBTCMarketCollision(model) {
        if (!window.btcMarketPolygon) return false;
        
        let isColliding = this.checkCollision(model, window.btcMarketPolygon);
        
        // 더 정확한 검사를 위해 거리도 확인
        if (isColliding) {
            const centerDistance = this.calculateDistance(
                model.circle.x, 
                model.circle.y, 
                window.btcMarketPolygon.x, 
                window.btcMarketPolygon.y
            );
            // 거리가 너무 멀면 충돌이 아니라고 판단
            if (centerDistance > 50) {
                isColliding = false;
            }
        }
        
        return isColliding;
    }

    // 트레이너 상태 정보 반환
    getTrainerStatus(model, config) {
        return {
            position: { x: model.circle.x, y: model.circle.y },
            target: { x: model.targetX, y: model.targetY },
            targetAction: model.targetAction,
            distanceToTarget: this.calculateDistance(
                model.circle.x, 
                model.circle.y, 
                model.targetX, 
                model.targetY
            ),
            distanceToSignalCenter: this.getDistanceToSignalCenter(model, config),
            distanceToBTCMarket: this.getDistanceToBTCMarket(model, config),
            distanceToNBGuild: this.getDistanceToNBGuild(model, config)
        };
    }
}

// 전역 객체로 등록
if (typeof window !== 'undefined') {
    window.trainerMovementManager = new TrainerMovementManager();
}

// 모듈 로딩 완료
