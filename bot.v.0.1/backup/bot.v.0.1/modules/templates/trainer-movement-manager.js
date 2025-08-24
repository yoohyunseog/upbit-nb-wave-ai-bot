// 트레이너 이동 관리자 모듈
// 트레이너의 이동, 상태 관리, 위치 계산을 담당

class TrainerMovementManager {
    constructor() {
        this.movementSpeed = 2;
        this.arrivalThreshold = 30;
    }

    // 트레이너 이동 처리
    updateTrainerMovement(model, config) {
        const modelX = model.circle.x;
        const modelY = model.circle.y;
        
        // 목표 위치까지의 거리 계산
        const distanceToTarget = Math.sqrt((model.targetX - modelX) ** 2 + (model.targetY - modelY) ** 2);
        
        // 목표에 도달했는지 확인
        if (distanceToTarget < this.arrivalThreshold) {
            this.handleArrival(model, config);
            return true; // 도달 완료
        }
        
        // 이동 방향 계산
        const dx = model.targetX - modelX;
        const dy = model.targetY - modelY;
        const distance = Math.sqrt(dx ** 2 + dy ** 2);
        
        // 정규화된 이동 벡터
        const moveX = (dx / distance) * this.movementSpeed;
        const moveY = (dy / distance) * this.movementSpeed;
        
        // 트레이너 위치 업데이트
        model.circle.x += moveX;
        model.circle.y += moveY;
        
        // 이름과 역할 텍스트도 함께 이동
        if (model.name) {
            model.name.x = model.circle.x;
            model.name.y = model.circle.y - 6;
        }
        if (model.role) {
            model.role.x = model.circle.x;
            model.role.y = model.circle.y + 6;
        }
        
        return false; // 아직 이동 중
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
