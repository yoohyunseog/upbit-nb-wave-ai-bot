// 트레이너 시각적 효과 모듈
// 트레이너의 색상 변경, 진동 효과, 애니메이션을 담당

class TrainerVisualEffects {
    constructor() {
        this.colorMap = {
            '매수': 0x0088ff,        // 파란색
            '매도': 0xff8800,        // 주황색
            '신호 대기': 0x88ccff,   // 하늘색
            'BTC 시장 방문': 0x0088ff, // 파란색
            'N/B 길드 방문': 0xff8800, // 주황색
            '대기': 0xffff00,        // 노란색
            'BTC 시장 탐색': 0x0088ff  // 파란색
        };
    }

    // 트레이너 색상 변경
    changeTrainerColor(model, action) {
        const color = this.colorMap[action] || 0x88ccff;
        model.circle.setFillStyle(color);
        
        // 로그 출력
        if (window.logManager) {
            const colorName = this.getColorName(color);
            window.logManager.addLog(`🎨 트레이너: ${action} 의사결정! ${colorName}으로 변경`);
        }
    }

    // 색상 이름 반환
    getColorName(color) {
        const colorNames = {
            0x0088ff: '파란색',
            0xff8800: '주황색',
            0x88ccff: '하늘색',
            0xffff00: '노란색'
        };
        return colorNames[color] || '기본색';
    }

    // N/B 길드에서 진동 효과
    createShakeEffect(model, duration = 2000) {
        const originalX = model.circle.x;
        const originalY = model.circle.y;
        let shakeCount = 0;
        const maxShakes = 10;
        
        const shakeInterval = setInterval(() => {
            if (model.circle && shakeCount < maxShakes) {
                // 360도 랜덤 방향으로 진동 (0~3px)
                const angle = Math.random() * 2 * Math.PI;
                const distance = Math.random() * 3;
                const shakeX = originalX + Math.cos(angle) * distance;
                const shakeY = originalY + Math.sin(angle) * distance;
                
                this.setModelPosition(model, shakeX, shakeY);
                shakeCount++;
            } else {
                clearInterval(shakeInterval);
                // 원래 위치로 복원
                this.setModelPosition(model, originalX, originalY);
            }
        }, 1);
    }

    // N/B 길드 다각형 깜빡임 효과
    createGuildBlinkEffect() {
        if (!window.nbGuildPolygon) return;
        
        const originalColor = 0x00ff00;
        let blinkCount = 0;
        const maxBlinks = 6;
        
        const blinkInterval = setInterval(() => {
            if (window.nbGuildPolygon && blinkCount < maxBlinks) {
                const isBright = blinkCount % 2 === 0;
                window.nbGuildPolygon.setFillStyle(isBright ? 0x00ff88 : originalColor);
                blinkCount++;
            } else {
                clearInterval(blinkInterval);
                if (window.nbGuildPolygon) {
                    window.nbGuildPolygon.setFillStyle(originalColor);
                }
            }
        }, 250);
    }

    // 트레이너 원형 깜빡임 효과
    createCircleBlinkEffect(model, duration = 2000) {
        const originalColor = model.circle.fillColor;
        let blinkCount = 0;
        const maxBlinks = 8;
        
        const circleBlinkInterval = setInterval(() => {
            if (model.circle && blinkCount < maxBlinks) {
                const isBright = blinkCount % 2 === 0;
                model.circle.setFillStyle(isBright ? 0xff8800 : originalColor);
                blinkCount++;
            } else {
                clearInterval(circleBlinkInterval);
                if (model.circle) {
                    model.circle.setFillStyle(originalColor);
                }
            }
        }, 250);
    }

    // 모델 위치 설정 (이름과 역할 텍스트 포함)
    setModelPosition(model, x, y) {
        model.circle.x = x;
        model.circle.y = y;
        if (model.name) {
            model.name.x = x;
            model.name.y = y - 6;
        }
        if (model.role) {
            model.role.x = x;
            model.role.y = y + 6;
        }
    }

    // N/B 길드 방문 시 모든 시각적 효과 실행
    createNBGuildEffects(model) {
        if (window.logManager) {
            window.logManager.addLog(`📳 N/B 길드 트레이너 원형 진동 효과 시작: 1~3초간 지속`);
        }
        
        // 다각형 깜빡임 효과
        this.createGuildBlinkEffect();
        
        // 원형 깜빡임 효과
        this.createCircleBlinkEffect(model);
        
        // 진동 효과
        this.createShakeEffect(model);
    }

    // 도착 효과 (목표에 도달했을 때)
    createArrivalEffect(model, targetName) {
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너: ${targetName}에 도착!`);
        }
        
        // 도착 시 잠깐 크기 변화 효과
        if (model.circle) {
            const originalScale = model.circle.scaleX;
            model.circle.setScale(originalScale * 1.2);
            
            setTimeout(() => {
                if (model.circle) {
                    model.circle.setScale(originalScale);
                }
            }, 500);
        }
    }

    // 이동 중 효과 (연결선 업데이트 등)
    updateConnectionLines(trainer, aiModels, nbCoinItems) {
        // 연결선 업데이트 로직은 별도로 구현
        // 여기서는 시각적 효과만 담당
    }
}

// 전역 인스턴스 생성
window.trainerVisualEffects = new TrainerVisualEffects();
