// 트레이너 이동 컨트롤러 모듈
// 트레이너의 이동 로직, 목표 설정, 이동 속도 관리를 담당

class TrainerMovementController {
    constructor() {
        this.movementSpeed = 2.0; // 이동 속도 조정 (적당한 속도로 변경)
        this.arrivalThreshold = 10;
        this.targetX = 0;
        this.targetY = 0;
        this.lastTargetAction = null; // 마지막 목표 액션 추적
        
        if (window.logManager) {
            window.logManager.addLog(`🚀 트레이너 이동 컨트롤러 초기화 완료 (이동속도: ${this.movementSpeed})`);
        }
    }

    // 트레이너 이동 업데이트
    updateTrainerMovement(model, config) {
        // 이동은 trainer-state-handler에서만 수행 (여기서는 델리게이트만)
        if (window.trainerStateHandler && typeof window.trainerStateHandler.updateTrainerMovement === 'function') {
            window.trainerStateHandler.updateTrainerMovement(model, config);
        }
    }

    // 목표 위치 설정
    setTargetPosition(model, config) {
        const centerX = config.width / 2;
        const centerY = config.height / 2;
        
        // 구역별 정확한 좌표 계산
        const spacing = 120;
        const startX = (config.width - (spacing * 2)) / 2;
        const topY = 60;
        
        // 현재 액션에 따른 목표 위치 결정
        switch (model.targetAction) {
            case '매수 수익률 계산':
            case '매수 의사결정':
            case '매수 실행':
            case 'N/B 코인 드랍':
                // 매수 구역 (상단 좌측)
                this.targetX = startX;
                this.targetY = topY;
                break;
                
            case '매도 수익률 계산':
            case '매도 의사결정':
            case '매도 실행':
                // 매도 구역 (상단 중앙)
                this.targetX = startX + spacing;
                this.targetY = topY;
                break;
                
            case 'BTC 시장 탐색':
            case '시장 분석 완료':
                // BTC 시장 탐색 구역 (우하단) - 더 정확한 위치 설정
                this.targetX = config.width - 100;
                this.targetY = config.height - 100;
                break;
                
            case 'N/B 코인 확인':
            case 'N/B 길드 방문':
                // N/B 길드 (좌상단)
                this.targetX = 100;
                this.targetY = 100;
                break;
                
            case '시장 신호 분석':
            case '다음 목적지 결정':
            case '신호 대기 센터 이동':
            case '신호 대기':
            default:
                // 신호 대기 센터 (중앙)
                this.targetX = centerX;
                this.targetY = centerY;
                break;
        }
        
        // model의 targetX, targetY도 업데이트
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        
        // 목표 위치 로그 (중복 방지를 위해 액션이 변경된 경우에만 출력)
        if (window.logManager && model.targetAction !== this.lastLoggedAction) {
            const targetPos = `(${Math.round(this.targetX)}, ${Math.round(this.targetY)})`;
            const currentPos = `(${Math.round(model.circle.x)}, ${Math.round(model.circle.y)})`;
            const distance = Math.sqrt((this.targetX - model.circle.x) ** 2 + (this.targetY - model.circle.y) ** 2);
            
            // 구역 정보 가져오기 (game-initializer의 함수 사용)
            const currentZone = window.gameInitializer?.getCurrentZoneName(model.circle.x, model.circle.y) || '기타영역';
            const targetZone = window.gameInitializer?.getCurrentZoneName(this.targetX, this.targetY) || '기타영역';
            
            window.logManager.addLog(`🎯 트레이너 의사결정: 현재위치 ${currentPos} (${currentZone}) | 목표위치 ${targetPos} (${targetZone}) | 목표까지거리 ${Math.round(distance)}px`);
            this.lastLoggedAction = model.targetAction;
        }
    }

    // 목표 위치로 이동
    moveToTarget(model) {
        // TrainerStateHandler가 이미 처리하고 있으므로 이 함수는 비활성화
        // 중복 이동 처리를 방지하기 위해 아무것도 하지 않음
        if (window.logManager && Math.floor(Date.now() / 1000) % 10 === 0) {
            window.logManager.addLog(`⚠️ TrainerMovementController.moveToTarget 비활성화: TrainerStateHandler가 처리 중`);
        }
        
        // 화면 위치 정보 업데이트만 수행
        if (window.trainerPositionInfo) {
            const currentPos = `(${model.circle.x.toFixed(1)}, ${model.circle.y.toFixed(1)})`;
            const targetPos = `(${this.targetX.toFixed(1)}, ${this.targetY.toFixed(1)})`;
            const distance = Math.sqrt((this.targetX - model.circle.x) ** 2 + (this.targetY - model.circle.y) ** 2);
            const positionText = `📍 위치: ${currentPos} | 목표: ${targetPos} | 거리: ${Math.round(distance)}px`;
            window.trainerPositionInfo.setText(positionText);
        }
    }

    // 도착 확인
    checkArrival(model) {
        const dx = this.targetX - model.circle.x;
        const dy = this.targetY - model.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // 도착 확인 (BTC 시장 탐색의 경우 더 넓은 범위로 설정)
        const threshold = (model.targetAction === 'BTC 시장 탐색') ? 30 : this.arrivalThreshold;
        
        if (distance <= threshold) {
            if (!model.arrivalLogged) {
                model.arrivalLogged = true;
                
                // BTC 시장 탐색의 경우 특별 처리
                if (model.targetAction === 'BTC 시장 탐색') {
                    if (window.logManager) {
                        window.logManager.addLog(`🎯 트레이너 BTC 시장 탐색 구역 도착! 수익률 계산 시작...`);
                    }
                    
                    // BTC 시장 도착 플래그 설정
                    model.btcMarketArrived = true;
                    
                    // 새로운 의사결정 필요 플래그 설정
                    model.needsNewDecision = true;
                    
                    // 직접 BTC 시장 도착 처리 함수 호출
                    if (window.trainerDecisionHandler && typeof window.trainerDecisionHandler.handleBTCMarketArrival === 'function') {
                        setTimeout(() => {
                            window.trainerDecisionHandler.handleBTCMarketArrival(model, window.gameConfig);
                        }, 100);
                    }
                } else {
                    if (window.logManager) {
                        window.logManager.addLog(`🎯 트레이너 목표 지점 도착! 구역 작업 시작...`);
                    }
                }
            }
            
            // 도착 후 대화창 업데이트
            if (window.trainerDialog) {
                const currentTime = new Date().toLocaleTimeString();
                const dialogText = `✅ 트레이너: 목표 지점 도착! 구역 작업 진행 중... | 시간: ${currentTime}`;
                window.trainerDialog.setText(dialogText);
                
                if (window.logManager) {
                    window.logManager.addLog(`📺 화면출력(트레이너대화창): ${dialogText}`);
                }
            }
            
            // BTC 시장 탐색 도착 시 즉시 수익률 계산 시작
            if (model.targetAction === 'BTC 시장 탐색' && model.btcMarketArrived) {
                if (window.logManager) {
                    window.logManager.addLog(`🎯 BTC 시장 도착 확인 - 즉시 수익률 계산 시작`);
                }
                
                // 즉시 수익률 계산 함수 호출
                if (window.trainerDecisionHandler && typeof window.trainerDecisionHandler.handleBTCMarketArrival === 'function') {
                    window.trainerDecisionHandler.handleBTCMarketArrival(model, window.gameConfig);
                }
            }
            
            return true; // 도착 완료
        } else {
            // 목표에서 멀어지면 도착 로그 리셋 (더 관대한 조건으로 변경)
            if (distance > threshold * 3) {
                model.arrivalLogged = false;
                if (model.targetAction === 'BTC 시장 탐색') {
                    model.btcMarketArrived = false;
                }
            }
        }
    }

    // 특정 위치로 즉시 이동 (텔레포트)
    teleportToPosition(model, x, y) {
        model.circle.x = x;
        model.circle.y = y;
        
        // 이름과 역할 텍스트도 함께 이동
        if (model.name) {
            model.name.x = x;
            model.name.y = y - 6;
        }
        if (model.role) {
            model.role.x = x;
            model.role.y = y + 6;
        }
        
        const teleportPos = `(${Math.round(x)}, ${Math.round(y)})`;
        if (window.logManager) {
            window.logManager.addLog(`⚡ 트레이너 텔레포트: ${teleportPos}`);
        }
    }

    // 이동 속도 조정
    setMovementSpeed(speed) {
        this.movementSpeed = Math.max(0.1, Math.min(10, speed));
        if (window.logManager) {
            window.logManager.addLog(`⚙️ 트레이너 이동 속도 조정: ${this.movementSpeed}`);
        }
    }

    // 도착 임계값 조정
    setArrivalThreshold(threshold) {
        this.arrivalThreshold = Math.max(1, Math.min(50, threshold));
        if (window.logManager) {
            window.logManager.addLog(`⚙️ 트레이너 도착 임계값 조정: ${this.arrivalThreshold}`);
        }
    }

    // 트레이너 이동 컨트롤러 재시작
    restart() {
        console.log('🔄 트레이너 이동 컨트롤러 재시작 시작...');
        
        try {
            // 기본 설정으로 초기화
            this.movementSpeed = 2;
            this.arrivalThreshold = 10;
            this.targetX = 0;
            this.targetY = 0;
            
            // 트레이너 모델 찾기 및 재시작
            if (window.aiModels && Array.isArray(window.aiModels)) {
                const trainerModel = window.aiModels.find(model => model.isTrainer);
                if (trainerModel) {
                    this.restartTrainer(trainerModel);
                }
            }
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 트레이너 이동 컨트롤러 재시작 완료`);
            }
            
            console.log('✅ 트레이너 이동 컨트롤러 재시작 완료');
        } catch (error) {
            console.error('❌ 트레이너 이동 컨트롤러 재시작 실패:', error);
        }
    }

    // 트레이너 재시작
    restartTrainer(model) {
        if (!model || !model.circle) {
            console.log('❌ 트레이너 재시작 실패: 모델이 유효하지 않음');
            return;
        }
        
        try {
            // 트레이너 상태 초기화
            model.targetAction = 'N/B 코인 확인';
            model.arrivalLogged = false;
            model.needsNewDecision = false;
            
            // N/B 길드 위치로 이동
            const nbGuildX = 150;
            const nbGuildY = 150;
            
            model.circle.x = nbGuildX;
            model.circle.y = nbGuildY;
            model.targetX = nbGuildX;
            model.targetY = nbGuildY;
            
            // 텍스트 위치 업데이트
            if (model.name) {
                model.name.x = nbGuildX;
                model.name.y = nbGuildY - 6;
            }
            if (model.role) {
                model.role.x = nbGuildX;
                model.role.y = nbGuildY + 6;
            }
            
            // 역할 텍스트 업데이트
            if (model.role) {
                model.role.setText(`트레이너 (${model.targetAction})`);
            }
            
            if (window.logManager) {
                window.logManager.addLog(`🔄 트레이너 재시작 완료 - N/B 길드 위치: (${Math.round(nbGuildX)}, ${Math.round(nbGuildY)})`);
            }
            
            console.log('🎯 트레이너 재시작 완료');
        } catch (error) {
            console.error('❌ 트레이너 재시작 실패:', error);
        }
    }

    // 트레이너 이동 상태 확인
    isTrainerMoving(model) {
        if (!model || !model.circle) return false;
        
        const dx = this.targetX - model.circle.x;
        const dy = this.targetY - model.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        return distance > this.arrivalThreshold;
    }

    // 트레이너 도착 상태 확인
    isTrainerArrived(model) {
        if (!model || !model.circle) return false;
        
        const dx = this.targetX - model.circle.x;
        const dy = this.targetY - model.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        return distance <= this.arrivalThreshold;
    }

    // 현재 이동 상태 정보 반환
    getMovementStatus(model) {
        const dx = this.targetX - model.circle.x;
        const dy = this.targetY - model.circle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        return {
            currentPosition: { x: model.circle.x, y: model.circle.y },
            targetPosition: { x: this.targetX, y: this.targetY },
            distance: distance,
            isArrived: distance <= this.arrivalThreshold,
            movementSpeed: this.movementSpeed,
            targetAction: model.targetAction
        };
    }

    // 매수 구역으로 이동
    moveToBuyArea(model, startX, topY) {
        this.targetX = startX;
        this.targetY = topY;
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        model.targetAction = '매수 수익률 계산';
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 목표 설정: 매수 구역 (${Math.round(this.targetX)}, ${Math.round(this.targetY)})`);
        }
    }

    // 매도 구역으로 이동
    moveToSellArea(model, startX, topY, spacing) {
        this.targetX = startX + spacing;
        this.targetY = topY;
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        model.targetAction = '매도 수익률 계산';
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 목표 설정: 매도 구역 (${Math.round(this.targetX)}, ${Math.round(this.targetY)})`);
        }
    }

    // BTC 시장 탐색 구역으로 이동
    moveToBTCMarket(model, config) {
        this.targetX = config.width - 100;
        this.targetY = config.height - 100;
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        model.targetAction = 'BTC 시장 탐색';
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 목표 설정: BTC 시장 탐색 구역 (${Math.round(this.targetX)}, ${Math.round(this.targetY)})`);
        }
    }

    // N/B 길드로 이동
    moveToNBGuild(model) {
        this.targetX = 100;
        this.targetY = 100;
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        model.targetAction = 'N/B 코인 확인';
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 목표 설정: N/B 길드 (${Math.round(this.targetX)}, ${Math.round(this.targetY)})`);
        }
    }

    // 신호 대기 센터로 이동
    moveToSignalCenter(model, config) {
        this.targetX = config.width / 2;
        this.targetY = config.height / 2;
        model.targetX = this.targetX;
        model.targetY = this.targetY;
        model.targetAction = '신호 대기';
        
        if (window.logManager) {
            window.logManager.addLog(`🎯 트레이너 목표 설정: 신호 대기 센터 (${Math.round(this.targetX)}, ${Math.round(this.targetY)})`);
        }
    }
}

// 전역 객체로 등록
window.TrainerMovementController = TrainerMovementController;

// 전역 인스턴스 생성
window.trainerMovementController = new TrainerMovementController();
