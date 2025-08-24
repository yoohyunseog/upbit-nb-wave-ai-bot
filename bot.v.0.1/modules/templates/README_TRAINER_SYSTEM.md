# Trainer System Documentation

## 개요
트레이너 시스템은 기존의 단일 파일에서 분리된 모듈형 아키텍처로 재구성되었습니다. 이는 코드의 유지보수성과 확장성을 향상시키고, 각 기능별로 독립적인 개발과 테스트가 가능하도록 합니다.

## 파일 구조

### 1. `trainer-decision-handler-fixed.js`
**역할**: 트레이너의 의사결정 로직을 담당
- 매수/매도 수익률 계산
- 구역별 단계 관리
- 의사결정 프로세스 제어

**주요 기능**:
- `calculateBuyProfitRate(model, config)`: 매수 수익률 계산
- `calculateSellProfitRate(model, config)`: 매도 수익률 계산
- `handleTrainerDecision()`: 메인 의사결정 처리
- 구역별 단계 관리 (`getZoneStep`, `nextZoneStep`, `resetZoneStep`)

### 2. `trainer-movement-controller.js`
**역할**: 트레이너의 이동 로직을 담당
- 목표 위치 설정
- 부드러운 이동 처리
- 도착 확인

**주요 기능**:
- `updateTrainerMovement(model, config)`: 이동 업데이트
- `setTargetPosition(model, config)`: 목표 위치 설정
- `moveToTarget(model)`: 목표로 이동
- `checkArrival(model)`: 도착 확인

### 3. `trainer-dialog-system.js`
**역할**: 트레이너의 UI 및 대화창 관리
- 대화창 텍스트 업데이트
- 위치 정보 표시
- 학습 상태 표시

**주요 기능**:
- `updateTrainerDialog(model)`: 대화창 업데이트
- `updateTrainerPositionInfo(model, config)`: 위치 정보 업데이트
- `updateLearningStatus(model, gameInitializer)`: 학습 상태 업데이트
- `updateAllUI()`: 모든 UI 요소 업데이트

### 4. `trainer-system-main.js`
**역할**: 모든 트레이너 모듈을 초기화하고 조정
- 모듈 초기화 및 관리
- 시스템 상태 모니터링
- 설정 관리

**주요 기능**:
- `initialize()`: 모든 모듈 초기화
- `getStatus()`: 시스템 상태 확인
- `restart()`: 시스템 재시작
- `updateSettings()`: 설정 업데이트

## 사용 방법

### 1. 기본 초기화
```javascript
// 자동 초기화 (페이지 로드 시)
document.addEventListener('DOMContentLoaded', function() {
    if (window.TrainerSystemMain) {
        window.trainerSystemMain = new window.TrainerSystemMain();
        window.trainerSystemMain.initialize();
    }
});
```

### 2. 수동 초기화
```javascript
// 수동으로 초기화
const trainerSystem = new window.TrainerSystemMain();
const success = trainerSystem.initialize();

if (success) {
    console.log('트레이너 시스템 초기화 성공');
} else {
    console.error('트레이너 시스템 초기화 실패');
}
```

### 3. 시스템 상태 확인
```javascript
const status = window.trainerSystemMain.getStatus();
console.log('시스템 상태:', status);
```

### 4. 설정 업데이트
```javascript
const settings = {
    movementSpeed: 3,
    arrivalThreshold: 15,
    dialogUpdateInterval: 2000
};

window.trainerSystemMain.updateSettings(settings);
```

### 5. 시스템 테스트
```javascript
const testResults = window.trainerSystemMain.test();
console.log('테스트 결과:', testResults);
```

## 수정된 오류

### 1. TypeError: Cannot read properties of undefined (reading 'x')
**원인**: `calculateBuyProfitRate` 메서드 호출 시 잘못된 매개변수 전달
**해결**: 
- 기존: `this.calculateBuyProfitRate(currentMajority, '매수영역')`
- 수정: `this.calculateBuyProfitRate(model, config)`

### 2. 매개변수 불일치
**원인**: 메서드 시그니처와 호출 시 매개변수가 일치하지 않음
**해결**: 모든 메서드의 매개변수를 일관성 있게 수정

## 아키텍처 장점

### 1. 모듈화
- 각 기능이 독립적인 모듈로 분리
- 코드 재사용성 향상
- 유지보수 용이성 증가

### 2. 확장성
- 새로운 기능 추가 시 기존 코드 영향 최소화
- 모듈별 독립적인 개발 가능
- 테스트 용이성 향상

### 3. 디버깅
- 각 모듈별 독립적인 디버깅 가능
- 명확한 책임 분리
- 오류 추적 용이

## 통합 방법

### 1. HTML에서 스크립트 로드
```html
<!-- 트레이너 시스템 모듈들 -->
<script src="trainer-decision-handler-fixed.js"></script>
<script src="trainer-movement-controller.js"></script>
<script src="trainer-dialog-system.js"></script>
<script src="trainer-system-main.js"></script>
```

### 2. game-initializer.js 수정
기존의 복잡한 `handleTrainer` 메서드를 간소화:
```javascript
handleTrainer(model, currentMajority, currentPriceText) {
    // 트레이너 의사결정 처리
    if (window.trainerDecisionHandler) {
        const targetAction = window.trainerDecisionHandler.handleTrainerDecision(
            model, this.game.config, currentMajority, this.gameData.nbCoins, 
            this.gameData.buyProfitRate, this.gameData.sellProfitRate, 
            (this.game.config.width - 240) / 2, 60, 120
        );
        model.targetAction = targetAction;
    }
    
    // 트레이너 이동 처리
    if (window.trainerMovementController) {
        window.trainerMovementController.updateTrainerMovement(model, this.game.config);
    }
    
    // 트레이너 역할 텍스트 업데이트
    model.role.setText(`트레이너 (${model.targetAction})`);
    
    // 트레이너 대화창 및 UI 업데이트
    if (window.trainerDialogSystem) {
        window.trainerDialogSystem.updateAllUI(model, this.game.config, this);
    }
}
```

## 주의사항

1. **로드 순서**: 모듈들이 올바른 순서로 로드되어야 합니다.
2. **의존성**: 각 모듈은 `window.logManager` 등의 전역 객체에 의존합니다.
3. **초기화**: `TrainerSystemMain.initialize()`가 성공적으로 실행되어야 합니다.
4. **호환성**: 기존 코드와의 호환성을 위해 전역 객체 이름을 유지했습니다.

## 문제 해결

### 1. 모듈이 로드되지 않는 경우
```javascript
// 각 모듈 클래스 확인
console.log('TrainerDecisionHandler:', !!window.TrainerDecisionHandler);
console.log('TrainerMovementController:', !!window.TrainerMovementController);
console.log('TrainerDialogSystem:', !!window.TrainerDialogSystem);
```

### 2. 초기화 실패 시
```javascript
// 시스템 상태 확인
const status = window.trainerSystemMain.getStatus();
console.log('시스템 상태:', status);

// 재시작 시도
window.trainerSystemMain.restart();
```

### 3. 디버그 정보 확인
```javascript
const debugInfo = window.trainerSystemMain.getDebugInfo();
console.log('디버그 정보:', debugInfo);
```
