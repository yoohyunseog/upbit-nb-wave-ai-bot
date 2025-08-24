# 🔄 새로고침 문제 해결 가이드

## 문제 상황
기존에는 페이지를 새로고침하면 주민들과 트레이너가 모두 멈춰버리는 문제가 있었습니다. 이는 게임 상태가 제대로 저장/복원되지 않아서 발생하는 문제였습니다.

## 해결 방법

### 1. 게임 상태 관리자 개선
- **자동 상태 저장**: 페이지 언로드 시 자동으로 게임 상태를 저장
- **자동 상태 복원**: 페이지 로드 시 자동으로 이전 상태를 복원
- **시스템 재시작**: 모든 AI 모델과 시스템을 자동으로 재시작

### 2. 각 시스템별 재시작 기능 추가

#### 탐색자 이동 시스템 (`explorer-movement-system.js`)
```javascript
// 개별 탐색자 재시작
restartExplorer(model, index)

// 전체 탐색 시스템 재시작
restart()
```

#### 트레이너 이동 컨트롤러 (`trainer-movement-controller.js`)
```javascript
// 트레이너 이동 컨트롤러 재시작
restart()

// 개별 트레이너 재시작
restartTrainer(model)
```

#### 트레이너 의사결정 핸들러 (`trainer-decision-handler.js`)
```javascript
// 트레이너 의사결정 핸들러 재시작
restart()

// 트레이너 의사결정 재시작
restartTrainerDecision(model)
```

#### 주민 수집 시스템 (`resident-collection-system.js`)
```javascript
// 주민 수집 시스템 재시작
restart()

// 개별 주민 재시작
restartResident(resident, index)
```

### 3. 게임 상태 관리자 (`game-state-manager.js`)
```javascript
// 새로고침 시 N/B 길드에서 새로 시작
restoreGameStateOnRefresh()

// 모든 시스템 재시작
restartAllSystems()

// 게임 루프 재시작
restartGameLoop()

// N/B 길드에서 새로운 탐색원 생성 (저장된 데이터 무시)
initializeDefaultAiModels()

// 모든 AI 모델 제거
clearAllAiModels()
```

## 사용법

### 1. 자동 복원 (기본)
페이지를 새로고침하면 자동으로 N/B 길드에서 새로 시작되고 모든 시스템이 재시작됩니다.

### 2. 수동 재시작
```javascript
// 모든 시스템 재시작
if (window.gameStateManager) {
    window.gameStateManager.restartAllSystems();
}

// 개별 시스템 재시작
if (window.explorerMovementSystem) {
    window.explorerMovementSystem.restart();
}

if (window.trainerMovementController) {
    window.trainerMovementController.restart();
}

if (window.residentCollectionSystem) {
    window.residentCollectionSystem.restart();
}
```

### 3. 상태 확인
```javascript
// 시스템 상태 확인
if (window.gameStateManager) {
    const status = window.gameStateManager.getStatus();
    console.log('시스템 상태:', status);
}
```

## 테스트 방법

### 1. 테스트 파일 사용
`test-refresh-fix.html` 파일을 사용하여 새로고침 문제 해결을 테스트할 수 있습니다.

```bash
# 테스트 파일 실행
open 8BIT/bot.v.0.1/test-refresh-fix.html
```

### 2. 테스트 단계
1. 페이지 로드 후 시스템 상태 확인
2. "N/B 길드에서 새로운 탐색원 생성" 버튼으로 4개 탐색원 생성 테스트
3. "새로고침 테스트" 버튼 클릭
4. 페이지 새로고침 후 N/B 길드에서 새로 시작 확인
5. "시스템 상태 확인" 버튼으로 모든 시스템 정상 작동 확인

## 주요 개선사항

### 1. 자동 상태 저장
- 페이지 언로드 시 자동으로 게임 상태 저장
- localStorage를 활용한 영구 저장
- 24시간 이내 저장된 상태만 유효

### 5. N/B 길드에서 새로 시작하는 시스템
- 새로고침 시 탐색원과 트레이너의 위치 데이터를 저장하지 않음
- N/B 길드에서 4개의 새로운 탐색원을 화면의 4개 구역에 분산 배치
- 각 탐색원은 독립적인 탐색 활동 수행
- 트레이너는 N/B 길드 중앙에 배치되어 전체 관리
- 저장된 탐색원과 트레이너의 모든 데이터를 제거하고 새로 시작

### 2. 자동 상태 복원
- 페이지 로드 시 자동으로 이전 상태 복원
- **N/B 길드에서 새로 시작**: 저장된 탐색원과 트레이너 데이터 무시하고 N/B 길드에서 새로 시작
- 게임 데이터 복원 (N/B 코인, 미네랄 등)
- 모든 시스템 자동 재시작

### 3. 시스템 재시작
- 각 시스템별 독립적인 재시작 기능
- AI 모델 움직임 자동 재개
- 의사결정 시스템 재활성화

### 4. 오류 처리
- 복원 실패 시 새 게임으로 자동 전환
- 상세한 로그 기록
- 오류 발생 시 안전한 폴백 처리

## 파일 구조

```
8BIT/bot.v.0.1/
├── modules/templates/
│   ├── game-state-manager.js          # 게임 상태 관리자 (개선됨)
│   ├── explorer-movement-system.js    # 탐색자 이동 시스템 (재시작 기능 추가)
│   ├── trainer-movement-controller.js # 트레이너 이동 컨트롤러 (재시작 기능 추가)
│   ├── trainer-decision-handler.js    # 트레이너 의사결정 핸들러 (재시작 기능 추가)
│   └── resident-collection-system.js  # 주민 수집 시스템 (재시작 기능 추가)
├── index.html                         # 메인 HTML (게임 상태 관리자 초기화 추가)
├── test-refresh-fix.html              # 테스트 파일 (새로 생성)
└── README_REFRESH_FIX.md              # 이 파일
```

## 문제 해결

### 1. 새로고침 후 여전히 멈춰있는 경우
```javascript
// 수동으로 모든 시스템 재시작
restartAllSystems();
```

### 2. 상태 복원이 안 되는 경우
```javascript
// 게임 상태 초기화 후 새로 시작
if (window.gameStateManager) {
    window.gameStateManager.clearGameState();
}
location.reload();
```

### 3. 특정 시스템만 문제가 있는 경우
```javascript
// 개별 시스템 재시작
if (window.explorerMovementSystem) {
    window.explorerMovementSystem.restart();
}
```

## 로그 확인

브라우저 개발자 도구의 콘솔에서 다음 로그들을 확인할 수 있습니다:

- `🔄 새로고침 감지 - 게임 상태 복원 시작...`
- `✅ 새로고침 후 게임 상태 복원 완료`
- `🔄 모든 시스템 재시작 완료`
- `🤖 X개의 AI 모델 복원 및 움직임 재시작 완료`

## 성능 최적화

- 상태 저장은 5초마다 자동으로 실행
- 복원 시 불필요한 중복 실행 방지
- 메모리 누수 방지를 위한 적절한 정리 작업

## 주의사항

1. **브라우저 호환성**: localStorage를 지원하는 모든 브라우저에서 작동
2. **저장 용량**: localStorage 용량 제한으로 인해 24시간 이내 데이터만 유지
3. **네트워크 의존성**: 오프라인에서도 정상 작동
4. **보안**: 민감한 데이터는 저장하지 않음

## 향후 개선 계획

1. **서버 저장**: localStorage 대신 서버에 상태 저장
2. **실시간 동기화**: 여러 탭 간 실시간 상태 동기화
3. **상태 버전 관리**: 상태 변경 이력 관리
4. **성능 모니터링**: 시스템 재시작 성능 최적화
