// 간단한 template-loader.js - 트레이너 관련 코드 제거
// 트레이너 처리는 TrainerStateHandler 모듈에서 담당

// 기존 template-loader.js의 핵심 기능만 유지하고 트레이너 관련 코드는 제거
// 이 파일은 임시로 사용되며, 나중에 원본 파일을 정리할 때 참고용

console.log('🔧 간단한 template-loader.js 로드됨 - 트레이너 처리는 TrainerStateHandler에서 담당');

// 기본 설정
const config = {
    width: 1080,
    height: 750
};

// AI 모델 초기화
const aiModels = [];
const initialPositions = [
    { x: config.width / 4, y: config.height / 4 },
    { x: config.width * 3/4, y: config.height / 4 },
    { x: config.width / 4, y: config.height * 3/4 },
    { x: config.width * 3/4, y: config.height * 3/4 },
    { x: config.width / 2, y: config.height / 2 }  // 트레이너를 화면 중앙에 배치
];

// 트레이너 모델 초기화
function initializeTrainerModel(model, config) {
    model.isTrainer = true;
    model.targetAction = 'N/B 코인 확인';
    model.targetX = 150;
    model.targetY = 150;
    model.circle.setFillStyle(0x88ccff);
    
    if (window.logManager) {
        window.logManager.addLog(`🔵 트레이너 모델 초기화 완료 - N/B 길드에서 시작`);
    }
}

// AI 시스템 알고리즘 (트레이너 처리 제거)
function aiSystemAlgorithm(models, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog) {
    models.forEach((model, index) => {
        if (model.isTrainer) {
            // 트레이너 모델: TrainerStateHandler를 통한 처리
            if (window.trainerStateHandler && typeof window.trainerStateHandler.updateTrainerState === 'function') {
                const targetAction = window.trainerStateHandler.updateTrainerState(
                    model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog
                );
                
                // 트레이너 액션 처리
                window.trainerStateHandler.handleTrainerActions(
                    model, config, currentMajority, nbCoins, buyProfitRate, sellProfitRate, trainerDialog
                );
            } else {
                console.warn('⚠️ TrainerStateHandler 모듈이 로드되지 않았습니다.');
                
                // 기본 처리 (fallback)
                if (typeof model.targetAction === 'undefined' || model.targetAction === '') {
                    model.targetAction = '신호 대기';
                    model.targetX = config.width / 2;
                    model.targetY = config.height / 2;
                    model.circle.setFillStyle(0x88ccff);
                    
                    if (window.logManager) {
                        window.logManager.addLog(`🔵 트레이너: 기본 처리 - targetAction 초기화`);
                    }
                }
            }
        } else {
            // 다른 AI 모델들의 처리 (기존 로직 유지)
            // ... 기존 코드
        }
    });
}

// 트레이너 이동 처리
function updateTrainerMovement(model, config) {
    // 트레이너 이동 처리 (TrainerStateHandler만 사용)
    if (model.isTrainer && window.trainerStateHandler && typeof window.trainerStateHandler.updateTrainerMovement === 'function') {
        window.trainerStateHandler.updateTrainerMovement(model, config);
    } else if (model.isTrainer) {
        // TrainerStateHandler가 없는 경우에만 경고 로그
        if (window.logManager && Math.floor(Date.now() / 1000) % 5 === 0) {
            window.logManager.addLog(`⚠️ TrainerStateHandler가 없어서 트레이너 이동 처리 스킵`);
        }
    }
    // fallback 로직 제거 - 중복 이동 처리 방지
}

// 전역 함수로 노출
window.initializeTrainerModel = initializeTrainerModel;
window.aiSystemAlgorithm = aiSystemAlgorithm;
window.updateTrainerMovement = updateTrainerMovement;

console.log('✅ 간단한 template-loader.js 초기화 완료');
