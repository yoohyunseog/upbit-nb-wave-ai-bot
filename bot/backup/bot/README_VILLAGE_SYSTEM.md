# 🏘️ 8BIT 마을 트레이딩 시스템

## 📖 개요

**8BIT 마을**은 N/B 길드의 지점장인 **촌장**이 이끄는 독특한 트레이딩 마을입니다. 마을 주민들은 각자의 역할과 전문성을 바탕으로 **실제 거래**와 **모의 거래**를 통해 마을 경제를 발전시키고, N/B 코인을 통해 학습하며 성장합니다.

---

## 🏛️ 촌장 (Mayor) - 마을 지도자

### 역할과 특성
- **위치**: 마을회관 (Town Hall)
- **역할**: N/B 길드 지점장, 마을 지도자
- **전문성**: 마을 리더십과 재정 관리
- **전략**: 적응형 리더십 (N/B 길드 지침 기반)

### 신뢰도 기반 지침 시스템

#### 🤖 ML Model Trust (40%)
- **의미**: 머신러닝 모델의 예측에 대한 신뢰도
- **해석**: 낮은 신뢰도로 ML 모델보다 개인 판단 우선
- **지침**: "개인 분석과 판단을 우선시하라"

#### 🏛️ N/B Guild Trust (85%)
- **의미**: N/B 길드의 전략과 지침에 대한 신뢰도
- **해석**: 높은 신뢰도로 길드 전통 존중
- **지침**: "길드의 전통적 지침을 존중하라"

### 구역별 전략

#### 🔵 BLUE 구역 (알파 구역)
- **접근법**: 공격적이고 자신감 있음
- **행동 패턴**: 70% 확률로 BUY (강한 상승 편향)
- **이유**: BLUE 구역은 우호적 영역으로, N/B 길드가 공격적으로 운영 가능

#### 🍊 ORANGE 구역 (베타 구역)
- **접근법**: 초신중하고 방어적
- **행동 패턴**: 60% 확률로 HOLD (극도의 신중함)
- **이유**: ORANGE 구역은 적대적 영역으로, N/B 길드가 방어적 입장을 유지

---

## 👥 마을 주민 시스템 (Guild Members)

### 1. 정찰병 (Scout) - 탐험가
- **위치**: 게이트 (Gate)
- **전문**: 빠른 신호 감지 (Quick Signals)
- **트레이너 카드**: minute1, minute3
- **전략**: 모멘텀 기반 빠른 거래
- **현재 상태**: 12분 동안 포지션 보유, P&L: +0.25%

### 2. 수호자 (Guardian) - 보호자
- **위치**: 시장 (Market)
- **전문**: 추세 보호 (Trend Protection)
- **트레이너 카드**: minute5, minute10
- **전략**: 평균 회귀 기반 안정적 거래
- **역할**: 실제 비트코인 거래 담당

### 3. 분석가 (Analyst) - 전략가
- **위치**: 타워 (Tower)
- **전문**: 전략적 분석 (Strategic Analysis)
- **트레이너 카드**: minute15, minute30
- **전략**: 돌파 기반 전략적 거래
- **역할**: 시장 분석 및 전략 개발

### 4. 어른 (Elder) - 고문
- **위치**: 여관 (Inn)
- **전문**: 장기적 지혜 (Long-term Wisdom)
- **트레이너 카드**: minute60, day
- **전략**: 추세 추종 기반 장기 거래
- **역할**: 멘토링 및 장기 계획

---

## 🚗 비트카 에너지 시스템

### 에너지 주입 과정
각 주민들은 마을 에너지를 자신의 **비트카**에 주입합니다:

```javascript
// 비트카 에너지 주입
const bitcar_energy_injection = {
    scout: { energy: 70, bitcar_model: "Quick Signal Runner" },
    guardian: { energy: 80, bitcar_model: "Trend Protector" },
    analyst: { energy: 90, bitcar_model: "Strategic Analyzer" },
    elder: { energy: 85, bitcar_model: "Wisdom Keeper" }
};
```

### ORANGE 구역으로의 여행
- **목적지**: ORANGE 구역 비트 시장
- **임무**: 촌장의 지침에 따른 트레이너 수행
- **특징**: 적대적 환경에서의 거래

---

## 📦 트레이너 창고 시스템

### 창고의 역할
각 트레이너는 자신만의 **개인 창고**를 가지고 있으며, 이곳에 모든 거래 기록이 실시간으로 저장됩니다.

### 📝 거래 일지 시스템
각 트레이너의 창고에는 **거래 일지**가 포함되어 있어, 촌장의 지침과 ML 모델의 판단을 기록합니다:

#### 일지 유형
1. **최근 일지**: 최근 10개의 거래 기록
2. **구역별 일지**: ORANGE/BLUE 구역별 거래 기록
3. **촌장 지침 일지**: 촌장의 지침에 따른 거래 결정 기록
4. **ML 모델 판단 일지**: ML 모델의 판단과 개인 확신 비교 기록

#### 일지 예시
```json
{
  "timestamp": "2025-01-27T08:15:00",
  "trainer": "scout",
  "action": "BUY",
  "zone": "ORANGE",
  "price": 161000000,
  "mayor_guidance": "ORANGE 구역에서 촌장의 방어적 지침을 무시하고 개인 확신으로 BUY 실행",
  "ml_decision": "ML 모델 신뢰도(40%)가 낮아 개인 판단(100%) 우선",
  "reasoning": "1분 차트에서 강력한 모멘텀 신호 감지",
  "lesson_learned": "개인 확신이 촌장 지침보다 우선시된 경우"
}
```

### 실시간 기록 시스템
```python
# 실시간 거래 기록 저장
def real_time_trade_recording(trainer, trade_data):
    warehouse = get_trainer_warehouse(trainer)
    
    # 거래 기록 저장
    warehouse['trade_records']['real_trades'].append({
        'timestamp': trade_data['timestamp'],
        'action': trade_data['action'],
        'price': trade_data['price'],
        'pnl': trade_data['pnl'],
        'strategy': trade_data['strategy'],
        'zone': trade_data['zone'],
        'confidence': trade_data['confidence']
    })
    
    # 수익/손실 업데이트
    update_profit_loss_history(warehouse, trade_data)
    
    # 학습 데이터 수집
    collect_learning_data(warehouse, trade_data)
```

### 창고 데이터 구조
```javascript
// 트레이너 창고 구조
const trainer_warehouse = {
    // 실시간 거래 기록
    trade_records: {
        real_trades: [],
        mock_trades: [],
        current_position: {}
    },
    
    // 수익/손실 기록
    profit_loss_history: {
        total_profit: 0,
        win_rate: 0,
        total_trades: 0,
        profitable_trades: 0,
        losing_trades: 0
    },
    
    // 학습 데이터
    learning_data: {
        successful_patterns: [],
        failed_patterns: [],
        market_conditions: [],
        strategy_effectiveness: {}
    }
};
```

---

## 🧠 학습 및 성장 시스템

### N/B 코인 기반 학습
각 주민들은 **N/B 코인**을 통해 학습하며 성장합니다:

```javascript
// 학습 시스템
function learnFromTrade(member, tradeResult) {
    if (tradeResult.profit > 0) {
        member.experience += tradeResult.profit * 0.1;
        member.nbCoins += tradeResult.profit * 0.05;
    } else {
        member.experience += Math.abs(tradeResult.profit) * 0.05;
    }
}
```

### 매매 전략 향상
창고의 기록들이 향후 매매 전략을 향상시킵니다:

```python
# 전략 향상 알고리즘
def strategy_improvement_algorithm(trainer):
    warehouse = get_trainer_warehouse(trainer)
    
    # 1. 성공 패턴 분석
    successful_patterns = analyze_successful_patterns(warehouse)
    
    # 2. 실패 패턴 분석
    failed_patterns = analyze_failed_patterns(warehouse)
    
    # 3. 시장 조건별 성과 분석
    market_performance = analyze_market_performance(warehouse)
    
    # 4. 전략 최적화
    optimized_strategy = optimize_strategy(successful_patterns, failed_patterns, market_performance)
    
    # 5. 새로운 전략 적용
    apply_improved_strategy(trainer, optimized_strategy)
```

---

## 🌐 API 엔드포인트

### 마을 시스템 API

#### 기본 정보
- `GET /api/village/status` - 마을 전체 상태 조회
- `GET /api/village/mayor/guidance` - 촌장의 신뢰도 기반 지침 조회
- `GET /api/village/residents` - 마을 주민 정보 조회
- `GET /api/village/system/overview` - 마을 시스템 전체 개요

#### 주민 관리
- `GET /api/village/resident/<trainer_name>` - 특정 주민 정보 조회
- `GET /api/village/scout/status` - Scout의 현재 상태 조회 (특별 API)

#### 창고 관리
- `GET /api/village/warehouse/<trainer_name>` - 트레이너 창고 정보 조회
- `GET /api/village/warehouse/<trainer_name>/analysis` - 창고 데이터 분석 조회

#### 거래 일지 관리
- `GET /api/village/journal/<trainer_name>` - 트레이너 거래 일지 조회
- `GET /api/village/journal/<trainer_name>/recent` - 최근 거래 일지 조회
- `GET /api/village/journal/<trainer_name>/zone/<zone>` - 구역별 거래 일지 조회
- `GET /api/village/journal/<trainer_name>/mayor-guidance` - 촌장 지침 일지 조회
- `GET /api/village/journal/<trainer_name>/ml-decisions` - ML 모델 판단 일지 조회
- `POST /api/village/journal/<trainer_name>/add` - 거래 일지 항목 추가
- `GET /api/village/journal/summary` - 전체 일지 요약 조회

#### 거래 및 에너지
- `POST /api/village/bitcar/energy` - 비트카 에너지 주입
- `POST /api/village/trade/record` - 거래 기록 저장
- `POST /api/village/trust/calculate` - 신뢰도 가중 평균 계산

---

## 🚀 사용 방법

### 1. 서버 실행
```bash
cd bot
python server.py
```

### 2. 시스템 테스트
```bash
python test_village_system.py
```

### 3. API 테스트
```bash
# 마을 상태 조회
curl http://localhost:5057/api/village/status

# 촌장 지침 조회
curl http://localhost:5057/api/village/mayor/guidance

# Scout 상태 조회
curl http://localhost:5057/api/village/scout/status
```

---

## 🎯 시스템 특징

### 🏛️ 촌장의 신뢰도 기반 지침
- ML Model Trust (40%): 개인 판단 우선
- N/B Guild Trust (85%): 길드 전통 존중
- 가중 평균 계산: 개인(60%) + ML(20%) + 길드(20%)

### 🚗 비트카 에너지 시스템
- 각 주민별 맞춤형 에너지 할당
- 마을 에너지 관리 및 소모
- ORANGE 구역으로의 여행 준비

### 📦 실시간 창고 기록
- 모든 거래의 실시간 저장
- 수익/손실 자동 계산
- 학습 데이터 수집 및 분석

### 📝 거래 일지 시스템
- 촌장 지침 기반 거래 결정 기록
- ML 모델 판단과 개인 확신 비교 기록
- 구역별 거래 패턴 분석
- 최근 10개 거래 일지 자동 관리

### 🧠 지속적 학습 시스템
- N/B 코인 기반 경험치 획득
- 성공/실패 패턴 분석
- 전략 자동 최적화

---

## 🏆 성과 지표

### 마을 전체
- **마을 에너지**: 150/100
- **주민 수**: 4명
- **활성 창고**: 4개
- **총 거래 수**: 실시간 업데이트

### 개별 주민 (예: Scout)
- **스킬 레벨**: 2.9
- **현재 포지션**: +0.25% (12분 보유)
- **전략**: 모멘텀
- **N/B 코인**: 0.001

---

## 🔮 향후 발전 계획

### 단기 계획
1. **UI 개선**: 마을 시스템 시각화
2. **알림 시스템**: 실시간 마을 상태 알림
3. **성과 대시보드**: 주민별 성과 추적

### 장기 계획
1. **AI 통합**: 고급 머신러닝 모델 적용
2. **새로운 주민**: 추가 트레이너 역할 도입
3. **마을 확장**: 더 많은 거래 시장 진출

---

## 📞 지원 및 문의

**8BIT 마을** 시스템에 대한 문의사항이나 개선 제안이 있으시면 언제든지 연락해주세요.

**"마을의 번영은 각 주민의 성장에서 시작됩니다"** 🏘️✨
