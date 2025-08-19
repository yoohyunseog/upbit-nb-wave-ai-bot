# 🏰 8BIT Village Trading System 시나리오

## 📖 개요

8BIT Village는 비트코인 거래를 위한 독특한 마을 시스템입니다. 마을 주민들은 각자의 역할과 전략을 가지고 있으며, 촌장의 지침에 따라 거래를 수행합니다.

## 🏘️ 마을 구조

### 🏛️ 촌장 (Mayor)
- **역할**: 마을의 지도자, N/B Guild Branch Manager
- **신뢰도 시스템**:
  - 🤖 ML Model Trust: 40% (낮은 신뢰도)
  - 🏛️ N/B Guild Trust: 82% (높은 신뢰도, 82개 히스토리)
- **지침**: Zone-Side Only (BUY@BLUE / SELL@ORANGE)
- **가중 신뢰도 계산**: (개인 확신도 * 0.6) + (ML Trust * 0.2) + (N/B Guild Trust * 0.2)

### 👥 마을 주민들 (Village Residents)

#### 1. Scout (Explorer) [Gate]
- **레벨**: 1.0-2.9
- **전문 분야**: Quick Signals
- **전략**: 1m & 3m 차트 모니터링, 빠른 기회 포착
- **특징**: momentum 전략, 60% 빈도

#### 2. Guardian (Protector) [Market]
- **레벨**: 1.0
- **전문 분야**: Trend Protection
- **전략**: 5m & 10m 트렌드 보호 및 리스크 관리
- **특징**: meanrev 전략, 50% 빈도

#### 3. Analyst (Strategist) [Tower]
- **레벨**: 1.0
- **전문 분야**: Strategic Analysis
- **전략**: 15m & 30m 패턴 분석
- **특징**: breakout 전략, 70% 빈도

#### 4. Elder (Advisor) [Inn]
- **레벨**: 1.0
- **전문 분야**: Long-term Wisdom
- **전략**: 1h & daily 관점의 지혜 제공
- **특징**: scalping 전략, 40% 빈도

#### 5. Trader_A ~ Trader_F
- **추가 트레이너들**: 각각 고유한 전략과 특성을 가짐

## 🚗 Bitcar Energy System

### 에너지 주입 과정
1. **마을 에너지 수집**: 마을 주민들이 마을의 에너지를 수집
2. **Bitcar 주입**: 수집된 에너지를 개인 Bitcar에 주입
3. **시장 진입**: 주입이 완료된 Bitcar를 타고 비트 시장으로 이동
4. **거래 실행**: 촌장의 지침에 따라 거래 수행

### 에너지 관리
- **HP (Health Points)**: 60-95/100
- **Stamina**: 70-90/100
- **에너지 소모**: 거래 시 에너지 소모
- **회복**: 휴식 시 에너지 회복

## 🏪 Trainer Warehouse System

### 실시간 거래 기록
- **수익/손실 기록**: 모든 거래의 수익률과 손실률 저장
- **패턴 분석**: 거래 패턴 분석 및 학습
- **전략 향상**: 창고 데이터를 통한 전략 개선

### 거래 일지 (Trade Journal)
- **촌장 지침 준수 여부**: 지침을 따랐는지 기록
- **학습 모델 판단**: ML 모델의 결정 사항 기록
- **거래 이유**: 왜 거래했는지, 왜 거래하지 않았는지 기록

## 🧠 ML Model Learning System

### 촌장 지침 학습
- **학습 내용**: "BUY@BLUE / SELL@ORANGE" 규칙 학습
- **자동 학습**: UI의 "🏛️ 촌장 지침 학습" 버튼으로 자동 학습
- **신뢰도 기반**: ML 40% + N/B 길드 82% 신뢰도 기반

### AI 트레이딩 설명
- **거래 이유 설명**: 왜 사지 않는지, 언제 살 것인지, 왜 팔 것인지, 언제 팔 것인지 설명
- **실시간 분석**: 현재 상황에 대한 AI 분석 제공

## 🎯 거래 시스템

### N/B (Narrative/Belief) System
- **주요 신호**: r 값 (0-1) 기반
- **구역 전환**: BLUE/ORANGE 구역 전환
- **임계값**: HIGH (0.55), LOW (0.45)
- **구역별 특성**:
  - 🔵 BLUE Zone: BUY 편향
  - 🟠 ORANGE Zone: SELL/HOLD 편향

### ML Model
- **확인 레이어**: N/B 신호에 대한 확인
- **예측**: 액션 예측 (-1, 0, 1)
- **구역 예측**: BLUE/ORANGE 구역 예측

### Position Lock System
- **사이클 강제**: BUY→SELL 사이클 강제
- **위험 관리**: 포지션 잠금으로 위험 관리

## 🔄 실시간 동기화

### UI 업데이트
- **5초 간격**: 실시간 데이터 업데이트
- **상태 유지**: localStorage를 통한 상태 저장
- **자동 복원**: 페이지 새로고침 시 자동 상태 복원

### 신뢰도 표시
- **실시간 신뢰도**: ML Model Trust, N/B Guild Trust 실시간 표시
- **Trust Balance**: ML: 40% | N/B: 82%
- **N/B Zone Status**: 1h BLUE/ORANGE
- **현재 시간**: 실시간 시간 표시
- **분봉 정보**: 현재 분봉 정보 표시

## 🏛️ 촌장의 실시간 지침

### 구역별 지침
- **🔵 BLUE 구역**: BUY만 허용 (SELL 금지)
- **🟠 ORANGE 구역**: SELL만 허용 (BUY 금지)

### 신뢰도 시스템
```
🤖 ML Model Trust: 40%
🏛️ N/B Guild Trust: 82% (82개 히스토리)
⚖️ Trust Balance: ML: 40% | N/B: 82%
📍 N/B Zone Status: 1h [현재구역]
⏰ 현재 시간: [실시간]
📊 분봉 정보: [현재분봉]
```

## 🎮 UI 시스템

### Guild Members Status
- **실시간 상태**: 각 주민의 현재 상태 표시
- **거래 기록**: 수익률, 승률, 최근 거래 정보
- **에너지 상태**: HP, Stamina 표시
- **촌장 지침 준수**: 지침 준수 여부 표시

### 실시간 동기화
- **N/B Zone**: 🟠ORANGE / 🔵BLUE
- **ML Zone**: 🟠ORANGE / 🔵BLUE
- **동기화 상태**: 실시간 동기화 상태 표시

## 🔧 기술적 구현

### 서버 사이드 (Python/Flask)
- **bot_ctrl**: 전역 상태 관리
- **trade_loop**: 실시간 거래 루프
- **API 엔드포인트**: 다양한 API 제공
- **ML 모델**: 학습 및 예측 시스템

### 클라이언트 사이드 (JavaScript/jQuery)
- **실시간 업데이트**: 5초 간격 데이터 업데이트
- **상태 저장**: localStorage를 통한 상태 관리
- **UI 동기화**: 실시간 UI 동기화
- **jQuery 사용**: AJAX 및 DOM 조작

## 🎯 시스템 특징

### 자동화
- **자동 학습**: 촌장 지침 자동 학습
- **자동 거래**: 조건 충족 시 자동 거래 실행
- **자동 기록**: 모든 거래 자동 기록

### 안전성
- **Position Lock**: 포지션 잠금 시스템
- **에너지 관리**: 에너지 기반 거래 제한
- **신뢰도 기반**: 신뢰도 기반 의사결정

### 확장성
- **모듈화**: 기능별 모듈 분리
- **API 기반**: RESTful API 구조
- **실시간**: 실시간 데이터 처리

## 🚀 향후 발전 방향

### 기능 확장
- **추가 주민**: 더 많은 트레이너 추가
- **고급 전략**: 더 복잡한 거래 전략
- **AI 강화**: 더 정교한 AI 분석

### 시스템 개선
- **성능 최적화**: 더 빠른 처리 속도
- **안정성 향상**: 더 안정적인 시스템
- **사용자 경험**: 더 나은 UI/UX

---

*이 시나리오는 8BIT Village Trading System의 완전한 가이드라인입니다. 시스템의 모든 측면을 포함하며, 지속적으로 업데이트됩니다.*
