import os
import math
import threading
import time
from collections import deque
from dataclasses import asdict
from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
import json
import pyupbit
import pandas as pd
import numpy as np
import joblib
import uuid
import requests
import hashlib
import random
from datetime import datetime, timedelta

from main import load_config, get_candles
from dotenv import load_dotenv
from strategy import decide_signal
from trade import Trader, TradeConfig

# ===== 8BIT 마을 시스템 =====

# 마을 에너지 시스템
VILLAGE_ENERGY = 150
MAX_VILLAGE_ENERGY = 100
ENERGY_ACCUMULATED = 150

# 촌장의 신뢰도 시스템
MAYOR_TRUST_SYSTEM = {
    "ML_Model_Trust": 40,    # 🤖 ML 모델 신뢰도
    "NB_Guild_Trust": 82,    # 🏛️ N/B 길드 신뢰도 (82개 히스토리)
    "last_guidance": None,
    "guidance_history": [],
    "auto_learning_enabled": True,  # 자동 촌장 지침 학습 활성화
    "last_learning_time": None,     # 마지막 학습 시간
    "learning_interval": 3600       # 학습 간격 (1시간)
}

# ===== 마을 출입 일지 시스템 =====
VILLAGE_ENTRY_EXIT_LOG = {
    "total_residents": 10,  # 총 주민 수
    "current_in_village": 4,  # 현재 마을 내 주민 수
    "current_in_orange": 3,   # 현재 ORANGE 구역 주민 수
    "current_in_blue": 3,     # 현재 BLUE 구역 주민 수
    "zone_logs": {
        "ORANGE": {
            "residents": [],  # ORANGE 구역 주민 목록
            "activities": [], # ORANGE 구역 활동 기록
            "entry_exit_log": []  # ORANGE 구역 출입 기록
        },
        "BLUE": {
            "residents": [],  # BLUE 구역 주민 목록
            "activities": [], # BLUE 구역 활동 기록
            "entry_exit_log": []  # BLUE 구역 출입 기록
        },
        "VILLAGE": {
            "residents": [],  # 마을 내 주민 목록
            "activities": [], # 마을 내 활동 기록
            "entry_exit_log": []  # 마을 출입 기록
        }
    },
    "resident_status": {}  # 각 주민별 현재 상태
}

# 마을 주민 시스템 (Guild Members)
VILLAGE_RESIDENTS = {
    "scout": {
        "name": "Scout",
        "hp": 85,
        "maxHp": 100,
        "stamina": 70,
        "maxStamina": 100,
        "location": "Gate",
        "role": "Explorer",
        "trainerCards": ["minute1", "minute3"],
        "specialty": "Quick Signals",
        "description": "Monitors 1m & 3m charts for rapid opportunities",
        "skillLevel": 2.9,
        "experience": 0,
        "learningRate": 0.1,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.6,
        "strategy": "momentum",
        "nbCoins": 0.001,
        "totalNbCoinsEarned": 0.0,
        "totalNbCoinsLost": 0.0,
        "openPosition": None,
        "positionHistory": [],
        "averagePrice": 0.0,
        "totalPositionSize": 0.0
    },
    "guardian": {
        "name": "Guardian",
        "hp": 95,
        "maxHp": 100,
        "stamina": 80,
        "maxStamina": 100,
        "location": "Market",
        "role": "Protector",
        "trainerCards": ["minute5", "minute10"],
        "specialty": "Trend Protection",
        "description": "Protects trends with 5m & 10m charts",
        "skillLevel": 1.0,
        "experience": 0,
        "learningRate": 0.15,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.4,
        "strategy": "mean_reversion",
        "nbCoins": 0.001,
        "totalNbCoinsEarned": 0.0,
        "totalNbCoinsLost": 0.0,
        "openPosition": None,
        "positionHistory": [],
        "averagePrice": 0.0,
        "totalPositionSize": 0.0
    },
    "analyst": {
        "name": "Analyst",
        "hp": 60,
        "maxHp": 100,
        "stamina": 90,
        "maxStamina": 100,
        "location": "Tower",
        "role": "Strategist",
        "trainerCards": ["minute15", "minute30"],
        "specialty": "Strategic Analysis",
        "description": "Develops strategies with 15m & 30m charts",
        "skillLevel": 1.0,
        "experience": 0,
        "learningRate": 0.12,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.3,
        "strategy": "breakout",
        "nbCoins": 0.001,
        "totalNbCoinsEarned": 0.0,
        "totalNbCoinsLost": 0.0,
        "openPosition": None,
        "positionHistory": [],
        "averagePrice": 0.0,
        "totalPositionSize": 0.0
    },
    "elder": {
        "name": "Elder",
        "hp": 75,
        "maxHp": 100,
        "stamina": 85,
        "maxStamina": 100,
        "location": "Inn",
        "role": "Advisor",
        "trainerCards": ["minute60", "day"],
        "specialty": "Long-term Wisdom",
        "description": "Provides wisdom with 1h & daily charts",
        "skillLevel": 1.0,
        "experience": 0,
        "learningRate": 0.08,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.2,
        "strategy": "trend_following",
        "nbCoins": 0.001,
        "totalNbCoinsEarned": 0.0,
        "totalNbCoinsLost": 0.0,
        "openPosition": None,
        "positionHistory": [],
        "averagePrice": 0.0,
        "totalPositionSize": 0.0
    }
}

# 트레이너 창고 시스템
TRAINER_WAREHOUSES = {}

def initialize_trainer_warehouses():
    """트레이너 창고 초기화"""
    for trainer_name, trainer_data in VILLAGE_RESIDENTS.items():
        TRAINER_WAREHOUSES[trainer_name] = {
            "location": f"{trainer_data['location']} Warehouse",
            "capacity": "무제한",
            "real_time_storage": True,
            "trade_records": {
                "real_trades": [],
                "mock_trades": [],
                "current_position": None
            },
            "profit_loss_history": {
                "total_profit": 0,
                "win_rate": 0,
                "total_trades": 0,
                "profitable_trades": 0,
                "losing_trades": 0
            },
            "learning_data": {
                "successful_patterns": [],
                "failed_patterns": [],
                "market_conditions": [],
                "strategy_effectiveness": {}
            },
            # 거래 일지 시스템 추가
            "trade_journal": {
                "recent_entries": [],  # 최근 10개 거래 일지
                "zone_entries": {      # 구역별 거래 일지
                    "ORANGE": [],
                    "BLUE": []
                },
                "mayor_guidance_log": [],  # 촌장 지침 기록
                "ml_model_decisions": []   # ML 모델 판단 기록
            }
        }

# 비트카 에너지 시스템
BITCAR_ENERGY_SYSTEM = {
    "scout": {"energy": 70, "bitcar_model": "Quick Signal Runner"},
    "guardian": {"energy": 80, "bitcar_model": "Trend Protector"},
    "analyst": {"energy": 90, "bitcar_model": "Strategic Analyzer"},
    "elder": {"energy": 85, "bitcar_model": "Wisdom Keeper"}
}

# 마을 시스템 초기화
initialize_trainer_warehouses()

# ===== 8BIT 마을 시스템 함수들 =====

def mayor_trust_guidance():
    """촌장의 신뢰도 기반 지침 생성"""
    global MAYOR_TRUST_SYSTEM
    
    guidance = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "location": "Town Hall",
        "announcement": "마을 주민 여러분, 신뢰도 기반 지침을 전달합니다.",
        
        "trust_analysis": {
            "ml_model_trust": MAYOR_TRUST_SYSTEM["ML_Model_Trust"],
            "nb_guild_trust": MAYOR_TRUST_SYSTEM["NB_Guild_Trust"],
            "interpretation": "신뢰도 분석 결과"
        },
        
        "guidance": {
            "zone": "ORANGE",
            "official_strategy": "신중한 방어적 접근",
            "trust_adjusted_strategy": "개인 판단 우선, ML 모델 참고",
            "energy_requirement": "최소 50 에너지",
            "special_instructions": "신뢰도 시스템 준수"
        }
    }
    
    MAYOR_TRUST_SYSTEM["last_guidance"] = guidance
    MAYOR_TRUST_SYSTEM["guidance_history"].append(guidance)
    
    return guidance

def generate_ai_trading_explanation(trainer_name, current_action, current_zone, r_value, confidence, position_status):
    """AI 거래 판단 설명 생성"""
    
    explanations = {
        "BUY": {
            "BLUE": {
                "reason": "✅ 촌장 지침 준수: BLUE 구역에서 BUY 허용",
                "timing": "🕐 즉시 실행 가능 (구역 조건 충족)",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}%",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (BLUE 구역 유지)",
                "strategy": "📈 공격적 매수 전략 (BLUE 구역 특성)"
            },
            "ORANGE": {
                "reason": "❌ 촌장 지침 위반: ORANGE 구역에서 BUY 금지",
                "timing": "⏳ BLUE 구역 전환 대기 필요 (r값 0.45 이하)",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}% (낮음)",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (ORANGE 구역)",
                "strategy": "⚠️ 개인 판단 우선 (촌장 지침 무시)"
            }
        },
        "SELL": {
            "BLUE": {
                "reason": "❌ 촌장 지침 위반: BLUE 구역에서 SELL 금지",
                "timing": "⏳ ORANGE 구역 전환 대기 필요 (r값 0.55 이상)",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}% (낮음)",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (BLUE 구역)",
                "strategy": "⚠️ 개인 판단 우선 (촌장 지침 무시)"
            },
            "ORANGE": {
                "reason": "✅ 촌장 지침 준수: ORANGE 구역에서 SELL 허용",
                "timing": "🕐 즉시 실행 가능 (구역 조건 충족)",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}%",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (ORANGE 구역 유지)",
                "strategy": "📉 방어적 매도 전략 (ORANGE 구역 특성)"
            }
        },
        "HOLD": {
            "BLUE": {
                "reason": "⏸️ BLUE 구역에서 관망 (BUY 대기)",
                "timing": "🕐 적절한 진입 시점 대기",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}%",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (BLUE 구역)",
                "strategy": "👀 관망 전략 (더 나은 진입점 대기)"
            },
            "ORANGE": {
                "reason": "⏸️ ORANGE 구역에서 관망 (SELL 대기)",
                "timing": "🕐 적절한 청산 시점 대기",
                "confidence": f"🤖 ML 모델 신뢰도: {confidence}%",
                "zone_status": f"📊 현재 r값: {r_value:.3f} (ORANGE 구역)",
                "strategy": "👀 관망 전략 (더 나은 청산점 대기)"
            }
        }
    }
    
    # 포지션 상태에 따른 추가 설명
    position_explanation = ""
    if position_status == "HAS_POSITION":
        if current_action == "SELL":
            position_explanation = "💼 포지션 보유 중 - 청산 시점 판단"
        elif current_action == "BUY":
            position_explanation = "💼 포지션 보유 중 - 추가 매수 고려"
        elif current_action == "HOLD":
            position_explanation = "💼 포지션 보유 중 - 관망 전략"
    else:
        position_explanation = "💼 포지션 없음 - 진입 시점 판단"
    
    base_explanation = explanations.get(current_action, {}).get(current_zone, {})
    
    # 기본값 설정으로 "알 수 없음" 방지
    default_reason = f"현재 {current_zone} 구역에서 {current_action} 판단"
    default_timing = "적절한 시점 모니터링 중"
    default_confidence = f"🤖 ML 모델 신뢰도: {confidence}%"
    default_zone_status = f"📊 현재 r값: {r_value:.3f} ({current_zone} 구역)"
    default_strategy = f"기본 {current_action} 전략"
    
    return {
        "trainer": trainer_name,
        "current_action": current_action,
        "current_zone": current_zone,
        "r_value": r_value,
        "confidence": confidence,
        "position_status": position_status,
        "explanation": {
            "reason": base_explanation.get("reason", default_reason),
            "timing": base_explanation.get("timing", default_timing),
            "confidence": base_explanation.get("confidence", default_confidence),
            "zone_status": base_explanation.get("zone_status", default_zone_status),
            "strategy": base_explanation.get("strategy", default_strategy),
            "position": position_explanation
        },
        "timestamp": datetime.now().isoformat()
    }

def auto_mayor_guidance_learning():
    """자동 촌장 지침 학습 실행"""
    global MAYOR_TRUST_SYSTEM
    
    try:
        # 자동 학습이 비활성화되어 있으면 스킵
        if not MAYOR_TRUST_SYSTEM.get("auto_learning_enabled", True):
            return
        
        current_time = time.time()
        last_learning_time = MAYOR_TRUST_SYSTEM.get("last_learning_time")
        learning_interval = MAYOR_TRUST_SYSTEM.get("learning_interval", 3600)  # 1시간
        
        # 학습 간격 체크
        if last_learning_time and (current_time - last_learning_time) < learning_interval:
            return
        
        print("🏛️ 자동 촌장 지침 학습 시작...")
        
        # 촌장 지침 학습 모델 훈련 실행
        cfg = load_config()
        window = 50
        ema_fast = 10
        ema_slow = 30
        horizon = 5
        count = 1800
        interval = cfg.candle
        
        df = get_candles(cfg.market, interval, count=count)
        
        # 촌장 지침 기반 특성 생성
        feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        
        # 촌장 지침 라벨링: Zone-Side Only
        r = _compute_r_from_ohlcv(df, window)
        HIGH = float(os.getenv('NB_HIGH', '0.55'))
        LOW = float(os.getenv('NB_LOW', '0.45'))
        labels = np.zeros(len(df), dtype=int)
        zone = None
        r_vals = r.values.tolist()
        
        for i in range(len(df)):
            rv = r_vals[i] if i < len(r_vals) else 0.5
            if zone not in ('BLUE','ORANGE'):
                zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            # hysteresis updates
            if zone == 'BLUE' and rv >= HIGH:
                zone = 'ORANGE'
            elif zone == 'ORANGE' and rv <= LOW:
                zone = 'BLUE'
            
            # 촌장 지침: BUY@BLUE / SELL@ORANGE
            if zone == 'BLUE':
                labels[i] = 1  # BUY
            elif zone == 'ORANGE':
                labels[i] = -1  # SELL
            else:
                labels[i] = 0  # HOLD
        
        idx_map = { ts: i for i, ts in enumerate(df.index) }
        y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        
        # 모델 훈련
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report, confusion_matrix
        
        # 특성 선택
        X = feat[['r', 'w', 'ema_diff', 'zone_flag', 'dist_high', 'dist_low', 'zone_conf']]
        
        # 모델 훈련
        model = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=3)
        model.fit(X.values, y)
        
        # 평가
        yhat = model.predict(X.values)
        report = classification_report(y, yhat, output_dict=True, zero_division=0)
        
        # 모델 저장
        pack = {
            'model': model,
            'window': window,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'horizon': horizon,
            'interval': interval,
            'label_mode': 'mayor_guidance',
            'trained_at': int(current_time * 1000),
            'feature_names': list(X.columns),
            'metrics': {
                'report': report
            }
        }
        
        # 모델 저장
        try:
            joblib.dump(pack, _model_path_for(interval))
            print(f"✅ 자동 촌장 지침 학습 완료 - 모델 저장됨")
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            joblib.dump(pack, ML_MODEL_PATH)
        
        # 학습 시간 업데이트
        MAYOR_TRUST_SYSTEM["last_learning_time"] = current_time
        
        # 학습 결과 로그
        classes = {
            '-1': int((y==-1).sum()),  # SELL (ORANGE)
            '0': int((y==0).sum()),    # HOLD
            '1': int((y==1).sum())     # BUY (BLUE)
        }
        print(f"📊 자동 학습 결과 - BUY: {classes['1']}, HOLD: {classes['0']}, SELL: {classes['-1']}")
        
    except Exception as e:
        print(f"❌ 자동 촌장 지침 학습 실패: {e}")

def calculate_weighted_confidence(personal_confidence, ml_trust, nb_guild_trust):
    """신뢰도 가중 평균 계산"""
    return (personal_confidence * 0.6) + (ml_trust * 0.2) + (nb_guild_trust * 0.2)

def real_time_trade_recording(trainer_name, trade_data):
    """실시간 거래 기록 저장"""
    global TRAINER_WAREHOUSES
    
    if trainer_name not in TRAINER_WAREHOUSES:
        return {"error": "트레이너를 찾을 수 없습니다."}
    
    warehouse = TRAINER_WAREHOUSES[trainer_name]
    
    # 거래 기록 저장
    trade_record = {
        'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
        'action': trade_data.get('action'),
        'price': trade_data.get('price'),
        'quantity': trade_data.get('quantity', 0),
        'pnl': trade_data.get('pnl', 0),
        'strategy': trade_data.get('strategy'),
        'zone': trade_data.get('zone'),
        'confidence': trade_data.get('confidence', 0),
        'trainer': trainer_name
    }
    
    if trade_data.get('is_real', False):
        warehouse['trade_records']['real_trades'].append(trade_record)
    else:
        warehouse['trade_records']['mock_trades'].append(trade_record)
    
    # 수익/손실 업데이트
    update_profit_loss_history(warehouse, trade_data)
    
    # 학습 데이터 수집
    collect_learning_data(warehouse, trade_data)
    
    return {"message": f"{trainer_name}의 거래 기록이 창고에 저장되었습니다."}

def update_profit_loss_history(warehouse, trade_data):
    """수익/손실 기록 업데이트"""
    history = warehouse['profit_loss_history']
    
    # 거래 수 증가
    history['total_trades'] += 1
    
    pnl = trade_data.get('pnl', 0)
    
    # 수익/손실 계산
    if pnl > 0:
        history['profitable_trades'] += 1
        history['total_profit'] += pnl
    else:
        history['losing_trades'] += 1
        history['total_profit'] += pnl
    
    # 승률 계산
    if history['total_trades'] > 0:
        history['win_rate'] = (history['profitable_trades'] / history['total_trades']) * 100

def collect_learning_data(warehouse, trade_data):
    """학습 데이터 수집"""
    learning_data = warehouse['learning_data']
    
    pattern_data = {
        'market_condition': trade_data.get('market_condition', 'unknown'),
        'strategy': trade_data.get('strategy', 'unknown'),
        'timing': trade_data.get('timing', 'unknown'),
        'confidence': trade_data.get('confidence', 0),
        'zone': trade_data.get('zone', 'unknown'),
        'timestamp': trade_data.get('timestamp', datetime.now().isoformat())
    }
    
    # 성공 패턴 수집
    if trade_data.get('pnl', 0) > 0:
        learning_data['successful_patterns'].append(pattern_data)
    else:
        # 실패 패턴 수집
        pattern_data['lesson_learned'] = trade_data.get('lesson_learned', '분석 필요')
        learning_data['failed_patterns'].append(pattern_data)

def inject_village_energy_to_bitcar(trainer_name, energy_amount):
    """마을 에너지를 비트카에 주입"""
    global VILLAGE_ENERGY, BITCAR_ENERGY_SYSTEM
    
    if VILLAGE_ENERGY >= energy_amount:
        if trainer_name in BITCAR_ENERGY_SYSTEM:
            BITCAR_ENERGY_SYSTEM[trainer_name]["energy"] = energy_amount
            VILLAGE_ENERGY -= energy_amount
            return f"{trainer_name}의 비트카에 {energy_amount} 에너지 주입 완료"
        else:
            return f"{trainer_name} 트레이너를 찾을 수 없습니다."
    else:
        return "마을 에너지 부족"

def get_trainer_warehouse_status(trainer_name):
    """트레이너 창고 상태 조회"""
    global TRAINER_WAREHOUSES
    
    if trainer_name not in TRAINER_WAREHOUSES:
        return {"error": "트레이너를 찾을 수 없습니다."}
    
    warehouse = TRAINER_WAREHOUSES[trainer_name]
    
    return {
        "trainer": trainer_name,
        "warehouse_location": warehouse["location"],
        "storage_usage": f"{len(warehouse['trade_records']['real_trades']) + len(warehouse['trade_records']['mock_trades'])} 거래 기록",
        "data_integrity": "100%",
        "last_backup": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "real_time_sync": "활성화",
        "profit_loss_summary": warehouse['profit_loss_history']
    }

def analyze_warehouse_data(trainer_name):
    """창고 데이터 기반 전략 분석"""
    global TRAINER_WAREHOUSES
    
    if trainer_name not in TRAINER_WAREHOUSES:
        return {"error": "트레이너를 찾을 수 없습니다."}
    
    warehouse = TRAINER_WAREHOUSES[trainer_name]
    
    analysis = {
        "trainer": trainer_name,
        "profitability_analysis": {
            "total_profit": warehouse['profit_loss_history']['total_profit'],
            "win_rate": warehouse['profit_loss_history']['win_rate'],
            "total_trades": warehouse['profit_loss_history']['total_trades']
        },
        "strategy_effectiveness": {
            "successful_patterns_count": len(warehouse['learning_data']['successful_patterns']),
            "failed_patterns_count": len(warehouse['learning_data']['failed_patterns'])
        },
        "recommendations": generate_strategy_recommendations(warehouse)
    }
    
    return analysis

def generate_strategy_recommendations(warehouse):
    """전략 개선 권장사항 생성"""
    successful_count = len(warehouse['learning_data']['successful_patterns'])
    failed_count = len(warehouse['learning_data']['failed_patterns'])
    
    if successful_count > failed_count:
        return "현재 전략이 효과적입니다. 계속 유지하세요."
    elif failed_count > successful_count:
        return "전략 개선이 필요합니다. 실패 패턴을 분석해보세요."
    else:
        return "전략이 균형을 이루고 있습니다. 더 많은 데이터를 수집해보세요."

# ===== 거래 일지 시스템 =====

def add_trade_journal_entry(trainer_name, entry_data):
    """거래 일지 항목 추가"""
    global TRAINER_WAREHOUSES
    
    if trainer_name not in TRAINER_WAREHOUSES:
        return {"error": "트레이너를 찾을 수 없습니다."}
    
    warehouse = TRAINER_WAREHOUSES[trainer_name]
    journal = warehouse['trade_journal']
    
    # 기본 일지 항목 생성
    journal_entry = {
        'timestamp': entry_data.get('timestamp', datetime.now().isoformat()),
        'trainer': trainer_name,
        'action': entry_data.get('action', 'UNKNOWN'),
        'zone': entry_data.get('zone', 'UNKNOWN'),
        'price': entry_data.get('price', 0),
        'pnl': entry_data.get('pnl', 0),
        'strategy': entry_data.get('strategy', 'unknown'),
        'confidence': entry_data.get('confidence', 0),
        'mayor_guidance': entry_data.get('mayor_guidance', ''),
        'ml_decision': entry_data.get('ml_decision', ''),
        'reasoning': entry_data.get('reasoning', ''),
        'lesson_learned': entry_data.get('lesson_learned', ''),
        'trade_type': entry_data.get('trade_type', 'mock')  # 'real' or 'mock'
    }
    
    # 최근 일지에 추가 (최대 10개 유지)
    journal['recent_entries'].append(journal_entry)
    if len(journal['recent_entries']) > 10:
        journal['recent_entries'] = journal['recent_entries'][-10:]
    
    # 구역별 일지에 추가
    zone = entry_data.get('zone', 'UNKNOWN')
    if zone in journal['zone_entries']:
        journal['zone_entries'][zone].append(journal_entry)
        if len(journal['zone_entries'][zone]) > 10:
            journal['zone_entries'][zone] = journal['zone_entries'][zone][-10:]
    
    # 촌장 지침 기록
    if entry_data.get('mayor_guidance'):
        mayor_entry = {
            'timestamp': journal_entry['timestamp'],
            'trainer': trainer_name,
            'guidance': entry_data['mayor_guidance'],
            'zone': zone,
            'action': entry_data.get('action', 'UNKNOWN')
        }
        journal['mayor_guidance_log'].append(mayor_entry)
        if len(journal['mayor_guidance_log']) > 10:
            journal['mayor_guidance_log'] = journal['mayor_guidance_log'][-10:]
    
    # ML 모델 판단 기록
    if entry_data.get('ml_decision'):
        ml_entry = {
            'timestamp': journal_entry['timestamp'],
            'trainer': trainer_name,
            'decision': entry_data['ml_decision'],
            'confidence': entry_data.get('confidence', 0),
            'zone': zone,
            'action': entry_data.get('action', 'UNKNOWN')
        }
        journal['ml_model_decisions'].append(ml_entry)
        if len(journal['ml_model_decisions']) > 10:
            journal['ml_model_decisions'] = journal['ml_model_decisions'][-10:]
    
    return {"message": f"{trainer_name}의 거래 일지에 항목이 추가되었습니다.", "entry": journal_entry}

def get_trade_journal(trainer_name, journal_type="recent", zone=None):
    """거래 일지 조회"""
    global TRAINER_WAREHOUSES
    
    if trainer_name not in TRAINER_WAREHOUSES:
        return {"error": "트레이너를 찾을 수 없습니다."}
    
    warehouse = TRAINER_WAREHOUSES[trainer_name]
    journal = warehouse['trade_journal']
    
    if journal_type == "recent":
        return {
            "trainer": trainer_name,
            "journal_type": "recent",
            "entries": journal['recent_entries'],
            "count": len(journal['recent_entries'])
        }
    elif journal_type == "zone" and zone:
        if zone in journal['zone_entries']:
            return {
                "trainer": trainer_name,
                "journal_type": "zone",
                "zone": zone,
                "entries": journal['zone_entries'][zone],
                "count": len(journal['zone_entries'][zone])
            }
        else:
            return {"error": f"구역 {zone}의 일지를 찾을 수 없습니다."}
    elif journal_type == "mayor_guidance":
        return {
            "trainer": trainer_name,
            "journal_type": "mayor_guidance",
            "entries": journal['mayor_guidance_log'],
            "count": len(journal['mayor_guidance_log'])
        }
    elif journal_type == "ml_decisions":
        return {
            "trainer": trainer_name,
            "journal_type": "ml_decisions",
            "entries": journal['ml_model_decisions'],
            "count": len(journal['ml_model_decisions'])
        }
    else:
        return {"error": "지원하지 않는 일지 유형입니다."}

def create_mayor_guidance_entry(trainer_name, zone, action, reasoning):
    """촌장 지침 기반 거래 일지 생성"""
    guidance_messages = {
        "ORANGE": {
            "BUY": "ORANGE 구역에서 촌장의 방어적 지침을 무시하고 개인 확신으로 BUY 실행",
            "SELL": "ORANGE 구역에서 촌장의 지침에 따라 신중한 SELL 실행",
            "HOLD": "ORANGE 구역에서 촌장의 방어적 지침에 따라 HOLD 결정"
        },
        "BLUE": {
            "BUY": "BLUE 구역에서 촌장의 공격적 지침에 따라 자신감 있는 BUY 실행",
            "SELL": "BLUE 구역에서 촌장의 지침을 무시하고 개인 판단으로 SELL 실행",
            "HOLD": "BLUE 구역에서 촌장의 공격적 지침을 고려하되 HOLD 결정"
        }
    }
    
    guidance = guidance_messages.get(zone, {}).get(action, "촌장의 지침을 고려한 거래 결정")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'trainer': trainer_name,
        'action': action,
        'zone': zone,
        'mayor_guidance': guidance,
        'reasoning': reasoning,
        'trade_type': 'mock'
    }

def create_ml_decision_entry(trainer_name, zone, action, ml_confidence, personal_confidence):
    """ML 모델 판단 기반 거래 일지 생성"""
    ml_trust = MAYOR_TRUST_SYSTEM["ML_Model_Trust"]
    
    if ml_confidence < ml_trust:
        decision = f"ML 모델 신뢰도({ml_confidence}%)가 낮아 개인 판단({personal_confidence}%) 우선"
    else:
        decision = f"ML 모델 신뢰도({ml_confidence}%)가 높아 ML 판단 채택"
    
    return {
        'timestamp': datetime.now().isoformat(),
        'trainer': trainer_name,
        'action': action,
        'zone': zone,
        'ml_decision': decision,
        'ml_confidence': ml_confidence,
        'personal_confidence': personal_confidence,
        'trade_type': 'mock'
    }

# ===== 마을 출입 일지 시스템 함수들 =====

def generate_resident_activity_log(resident_name, zone, activity_type, duration=None):
    """주민 활동 일지 생성 (AI 자동 작성)"""
    activities = {
        "ORANGE": {
            "rest": [
                f"{resident_name}이 ORANGE 구역에서 {duration}간 휴식을 취하며 시장 상황을 관찰했습니다.",
                f"{resident_name}이 ORANGE 구역의 적대적 환경에서 {duration}간 안전한 휴식을 취했습니다.",
                f"{resident_name}이 ORANGE 구역에서 {duration}간 신중한 관찰을 통해 시장 동향을 파악했습니다."
            ],
            "training": [
                f"{resident_name}이 ORANGE 구역에서 {duration}간 방어적 트레이닝을 수행했습니다.",
                f"{resident_name}이 ORANGE 구역에서 {duration}간 신중한 거래 연습을 했습니다.",
                f"{resident_name}이 ORANGE 구역에서 {duration}간 베타 관계 형성에 주의하며 트레이닝했습니다."
            ],
            "observation": [
                f"{resident_name}이 ORANGE 구역에서 {duration}간 적대적 시장 환경을 관찰했습니다.",
                f"{resident_name}이 ORANGE 구역에서 {duration}간 빠른 수익 실현 기회를 모색했습니다.",
                f"{resident_name}이 ORANGE 구역에서 {duration}간 방어적 입장을 유지하며 시장을 분석했습니다."
            ]
        },
        "BLUE": {
            "rest": [
                f"{resident_name}이 BLUE 구역에서 {duration}간 편안한 휴식을 취하며 시장 기회를 기다렸습니다.",
                f"{resident_name}이 BLUE 구역의 우호적 환경에서 {duration}간 여유로운 휴식을 취했습니다.",
                f"{resident_name}이 BLUE 구역에서 {duration}간 자신감을 회복하며 휴식을 취했습니다."
            ],
            "training": [
                f"{resident_name}이 BLUE 구역에서 {duration}간 공격적 트레이닝을 수행했습니다.",
                f"{resident_name}이 BLUE 구역에서 {duration}간 자신감 있는 거래 연습을 했습니다.",
                f"{resident_name}이 BLUE 구역에서 {duration}간 알파 접근법으로 트레이닝했습니다."
            ],
            "observation": [
                f"{resident_name}이 BLUE 구역에서 {duration}간 우호적 시장 환경을 관찰했습니다.",
                f"{resident_name}이 BLUE 구역에서 {duration}간 강한 매수 기회를 모색했습니다.",
                f"{resident_name}이 BLUE 구역에서 {duration}간 공격적 입장을 유지하며 시장을 분석했습니다."
            ]
        },
        "VILLAGE": {
            "rest": [
                f"{resident_name}이 마을에서 {duration}간 편안한 휴식을 취했습니다.",
                f"{resident_name}이 마을에서 {duration}간 동료들과 대화하며 경험을 나눴습니다.",
                f"{resident_name}이 마을에서 {duration}간 촌장의 지침을 받으며 휴식을 취했습니다."
            ],
            "training": [
                f"{resident_name}이 마을에서 {duration}간 이론적 트레이닝을 수행했습니다.",
                f"{resident_name}이 마을에서 {duration}간 동료들과 함께 전략을 논의했습니다.",
                f"{resident_name}이 마을에서 {duration}간 촌장의 멘토링을 받으며 학습했습니다."
            ],
            "observation": [
                f"{resident_name}이 마을에서 {duration}간 시장 동향을 분석했습니다.",
                f"{resident_name}이 마을에서 {duration}간 창고의 거래 기록을 검토했습니다.",
                f"{resident_name}이 마을에서 {duration}간 향후 전략을 계획했습니다."
            ]
        }
    }
    
    import random
    activity_list = activities.get(zone, {}).get(activity_type, [f"{resident_name}이 {zone}에서 활동했습니다."])
    return random.choice(activity_list)

def record_resident_entry_exit(resident_name, from_zone, to_zone, activity_type="training", duration="몇 시간"):
    """주민 출입 기록"""
    global VILLAGE_ENTRY_EXIT_LOG
    
    timestamp = datetime.now().isoformat()
    
    # 출입 기록 생성
    entry_exit_record = {
        'timestamp': timestamp,
        'resident': resident_name,
        'from_zone': from_zone,
        'to_zone': to_zone,
        'activity_type': activity_type,
        'duration': duration,
        'activity_description': generate_resident_activity_log(resident_name, from_zone, activity_type, duration)
    }
    
    # 출발 구역에서 제거
    if from_zone in VILLAGE_ENTRY_EXIT_LOG['zone_logs']:
        if resident_name in VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['residents']:
            VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['residents'].remove(resident_name)
        VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['entry_exit_log'].append(entry_exit_record)
        if len(VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['entry_exit_log']) > 10:
            VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['entry_exit_log'] = VILLAGE_ENTRY_EXIT_LOG['zone_logs'][from_zone]['entry_exit_log'][-10:]
    
    # 도착 구역에 추가
    if to_zone in VILLAGE_ENTRY_EXIT_LOG['zone_logs']:
        if resident_name not in VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['residents']:
            VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['residents'].append(resident_name)
        VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['entry_exit_log'].append(entry_exit_record)
        if len(VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['entry_exit_log']) > 10:
            VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['entry_exit_log'] = VILLAGE_ENTRY_EXIT_LOG['zone_logs'][to_zone]['entry_exit_log'][-10:]
    
    # 주민 상태 업데이트
    VILLAGE_ENTRY_EXIT_LOG['resident_status'][resident_name] = {
        'current_zone': to_zone,
        'last_activity': activity_type,
        'last_update': timestamp,
        'duration_in_current_zone': duration
    }
    
    # 구역별 인원 수 업데이트
    _update_zone_population_counts()
    
    return entry_exit_record

def _update_zone_population_counts():
    """구역별 인원 수 업데이트"""
    global VILLAGE_ENTRY_EXIT_LOG
    
    VILLAGE_ENTRY_EXIT_LOG['current_in_village'] = len(VILLAGE_ENTRY_EXIT_LOG['zone_logs']['VILLAGE']['residents'])
    VILLAGE_ENTRY_EXIT_LOG['current_in_orange'] = len(VILLAGE_ENTRY_EXIT_LOG['zone_logs']['ORANGE']['residents'])
    VILLAGE_ENTRY_EXIT_LOG['current_in_blue'] = len(VILLAGE_ENTRY_EXIT_LOG['zone_logs']['BLUE']['residents'])

def get_zone_entry_exit_log(zone):
    """구역별 출입 일지 조회"""
    global VILLAGE_ENTRY_EXIT_LOG
    
    if zone not in VILLAGE_ENTRY_EXIT_LOG['zone_logs']:
        return {"error": f"구역 {zone}를 찾을 수 없습니다."}
    
    return {
        "zone": zone,
        "current_residents": VILLAGE_ENTRY_EXIT_LOG['zone_logs'][zone]['residents'],
        "entry_exit_log": VILLAGE_ENTRY_EXIT_LOG['zone_logs'][zone]['entry_exit_log'],
        "total_entries": len(VILLAGE_ENTRY_EXIT_LOG['zone_logs'][zone]['entry_exit_log'])
    }

def get_all_residents_status():
    """모든 주민 상태 조회"""
    global VILLAGE_ENTRY_EXIT_LOG
    
    return {
        "total_residents": VILLAGE_ENTRY_EXIT_LOG['total_residents'],
        "current_in_village": VILLAGE_ENTRY_EXIT_LOG['current_in_village'],
        "current_in_orange": VILLAGE_ENTRY_EXIT_LOG['current_in_orange'],
        "current_in_blue": VILLAGE_ENTRY_EXIT_LOG['current_in_blue'],
        "resident_status": VILLAGE_ENTRY_EXIT_LOG['resident_status']
    }

def simulate_resident_movement():
    """주민 이동 시뮬레이션 (자동화된 시스템)"""
    import random
    import time
    
    # 주민 목록 (10명)
    residents = [
        "Scout", "Guardian", "Analyst", "Elder",
        "Trader_A", "Trader_B", "Trader_C", "Trader_D", "Trader_E", "Trader_F"
    ]
    
    zones = ["VILLAGE", "ORANGE", "BLUE"]
    activities = ["rest", "training", "observation"]
    durations = ["몇 시간", "하루", "며칠", "일주일", "몇 주", "한 달"]
    
    # 랜덤 주민 선택
    resident = random.choice(residents)
    
    # 현재 상태 확인
    current_zone = VILLAGE_ENTRY_EXIT_LOG['resident_status'].get(resident, {}).get('current_zone', 'VILLAGE')
    
    # 새로운 구역 선택 (현재 구역과 다른 곳)
    available_zones = [z for z in zones if z != current_zone]
    new_zone = random.choice(available_zones)
    
    # 활동 유형과 기간 선택
    activity = random.choice(activities)
    duration = random.choice(durations)
    
    # 출입 기록
    record = record_resident_entry_exit(resident, current_zone, new_zone, activity, duration)
    
    return record

# ===== 8BIT 마을 API 엔드포인트 (Flask 앱 정의 후에 이동됨) =====
def get_village_status():
    """마을 전체 상태 조회"""
    return jsonify({
        "village_name": "8BIT 마을",
        "mayor": "촌장 (N/B 길드 지점장)",
        "village_energy": VILLAGE_ENERGY,
        "max_village_energy": MAX_VILLAGE_ENERGY,
        "energy_accumulated": ENERGY_ACCUMULATED,
        "residents_count": len(VILLAGE_RESIDENTS),
        "warehouses_count": len(TRAINER_WAREHOUSES),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# 중복 라우트 제거됨 - Flask 앱 정의 이후로 이동됨

# ===== 기존 코드 계속 =====

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return (
        """
        <html>
          <head><meta charset="utf-8"><title>8BIT Bot Server</title></head>
          <body style="font-family:system-ui,Segoe UI,Arial; padding:20px;">
            <h2>8BIT PyUpbit Bot Server</h2>
            <p>Server is running. API endpoint: <a href="/api/state">/api/state</a></p>
            <div id="s">Loading...</div>
            <script>
              fetch('/api/state').then(r=>r.json()).then(j=>{
                document.getElementById('s').textContent = JSON.stringify(j);
              }).catch(()=>{ document.getElementById('s').textContent='Failed to load /api/state'; });
            </script>
          </body>
        </html>
        """,
        200,
        {"Content-Type": "text/html; charset=utf-8"}
    )


@app.route("/ui")
def serve_ui():
    # Serve the embedded chart UI from bot/static/ui.html
    return send_from_directory('static', 'ui.html')

@app.route("/game")
def serve_game():
    # Serve the village simulator from bot/game/village.html
    return send_from_directory('game', 'village.html')

@app.route('/static/<path:filename>')
def serve_static(filename: str):
    return send_from_directory('static', filename)

@app.route('/api/save-chart-data', methods=['POST'])
def save_chart_data():
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data or 'data' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        
        filename = data['filename']
        chart_data = data['data']
        
        # Validate filename
        import re
        if not re.match(r'^chart_data_[a-zA-Z0-9_-]+\.json$', filename):
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Create data directory structure
        import datetime
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, '..', 'data', 'chart_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories by date
        date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        date_dir = os.path.join(data_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        
        # Create subdirectories by interval
        interval = chart_data.get('interval', 'unknown')
        interval_dir = os.path.join(date_dir, interval)
        os.makedirs(interval_dir, exist_ok=True)
        
        # Full file path
        filepath = os.path.join(interval_dir, filename)
        
        # Write JSON data with pretty formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, indent=2, ensure_ascii=False)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'fileSize': file_size,
            'totalCandles': chart_data.get('totalCandles', 0),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

# ===== 8BIT 마을 API 엔드포인트 =====

@app.route('/api/village/status')
def get_village_status():
    """마을 전체 상태 조회"""
    return jsonify({
        "village_name": "8BIT 마을",
        "mayor": "촌장 (N/B 길드 지점장)",
        "village_energy": VILLAGE_ENERGY,
        "max_village_energy": MAX_VILLAGE_ENERGY,
        "energy_accumulated": ENERGY_ACCUMULATED,
        "residents_count": len(VILLAGE_RESIDENTS),
        "warehouses_count": len(TRAINER_WAREHOUSES),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/village/mayor/guidance')
def get_mayor_guidance():
    """촌장의 신뢰도 기반 지침 조회"""
    return jsonify(mayor_trust_guidance())

@app.route('/api/village/residents')
def get_village_residents():
    """마을 주민 정보 조회"""
    return jsonify({
        "residents": VILLAGE_RESIDENTS,
        "total_count": len(VILLAGE_RESIDENTS)
    })

@app.route('/api/village/resident/<trainer_name>')
def get_resident_info(trainer_name):
    """특정 주민 정보 조회"""
    if trainer_name not in VILLAGE_RESIDENTS:
        return jsonify({"error": "주민을 찾을 수 없습니다."}), 404
    
    return jsonify({
        "resident": VILLAGE_RESIDENTS[trainer_name],
        "warehouse_status": get_trainer_warehouse_status(trainer_name)
    })

@app.route('/api/village/warehouse/<trainer_name>')
def get_warehouse_info(trainer_name):
    """트레이너 창고 정보 조회"""
    if trainer_name not in TRAINER_WAREHOUSES:
        return jsonify({"error": "창고를 찾을 수 없습니다."}), 404
    
    return jsonify({
        "warehouse": TRAINER_WAREHOUSES[trainer_name],
        "status": get_trainer_warehouse_status(trainer_name)
    })

@app.route('/api/village/warehouse/<trainer_name>/analysis')
def get_warehouse_analysis(trainer_name):
    """창고 데이터 분석 조회"""
    return jsonify(analyze_warehouse_data(trainer_name))

@app.route('/api/village/bitcar/energy', methods=['POST'])
def inject_bitcar_energy():
    """비트카 에너지 주입"""
    data = request.get_json()
    trainer_name = data.get('trainer_name')
    energy_amount = data.get('energy_amount', 50)
    
    if not trainer_name:
        return jsonify({"error": "트레이너 이름이 필요합니다."}), 400
    
    result = inject_village_energy_to_bitcar(trainer_name, energy_amount)
    return jsonify({"message": result})

@app.route('/api/village/trade/record', methods=['POST'])
def record_trade():
    """거래 기록 저장"""
    data = request.get_json()
    trainer_name = data.get('trainer_name')
    
    if not trainer_name:
        return jsonify({"error": "트레이너 이름이 필요합니다."}), 400
    
    result = real_time_trade_recording(trainer_name, data)
    return jsonify(result)

@app.route('/api/village/trust/calculate', methods=['POST'])
def calculate_trust():
    """신뢰도 가중 평균 계산"""
    data = request.get_json()
    personal_confidence = data.get('personal_confidence', 0)
    ml_trust = data.get('ml_trust', MAYOR_TRUST_SYSTEM["ML_Model_Trust"])
    nb_guild_trust = data.get('nb_guild_trust', MAYOR_TRUST_SYSTEM["NB_Guild_Trust"])
    
    weighted_confidence = calculate_weighted_confidence(personal_confidence, ml_trust, nb_guild_trust)
    
    return jsonify({
        "personal_confidence": personal_confidence,
        "ml_trust": ml_trust,
        "nb_guild_trust": nb_guild_trust,
        "weighted_confidence": weighted_confidence,
        "weights": {
            "personal": 0.6,
            "ml_model": 0.2,
            "nb_guild": 0.2
        }
    })

@app.route('/api/village/system/overview')
def get_system_overview():
    """마을 시스템 전체 개요"""
    return jsonify({
        "system_name": "8BIT 마을 트레이딩 시스템",
        "description": "촌장의 지침에 따라 운영되는 AI 트레이더 마을",
        "components": {
            "mayor_system": "촌장 신뢰도 기반 지침 시스템",
            "residents": "10명의 트레이너 주민",
            "warehouses": "실시간 거래 기록 창고",
            "bitcar_system": "비트카 에너지 주입 시스템",
            "auto_learning": "자동 촌장 지침 학습 시스템"
        },
        "current_status": {
            "village_energy": VILLAGE_ENERGY,
            "residents_count": len(VILLAGE_RESIDENTS),
            "warehouses_count": len(TRAINER_WAREHOUSES),
            "auto_learning_enabled": MAYOR_TRUST_SYSTEM.get("auto_learning_enabled", True)
        }
    })

@app.route('/api/village/scout/status')
def get_scout_status():
    """Scout의 현재 상태 조회 (특별 API)"""
    if 'scout' not in VILLAGE_RESIDENTS:
        return jsonify({"error": "Scout를 찾을 수 없습니다."}), 404
    
    scout = VILLAGE_RESIDENTS['scout']
    warehouse = TRAINER_WAREHOUSES['scout']
    
    # Scout의 현재 포지션 정보 (예시)
    current_position = {
        "entry_time": "2025-01-27 08:15:00",
        "entry_price": 161000000,
        "current_price": 161401000,
        "pnl": "+0.25%",
        "duration": "12분",
        "strategy": "momentum"
    }
    
    # 거래 일지 정보 추가
    recent_journal = get_trade_journal('scout', "recent")
    mayor_journal = get_trade_journal('scout', "mayor_guidance")
    ml_journal = get_trade_journal('scout', "ml_decisions")
    
    return jsonify({
        "trainer": "Scout",
        "status": {
            "name": scout['name'],
            "hp": scout['hp'],
            "stamina": scout['stamina'],
            "location": scout['location'],
            "role": scout['role'],
            "specialty": scout['specialty'],
            "skillLevel": scout['skillLevel'],
            "strategy": scout['strategy'],
            "nbCoins": scout['nbCoins']
        },
        "current_position": current_position,
        "warehouse_summary": {
            "total_trades": warehouse['profit_loss_history']['total_trades'],
            "total_profit": warehouse['profit_loss_history']['total_profit'],
            "win_rate": warehouse['profit_loss_history']['win_rate'],
            "successful_patterns": len(warehouse['learning_data']['successful_patterns']),
            "failed_patterns": len(warehouse['learning_data']['failed_patterns'])
        },
        "mayor_guidance": {
            "ml_model_trust": MAYOR_TRUST_SYSTEM["ML_Model_Trust"],
            "nb_guild_trust": MAYOR_TRUST_SYSTEM["NB_Guild_Trust"],
            "current_zone": "ORANGE",
            "guidance": "신중한 방어적 접근, 개인 판단 우선"
        },
        "trade_journal": {
            "recent_entries_count": recent_journal.get("count", 0),
            "mayor_guidance_count": mayor_journal.get("count", 0),
            "ml_decisions_count": ml_journal.get("count", 0),
            "latest_entry": recent_journal.get("entries", [])[-1] if recent_journal.get("entries") else None
        }
    })

@app.route('/api/village/current-zone')
def get_current_zone():
    """현재 구역 정보 조회 - UI에서 이미 계산된 값 사용"""
    try:
        # UI에서 이미 계산된 값들을 그대로 사용
        r_value = bot_ctrl.get('r_value', 0.5)
        nb_zone = bot_ctrl.get('nb_zone', 'ORANGE')
        last_signal = bot_ctrl.get('last_signal', 'HOLD')
        position = bot_ctrl.get('position', 'FLAT')
        
        # ML 구역은 N/B 구역과 동일하게 설정 (UI에서 이미 계산됨)
        ml_zone = nb_zone
        
        # 현재 활성 구역 (N/B 시스템 기준)
        current_zone = nb_zone
        
        # 촌장의 신뢰도 정보 추가
        ml_trust = MAYOR_TRUST_SYSTEM.get("ML_Model_Trust", 40)
        nb_trust = MAYOR_TRUST_SYSTEM.get("NB_Guild_Trust", 82)  # 82개 히스토리로 업데이트
        
        return jsonify({
            'current_zone': current_zone,
            'nb_zone': nb_zone,
            'ml_zone': ml_zone,
            'last_signal': last_signal,
            'position': position,
            'r_value': r_value,
            'ml_trust': ml_trust,
            'nb_trust': nb_trust,
            'timestamp': int(time.time() * 1000)
        })
    except Exception as e:
        return jsonify({'error': f'구역 정보 조회 실패: {str(e)}'}), 500

@app.route('/api/village/auto-learning/toggle', methods=['POST'])
def toggle_auto_learning():
    """자동 촌장 지침 학습 토글"""
    global MAYOR_TRUST_SYSTEM
    
    try:
        # 현재 상태 토글
        current_status = MAYOR_TRUST_SYSTEM.get("auto_learning_enabled", True)
        MAYOR_TRUST_SYSTEM["auto_learning_enabled"] = not current_status
        
        return jsonify({
            'ok': True,
            'auto_learning_enabled': MAYOR_TRUST_SYSTEM["auto_learning_enabled"],
            'message': f"자동 촌장 지침 학습이 {'활성화' if MAYOR_TRUST_SYSTEM['auto_learning_enabled'] else '비활성화'}되었습니다.",
            'learning_interval': MAYOR_TRUST_SYSTEM.get("learning_interval", 3600),
            'last_learning_time': MAYOR_TRUST_SYSTEM.get("last_learning_time")
        })
        
    except Exception as e:
        return jsonify({'ok': False, 'error': f'자동 학습 토글 실패: {str(e)}'}), 500

@app.route('/api/ml/train-mayor-guidance', methods=['POST'])
def train_mayor_guidance_model():
    """촌장 지침 학습 모델 훈련"""
    try:
        payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
        
        # 촌장 지침 학습 파라미터
        window = int(payload.get('window', 50))
        ema_fast = int(payload.get('ema_fast', 10))
        ema_slow = int(payload.get('ema_slow', 30))
        horizon = int(payload.get('horizon', 5))
        count = int(payload.get('count', 1800))
        interval = payload.get('interval') or load_config().candle
        
        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        
        # 촌장 지침 기반 특성 생성
        feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        
        # 촌장 지침 라벨링: Zone-Side Only
        r = _compute_r_from_ohlcv(df, window)
        HIGH = float(os.getenv('NB_HIGH', '0.55'))
        LOW = float(os.getenv('NB_LOW', '0.45'))
        labels = np.zeros(len(df), dtype=int)
        zone = None
        r_vals = r.values.tolist()
        
        for i in range(len(df)):
            rv = r_vals[i] if i < len(r_vals) else 0.5
            if zone not in ('BLUE','ORANGE'):
                zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            # hysteresis updates
            if zone == 'BLUE' and rv >= HIGH:
                zone = 'ORANGE'
            elif zone == 'ORANGE' and rv <= LOW:
                zone = 'BLUE'
            
            # 촌장 지침: BUY@BLUE / SELL@ORANGE
            if zone == 'BLUE':
                labels[i] = 1  # BUY
            elif zone == 'ORANGE':
                labels[i] = -1  # SELL
            else:
                labels[i] = 0  # HOLD
        
        idx_map = { ts: i for i, ts in enumerate(df.index) }
        y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        
        # 모델 훈련
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        from sklearn.metrics import classification_report, confusion_matrix
        
        # 특성 선택
        X = feat[['r', 'w', 'ema_diff', 'zone_flag', 'dist_high', 'dist_low', 'zone_conf']]
        
        # 시계열 교차 검증
        tscv = TimeSeriesSplit(n_splits=3)
        model = GradientBoostingClassifier(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=3)
        
        # 훈련
        model.fit(X.values, y)
        
        # 평가
        yhat = model.predict(X.values)
        report = classification_report(y, yhat, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, yhat, labels=[-1,0,1]).tolist()
        
        # 모델 저장
        pack = {
            'model': model,
            'window': window,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'horizon': horizon,
            'interval': interval,
            'label_mode': 'mayor_guidance',
            'trained_at': int(time.time() * 1000),
            'feature_names': list(X.columns),
            'metrics': {
                'report': report,
                'confusion': cm
            }
        }
        
        # 모델 저장
        try:
            joblib.dump(pack, _model_path_for(interval))
        except Exception:
            joblib.dump(pack, ML_MODEL_PATH)
        
        return jsonify({
            'ok': True,
            'message': '촌장 지침 학습 모델 훈련 완료',
            'label_mode': 'mayor_guidance',
            'classes': {
                '-1': int((y==-1).sum()),  # SELL (ORANGE)
                '0': int((y==0).sum()),    # HOLD
                '1': int((y==1).sum())     # BUY (BLUE)
            },
            'report': report,
            'confusion': cm
        })
        
    except Exception as e:
        return jsonify({'ok': False, 'error': f'촌장 지침 학습 실패: {str(e)}'}), 500

@app.route('/api/village/ai-explanation/<trainer_name>')
def get_ai_trading_explanation(trainer_name):
    """AI 거래 판단 설명 조회"""
    try:
        # 현재 구역 정보 가져오기
        current_zone = bot_ctrl.get('nb_zone', 'ORANGE')
        last_signal = bot_ctrl.get('last_signal', 'HOLD')
        position = bot_ctrl.get('position', 'FLAT')
        
        # r값 계산 (실제 구현에서는 실제 r값을 가져와야 함)
        r_value = 0.5  # 기본값, 실제로는 계산된 값 사용
        
        # 포지션 상태 판단
        position_status = "HAS_POSITION" if position != "FLAT" else "NO_POSITION"
        
        # 현재 액션 판단
        current_action = last_signal if last_signal in ['BUY', 'SELL', 'HOLD'] else 'HOLD'
        
        # 신뢰도 계산 (예시)
        confidence = 60  # 실제로는 계산된 신뢰도 사용
        
        # AI 거래 설명 생성
        explanation = generate_ai_trading_explanation(
            trainer_name, 
            current_action, 
            current_zone, 
            r_value, 
            confidence, 
            position_status
        )
        
        return jsonify(explanation)
        
    except Exception as e:
        return jsonify({'error': f'AI 거래 설명 생성 실패: {str(e)}'}), 500

state = {
    "price": 0.0,
    "signal": "HOLD",
    "ema_fast": 10,
    "ema_slow": 30,
    "market": "KRW-BTC",
    "candle": "minute10",
    "history": deque(maxlen=200),  # (ts, price)
}

# ML training state
ml_state = {
    'train_count': 0,
}

# Grouped NB observations (time-bucketed)
GROUP_BUCKET_SEC = int(os.getenv('NB_GROUP_BUCKET_SEC', '60'))  # group by 1m default
GROUP_MIN_SIZE = int(os.getenv('NB_GROUP_MIN_SIZE', '25'))
_nb_groups: dict[int, list] = {}
_npc_hashes: set[str] = set()

# Zone reputation learned from narratives/policy (-1 .. +1)
_zone_reputation: dict[str, dict] = {
    'ORANGE': {'score': 0.0, 'updated_ms': None, 'notes': []},
    'BLUE':   {'score': 0.0, 'updated_ms': None, 'notes': []},
}

# Information trust configuration
_trust_config: dict = {
    'ml_trust': 50.0,  # ML Model trust level (0-100)
    'nb_trust': 50.0,  # N/B Guild trust level (0-100)
    'last_updated': None
}

# Trainer storage warehouses (각 트레이너별 저장 창고)
_trainer_storage: dict[str, dict] = {
    'Scout': {
        'coins': 0.0,  # 보유 코인 수량
        'entry_price': 0.0,  # 매수 가격
        'last_update': None,  # 마지막 업데이트 시간
        'total_profit': 0.0,  # 총 수익
        'ticks': 0,  # 거래 틱 카운터
        'trades': []  # 거래 기록
    },
    'Guardian': {
        'coins': 0.0,
        'entry_price': 0.0,
        'last_update': None,
        'total_profit': 0.0,
        'ticks': 0,
        'trades': []
    },
    'Analyst': {
        'coins': 0.0,
        'entry_price': 0.0,
        'last_update': None,
        'total_profit': 0.0,
        'ticks': 0,
        'trades': []
    },
    'Elder': {
        'coins': 0.0,
        'entry_price': 0.0,
        'last_update': None,
        'total_profit': 0.0,
        'ticks': 0,
        'trades': []
    }
}

def _narrative_store_path() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'narratives.jsonl')
    except Exception:
        return 'narratives.jsonl'

def _trainer_storage_path() -> str:
    """트레이너 저장 창고 데이터 파일 경로"""
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'trainer_storage.json')
    except Exception:
        return 'trainer_storage.json'

def _trust_config_path() -> str:
    """신뢰도 설정 파일 경로"""
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'trust_config.json')
    except Exception:
        return 'trust_config.json'

def _load_trainer_storage() -> dict:
    """트레이너 저장 창고 데이터 로드"""
    try:
        path = _trainer_storage_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 기존 데이터와 새 구조 병합
                for trainer in ['Scout', 'Guardian', 'Analyst', 'Elder']:
                    if trainer not in data:
                        data[trainer] = {
                            'coins': 0.0,
                            'entry_price': 0.0,
                            'last_update': None,
                            'total_profit': 0.0,
                            'ticks': 0,
                            'trades': []
                        }
                    # 기존 데이터에 틱 카운터가 없으면 추가
                    if 'ticks' not in data[trainer]:
                        data[trainer]['ticks'] = 0
                return data
    except Exception:
        pass
    return _trainer_storage.copy()

def _save_trainer_storage():
    """트레이너 저장 창고 데이터 저장"""
    try:
        path = _trainer_storage_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_trainer_storage, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _load_trust_config() -> dict:
    """신뢰도 설정 로드"""
    try:
        with open(_trust_config_path(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'ml_trust': 50.0, 'nb_trust': 50.0, 'last_updated': None}

def _save_trust_config():
    """신뢰도 설정 저장"""
    try:
        _trust_config['last_updated'] = int(time.time() * 1000)
        with open(_trust_config_path(), 'w', encoding='utf-8') as f:
            json.dump(_trust_config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving trust config: {e}")

def _update_trainer_storage(trainer: str, action: str, price: float, size: float, profit: float = 0.0):
    """트레이너 저장 창고 업데이트"""
    try:
        if trainer not in _trainer_storage:
            return
        
        storage = _trainer_storage[trainer]
        now = int(time.time() * 1000)
        
        # 틱 카운터 초기화 (없으면)
        if 'ticks' not in storage:
            storage['ticks'] = 0
        
        if action.upper() == 'BUY':
            # 매수: 코인 추가
            storage['coins'] += size
            storage['entry_price'] = price
            storage['last_update'] = now
            storage['ticks'] += 1  # 거래 시 틱 증가
            storage['trades'].append({
                'ts': now,
                'action': 'BUY',
                'price': price,
                'size': size,
                'profit': 0.0
            })
            
        elif action.upper() == 'SELL':
            # 매도: 코인 차감 및 수익 계산
            if storage['coins'] >= size:
                storage['coins'] -= size
                if storage['entry_price'] > 0:
                    profit = (price - storage['entry_price']) * size
                    storage['total_profit'] += profit
                
                storage['last_update'] = now
                storage['ticks'] += 1  # 거래 시 틱 증가
                storage['trades'].append({
                    'ts': now,
                    'action': 'SELL',
                    'price': price,
                    'size': size,
                    'profit': profit
                })
                
                # 모든 코인을 매도한 경우 entry_price 초기화
                if storage['coins'] <= 0:
                    storage['entry_price'] = 0.0
        
        # 거래 기록은 최근 100개만 유지
        if len(storage['trades']) > 100:
            storage['trades'] = storage['trades'][-100:]
            
        _save_trainer_storage()
        
    except Exception:
        pass

def _update_zone_reputation(zone: str, delta: float, note: str | None = None) -> dict:
    try:
        z = str(zone or '').upper()
        if z not in _zone_reputation:
            _zone_reputation[z] = {'score': 0.0, 'updated_ms': None, 'notes': []}
        row = _zone_reputation[z]
        row['score'] = float(max(-1.0, min(1.0, float(row.get('score', 0.0)) + float(delta))))
        row['updated_ms'] = int(time.time()*1000)
        if note:
            notes = row.get('notes') or []
            notes.append(str(note))
            # cap notes list
            if len(notes) > 20:
                notes = notes[-20:]
            row['notes'] = notes
        return row
    except Exception:
        return {'score': 0.0}

def _bucket_ts(ts_ms: int | None = None, bucket_sec: int | None = None) -> int:
    try:
        b = int(bucket_sec or GROUP_BUCKET_SEC)
        t = int((ts_ms or int(time.time()*1000)) / 1000)
        return (t // b) * b
    except Exception:
        return int(time.time())

def _record_group_observation(interval: str, window: int, r_val: float,
                              pct_blue: float, pct_orange: float, ts_ms: int | None = None):
    try:
        bt = _bucket_ts(ts_ms, GROUP_BUCKET_SEC)
        row = {
            'ts': int(ts_ms or int(time.time()*1000)),
            'bucket': int(bt),
            'interval': str(interval),
            'window': int(window),
            'r': float(r_val),
            'pct_blue': float(pct_blue),
            'pct_orange': float(pct_orange),
        }
        _nb_groups.setdefault(bt, []).append(row)
        # trim old buckets to keep memory bounded
        if len(_nb_groups) > 1000:
            for k in sorted(list(_nb_groups.keys()))[:-900]:
                _nb_groups.pop(k, None)
    except Exception:
        pass

# In-memory order log for UI markers
orders = deque(maxlen=500)  # each item: {ts, side, price, size, paper, market}
# ML signal log (in-memory; optionally persisted)
signals = []  # each: {id, ts, zone, extreme, price, pct_major, slope_bp, horizon, pred_nb, interval, market, score0, realized_score}

# N/B COIN tracking per candle bucket
_nb_coin_store: dict[str, dict] = {}
_nb_coin_counter: dict[str, int] = {}          # per-interval coin count (card-level)
_nb_open_entry: dict[str, float] = {}           # per-interval open entry price for BUY→SELL cycle
_nb_rest_until: dict[str, int] = {}             # per-interval rest window end bucket (exclusive)
_village_energy: dict[str, dict] = {}           # per-interval energy state: { E: float(0..100), last_ms: int, idle_bars: int }

# Village Council (trainer consensus) state
_council_state: dict = {
    'ts': None,
    'intervals': {},   # iv -> { chosen, intent, feasible, zone, slope_bp }
    'consensus': {'intent': 'HOLD', 'votes': {}},
}
_council_thread: threading.Thread | None = None
_council_running: bool = False

def _energy_state(iv: str) -> dict:
    try:
        iv = str(iv)
        st = _village_energy.get(iv)
        if not st:
            st = { 'E': 50.0, 'last_ms': int(time.time()*1000), 'idle_bars': 0 }
            _village_energy[iv] = st
        return st
    except Exception:
        return { 'E': 50.0, 'last_ms': int(time.time()*1000), 'idle_bars': 0 }

def _energy_tick(iv: str) -> float:
    try:
        st = _energy_state(iv)
        now = int(time.time()*1000)
        dt_sec = max(0.0, (now - int(st.get('last_ms') or now)) / 1000.0)
        decay = float(os.getenv('ENERGY_DECAY_PER_SEC', '0.001'))
        st['E'] = float(max(0.0, min(99999.0, float(st.get('E', 50.0)) - decay * dt_sec)))
        st['last_ms'] = now
        return float(st['E'])
    except Exception:
        return 0.0

def _energy_adjust(iv: str, delta: float, reason: str | None = None) -> float:
    try:
        st = _energy_state(iv)
        _energy_tick(iv)
        st['E'] = float(max(0.0, min(99999.0, float(st.get('E', 50.0)) + float(delta))))
        if reason:
            st['last_reason'] = str(reason)
        return float(st['E'])
    except Exception:
        return 0.0

@app.route('/api/village/state')
def api_village_state():
    try:
        iv = request.args.get('interval') if request.args else None
        if not iv:
            iv = state.get('candle') or load_config().candle
        # tick and read
        E = _energy_tick(str(iv))
        st = _energy_state(str(iv))
        last_reason = st.get('last_reason')
        # attach learned zone reputation snapshot
        rep = {
            'BLUE': dict(_zone_reputation.get('BLUE', {})),
            'ORANGE': dict(_zone_reputation.get('ORANGE', {})),
        }
        # compose minimal treasury snapshot via existing summary
        try:
            total_owned = int(sum(int(v) for v in _nb_coin_counter.values()))
        except Exception:
            total_owned = 0
        # KRW/price/ buyable from summary helper (reuse logic inline)
        price_per_coin = int(getattr(_resolve_config(), 'order_krw', 5100))
        krw = 0.0
        try:
            cfg = _resolve_config()
            if (not cfg.paper) and cfg.access_key and cfg.secret_key:
                upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
                if upbit:
                    krw = float(upbit.get_balance('KRW') or 0.0)
        except Exception:
            krw = 0.0
        buyable = int(krw // max(1, price_per_coin))
        return jsonify({ 'ok': True, 'interval': str(iv), 'energy': E, 'last_reason': last_reason, 'reputation': rep, 'treasury': { 'krw': krw, 'coins': total_owned, 'price_per_coin': price_per_coin, 'buyable': buyable } })
    except Exception as e:
        return jsonify({ 'ok': False, 'error': str(e) }), 500

@app.route('/api/village/energy/fill', methods=['POST'])
def api_village_energy_fill():
    try:
        iv = request.args.get('interval') if request.args else None
        if not iv:
            iv = state.get('candle') or load_config().candle
        
        # Fill energy to 99999
        current_energy = _energy_tick(str(iv))
        energy_needed = 99999.0 - current_energy
        new_energy = _energy_adjust(str(iv), energy_needed, 'manual_fill')
        
        print(f"✅ Village energy filled: {current_energy:.1f}% → {new_energy:.1f}% (interval: {iv})")
        return jsonify({ 'ok': True, 'interval': str(iv), 'previous_energy': current_energy, 'new_energy': new_energy })
    except Exception as e:
        print(f"❌ Error filling village energy: {e}")
        return jsonify({ 'ok': False, 'error': str(e) }), 500

def _interval_to_sec(iv: str) -> int:
    try:
        s = str(iv or 'minute1')
        if s.startswith('minute'):
            return int(s.replace('minute','')) * 60
        if s == 'day':
            return 86400
        if s == 'week':
            return 7*86400
        if s == 'month':
            return 30*86400
    except Exception:
        pass
    return 60

def _bucket_ts_interval(ts_ms: int | None, iv: str) -> int:
    try:
        sec = _interval_to_sec(iv)
        t = int((ts_ms or int(time.time()*1000)) / 1000)
        return (t // sec) * sec
    except Exception:
        return int(time.time())

def _coin_key(interval: str, market: str, bucket_sec: int) -> str:
    return f"{market}|{interval}|{bucket_sec}"

def _coin_store_path() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'nb_coins_store.json')
    except Exception:
        return 'nb_coins_store.json'

def _npc_store_path() -> str:
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, 'npc_messages.jsonl')
    except Exception:
        return 'npc_messages.jsonl'

def _load_npc_hashes() -> int:
    try:
        path = _npc_store_path()
        if not os.path.exists(path):
            return 0
        cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    h = str(obj.get('hash') or _hash_text(str(obj.get('text') or '')))
                    if h not in _npc_hashes:
                        _npc_hashes.add(h)
                        cnt += 1
                except Exception:
                    continue
        return cnt
    except Exception:
        return 0

def _save_nb_coins() -> bool:
    try:
        path = _coin_store_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_nb_coin_store, f, ensure_ascii=False)
        return True
    except Exception:
        return False

def _load_nb_coins() -> int:
    try:
        path = _coin_store_path()
        if not os.path.exists(path):
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            _nb_coin_store.clear()
            _nb_coin_store.update(data)
            return len(_nb_coin_store)
        return 0
    except Exception:
        return 0

def _hash_text(s: str) -> str:
    try:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()
    except Exception:
        return str(uuid.uuid4())

def _npc_add(msg: dict) -> bool:
    try:
        text = str(msg.get('text') or '')
        h = _hash_text(text)
        if h in _npc_hashes:
            return False
        _npc_hashes.add(h)
        msg['id'] = str(uuid.uuid4())
        msg['hash'] = h
        path = _npc_store_path()
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        return True
    except Exception:
        return False

def _ensure_nb_coin(interval: str, market: str, bucket_sec: int) -> dict:
    key = _coin_key(interval, market, bucket_sec)
    if key not in _nb_coin_store:
        _nb_coin_store[key] = {
            'bucket': int(bucket_sec),
            'interval': str(interval),
            'market': str(market),
            'side': 'NONE',  # NONE | BUY | SELL
            'orders': [],
            'ts': int(time.time()*1000),
            'reasons': [],            # list of strings describing why no trade yet
            'checked_ts': None,       # last time we evaluated trade conditions
            'blocks': {},             # aggregated counters per reason
            'coin_count': int(_nb_coin_counter.get(str(interval), 0)),
            'rest_until': int(_nb_rest_until.get(str(interval), 0)),
        }
        # trim to last ~2000 coins
        if len(_nb_coin_store) > 2500:
            for k in sorted(_nb_coin_store.keys())[:-2000]:
                try:
                    del _nb_coin_store[k]
                except Exception:
                    pass
        try:
            _save_nb_coins()
        except Exception:
            pass
    return _nb_coin_store[key]

def _mark_nb_coin(interval: str, market: str, side: str, ts_ms: int | None = None, order_obj: dict | None = None):
    try:
        b = _bucket_ts_interval(ts_ms or int(time.time()*1000), interval)
        coin = _ensure_nb_coin(interval, market, b)
        # Once any order happens in the bucket, mark the side (prefer SELL over BUY if multiple; or latest wins)
        coin['side'] = str(side).upper()
        
        # Store position size for BUY orders
        if str(side).upper() == 'BUY' and order_obj:
            try:
                size = float(order_obj.get('size') or 0.0)
                if size > 0:
                    coin['position_size'] = size
                    coin['entry_price'] = float(order_obj.get('price') or 0.0)
            except Exception:
                pass
        
        if order_obj:
            try:
                coin['orders'].append({
                    'ts': int(order_obj.get('ts') or int(time.time()*1000)),
                    'side': str(order_obj.get('side') or side).upper(),
                    'price': float(order_obj.get('price') or 0.0),
                    'size': float(order_obj.get('size') or 0.0),
                    'paper': bool(order_obj.get('paper')),
                })
            except Exception:
                pass
    except Exception:
        pass
    try:
        _save_nb_coins()
    except Exception:
        pass

def _apply_coin_accounting(interval: str, price: float, side: str):
    try:
        iv = str(interval)
        if side.upper() == 'BUY' and (price or 0) > 0:
            if iv not in _nb_open_entry:
                _nb_open_entry[iv] = float(price)
                # On BUY success, save 1 coin
                prev = int(_nb_coin_counter.get(iv, 0))
                _nb_coin_counter[iv] = prev + 1
                # If this is the first coin (0 -> 1), schedule rest window
                try:
                    if prev <= 0 and (_nb_coin_counter.get(iv, 0) or 0) >= 1:
                        rest_on = (os.getenv('REST_AFTER_FIRST_COIN', 'true').lower() == 'true')
                        rest_bars = int(os.getenv('REST_BARS', '3'))
                        if rest_on and rest_bars > 0:
                            b = _bucket_ts_interval(int(time.time()*1000), iv)
                            _nb_rest_until[iv] = int(b + rest_bars)
                except Exception:
                    pass
        elif side.upper() == 'SELL' and (price or 0) > 0:
            if iv in _nb_open_entry:
                entry = float(_nb_open_entry.get(iv) or 0.0)
                profit = (float(price) - entry) > 0
                if profit:
                    # profit: add one more coin
                    _nb_coin_counter[iv] = int(_nb_coin_counter.get(iv, 0)) + 1
                    try:
                        _energy_adjust(iv, +1.5, 'sell_profit')
                    except Exception:
                        pass
                else:
                    # loss: remove coin(s); stronger penalty if Elder guidance was violated
                    # Elder guidance: BUY only in BLUE, SELL only in ORANGE
                    try:
                        z = str((_nb_coin_store.get(_coin_key(iv, load_config().market, _bucket_ts_interval(int(time.time()*1000), iv)) ) or {}).get('zone') or '').upper()
                    except Exception:
                        z = ''
                    violated = False
                    try:
                        # If last known zone is BLUE and we SOLD, or ORANGE and we BOUGHT (opposite of guidance)
                        violated = (z == 'BLUE' and True)  # SELL in BLUE is violation; if z unknown keep False
                    except Exception:
                        violated = False
                    penalty = int(os.getenv('ELDER_VIOLATION_PENALTY', '2'))
                    if violated:
                        _nb_coin_counter[iv] = int(_nb_coin_counter.get(iv, 0)) - max(1, penalty)
                        try:
                            _energy_adjust(iv, -2.0, 'sell_loss_violation')
                        except Exception:
                            pass
                    else:
                        _nb_coin_counter[iv] = int(_nb_coin_counter.get(iv, 0)) - 1
                        try:
                            _energy_adjust(iv, -1.0, 'sell_loss')
                        except Exception:
                            pass
                # close the open cycle
                _nb_open_entry.pop(iv, None)
        # reflect latest coin_count into current bucket coin if exists
        try:
            b = _bucket_ts_interval(int(time.time()*1000), iv)
            key = _coin_key(iv, load_config().market, b)
            if key in _nb_coin_store:
                _nb_coin_store[key]['coin_count'] = int(_nb_coin_counter.get(iv, 0))
        except Exception:
            pass
    except Exception:
        pass


def _score_strategies(interval: str) -> dict:
    """Return simple heuristic scores for four strategies and a suggested action.
    Heads: trend, meanrev, breakout, pullback
    """
    try:
        iv = str(interval)
        cfg = _resolve_config()
        df = get_candles(cfg.market, iv, count=max(200, cfg.ema_slow+50))
        window = int(load_nb_params().get('window', 50))
        ins = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, iv, None) or {}
        zone = str(ins.get('zone') or '').upper()
        rv = float(ins.get('r', 0.5) or 0.5)
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55')); LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH,LOW = 0.55,0.45
        rng = max(1e-9, HIGH-LOW)
        # slope approx
        slope_bp = 0.0
        try:
            n_tail = max(20, min(120, window))
            closes = df['close'].astype(float).tail(n_tail)
            if len(closes) >= 5:
                import numpy as _np
                y = _np.log(closes.replace(0, _np.nan)).fillna(method='bfill').fillna(method='ffill').values
                x = _np.arange(len(y), dtype=float)
                b1 = _np.polyfit(x, y, 1)[0]
                slope_bp = float(b1*10000.0)
        except Exception:
            slope_bp = 0.0
        # features for heads
        trend_align = (zone=='BLUE' and slope_bp>0) or (zone=='ORANGE' and slope_bp<0)
        near_extreme = (zone=='BLUE' and (rv-LOW) <= (0.15*rng)) or (zone=='ORANGE' and (HIGH-rv) <= (0.15*rng))
        try:
            hi = float(df['high'].rolling(window).max().iloc[-1]); lo = float(df['low'].rolling(window).min().iloc[-1]); c = float(df['close'].iloc[-1])
        except Exception:
            hi=lo=c=0.0
        breakout_up = c >= (hi*0.999)
        breakout_dn = c <= (lo*1.001)
        eg = float(ins.get('extreme_gap', 0.0) or 0.0); age = int(ins.get('zone_extreme_age', 0) or 0)
        try:
            pb_r = float(os.getenv('PULLBACK_R', '0.02'))
            pb_bars = int(os.getenv('PULLBACK_BARS', '2'))
        except Exception:
            pb_r, pb_bars = 0.02, 2
        pull_ok = (eg >= pb_r) and (age >= pb_bars)
        # scores (0..1)
        s_trend = 1.0 if trend_align else 0.2
        s_mean = 1.0 if ((zone=='BLUE' and slope_bp<0 and near_extreme) or (zone=='ORANGE' and slope_bp>0 and near_extreme)) else 0.2
        s_break = 1.0 if (breakout_up or breakout_dn) else 0.2
        s_pull = 1.0 if pull_ok else 0.2
        # Reputation-aware adjustment: penalize actions that conflict with learned zone reputation
        rep_orange = float((_zone_reputation.get('ORANGE') or {}).get('score') or 0.0)
        rep_blue = float((_zone_reputation.get('BLUE') or {}).get('score') or 0.0)
        rep_penalty = 0.15
        if zone == 'ORANGE' and rep_orange < 0:
            s_trend *= (1.0 + rep_orange * rep_penalty)
            s_mean  *= (1.0 + rep_orange * rep_penalty)
            s_pull  *= (1.0 + rep_orange * rep_penalty)
        if zone == 'BLUE' and rep_blue < 0:
            s_trend *= (1.0 + rep_blue * rep_penalty)
            s_mean  *= (1.0 + rep_blue * rep_penalty)
            s_pull  *= (1.0 + rep_blue * rep_penalty)
        head_scores = {'trend': s_trend, 'meanrev': s_mean, 'breakout': s_break, 'pullback': s_pull}
        # choose best (favor recent realized pnl via simple tie-break)
        chosen = max(head_scores.items(), key=lambda x: (x[1], 0))[0]
        # intent
        intent = 'HOLD'
        if chosen=='trend':
            intent = 'BUY' if zone=='BLUE' and slope_bp>0 else ('SELL' if zone=='ORANGE' and slope_bp<0 else 'HOLD')
        elif chosen=='meanrev':
            intent = 'BUY' if zone=='BLUE' and slope_bp<0 and near_extreme else ('SELL' if zone=='ORANGE' and slope_bp>0 and near_extreme else 'HOLD')
        elif chosen=='breakout':
            intent = 'BUY' if breakout_up else ('SELL' if breakout_dn else 'HOLD')
        elif chosen=='pullback':
            intent = 'BUY' if zone=='BLUE' and pull_ok else ('SELL' if zone=='ORANGE' and pull_ok else 'HOLD')
        # feasibility
        coin = int(_nb_coin_counter.get(iv, 0))
        price_per_coin = int(getattr(cfg, 'order_krw', 5100))
        avail_krw = 0.0
        try:
            upbit = None
            if (not cfg.paper) and cfg.access_key and cfg.secret_key:
                upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
            if upbit:
                avail_krw = float(upbit.get_balance('KRW') or 0.0)
        except Exception:
            avail_krw = 0.0
        buyable = int(avail_krw // max(1, price_per_coin))
        feasible = {'can_buy': buyable>0, 'can_sell': coin>0}
        return {
            'ok': True,
            'interval': iv,
            'insight': ins,
            'slope_bp': slope_bp,
            'head_scores': head_scores,
            'chosen': chosen,
            'intent': intent,
            'feasible': feasible,
            'coin_count': coin,
            'buyable_by_krw': buyable,
            'reputation': {
                'BLUE': float(rep_blue),
                'ORANGE': float(rep_orange),
            },
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


@app.route('/api/trainer/suggest')
def api_trainer_suggest():
    try:
        iv = request.args.get('interval') if request.args else None
        if not iv:
            iv = state.get('candle') or load_config().candle
        res = _score_strategies(str(iv))
        # update council view for this interval
        try:
            if res.get('ok'):
                _council_state['ts'] = int(time.time()*1000)
                ivs = _council_state.setdefault('intervals', {})
                ivs[str(iv)] = {
                    'chosen': res.get('chosen'),
                    'intent': res.get('intent'),
                    'feasible': res.get('feasible'),
                    'zone': (res.get('insight') or {}).get('zone'),
                    'slope_bp': res.get('slope_bp'),
                }
                # derive a simple consensus by majority of intents among feasible ones
                votes = {}
                for _, row in ivs.items():
                    intent = str(row.get('intent') or 'HOLD').upper()
                    feas = row.get('feasible') or {}
                    if intent == 'BUY' and not feas.get('can_buy'): intent = 'HOLD'
                    if intent == 'SELL' and not feas.get('can_sell'): intent = 'HOLD'
                    votes[intent] = votes.get(intent, 0) + 1
                if votes:
                    intent_cons = max(votes.items(), key=lambda x: x[1])[0]
                    _council_state['consensus'] = { 'intent': intent_cons, 'votes': votes }
        except Exception:
            pass
        return jsonify(res), (200 if res.get('ok') else 500)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/narrative/add', methods=['POST'])
def api_narrative_add():
    try:
        payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
        text = str(payload.get('text') or '')
        zone = str(payload.get('zone') or '').upper()
        # simple sentiment mapping: if explicit negative, penalize; else small nudge
        negative = bool(payload.get('negative') or ('negative' in text.lower()) or ('risk' in text.lower()) or ('lock' in text.lower()))
        delta = float(payload.get('delta') or (-0.3 if negative else 0.1))
        row = _update_zone_reputation(zone, delta, note=(payload.get('title') or text[:120]))
        # persist narrative
        obj = {
            'id': str(uuid.uuid4()),
            'ts': int(time.time()*1000),
            'zone': zone,
            'text': text,
            'delta': delta,
            'rep_after': float(row.get('score', 0.0)),
        }
        try:
            with open(_narrative_store_path(), 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        except Exception:
            pass
        # broadcast a brief NPC line
        _npc_add({'text': f"Narrative updated: {zone} reputation {row.get('score',0.0):.2f}.", 'ts': obj['ts']})
        return jsonify({'ok': True, 'reputation': _zone_reputation, 'saved': obj})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/council/state')
def api_council_state():
    try:
        return jsonify({ 'ok': True, 'state': _council_state })
    except Exception as e:
        return jsonify({ 'ok': False, 'error': str(e) }), 500

def _mark_nb_coin_block(interval: str, market: str, reasons: list[str] | None = None, ts_ms: int | None = None, meta: dict | None = None):
    try:
        b = _bucket_ts_interval(ts_ms or int(time.time()*1000), interval)
        coin = _ensure_nb_coin(interval, market, b)
        # Rest-after-first-coin gate annotation
        try:
            iv = str(interval)
            rest_until = int(_nb_rest_until.get(iv) or 0)
            if rest_until and b < rest_until:
                if reasons is None:
                    reasons = []
                if 'rest:scheduled' not in reasons:
                    reasons = list(reasons) + ['rest:scheduled']
        except Exception:
            pass
        coin['checked_ts'] = int(time.time()*1000)
        # Do not override side if already traded; still record reasons for diagnostics
        rs = reasons or []
        if rs:
            # append unique recent reasons (cap 20)
            for r in rs:
                try:
                    r = str(r)
                except Exception:
                    continue
                coin['reasons'].append(r)
                if isinstance(coin.get('blocks'), dict):
                    coin['blocks'][r] = int(coin['blocks'].get(r, 0)) + 1
            if len(coin['reasons']) > 20:
                coin['reasons'] = coin['reasons'][-20:]
        if meta and isinstance(meta, dict):
            # store a tiny snapshot
            coin['meta'] = {k: meta[k] for k in list(meta.keys())[:12]}
    except Exception:
        pass
    try:
        _save_nb_coins()
    except Exception:
        pass

def _record_nb_attempt(interval: str, market: str, side: str, ok: bool, error: str | None = None, ts_ms: int | None = None, meta: dict | None = None):
    try:
        b = _bucket_ts_interval(ts_ms or int(time.time()*1000), interval)
        coin = _ensure_nb_coin(interval, market, b)
        arr = coin.setdefault('attempts', [])
        item = {
            'ts': int(time.time()*1000),
            'side': str(side).upper(),
            'ok': bool(ok),
            'error': (str(error) if error else None),
        }
        if isinstance(meta, dict):
            item['meta'] = {k: meta[k] for k in list(meta.keys())[:12]}
        arr.append(item)
        # aggregate blocks
        key = (f"attempt_ok_{str(side).upper()}" if ok else f"error:{str(error)}:{str(side).upper()}")
        coin.setdefault('blocks', {})
        coin['blocks'][key] = int(coin['blocks'].get(key, 0)) + 1
        if not ok and error:
            coin.setdefault('reasons', [])
            coin['reasons'].append(f"error:{str(error)}:{str(side).upper()}")
            if len(coin['reasons']) > 20:
                coin['reasons'] = coin['reasons'][-20:]
    except Exception:
        pass
    try:
        _save_nb_coins()
    except Exception:
        pass

def _prefill_nb_coins(interval: str, market: str, how_many: int = 50) -> None:
    try:
        now_ms = int(time.time()*1000)
        now_b = _bucket_ts_interval(now_ms, interval)
        sec = _interval_to_sec(interval)
        for i in range(max(1, how_many)):
            b = now_b - i*sec
            _ensure_nb_coin(str(interval), str(market), int(b))
    except Exception:
        pass

# Bot controller for start/stop from UI
bot_ctrl = {
    'running': False,
    'thread': None,
    'last_signal': 'HOLD',
    'last_order': None,
    'nb_zone': 'ORANGE',  # 'BLUE' or 'ORANGE'
    'ml_zone': 'ORANGE',  # 'BLUE' or 'ORANGE'
    'r_value': 0.5,  # Current r value
    'position': 'FLAT',  # 'FLAT' or 'LONG' (single-cycle enforcement)
    'cfg_override': {  # values can be overridden via /api/bot/config
        'paper': None,
        'order_krw': None,
        'pnl_ratio': None,
        'pnl_profit_ratio': None,
        'pnl_loss_ratio': None,
        'ema_fast': None,
        'ema_slow': None,
        'candle': None,
        'market': None,
        'interval_sec': None,
        'require_ml': None,  # if true, require ML confirmation to place orders
        'zone100_only': None,  # if true, place orders only when zone prob is 100%
        'require_group': None,  # if true, require multi-timeframe group consensus
        'group_intervals': None,  # e.g., ["minute1","minute3","minute5"]
        'group_buy_th': None,    # 0~100
        'group_sell_th': None,   # 0~100
        'min_order_gap_sec': None, # enforce minimal seconds between orders
        'require_pullback': None,   # require pullback from extreme before ordering
        'pullback_r': None,         # minimum extreme_gap in r (e.g., 0.02)
        'pullback_bars': None,      # minimum bars since extreme (zone_extreme_age)
        # Enforce side by zone: ONLY BUY in BLUE, ONLY SELL in ORANGE
        'enforce_zone_side': None,
        'nb_force': None,  # if true, place order immediately on NB signal (skip ML/pullback/group/zone100)
        # NB window override from UI to align server signals with chart
        'nb_window': None,
        # runtime key injection (avoid restarting server)
        'access_key': None,
        'secret_key': None,
        'open_api_access_key': None,
        'open_api_secret_key': None,
    }
}

# ---------------- NB auto-tune persistence ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PARAMS_PATH = os.path.join(DATA_DIR, 'nb_params.json')

def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def load_nb_params():
    try:
        _ensure_data_dir()
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return { 'buy': 0.70, 'sell': 0.30, 'window': 50, 'updated_at': None }

def save_nb_params(params: dict):
    try:
        _ensure_data_dir()
        params = dict(params)
        params['updated_at'] = int(time.time()*1000)
        with open(PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False)
        return True
    except Exception:
        return False

# ---------------- ML training/prediction (development) ----------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
ML_MODEL_PATH = os.path.join(MODELS_DIR, 'nb_ml.pkl')

def _model_path_for(interval: str) -> str:
    try:
        safe = str(interval or 'minute10').replace('/', '_')
    except Exception:
        safe = 'minute10'
    return os.path.join(MODELS_DIR, f'nb_ml_{safe}.pkl')

def _ensure_models_dir():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception:
        pass

def _build_features(df: pd.DataFrame, window: int, ema_fast: int = 10, ema_slow: int = 30, horizon: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['close'] = pd.to_numeric(df['close'], errors='coerce')
    out['high'] = pd.to_numeric(df['high'], errors='coerce')
    out['low'] = pd.to_numeric(df['low'], errors='coerce')
    # NB r
    r = _compute_r_from_ohlcv(df, window)
    out['r'] = r
    out['w'] = (out['high'].rolling(window).max() - out['low'].rolling(window).min()) / ((out['high'] + out['low'])/2).replace(0, np.nan)
    # EMA features
    out['ema_f'] = out['close'].ewm(span=ema_fast, adjust=False).mean()
    out['ema_s'] = out['close'].ewm(span=ema_slow, adjust=False).mean()
    out['ema_diff'] = out['ema_f'] - out['ema_s']
    # r smoothed and slopes
    out['r_ema3'] = out['r'].ewm(span=3, adjust=False).mean()
    out['r_ema5'] = out['r'].ewm(span=5, adjust=False).mean()
    out['dr'] = out['r'].diff()
    out['ret1'] = out['close'].pct_change(1)
    out['ret3'] = out['close'].pct_change(3)
    out['ret5'] = out['close'].pct_change(5)
    # Zone-aware helper features so model can learn BLUE/ORANGE context explicitly
    try:
        HIGH = float(os.getenv('NB_HIGH', '0.55'))
        LOW = float(os.getenv('NB_LOW', '0.45'))
    except Exception:
        HIGH, LOW = 0.55, 0.45
    rng = max(1e-9, HIGH - LOW)
    zone_flag = []  # +1=BLUE, -1=ORANGE
    dist_high = []  # max(0, r-HIGH)
    dist_low = []   # max(0, LOW-r)
    extreme_gap = []
    zone_conf = []  # confidence within current zone (0~1)
    # Zone extrema tracking (min/max r and corresponding prices within the current zone)
    zone_min_r_list = []
    zone_max_r_list = []
    zone_min_price_list = []
    zone_max_price_list = []
    zone_extreme_r_list = []     # r of current zone's defining extreme (min for BLUE, max for ORANGE)
    zone_extreme_price_list = [] # price at that extreme
    zone_extreme_age_list = []   # bars since that extreme was set/updated
    cur_zone = None
    cur_zone_min_r = None
    cur_zone_max_r = None
    cur_zone_min_idx = None
    cur_zone_max_idx = None
    cur_extreme_idx = None
    # Zone progression helpers
    zone_start_idx = 0
    zmin_prev = None
    zmax_prev = None
    zmin_slope_list = []
    zmax_slope_list = []
    zone_len_list = []
    zone_pos_list = []  # 0~1, position of current zone segment within the last `window` bars (0=left,1=right)
    # Previous completed zone extrema (for BLUE min and ORANGE max)
    prev_blue_min_completed = None
    prev_orange_max_completed = None
    zmin_vs_prev_list = []
    zmax_vs_prev_list = []
    blue_min_last_list = []
    orange_max_last_list = []
    blue_min_cur_list = []
    orange_max_cur_list = []
    close_vals = out['close'].astype(float).fillna(method='bfill').fillna(method='ffill').fillna(0.0).values.tolist()
    r_vals = r.fillna(0.5).astype(float).values.tolist()
    for i, rv in enumerate(r_vals):
        if cur_zone not in ('BLUE','ORANGE'):
            cur_zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
            zone_start_idx = i
        # update extremes per zone
        if cur_zone == 'BLUE' and rv >= HIGH:
            # BLUE completed → record its min before switching
            try:
                prev_blue_min_completed = float(cur_zone_min_r if cur_zone_min_r is not None else rv)
            except Exception:
                prev_blue_min_completed = float(rv)
            cur_zone = 'ORANGE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
            zone_start_idx = i
        elif cur_zone == 'ORANGE' and rv <= LOW:
            # ORANGE completed → record its max before switching
            try:
                prev_orange_max_completed = float(cur_zone_max_r if cur_zone_max_r is not None else rv)
            except Exception:
                prev_orange_max_completed = float(rv)
            cur_zone = 'BLUE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
            zone_start_idx = i
        # track within-zone min/max r and their indices
        cur_zone_min_r = rv if cur_zone_min_r is None else min(cur_zone_min_r, rv)
        cur_zone_max_r = rv if cur_zone_max_r is None else max(cur_zone_max_r, rv)
        if cur_zone_min_r == rv:
            cur_zone_min_idx = i
        if cur_zone_max_r == rv:
            cur_zone_max_idx = i
        if cur_zone == 'BLUE':
            cur_extreme_idx = cur_zone_min_idx if cur_zone_min_idx is not None else i
            zone_flag.append(1)
            zone_conf.append(max(0.0, (HIGH - rv) / rng))
        else:
            cur_extreme_idx = cur_zone_max_idx if cur_zone_max_idx is not None else i
            zone_flag.append(-1)
            zone_conf.append(max(0.0, (rv - LOW) / rng))
        dist_high.append(max(0.0, rv - HIGH))
        dist_low.append(max(0.0, LOW - rv))
        # current zone's defining extreme r
        cur_extreme_r = (cur_zone_min_r if cur_zone == 'BLUE' else cur_zone_max_r)
        extreme_gap.append(abs(rv - float(cur_extreme_r)))
        # slopes of zone extrema (delta since previous bar)
        try:
            zmin_slope = (0.0 if zmin_prev is None else float(cur_zone_min_r) - float(zmin_prev))
        except Exception:
            zmin_slope = 0.0
        try:
            zmax_slope = (0.0 if zmax_prev is None else float(cur_zone_max_r) - float(zmax_prev))
        except Exception:
            zmax_slope = 0.0
        zmin_prev = float(cur_zone_min_r if cur_zone_min_r is not None else rv)
        zmax_prev = float(cur_zone_max_r if cur_zone_max_r is not None else rv)
        zmin_slope_list.append(zmin_slope)
        zmax_slope_list.append(zmax_slope)
        # bars since current zone started
        try:
            zone_len_list.append(int(i - zone_start_idx))
        except Exception:
            zone_len_list.append(0)
        # zone position within the last `window` bars
        try:
            win_start = max(0, i - window + 1)
            z_start = max(zone_start_idx, win_start)
            z_end = i
            denom = max(1, (i - win_start))
            zone_mid = (z_start + z_end) / 2.0
            zone_pos = (zone_mid - win_start) / denom  # 0=left, 1=right
            if not np.isfinite(zone_pos): zone_pos = 0.5
        except Exception:
            zone_pos = 0.5
        zone_pos_list.append(float(max(0.0, min(1.0, zone_pos))))
        # compare current zone's extreme vs previous completed same-zone extreme
        if cur_zone == 'BLUE':
            try:
                zmin_vs_prev = (float(cur_zone_min_r) - float(prev_blue_min_completed)) if prev_blue_min_completed is not None else 0.0
            except Exception:
                zmin_vs_prev = 0.0
            zmax_vs_prev = 0.0
        else:
            try:
                zmax_vs_prev = (float(cur_zone_max_r) - float(prev_orange_max_completed)) if prev_orange_max_completed is not None else 0.0
            except Exception:
                zmax_vs_prev = 0.0
            zmin_vs_prev = 0.0
        zmin_vs_prev_list.append(zmin_vs_prev)
        zmax_vs_prev_list.append(zmax_vs_prev)
        # emit both BLUE and ORANGE extrema regardless of current zone
        try:
            blue_min_last = float(prev_blue_min_completed) if prev_blue_min_completed is not None else float(zmin_prev)
        except Exception:
            blue_min_last = float(rv)
        try:
            orange_max_last = float(prev_orange_max_completed) if prev_orange_max_completed is not None else float(zmax_prev)
        except Exception:
            orange_max_last = float(rv)
        blue_min_last_list.append(blue_min_last)
        orange_max_last_list.append(orange_max_last)
        # current estimates: current zone's extreme if matching, else last completed for that zone
        try:
            blue_min_cur = float(cur_zone_min_r) if cur_zone == 'BLUE' and cur_zone_min_r is not None else blue_min_last
        except Exception:
            blue_min_cur = blue_min_last
        try:
            orange_max_cur = float(cur_zone_max_r) if cur_zone == 'ORANGE' and cur_zone_max_r is not None else orange_max_last
        except Exception:
            orange_max_cur = orange_max_last
        blue_min_cur_list.append(blue_min_cur)
        orange_max_cur_list.append(orange_max_cur)
        # append zone-wide extrema and their prices
        zone_min_r_list.append(float(cur_zone_min_r if cur_zone_min_r is not None else rv))
        zone_max_r_list.append(float(cur_zone_max_r if cur_zone_max_r is not None else rv))
        zmin_px = float(close_vals[cur_zone_min_idx]) if cur_zone_min_idx is not None else float(close_vals[i])
        zmax_px = float(close_vals[cur_zone_max_idx]) if cur_zone_max_idx is not None else float(close_vals[i])
        zone_min_price_list.append(zmin_px)
        zone_max_price_list.append(zmax_px)
        zone_extreme_r_list.append(float(cur_extreme_r))
        zext_px = float(close_vals[cur_extreme_idx]) if cur_extreme_idx is not None else float(close_vals[i])
        zone_extreme_price_list.append(zext_px)
        zone_extreme_age_list.append(int(i - (cur_extreme_idx if cur_extreme_idx is not None else i)))
    try:
        out['zone_flag'] = pd.Series(zone_flag, index=out.index)
        out['dist_high'] = pd.Series(dist_high, index=out.index)
        out['dist_low'] = pd.Series(dist_low, index=out.index)
        out['extreme_gap'] = pd.Series(extreme_gap, index=out.index)
        out['zone_conf'] = pd.Series(zone_conf, index=out.index)
        # new: zone extrema features (learning + insight)
        out['zone_min_r'] = pd.Series(zone_min_r_list, index=out.index)
        out['zone_max_r'] = pd.Series(zone_max_r_list, index=out.index)
        out['zone_min_price'] = pd.Series(zone_min_price_list, index=out.index)
        out['zone_max_price'] = pd.Series(zone_max_price_list, index=out.index)
        out['zone_extreme_r'] = pd.Series(zone_extreme_r_list, index=out.index)
        out['zone_extreme_price'] = pd.Series(zone_extreme_price_list, index=out.index)
        out['zone_extreme_age'] = pd.Series(zone_extreme_age_list, index=out.index)
        # trend helpers: extrema slopes and prior comparisons
        out['zmin_slope'] = pd.Series(zmin_slope_list, index=out.index)
        out['zmax_slope'] = pd.Series(zmax_slope_list, index=out.index)
        out['zone_len'] = pd.Series(zone_len_list, index=out.index)
        out['zone_pos'] = pd.Series(zone_pos_list, index=out.index)
        out['zmin_vs_prev'] = pd.Series(zmin_vs_prev_list, index=out.index)
        out['zmax_vs_prev'] = pd.Series(zmax_vs_prev_list, index=out.index)
        out['blue_min_last'] = pd.Series(blue_min_last_list, index=out.index)
        out['orange_max_last'] = pd.Series(orange_max_last_list, index=out.index)
        out['blue_min_cur'] = pd.Series(blue_min_cur_list, index=out.index)
        out['orange_max_cur'] = pd.Series(orange_max_cur_list, index=out.index)
    except Exception:
        pass
    # Time-of-day and weekly cycle features (help model learn time-localized BLUE/ORANGE behaviors)
    try:
        idx = out.index
        hours = pd.Index(getattr(idx, 'hour', pd.Series(idx).map(lambda x: getattr(x, 'hour', 0))))
        minutes = pd.Index(getattr(idx, 'minute', pd.Series(idx).map(lambda x: getattr(x, 'minute', 0))))
        tod_min = (hours.astype(int) * 60 + minutes.astype(int)).astype(float)
        out['tod_sin'] = np.sin(2 * np.pi * tod_min / (24*60))
        out['tod_cos'] = np.cos(2 * np.pi * tod_min / (24*60))
        # Day-of-week cyclic
        dows = pd.Index(getattr(idx, 'dayofweek', pd.Series(idx).map(lambda x: getattr(x, 'dayofweek', 0)))).astype(float)
        out['dow_sin'] = np.sin(2 * np.pi * dows / 7.0)
        out['dow_cos'] = np.cos(2 * np.pi * dows / 7.0)
        # Rough global sessions in KST: ASIA 09-17, EU 16-24, US 22-06
        h = hours.astype(int)
        out['sess_asia'] = ((h>=9) & (h<17)).astype(int)
        out['sess_eu'] = ((h>=16) | (h<0)).astype(int)  # 16~23
        out['sess_us'] = ((h>=22) | (h<6)).astype(int)
    except Exception:
        pass
    # forward return for labeling
    out['fwd'] = out['close'].shift(-horizon) / out['close'] - 1.0
    return out

def _train_ml(X: pd.DataFrame, y: np.ndarray):
    # Try scikit-learn; fall back to logistic regression if needed
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.utils.class_weight import compute_class_weight
        cls = GradientBoostingClassifier(random_state=42)
        # simple fit; for dev we skip CV heavy compute
        cls.fit(X, y)
        return cls
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Please run: pip install scikit-learn. Cause: %s" % e)

def _load_ml(interval: str | None = None):
    _ensure_models_dir()
    try:
        path = _model_path_for(interval or state.get('candle') or load_config().candle)
    except Exception:
        path = ML_MODEL_PATH
    if os.path.exists(path):
        return joblib.load(path)
    # Backward compatibility fallback
    if os.path.exists(ML_MODEL_PATH):
        return joblib.load(ML_MODEL_PATH)
    return None

def _make_insight(df: pd.DataFrame, window: int, ema_fast: int, ema_slow: int, interval: str, pack: dict | None = None) -> dict:
    try:
        feat = _build_features(df, window, ema_fast, ema_slow, 5).dropna().copy()
        if feat.empty:
            return {}
        last = feat.iloc[-1]
        zone_flag = int(round(float(last.get('zone_flag', 0))))
        zone = 'BLUE' if zone_flag == 1 else ('ORANGE' if zone_flag == -1 else 'UNKNOWN')
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH, LOW = 0.55, 0.45
        rng = max(1e-9, HIGH - LOW)
        rv = float(last.get('r', 0.5))
        p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
        p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
        s0 = p_blue_raw + p_orange_raw
        if s0 > 0:
            p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
        # Trend weighting
        try:
            trend_k = int(os.getenv('NB_TREND_K', '30'))
            trend_alpha = float(os.getenv('NB_TREND_ALPHA', '0.5'))
        except Exception:
            trend_k, trend_alpha = 30, 0.5
        p_blue, p_orange = p_blue_raw, p_orange_raw
        try:
            r_series = _compute_r_from_ohlcv(df, window).astype(float)
            if len(r_series) >= trend_k*2:
                tail_now = r_series.iloc[-trend_k:]
                tail_prev = r_series.iloc[-trend_k*2:-trend_k]
                zmax_now, zmax_prev = float(tail_now.max()), float(tail_prev.max())
                zmin_now, zmin_prev = float(tail_now.min()), float(tail_prev.min())
                trend_orange = max(0.0, (zmax_prev - zmax_now) / rng)
                trend_blue = max(0.0, (zmin_now - zmin_prev) / rng)
                p_orange = max(0.0, min(1.0, p_orange_raw * (1.0 - trend_alpha * trend_orange)))
                p_blue = max(0.0, min(1.0, p_blue_raw * (1.0 - trend_alpha * trend_blue)))
                s = p_blue + p_orange
                if s > 0:
                    p_blue, p_orange = p_blue/s, p_orange/s
        except Exception:
            pass
        ins = {
            'r': rv,
            'zone_flag': zone_flag,
            'zone': zone,
            'zone_conf': float(last.get('zone_conf', 0.0)),
            'dist_high': float(last.get('dist_high', 0.0)),
            'dist_low': float(last.get('dist_low', 0.0)),
            'extreme_gap': float(last.get('extreme_gap', 0.0)),
            'zone_min_r': float(last.get('zone_min_r', rv)),
            'zone_max_r': float(last.get('zone_max_r', rv)),
            'zone_extreme_r': float(last.get('zone_extreme_r', rv)),
            'zone_extreme_age': int(last.get('zone_extreme_age', 0)),
            'zone_min_price': float(last.get('zone_min_price', last.get('close', 0.0))),
            'zone_max_price': float(last.get('zone_max_price', last.get('close', 0.0))),
            'zone_extreme_price': float(last.get('zone_extreme_price', last.get('close', 0.0))),
            'w': float(last.get('w', 0.0)),
            'ema_diff': float(last.get('ema_diff', 0.0)),
            'pct_blue_raw': float(p_blue_raw*100.0),
            'pct_orange_raw': float(p_orange_raw*100.0),
            'pct_blue': float(p_blue*100.0),
            'pct_orange': float(p_orange*100.0),
        }
        # record observation bucket for grouping
        try:
            _record_group_observation(interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
        except Exception:
            pass
        return ins
    except Exception:
        return {}

def _simulate_pnl_from_preds(prices: pd.Series, preds: np.ndarray, fee_bps: float = 10.0) -> dict:
    pos = 0
    entry = 0.0
    pnl = 0.0
    wins = 0
    trades = 0
    for p, y in zip(prices.astype(float).values, preds.tolist()):
        if pos == 0 and y > 0:
            pos = 1
            entry = float(p)
            trades += 1
        elif pos == 1 and y < 0:
            ret = float(p) - entry
            ret -= abs(entry) * (fee_bps / 10000.0)
            ret -= abs(p) * (fee_bps / 10000.0)
            pnl += ret
            if ret > 0:
                wins += 1
            pos = 0
            entry = 0.0
    if pos == 1:
        p = float(prices.iloc[-1])
        ret = p - entry
        ret -= abs(entry) * (fee_bps / 10000.0)
        ret -= abs(p) * (fee_bps / 10000.0)
        pnl += ret
        if ret > 0:
            wins += 1
        pos = 0
    win_rate = (wins / trades * 100.0) if trades else 0.0
    return { 'pnl': float(pnl), 'trades': int(trades), 'wins': int(wins), 'win_rate': float(win_rate) }

@app.route('/api/ml/train', methods=['GET','POST'])
def api_ml_train():
    try:
        try:
            if request.method == 'POST':
                payload = request.get_json(force=True) if request.is_json else (request.form.to_dict() if request.form else {})
            else:
                payload = request.args.to_dict()
        except Exception:
            payload = {}
        window = int(payload.get('window', load_nb_params().get('window', 50)))
        ema_fast = int(payload.get('ema_fast', 10))
        ema_slow = int(payload.get('ema_slow', 30))
        horizon = int(payload.get('horizon', 5))
        tau = float(payload.get('tau', 0.002))  # 0.2%
        count = int(payload.get('count', 1800))
        interval = payload.get('interval') or load_config().candle
        # Default label mode can be overridden via env NB_LABEL_MODE_DEFAULT
        try:
            _lm_def = os.getenv('NB_LABEL_MODE_DEFAULT', 'zone')
        except Exception:
            _lm_def = 'zone'
        label_mode = str(payload.get('label_mode', _lm_def))  # 'zone' | 'nb_zone' | 'fwd_return' | 'nb_extreme' | 'nb_best_trade'
        # Optional: extreme-based labels tuning
        try:
            pullback_pct = float(payload.get('pullback_pct', os.getenv('NB_PULLBACK_PCT', '40')))
        except Exception:
            pullback_pct = 40.0
        try:
            confirm_bars = int(payload.get('confirm_bars', os.getenv('NB_CONFIRM_BARS', '2')))
        except Exception:
            confirm_bars = 2

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        # Prefill NB COINs for the training interval so UI has coins during random learning
        try:
            _prefill_nb_coins(str(interval), str(cfg.market), how_many=min(200, max(60, count)))
        except Exception:
            pass
        feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        # label: depends on label_mode
        if label_mode == 'fwd_return':
            fwd = feat['fwd']
            y = np.where(fwd >= tau, 1, np.where(fwd <= -tau, -1, 0))
        elif label_mode in ('zone','zone_flag'):
            # Learn zone as target: BLUE(+1), ORANGE(-1) using hysteresis to reduce churn
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            labels = np.zeros(len(df), dtype=int)
            zone = None
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                # hysteresis updates
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                labels[i] = (1 if zone=='BLUE' else -1)
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
            # Safety: ensure no zeros remain in zone targets
            if np.any(y == 0):
                try:
                    rv_feat = feat['r'].astype(float).values
                    y = np.where(y == 0, np.where(rv_feat >= 0.5, -1, 1), y)
                except Exception:
                    y = np.where(y == 0, 1, y)
        elif label_mode == 'mayor_guidance':
            # 촌장 지침 학습: Zone-Side Only (BUY@BLUE / SELL@ORANGE)
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            labels = np.zeros(len(df), dtype=int)
            zone = None
            r_vals = r.values.tolist()
            
            # 촌장 지침 기반 라벨링
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                # hysteresis updates
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                
                # 촌장 지침에 따른 라벨링:
                # BLUE 구역: BUY(+1)만 허용, SELL(-1) 금지
                # ORANGE 구역: SELL(-1)만 허용, BUY(+1) 금지
                if zone == 'BLUE':
                    labels[i] = 1  # BUY만 허용
                elif zone == 'ORANGE':
                    labels[i] = -1  # SELL만 허용
                else:
                    labels[i] = 0  # HOLD
            
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        elif label_mode == 'nb_extreme':
            # Learn BLUE/ORANGE extremes with pullback confirmation; one BUY then one SELL
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            RANGE = max(1e-9, HIGH - LOW)
            pull_r = RANGE * (max(0.0, min(100.0, float(pullback_pct))) / 100.0)
            labels = np.zeros(len(df), dtype=int)
            zone = None
            zone_extreme = None
            prev_r = None
            confirm_up = 0
            confirm_dn = 0
            position = 'FLAT'
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                # init zone
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                # zone transitions reset extremes
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                # track extremes
                if zone == 'BLUE':
                    zone_extreme = min(zone_extreme, rv) if zone_extreme is not None else rv
                else:
                    zone_extreme = max(zone_extreme, rv) if zone_extreme is not None else rv
                # confirmations
                if prev_r is not None:
                    if rv > prev_r: confirm_up += 1
                    else: confirm_up = 0
                    if rv < prev_r: confirm_dn += 1
                    else: confirm_dn = 0
                prev_r = rv
                # decisions
                if position == 'FLAT' and zone == 'BLUE':
                    if (rv - zone_extreme) >= pull_r and confirm_up >= int(confirm_bars):
                        labels[i] = 1
                        position = 'LONG'
                        confirm_up = 0; confirm_dn = 0
                elif position == 'LONG' and zone == 'ORANGE':
                    if (zone_extreme - rv) >= pull_r and confirm_dn >= int(confirm_bars):
                        labels[i] = -1
                        position = 'FLAT'
                        confirm_up = 0; confirm_dn = 0
            # align labels to feature index
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        elif label_mode == 'nb_best_trade':
            # Build NB zone transitions, form BUY/SELL pairs, pick the single best PnL pair
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            zone = None
            signals = []  # (idx, side)
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    signals.append((i, -1))  # SELL
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    signals.append((i, 1))   # BUY
            # normalize to alternating BUY/SELL starting with BUY
            norm = []
            last = None
            for i, s in signals:
                if s == last:
                    continue
                norm.append((i, s))
                last = s
            while norm and norm[0][1] != 1:
                norm.pop(0)
            # pair and score
            prices = df['close'].astype(float).values.tolist()
            best = None
            for k in range(0, len(norm)-1, 2):
                bi, bs = norm[k]
                if k+1 >= len(norm):
                    break
                si, ss = norm[k+1]
                if bs != 1 or ss != -1:
                    continue
                if si <= bi or bi < 0 or si >= len(prices):
                    continue
                ret = float(prices[si]) - float(prices[bi])
                # approx fees: 0.1% in/out
                fee_bps = 10.0
                ret -= float(prices[bi]) * (fee_bps/10000.0)
                ret -= float(prices[si]) * (fee_bps/10000.0)
                if (best is None) or (ret > best['pnl']):
                    best = { 'buy_idx': bi, 'sell_idx': si, 'pnl': ret }
            labels = np.zeros(len(df), dtype=int)
            if best is not None:
                labels[best['buy_idx']] = 1
                labels[best['sell_idx']] = -1
            # align labels to feature index
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        else:
            # NB zone transition labels consistent with live trading loop
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            labels = np.zeros(len(df), dtype=int)
            zone = None
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE', 'ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                sig = 0
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    sig = -1  # SELL
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    sig = 1   # BUY
                labels[i] = sig
            # align labels to feature frame
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        base_cols = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
        ext_cols = ['zone_flag','dist_high','dist_low','extreme_gap','zone_conf','zone_min_r','zone_max_r','zone_extreme_r','zone_extreme_age','zmin_slope','zmax_slope','zone_len','zmin_vs_prev','zmax_vs_prev']
        use_cols = base_cols + [c for c in ext_cols if c in feat.columns]
        X = feat[use_cols]
        # Sample weights: class-balance + zone-time/extreme-aware weighting
        total_n = len(X)
        c_neg = int((y==-1).sum()); c_zero = int((y==0).sum()); c_pos = int((y==1).sum())
        w_neg = float(total_n) / max(1, 3*c_neg)
        w_zero = float(total_n) / max(1, 3*c_zero) if c_zero>0 else float(total_n)
        w_pos = float(total_n) / max(1, 3*c_pos)
        w = np.where(y==-1, w_neg, np.where(y==0, w_zero, w_pos)).astype(float)
        # Context multiplier:
        # - SELL(-1): emphasize when zones are far apart (long zone_len) and ORANGE max exceeds previous (zmax_vs_prev > 0)
        # - BUY(+1): emphasize when zones are close (short zone_len) and BLUE min exceeds previous (zmin_vs_prev > 0)
        try:
            zone_len = feat['zone_len'].reindex(X.index) if hasattr(X, 'index') else feat['zone_len']
            zmin_vs_prev = feat['zmin_vs_prev'].reindex(X.index) if hasattr(X, 'index') else feat['zmin_vs_prev']
            zmax_vs_prev = feat['zmax_vs_prev'].reindex(X.index) if hasattr(X, 'index') else feat['zmax_vs_prev']
            # normalize zone_len by window
            zl = np.clip((zone_len.astype(float).values / max(1, window)), 0.0, 1.0)
            zp = feat['zone_pos'].reindex(X.index).astype(float).values if 'zone_pos' in feat.columns else np.zeros_like(zl)
            zvp_min = np.clip(np.maximum(0.0, zmin_vs_prev.astype(float).values), 0.0, 1.0)
            zvp_max = np.clip(np.maximum(0.0, zmax_vs_prev.astype(float).values), 0.0, 1.0)
            try:
                alpha_buy = float(os.getenv('TW_ALPHA_BUY', '0.5'))
            except Exception:
                alpha_buy = 0.5
            try:
                alpha_sell = float(os.getenv('TW_ALPHA_SELL', '0.5'))
            except Exception:
                alpha_sell = 0.5
            ctx = np.ones_like(w, dtype=float)
            # SELL: farther zones (zl high) + positioned to the right (zp high) + stronger ORANGE max (zvp_max high)
            ctx = np.where(y==-1, ctx * (1.0 + alpha_sell * (zvp_max * zl * (0.5 + 0.5*zp))), ctx)
            # BUY: closer zones (zl low) + positioned to the left (zp low) + stronger BLUE min (zvp_min high)
            ctx = np.where(y== 1, ctx * (1.0 + alpha_buy  * (zvp_min * (1.0 - zl) * (1.0 - 0.5*zp))), ctx)
            w = w * ctx
        except Exception:
            pass

        # Hyperparameter search with time-series CV (weighted)
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        Xv = X.values
        tscv = TimeSeriesSplit(n_splits=3)
        grid = [
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 2},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 2},
            {'n_estimators': 150, 'learning_rate': 0.10, 'max_depth': 3},
        ]
        best_params = None
        best_score = -1e9
        best_pnl = -1e18
        # prices aligned to feature index
        prices = feat['close'].loc[X.index] if hasattr(X, 'index') else feat['close']
        for params in grid:
            accs=[]; f1s=[]; cms=None; pnl_sum=0.0
            for tr_idx, va_idx in tscv.split(Xv):
                cls = GradientBoostingClassifier(random_state=42, **params)
                cls.fit(Xv[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
                yp = cls.predict(Xv[va_idx])
                accs.append(accuracy_score(y[va_idx], yp))
                f1s.append(f1_score(y[va_idx], yp, average='macro', zero_division=0))
                cm = confusion_matrix(y[va_idx], yp, labels=[-1,0,1])
                cms = (cm if cms is None else (cms + cm))
                # pnl on validation slice
                try:
                    prices_va = prices.iloc[va_idx]
                    st = _simulate_pnl_from_preds(prices_va, yp)
                    pnl_sum += st['pnl']
                except Exception:
                    pass
            avg_f1 = float(np.mean(f1s)) if f1s else 0.0
            score = avg_f1
            if (score > best_score + 1e-9) or (abs(score - best_score) <= 1e-9 and pnl_sum > best_pnl):
                best_score = score
                best_params = params
                best_pnl = pnl_sum
        # Fit best model on all data with weights
        base = GradientBoostingClassifier(random_state=42, **(best_params or {}))
        base.fit(Xv, y, sample_weight=w)
        _ensure_models_dir()
        # compute reports
        yhat_in = base.predict(Xv)
        report_in = classification_report(y, yhat_in, output_dict=True, zero_division=0)
        cm_in = confusion_matrix(y, yhat_in, labels=[-1,0,1]).tolist()
        # summarize CV again for metrics payload
        metrics = {
            'in_sample': { 'report': report_in, 'confusion': cm_in },
            'cv': { 'f1_macro': float(best_score), 'pnl_sum': float(best_pnl) },
            'params': best_params,
        }
        # persist the exact feature order used for training
        try:
            feature_names = list(X.columns)
        except Exception:
            feature_names = use_cols
        pack = { 'model': base, 'window': window, 'ema_fast': ema_fast, 'ema_slow': ema_slow, 'horizon': horizon, 'tau': tau, 'interval': interval, 'metrics': metrics, 'trained_at': int(time.time()*1000), 'feature_names': feature_names, 'label_mode': label_mode }
        
        # Optional slope regressor: predict steepness over horizon (per-bar pct return)
        try:
            closes = feat['close'].astype(float).reindex(X.index)
            fwd_close = closes.shift(-horizon)
            slope_y = ((fwd_close - closes) / (closes.replace(0, np.nan) * max(1, horizon))).fillna(0.0).values
            reg = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=2)
            reg.fit(X.values, slope_y)
            pack['slope_model'] = reg
        except Exception:
            pass
        # save model per-interval
        try:
            joblib.dump(pack, _model_path_for(interval))
        except Exception:
            joblib.dump(pack, ML_MODEL_PATH)
        ml_state['train_count'] = int(ml_state.get('train_count', 0)) + 1
        classes = { '-1': int((y==-1).sum()), '0': int((y==0).sum()), '1': int((y==1).sum()) }
        return jsonify({'ok': True, 'classes': classes, 'report': report_in, 'cv': metrics['cv'], 'params': best_params, 'train_count': ml_state['train_count']})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/ml/predict', methods=['GET'])
def api_ml_predict():
    try:
        # load model for requested or current interval
        try:
            req_iv = request.args.get('interval') if request.args else None
        except Exception:
            req_iv = None
        cur_interval = str(req_iv or (state.get('candle') or load_config().candle))
        pack = _load_ml(cur_interval)
        if not pack:
            # Graceful fallback: return lightweight insight so UI narrative can render
            cfg = load_config()
            try:
                window = int(load_nb_params().get('window', 50))
            except Exception:
                window = 50
            try:
                df = get_candles(cfg.market, cur_interval, count=max(400, window*3))
            except Exception:
                df = pd.DataFrame()
            # Build minimal insight
            ins = {}
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            try:
                r_series = _compute_r_from_ohlcv(df, window)
                rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
            except Exception:
                rv = 0.5
            p_blue = max(0.0, min(1.0, (HIGH - rv) / rng))
            p_orange = max(0.0, min(1.0, (rv - LOW) / rng))
            s = p_blue + p_orange
            if s > 0:
                p_blue, p_orange = p_blue/s, p_orange/s
            zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            ins = {
                'r': rv,
                'zone_flag': (-1 if zone=='ORANGE' else 1),
                'zone': zone,
                'zone_conf': float(max(0.0, (rv-LOW)/rng) if zone=='ORANGE' else max(0.0, (HIGH-rv)/rng)),
                'dist_high': float(max(0.0, rv - HIGH)),
                'dist_low': float(max(0.0, LOW - rv)),
                'extreme_gap': 0.0,
                'w': 0.0,
                'ema_diff': 0.0,
                'pct_blue': float(p_blue*100.0),
                'pct_orange': float(p_orange*100.0),
            }
            try:
                _record_group_observation(cur_interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
            except Exception:
                pass
            label_mode = 'zone'
            action = ('BLUE' if zone=='BLUE' else 'ORANGE')
            return jsonify({'ok': True, 'action': action, 'pred': 0, 'probs': [], 'train_count': int(ml_state.get('train_count', 0)), 'insight': ins, 'zone_actions': {'sell_in_orange': False, 'buy_in_blue': False}, 'label_mode': label_mode, 'steep': None, 'pred_nb': None, 'horizon': 5, 'interval': cur_interval})
        model = pack['model']
        window = int(pack.get('window', 50))
        ema_fast = int(pack.get('ema_fast', 10))
        ema_slow = int(pack.get('ema_slow', 30))
        horizon = int(pack.get('horizon', 5))
        cfg = load_config()
        df = get_candles(cfg.market, cur_interval, count=max(400, window*3))
        try:
            feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        except Exception:
            feat = pd.DataFrame()
        # IMPORTANT: Use same feature set and order as model was trained on
        base_cols = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
        ext_cols = ['zone_flag','dist_high','dist_low','extreme_gap','zone_conf','zone_min_r','zone_max_r','zone_extreme_r','zone_extreme_age']
        trained_cols = list(pack.get('feature_names') or [])
        if not trained_cols:
            # fallback: constrain to model.n_features_in_ if present
            cand = base_cols + [c for c in ext_cols if c in feat.columns]
            try:
                need = int(getattr(model, 'n_features_in_', len(cand)))
            except Exception:
                need = len(cand)
            trained_cols = cand[:need]
        X = feat[[c for c in trained_cols if c in feat.columns]]
        probs = None
        try:
            probs = model.predict_proba(X.values)[-1].tolist()
        except Exception:
            probs = []
        pred = int(model.predict(X.values)[-1])
        # Optional: slope prediction
        slope_hat = None
        try:
            reg = pack.get('slope_model')
            if reg is not None:
                slope_hat = float(reg.predict(X.values)[-1])
        except Exception:
            slope_hat = None
        # Fallback slope if model missing: use recent log-price linear slope per bar
        if slope_hat is None:
            try:
                n_tail = max(20, min(120, window))
                closes_tail = df['close'].astype(float).tail(n_tail)
                if len(closes_tail) >= 5:
                    import numpy as _np
                    y = _np.log(closes_tail.replace(0, _np.nan)).fillna(method='bfill').fillna(method='ffill').values
                    x = _np.arange(len(y), dtype=float)
                    b1 = _np.polyfit(x, y, 1)[0]  # slope of log(price) per bar
                    # approximate per-bar fractional return slope
                    slope_hat = float(b1)
            except Exception:
                slope_hat = None
        # Build insight payload from last feature row
        ins = {}
        try:
            last = feat.iloc[-1]
            zone_flag = int(round(float(last.get('zone_flag', 0))))
            zone = 'BLUE' if zone_flag == 1 else ('ORANGE' if zone_flag == -1 else 'UNKNOWN')
            # heuristic zone probabilities from r distances to thresholds
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            rv = float(last.get('r', 0.5))
            p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
            p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
            s0 = p_blue_raw + p_orange_raw
            if s0 > 0:
                p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
            # Trend-weighted adjustment using recent r trajectory
            try:
                trend_k = int(os.getenv('NB_TREND_K', '30'))
                trend_alpha = float(os.getenv('NB_TREND_ALPHA', '0.5'))
            except Exception:
                trend_k, trend_alpha = 30, 0.5
            try:
                r_series = _compute_r_from_ohlcv(df, window).astype(float)
                if len(r_series) >= trend_k*2:
                    tail_now = r_series.iloc[-trend_k:]
                    tail_prev = r_series.iloc[-trend_k*2:-trend_k]
                    zmax_now, zmax_prev = float(tail_now.max()), float(tail_prev.max())
                    zmin_now, zmin_prev = float(tail_now.min()), float(tail_prev.min())
                    # ORANGE weakening when recent peak < previous peak
                    trend_orange = max(0.0, (zmax_prev - zmax_now) / rng)
                    # BLUE weakening when recent trough > previous trough
                    trend_blue = max(0.0, (zmin_now - zmin_prev) / rng)
                    p_orange = max(0.0, min(1.0, p_orange_raw * (1.0 - trend_alpha * trend_orange)))
                    p_blue = max(0.0, min(1.0, p_blue_raw * (1.0 - trend_alpha * trend_blue)))
                    # If both collapse to zero, fall back to raw
                    s = p_blue + p_orange
                    if s <= 1e-9:
                        p_blue, p_orange = p_blue_raw, p_orange_raw
                        s = p_blue + p_orange
                    if s > 0:
                        p_blue, p_orange = p_blue/s, p_orange/s
                else:
                    p_blue, p_orange = p_blue_raw, p_orange_raw
            except Exception:
                p_blue, p_orange = p_blue_raw, p_orange_raw
            ins = {
                'r': rv,
                'zone_flag': zone_flag,
                'zone': zone,
                'zone_conf': float(last.get('zone_conf', 0.0)),
                'dist_high': float(last.get('dist_high', 0.0)),
                'dist_low': float(last.get('dist_low', 0.0)),
                'extreme_gap': float(last.get('extreme_gap', 0.0)),
                # expose zone extrema for UI insight
                'zone_min_r': float(last.get('zone_min_r', rv)),
                'zone_max_r': float(last.get('zone_max_r', rv)),
                'zone_extreme_r': float(last.get('zone_extreme_r', rv)),
                'zone_extreme_age': int(last.get('zone_extreme_age', 0)),
                # also expose corresponding prices
                'zone_min_price': float(last.get('zone_min_price', last.get('close', 0.0))),
                'zone_max_price': float(last.get('zone_max_price', last.get('close', 0.0))),
                'zone_extreme_price': float(last.get('zone_extreme_price', last.get('close', 0.0))),
                # cross-zone extrema snapshots
                'blue_min_last': float(last.get('blue_min_last', rv)),
                'orange_max_last': float(last.get('orange_max_last', rv)),
                'blue_min_cur': float(last.get('blue_min_cur', rv)),
                'orange_max_cur': float(last.get('orange_max_cur', rv)),
                'w': float(last.get('w', 0.0)),
                'ema_diff': float(last.get('ema_diff', 0.0)),
                'pct_blue_raw': float(p_blue_raw*100.0),
                'pct_orange_raw': float(p_orange_raw*100.0),
                'pct_blue': float(p_blue*100.0),
                'pct_orange': float(p_orange*100.0),
            }
            try:
                _record_group_observation(cur_interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
            except Exception:
                pass
        except Exception:
            ins = {}
        # Fallback insight if feature frame is empty
        if not ins:
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            r_series = _compute_r_from_ohlcv(df, window)
            rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
            p_blue = max(0.0, min(1.0, (HIGH - rv) / rng))
            p_orange = max(0.0, min(1.0, (rv - LOW) / rng))
            s = p_blue + p_orange
            if s > 0:
                p_blue, p_orange = p_blue/s, p_orange/s
            zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            ins = {
                'r': rv,
                'zone_flag': (-1 if zone=='ORANGE' else 1),
                'zone': zone,
                'zone_conf': float(max(0.0, (rv-LOW)/rng) if zone=='ORANGE' else max(0.0, (HIGH-rv)/rng)),
                'dist_high': float(max(0.0, rv - HIGH)),
                'dist_low': float(max(0.0, LOW - rv)),
                'extreme_gap': 0.0,
                'w': float(((df['high'].rolling(window).max() - df['low'].rolling(window).min()) / ((df['high'] + df['low'])/2).replace(0, np.nan)).iloc[-1]) if len(df) else 0.0,
                'ema_diff': float((df['close'].ewm(span=ema_fast, adjust=False).mean().iloc[-1] - df['close'].ewm(span=ema_slow, adjust=False).mean().iloc[-1])) if len(df) else 0.0,
                'pct_blue': float(p_blue*100.0),
                'pct_orange': float(p_orange*100.0),
            }
            try:
                _record_group_observation(cur_interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
            except Exception:
                pass
        # Map prediction to action; if model was trained on 'zone', action is the zone itself
        label_mode = str(pack.get('label_mode') or 'zone')
        action = 'HOLD'
        if label_mode in ('zone','zone_flag'):
            action = ('BLUE' if pred>0 else 'ORANGE')
        elif label_mode == 'mayor_guidance':
            # 촌장 지침 기반 액션 매핑
            if pred > 0:
                action = 'BUY'  # BLUE 구역에서 BUY
            elif pred < 0:
                action = 'SELL'  # ORANGE 구역에서 SELL
            else:
                action = 'HOLD'
        elif pred > 0:
            action = 'BUY'
        elif pred < 0:
            action = 'SELL'
        # Zone-aware intent: whether model would act in the current zone context
        try:
            z_now = str(ins.get('zone') or '').upper()
        except Exception:
            z_now = 'UNKNOWN'
        zone_actions = {
            'sell_in_orange': bool(z_now == 'ORANGE' and pred < 0),
            'buy_in_blue': bool(z_now == 'BLUE' and pred > 0),
        }
        # Zone-conditional steepness
        try:
            steep = None
            if slope_hat is not None:
                if str(ins.get('zone') or '').upper() == 'BLUE':
                    steep = {'blue_up_slope': slope_hat, 'orange_down_slope': None}
                elif str(ins.get('zone') or '').upper() == 'ORANGE':
                    steep = {'blue_up_slope': None, 'orange_down_slope': slope_hat}
            # Predict NB flip timing using a simple r-step projection
            pred_nb = None
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
                rv = float(ins.get('r', 0.5))
                z = str(ins.get('zone') or '').upper()
                # seconds per bar from interval
                def sec_from_iv(iv:str)->int:
                    if iv.startswith('minute'):
                        m=int(iv.replace('minute','') or '1'); return m*60
                    if iv=='day': return 86400
                    return 60
                bar_sec = sec_from_iv(cur_interval)
                # map slope -> r step per bar
                k_env = float(os.getenv('NB_R_STEP_K','0.2'))
                min_step = float(os.getenv('NB_R_STEP_MIN','0.003'))
                r_step = max(min_step, min(0.2, abs(float(slope_hat or 0.0)) * k_env)) if slope_hat is not None else 0.0
                last_ts_ms = int(df.index[-1].timestamp()*1000) if len(df) else int(time.time()*1000)
                if z=='BLUE':
                    dist = max(0.0, HIGH - rv)
                    # need positive slope to approach HIGH
                    if (slope_hat or 0.0) > 0 and r_step>0:
                        bars = int(math.ceil(dist / r_step))
                        if bars>0 and bars <= max(1, horizon*2):
                            pred_nb = {'side':'SELL','bars':bars,'ts': last_ts_ms + bars*bar_sec*1000}
                elif z=='ORANGE':
                    dist = max(0.0, rv - LOW)
                    # need negative slope to approach LOW
                    if (slope_hat or 0.0) < 0 and r_step>0:
                        bars = int(math.ceil(dist / r_step))
                        if bars>0 and bars <= max(1, horizon*2):
                            pred_nb = {'side':'BUY','bars':bars,'ts': last_ts_ms + bars*bar_sec*1000}
            except Exception:
                pred_nb = None
            # derive a simple confidence and default score0
            try:
                pct_major = max(float(ins.get('pct_blue') or ins.get('pct_blue_raw') or 0.0), float(ins.get('pct_orange') or ins.get('pct_orange_raw') or 0.0))
            except Exception:
                pct_major = 0.0
            score0 = float(max(0.0, min(1.0, pct_major/100.0)))
            return jsonify({'ok': True, 'action': action, 'pred': pred, 'probs': probs, 'train_count': ml_state.get('train_count', 0), 'insight': ins, 'zone_actions': zone_actions, 'label_mode': label_mode, 'steep': steep, 'pred_nb': pred_nb, 'horizon': horizon, 'interval': cur_interval, 'score0': score0})
        except Exception:
            return jsonify({'ok': True, 'action': action, 'pred': pred, 'probs': probs, 'train_count': ml_state.get('train_count', 0), 'insight': ins, 'zone_actions': zone_actions, 'label_mode': label_mode, 'pred_nb': None, 'horizon': horizon, 'interval': cur_interval, 'score0': 0.0})
    except Exception as e:
        # Robust fallback: never 500; return minimal insight so UI can render
        try:
            cur_interval = state.get('candle') or load_config().candle
            cfg = load_config()
            window = int(load_nb_params().get('window', 50))
            df = get_candles(cfg.market, cur_interval, count=max(200, window*2))
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            try:
                r_series = _compute_r_from_ohlcv(df, window)
                rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
            except Exception:
                rv = 0.5
            p_blue = max(0.0, min(1.0, (HIGH - rv) / rng)); p_orange = max(0.0, min(1.0, (rv - LOW) / rng))
            s = p_blue + p_orange
            if s > 0: p_blue, p_orange = p_blue/s, p_orange/s
            zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            ins = {'r': rv, 'zone_flag': (-1 if zone=='ORANGE' else 1), 'zone': zone, 'pct_blue': float(p_blue*100.0), 'pct_orange': float(p_orange*100.0)}
            return jsonify({'ok': True, 'action': zone, 'pred': 0, 'probs': [], 'train_count': int(ml_state.get('train_count', 0)), 'insight': ins, 'zone_actions': {'sell_in_orange': False, 'buy_in_blue': False}, 'label_mode': 'zone', 'steep': None, 'pred_nb': None, 'horizon': 5, 'interval': cur_interval, 'score0': float(max(p_blue, p_orange))})
        except Exception as e2:
            return jsonify({'ok': False, 'error': f'predict_fallback_failed: {e2}'}), 500

@app.route('/api/ml/metrics', methods=['GET'])
def api_ml_metrics():
    try:
        try:
            req_iv = request.args.get('interval') if request.args else None
        except Exception:
            req_iv = None
        cur_interval = str(req_iv or (state.get('candle') or load_config().candle))
        pack = _load_ml(cur_interval)
        if not pack:
            return jsonify({'ok': False, 'error': 'model_not_trained'}), 400
        metrics = pack.get('metrics', {}) or {}
        # If metrics missing (old model), recompute lightweight metrics on recent data
        if not metrics or not metrics.get('in_sample'):
            try:
                model = pack['model']
                window = int(pack.get('window', 50))
                ema_fast = int(pack.get('ema_fast', 10))
                ema_slow = int(pack.get('ema_slow', 30))
                horizon = int(pack.get('horizon', 5))
                cfg = load_config()
                df = get_candles(cfg.market, cur_interval, count=max(800, window*3))
                feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
                X = feat[['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']]
                # default NB zone labels for comparison
                r = _compute_r_from_ohlcv(df, window)
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
                labels = np.zeros(len(df), dtype=int)
                zone = None
                r_vals = r.values.tolist()
                for i in range(len(df)):
                    rv = r_vals[i] if i < len(r_vals) else 0.5
                    if zone not in ('BLUE','ORANGE'):
                        zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                    sig = 0
                    if zone == 'BLUE' and rv >= HIGH:
                        zone = 'ORANGE'; sig = -1
                    elif zone == 'ORANGE' and rv <= LOW:
                        zone = 'BLUE'; sig = 1
                    labels[i] = sig
                idx_map = { ts: i for i, ts in enumerate(df.index) }
                y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
                from sklearn.metrics import classification_report, confusion_matrix, f1_score
                from sklearn.model_selection import TimeSeriesSplit
                yhat = model.predict(X.values)
                rep = classification_report(y, yhat, output_dict=True, zero_division=0)
                cm = confusion_matrix(y, yhat, labels=[-1,0,1]).tolist()
                # quick CV
                tscv = TimeSeriesSplit(n_splits=3)
                f1s=[]; pnl_sum=0.0
                for tr_idx, va_idx in tscv.split(X.values):
                    yp = model.predict(X.values[va_idx])
                    f1s.append(f1_score(y[va_idx], yp, average='macro', zero_division=0))
                    try:
                        prices_va = feat['close'].iloc[va_idx]
                        st = _simulate_pnl_from_preds(prices_va, yp)
                        pnl_sum += st['pnl']
                    except Exception:
                        pass
                metrics = {
                    'in_sample': { 'report': rep, 'confusion': cm },
                    'cv': { 'f1_macro': float(np.mean(f1s)) if f1s else 0.0, 'pnl_sum': float(pnl_sum) },
                    'params': None,
                }
                # persist back for faster future reads
                try:
                    pack['metrics'] = metrics
                    joblib.dump(pack, _model_path_for(cur_interval))
                except Exception:
                    pass
            except Exception:
                metrics = {}
        return jsonify({'ok': True, 'interval': pack.get('interval', cur_interval), 'metrics': metrics, 'params': metrics.get('params'), 'trained_at': pack.get('trained_at'), 'train_count': ml_state.get('train_count', 0)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def updater():
    cfg = load_config()
    try:
        _load_nb_coins()
    except Exception:
        pass
    state["ema_fast"] = cfg.ema_fast
    state["ema_slow"] = cfg.ema_slow
    state["market"] = cfg.market
    state["candle"] = cfg.candle
    # Prefill N/B COIN buckets for recent candles
    try:
        _prefill_nb_coins(str(cfg.candle), str(cfg.market), how_many=120)
    except Exception:
        pass
    try:
        _load_npc_hashes()
    except Exception:
        pass
    # Initial seed with candles
    try:
        df = get_candles(cfg.market, cfg.candle, count=max(cfg.ema_slow + 60, 120))
        sig = decide_signal(df, cfg.ema_fast, cfg.ema_slow)
        tail = df.tail(60)
        for t, p in zip(tail.index, tail["close"].astype(float)):
            state["history"].append((int(t.timestamp()*1000), float(p)))
        state["price"] = float(tail["close"].iloc[-1])
        state["signal"] = sig
    except Exception:
        pass

    tick = 0
    tick_sec = int(os.getenv("UI_TICK_SEC", "1"))
    recalc_every = int(os.getenv("UI_RECALC_SEC", "30"))
    while True:
        try:
            # Live price via ticker
            cp = pyupbit.get_current_price(cfg.market)
            if cp:
                now_ms = int(time.time() * 1000)
                state["price"] = float(cp)
                state["history"].append((now_ms, float(cp)))
            # Periodic recalc of signal from candles
            if tick % max(recalc_every, 1) == 0:
                df = get_candles(cfg.market, cfg.candle, count=max(cfg.ema_slow + 5, 60))
                state["signal"] = decide_signal(df, cfg.ema_fast, cfg.ema_slow)
        except Exception:
            pass
        tick += tick_sec
        time.sleep(tick_sec)


def _resolve_config():
    base = load_config()
    ov = bot_ctrl['cfg_override']
    # merge overrides if present
    base.paper = base.paper if ov['paper'] is None else bool(ov['paper'])
    base.order_krw = base.order_krw if ov['order_krw'] is None else int(ov['order_krw'])
    # attach pnl_ratio dynamically to base for Trader
    try:
        base.pnl_ratio = float(ov['pnl_ratio']) if ov['pnl_ratio'] is not None else float(getattr(base, 'pnl_ratio', 0.0))
    except Exception:
        base.pnl_ratio = float(getattr(base, 'pnl_ratio', 0.0))
    # Attach new ratios for profit/loss mapping
    try:
        base.pnl_profit_ratio = float(ov['pnl_profit_ratio']) if ov['pnl_profit_ratio'] is not None else float(getattr(base, 'pnl_profit_ratio', 0.0))
    except Exception:
        base.pnl_profit_ratio = float(getattr(base, 'pnl_profit_ratio', 0.0))
    try:
        base.pnl_loss_ratio = float(ov['pnl_loss_ratio']) if ov['pnl_loss_ratio'] is not None else float(getattr(base, 'pnl_loss_ratio', 0.0))
    except Exception:
        base.pnl_loss_ratio = float(getattr(base, 'pnl_loss_ratio', 0.0))
    base.ema_fast = base.ema_fast if ov['ema_fast'] is None else int(ov['ema_fast'])
    base.ema_slow = base.ema_slow if ov['ema_slow'] is None else int(ov['ema_slow'])
    base.candle = base.candle if ov['candle'] is None else str(ov['candle'])
    base.market = base.market if ov['market'] is None else str(ov['market'])
    base.interval_sec = base.interval_sec if ov['interval_sec'] is None else int(ov['interval_sec'])
    # keys (if provided via API)
    base.access_key = base.access_key if ov['access_key'] is None else str(ov['access_key'])
    base.secret_key = base.secret_key if ov['secret_key'] is None else str(ov['secret_key'])
    # Feature flag: ML-only autotrade (ignore zone-side/order checks except min notional)
    try:
        base.ml_only = bool(ov.get('ml_only'))
    except Exception:
        base.ml_only = False
    try:
        base.ml_seg_only = bool(ov.get('ml_seg_only'))
    except Exception:
        base.ml_seg_only = False
    return base

def _get_runtime_keys():
    """Return a tuple of (std_ak, std_sk, open_ak, open_sk) from overrides/env."""
    ov = bot_ctrl['cfg_override']
    std_ak = (ov.get('access_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_ACCESS_KEY')
    std_sk = (ov.get('secret_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_SECRET_KEY')
    open_ak = (ov.get('open_api_access_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
    open_sk = (ov.get('open_api_secret_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_OPEN_API_SECRET_KEY')
    return std_ak, std_sk, open_ak, open_sk

def _mask_key(v: str | None) -> str:
    if not v:
        return ''
    try:
        s = str(v)
        if len(s) <= 8:
            return s[:2] + ('*' * max(0, len(s) - 4)) + s[-2:]
        return s[:4] + ('*' * (len(s) - 8)) + s[-4:]
    except Exception:
        return '<?>'

def log_env_keys():
    std_ak, std_sk, open_ak, open_sk = _get_runtime_keys()
    print(f"[ENV] UPBIT_ACCESS_KEY={_mask_key(std_ak)} UPBIT_SECRET_KEY={_mask_key(std_sk)}")
    print(f"[ENV] UPBIT_OPEN_API_ACCESS_KEY={_mask_key(open_ak)} UPBIT_OPEN_API_SECRET_KEY={_mask_key(open_sk)}")

def _reload_env_vars() -> bool:
    try:
        # project root
        load_dotenv()
        load_dotenv("env.local", override=False)
        # bot dir (this file)
        base_dir = os.path.dirname(__file__)
        load_dotenv(os.path.join(base_dir, ".env"), override=True)
        load_dotenv(os.path.join(base_dir, "env.local"), override=True)
        return True
    except Exception:
        return False


def trade_loop():
    try:
        cfg = _resolve_config()
        upbit = None
        if not cfg.paper and cfg.access_key and cfg.secret_key:
            upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
        trader = Trader(
            upbit,
            TradeConfig(
                market=cfg.market,
                order_krw=cfg.order_krw,
                paper=cfg.paper,
                pnl_ratio=float(getattr(cfg, 'pnl_ratio', 0.0)),
                pnl_profit_ratio=float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
                pnl_loss_ratio=float(getattr(cfg, 'pnl_loss_ratio', 0.0)),
            )
        )
        last_signal = 'HOLD'
        # ML model cache for confirmation
        ml_pack = None
        ml_interval = None
        last_order_ts = 0
        # Prevent multiple orders within the same candle/bar
        last_order_bar_ts = 0
        
        # ===== 8BIT 마을 시스템 통합 =====
        # 촌장의 신뢰도 기반 지침 생성
        mayor_guidance = mayor_trust_guidance()
        print(f"🏛️ 촌장 지침: {mayor_guidance['guidance']['official_strategy']}")
        
        # 자동 촌장 지침 학습 체크 및 실행
        auto_mayor_guidance_learning()
        
        # 마을 주민들의 비트카 에너지 주입
        for trainer_name in VILLAGE_RESIDENTS.keys():
            energy_amount = BITCAR_ENERGY_SYSTEM[trainer_name]["energy"]
            result = inject_village_energy_to_bitcar(trainer_name, energy_amount)
            print(f"🚗 {trainer_name} 비트카: {result}")
        
        print("🍊 ORANGE 구역으로 출발합니다!")
        # ===== 마을 시스템 통합 완료 =====
        
        while bot_ctrl['running']:
            try:
                cfg = _resolve_config()
                # Use NB wave zone transitions: one SELL when entering ORANGE, one BUY when entering BLUE
                df = get_candles(cfg.market, cfg.candle, count=max(120, cfg.ema_slow + 5))
                price = float(df['close'].iloc[-1])
                # Compute r in [0,1]
                try:
                    ui_win = bot_ctrl['cfg_override'].get('nb_window')
                    window = int(ui_win) if ui_win is not None else int(load_nb_params().get('window', 50))
                except Exception:
                    window = 50
                r = _compute_r_from_ohlcv(df, window)
                r_last = float(r.iloc[-1]) if len(r) else 0.5
                # Update bot_ctrl with current r_value
                bot_ctrl['r_value'] = r_last
                
                # Current bar timestamp (ms) to dedupe orders per bar
                try:
                    bar_ts = int(df.index[-1].timestamp() * 1000)
                except Exception:
                    bar_ts = int(time.time() * 1000)
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
                if bot_ctrl.get('nb_zone') not in ('BLUE','ORANGE'):
                    bot_ctrl['nb_zone'] = 'ORANGE' if r_last >= 0.5 else 'BLUE'
                
                # Update ml_zone to match nb_zone for now (can be enhanced later)
                bot_ctrl['ml_zone'] = bot_ctrl['nb_zone']
                sig = 'HOLD'
                if bot_ctrl['nb_zone'] == 'BLUE' and r_last >= HIGH:
                    bot_ctrl['nb_zone'] = 'ORANGE'
                    sig = 'SELL'
                elif bot_ctrl['nb_zone'] == 'ORANGE' and r_last <= LOW:
                    bot_ctrl['nb_zone'] = 'BLUE'
                    sig = 'BUY'
                state['signal'] = sig if sig != 'HOLD' else state.get('signal', 'HOLD')
                state['price'] = price
                if sig in ('BUY','SELL') and sig != last_signal:
                    # One-order-per-bar: skip if we already ordered on this bar
                    if last_order_bar_ts and bar_ts == last_order_bar_ts:
                        # already ordered this bar; record reason and skip
                        try:
                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:already_ordered_this_bar"], int(time.time()*1000), { 'price': price })
                        except Exception:
                            pass
                        last_signal = sig
                        bot_ctrl['last_signal'] = sig
                        time.sleep(max(1, _resolve_config().interval_sec))
                        continue
                    # cooldown between orders (to avoid near-simultaneous flips)
                    try:
                        min_gap = int(bot_ctrl['cfg_override'].get('min_order_gap_sec') or os.getenv('MIN_ORDER_GAP_SEC', '10'))
                    except Exception:
                        min_gap = 10
                    now_ms = int(time.time()*1000)
                    if last_order_ts and (now_ms - last_order_ts) < max(0,min_gap)*1000:
                        try:
                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), [f"blocked:cooldown({min_gap}s)"], now_ms, { 'price': price })
                        except Exception:
                            pass
                        try:
                            _energy_tick(str(cfg.candle))
                        except Exception:
                            pass
                        last_signal = sig
                        bot_ctrl['last_signal'] = sig
                        time.sleep(max(1, _resolve_config().interval_sec))
                        continue
                    # Enforce single BUY→SELL cycle using position lock
                    try:
                        pos = str(bot_ctrl.get('position') or 'FLAT').upper()
                    except Exception:
                        pos = 'FLAT'
                    # Disallow consecutive BUYs; require SELL to flatten first
                    if sig == 'BUY' and pos == 'LONG':
                        try:
                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:already_long"], int(time.time()*1000), { 'price': price })
                        except Exception:
                            pass
                        try:
                            _energy_adjust(str(cfg.candle), -0.5, 'already_long')
                        except Exception:
                            pass
                        last_signal = sig
                        bot_ctrl['last_signal'] = sig
                        time.sleep(max(1, _resolve_config().interval_sec))
                        continue
                    # Disallow SELL when already flat (no prior BUY)
                    if sig == 'SELL' and pos != 'LONG':
                        try:
                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:not_long"], int(time.time()*1000), { 'price': price })
                        except Exception:
                            pass
                        try:
                            _energy_adjust(str(cfg.candle), -0.5, 'not_long')
                        except Exception:
                            pass
                        last_signal = sig
                        bot_ctrl['last_signal'] = sig
                        time.sleep(max(1, _resolve_config().interval_sec))
                        continue
                    # Optional: require ML confirmation
                    try:
                        require_ml = bool(bot_ctrl['cfg_override'].get('require_ml')) if bot_ctrl['cfg_override'].get('require_ml') is not None else (os.getenv('REQUIRE_ML_CONFIRM', 'false').lower()=='true')
                    except Exception:
                        require_ml = False
                    # Rest-after-first-coin: if within rest window, skip placing orders
                    try:
                        iv_rest = str(cfg.candle)
                        bnow = _bucket_ts_interval(int(time.time()*1000), iv_rest)
                        ru = int(_nb_rest_until.get(iv_rest) or 0)
                        if ru and bnow < ru:
                            _mark_nb_coin_block(iv_rest, str(cfg.market), ["rest:scheduled"], int(time.time()*1000), { 'price': price })
                            last_signal = sig
                            bot_ctrl['last_signal'] = sig
                            time.sleep(max(1, _resolve_config().interval_sec))
                            continue
                    except Exception:
                        pass
                    # Optional: require 100% zone probability
                    try:
                        zone100_only = bool(bot_ctrl['cfg_override'].get('zone100_only')) if bot_ctrl['cfg_override'].get('zone100_only') is not None else (os.getenv('ZONE100_ONLY', 'false').lower()=='true')
                    except Exception:
                        zone100_only = False
                    # If nb_force is true, skip optional gates and place order (respect cooldown/position lock)
                    try:
                        nb_force = bool(bot_ctrl['cfg_override'].get('nb_force')) if bot_ctrl['cfg_override'].get('nb_force') is not None else (os.getenv('NB_FORCE','false').lower()=='true')
                    except Exception:
                        nb_force = False

                    # Energy-aware gating (E low → enforce stronger guards; very low → pause)
                    try:
                        E = float(_energy_tick(str(cfg.candle)))
                        e_block = float(os.getenv('ENERGY_BLOCK_TH', '5'))
                        e_pull = float(os.getenv('ENERGY_ENFORCE_PULLBACK_TH', '30'))
                        e_zone = float(os.getenv('ENERGY_ENFORCE_ZONE100_TH', '30'))
                        if E <= e_block:
                            try:
                                _mark_nb_coin_block(str(cfg.candle), str(cfg.market), [f"blocked:energy_low({E:.1f})"], int(time.time()*1000), { 'price': price })
                            except Exception:
                                pass
                            last_signal = sig
                            bot_ctrl['last_signal'] = sig
                            time.sleep(max(1, _resolve_config().interval_sec))
                            continue
                        # below thresholds → tighten gates
                        energy_enforce_pullback = (E < e_pull)
                        energy_enforce_zone100 = (E < e_zone)
                    except Exception:
                        energy_enforce_pullback = False
                        energy_enforce_zone100 = False

                    if not nb_force and require_ml:
                        try:
                            if ml_interval != cfg.candle or ml_pack is None:
                                ml_pack = _load_ml(cfg.candle)
                                ml_interval = cfg.candle
                            if ml_pack is not None:
                                model = ml_pack['model']
                                window = int(ml_pack.get('window', 50))
                                ema_fast = int(ml_pack.get('ema_fast', 10))
                                ema_slow = int(ml_pack.get('ema_slow', 30))
                                feat = _build_features(df, window, ema_fast, ema_slow, 5).dropna().copy()
                                # Respect trained feature order if available
                                trained_cols = list(ml_pack.get('feature_names') or [])
                                if not trained_cols:
                                    base_cols = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
                                    cols_ext = ['zone_flag','dist_high','dist_low','extreme_gap','zone_conf','zone_min_r','zone_max_r','zone_extreme_r','zone_extreme_age']
                                    cand = base_cols + [c for c in cols_ext if c in feat.columns]
                                    try:
                                        need = int(getattr(model, 'n_features_in_', len(cand)))
                                    except Exception:
                                        need = len(cand)
                                    trained_cols = cand[:need]
                                Xv = feat[[c for c in trained_cols if c in feat.columns]].values
                                ml_pred = int(model.predict(Xv)[-1]) if len(Xv) else 0
                                # Auto-sync server candle to ML model interval if they diverge
                                try:
                                    ml_used_interval = str(ml_pack.get('interval') or cfg.candle)
                                except Exception:
                                    ml_used_interval = cfg.candle
                                if ml_used_interval and ml_used_interval != cfg.candle:
                                    bot_ctrl['cfg_override']['candle'] = ml_used_interval
                                    state['candle'] = ml_used_interval
                                    # Skip this tick to reload with new interval
                                    try:
                                        _mark_nb_coin_block(str(cfg.candle), str(cfg.market), [f"blocked:ml_interval_switch->{ml_used_interval}"])
                                    except Exception:
                                        pass
                                    last_signal = sig
                                    bot_ctrl['last_signal'] = sig
                                    time.sleep(max(1, _resolve_config().interval_sec))
                                    continue
                                # Pullback from extreme enforcement (may be forced by low energy)
                                allow_by_pullback = True
                                try:
                                    need_pullback = bool(bot_ctrl['cfg_override'].get('require_pullback') or os.getenv('REQUIRE_PULLBACK', 'false').lower()=='true')
                                except Exception:
                                    need_pullback = False
                                # Energy may force pullback requirement
                                if energy_enforce_pullback:
                                    need_pullback = True
                                try:
                                    pullback_r = float(bot_ctrl['cfg_override'].get('pullback_r') or os.getenv('PULLBACK_R', '0.02'))
                                except Exception:
                                    pullback_r = 0.02
                                try:
                                    pullback_bars = int(bot_ctrl['cfg_override'].get('pullback_bars') or os.getenv('PULLBACK_BARS', '2'))
                                except Exception:
                                    pullback_bars = 2
                                if need_pullback:
                                    try:
                                        snap_pb = snap if 'snap' in locals() and isinstance(snap, dict) else _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, ml_pack)
                                        eg = float(snap_pb.get('extreme_gap', 0.0) or 0.0)
                                        age = int(snap_pb.get('zone_extreme_age', 0) or 0)
                                        allow_by_pullback = (eg >= pullback_r) and (age >= pullback_bars)
                                    except Exception:
                                        allow_by_pullback = False
                                # Zone 100% enforcement using latest insight snapshot
                                allow_by_zone100 = True
                                if zone100_only or energy_enforce_zone100:
                                    try:
                                        snap = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, ml_pack)
                                        pb = float(snap.get('pct_blue', 0.0) or 0.0)
                                        po = float(snap.get('pct_orange', 0.0) or 0.0)
                                        allow_by_zone100 = (pb >= 99.95 or po >= 99.95)
                                    except Exception:
                                        allow_by_zone100 = False
                                # Multi-timeframe group consensus
                                allow_by_group = True
                                try:
                                    need_group = bool(bot_ctrl['cfg_override'].get('require_group') or os.getenv('REQUIRE_GROUP', 'false').lower()=='true')
                                except Exception:
                                    need_group = False
                                if need_group:
                                    try:
                                        intervals = bot_ctrl['cfg_override'].get('group_intervals') or ['minute1','minute3','minute5']
                                        buy_th = float(bot_ctrl['cfg_override'].get('group_buy_th') or os.getenv('GROUP_BUY_TH','70'))
                                        sell_th = float(bot_ctrl['cfg_override'].get('group_sell_th') or os.getenv('GROUP_SELL_TH','70'))
                                        blue_sum=0.0; orange_sum=0.0; cnt=0
                                        for iv in intervals:
                                            dfx = get_candles(cfg.market, iv, count=max(120, window*2))
                                            rvx = float(_compute_r_from_ohlcv(dfx, window).iloc[-1]) if len(dfx) else 0.5
                                            HIGH = float(os.getenv('NB_HIGH', '0.55')); LOW = float(os.getenv('NB_LOW', '0.45'))
                                            rng = max(1e-9, HIGH-LOW)
                                            pbx = max(0.0, min(1.0, (HIGH - rvx)/rng))
                                            pox = max(0.0, min(1.0, (rvx - LOW)/rng))
                                            s0 = pbx+pox
                                            if s0>0: pbx,pox=pbx/s0,pox/s0
                                            blue_sum += pbx; orange_sum += pox; cnt += 1
                                        pb = (blue_sum/cnt*100.0) if cnt else 0.0
                                        po = (orange_sum/cnt*100.0) if cnt else 0.0
                                        if sig=='BUY': allow_by_group = (pb >= buy_th)
                                        elif sig=='SELL': allow_by_group = (po >= sell_th)
                                    except Exception:
                                        allow_by_group = False
                                cfg_now = _resolve_config()
                                if getattr(cfg_now, 'ml_only', False):
                                    # ML-only: only require ML direction to match NB signal
                                    if (ml_pred == 0) or (ml_pred == 1 and sig != 'BUY') or (ml_pred == -1 and sig != 'SELL'):
                                        try:
                                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), [f"blocked:ml_dir_mismatch pred={ml_pred} sig={sig}"])
                                        except Exception:
                                            pass
                                        try:
                                            _energy_adjust(str(cfg.candle), -0.5, 'ml_dir_mismatch')
                                        except Exception:
                                            pass
                                        last_signal = sig
                                        bot_ctrl['last_signal'] = sig
                                        time.sleep(max(1, _resolve_config().interval_sec))
                                        continue
                                else:
                                    if (ml_pred == 0) or (ml_pred == 1 and sig != 'BUY') or (ml_pred == -1 and sig != 'SELL') or (not allow_by_pullback) or (not allow_by_zone100) or (not allow_by_group):
                                        try:
                                            rs = []
                                            if ml_pred == 0: rs.append('blocked:ml_hold')
                                            if (ml_pred == 1 and sig != 'BUY') or (ml_pred == -1 and sig != 'SELL'):
                                                rs.append('blocked:ml_dir_mismatch')
                                            if not allow_by_pullback: rs.append('blocked:pullback')
                                            if not allow_by_zone100: rs.append('blocked:zone100')
                                            if not allow_by_group: rs.append('blocked:group')
                                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), rs)
                                        except Exception:
                                            pass
                                        try:
                                            _energy_adjust(str(cfg.candle), -0.5, 'blocked')
                                        except Exception:
                                            pass
                                        last_signal = sig
                                        bot_ctrl['last_signal'] = sig
                                        time.sleep(max(1, _resolve_config().interval_sec))
                                        continue
                        except Exception:
                            pass
                    # Enforce: only BUY in BLUE zone, only SELL in ORANGE zone (toggle-able)
                    try:
                        need_enforce = bool(bot_ctrl['cfg_override'].get('enforce_zone_side')) if bot_ctrl['cfg_override'].get('enforce_zone_side') is not None else (os.getenv('ENFORCE_ZONE_SIDE','false').lower()=='true')
                    except Exception:
                        need_enforce = False
                    if need_enforce:
                        try:
                            snap_guard = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, ml_pack)
                            z_now = str(snap_guard.get('zone') or ('ORANGE' if r_last >= 0.5 else 'BLUE')).upper()
                            if (sig == 'BUY' and z_now != 'BLUE') or (sig == 'SELL' and z_now != 'ORANGE'):
                                try:
                                    _mark_nb_coin_block(str(cfg.candle), str(cfg.market), [f"blocked:enforce_zone_side zone={z_now} sig={sig}"])
                                except Exception:
                                    pass
                                try:
                                    _energy_adjust(str(cfg.candle), -0.5, 'enforce_zone_side')
                                except Exception:
                                    pass
                                last_signal = sig
                                bot_ctrl['last_signal'] = sig
                                time.sleep(max(1, _resolve_config().interval_sec))
                                continue
                        except Exception:
                            pass
                    # Finance-aware gating by residents (live only)
                    try:
                        if not cfg.paper:
                            res = _score_strategies(str(cfg.candle))
                            feas = res.get('feasible') if isinstance(res, dict) else None
                            if sig == 'BUY' and (not feas or not feas.get('can_buy')):
                                _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:finance:no_buyable"], int(time.time()*1000), { 'price': price })
                                last_signal = sig
                                bot_ctrl['last_signal'] = sig
                                time.sleep(max(1, _resolve_config().interval_sec))
                                continue
                            if sig == 'SELL' and (not feas or not feas.get('can_sell')):
                                _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:finance:no_inventory"], int(time.time()*1000), { 'price': price })
                                last_signal = sig
                                bot_ctrl['last_signal'] = sig
                                time.sleep(max(1, _resolve_config().interval_sec))
                                continue
                    except Exception:
                        pass
                    # Update trader's dynamic pnl_ratio before each order
                    try:
                        trader.cfg.pnl_ratio = float(getattr(cfg, 'pnl_ratio', 0.0))
                    except Exception:
                        trader.cfg.pnl_ratio = 0.0
                    o = None
                    try:
                        o = trader.place(sig, price)
                    except Exception:
                        o = None
                    # snapshot current insight at order time
                    try:
                        snap_insight = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, ml_pack)
                    except Exception:
                        snap_insight = {}
                    # If live mode and order was not placed (e.g., min notional, no balance), skip logging
                    if (not cfg.paper) and (not isinstance(o, dict)):
                        try:
                            _mark_nb_coin_block(str(cfg.candle), str(cfg.market), ["blocked:live_min_notional_or_balance"])
                        except Exception:
                            pass
                        try:
                            _energy_adjust(str(cfg.candle), -1.0, 'live_fail')
                        except Exception:
                            pass
                        last_signal = sig
                        bot_ctrl['last_signal'] = sig
                        time.sleep(max(1, _resolve_config().interval_sec))
                        continue
                    order = {
                        'ts': int(time.time()*1000),
                        'side': sig,
                        'price': price,
                        'size': (o.get('size') if isinstance(o, dict) else None) or 0,
                        'paper': cfg.paper or bool((isinstance(o, dict) and o.get('paper'))),
                        'market': cfg.market,
                        'interval': str(cfg.candle),
                        'live_ok': bool(o.get('live_ok')) if isinstance(o, dict) else False,
                        'nb_signal': sig,
                        'nb_window': int(window),
                        'nb_r': float(r_last),
                        'insight': snap_insight,
                    }
                    orders.append(order)
                    try:
                        _mark_nb_coin(str(cfg.candle), str(cfg.market), sig, order.get('ts'), order)
                    except Exception:
                        pass
                    
                    # ===== 8BIT 마을 시스템 거래 기록 =====
                    # 각 트레이너의 창고에 거래 기록 저장
                    for trainer_name in VILLAGE_RESIDENTS.keys():
                        try:
                            # 신뢰도 계산
                            personal_confidence = VILLAGE_RESIDENTS[trainer_name].get('skillLevel', 1.0) * 100
                            weighted_confidence = calculate_weighted_confidence(
                                personal_confidence, 
                                MAYOR_TRUST_SYSTEM["ML_Model_Trust"], 
                                MAYOR_TRUST_SYSTEM["NB_Guild_Trust"]
                            )
                            
                            # 거래 데이터 준비
                            trade_data = {
                                'timestamp': datetime.now().isoformat(),
                                'action': sig,
                                'price': price,
                                'quantity': order.get('size', 0),
                                'pnl': 0,  # 나중에 계산
                                'strategy': VILLAGE_RESIDENTS[trainer_name].get('strategy', 'unknown'),
                                'zone': bot_ctrl.get('nb_zone', 'unknown'),
                                'confidence': weighted_confidence,
                                'is_real': not cfg.paper,
                                'market_condition': 'ORANGE' if bot_ctrl.get('nb_zone') == 'ORANGE' else 'BLUE',
                                'timing': 'immediate',
                                'lesson_learned': '거래 실행됨'
                            }
                            
                            # 창고에 거래 기록 저장
                            real_time_trade_recording(trainer_name, trade_data)
                            
                            # ===== 거래 일지 추가 =====
                            # 촌장 지침 기반 일지 생성
                            mayor_entry = create_mayor_guidance_entry(
                                trainer_name, 
                                bot_ctrl.get('nb_zone', 'unknown'), 
                                sig, 
                                f"{trainer_name}의 {sig} 거래 실행"
                            )
                            
                            # ML 모델 판단 기반 일지 생성
                            ml_entry = create_ml_decision_entry(
                                trainer_name,
                                bot_ctrl.get('nb_zone', 'unknown'),
                                sig,
                                MAYOR_TRUST_SYSTEM["ML_Model_Trust"],
                                personal_confidence
                            )
                            
                            # 일지에 추가
                            add_trade_journal_entry(trainer_name, mayor_entry)
                            add_trade_journal_entry(trainer_name, ml_entry)
                            
                            print(f"📦 {trainer_name} 창고에 거래 기록 저장: {sig} @ {price}")
                            print(f"📝 {trainer_name} 거래 일지 업데이트: {mayor_entry['mayor_guidance']}")
                            
                        except Exception as e:
                            print(f"❌ {trainer_name} 거래 기록 저장 실패: {e}")
                    # ===== 마을 시스템 거래 기록 완료 =====
                    last_order_ts = int(order['ts'])
                    last_order_bar_ts = int(bar_ts)
                    bot_ctrl['last_order'] = order
                    # Update position lock
                    try:
                        if sig == 'BUY':
                            bot_ctrl['position'] = 'LONG'
                        elif sig == 'SELL':
                            bot_ctrl['position'] = 'FLAT'
                    except Exception:
                        pass
                    # Energy reward/penalty on order outcome will be applied when accounting updates coin_count
                # No state change (HOLD) or after handling
                last_signal = sig
                bot_ctrl['last_signal'] = sig
            except Exception:
                pass
            time.sleep(max(1, _resolve_config().interval_sec))
    finally:
        bot_ctrl['running'] = False


@app.route('/api/stream')
def api_stream():
    def gen():
        last_ts = None
        last_order_ts = None
        while True:
            try:
                ts = state["history"][-1][0] if state["history"] else None
                if ts and ts != last_ts:
                    last_ts = ts
                    payload = {
                        "ts": ts,
                        "price": state.get("price", 0),
                        "signal": state.get("signal", "HOLD"),
                        "market": state.get("market"),
                        "candle": state.get("candle"),
                        "ema_fast": state.get("ema_fast"),
                        "ema_slow": state.get("ema_slow"),
                    }
                    # Include latest order only when there's a new one
                    if orders:
                        o = orders[-1]
                        if last_order_ts != o.get("ts"):
                            payload["order"] = o
                            last_order_ts = o.get("ts")
                    yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.5)
                continue
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    return Response(gen(), mimetype='text/event-stream', headers=headers)


@app.route("/api/state")
def api_state():
    return jsonify({
        "price": state["price"],
        "signal": state["signal"],
        "ema_fast": state["ema_fast"],
        "ema_slow": state["ema_slow"],
        "market": state["market"],
        "candle": state["candle"],
        "history": list(state["history"]),
    })


@app.route('/api/ohlcv')
def api_ohlcv():
    try:
        cfg = load_config()
        count = int((request.args.get('count') or 300))
        interval = request.args.get('interval') or cfg.candle
        df = get_candles(cfg.market, interval, count=count)
        out = []
        for idx, row in df.iterrows():
            out.append({
                'time': int(idx.timestamp()*1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0.0,
            })
        return jsonify({'market': state.get('market'), 'candle': state.get('candle'), 'data': out})
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@app.route('/api/orders', methods=['GET'])
def api_orders():
    """Return recent orders for plotting markers on the chart."""
    try:
        return jsonify({'ok': True, 'market': state.get('market'), 'data': list(orders)})
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@app.route('/api/order', methods=['POST'])
def api_order_create():
    """Accept order notifications from the trader (paper or live)."""
    try:
        if request.is_json:
            payload = request.get_json(force=True)
        else:
            payload = request.form.to_dict()
        # Normalize fields
        order = {
            'ts': int(payload.get('ts') or int(time.time() * 1000)),
            'side': str(payload.get('side', '')).upper(),
            'price': float(payload.get('price', 0) or 0),
            'size': float(payload.get('size', 0) or 0),
            'paper': bool(payload.get('paper', True) in (True, 'true', '1', 1, 'True')),
            'market': payload.get('market') or state.get('market'),
        }
        orders.append(order)
        try:
            _mark_nb_coin(str(state.get('candle') or load_config().candle), str(order.get('market') or state.get('market') or load_config().market), str(order.get('side') or 'NONE'), int(order.get('ts') or int(time.time()*1000)), order)
        except Exception:
            pass
        return jsonify({'ok': True, 'order': order})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/orders/clear', methods=['POST'])
def api_orders_clear():
    """Clear in-memory order log and return ok."""
    try:
        orders.clear()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/signal/log', methods=['POST'])
def api_signal_log():
    """Append an ML signal marker for later scoring/training.
    Body: { ts, zone, extreme, price, pct_major, slope_bp, horizon, pred_nb, interval }
    """
    try:
        payload = request.get_json(force=True)
        s = {
            'id': int(time.time()*1000),
            'ts': int(payload.get('ts')),
            'zone': str(payload.get('zone','')).upper(),
            'extreme': str(payload.get('extreme','')).upper(),
            'price': float(payload.get('price') or 0.0),
            'pct_major': float(payload.get('pct_major') or 0.0),
            'slope_bp': float(payload.get('slope_bp') or 0.0),
            'horizon': int(payload.get('horizon') or 0),
            'pred_nb': payload.get('pred_nb'),
            'interval': str(payload.get('interval') or (state.get('candle') or 'minute5')),
            'market': str(state.get('market') or load_config().market),
            'score0': max(0.0, min(1.0, float(payload.get('score0') or 0.0))),
            'realized_score': None,
        }
        signals.append(s)
        try:
            _mark_nb_coin(str(s.get('interval') or (state.get('candle') or 'minute5')),
                          str(s.get('market') or (state.get('market') or load_config().market)),
                          'BUY' if str(s.get('zone')).upper()=='BLUE' else ('SELL' if str(s.get('zone')).upper()=='ORANGE' else 'NONE'),
                          int(s.get('ts') or int(time.time()*1000)), None)
        except Exception:
            pass
        # optional: append to disk
        try:
            base_dir = os.path.dirname(__file__)
            path = os.path.join(base_dir, 'data', 'signals.jsonl')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        except Exception:
            pass
        return jsonify({'ok': True, 'signal': s})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


def _compute_r_from_ohlcv(df: pd.DataFrame, window: int) -> pd.Series:
    hi = df['high'].rolling(window=window, min_periods=window).max()
    lo = df['low'].rolling(window=window, min_periods=window).min()
    span = (hi - lo).replace(0, np.nan)
    r = (df['close'] - lo) / span
    # pandas 2.x: fillna(method=...) deprecated → use bfill()
    return r.clip(0, 1).bfill().fillna(0.5)


def _simulate_pnl_from_r(prices: pd.Series, r: pd.Series, buy_th: float, sell_th: float,
                         debounce: int = 0, fee_bps: float = 0.0) -> dict:
    pos = 0
    entry = 0.0
    pnl = 0.0
    wins = 0
    trades = 0
    peak = 0.0
    maxdd = 0.0
    last_sig_idx = -10**9
    for i, (p, rv) in enumerate(zip(prices.values, r.values)):
        if pos == 0 and rv >= buy_th and (i - last_sig_idx) >= debounce:
            pos = 1
            entry = float(p)
            trades += 1
            last_sig_idx = i
        elif pos == 1 and rv <= sell_th and (i - last_sig_idx) >= debounce:
            ret = float(p) - entry
            # apply fee (approx market in/out)
            ret -= abs(entry) * (fee_bps / 10000.0)
            ret -= abs(p) * (fee_bps / 10000.0)
            pnl += ret
            if ret > 0:
                wins += 1
            pos = 0
            entry = 0.0
            last_sig_idx = i
        peak = max(peak, pnl)
        maxdd = max(maxdd, peak - pnl)
    # close at last
    if pos == 1:
        p = float(prices.iloc[-1])
        ret = p - entry
        ret -= abs(entry) * (fee_bps / 10000.0)
        ret -= abs(p) * (fee_bps / 10000.0)
        pnl += ret
        if ret > 0:
            wins += 1
        pos = 0
    win_rate = (wins / trades * 100.0) if trades else 0.0
    return {
        'pnl': float(pnl),
        'trades': trades,
        'wins': wins,
        'win_rate': win_rate,
        'max_dd': float(maxdd),
    }


@app.route('/api/nb/optimize', methods=['POST'])
def api_nb_optimize():
    """Grid-search NB thresholds to maximize PnL on recent OHLCV.
    Body JSON: { window: int, buy: [start, stop, step], sell: [start, stop, step], debounce: int, fee_bps: float, count: int, interval: str }
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        window = int(payload.get('window', 50))
        buy_grid = payload.get('buy', [0.6, 0.85, 0.02])
        sell_grid = payload.get('sell', [0.15, 0.45, 0.02])
        debounce = int(payload.get('debounce', 6))
        fee_bps = float(payload.get('fee_bps', 10.0))  # 0.1%
        count = int(payload.get('count', 600))
        interval = payload.get('interval') or load_config().candle

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        if not {'open','high','low','close'}.issubset(df.columns):
            return jsonify({'ok': False, 'error': 'OHLCV missing', 'data': {}}), 400
        r = _compute_r_from_ohlcv(df, window)
        prices = df['close']

        b_start, b_stop, b_step = buy_grid
        s_start, s_stop, s_step = sell_grid
        best = None
        best_stats = None
        b = b_start
        while b <= b_stop + 1e-9:
            s = s_start
            while s <= s_stop + 1e-9:
                stats = _simulate_pnl_from_r(prices, r, b, s, debounce=debounce, fee_bps=fee_bps)
                if best is None or stats['pnl'] > best_stats['pnl']:
                    best = {'buy': round(b, 3), 'sell': round(s, 3)}
                    best_stats = stats
                s += s_step
            b += b_step

        # persist best and respond
        if best:
            save_nb_params({ 'buy': best['buy'], 'sell': best['sell'], 'window': window })
        return jsonify({'ok': True, 'best': best, 'stats': best_stats, 'saved': bool(best)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/zone')
def api_nb_zone():
    """Return current NB r and zone. Optional query params:
    - r: float (if provided, use this r directly)
    - interval: str (default: config.candle)
    - count: int (default: 300)
    - window: int (default: saved nb_params.window)
    """
    try:
        # thresholds: prefer env, else defaults
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH, LOW = 0.55, 0.45
        rng = max(1e-9, HIGH - LOW)

        q = request.args
        r_q = q.get('r')
        if r_q is not None:
            rv = float(r_q)
            interval = q.get('interval') or state.get('candle') or load_config().candle
            window = int(q.get('window') or load_nb_params().get('window', 50))
        else:
            cfg = load_config()
            interval = q.get('interval') or state.get('candle') or cfg.candle
            count = int(q.get('count') or 300)
            window = int(q.get('window') or load_nb_params().get('window', 50))
            df = get_candles(cfg.market, interval, count=count)
            r_series = _compute_r_from_ohlcv(df, window)
            rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
        p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
        p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
        s0 = p_blue_raw + p_orange_raw
        if s0 > 0:
            p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
        # Optional trend weighting when data available
        p_blue, p_orange = p_blue_raw, p_orange_raw
        try:
            trend_k = int(os.getenv('NB_TREND_K', '30'))
            trend_alpha = float(os.getenv('NB_TREND_ALPHA', '0.5'))
        except Exception:
            trend_k, trend_alpha = 30, 0.5
        if r_q is None:
            try:
                r_series = _compute_r_from_ohlcv(df, window).astype(float)
                if len(r_series) >= trend_k*2:
                    tail_now = r_series.iloc[-trend_k:]
                    tail_prev = r_series.iloc[-trend_k*2:-trend_k]
                    zmax_now, zmax_prev = float(tail_now.max()), float(tail_prev.max())
                    zmin_now, zmin_prev = float(tail_now.min()), float(tail_prev.min())
                    trend_orange = max(0.0, (zmax_prev - zmax_now) / rng)
                    trend_blue = max(0.0, (zmin_now - zmin_prev) / rng)
                    p_orange = max(0.0, min(1.0, p_orange_raw * (1.0 - trend_alpha * trend_orange)))
                    p_blue = max(0.0, min(1.0, p_blue_raw * (1.0 - trend_alpha * trend_blue)))
                    s = p_blue + p_orange
                    if s > 0:
                        p_blue, p_orange = p_blue/s, p_orange/s
            except Exception:
                pass
        zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
        return jsonify({
            'ok': True,
            'interval': interval,
            'window': window,
            'r': float(rv),
            'zone': zone,
            'pct_blue_raw': float(p_blue_raw*100.0),
            'pct_orange_raw': float(p_orange_raw*100.0),
            'pct_blue': float(p_blue*100.0),
            'pct_orange': float(p_orange*100.0),
            'high': float(HIGH),
            'low': float(LOW),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/group', methods=['POST'])
def api_nb_group():
    """Group multiple intervals at the current time and return per-interval NB stats and a consensus.
    Body JSON (all optional):
      - intervals: ["minute1","minute3","minute5","minute10"]
      - window: int (default saved nb_params.window)
      - weights: { interval: number }
      - tolerance_sec: number (default: interval length in sec)
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH, LOW = 0.55, 0.45
        rng = max(1e-9, HIGH - LOW)
        def interval_seconds(iv: str) -> int:
            if iv.startswith('minute'):
                try:
                    m = int(iv.replace('minute',''))
                except Exception:
                    m = 1
                return max(60, m*60)
            if iv == 'day':
                return 24*60*60
            if iv == 'minute60':
                return 60*60
            return 600
        cfg = load_config()
        intervals = payload.get('intervals') or ['minute1','minute3','minute5','minute10']
        base_window = int(payload.get('window', load_nb_params().get('window', 50)))
        weights = payload.get('weights') or { iv: max(1, interval_seconds(iv)//60) for iv in intervals }
        tol_sec = int(payload.get('tolerance_sec', 0))  # per-interval fallback below
        now = int(time.time())
        rows = []
        w_sum = 0.0
        blue_sum = 0.0
        orange_sum = 0.0
        for iv in intervals:
            try:
                sec = interval_seconds(iv)
                tol = tol_sec if tol_sec>0 else sec
                df = get_candles(cfg.market, iv, count=max(200, base_window*3))
                if df is None or df.empty:
                    continue
                ts_ms = int(df.index[-1].timestamp()*1000)
                ts_s = ts_ms//1000
                if abs(now - ts_s) > tol:
                    # skip very stale bars
                    continue
                r_series = _compute_r_from_ohlcv(df, base_window)
                rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
                p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
                p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
                s0 = p_blue_raw + p_orange_raw
                if s0>0:
                    p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
                z = 'ORANGE' if rv >= 0.5 else 'BLUE'
                w = float(weights.get(iv, 1.0))
                w_sum += w
                blue_sum += w * p_blue_raw
                orange_sum += w * p_orange_raw
                rows.append({
                    'interval': iv,
                    'time_ms': ts_ms,
                    'r': rv,
                    'zone': z,
                    'pct_blue_raw': float(p_blue_raw*100.0),
                    'pct_orange_raw': float(p_orange_raw*100.0),
                    'weight': w,
                })
            except Exception:
                continue
        consensus = {
            'pct_blue': float(blue_sum/w_sum*100.0) if w_sum>0 else 0.0,
            'pct_orange': float(orange_sum/w_sum*100.0) if w_sum>0 else 0.0,
            'count': len(rows),
        }
        return jsonify({ 'ok': True, 'intervals': intervals, 'window': base_window, 'items': rows, 'consensus': consensus })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/nb/train', methods=['POST'])
def api_nb_train():
    """Auto period split training (grid search per segment) and persist best.
    Body JSON: { count: int(1800), segments: int(3), window: int, debounce: int, fee_bps: float, interval: str }
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        count = int(payload.get('count', 1800))
        segments = max(1, int(payload.get('segments', 3)))
        window = int(payload.get('window', load_nb_params().get('window', 50)))
        debounce = int(payload.get('debounce', 6))
        fee_bps = float(payload.get('fee_bps', 10.0))
        interval = payload.get('interval') or load_config().candle

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        if len(df) < max(window*2, segments*50):
            return jsonify({'ok': False, 'error': 'Not enough data'}), 400
        r_all = _compute_r_from_ohlcv(df, window)
        prices_all = df['close']

        seg_len = len(df) // segments
        results = []
        def search_best(prices: pd.Series, r: pd.Series):
            best=None; best_stats=None
            b=0.6
            while b<=0.85+1e-9:
                s=0.15
                while s<=0.45+1e-9:
                    st = _simulate_pnl_from_r(prices, r, b, s, debounce=debounce, fee_bps=fee_bps)
                    if best is None or st['pnl']>best_stats['pnl']:
                        best={'buy':round(b,3),'sell':round(s,3)}; best_stats=st
                    s+=0.02
                b+=0.02
            return best, best_stats

        for i in range(segments):
            start = i*seg_len
            end = (i+1)*seg_len if i<segments-1 else len(df)
            r_seg = r_all.iloc[start:end]
            p_seg = prices_all.iloc[start:end]
            best, stats = search_best(p_seg, r_seg)
            results.append({'segment': i+1, 'start': int(df.index[start].timestamp()*1000), 'end': int(df.index[end-1].timestamp()*1000), 'best': best, 'stats': stats})

        # choose best by highest pnl; fallback to last segment if tie
        results_sorted = sorted(results, key=lambda x: x['stats']['pnl'], reverse=True)
        chosen = results_sorted[0]
        save_nb_params({ 'buy': chosen['best']['buy'], 'sell': chosen['best']['sell'], 'window': window })
        return jsonify({'ok': True, 'chosen': chosen, 'results': results, 'saved': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/params', methods=['GET', 'POST'])
def api_nb_params():
    try:
        if request.method == 'GET':
            return jsonify({ 'ok': True, 'params': load_nb_params() })
        # POST to manually set/override
        payload = request.get_json(force=True)
        p = load_nb_params()
        for k in ('buy','sell','window'):
            if k in payload:
                p[k] = payload[k]
        ok = save_nb_params(p)
        return jsonify({ 'ok': ok, 'params': p })
    except Exception as e:
        return jsonify({ 'ok': False, 'error': str(e)}), 500


def nb_auto_opt_loop():
    """Background auto-optimizer: periodically updates NB parameters."""
    while True:
        try:
            cfg = load_config()
            # quick grid for development
            payload = {
                'window': load_nb_params().get('window', 50),
                'buy': [0.6, 0.85, 0.025],
                'sell': [0.15, 0.45, 0.025],
                'debounce': 6,
                'fee_bps': 10.0,
                'count': 800,
                'interval': state.get('candle') or cfg.candle,
            }
            # run optimize inline
            try:
                # reuse internal helpers
                df = get_candles(cfg.market, payload['interval'], count=payload['count'])
                r = _compute_r_from_ohlcv(df, payload['window'])
                prices = df['close']
                best=None; best_stats=None
                b=payload['buy'][0]
                while b <= payload['buy'][1] + 1e-9:
                    s=payload['sell'][0]
                    while s <= payload['sell'][1] + 1e-9:
                        stats = _simulate_pnl_from_r(prices, r, b, s, debounce=payload['debounce'], fee_bps=payload['fee_bps'])
                        if best is None or stats['pnl'] > best_stats['pnl']:
                            best={'buy': round(b,3), 'sell': round(s,3)}; best_stats=stats
                        s += payload['sell'][2]
                    b += payload['buy'][2]
                if best:
                    save_nb_params({ 'buy': best['buy'], 'sell': best['sell'], 'window': payload['window'] })
            except Exception:
                pass
        finally:
            # sleep (dev: 10 minutes; configurable via NB_OPT_MIN env)
            mins = int(os.getenv('NB_OPT_MIN', '10'))
            time.sleep(max(60, mins*60))

@app.route('/api/balance')
def api_balance():
    """Return Upbit balances (requires API keys and PAPER=false).
    Uses runtime-resolved config so UI Paper toggle takes effect.
    """
    try:
        cfg = _resolve_config()
        if cfg.paper:
            return jsonify({'ok': True, 'paper': True, 'balances': []})
        # Prefer standard keys from config; otherwise support UPBIT_OPEN_API_* env style (JWT direct call)
        bals = None
        std_ak, std_sk, open_ak, open_sk = _get_runtime_keys()
        if std_ak and std_sk:
            up = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
            bals = up.get_balances()
        else:
            # Try JWT-based private API using env: UPBIT_OPEN_API_ACCESS_KEY, UPBIT_OPEN_API_SECRET_KEY
            ak = open_ak or std_ak
            sk = open_sk or std_sk
            server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')
            if not ak or not sk:
                return jsonify({'ok': False, 'error': 'missing_keys'}), 400
            try:
                # Lazy import PyJWT
                import jwt as pyjwt  # type: ignore
            except Exception:
                return jsonify({'ok': False, 'error': 'pyjwt_not_installed'}), 500
            payload = {
                'access_key': ak,
                'nonce': str(uuid.uuid4()),
            }
            token = pyjwt.encode(payload, sk, algorithm='HS256')
            headers = { 'Authorization': f'Bearer {token}', 'Accept': 'application/json' }
            resp = requests.get(server_url.rstrip('/') + '/v1/accounts', headers=headers, timeout=10)
            if resp.status_code >= 400:
                return jsonify({'ok': False, 'error': f'upbit_http_{resp.status_code}', 'body': resp.text[:200]}), 400
            try:
                bals = resp.json()
            except Exception as e:
                return jsonify({'ok': False, 'error': f'invalid_json: {e}', 'body': resp.text[:200]}), 500
        cleaned = []
        for b in (bals or []):
            try:
                cleaned.append({
                    'currency': b.get('currency'),
                    'balance': float(b.get('balance', 0) or 0),
                    'locked': float(b.get('locked', 0) or 0),
                    'avg_buy_price': float(b.get('avg_buy_price', 0) or 0),
                    'unit_currency': b.get('unit_currency', 'KRW'),
                })
            except Exception:
                continue
        # Enrich with current KRW price and asset_value
        out = []
        for row in cleaned:
            try:
                cur = (row.get('currency') or '').upper()
                bal = float(row.get('balance') or 0)
                if cur == 'KRW':
                    price = 1.0
                    asset_value = bal
                else:
                    try:
                        price = float(pyupbit.get_current_price(f"KRW-{cur}") or 0.0)
                    except Exception:
                        price = 0.0
                    asset_value = float(bal * price)
                row['price'] = price
                row['asset_value'] = asset_value
                out.append(row)
            except Exception:
                out.append(row)
        return jsonify({'ok': True, 'paper': False, 'balances': out})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/trade/preflight', methods=['GET'])
def api_trade_preflight():
    """Return whether live trading is feasible right now without placing an order."""
    try:
        cfg = _resolve_config()
        std_ak, std_sk, open_ak, open_sk = _get_runtime_keys()
        resp = {
            'paper': bool(cfg.paper),
            'has_keys': bool((std_ak and std_sk) or (open_ak and open_sk)),
            'has_std_keys': bool(std_ak and std_sk),
            'has_open_keys': bool(open_ak and open_sk),
            'market': cfg.market,
            'candle': cfg.candle,
        }
        # price
        price = 0.0
        try:
            price = float(pyupbit.get_current_price(cfg.market) or 0.0)
            if price > 0:
                resp['price_source'] = 'ticker'
        except Exception:
            price = 0.0
        # Fallback: if ticker price unavailable, use last candle close
        if price <= 0:
            try:
                dfx = get_candles(cfg.market, cfg.candle, count=1)
                if len(dfx):
                    price = float(dfx['close'].iloc[-1])
                    resp['price_source'] = 'candle'
            except Exception:
                pass
        resp['price'] = price
        # balances
        avail_krw = 0.0; coin_bal = 0.0
        if not cfg.paper and std_ak and std_sk:
            try:
                up = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
                avail_krw = float(up.get_balance('KRW') or 0.0)
                coin = cfg.market.split('-')[-1]
                coin_bal = float(up.get_balance(coin) or 0.0)
            except Exception:
                pass
        else:
            # No standard keys available for live queries; cannot trade live
            if not cfg.paper and (not std_ak or not std_sk):
                resp['reason'] = 'missing_standard_keys'
        resp['krw'] = avail_krw
        resp['coin_balance'] = coin_bal
        # planned amounts (same normalization rules)
        try:
            ratio = float(getattr(cfg, 'pnl_ratio', 0.0))
        except Exception:
            ratio = 0.0
        spend = None
        if ratio > 0 and avail_krw > 0:
            try:
                spend = int(max(0, (avail_krw * (max(0.0, min(100.0, ratio)) / 100.0))))
                spend = (spend // 1000) * 1000
                spend = max(5000, min(spend, int(avail_krw)))
            except Exception:
                spend = None
        fallback = int(getattr(cfg, 'order_krw', 5000))
        fallback = (fallback // 1000) * 1000
        if fallback < 5000:
            fallback = 5000
        buy_krw = spend if (spend and spend >= 5000) else fallback
        resp['planned_buy_krw'] = buy_krw
        sell_size = coin_bal
        if ratio > 0 and coin_bal > 0:
            sell_size = coin_bal * (max(0.0, min(100.0, ratio)) / 100.0)
        try:
            sell_size = math.floor(float(sell_size) * 1e8) / 1e8
        except Exception:
            pass
        resp['planned_sell_size'] = float(sell_size)
        min_ok_buy = (not cfg.paper) and bool(std_ak and std_sk) and (avail_krw >= 5000) and (buy_krw >= 5000)
        min_ok_sell = (not cfg.paper) and bool(std_ak and std_sk) and (price > 0) and (sell_size > 0) and ((sell_size * price) >= 5000)
        resp['can_buy'] = bool(min_ok_buy)
        resp['can_sell'] = bool(min_ok_sell)
        return jsonify({'ok': True, 'preflight': resp})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/trade/buy', methods=['POST'])
def api_trade_buy():
    try:
        payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception:
        payload = {}
    cfg = _resolve_config()
    market = str(payload.get('market') or cfg.market)
    try:
        krw = int(payload.get('krw')) if payload.get('krw') is not None else int(cfg.order_krw)
    except Exception:
        krw = int(cfg.order_krw)
    try:
        pnl_ratio = float(payload.get('pnl_ratio')) if payload.get('pnl_ratio') is not None else float(getattr(cfg, 'pnl_ratio', 0.0))
    except Exception:
        pnl_ratio = float(getattr(cfg, 'pnl_ratio', 0.0))
    paper = cfg.paper if ('paper' not in payload) else bool(payload.get('paper') in (True, 'true', '1', 1, 'True'))
    # optional: record attempts under a specific bucket (sec epoch) for UI card association
    try:
        bucket_override = payload.get('bucket')
        bucket_ts_ms = int(bucket_override)*1000 if bucket_override is not None else None
    except Exception:
        bucket_ts_ms = None
    upbit = None
    if not paper and cfg.access_key and cfg.secret_key:
        upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
    trader = Trader(upbit, TradeConfig(market=market, order_krw=krw, paper=paper, pnl_ratio=pnl_ratio,
                                       pnl_profit_ratio=float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
                                       pnl_loss_ratio=float(getattr(cfg, 'pnl_loss_ratio', 0.0))))
    try:
        df = get_candles(market, cfg.candle, count=max(60, cfg.ema_slow+5))
        price = float(df['close'].iloc[-1]) if len(df) else 0.0
    except Exception:
        price = 0.0
    # Zone gating: require BLUE and near-100% confidence
    try:
        window = int(load_nb_params().get('window', 50))
    except Exception:
        window = 50
    try:
        ins = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, None)
    except Exception:
        ins = {}
    try:
        th = float(os.getenv('ZONE100_TH', '99.95'))
    except Exception:
        th = 99.95
    z = str(ins.get('zone') or '').upper()
    pb = float(ins.get('pct_blue') or ins.get('pct_blue_raw') or 0.0)
    po = float(ins.get('pct_orange') or ins.get('pct_orange_raw') or 0.0)
    if not (z == 'BLUE' and max(pb, po) >= th):
        try:
            _record_nb_attempt(str(cfg.candle), str(cfg.market), 'BUY', ok=False, error='blocked_by_zone_rule', ts_ms=(bucket_ts_ms or int(time.time()*1000)), meta={'zone': z, 'pct_blue': pb, 'pct_orange': po})
        except Exception:
            pass
        return jsonify({'ok': False, 'error': 'blocked_by_zone_rule', 'zone': z, 'pct_blue': pb, 'pct_orange': po})
    # Estimate intended spend/size for logging
    attempt_krw = 0
    attempt_size = 0.0
    try:
        if pnl_ratio > 0:
            try:
                avail_krw = float((upbit.get_balance('KRW') if upbit else 0.0) or 0.0)
            except Exception:
                avail_krw = 0.0
            attempt_krw = int(max(0, (avail_krw * (max(0.0, min(100.0, pnl_ratio)) / 100.0))))
            attempt_krw = (attempt_krw // 1000) * 1000
            if attempt_krw < 5000:
                attempt_krw = 5000
        else:
            attempt_krw = int(krw)
            attempt_krw = (attempt_krw // 1000) * 1000
            if attempt_krw < 5000:
                attempt_krw = 5000
        attempt_size = (float(attempt_krw) / float(price)) if price > 0 else 0.0
    except Exception:
        attempt_krw = int(krw)
        attempt_size = 0.0
    o = trader.place('BUY', price)
    if o is None or (not paper and not (isinstance(o, dict) and o.get('live_ok'))):
        try:
            _record_nb_attempt(str(cfg.candle), str(cfg.market), 'BUY', ok=False, error='buy_failed', ts_ms=(bucket_ts_ms or int(time.time()*1000)), meta={'price': price})
        except Exception:
            pass
        return jsonify({'ok': False, 'error': 'buy_failed'})
    # ins already computed above
    order = {
        'ts': int(time.time()*1000),
        'side': 'BUY',
        'price': float(price),
        'size': float(o.get('size') or attempt_size) if isinstance(o, dict) else float(attempt_size),
        'paper': bool(paper),
        'market': market,
        'live_ok': bool(o.get('live_ok')) if isinstance(o, dict) else False,
        'insight': ins,
    }
    try:
        orders.append(order)
    except Exception:
        pass
    try:
        _mark_nb_coin(str(cfg.candle), str(cfg.market), 'BUY', order.get('ts'), order)
    except Exception:
        pass
    try:
        _apply_coin_accounting(str(cfg.candle), float(order.get('price') or 0.0), 'BUY')
    except Exception:
        pass
    try:
        _record_nb_attempt(str(cfg.candle), str(cfg.market), 'BUY', ok=True, error=None, ts_ms=(bucket_ts_ms or order.get('ts')), meta={'price': order.get('price'), 'size': order.get('size')})
    except Exception:
        pass
    
    # Update trainer storage warehouse
    try:
        trainer = payload.get('trainer', 'Scout')  # 기본값은 Scout
        if trainer in ['Scout', 'Guardian', 'Analyst', 'Elder']:
            _update_trainer_storage(
                trainer=trainer,
                action='BUY',
                price=float(order.get('price') or 0.0),
                size=float(order.get('size') or 0.0)
            )
    except Exception:
        pass
    
    return jsonify({'ok': True, 'order': order})

@app.route('/api/trade/sell', methods=['POST'])
def api_trade_sell():
    try:
        payload = request.get_json(force=True) if request.is_json else request.form.to_dict()
    except Exception:
        payload = {}
    cfg = _resolve_config()
    market = str(payload.get('market') or cfg.market)
    try:
        size_override = float(payload.get('size')) if payload.get('size') is not None else None
    except Exception:
        size_override = None
    try:
        pnl_ratio = float(payload.get('pnl_ratio')) if payload.get('pnl_ratio') is not None else float(getattr(cfg, 'pnl_ratio', 0.0))
    except Exception:
        pnl_ratio = float(getattr(cfg, 'pnl_ratio', 0.0))
    paper = cfg.paper if ('paper' not in payload) else bool(payload.get('paper') in (True, 'true', '1', 1, 'True'))
    # optional: record attempts under a specific bucket (sec epoch) for UI card association
    try:
        bucket_override = payload.get('bucket')
        bucket_ts_ms = int(bucket_override)*1000 if bucket_override is not None else None
    except Exception:
        bucket_ts_ms = None
    upbit = None
    if not paper and cfg.access_key and cfg.secret_key:
        upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
    trader = Trader(upbit, TradeConfig(market=market, order_krw=int(cfg.order_krw), paper=paper, pnl_ratio=pnl_ratio,
                                       pnl_profit_ratio=float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
                                       pnl_loss_ratio=float(getattr(cfg, 'pnl_loss_ratio', 0.0))))
    try:
        df = get_candles(market, cfg.candle, count=max(60, cfg.ema_slow+5))
        price = float(df['close'].iloc[-1]) if len(df) else 0.0
    except Exception:
        price = 0.0
    # Zone gating: require ORANGE and near-100% confidence
    try:
        window = int(load_nb_params().get('window', 50))
    except Exception:
        window = 50
    try:
        ins = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, None)
    except Exception:
        ins = {}
    try:
        th = float(os.getenv('ZONE100_TH', '99.95'))
    except Exception:
        th = 99.95
    z = str(ins.get('zone') or '').upper()
    pb = float(ins.get('pct_blue') or ins.get('pct_blue_raw') or 0.0)
    po = float(ins.get('pct_orange') or ins.get('pct_orange_raw') or 0.0)
    if not (z == 'ORANGE' and max(pb, po) >= th):
        try:
            _record_nb_attempt(str(cfg.candle), str(cfg.market), 'SELL', ok=False, error='blocked_by_zone_rule', ts_ms=(bucket_ts_ms or int(time.time()*1000)), meta={'zone': z, 'pct_blue': pb, 'pct_orange': po})
        except Exception:
            pass
        return jsonify({'ok': False, 'error': 'blocked_by_zone_rule', 'zone': z, 'pct_blue': pb, 'pct_orange': po})
    if (not paper) and size_override and price>0 and (size_override*price)>=5000:
        try:
            o = upbit.sell_market_order(market, size_override)
            if isinstance(o, dict): o['live_ok'] = True
        except Exception:
            o = None
    else:
        # Estimate intended sell size for logging
        attempt_size = 0.0
        try:
            coin = market.split('-')[-1]
            bal = float((upbit.get_balance(coin) if upbit else 0.0) or 0.0)
        except Exception:
            bal = 0.0
        try:
            if size_override:
                attempt_size = float(size_override)
            elif pnl_ratio > 0 and bal > 0:
                attempt_size = bal * (max(0.0, min(100.0, pnl_ratio)) / 100.0)
            else:
                attempt_size = bal
            # round to 8dp
            attempt_size = math.floor(float(attempt_size) * 1e8) / 1e8
        except Exception:
            attempt_size = 0.0
        o = trader.place('SELL', price)
    if o is None or (not paper and not (isinstance(o, dict) and o.get('live_ok'))):
        try:
            _record_nb_attempt(str(cfg.candle), str(cfg.market), 'SELL', ok=False, error='sell_failed_or_min_notional', ts_ms=(bucket_ts_ms or int(time.time()*1000)), meta={'price': price, 'size': float(size_override or 0.0)})
        except Exception:
            pass
        return jsonify({'ok': False, 'error': 'sell_failed_or_min_notional'})
    try:
        window = int(load_nb_params().get('window', 50))
    except Exception:
        window = 50
    try:
        ins = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, cfg.candle, None)
    except Exception:
        ins = {}
    order = {
        'ts': int(time.time()*1000),
        'side': 'SELL',
        'price': float(price),
        'size': float(o.get('size') or (size_override if size_override else attempt_size)) if isinstance(o, dict) else float(size_override if size_override else attempt_size),
        'paper': bool(paper),
        'market': market,
        'live_ok': bool(o.get('live_ok')) if isinstance(o, dict) else False,
        'insight': ins,
    }
    try:
        orders.append(order)
    except Exception:
        pass
    try:
        _mark_nb_coin(str(cfg.candle), str(cfg.market), 'SELL', order.get('ts'), order)
    except Exception:
        pass
    try:
        _apply_coin_accounting(str(cfg.candle), float(order.get('price') or 0.0), 'SELL')
    except Exception:
        pass
    try:
        _record_nb_attempt(str(cfg.candle), str(cfg.market), 'SELL', ok=True, error=None, ts_ms=(bucket_ts_ms or order.get('ts')), meta={'price': order.get('price'), 'size': order.get('size')})
    except Exception:
        pass
    
    # Update trainer storage warehouse
    try:
        trainer = payload.get('trainer', 'Scout')  # 기본값은 Scout
        if trainer in ['Scout', 'Guardian', 'Analyst', 'Elder']:
            _update_trainer_storage(
                trainer=trainer,
                action='SELL',
                price=float(order.get('price') or 0.0),
                size=float(order.get('size') or 0.0)
            )
    except Exception:
        pass
    
    return jsonify({'ok': True, 'order': order})
@app.route('/api/bot/config', methods=['POST'])
def api_bot_config():
    try:
        data = request.get_json(force=True)
        # Optional: reload env vars on demand
        if data.get('reload_env'):
            _reload_env_vars()
        ov = bot_ctrl['cfg_override']
        for k in ('paper','order_krw','pnl_ratio','pnl_profit_ratio','pnl_loss_ratio','ema_fast','ema_slow','candle','market','interval_sec','require_ml','enforce_zone_side','nb_force','nb_window','ml_only','ml_seg_only',
                  'access_key','secret_key','open_api_access_key','open_api_secret_key'):
            if k in data:
                ov[k] = data[k]
        # reflect into global state for UI
        cfg = _resolve_config()
        state['ema_fast'] = cfg.ema_fast
        state['ema_slow'] = cfg.ema_slow
        state['market'] = cfg.market
        state['candle'] = cfg.candle
        return jsonify({'ok': True, 'config': {
            'paper': cfg.paper,
            'order_krw': cfg.order_krw,
            'pnl_ratio': float(getattr(cfg, 'pnl_ratio', 0.0)),
            'ema_fast': cfg.ema_fast,
            'ema_slow': cfg.ema_slow,
            'candle': cfg.candle,
            'market': cfg.market,
            'interval_sec': cfg.interval_sec,
            'pnl_profit_ratio': float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
            'pnl_loss_ratio': float(getattr(cfg, 'pnl_loss_ratio', 0.0)),
            'has_keys': bool((_get_runtime_keys()[0] and _get_runtime_keys()[1]) or (_get_runtime_keys()[2] and _get_runtime_keys()[3]))
        }})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    if bot_ctrl['running']:
        return jsonify({'ok': True, 'running': True})
    bot_ctrl['running'] = True
    t = threading.Thread(target=trade_loop, daemon=True)
    bot_ctrl['thread'] = t
    t.start()
    return jsonify({'ok': True, 'running': True})


@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    bot_ctrl['running'] = False
    return jsonify({'ok': True, 'running': False})


@app.route('/api/trainer/storage', methods=['GET'])
def api_trainer_storage():
    """트레이너별 저장 창고 정보 조회"""
    try:
        trainer = request.args.get('trainer')
        if trainer and trainer in _trainer_storage:
            return jsonify({'ok': True, 'storage': _trainer_storage[trainer]})
        else:
            return jsonify({'ok': True, 'storage': _trainer_storage})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/trainer/storage/modify', methods=['POST'])
def api_trainer_storage_modify():
    """트레이너별 저장 창고 수정 (N/B 길드 NPC 제어)"""
    try:
        data = request.get_json(force=True)
        trainer = data.get('trainer')
        amount = float(data.get('amount', 0.0))
        
        if not trainer or trainer not in ['Scout', 'Guardian', 'Analyst', 'Elder']:
            return jsonify({'ok': False, 'error': 'Invalid trainer name'}), 400
        
        # Get current price for entry price calculation
        current_price = 0.0
        try:
            # Try to get current price from preflight API
            cfg = _resolve_config()
            if cfg.access_key and cfg.secret_key:
                upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
                ticker = upbit.get_ticker(cfg.market)
                if ticker and 'trade_price' in ticker:
                    current_price = float(ticker['trade_price'])
            else:
                # Fallback: try to get from market data
                market_data = _get_market_data()
                if market_data and 'price' in market_data:
                    current_price = float(market_data['price'])
        except Exception as e:
            print(f"Warning: Could not get current price: {e}")
            # Use a fallback price if available
            current_price = 161000000  # fallback price
        
        # Update trainer storage
        if trainer in _trainer_storage:
            current_coins = _trainer_storage[trainer]['coins']
            new_coins = max(0.0, current_coins + amount)  # Prevent negative coins
            
            # Update coins
            _trainer_storage[trainer]['coins'] = new_coins
            
            # Update entry price if adding coins
            if amount > 0 and current_price > 0:
                if current_coins > 0:
                    # Weighted average of existing and new coins
                    total_value = (current_coins * _trainer_storage[trainer]['entry_price']) + (amount * current_price)
                    _trainer_storage[trainer]['entry_price'] = total_value / new_coins
                else:
                    # First time adding coins
                    _trainer_storage[trainer]['entry_price'] = current_price
            
            # Update last update time
            _trainer_storage[trainer]['last_update'] = int(time.time())
            
            # Add to trade history
            _trainer_storage[trainer]['trades'].append({
                'timestamp': int(time.time()),
                'action': 'MANUAL_MODIFY',
                'amount': amount,
                'price': current_price,
                'new_balance': new_coins
            })
            
            # Save to file
            _save_trainer_storage()
            
            print(f"✅ Trainer storage modified: {trainer} {amount:+.8f} BTC (new balance: {new_coins:.8f} BTC)")
            
            return jsonify({
                'ok': True, 
                'trainer': trainer,
                'amount': amount,
                'new_balance': new_coins,
                'entry_price': _trainer_storage[trainer]['entry_price']
            })
        else:
            return jsonify({'ok': False, 'error': 'Trainer not found in storage'}), 404
            
    except Exception as e:
        print(f"❌ Error modifying trainer storage: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/trainer/storage/reset', methods=['POST'])
def api_trainer_storage_reset():
    """트레이너별 저장 창고 평균가 초기화"""
    try:
        data = request.get_json(force=True)
        trainer = data.get('trainer')
        
        if not trainer or trainer not in ['Scout', 'Guardian', 'Analyst', 'Elder']:
            return jsonify({'ok': False, 'error': 'Invalid trainer name'}), 400
        
        if trainer in _trainer_storage:
            # 평균가 초기화
            _trainer_storage[trainer]['entry_price'] = 0.0
            _trainer_storage[trainer]['last_update'] = int(time.time())
            
            # 거래 기록에 추가
            _trainer_storage[trainer]['trades'].append({
                'timestamp': int(time.time()),
                'action': 'RESET_AVG_PRICE',
                'amount': 0.0,
                'price': 0.0,
                'new_balance': _trainer_storage[trainer]['coins']
            })
            
            # Save to file
            _save_trainer_storage()
            
            print(f"✅ Trainer storage average price reset: {trainer}")
            
            return jsonify({
                'ok': True, 
                'trainer': trainer,
                'entry_price': 0.0
            })
        else:
            return jsonify({'ok': False, 'error': 'Trainer not found in storage'}), 404
            
    except Exception as e:
        print(f"❌ Error resetting trainer storage average price: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/trainer/storage/tick', methods=['POST'])
def api_trainer_storage_tick():
    """트레이너별 저장 창고 틱 조작"""
    try:
        data = request.get_json(force=True)
        trainer = data.get('trainer')
        delta = int(data.get('delta', 0))  # +1 or -1
        
        if not trainer or trainer not in ['Scout', 'Guardian', 'Analyst', 'Elder']:
            return jsonify({'ok': False, 'error': 'Invalid trainer name'}), 400
        
        if trainer in _trainer_storage:
            # 틱 카운터 조작
            current_ticks = _trainer_storage[trainer].get('ticks', 0)
            new_ticks = max(0, current_ticks + delta)  # Prevent negative ticks
            _trainer_storage[trainer]['ticks'] = new_ticks
            _trainer_storage[trainer]['last_update'] = int(time.time())
            
            # 거래 기록에 추가
            _trainer_storage[trainer]['trades'].append({
                'timestamp': int(time.time()),
                'action': 'MANUAL_TICK',
                'delta': delta,
                'old_ticks': current_ticks,
                'new_ticks': new_ticks
            })
            
            # Save to file
            _save_trainer_storage()
            
            print(f"✅ Trainer storage tick modified: {trainer} {delta:+d} (new ticks: {new_ticks})")
            
            return jsonify({
                'ok': True, 
                'trainer': trainer,
                'delta': delta,
                'new_ticks': new_ticks
            })
        else:
            return jsonify({'ok': False, 'error': 'Trainer not found in storage'}), 404
            
    except Exception as e:
        print(f"❌ Error modifying trainer storage ticks: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/trust/config', methods=['GET', 'POST'])
def api_trust_config():
    """신뢰도 설정 조회 및 수정"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            ml_trust = float(data.get('ml_trust', 50.0))
            nb_trust = float(data.get('nb_trust', 50.0))
            
            # 값 범위 제한 (0-100)
            ml_trust = max(0.0, min(100.0, ml_trust))
            nb_trust = max(0.0, min(100.0, nb_trust))
            
            _trust_config['ml_trust'] = ml_trust
            _trust_config['nb_trust'] = nb_trust
            _trust_config['last_updated'] = int(time.time() * 1000)
            
            _save_trust_config()
            
            return jsonify({
                'ok': True,
                'ml_trust': ml_trust,
                'nb_trust': nb_trust,
                'last_updated': _trust_config['last_updated']
            })
        else:
            # GET: 현재 설정 반환
            return jsonify({
                'ok': True,
                'ml_trust': _trust_config['ml_trust'],
                'nb_trust': _trust_config['nb_trust'],
                'last_updated': _trust_config['last_updated']
            })
            
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/bot/status')
def api_bot_status():
    cfg = _resolve_config()
    # Log masked env keys on each status request for visibility
    try:
        log_env_keys()
    except Exception:
        pass
    # current N/B coin for this interval bucket
    try:
        b = _bucket_ts_interval(int(time.time()*1000), str(cfg.candle))
        coin = _nb_coin_store.get(_coin_key(str(cfg.candle), str(cfg.market), b))
    except Exception:
        coin = None
    return jsonify({
        'running': bot_ctrl['running'],
        'last_signal': bot_ctrl.get('last_signal', 'HOLD'),
        'last_order': bot_ctrl.get('last_order'),
        'coin': coin,
        'trainer_storage': _trainer_storage,  # 트레이너 저장 창고 정보 추가
        'config': {
            'paper': cfg.paper,
            'order_krw': cfg.order_krw,
            'pnl_ratio': float(getattr(cfg, 'pnl_ratio', 0.0)),
            'ema_fast': cfg.ema_fast,
            'ema_slow': cfg.ema_slow,
            'candle': cfg.candle,
            'market': cfg.market,
            'interval_sec': cfg.interval_sec,
            'has_keys': bool((_get_runtime_keys()[0] and _get_runtime_keys()[1]) or (_get_runtime_keys()[2] and _get_runtime_keys()[3]))
        }
    })


@app.route('/api/nb/coin', methods=['GET'])
def api_nb_coin():
    """Return current and recent N/B COINs (per-candle buckets)."""
    try:
        cfg = _resolve_config()
        iv = str(request.args.get('interval') or cfg.candle)
        market = str(request.args.get('market') or cfg.market)
        now_b = _bucket_ts_interval(int(time.time()*1000), iv)
        # collect recent N buckets
        try:
            n = int(request.args.get('n') or 50)
        except Exception:
            n = 50
        sec = _interval_to_sec(iv)
        buckets = [(now_b - i*sec) for i in range(max(1, n))]
        coins = []
        for b in buckets:
            c = _nb_coin_store.get(_coin_key(iv, market, b))
            if not c:
                c = _ensure_nb_coin(iv, market, int(b))
            coins.append(c)
        cur = _nb_coin_store.get(_coin_key(iv, market, now_b))
        return jsonify({'ok': True, 'current': cur, 'recent': coins})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/coins/summary', methods=['GET'])
def api_nb_coins_summary():
    try:
        cfg = _resolve_config()
        # total owned coins = sum of per-interval counters
        try:
            total_owned = int(sum(int(v) for v in _nb_coin_counter.values()))
        except Exception:
            total_owned = 0
        # price per coin from setting (order_krw), default 5100
        try:
            price_per_coin = int(getattr(cfg, 'order_krw', 5100))
        except Exception:
            price_per_coin = 5100
        # available KRW
        avail_krw = 0.0
        try:
            upbit = None
            if (not cfg.paper) and cfg.access_key and cfg.secret_key:
                upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
            if upbit:
                avail_krw = float(upbit.get_balance('KRW') or 0.0)
        except Exception:
            avail_krw = 0.0
        try:
            buyable = int(avail_krw // max(1, int(price_per_coin)))
        except Exception:
            buyable = 0
        return jsonify({'ok': True, 'total_owned': total_owned, 'price_per_coin': int(price_per_coin), 'krw': float(avail_krw), 'buyable_by_krw': int(buyable)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/npc/generate', methods=['POST'])
def api_npc_generate():
    """Generate N random NPC dialogue messages based on current narrative/state.
    Body: { n?: int, interval?: string }
    Writes unique messages to data/npc_messages.jsonl and returns the new ones.
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        try:
            n = max(1, min(50, int(payload.get('n', 10))))
        except Exception:
            n = 10
        try:
            iv = str(payload.get('interval')) if payload.get('interval') else (state.get('candle') or load_config().candle)
        except Exception:
            iv = state.get('candle') or load_config().candle
        # lightweight insight snapshot (avoid calling Flask handlers directly)
        cfg = _resolve_config()
        try:
            df = get_candles(cfg.market, iv, count=max(120, cfg.ema_slow + 5))
        except Exception:
            df = pd.DataFrame()
        try:
            window = int(load_nb_params().get('window', 50))
        except Exception:
            window = 50
        try:
            ins = _make_insight(df, window, cfg.ema_fast, cfg.ema_slow, iv, None) or {}
        except Exception:
            ins = {}
        zone = str(ins.get('zone') or '').upper() if ins else None
        # approximate slope per bar (bp) if possible
        slope = None
        try:
            closes = df['close'].astype(float).tail(max(20, min(120, window)))
            if len(closes) >= 5:
                import numpy as _np
                y = _np.log(closes.replace(0, _np.nan)).fillna(method='bfill').fillna(method='ffill').values
                x = _np.arange(len(y), dtype=float)
                b1 = _np.polyfit(x, y, 1)[0]
                slope = float(b1)  # per-bar log slope (approx bp/bar after scale)
        except Exception:
            slope = None
        flip = None  # optional: can be added later
        # templates
        personas = ['Analyst','Scout','Guardian','Elder']
        frames = [
            "{p}({iv}): {zone} with slope {s} bp/bar. Flip ETA: {f} bars.",
            "{p}({iv}): I favor {act} while momentum holds. {guard}",
            "{p}({iv}): Feasibility → BUY={can_buy} SELL={can_sell}. coin={coin} buyable={buy}",
            "{p}({iv}): If conditions soften, I will stand down and wait for better alignment."
        ]
        # feasibility snapshot
        coin = int(_nb_coin_counter.get(iv, 0))
        # buyable via KRW balance and order_krw(coin price)
        try:
            price_per_coin = int(getattr(cfg, 'order_krw', 5100))
        except Exception:
            price_per_coin = 5100
        avail_krw = 0.0
        try:
            upbit = None
            if (not cfg.paper) and cfg.access_key and cfg.secret_key:
                upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
            if upbit:
                avail_krw = float(upbit.get_balance('KRW') or 0.0)
        except Exception:
            avail_krw = 0.0
        try:
            buy = int(avail_krw // max(1, price_per_coin))
        except Exception:
            buy = 0
        can_buy = (buy > 0); can_sell = (coin > 0)
        guard = "Zone-side & cooldown OK"  # placeholder; detailed guards available elsewhere
        # If OpenAI key present or provider specified, generate via GPT-4o-mini first
        provider = str(payload.get('provider') or '').lower()
        openai_key = os.getenv('OPENAI_API_KEY')
        out = []
        if openai_key and (provider == 'openai' or os.getenv('NPC_PROVIDER','').lower()=='openai'):
            try:
                url = 'https://api.openai.com/v1/chat/completions'
                headers = { 'Authorization': f'Bearer {openai_key}', 'Content-Type': 'application/json' }
                sys = "You are an NPC villager speaking concise, context-aware trading lines in English. Keep each line short (<= 140 chars), natural, and grounded in the given signals."
                context = f"interval={iv}, zone={zone}, slope={slope}, flip={flip}, coin_count={coin}, buyable={buy}, can_buy={can_buy}, can_sell={can_sell}"
                # we will request one-by-one to enforce de-duplication and keep responses crisp
                tries = 0
                while len(out) < n and tries < n*3:
                    tries += 1
                    persona = random.choice(personas)
                    usr = f"As {persona} at {iv}, say ONE short line about: {context}. Include a clear intent (BUY/SELL/HOLD) only if feasible."
                    body = {
                        'model': 'gpt-4o-mini',
                        'messages': [
                            { 'role': 'system', 'content': sys },
                            { 'role': 'user', 'content': usr }
                        ],
                        'temperature': 0.7,
                        'max_tokens': 60
                    }
                    resp = requests.post(url, headers=headers, json=body, timeout=20)
                    if resp.status_code >= 400:
                        break
                    data = resp.json()
                    txt = (data.get('choices') or [{}])[0].get('message', {}).get('content') or ''
                    text = f"{persona}({iv}): {txt.strip()}"
                    msg = { 'ts': int(time.time()*1000), 'interval': iv, 'persona': persona, 'text': text }
                    if _npc_add(msg):
                        out.append(msg)
            except Exception:
                out = []
        # fallback: template generator
        out = []
        tries = 0
        while len(out) < n and tries < n*5:
            tries += 1
            p = random.choice(personas)
            act = 'BUY' if (zone=='BLUE') else ('SELL' if zone=='ORANGE' else 'HOLD')
            s = None if slope is None else (round(float(slope)*10000, 2))
            f = (flip if isinstance(flip, int) else '-')
            text = random.choice(frames).format(p=p, iv=iv, zone=(zone or '-'), s=(s if s is not None else '-'), f=f, act=act, guard=guard, can_buy=can_buy, can_sell=can_sell, coin=coin, buy=buy)
            msg = { 'ts': int(time.time()*1000), 'interval': iv, 'persona': p, 'text': text }
            if _npc_add(msg):
                out.append(msg)
        return jsonify({'ok': True, 'count': len(out), 'items': out})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def run():
    # Load saved trainer storage data
    global _trainer_storage
    try:
        saved_data = _load_trainer_storage()
        if saved_data:
            _trainer_storage.update(saved_data)
            print("✅ Trainer storage data loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load trainer storage data: {e}")
    
    # Load trust configuration
    global _trust_config
    try:
        saved_trust = _load_trust_config()
        if saved_trust:
            _trust_config.update(saved_trust)
            print(f"✅ Trust config loaded: ML={_trust_config['ml_trust']}%, N/B={_trust_config['nb_trust']}%")
    except Exception as e:
        print(f"⚠️ Failed to load trust config: {e}")
    
    threading.Thread(target=updater, daemon=True).start()
    threading.Thread(target=nb_auto_opt_loop, daemon=True).start()
    use_https = os.getenv("UI_HTTPS", "false").lower() == "true"
    ssl_ctx = 'adhoc' if use_https else None
    app.run(host="127.0.0.1", port=int(os.getenv("UI_PORT", "5057")), ssl_context=ssl_ctx)


if __name__ == "__main__":
    run()


