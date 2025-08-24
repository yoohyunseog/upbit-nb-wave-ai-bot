# ===== Central Hub - Game Engine =====
"""
Central Hub - Game Engine
마을 시스템을 통합 관리하는 게임 엔진

구성 요소:
- 주민 4명 (각각 다른 역할과 특성)
- 촌장 1명 (마을 관리자)
- N/B 길드 1개 (거래 및 전략)
- 비트코인 시장 1개 (실시간 시장 데이터)
"""

import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

class CentralHubGameEngine:
    """Central Hub - Game Engine"""
    
    def __init__(self):
        self.engine_status = "initializing"
        self.start_time = datetime.now()
        
        # 마을 기본 정보
        self.village_info = {
            "name": "8BIT 마을",
            "population": 5,  # 주민 4명 + 촌장 1명
            "founded": datetime.now().isoformat(),
            "energy": 100,
            "max_energy": 100,
            "reputation": 50,
            "economy_level": 1
        }
        
        # 주민 시스템 초기화
        self.residents = self._initialize_residents()
        
        # 촌장 시스템 초기화
        self.mayor = self._initialize_mayor()
        
        # N/B 길드 시스템 초기화
        self.nb_guild = self._initialize_nb_guild()
        
        # 비트코인 시장 시스템 초기화
        self.bitcoin_market = self._initialize_bitcoin_market()
        
        # 게임 상태
        self.game_state = {
            "current_time": datetime.now().isoformat(),
            "day_cycle": 1,
            "weather": "sunny",
            "events": []
        }
        
        # 시스템 로그
        self.system_log = []
        
        self.engine_status = "running"
        self._log("Central Hub - Game Engine 시작됨")
        
        # 백그라운드 업데이트 시작
        self._start_background_updates()
    
    def _initialize_residents(self) -> Dict:
        """주민 4명 초기화"""
        residents = {
            "resident_001": {
                "id": "resident_001",
                "name": "김철수",
                "age": 28,
                "role": "농부",
                "specialty": "농작물 재배",
                "hp": 100,
                "max_hp": 100,
                "stamina": 80,
                "max_stamina": 100,
                "skills": {
                    "farming": 8,
                    "trading": 3,
                    "mining": 2
                },
                "inventory": {
                    "wheat": 50,
                    "corn": 30,
                    "money": 1000
                },
                "location": "농장",
                "status": "working",
                "last_activity": datetime.now().isoformat(),
                "relationships": {
                    "mayor": 7,
                    "resident_002": 6,
                    "resident_003": 4,
                    "resident_004": 5
                }
            },
            "resident_002": {
                "id": "resident_002",
                "name": "이영희",
                "age": 32,
                "role": "상인",
                "specialty": "상거래",
                "hp": 90,
                "max_hp": 100,
                "stamina": 70,
                "max_stamina": 100,
                "skills": {
                    "farming": 2,
                    "trading": 9,
                    "mining": 1
                },
                "inventory": {
                    "tools": 10,
                    "clothes": 20,
                    "money": 5000
                },
                "location": "상점",
                "status": "trading",
                "last_activity": datetime.now().isoformat(),
                "relationships": {
                    "mayor": 8,
                    "resident_001": 6,
                    "resident_003": 7,
                    "resident_004": 6
                }
            },
            "resident_003": {
                "id": "resident_003",
                "name": "박민수",
                "age": 25,
                "role": "광부",
                "specialty": "광물 채굴",
                "hp": 95,
                "max_hp": 100,
                "stamina": 90,
                "max_stamina": 100,
                "skills": {
                    "farming": 1,
                    "trading": 2,
                    "mining": 9
                },
                "inventory": {
                    "iron_ore": 100,
                    "coal": 80,
                    "money": 2000
                },
                "location": "광산",
                "status": "mining",
                "last_activity": datetime.now().isoformat(),
                "relationships": {
                    "mayor": 6,
                    "resident_001": 4,
                    "resident_002": 7,
                    "resident_004": 8
                }
            },
            "resident_004": {
                "id": "resident_004",
                "name": "최지영",
                "age": 29,
                "role": "기술자",
                "specialty": "도구 제작",
                "hp": 85,
                "max_hp": 100,
                "stamina": 75,
                "max_stamina": 100,
                "skills": {
                    "farming": 3,
                    "trading": 4,
                    "mining": 3,
                    "crafting": 9
                },
                "inventory": {
                    "tools": 5,
                    "materials": 50,
                    "money": 3000
                },
                "location": "공방",
                "status": "crafting",
                "last_activity": datetime.now().isoformat(),
                "relationships": {
                    "mayor": 7,
                    "resident_001": 5,
                    "resident_002": 6,
                    "resident_003": 8
                }
            }
        }
        return residents
    
    def _initialize_mayor(self) -> Dict:
        """촌장 시스템 초기화"""
        mayor = {
            "id": "mayor",
            "name": "정촌장",
            "age": 45,
            "role": "촌장",
            "specialty": "마을 관리",
            "hp": 100,
            "max_hp": 100,
            "stamina": 100,
            "max_stamina": 100,
            "skills": {
                "leadership": 10,
                "diplomacy": 8,
                "management": 9
            },
            "inventory": {
                "village_funds": 10000,
                "documents": 20,
                "seal": 1
            },
            "location": "촌장실",
            "status": "managing",
            "last_activity": datetime.now().isoformat(),
            "authority_level": 10,
            "trust_level": 8,
            "policies": {
                "tax_rate": 0.05,
                "trade_fee": 0.02,
                "welfare_rate": 0.03
            },
            "relationships": {
                "resident_001": 7,
                "resident_002": 8,
                "resident_003": 6,
                "resident_004": 7
            }
        }
        return mayor
    
    def _initialize_nb_guild(self) -> Dict:
        """N/B 길드 시스템 초기화"""
        guild = {
            "id": "nb_guild",
            "name": "N/B 거래 길드",
            "type": "trading_guild",
            "founded": datetime.now().isoformat(),
            "members": {
                "mayor": {"role": "advisor", "join_date": datetime.now().isoformat()},
                "resident_002": {"role": "trader", "join_date": datetime.now().isoformat()},
                "resident_004": {"role": "supplier", "join_date": datetime.now().isoformat()}
            },
            "resources": {
                "guild_funds": 50000,
                "trading_volume": 0,
                "reputation": 75
            },
            "activities": {
                "daily_trades": 0,
                "weekly_profit": 0,
                "monthly_growth": 0
            },
            "strategies": {
                "current_strategy": "balanced",
                "risk_level": "medium",
                "target_markets": ["local", "regional"]
            },
            "location": "길드 회관",
            "status": "active",
            "last_activity": datetime.now().isoformat()
        }
        return guild
    
    def _initialize_bitcoin_market(self) -> Dict:
        """비트코인 시장 시스템 초기화"""
        market = {
            "id": "bitcoin_market",
            "name": "8BIT 비트코인 시장",
            "type": "cryptocurrency_market",
            "opened": datetime.now().isoformat(),
            "current_price": 50000000,  # 5천만원
            "price_history": [],
            "volume_24h": 0,
            "market_cap": 0,
            "participants": {
                "mayor": {"role": "regulator", "balance": 1000000},
                "resident_002": {"role": "trader", "balance": 500000},
                "nb_guild": {"role": "institution", "balance": 2000000}
            },
            "trading_pairs": {
                "BTC/KRW": {"price": 50000000, "volume": 0},
                "BTC/USD": {"price": 40000, "volume": 0}
            },
            "market_data": {
                "price_change_24h": 0,
                "price_change_7d": 0,
                "volatility": 0.05,
                "market_sentiment": "neutral"
            },
            "status": "open",
            "last_update": datetime.now().isoformat()
        }
        return market
    
    def _log(self, message: str):
        """시스템 로그 기록"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": "info"
        }
        self.system_log.append(log_entry)
        
        # 로그 크기 제한 (최근 1000개만 유지)
        if len(self.system_log) > 1000:
            self.system_log = self.system_log[-1000:]
    
    def _start_background_updates(self):
        """백그라운드 업데이트 시작"""
        def background_worker():
            while self.engine_status == "running":
                try:
                    self._update_game_state()
                    self._update_residents()
                    self._update_market()
                    time.sleep(5)  # 5초마다 업데이트
                except Exception as e:
                    self._log(f"백그라운드 업데이트 오류: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
        self._log("백그라운드 업데이트 시작됨")
    
    def _update_game_state(self):
        """게임 상태 업데이트"""
        self.game_state["current_time"] = datetime.now().isoformat()
        
        # 시간에 따른 변화
        current_hour = datetime.now().hour
        if 6 <= current_hour < 18:
            self.game_state["weather"] = "sunny"
        else:
            self.game_state["weather"] = "night"
        
        # 마을 에너지 회복
        if self.village_info["energy"] < self.village_info["max_energy"]:
            self.village_info["energy"] = min(
                self.village_info["max_energy"],
                self.village_info["energy"] + 1
            )
    
    def _update_residents(self):
        """주민 상태 업데이트"""
        for resident_id, resident in self.residents.items():
            # 체력 회복
            if resident["hp"] < resident["max_hp"]:
                resident["hp"] = min(
                    resident["max_hp"],
                    resident["hp"] + 2
                )
            
            # 스태미나 회복
            if resident["stamina"] < resident["max_stamina"]:
                resident["stamina"] = min(
                    resident["max_stamina"],
                    resident["stamina"] + 3
                )
            
            # 활동 업데이트
            resident["last_activity"] = datetime.now().isoformat()
    
    def _update_market(self):
        """시장 데이터 업데이트"""
        # 비트코인 가격 변동 (시뮬레이션)
        current_price = self.bitcoin_market["current_price"]
        change_percent = random.uniform(-0.02, 0.02)  # ±2% 변동
        new_price = current_price * (1 + change_percent)
        
        self.bitcoin_market["current_price"] = int(new_price)
        self.bitcoin_market["price_change_24h"] = change_percent * 100
        
        # 거래량 업데이트
        self.bitcoin_market["volume_24h"] += random.randint(0, 1000)
        
        # 가격 히스토리 업데이트
        price_entry = {
            "timestamp": datetime.now().isoformat(),
            "price": self.bitcoin_market["current_price"]
        }
        self.bitcoin_market["price_history"].append(price_entry)
        
        # 히스토리 크기 제한 (최근 1000개만 유지)
        if len(self.bitcoin_market["price_history"]) > 1000:
            self.bitcoin_market["price_history"] = self.bitcoin_market["price_history"][-1000:]
        
        self.bitcoin_market["last_update"] = datetime.now().isoformat()
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        return {
            "engine_status": self.engine_status,
            "uptime": str(datetime.now() - self.start_time),
            "village_info": self.village_info,
            "game_state": self.game_state,
            "active_residents": len([r for r in self.residents.values() if r["status"] != "sleeping"]),
            "market_status": self.bitcoin_market["status"],
            "guild_status": self.nb_guild["status"]
        }
    
    def get_residents_info(self) -> Dict:
        """주민 정보 조회"""
        return {
            "residents": self.residents,
            "mayor": self.mayor,
            "total_population": len(self.residents) + 1
        }
    
    def get_market_info(self) -> Dict:
        """시장 정보 조회"""
        return {
            "bitcoin_market": self.bitcoin_market,
            "nb_guild": self.nb_guild
        }
    
    def execute_resident_action(self, resident_id: str, action: str, target: Optional[str] = None) -> Dict:
        """주민 행동 실행"""
        if resident_id not in self.residents:
            return {"success": False, "error": "주민을 찾을 수 없습니다"}
        
        resident = self.residents[resident_id]
        
        # 행동에 따른 효과
        if action == "work":
            if resident["stamina"] >= 10:
                resident["stamina"] -= 10
                # 역할에 따른 보상
                if resident["role"] == "농부":
                    resident["inventory"]["wheat"] += 10
                elif resident["role"] == "상인":
                    resident["inventory"]["money"] += 100
                elif resident["role"] == "광부":
                    resident["inventory"]["iron_ore"] += 15
                elif resident["role"] == "기술자":
                    resident["inventory"]["tools"] += 1
                
                self._log(f"{resident['name']}이(가) 일했습니다")
                return {"success": True, "message": f"{resident['name']}이(가) 일했습니다"}
            else:
                return {"success": False, "error": "스태미나가 부족합니다"}
        
        elif action == "rest":
            resident["stamina"] = min(resident["max_stamina"], resident["stamina"] + 20)
            resident["hp"] = min(resident["max_hp"], resident["hp"] + 10)
            self._log(f"{resident['name']}이(가) 휴식을 취했습니다")
            return {"success": True, "message": f"{resident['name']}이(가) 휴식을 취했습니다"}
        
        elif action == "trade":
            if target and target in self.residents:
                # 간단한 거래 시뮬레이션
                resident["inventory"]["money"] += 50
                self._log(f"{resident['name']}이(가) {self.residents[target]['name']}과 거래했습니다")
                return {"success": True, "message": f"{resident['name']}이(가) 거래했습니다"}
        
        return {"success": False, "error": "알 수 없는 행동입니다"}
    
    def get_mayor_actions(self) -> List[Dict]:
        """촌장이 할 수 있는 행동 목록"""
        return [
            {"action": "collect_tax", "name": "세금 징수", "cost": 0},
            {"action": "improve_infrastructure", "name": "인프라 개선", "cost": 5000},
            {"action": "hold_meeting", "name": "주민 회의", "cost": 1000},
            {"action": "trade_negotiation", "name": "거래 협상", "cost": 2000}
        ]
    
    def execute_mayor_action(self, action: str) -> Dict:
        """촌장 행동 실행"""
        if action == "collect_tax":
            total_tax = 0
            for resident in self.residents.values():
                tax_amount = int(resident["inventory"]["money"] * self.mayor["policies"]["tax_rate"])
                resident["inventory"]["money"] -= tax_amount
                total_tax += tax_amount
            
            self.mayor["inventory"]["village_funds"] += total_tax
            self._log(f"촌장이 세금 {total_tax}원을 징수했습니다")
            return {"success": True, "message": f"세금 {total_tax}원을 징수했습니다"}
        
        elif action == "improve_infrastructure":
            if self.mayor["inventory"]["village_funds"] >= 5000:
                self.mayor["inventory"]["village_funds"] -= 5000
                self.village_info["reputation"] += 5
                self.village_info["economy_level"] += 1
                self._log("촌장이 인프라를 개선했습니다")
                return {"success": True, "message": "인프라가 개선되었습니다"}
            else:
                return {"success": False, "error": "마을 자금이 부족합니다"}
        
        return {"success": False, "error": "알 수 없는 행동입니다"}

# 전역 인스턴스 생성
central_hub_engine = CentralHubGameEngine()
