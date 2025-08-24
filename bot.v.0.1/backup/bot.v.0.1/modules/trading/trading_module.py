# ===== Trading Module - Python Backend =====

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyupbit
import json
from flask import jsonify

class TradingModule:
    """Trading Dashboard 모듈"""
    
    def __init__(self):
        self.current_timeframe = 'minute1'
        self.auto_rotation = False
        self.last_update = datetime.now()
        self.cached_data = {}
        
    def get_candles(self, market: str, candle: str, count: int = 200) -> pd.DataFrame:
        """PyUpbit에서 캔들 데이터 가져오기"""
        try:
            if candle.startswith("minute"):
                unit = int(candle.replace("minute", ""))
                data = pyupbit.get_ohlcv(ticker=market, interval=f"minute{unit}", count=count)
            else:
                data = pyupbit.get_ohlcv(ticker=market, interval=candle, count=count)
            
            if data is None or data.empty:
                raise RuntimeError("Failed to fetch OHLCV data")
            return data
        except Exception as e:
            print(f"Error fetching candles for {market} {candle}: {e}")
            # Fallback: 더미 데이터 반환
            dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
            dummy_data = pd.DataFrame({
                'open': [50000000] * count,
                'high': [51000000] * count,
                'low': [49000000] * count,
                'close': [50000000] * count,
                'volume': [100] * count
            }, index=dates)
            return dummy_data
    
    def calculate_nb_wave(self, df: pd.DataFrame) -> list:
        """N/B Wave 계산"""
        zones = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            open_price = row['open']
            close = row['close']
            high = row['high']
            low = row['low']
            volume = row['volume']
            
            # 가격 변화율
            price_change = (close - open_price) / open_price
            
            # 볼륨 가중 가격 변화
            volume_weighted_change = price_change * (volume / df['volume'].mean())
            
            # N/B 판단 (임시 로직) - 더 민감하게 설정
            if volume_weighted_change > 0.0005:  # 0.05% 이상 상승
                zone = 'ORANGE'
                strength = min(0.95, 0.5 + abs(volume_weighted_change) * 200)
            elif volume_weighted_change < -0.0005:  # 0.05% 이상 하락
                zone = 'BLUE'
                strength = min(0.95, 0.5 + abs(volume_weighted_change) * 200)
            else:
                # 중립 구간 - 이전 구간 유지하거나 랜덤하게 설정
                if zones:
                    zone = zones[-1]['zone']
                    strength = 0.4
                else:
                    import random
                    zone = 'ORANGE' if random.random() > 0.5 else 'BLUE'
                    strength = 0.4
            
            zones.append({
                'zone': zone,
                'strength': round(strength, 2),
                'price': float(close),
                'volume': float(volume),
                'change': round(price_change * 100, 2)
            })
        
        return zones
    
    def get_trading_data(self, timeframe: str = 'minute1') -> dict:
        """Trading 데이터 조회"""
        try:
            # 캐시된 데이터가 있고 최근이면 반환
            cache_key = f"trading_{timeframe}"
            if cache_key in self.cached_data:
                cached_time = self.cached_data[cache_key].get('timestamp')
                if cached_time and (datetime.now() - cached_time).seconds < 30:
                    return self.cached_data[cache_key]['data']
            
            # 새로운 데이터 수집
            df = self.get_candles("KRW-BTC", timeframe, 100)
            
            # 라벨 생성
            labels = [d.strftime('%H:%M') for d in df.index]
            
            # N/B Wave 계산
            zones = self.calculate_nb_wave(df)
            
            # 요약 정보
            summary = {
                'orange': sum(1 for z in zones if z['zone'] == 'ORANGE'),
                'blue': sum(1 for z in zones if z['zone'] == 'BLUE'),
                'current_price': float(df['close'].iloc[-1]),
                'price_change_24h': round(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100, 2)
            }
            
            result = {
                'timeframe': timeframe,
                'labels': labels,
                'zones': zones,
                'summary': summary,
                'last_update': datetime.now().isoformat()
            }
            
            # 캐시 업데이트
            self.cached_data[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            print(f"Error fetching trading data: {e}")
            return {
                'error': str(e),
                'timeframe': timeframe,
                'labels': [],
                'zones': [],
                'summary': {'orange': 0, 'blue': 0}
            }
    
    def get_price_chart_data(self, timeframe: str = 'minute1') -> dict:
        """가격 차트 데이터 조회"""
        try:
            df = self.get_candles("KRW-BTC", timeframe, 100)
            
            return {
                'timeframe': timeframe,
                'labels': [d.strftime('%H:%M') for d in df.index],
                'prices': df['close'].tolist(),
                'volumes': df['volume'].tolist(),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching price chart data: {e}")
            return {
                'error': str(e),
                'timeframe': timeframe,
                'labels': [],
                'prices': [],
                'volumes': []
            }
    
    def set_timeframe(self, timeframe: str):
        """타임프레임 설정"""
        self.current_timeframe = timeframe
        print(f"Trading timeframe set to: {timeframe}")
    
    def toggle_auto_rotation(self):
        """자동 순환 토글"""
        self.auto_rotation = not self.auto_rotation
        print(f"Auto rotation: {'ON' if self.auto_rotation else 'OFF'}")
        return self.auto_rotation
    
    def get_status(self) -> dict:
        """모듈 상태 조회"""
        return {
            'current_timeframe': self.current_timeframe,
            'auto_rotation': self.auto_rotation,
            'last_update': self.last_update.isoformat(),
            'cached_data_keys': list(self.cached_data.keys())
        }

# 전역 인스턴스
trading_module = TradingModule()
