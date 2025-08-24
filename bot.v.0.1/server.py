import os
import math
import threading
import time
from collections import deque
from dataclasses import asdict
from flask import Flask, jsonify, Response, request, send_from_directory, render_template
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
import wave
import struct
import pygame
import tempfile
import websocket
import hmac
import base64
from urllib.parse import urlencode
import jwt
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# ===== Rate Limit 관리 시스템 =====

class UpbitRateLimiter:
    """업비트 API Rate Limit 관리자"""
    
    def __init__(self):
        self.rate_limits = {
            'market': {'limit': 10, 'current': 0, 'last_reset': time.time()},
            'candle': {'limit': 10, 'current': 0, 'last_reset': time.time()},
            'trade': {'limit': 10, 'current': 0, 'last_reset': time.time()},
            'ticker': {'limit': 10, 'current': 0, 'last_reset': time.time()},
            'orderbook': {'limit': 10, 'current': 0, 'last_reset': time.time()},
            'default': {'limit': 30, 'current': 0, 'last_reset': time.time()},
            'order': {'limit': 8, 'current': 0, 'last_reset': time.time()},
            'order-cancel-all': {'limit': 1, 'current': 0, 'last_reset': time.time(), 'window': 2},
            'websocket-connect': {'limit': 5, 'current': 0, 'last_reset': time.time()},
            'websocket-message': {'limit': 5, 'current': 0, 'last_reset': time.time(), 'minute_limit': 100}
        }
        self.last_request_time = {}
        self.min_request_interval = 0.1  # 최소 100ms 간격
        
    def can_make_request(self, group='default'):
        """요청 가능 여부 확인"""
        current_time = time.time()
        limit_info = self.rate_limits.get(group, self.rate_limits['default'])
        
        # 1초마다 카운터 리셋
        if current_time - limit_info['last_reset'] >= 1:
            limit_info['current'] = 0
            limit_info['last_reset'] = current_time
        
        # 요청 수 제한 확인
        if limit_info['current'] >= limit_info['limit']:
            return False
        
        # 최소 요청 간격 확인
        last_request = self.last_request_time.get(group, 0)
        if current_time - last_request < self.min_request_interval:
            return False
        
        return True
    
    def record_request(self, group='default'):
        """요청 기록"""
        current_time = time.time()
        limit_info = self.rate_limits.get(group, self.rate_limits['default'])
        
        # 1초마다 카운터 리셋
        if current_time - limit_info['last_reset'] >= 1:
            limit_info['current'] = 0
            limit_info['last_reset'] = current_time
        
        limit_info['current'] += 1
        self.last_request_time[group] = current_time
    
    def wait_if_needed(self, group='default'):
        """필요시 대기"""
        while not self.can_make_request(group):
            time.sleep(0.1)
    
    def get_status(self):
        """Rate Limit 상태 반환"""
        return {
            group: {
                'limit': info['limit'],
                'current': info['current'],
                'remaining': info['limit'] - info['current'],
                'last_reset': info['last_reset']
            }
            for group, info in self.rate_limits.items()
        }

# 전역 Rate Limiter 인스턴스
rate_limiter = UpbitRateLimiter()

# ===== 백그라운드 프로세스 관리 시스템 =====

class BackgroundProcessManager:
    """백그라운드 프로세스 관리자"""
    
    def __init__(self):
        self.processes = {}
        self.is_running = False
        self.data_cache = {}
        self.event_listeners = {}
        self.last_update = datetime.now()
        
    def start_system(self):
        """백그라운드 시스템 시작"""
        if self.is_running:
            return
            
        self.is_running = True
        print("🔄 Background system started")
        
        # 모든 백그라운드 프로세스 시작
        self.start_trading_monitor()
        self.start_wallet_monitor()
        self.start_market_monitor()
        
        # 메인 모니터링 스레드 시작
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_system(self):
        """백그라운드 시스템 중지"""
        self.is_running = False
        print("⏹️ Background system stopped")
        
    def _monitor_loop(self):
        """메인 모니터링 루프"""
        while self.is_running:
            try:
                # 데이터 수집 및 캐시 업데이트
                self._collect_all_data()
                
                # 이벤트 발생
                self._emit_events()
                
                # 5초 대기
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ Error in monitor loop: {e}")
                time.sleep(10)
    
    def _collect_all_data(self):
        """모든 데이터 수집"""
        try:
            # Trading 데이터 수집
            trading_data = self._collect_trading_data()
            self.data_cache['trading'] = trading_data
            
            # Wallet 데이터 수집
            wallet_data = self._collect_wallet_data()
            self.data_cache['wallet'] = wallet_data
            
            # Market 데이터 수집
            market_data = self._collect_market_data()
            self.data_cache['market'] = market_data
            
            self.last_update = datetime.now()
            
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
    
    def _collect_trading_data(self):
        """Trading 데이터 수집"""
        try:
            # BTC 1분봉 데이터
            df = get_candles("KRW-BTC", "minute1", 100)
            
            # N/B Wave 계산
            zones = []
            for i in range(len(df)):
                row = df.iloc[i]
                open_price = row['open']
                close = row['close']
                high = row['high']
                low = row['low']
                volume = row['volume']
                
                price_change = (close - open_price) / open_price
                volume_weighted_change = price_change * (volume / df['volume'].mean())
                
                if volume_weighted_change > 0.0005:
                    zone = 'ORANGE'
                    strength = min(0.95, 0.5 + abs(volume_weighted_change) * 200)
                elif volume_weighted_change < -0.0005:
                    zone = 'BLUE'
                    strength = min(0.95, 0.5 + abs(volume_weighted_change) * 200)
                else:
                    zone = 'NEUTRAL'
                    strength = 0.4
                
                zones.append({
                    'zone': zone,
                    'strength': round(strength, 2),
                    'price': float(close),
                    'volume': float(volume),
                    'change': round(price_change * 100, 2),
                    'timestamp': df.index[i].isoformat()
                })
            
            return {
                'current_price': float(df['close'].iloc[-1]),
                'price_change_24h': round(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100, 2),
                'zones': zones,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error collecting trading data: {e}")
            return {'error': str(e)}
    
    def _collect_wallet_data(self):
        """Wallet 데이터 수집"""
        try:
            # Upbit API 키가 설정되어 있으면 잔고 조회
            # 실제 구현에서는 설정된 API 키 사용
            return {
                'total_balance': 0,
                'btc_balance': 0,
                'krw_balance': 0,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error collecting wallet data: {e}")
            return {'error': str(e)}
    
    def _collect_market_data(self):
        """Market 데이터 수집"""
        try:
            # 시장 전체 데이터
            return {
                'market_cap': 0,
                'volume_24h': 0,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error collecting market data: {e}")
            return {'error': str(e)}
    
    def _emit_events(self):
        """이벤트 발생"""
        events = []
        
        # Trading 이벤트
        if 'trading' in self.data_cache:
            trading_data = self.data_cache['trading']
            if 'current_price' in trading_data:
                events.append({
                    'type': 'trading:price:update',
                    'data': trading_data,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Wallet 이벤트
        if 'wallet' in self.data_cache:
            wallet_data = self.data_cache['wallet']
            events.append({
                'type': 'wallet:balance:update',
                'data': wallet_data,
                'timestamp': datetime.now().isoformat()
            })
        
        # 이벤트 리스너들에게 전송
        for event in events:
            self._notify_listeners(event)
    
    def _notify_listeners(self, event):
        """이벤트 리스너들에게 알림"""
        if event['type'] in self.event_listeners:
            for listener in self.event_listeners[event['type']]:
                try:
                    listener(event)
                except Exception as e:
                    print(f"❌ Error in event listener: {e}")
    
    def add_event_listener(self, event_type, callback):
        """이벤트 리스너 추가"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(callback)
    
    def get_cached_data(self, data_type):
        """캐시된 데이터 조회"""
        return self.data_cache.get(data_type, {})
    
    def get_system_status(self):
        """시스템 상태 조회"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat(),
            'active_processes': list(self.processes.keys()),
            'cache_keys': list(self.data_cache.keys())
        }
    
    def start_trading_monitor(self):
        """Trading 모니터 시작"""
        self.processes['trading'] = {
            'status': 'running',
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        print("📊 Trading monitor started")
    
    def start_wallet_monitor(self):
        """Wallet 모니터 시작"""
        self.processes['wallet'] = {
            'status': 'running',
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        print("💰 Wallet monitor started")
    
    def start_market_monitor(self):
        """Market 모니터 시작"""
        self.processes['market'] = {
            'status': 'running',
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }
        print("📈 Market monitor started")

# 전역 백그라운드 프로세스 매니저 인스턴스
background_manager = BackgroundProcessManager()

# ===== 모듈 Import =====
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Trading 모듈
from modules.trading.trading_module import trading_module
# Wallet 모듈
from modules.wallet.wallet_module import wallet_module
# Settings 모듈
from modules.settings.settings_module import settings_module
# Central Hub - Game Engine
import central_hub_game_engine
central_hub_engine = central_hub_game_engine.central_hub_engine

# ===== PyUpbit 유틸리티 함수 =====

def get_current_price(market: str) -> float:
    """현재가 조회 (Rate Limit 적용)"""
    try:
        # Rate Limit 확인 및 대기
        rate_limiter.wait_if_needed('ticker')
        
        price = pyupbit.get_current_price(market)
        
        # 요청 기록
        rate_limiter.record_request('ticker')
        
        return price
    except Exception as e:
        print(f"Error fetching current price for {market}: {e}")
        return None

def get_candles(market: str, candle: str, count: int = 200) -> pd.DataFrame:
    """PyUpbit에서 캔들 데이터 가져오기 (Rate Limit 적용)"""
    try:
        # Rate Limit 확인 및 대기
        rate_limiter.wait_if_needed('candle')
        
        if candle.startswith("minute"):
            unit = int(candle.replace("minute", ""))
            data = pyupbit.get_ohlcv(ticker=market, interval=f"minute{unit}", count=count)
        else:
            data = pyupbit.get_ohlcv(ticker=market, interval=candle, count=count)
        
        # 요청 기록
        rate_limiter.record_request('candle')
        
        if data is None or data.empty:
            raise RuntimeError("Failed to fetch OHLCV data")
        return data
    except Exception as e:
        print(f"Error fetching candles for {market} {candle}: {e}")
        # 더미 데이터 대신 유효 데이터가 올 때까지 대기/재시도
        # 환경변수 CANDLES_MAX_WAIT_SEC 가 0(기본)이면 무기한 대기, >0이면 해당 초 만큼만 대기 후 예외
        try:
            max_wait_sec = int(os.environ.get('CANDLES_MAX_WAIT_SEC', '0'))
        except Exception:
            max_wait_sec = 0
        start_ts = time.time()
        attempt = 0
        while True:
            try:
                attempt += 1
                # 소폭 대기 후 재시도 (지수 백오프 한도 5초)
                sleep_sec = min(5.0, 1.0 + (attempt * 0.5))
                time.sleep(sleep_sec)
                rate_limiter.wait_if_needed('candle')
                if candle.startswith("minute"):
                    unit = int(candle.replace("minute", ""))
                    data = pyupbit.get_ohlcv(ticker=market, interval=f"minute{unit}", count=count)
                else:
                    data = pyupbit.get_ohlcv(ticker=market, interval=candle, count=count)
                rate_limiter.record_request('candle')
                if data is not None and not data.empty:
                    print(f"✅ Candles fetched after retry (attempt={attempt}) for {market} {candle}")
                    return data
                else:
                    print(f"⏳ Waiting for valid candles... (attempt={attempt}) {market} {candle}")
            except Exception as re:
                print(f"⏳ Retry error fetching candles ({attempt}) for {market} {candle}: {re}")
            # 시간 제한 체크
            if max_wait_sec > 0 and (time.time() - start_ts) >= max_wait_sec:
                raise RuntimeError(f"Timed out waiting for candles: {market} {candle}")

# ===== 8BIT Trading System v0.1 =====

app = Flask(__name__)
CORS(app)

# pygame 초기화
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
print("🎵 Pygame mixer initialized")

def generate_beep_sound(frequency=800, duration=0.1, volume=0.5):
    """비프음 생성"""
    try:
        # 오디오 파라미터
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        
        # 사인파 생성
        samples = []
        for i in range(num_samples):
            sample = int(volume * 32767 * np.sin(2 * np.pi * frequency * i / sample_rate))
            samples.append(sample)
        
        # WAV 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(struct.pack('h' * len(samples), *samples))
            
            return temp_file.name
            
    except Exception as e:
        print(f"🎵 Error generating beep sound: {e}")
        return None

def play_sound_file(file_path):
    """오디오 파일 재생 (메모리 안전)"""
    try:
        # 이전 재생 중지
        pygame.mixer.music.stop()
        
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"🎵 Playing sound file: {file_path}")
        
        # 재생 완료 후 파일 삭제 (더 안전한 방식)
        def cleanup():
            time.sleep(0.3)  # 재생 완료 대기 시간 증가
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"🎵 Cleaned up: {file_path}")
            except Exception as e:
                print(f"🎵 Cleanup error: {e}")
        
        # 스레드 수 제한
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
        
    except Exception as e:
        print(f"🎵 Error playing sound: {e}")

def play_sound_by_type(sound_type, volume=0.3):
    """사운드 타입에 따른 스타크래프트 사운드 재생"""
    try:
        # 스타크래프트 사운드 파일 매핑
        sound_files = {
            'click': 'audio/bigbox-starcraft-sfx-master/Starcraft Set 1 by Emulga/Select.wav',
            'success': 'audio/bigbox-starcraft-sfx-master/Starcraft Set 1 by Emulga/Move.wav',
            'error': 'audio/bigbox-starcraft-sfx-master/Starcraft Set 1 by Emulga/Back.wav',
            'type': 'audio/bigbox-starcraft-sfx-master/Starcraft Set 1 by Emulga/Startup.wav'
        }
        
        sound_file_path = sound_files.get(sound_type)
        if not sound_file_path:
            print(f"🎵 Unknown sound type: {sound_type}")
            return False
        
        print(f"🎵 Playing {sound_type} sound: {sound_file_path} (volume: {volume})")
        
        # 스타크래프트 사운드 파일 재생
        play_starcraft_sound(sound_file_path, volume)
        return True
            
    except Exception as e:
        print(f"🎵 Error in play_sound_by_type: {e}")
        return False

def play_starcraft_sound(file_path, volume=0.3):
    """스타크래프트 사운드 파일 재생 (메모리 안전)"""
    try:
        # 이전 재생 중지
        pygame.mixer.music.stop()
        
        # 볼륨 설정
        pygame.mixer.music.set_volume(volume)
        
        # 사운드 파일 로드 및 재생
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        print(f"🎵 Playing Starcraft sound: {file_path}")
        
    except Exception as e:
        print(f"🎵 Error playing Starcraft sound: {e}")
        # 실패 시 비프음으로 fallback
        fallback_beep_sound('click', volume)

def fallback_beep_sound(sound_type, volume=0.3):
    """비프음 fallback"""
    try:
        frequencies = {
            'click': 800,
            'success': 1000,
            'error': 400,
            'type': 600
        }
        
        frequency = frequencies.get(sound_type, 800)
        duration = 0.1
        
        print(f"🎵 Fallback: Generating {sound_type} beep sound (freq: {frequency}Hz, volume: {volume})")
        
        # 비프음 파일 생성
        sound_file = generate_beep_sound(frequency, duration, volume)
        
        if sound_file:
            # 별도 스레드에서 재생
            threading.Thread(target=play_sound_file, args=(sound_file,), daemon=True).start()
            
    except Exception as e:
        print(f"🎵 Error in fallback_beep_sound: {e}")

# 간단한 사운드 API (실제 오디오 재생)
# 사운드 API Rate Limit 관리
sound_rate_limiter = {
    'last_call': 0,
    'min_interval': 0.1  # 최소 100ms 간격
}

@app.route('/api/play-sound', methods=['POST'])
def api_play_sound():
    """사운드 재생 API (Rate Limit 적용)"""
    try:
        # Rate Limit 확인
        current_time = time.time()
        if current_time - sound_rate_limiter['last_call'] < sound_rate_limiter['min_interval']:
            return jsonify({
                'success': False,
                'message': 'Rate limit exceeded - too many sound requests'
            }), 429
        
        sound_rate_limiter['last_call'] = current_time
        
        data = request.get_json()
        sound_type = data.get('type', 'click')
        volume = data.get('volume', 0.3)  # 기본 볼륨 0.3, 클라이언트에서 전달받은 볼륨 사용
        
        print(f"🎵 Sound API called: {sound_type} (volume: {volume})")
        
        # 실제 오디오 재생 (볼륨 적용)
        success = play_sound_by_type(sound_type, volume)
        
        return jsonify({
            'success': success,
            'message': f'Sound {sound_type} {"played" if success else "failed"} (volume: {volume})',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"🎵 Sound API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-upbit', methods=['POST'])
def api_test_upbit():
    """업비트 API 연결 테스트 (설정값 사용)"""
    try:
        upbit_settings = settings_module.get_settings('upbit')
        access_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')

        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Upbit API keys are not set in settings (upbitAccessKey/upbitSecretKey).'
            }), 400

        print(f"🔑 Testing Upbit API connection (settings)...")

        try:
            upbit = pyupbit.Upbit(access_key, secret_key)
            balance = upbit.get_balance("KRW")
            if balance is not None:
                return jsonify({
                    'success': True,
                    'balance': f"{balance:,.0f}",
                    'message': 'Upbit API connection successful'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Balance check failed - Please verify your API keys in settings.'
                })
        except Exception as api_error:
            print(f"🔑 Upbit API error: {api_error}")
            return jsonify({
                'success': False,
                'error': f'API connection failed: {str(api_error)}'
            })
    except Exception as e:
        print(f"🔑 Test Upbit API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upbit-balance', methods=['POST'])
def api_upbit_balance():
    """Get Upbit account balance (설정값 사용)"""
    try:
        upbit_settings = settings_module.get_settings('upbit')
        access_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')

        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Upbit API keys are not set in settings (upbitAccessKey/upbitSecretKey).'
            }), 400

        print(f"💰 Fetching Upbit balance (settings)...")

        try:
            upbit = pyupbit.Upbit(access_key, secret_key)
            balances = upbit.get_balances()
            if not balances:
                return jsonify({'success': True, 'balances': [], 'total_portfolio_value': 0})

            processed_balances = []
            total_portfolio_value = 0

            for balance in balances:
                currency = balance['currency']
                balance_amount = float(balance['balance'])
                locked_amount = float(balance['locked'])
                if balance_amount == 0 and locked_amount == 0:
                    continue

                current_price = 0
                avg_buy_price = 0
                asset_value = 0

                if currency == 'KRW':
                    current_price = 1
                    avg_buy_price = 1
                    asset_value = balance_amount
                else:
                    try:
                        rate_limiter.wait_if_needed('ticker')
                        ticker = get_current_price(f"KRW-{currency}")
                        if ticker and ticker > 0:
                            current_price = ticker
                            asset_value = balance_amount * current_price
                        else:
                            current_price = 0
                            asset_value = 0
                        avg_buy_price = float(balance.get('avg_buy_price', 0))
                        if avg_buy_price == 0 and current_price > 0:
                            avg_buy_price = current_price
                    except Exception as price_error:
                        if "Code not found" not in str(price_error):
                            print(f"⚠️ Price fetch error for {currency}: {price_error}")
                        current_price = 0
                        avg_buy_price = 0
                        asset_value = 0

                total_portfolio_value += asset_value
                processed_balances.append({
                    'currency': currency,
                    'balance': balance_amount,
                    'locked': locked_amount,
                    'avg_buy_price': avg_buy_price,
                    'price': current_price,
                    'asset_value': asset_value
                })

            return jsonify({
                'success': True,
                'balances': processed_balances,
                'total_portfolio_value': total_portfolio_value
            })
        except Exception as api_error:
            print(f"💰 Upbit balance API error: {api_error}")
            return jsonify({'success': False, 'error': f'Balance fetch failed: {str(api_error)}'})
    except Exception as e:
        print(f"💰 Upbit balance error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 오디오 테스트 API
@app.route('/api/test-audio', methods=['GET'])
def api_test_audio():
    """오디오 시스템 테스트 API"""
    try:
        # 간단한 테스트 사운드 재생
        # audio_manager.play_sound_async('click') # Removed audio_manager
        
        return jsonify({
            'success': True,
            'message': 'Test sound played',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 오디오 상태 확인 API
@app.route('/api/audio-status', methods=['GET'])
def api_audio_status():
    """오디오 시스템 상태 확인"""
    try:
        # status = audio_manager.get_status() # Removed audio_manager
        status = {
            'audio_enabled': False,
            'pygame_initialized': False,
            'pygame_info': {},
            'is_initialized': False,
            'error': "Audio system disabled",
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'audio_enabled': False,
            'pygame_initialized': False,
            'pygame_info': {},
            'is_initialized': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

# 오디오 설정 API
@app.route('/api/audio-settings', methods=['GET', 'POST'])
def api_audio_settings():
    """오디오 설정 API"""
    if request.method == 'POST':
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        # audio_manager.enable_audio() # Removed audio_manager
        # audio_manager.disable_audio() # Removed audio_manager
        
        return jsonify({
            'success': True,
            'enabled': False, # Always disabled
            'message': 'Audio system disabled'
        })
    
    return jsonify({
        'enabled': False, # Always disabled
        'available_sounds': []
    })

# 페이지 로딩 감지 API
@app.route('/api/page-load', methods=['POST'])
def api_page_load():
    """페이지 로딩 감지 API"""
    try:
        data = request.get_json()
        page_type = data.get('page', 'unknown')
        
        # 페이지 로딩 정보 업데이트
        # page_monitor.last_page_load = datetime.now() # Removed page_monitor
        # page_monitor.page_load_count += 1 # Removed page_monitor
        
        print(f"🎵 Page load detected: {page_type}")
        
        # 페이지 로딩 시 자동 사운드 재생
        # if page_monitor.page_load_count == 1: # Removed page_monitor
        #     # 첫 번째 로딩
        #     print("🎵 First page load - playing welcome sound sequence...")
        #     audio_manager.play_sound_async('success')
        #     time.sleep(0.5)
        #     audio_manager.play_sound_async('click')
        #     time.sleep(0.5)
        #     audio_manager.play_sound_async('type')
        # else:
        #     # 일반 로딩
        #     print("🎵 Page reload - playing notification sound...")
        #     audio_manager.play_sound_async('click')
        
        return jsonify({
            'success': True,
            'message': f'Page load detected: {page_type}',
            # 'load_count': page_monitor.page_load_count, # Removed page_monitor
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 페이지 상태 확인 API
@app.route('/api/page-status', methods=['GET'])
def api_page_status():
    """페이지 상태 확인 API"""
    try:
        status = {
            # 'monitoring': page_monitor.monitoring, # Removed page_monitor
            # 'load_count': page_monitor.page_load_count, # Removed page_monitor
            # 'last_load': page_monitor.last_page_load.isoformat() if page_monitor.last_page_load else None, # Removed page_monitor
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WebSocket 관련 변수들
websocket_connection = None
order_history = []
order_history_lock = threading.Lock()

def get_upbit_websocket_token(access_key, secret_key):
    """Upbit WebSocket 인증 토큰 생성"""
    try:
        print(f"🔐 WebSocket 토큰 생성 시작...")
        print(f"🔐 Access Key: {access_key[:10]}...")
        print(f"🔐 Secret Key: {secret_key[:10]}...")
        
        # Upbit WebSocket API JWT 토큰 생성
        payload = {
            "access_key": access_key,
            "nonce": str(uuid.uuid4())
        }
        
        jwt_token = jwt.encode(payload, secret_key, algorithm="HS256")
        print(f"🔐 JWT 토큰 생성 성공: {jwt_token[:50]}...")
        return jwt_token
    except Exception as e:
        print(f"🔐 WebSocket 토큰 생성 실패: {e}")
        print(f"🔐 에러 타입: {type(e)}")
        return None

def on_websocket_message(ws, message):
    """WebSocket 메시지 수신 처리"""
    try:
        data = json.loads(message)
        
        if data.get('type') == 'myOrder':
            with order_history_lock:
                # 주문 데이터를 히스토리에 추가
                order_data = {
                    'timestamp': datetime.now().isoformat(),
                    'order_data': data
                }
                order_history.append(order_data)
                
                # 최근 100개만 유지
                if len(order_history) > 100:
                    order_history.pop(0)
                
                print(f"📊 주문 데이터 수신: {data.get('code')} - {data.get('state')}")
                
    except Exception as e:
        print(f"⚠️ WebSocket 메시지 처리 오류: {e}")

def on_websocket_error(ws, error):
    """WebSocket 에러 처리"""
    print(f"❌ WebSocket 에러: {error}")

def on_websocket_close(ws, close_status_code, close_msg):
    """WebSocket 연결 종료 처리"""
    print(f"🔌 WebSocket 연결 종료: {close_status_code} - {close_msg}")

def on_websocket_open(ws):
    """WebSocket 연결 성공 처리"""
    print("🔗 WebSocket 연결 성공")
    
    # 주문 데이터 구독 요청 (실시간 이벤트만)
    subscribe_message = [
        {"ticket": "my-order-" + str(int(time.time()))},
        {
            "type": "myOrder",
            "codes": ["KRW-BTC"]  # BTC 마켓만 구독 (테스트용)
        }
    ]
    
    ws.send(json.dumps(subscribe_message))
    print("📡 실시간 주문 이벤트 구독 요청 전송")
    print("📡 참고: 연결 후 주문/체결이 발생해야 데이터가 수신됩니다")

def start_websocket_connection(access_key, secret_key):
    """WebSocket 연결 시작"""
    global websocket_connection
    
    try:
        # 기존 연결이 있으면 종료
        if websocket_connection:
            websocket_connection.close()
        
        # JWT 토큰 생성
        jwt_token = get_upbit_websocket_token(access_key, secret_key)
        if not jwt_token:
            return False
        
        # WebSocket 연결
        websocket_url = "wss://api.upbit.com/websocket/v1"
        headers = ["Authorization: Bearer " + jwt_token]
        
        websocket_connection = websocket.WebSocketApp(
            websocket_url,
            header=headers,
            on_open=on_websocket_open,
            on_message=on_websocket_message,
            on_error=on_websocket_error,
            on_close=on_websocket_close
        )
        
        # 별도 스레드에서 WebSocket 실행
        wst = threading.Thread(target=websocket_connection.run_forever)
        wst.daemon = True
        wst.start()
        
        print("🚀 WebSocket 연결 시작됨")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket 연결 실패: {e}")
        return False

@app.route('/api/start-websocket', methods=['POST'])
def api_start_websocket():
    """WebSocket 연결 시작 API (설정값 사용)"""
    try:
        upbit_settings = settings_module.get_settings('upbit')
        access_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')
        if not access_key or not secret_key:
            return jsonify({'success': False, 'error': 'Upbit API keys are not set in settings.'}), 400

        success = start_websocket_connection(access_key, secret_key)
        return jsonify({'success': success, 'message': 'WebSocket connection started successfully.' if success else 'Failed to start WebSocket connection.'})
    except Exception as e:
        print(f"❌ WebSocket 시작 API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/order-history', methods=['POST'])
def api_order_history():
    """주문 히스토리 조회 API (Upbit REST API 사용, 설정값)"""
    try:
        upbit_settings = settings_module.get_settings('upbit')
        access_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')
        if not access_key or not secret_key:
            return jsonify({'success': False, 'error': 'Upbit API keys are not set in settings.'}), 400

        upbit = pyupbit.Upbit(access_key, secret_key)
        
        # 최근 주문 내역 조회 (완료된 주문들)
        orders = upbit.get_orders(state='done', limit=50)
        
        if not orders:
            return jsonify({
                'success': True,
                'orders': []
            })
        
        # 주문 데이터를 클라이언트 형식으로 변환
        formatted_orders = []
        
        for order in orders:
            # 주문 상태에 따른 표시 텍스트
            state_text = {
                'wait': 'PENDING',
                'watch': 'WATCHING',
                'trade': 'TRADING',
                'done': 'COMPLETED',
                'cancel': 'CANCELLED',
                'prevented': 'PREVENTED'
            }.get(order.get('state', ''), order.get('state', '').upper())
            
            # 매수/매도 구분
            order_type = 'BUY' if order.get('side') == 'bid' else 'SELL'
            
            # 가격 정보
            price = float(order.get('price', 0))
            volume = float(order.get('volume', 0))
            executed_volume = float(order.get('executed_volume', 0))
            executed_funds = float(order.get('executed_funds', 0))
            
            # 타임스탬프 변환
            created_at = order.get('created_at', '')
            if created_at:
                # ISO 형식의 문자열을 파싱
                try:
                    order_time = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    order_time = created_at
            else:
                order_time = 'Unknown'
            
            formatted_orders.append({
                'type': order_type,
                'state': state_text,
                'code': order.get('market', ''),
                'price': price,
                'volume': volume,
                'executed_volume': executed_volume,
                'executed_funds': executed_funds,
                'time': order_time,
                'uuid': order.get('uuid', '')
            })
        
        # 최신 주문부터 정렬
        formatted_orders.sort(key=lambda x: x['time'], reverse=True)
        
        return jsonify({
            'success': True,
            'orders': formatted_orders
        })
        
    except Exception as e:
        print(f"❌ 주문 히스토리 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===== 게임 시스템 변수 =====
GAME_STATE = {
    "minerals": 1000,
    "gas": 500,
    "supply": 8,
    "max_supply": 10,
    "current_module": None,
    "system_status": "Online",
    "connection_status": "Stable"
}

# ===== 마을 시스템 (기존 시스템 유지) =====
VILLAGE_ENERGY = 150
MAX_VILLAGE_ENERGY = 100
ENERGY_ACCUMULATED = 150

# 촌장의 신뢰도 시스템
MAYOR_TRUST_SYSTEM = {
    "ML_Model_Trust": 40,
    "NB_Guild_Trust": 82,
    "last_guidance": None,
    "guidance_history": [],
    "auto_learning_enabled": True,
    "last_learning_time": None,
    "learning_interval": 3600
}

# 마을 출입 일지 시스템
VILLAGE_ENTRY_EXIT_LOG = {
    "total_residents": 10,
    "current_in_village": 4,
    "current_in_orange": 3,
    "current_in_blue": 3,
    "zone_logs": {
        "ORANGE": {
            "residents": [],
            "activities": [],
            "entry_exit_log": []
        },
        "BLUE": {
            "residents": [],
            "activities": [],
            "entry_exit_log": []
        },
        "VILLAGE": {
            "residents": [],
            "activities": [],
            "entry_exit_log": []
        }
    },
    "resident_status": {}
}

# 마을 주민 시스템
VILLAGE_RESIDENTS = {
    "scout": {
        "name": "Scout",
        "hp": 85,
        "maxHp": 100,
        "stamina": 70,
        "maxStamina": 100,
        "location": "Gate",
        "role": "Explorer",
        "assignedTimeframes": ["minute1", "minute3"],
        "specialty": "Quick Signals",
        "description": "Monitors 1m & 3m charts for rapid opportunities",
        "skillLevel": 2.9,
        "experience": 0,
        "learningRate": 0.1,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.6,
        "strategy": "momentum"
    },
    "marine": {
        "name": "Marine",
        "hp": 100,
        "maxHp": 100,
        "stamina": 90,
        "maxStamina": 100,
        "location": "Barracks",
        "role": "Defender",
        "assignedTimeframes": ["minute5", "minute10"],
        "specialty": "Trend Analysis",
        "description": "Analyzes 5m & 10m trends for strategic positions",
        "skillLevel": 3.2,
        "experience": 0,
        "learningRate": 0.08,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.4,
        "strategy": "trend"
    },
    "medic": {
        "name": "Medic",
        "hp": 75,
        "maxHp": 100,
        "stamina": 85,
        "maxStamina": 100,
        "location": "Academy",
        "role": "Support",
        "assignedTimeframes": ["minute15", "minute30"],
        "specialty": "Risk Management",
        "description": "Manages risk on 15m & 30m timeframes",
        "skillLevel": 2.8,
        "experience": 0,
        "learningRate": 0.12,
        "autoTradingEnabled": True,
        "lastAutoTrade": None,
        "tradeFrequency": 0.3,
        "strategy": "conservative"
    }
}

# ===== 라우트 정의 =====

@app.route('/')
def index():
    """메인 게임 UI 페이지"""
    return send_from_directory('.', 'index.html')

@app.route('/ui')
def ui():
    """기존 /ui 경로 호환성"""
    return send_from_directory('.', 'index.html')

@app.route('/village')
def village_ui():
    """마을 UI - Central Hub Game Engine"""
    return send_from_directory('.', 'village-ui.html')

@app.route('/style.css')
def style():
    """CSS 파일 서빙"""
    return send_from_directory('.', 'style.css')

@app.route('/trainer-decision-log')
def trainer_decision_log_page():
    """Trainer decision log viewer page"""
    return send_from_directory('.', 'trainer-decision-log.html')

@app.route('/trainer-activity-log')
def trainer_activity_log_page():
    """Trainer activity log viewer page"""
    return send_from_directory('.', 'trainer-activity-log.html')

@app.route('/app.js')
def app_js():
    """JavaScript 파일 서빙"""
    return send_from_directory('.', 'app.js')

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """오디오 파일 서빙"""
    return send_from_directory('audio', filename)

@app.route('/api/game-state')
def get_game_state():
    """게임 상태 조회"""
    return jsonify(GAME_STATE)

@app.route('/api/update-resources', methods=['POST'])
def update_resources():
    """자원 업데이트"""
    data = request.json
    if 'minerals' in data:
        GAME_STATE['minerals'] = data['minerals']
    if 'gas' in data:
        GAME_STATE['gas'] = data['gas']
    if 'supply' in data:
        GAME_STATE['supply'] = data['supply']
    return jsonify({"status": "success", "game_state": GAME_STATE})



@app.route('/api/residents')
def get_residents():
    """주민 정보 조회"""
    return jsonify(central_hub_engine.get_residents_info())

@app.route('/api/update-resident', methods=['POST'])
def update_resident():
    """주민 정보 업데이트"""
    data = request.json
    resident_id = data.get('id')
    action = data.get('action', 'work')
    target = data.get('target')
    
    result = central_hub_engine.execute_resident_action(resident_id, action, target)
    return jsonify(result)

@app.route('/api/mayor-actions')
def get_mayor_actions():
    """촌장 행동 목록 조회"""
    actions = central_hub_engine.get_mayor_actions()
    return jsonify({"actions": actions})

@app.route('/api/execute-mayor-action', methods=['POST'])
def execute_mayor_action():
    """촌장 행동 실행"""
    data = request.json
    action = data.get('action')
    
    if not action:
        return jsonify({"success": False, "error": "Action is required"}), 400
    
    result = central_hub_engine.execute_mayor_action(action)
    return jsonify(result)

@app.route('/api/market-info')
def get_market_info():
    """시장 정보 조회"""
    return jsonify(central_hub_engine.get_market_info())

@app.route('/api/village-status')
def get_village_status():
    """마을 상태 조회"""
    return jsonify({
        "village_info": central_hub_engine.village_info,
        "game_state": central_hub_engine.game_state,
        "system_log": central_hub_engine.system_log[-10:]  # 최근 10개 로그만
    })

@app.route('/api/trading-data')
def get_legacy_trading_data():
    """실제 업비트 데이터 기반 트레이딩 정보"""
    try:
        # URL 파라미터에서 시간대와 코인 가져오기
        timeframe = request.args.get('timeframe', 'minute1')
        coin = request.args.get('coin', 'BTC')
        
        # 마켓 코드 생성
        market = f"KRW-{coin}"
        print(f"🪙 Trading data requested for {market}")
        
        # auto 옵션 처리
        if timeframe == 'auto':
            # 모든 시간대의 데이터를 순차적으로 가져오기
            timeframes = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'minute240', 'day', 'week', 'month']
            all_data = {}
            
            for tf in timeframes:
                try:
                    # 현재 가격
                    current_price_raw = get_current_price(market)
                    current_price = float(current_price_raw) if current_price_raw is not None else 0
                    
                    # 선택된 시간대 데이터 (최근 60개)
                    df = get_candles(market, tf, count=60)
                    
                    # 데이터 유효성 검사
                    if df is None or len(df) < 2:
                        print(f"❌ {tf} 데이터가 유효하지 않습니다")
                        continue
                    
                    # 가격 변화율 계산 (JSON 직렬화 가능하도록 변환)
                    price_change = float(((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100) if len(df) > 1 else 0.0
                    
                    # 거래량 (JSON 직렬화 가능하도록 변환)
                    volume_24h = float(df['volume'].sum())
                    
                    # 시그널 생성
                    signals = []
                    if price_change > 0.5:
                        signals.append({"type": "buy", "strength": min(0.9, abs(price_change) / 2), "timeframe": tf})
                    elif price_change < -0.5:
                        signals.append({"type": "sell", "strength": min(0.9, abs(price_change) / 2), "timeframe": tf})
                    
                    # 차트 데이터 (최근 20개 포인트)
                    chart_labels = [ts.strftime('%H:%M') for ts in df.index[-20:]]
                    chart_prices = [float(price) for price in df['close'].iloc[-20:]]
                    
                    all_data[tf] = {
                        "current_price": current_price,
                        "price_change": round(price_change, 2),
                        "volume": volume_24h,
                        "signals": signals,
                        "timeframe": tf,
                        "chart_data": {
                            "labels": chart_labels,
                            "prices": chart_prices
                        }
                    }
                except Exception as e:
                    print(f"❌ {tf} 데이터 조회 실패: {e}")
                    continue
            
            return jsonify({
                "success": True,
                "mode": "auto",
                "timeframes": all_data
            })
        else:
            # 단일 시간대 데이터
            # 현재 가격
            current_price_raw = get_current_price(market)
            current_price = float(current_price_raw) if current_price_raw is not None else 0
            
            # 선택된 시간대 데이터 (최근 60개)
            df = get_candles(market, timeframe, count=60)
            
            # 데이터 유효성 검사
            if df is None or len(df) < 2:
                raise Exception(f"Invalid data for timeframe {timeframe}")
            
            # 가격 변화율 계산 (JSON 직렬화 가능하도록 변환)
            price_change = float(((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100) if len(df) > 1 else 0.0
            
            # 거래량 (JSON 직렬화 가능하도록 변환)
            volume_24h = float(df['volume'].sum())
            
            # 시그널 생성 (간단한 예시)
            signals = []
            if price_change > 0.5:
                signals.append({"type": "buy", "strength": min(0.9, abs(price_change) / 2), "timeframe": timeframe})
            elif price_change < -0.5:
                signals.append({"type": "sell", "strength": min(0.9, abs(price_change) / 2), "timeframe": timeframe})
            
            # 차트 데이터 (최근 20개 포인트)
            chart_labels = [ts.strftime('%H:%M') for ts in df.index[-20:]]
            chart_prices = [float(price) for price in df['close'].iloc[-20:]]
            
            return jsonify({
                "success": True,
                "mode": "single",
                "current_price": current_price,
                "price_change": round(price_change, 2),
                "volume": volume_24h,
                "signals": signals,
                "timeframe": timeframe,
                "chart_data": {
                    "labels": chart_labels,
                    "prices": chart_prices
                }
            })
        
    except Exception as e:
        print(f"Error fetching trading data: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "current_price": 0,
            "price_change": 0,
            "volume": 0,
            "signals": [],
            "timeframe": timeframe,
            "chart_data": {"labels": [], "prices": []}
        }), 500

@app.route('/api/nb-wave')
def get_nb_wave():
    """실제 업비트 데이터 기반 N/B Wave 계산"""
    timeframe = request.args.get('timeframe', 'minute1')
    coin = request.args.get('coin', 'BTC')
    try:
        bars = int(request.args.get('bars', 120))
    except Exception:
        bars = 120
    
    # 마켓 코드 생성
    market = f"KRW-{coin}"
    print(f"🪙 NB Wave data requested for {market}")

    try:
        # auto 옵션 처리
        if timeframe == 'auto':
            # 모든 시간대의 NB Wave 데이터를 순차적으로 가져오기
            timeframes = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'minute240', 'day', 'week', 'month']
            all_data = {}
            
            for tf in timeframes:
                try:
                    # 실제 업비트 데이터 가져오기
                    df = get_candles(market, tf, count=bars)
                    
                    # N/B Wave 계산
                    zones = []
                    labels = []
                    
                    for i, (timestamp, row) in enumerate(df.iterrows()):
                        # 시간 라벨
                        labels.append(timestamp.strftime('%H:%M'))
                        
                        # 간단한 N/B 판단 (실제 로직으로 교체 필요)
                        close = row['close']
                        open_price = row['open']
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
                                zone = 'ORANGE' if random.random() > 0.5 else 'BLUE'
                                strength = 0.4
                        
                        zones.append({
                            'zone': zone,
                            'strength': round(strength, 2),
                            'price': float(close),
                            'volume': float(volume),
                            'change': round(price_change * 100, 2)
                        })
                    
                    # 통계 계산
                    orange_count = sum(1 for z in zones if z['zone'] == 'ORANGE')
                    blue_count = sum(1 for z in zones if z['zone'] == 'BLUE')
                    neutral_count = sum(1 for z in zones if z['zone'] == 'NEUTRAL')
                    
                    all_data[tf] = {
                        'timeframe': tf,
                        'zones': zones,
                        'labels': labels,
                        'summary': {
                            'orange': orange_count,
                            'blue': blue_count,
                            'neutral': neutral_count,
                            'current_price': float(df['close'].iloc[-1]) if len(df) > 0 else 0
                        },
                        'last_update': datetime.now().isoformat()
                    }
                except Exception as e:
                    print(f"❌ {tf} NB Wave 데이터 조회 실패: {e}")
                    continue
            
            return jsonify({
                'success': True,
                'mode': 'auto',
                'timeframes': all_data
            })
        else:
            # 단일 시간대 NB Wave 데이터
            # 실제 업비트 데이터 가져오기
            df = get_candles(market, timeframe, count=bars)
            
            # N/B Wave 계산 (간단한 예시 - 실제 로직으로 교체 필요)
            zones = []
            labels = []
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                # 시간 라벨
                labels.append(timestamp.strftime('%H:%M'))
                
                # 간단한 N/B 판단 (실제 로직으로 교체 필요)
                close = row['close']
                open_price = row['open']
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
                        zone = 'ORANGE' if random.random() > 0.5 else 'BLUE'
                        strength = 0.4
                
                zones.append({
                    'zone': zone,
                    'strength': round(strength, 2),
                    'price': float(close),
                    'volume': float(volume),
                    'change': round(price_change * 100, 2)
                })
        
        summary = {
            'orange': sum(1 for z in zones if z['zone'] == 'ORANGE'),
            'blue': sum(1 for z in zones if z['zone'] == 'BLUE'),
            'current_price': float(df['close'].iloc[-1]),
            'price_change_24h': round(((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100, 2)
        }

        return jsonify({
            'timeframe': timeframe,
            'labels': labels,
            'zones': zones,
            'summary': summary,
            'last_update': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error fetching NB wave data: {e}")
        return jsonify({
            'error': str(e),
            'timeframe': timeframe,
            'labels': [],
            'zones': [],
            'summary': {'orange': 0, 'blue': 0}
        }), 500

# ===== 백그라운드 시스템 API 엔드포인트 =====

@app.route('/api/background/start')
def start_background_system():
    """백그라운드 시스템 시작"""
    try:
        background_manager.start_system()
        return jsonify({
            "status": "success",
            "message": "Background system started",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/background/stop')
def stop_background_system():
    """백그라운드 시스템 중지"""
    try:
        background_manager.stop_system()
        return jsonify({
            "status": "success",
            "message": "Background system stopped",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/background/status')
def get_background_status():
    """백그라운드 시스템 상태 조회"""
    try:
        status = background_manager.get_system_status()
        return jsonify({
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/background/data/<data_type>')
def get_background_data(data_type):
    """백그라운드에서 수집된 데이터 조회"""
    try:
        data = background_manager.get_cached_data(data_type)
        return jsonify({
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/background/events')
def get_background_events():
    """백그라운드 이벤트 스트림 (Server-Sent Events)"""
    def generate():
        while True:
            try:
                # 현재 캐시된 데이터를 이벤트로 전송
                trading_data = background_manager.get_cached_data('trading')
                wallet_data = background_manager.get_cached_data('wallet')
                market_data = background_manager.get_cached_data('market')
                
                events = {
                    'trading': trading_data,
                    'wallet': wallet_data,
                    'market': market_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(events)}\n\n"
                time.sleep(5)  # 5초마다 업데이트
                
            except Exception as e:
                print(f"❌ Error in event stream: {e}")
                time.sleep(10)
    
    return Response(generate(), mimetype='text/event-stream')

# ===== Trading Module API =====

@app.route('/api/trading/data/<timeframe>')
def get_trading_data(timeframe):
    """Trading 데이터 조회"""
    try:
        data = trading_module.get_trading_data(timeframe)
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/trading/price-chart/<timeframe>')
def get_price_chart_data(timeframe):
    """가격 차트 데이터 조회"""
    try:
        data = trading_module.get_price_chart_data(timeframe)
        return jsonify({
            "status": "success",
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/trading/timeframe', methods=['POST'])
def set_trading_timeframe():
    """Trading 타임프레임 설정"""
    try:
        data = request.get_json()
        timeframe = data.get('timeframe', 'minute1')
        trading_module.set_timeframe(timeframe)
        return jsonify({
            "status": "success",
            "message": f"Timeframe set to {timeframe}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/trading/auto-rotation', methods=['POST'])
def toggle_trading_auto_rotation():
    """Trading 자동 순환 토글"""
    try:
        is_active = trading_module.toggle_auto_rotation()
        return jsonify({
            "status": "success",
            "auto_rotation": is_active
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/trading/status')
def get_trading_status():
    """Trading 모듈 상태 조회"""
    try:
        status = trading_module.get_status()
        return jsonify({
            "status": "success",
            "data": status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ===== Wallet Module API =====

@app.route('/api/wallet/balance')
def get_wallet_balance():
    """Wallet 잔고 조회"""
    try:
        light_mode = request.args.get('light', 'false').lower() == 'true'
        print(f"🔍 /api/wallet/balance called (light={light_mode})")
        
        # Settings에서 API 키 가져오기
        upbit_settings = settings_module.get_settings('upbit')
        print(f"📋 Upbit settings: {upbit_settings}")
        api_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')
        print(f"🔑 API Key exists: {bool(api_key)}, Secret Key exists: {bool(secret_key)}")
        
        # API 키가 설정되어 있으면 wallet_module에 설정
        if api_key and secret_key:
            print(f"✅ Setting API keys to wallet_module")
            wallet_module.set_api_keys(api_key, secret_key)
        else:
            print(f"❌ API keys not found in settings")
        
        result = wallet_module.get_balance()
        print(f"📊 Balance result: {result.get('status', 'unknown')}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in get_wallet_balance: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/wallet/transactions')
def get_wallet_transactions():
    """Wallet 거래 내역 조회"""
    try:
        # Settings에서 API 키 가져오기
        upbit_settings = settings_module.get_settings('upbit')
        api_key = upbit_settings.get('upbitAccessKey', '')
        secret_key = upbit_settings.get('upbitSecretKey', '')
        
        # API 키가 설정되어 있으면 wallet_module에 설정
        if api_key and secret_key:
            wallet_module.set_api_keys(api_key, secret_key)
        
        limit = request.args.get('limit', 20, type=int)
        result = wallet_module.get_transactions(limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/wallet/test-connection', methods=['POST'])
def test_wallet_connection():
    """Wallet API 연결 테스트"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '')
        secret_key = data.get('secret_key', '')
        
        # 전달받은 API 키로 테스트
        if api_key and secret_key:
            wallet_module.set_api_keys(api_key, secret_key)
        
        result = wallet_module.test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/wallet/status')
def get_wallet_status():
    """Wallet 모듈 상태 조회"""
    try:
        status = wallet_module.get_status()
        return jsonify({
            "status": "success",
            "data": status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ===== Settings Module API =====

@app.route('/api/settings/get')
def get_settings():
    """설정 조회"""
    try:
        section = request.args.get('section')
        settings = settings_module.get_settings(section)
        return jsonify({
            "status": "success",
            "data": settings
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    """설정 업데이트"""
    try:
        data = request.get_json()
        section = data.get('section')
        key = data.get('key')
        value = data.get('value')
        
        result = settings_module.update_settings(section, key, value)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """설정 초기화"""
    try:
        data = request.get_json()
        section = data.get('section')
        
        result = settings_module.reset_settings(section)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/export')
def export_settings():
    """설정 내보내기"""
    try:
        result = settings_module.export_settings()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/import', methods=['POST'])
def import_settings():
    """설정 가져오기"""
    try:
        data = request.get_json()
        result = settings_module.import_settings(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/test-upbit', methods=['POST'])
def test_upbit_connection():
    """Upbit API 연결 테스트"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '')
        secret_key = data.get('secret_key', '')
        
        result = settings_module.test_upbit_connection(api_key, secret_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/sound')
def get_sound_settings():
    """사운드 설정 조회"""
    try:
        settings = settings_module.get_sound_settings()
        return jsonify({
            "status": "success",
            "data": settings
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/sound', methods=['POST'])
def update_sound_settings():
    """사운드 설정 업데이트"""
    try:
        data = request.get_json()
        result = settings_module.update_sound_settings(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/settings/status')
def get_settings_status():
    """Settings 모듈 상태 조회"""
    try:
        status = settings_module.get_status()
        return jsonify({
            "status": "success",
            "data": status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ===== Central Hub - Game Engine API =====

@app.route('/api/game-engine/status')
def get_game_engine_status():
    """게임 엔진 상태 조회"""
    try:
        status = central_hub_engine.get_system_status()
        return jsonify({
            "status": "success",
            "data": status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/game-engine/residents')
def get_game_engine_residents():
    """게임 엔진 주민 정보 조회"""
    try:
        residents = central_hub_engine.get_residents_info()
        return jsonify({
            "status": "success",
            "data": residents
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/game-engine/market')
def get_game_engine_market():
    """게임 엔진 시장 정보 조회"""
    try:
        market = central_hub_engine.get_market_info()
        return jsonify({
            "status": "success",
            "data": market
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/game-engine/execute-action', methods=['POST'])
def execute_game_action():
    """게임 엔진 행동 실행"""
    try:
        data = request.json
        action_type = data.get('type')  # 'resident' or 'mayor'
        
        if action_type == 'resident':
            resident_id = data.get('resident_id')
            action = data.get('action')
            target = data.get('target')
            result = central_hub_engine.execute_resident_action(resident_id, action, target)
        elif action_type == 'mayor':
            action = data.get('action')
            result = central_hub_engine.execute_mayor_action(action)
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid action type"
            }), 400
        
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/system-status')
def get_system_status():
    """시스템 상태 조회"""
    return jsonify({
        "server": "Online",
        "connection": "Stable",
        "database": "Connected",
        "background_system": background_manager.get_system_status(),
        "central_hub_engine": central_hub_engine.get_system_status(),
        "rate_limits": rate_limiter.get_status(),
        "last_update": datetime.now().isoformat()
    })

# ===== Left Panel Logging (use safe port 5057) =====

MAX_LEFTPANEL_LINES = 100
LEFT_LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
LEFT_LOG_PATH = os.path.join(LEFT_LOG_DIR, 'left_panel.log')
SNAPSHOT_DIR = os.path.join(LEFT_LOG_DIR, 'snapshots')
TRAINER_DIR = os.path.join(LEFT_LOG_DIR, 'trainer')
TRAINER_DECISION_LOG = os.path.join(TRAINER_DIR, 'trainer_decision.log')
TRAINER_ACTIVITY_LOG = os.path.join(TRAINER_DIR, 'trainer_activity.log')

def _ensure_dirs():
    try:
        if not os.path.exists(LEFT_LOG_DIR):
            os.makedirs(LEFT_LOG_DIR, exist_ok=True)
        if not os.path.exists(SNAPSHOT_DIR):
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        if not os.path.exists(LEFT_LOG_PATH):
            with open(LEFT_LOG_PATH, 'w', encoding='utf-8') as f:
                f.write('')
        if not os.path.exists(TRAINER_DIR):
            os.makedirs(TRAINER_DIR, exist_ok=True)
        if not os.path.exists(TRAINER_DECISION_LOG):
            with open(TRAINER_DECISION_LOG, 'w', encoding='utf-8') as f:
                f.write('')
        if not os.path.exists(TRAINER_ACTIVITY_LOG):
            with open(TRAINER_ACTIVITY_LOG, 'w', encoding='utf-8') as f:
                f.write('')
    except Exception:
        pass

def _tail_limit_file(path: str, limit: int):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > limit:
            keep = lines[-limit:]
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(keep)
    except Exception:
        pass

@app.route('/api/leftpanel/log', methods=['POST', 'GET'])
def api_leftpanel_log():
    """Left panel status log (keeps last 100 lines)."""
    try:
        _ensure_dirs()
        if request.method == 'GET':
            try:
                with open(LEFT_LOG_PATH, 'r', encoding='utf-8') as f:
                    data = [line.rstrip('\n') for line in f.readlines()[-MAX_LEFTPANEL_LINES:]]
                return jsonify({'ok': True, 'lines': data})
            except Exception as e:
                return jsonify({'ok': False, 'error': str(e)}), 500

        payload = request.get_json(force=True, silent=True) or {}
        record = {
            'tf': payload.get('tf'),
            'text': payload.get('text'),
            'ts': int(payload.get('ts') or 0),
            'mode': payload.get('mode'),
            'type': payload.get('type') or 'status'
        }
        with open(LEFT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        _tail_limit_file(LEFT_LOG_PATH, MAX_LEFTPANEL_LINES)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

def _snapshot_path_for_ts(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(int(ts_ms)/1000.0)
    return os.path.join(SNAPSHOT_DIR, dt.strftime('%Y-%m-%d') + '.log')

@app.route('/api/leftpanel/snapshot', methods=['POST', 'GET'])
def api_leftpanel_snapshot():
    """Left panel snapshot log (day file, keeps last 100 lines)."""
    try:
        _ensure_dirs()
        if request.method == 'GET':
            day = request.args.get('day')
            if not day:
                day = datetime.utcnow().strftime('%Y-%m-%d')
            path = os.path.join(SNAPSHOT_DIR, f'{day}.log')
            if not os.path.exists(path):
                return jsonify({'ok': True, 'lines': []})
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n') for line in f.readlines()[-MAX_LEFTPANEL_LINES:]]
            return jsonify({'ok': True, 'lines': lines})

        payload = request.get_json(force=True, silent=True) or {}
        ts = int(payload.get('ts') or 0) or int(datetime.utcnow().timestamp() * 1000)
        # 전체 필드를 보존하되, ts는 보정해서 기록
        record = dict(payload)
        record['ts'] = ts
        path = _snapshot_path_for_ts(ts)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        _tail_limit_file(path, MAX_LEFTPANEL_LINES)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ===== Trainer Decision Log (single file, capped to 100) =====

@app.route('/api/trainer/decision-log', methods=['GET', 'POST'])
def api_trainer_decision_log():
    """Trainer decision log endpoint. Keeps last 100 records in a single file."""
    try:
        _ensure_dirs()
        if request.method == 'GET':
            try:
                with open(TRAINER_DECISION_LOG, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                # Return last 100 lines as parsed JSON if possible, else raw text
                trimmed = lines[-MAX_LEFTPANEL_LINES:]
                items = []
                for ln in trimmed:
                    ln = ln.strip('\n')
                    try:
                        items.append(json.loads(ln))
                    except Exception:
                        items.append({'text': ln})
                return jsonify({'ok': True, 'items': items})
            except Exception as e:
                return jsonify({'ok': False, 'error': str(e)}), 500

        # POST
        payload = request.get_json(force=True, silent=True) or {}
        # Attach server timestamp
        if 'ts' not in payload:
            payload['ts'] = int(datetime.utcnow().timestamp() * 1000)
        # Append record
        with open(TRAINER_DECISION_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        # Trim to 100
        _tail_limit_file(TRAINER_DECISION_LOG, MAX_LEFTPANEL_LINES)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ===== Trainer Activity Log (single file, capped to 200) =====

@app.route('/api/trainer/activity-log', methods=['GET', 'POST'])
def api_trainer_activity_log():
    """Trainer activity log endpoint. Keeps last 200 records in a single file."""
    try:
        _ensure_dirs()
        if request.method == 'GET':
            try:
                with open(TRAINER_ACTIVITY_LOG, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                trimmed = lines[-200:]
                items = []
                for ln in trimmed:
                    ln = ln.strip('\n')
                    try:
                        items.append(json.loads(ln))
                    except Exception:
                        items.append({'text': ln})
                return jsonify({'ok': True, 'items': items})
            except Exception as e:
                return jsonify({'ok': False, 'error': str(e)}), 500

        payload = request.get_json(force=True, silent=True) or {}
        if 'ts' not in payload:
            payload['ts'] = int(datetime.utcnow().timestamp() * 1000)
        with open(TRAINER_ACTIVITY_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        _tail_limit_file(TRAINER_ACTIVITY_LOG, 200)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/rate-limits')
def get_rate_limits():
    """Rate Limit 상태 조회"""
    try:
        return jsonify({
            "status": "success",
            "rate_limits": rate_limiter.get_status(),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/save-log', methods=['POST'])
def save_log():
    """로그 저장 API"""
    try:
        data = request.get_json()
        log_data = data.get('logData', '')
        file_name = data.get('fileName', 'ai-trainer-logs.txt')
        
        # log 폴더 생성 (없으면)
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"📁 로그 폴더 생성: {log_dir}")
        
        # 파일 경로 설정
        file_path = os.path.join(log_dir, file_name)
        
        # 로그 데이터를 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_data)
        
        print(f"✅ 로그 저장 완료: {file_path}")
        
        return jsonify({
            'success': True,
            'message': f'로그가 성공적으로 저장되었습니다. ({file_path})',
            'file_path': file_path
        })
        
    except Exception as e:
        print(f"❌ 로그 저장 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/save-log-file', methods=['POST'])
def save_log_file():
    """수익률 계산 로그 파일 저장 API (100줄 제한)"""
    try:
        data = request.get_json()
        filename = data.get('filename', '')
        content = data.get('content', '')
        log_count = data.get('logCount', 0)
        
        if not filename or not content:
            return jsonify({
                'success': False,
                'error': 'Filename and content are required'
            }), 400
        
        # trainer 폴더 경로 설정
        trainer_dir = os.path.join(os.path.dirname(__file__), 'log', 'trainer')
        if not os.path.exists(trainer_dir):
            os.makedirs(trainer_dir, exist_ok=True)
            print(f"📁 trainer 폴더 생성: {trainer_dir}")
        
        # 파일명에서 경로 제거하고 실제 파일명만 사용
        actual_filename = os.path.basename(filename)
        file_path = os.path.join(trainer_dir, actual_filename)
        
        # 새로운 로그 라인들을 분리하고 줄바꿈 추가
        new_lines = content.split('\n')
        if new_lines and new_lines[-1] == '':  # 마지막 빈 줄 제거
            new_lines = new_lines[:-1]
        
        # 각 라인에 줄바꿈 추가
        formatted_lines = []
        for line in new_lines:
            if line.strip():  # 빈 줄이 아닌 경우만 처리
                formatted_lines.append(line + '\n')
        
        # 100줄 제한 적용
        if len(formatted_lines) > 100:
            removed_count = len(formatted_lines) - 100
            formatted_lines = formatted_lines[removed_count:]  # 가장 오래된 데이터 제거
            print(f"📊 100줄 제한 적용: {removed_count}개 오래된 로그 제거됨")
        
        # 파일에 저장 (기존 파일 덮어쓰기)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(formatted_lines)
        
        final_line_count = len(formatted_lines)
        print(f"💾 수익률 로그 파일 저장 완료: {file_path} (총 {final_line_count}줄)")
        
        return jsonify({
            'success': True,
            'message': f'수익률 로그 파일이 성공적으로 저장되었습니다. (총 {final_line_count}줄)',
            'filepath': file_path,
            'log_count': final_line_count
        })
        
    except Exception as e:
        print(f"❌ 수익률 로그 파일 저장 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/explorer-log', methods=['POST'])
def save_explorer_log():
    """탐색원 로그 저장 API"""
    try:
        data = request.get_json()
        log_data = data.get('logData', '')
        file_name = data.get('fileName', 'explorer_movement.log')
        
        # log 폴더 생성 (없으면)
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"📁 로그 폴더 생성: {log_dir}")
        
        # 파일 경로 설정
        file_path = os.path.join(log_dir, file_name)
        
        # 로그 데이터를 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_data)
        
        print(f"✅ 탐색원 로그 저장 완료: {file_path}")
        
        return jsonify({
            'success': True,
            'message': f'탐색원 로그가 성공적으로 저장되었습니다. ({file_path})',
            'file_path': file_path
        })
        
    except Exception as e:
        print(f"❌ 탐색원 로그 저장 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/explorer-log', methods=['GET'])
def get_explorer_log():
    """탐색원 로그 읽기 API"""
    try:
        file_name = request.args.get('fileName', 'explorer_movement.log')
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        file_path = os.path.join(log_dir, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return jsonify({
                'success': True,
                'content': content,
                'file_path': file_path
            })
        else:
            return jsonify({
                'success': False,
                'error': '로그 파일이 존재하지 않습니다.'
            }), 404
            
    except Exception as e:
        print(f"❌ 탐색원 로그 읽기 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/explorer-log-stats', methods=['GET'])
def get_explorer_log_stats():
    """탐색원 로그 통계 정보 API"""
    try:
        file_name = request.args.get('fileName', 'explorer_movement.log')
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        file_path = os.path.join(log_dir, file_name)
        
        stats = {
            'file_exists': os.path.exists(file_path),
            'file_path': file_path,
            'log_dir': log_dir
        }
        
        if stats['file_exists']:
            file_size = os.path.getsize(file_path)
            stats['file_size'] = file_size
            
            # 파일의 줄 수 계산
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                stats['line_count'] = len(lines)
                stats['message_count'] = len([line for line in lines if line.strip()])
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ 탐색원 로그 통계 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===== 카드 저장소 API =====

def get_card_storage_file():
    """카드 저장소 파일 경로 반환"""
    return os.path.join(os.path.dirname(__file__), 'data', 'card_storage.json')

def ensure_data_directory():
    """데이터 디렉토리 생성"""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 데이터 디렉토리 생성: {data_dir}")

def load_card_storage():
    """카드 저장소 데이터 로드"""
    try:
        file_path = get_card_storage_file()
        ensure_data_directory()
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"📂 카드 저장소 로드 완료: {len(data)}개 타임프레임")
                return data
        else:
            print("📂 새로운 카드 저장소 생성")
            return {}
    except Exception as e:
        print(f"❌ 카드 저장소 로드 실패: {e}")
        return {}

def save_card_storage(data):
    """카드 저장소 데이터 저장"""
    try:
        file_path = get_card_storage_file()
        ensure_data_directory()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 카드 저장소 저장 완료: {len(data)}개 타임프레임")
        return True
    except Exception as e:
        print(f"❌ 카드 저장소 저장 실패: {e}")
        return False

@app.route('/api/card-storage', methods=['GET'])
def get_card_storage():
    """카드 저장소 전체 데이터 조회"""
    try:
        data = load_card_storage()
        return jsonify({
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"❌ 카드 저장소 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/card-storage/<timeframe>', methods=['GET'])
def get_card_storage_timeframe(timeframe):
    """특정 타임프레임 카드 저장소 조회"""
    try:
        data = load_card_storage()
        if timeframe in data:
            return jsonify({
                'success': True,
                'data': data[timeframe],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'타임프레임 {timeframe}을 찾을 수 없습니다.'
            }), 404
    except Exception as e:
        print(f"❌ 타임프레임 {timeframe} 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/card-storage/<timeframe>', methods=['POST'])
def update_card_storage(timeframe):
    """카드 저장소 데이터 업데이트"""
    try:
        data = load_card_storage()
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({
                'success': False,
                'error': '요청 데이터가 없습니다.'
            }), 400
        
        # 기본 구조 생성
        if timeframe not in data:
            data[timeframe] = {
                'timeframe': timeframe,
                'nbCoins': 0,
                'nbMinerals': 0.0,
                'buyCount': 0,
                'sellCount': 0,
                'totalProfit': 0,
                'lastBuyPrice': 0,
                'lastSellPrice': 0,
                'lastBuyTime': 0,
                'lastSellTime': 0,
                'createdAt': datetime.now().isoformat(),
                'lastUpdated': datetime.now().isoformat()
            }
        
        # 데이터 업데이트
        for key, value in request_data.items():
            if key in data[timeframe]:
                data[timeframe][key] = value
        
        data[timeframe]['lastUpdated'] = datetime.now().isoformat()
        
        # 저장
        if save_card_storage(data):
            return jsonify({
                'success': True,
                'data': data[timeframe],
                'message': f'타임프레임 {timeframe} 업데이트 완료'
            })
        else:
            return jsonify({
                'success': False,
                'error': '저장에 실패했습니다.'
            }), 500
            
    except Exception as e:
        print(f"❌ 타임프레임 {timeframe} 업데이트 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/card-storage/<timeframe>/nb-coin', methods=['POST'])
def update_nb_coin(timeframe):
    """N/B 코인 업데이트"""
    try:
        data = load_card_storage()
        request_data = request.get_json()
        
        if not request_data or 'action' not in request_data:
            return jsonify({
                'success': False,
                'error': 'action 파라미터가 필요합니다.'
            }), 400
        
        action = request_data['action']  # 'add' 또는 'remove'
        count = request_data.get('count', 1)
        
        if timeframe not in data:
            data[timeframe] = {
                'timeframe': timeframe,
                'nbCoins': 0,
                'nbMinerals': 0.0,
                'buyCount': 0,
                'sellCount': 0,
                'totalProfit': 0,
                'lastBuyPrice': 0,
                'lastSellPrice': 0,
                'lastBuyTime': 0,
                'lastSellTime': 0,
                'createdAt': datetime.now().isoformat(),
                'lastUpdated': datetime.now().isoformat()
            }
        
        if action == 'add':
            data[timeframe]['nbCoins'] += count
            message = f'N/B 코인 +{count} 추가'
        elif action == 'remove':
            if data[timeframe]['nbCoins'] >= count:
                data[timeframe]['nbCoins'] -= count
                message = f'N/B 코인 -{count} 제거'
            else:
                return jsonify({
                    'success': False,
                    'error': f'N/B 코인이 부족합니다. 현재: {data[timeframe]["nbCoins"]}개, 요청: {count}개'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': '잘못된 action입니다. (add/remove)'
            }), 400
        
        # 음수 방지
        data[timeframe]['nbCoins'] = max(0, data[timeframe]['nbCoins'])
        data[timeframe]['lastUpdated'] = datetime.now().isoformat()
        
        # 저장
        if save_card_storage(data):
            return jsonify({
                'success': True,
                'data': data[timeframe],
                'message': message,
                'nbCoins': data[timeframe]['nbCoins']
            })
        else:
            return jsonify({
                'success': False,
                'error': '저장에 실패했습니다.'
            }), 500
            
    except Exception as e:
        print(f"❌ 타임프레임 {timeframe} N/B 코인 업데이트 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/card-storage/<timeframe>/nb-mineral', methods=['POST'])
def update_nb_mineral(timeframe):
    """N/B 미네랄 업데이트"""
    try:
        data = load_card_storage()
        request_data = request.get_json()
        
        if not request_data or 'action' not in request_data:
            return jsonify({
                'success': False,
                'error': 'action 파라미터가 필요합니다.'
            }), 400
        
        action = request_data['action']  # 'add' 또는 'remove'
        amount = request_data.get('amount', 1.0)
        
        if timeframe not in data:
            data[timeframe] = {
                'timeframe': timeframe,
                'nbCoins': 0,
                'nbMinerals': 0.0,
                'buyCount': 0,
                'sellCount': 0,
                'totalProfit': 0,
                'lastBuyPrice': 0,
                'lastSellPrice': 0,
                'lastBuyTime': 0,
                'lastSellTime': 0,
                'createdAt': datetime.now().isoformat(),
                'lastUpdated': datetime.now().isoformat()
            }
        
        if action == 'add':
            data[timeframe]['nbMinerals'] += amount
            message = f'N/B 미네랄 +{amount}% 추가'
        elif action == 'remove':
            if data[timeframe]['nbMinerals'] >= amount:
                data[timeframe]['nbMinerals'] -= amount
                message = f'N/B 미네랄 -{amount}% 제거'
            else:
                return jsonify({
                    'success': False,
                    'error': f'N/B 미네랄이 부족합니다. 현재: {data[timeframe]["nbMinerals"]}%, 요청: {amount}%'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': '잘못된 action입니다. (add/remove)'
            }), 400
        
        data[timeframe]['lastUpdated'] = datetime.now().isoformat()
        
        # 저장
        if save_card_storage(data):
            return jsonify({
                'success': True,
                'data': data[timeframe],
                'message': message,
                'nbMinerals': data[timeframe]['nbMinerals']
            })
        else:
            return jsonify({
                'success': False,
                'error': '저장에 실패했습니다.'
            }), 500
            
    except Exception as e:
        print(f"❌ 타임프레임 {timeframe} N/B 미네랄 업데이트 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/card-storage/reset', methods=['POST'])
def reset_card_storage():
    """카드 저장소 초기화"""
    try:
        request_data = request.get_json() or {}
        timeframe = request_data.get('timeframe')
        
        if timeframe:
            # 특정 타임프레임만 초기화
            data = load_card_storage()
            if timeframe in data:
                del data[timeframe]
                save_card_storage(data)
                return jsonify({
                    'success': True,
                    'message': f'타임프레임 {timeframe} 초기화 완료'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'타임프레임 {timeframe}을 찾을 수 없습니다.'
                }), 404
        else:
            # 전체 초기화
            save_card_storage({})
            return jsonify({
                'success': True,
                'message': '전체 카드 저장소 초기화 완료'
            })
            
    except Exception as e:
        print(f"❌ 카드 저장소 초기화 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ===== 정적 파일 서빙 =====

@app.route('/<path:filename>')
def static_files(filename):
    """정적 파일 서빙"""
    return send_from_directory('.', filename)

# ===== 에러 핸들러 =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": "The requested resource was not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": "Something went wrong"}), 500

# ===== 메인 실행 =====

if __name__ == '__main__':
    print("🚀 8BIT Trading System v0.1 - Starcraft Style UI")
    print("📍 Server starting on http://127.0.0.1:5057")
    print("🎮 Game UI available at http://127.0.0.1:5057/ui")
    print("📊 API endpoints available at http://127.0.0.1:5057/api/")
    print("=" * 50)
    
    # 페이지 로딩 모니터 시작
    # page_monitor.start_monitoring() # Removed page_monitor
    
    app.run(host='127.0.0.1', port=5057, debug=True)
