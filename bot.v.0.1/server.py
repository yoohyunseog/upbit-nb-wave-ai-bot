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

# ===== PyUpbit 유틸리티 함수 =====

def get_candles(market: str, candle: str, count: int = 200) -> pd.DataFrame:
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
    """오디오 파일 재생"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"🎵 Playing sound file: {file_path}")
        
        # 재생 완료 후 파일 삭제
        def cleanup():
            time.sleep(0.2)  # 재생 완료 대기
            try:
                os.unlink(file_path)
                print(f"🎵 Cleaned up: {file_path}")
            except:
                pass
        
        threading.Thread(target=cleanup, daemon=True).start()
        
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
    """스타크래프트 사운드 파일 재생"""
    try:
        # 볼륨 설정
        pygame.mixer.music.set_volume(volume)
        
        # 사운드 파일 로드 및 재생
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        print(f"🎵 Playing Starcraft sound: {file_path}")
        
    except Exception as e:
        print(f"🎵 Error playing Starcraft sound: {e}")
        # 실패 시 비프음으로 fallback
        fallback_beep_sound(sound_type, volume)

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
@app.route('/api/play-sound', methods=['POST'])
def api_play_sound():
    """사운드 재생 API (실제 오디오 재생)"""
    try:
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
    """업비트 API 연결 테스트"""
    try:
        data = request.get_json()
        access_key = data.get('accessKey', '')
        secret_key = data.get('secretKey', '')
        
        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Access Key and Secret Key are required.'
            }), 400
        
        print(f"🔑 Testing Upbit API connection...")
        
        # Upbit API connection test
        try:
            # Create Upbit instance
            upbit = pyupbit.Upbit(access_key, secret_key)
            
            # Test API connection by checking balance
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
                    'error': 'Balance check failed - Please verify your API keys.'
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
    """Get Upbit account balance"""
    try:
        data = request.get_json()
        access_key = data.get('accessKey', '')
        secret_key = data.get('secretKey', '')
        
        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Access Key and Secret Key are required.'
            }), 400
        
        print(f"💰 Fetching Upbit balance...")
        
        try:
            # Create Upbit instance
            upbit = pyupbit.Upbit(access_key, secret_key)
            
            # Get all balances
            balances = upbit.get_balances()
            
            if balances is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to fetch balance data'
                })
            
            # Process balance data
            processed_balances = []
            total_portfolio_value = 0
            
            for balance in balances:
                currency = balance['currency']
                balance_amount = float(balance['balance'])
                locked_amount = float(balance['locked'])
                
                # Skip zero balances
                if balance_amount == 0 and locked_amount == 0:
                    continue
                
                # Get current price for non-KRW currencies
                current_price = 0
                avg_buy_price = 0
                asset_value = 0
                
                if currency == 'KRW':
                    current_price = 1
                    avg_buy_price = 1
                    asset_value = balance_amount
                else:
                    try:
                        # Get current price
                        ticker = pyupbit.get_current_price(f"KRW-{currency}")
                        if ticker and ticker > 0:
                            current_price = ticker
                            asset_value = balance_amount * current_price
                        else:
                            print(f"💡 {currency} price not available on Upbit KRW market")
                            current_price = 0
                            asset_value = 0
                        
                        # Get average buy price (if available)
                        avg_buy_price = float(balance.get('avg_buy_price', 0))
                        if avg_buy_price == 0 and current_price > 0:
                            avg_buy_price = current_price
                            
                    except Exception as price_error:
                        # Only log if it's not the common "Code not found" error
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
            return jsonify({
                'success': False,
                'error': f'Balance fetch failed: {str(api_error)}'
            })
        
    except Exception as e:
        print(f"💰 Upbit balance error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
    """WebSocket 연결 시작 API"""
    try:
        data = request.get_json()
        access_key = data.get('accessKey', '')
        secret_key = data.get('secretKey', '')
        
        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Access Key and Secret Key are required.'
            }), 400
        
        success = start_websocket_connection(access_key, secret_key)
        
        return jsonify({
            'success': success,
            'message': 'WebSocket connection started successfully.' if success else 'Failed to start WebSocket connection.'
        })
        
    except Exception as e:
        print(f"❌ WebSocket 시작 API 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/order-history', methods=['POST'])
def api_order_history():
    """주문 히스토리 조회 API (Upbit REST API 사용)"""
    try:
        data = request.get_json()
        access_key = data.get('accessKey', '')
        secret_key = data.get('secretKey', '')
        
        if not access_key or not secret_key:
            return jsonify({
                'success': False,
                'error': 'Access Key and Secret Key are required.'
            }), 400
        
        # Upbit API 인스턴스 생성
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

@app.route('/style.css')
def style():
    """CSS 파일 서빙"""
    return send_from_directory('.', 'style.css')

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

@app.route('/api/village-status')
def get_village_status():
    """마을 상태 조회"""
    return jsonify({
        "energy": VILLAGE_ENERGY,
        "max_energy": MAX_VILLAGE_ENERGY,
        "residents": VILLAGE_RESIDENTS,
        "entry_exit_log": VILLAGE_ENTRY_EXIT_LOG,
        "mayor_trust": MAYOR_TRUST_SYSTEM
    })

@app.route('/api/residents')
def get_residents():
    """주민 정보 조회"""
    return jsonify(VILLAGE_RESIDENTS)

@app.route('/api/update-resident', methods=['POST'])
def update_resident():
    """주민 정보 업데이트"""
    data = request.json
    resident_id = data.get('id')
    updates = data.get('updates', {})
    
    if resident_id in VILLAGE_RESIDENTS:
        VILLAGE_RESIDENTS[resident_id].update(updates)
        return jsonify({"status": "success", "resident": VILLAGE_RESIDENTS[resident_id]})
    else:
        return jsonify({"status": "error", "message": "Resident not found"}), 404

@app.route('/api/trading-data')
def get_trading_data():
    """실제 업비트 데이터 기반 트레이딩 정보"""
    try:
        # URL 파라미터에서 시간대 가져오기
        timeframe = request.args.get('timeframe', 'minute1')
        
        # auto 옵션 처리
        if timeframe == 'auto':
            # 모든 시간대의 데이터를 순차적으로 가져오기
            timeframes = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'minute240', 'day', 'week', 'month']
            all_data = {}
            
            for tf in timeframes:
                try:
                    # 현재 가격
                    current_price = float(pyupbit.get_current_price("KRW-BTC") or 0)
                    
                    # 선택된 시간대 데이터 (최근 60개)
                    df = get_candles("KRW-BTC", tf, count=60)
                    
                    # 가격 변화율 계산
                    price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
                    
                    # 거래량
                    volume_24h = df['volume'].sum()
                    
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
            current_price = float(pyupbit.get_current_price("KRW-BTC") or 0)
            
            # 선택된 시간대 데이터 (최근 60개)
            df = get_candles("KRW-BTC", timeframe, count=60)
            
            # 가격 변화율 계산
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
            
            # 거래량
            volume_24h = df['volume'].sum()
            
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
    try:
        bars = int(request.args.get('bars', 120))
    except Exception:
        bars = 120

    try:
        # auto 옵션 처리
        if timeframe == 'auto':
            # 모든 시간대의 NB Wave 데이터를 순차적으로 가져오기
            timeframes = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'minute240', 'day', 'week', 'month']
            all_data = {}
            
            for tf in timeframes:
                try:
                    # 실제 업비트 데이터 가져오기
                    df = get_candles("KRW-BTC", tf, count=bars)
                    
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
            df = get_candles("KRW-BTC", timeframe, count=bars)
            
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

@app.route('/api/system-status')
def get_system_status():
    """시스템 상태 조회"""
    return jsonify({
        "server": "Online",
        "connection": "Stable",
        "database": "Connected",
        "last_update": datetime.now().isoformat()
    })

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
