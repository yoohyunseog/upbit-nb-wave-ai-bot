# ===== Wallet Module - Python Backend =====

import pyupbit
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Settings 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.settings_module import SettingsModule

class WalletModule:
    """Wallet 모듈 - Settings와 완전 연동"""
    
    def __init__(self):
        self.api_key = None
        self.secret_key = None
        self.is_connected = False
        self.last_update = None
        self.cache_duration = 30  # 30초 캐시
        self.settings_module = SettingsModule()
        
    def set_api_keys(self, api_key: str, secret_key: str) -> bool:
        """API 키 설정"""
        try:
            print(f"🔑 Setting API keys - API Key: {api_key[:10]}..., Secret: {secret_key[:10]}...")
            self.api_key = api_key
            self.secret_key = secret_key
            self.is_connected = True
            print(f"💰 Wallet API keys set - is_connected: {self.is_connected}")
            return True
        except Exception as e:
            print(f"❌ Error setting API keys: {e}")
            self.is_connected = False
            return False
    
    def get_balance(self) -> Dict:
        """잔고 조회"""
        try:
            print(f"🔍 get_balance called - is_connected: {self.is_connected}")
            
            if not self.is_connected or not self.api_key or not self.secret_key:
                print(f"❌ Not connected to Upbit API")
                return {
                    'status': 'error',
                    'message': 'Not connected to Upbit API',
                    'data': {
                        'total_value': 0,
                        'total_krw': 0,
                        'total_btc_value': 0,
                        'balances': []
                    }
                }
            
            # Upbit API 호출
            upbit = pyupbit.Upbit(self.api_key, self.secret_key)
            balances = upbit.get_balances()
            
            if not balances:
                return {
                    'status': 'error',
                    'message': 'Failed to fetch balance data',
                    'data': {
                        'total_value': 0,
                        'total_krw': 0,
                        'total_btc_value': 0,
                        'balances': []
                    }
                }
            
            # 잔고 데이터 처리
            total_value = 0
            total_krw = 0
            total_btc_value = 0
            processed_balances = []
            
            for balance in balances:
                currency = balance['currency']
                balance_amount = float(balance['balance'])
                locked_amount = float(balance['locked'])
                
                # 0 잔고는 제외하되, 선택된 코인은 포함 (평균 매수가 표시를 위해)
                upbit_settings = self.settings_module.get_settings('upbit')
                selected_coin = upbit_settings.get('defaultKrwCoin', 'BTC')
                if balance_amount == 0 and locked_amount == 0 and currency != selected_coin:
                    continue
                
                # KRW 처리
                if currency == 'KRW':
                    total_krw = balance_amount
                    total_value += balance_amount
                    processed_balances.append({
                        'currency': currency,
                        'balance': balance_amount,
                        'locked': locked_amount,
                        'avg_buy_price': 1,
                        'current_price': 1,
                        'asset_value': balance_amount
                    })
                else:
                    # 암호화폐 처리
                    try:
                        # 현재가 조회
                        ticker = pyupbit.get_current_price(f"KRW-{currency}")
                        current_price = ticker if ticker else 0
                        asset_value = balance_amount * current_price
                        
                        # 평균 매수가
                        avg_buy_price = float(balance.get('avg_buy_price', 0))
                        
                        # 평균 매수가가 0이면 이전 거래 내역에서 가져오기
                        if avg_buy_price == 0:
                            try:
                                orders = upbit.get_orders(state='done', limit=50)
                                for order in orders:
                                    if order.get('market') == f'KRW-{currency}' and order.get('side') == 'bid':
                                        # 매수 거래에서 평균 매수가 계산
                                        executed_volume = float(order.get('executed_volume', 0))
                                        price = float(order.get('price', 0))
                                        if executed_volume > 0:
                                            avg_buy_price = price
                                            break
                            except Exception as e:
                                print(f"⚠️ Error fetching avg price for {currency}: {e}")
                                
                        # 여전히 0이면 현재가로 설정
                        if avg_buy_price == 0 and current_price > 0:
                            avg_buy_price = current_price
                        
                        # BTC 가치 계산 (BTC 기준)
                        if currency == 'BTC':
                            total_btc_value = asset_value
                        else:
                            btc_price = pyupbit.get_current_price("KRW-BTC")
                            if btc_price:
                                total_btc_value += asset_value / btc_price
                        
                        total_value += asset_value
                        
                        processed_balances.append({
                            'currency': currency,
                            'balance': balance_amount,
                            'locked': locked_amount,
                            'avg_buy_price': avg_buy_price,
                            'current_price': current_price,
                            'asset_value': asset_value
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Error processing {currency}: {e}")
                        continue
            
            # 선택된 코인이 없으면 0으로 추가
            upbit_settings = self.settings_module.get_settings('upbit')
            selected_coin = upbit_settings.get('defaultKrwCoin', 'BTC')
            has_selected_coin = any(balance['currency'] == selected_coin for balance in processed_balances)
            
            if not has_selected_coin:
                # 선택된 코인의 이전 거래 내역에서 평균 매수가 가져오기
                try:
                    orders = upbit.get_orders(state='done', limit=50)
                    selected_coin_avg_price = 0
                    
                    for order in orders:
                        if order.get('market') == f'KRW-{selected_coin}' and order.get('side') == 'bid':
                            # 매수 거래에서 평균 매수가 계산
                            executed_volume = float(order.get('executed_volume', 0))
                            price = float(order.get('price', 0))
                            if executed_volume > 0:
                                selected_coin_avg_price = price
                                break
                    
                    processed_balances.append({
                        'currency': selected_coin,
                        'balance': 0,
                        'locked': 0,
                        'avg_buy_price': selected_coin_avg_price,
                        'current_price': 0,
                        'asset_value': 0
                    })
                except Exception as e:
                    print(f"⚠️ Error adding selected coin: {e}")
                    processed_balances.append({
                        'currency': selected_coin,
                        'balance': 0,
                        'locked': 0,
                        'avg_buy_price': 0,
                        'current_price': 0,
                        'asset_value': 0
                    })
            
            self.last_update = datetime.now()
            
            return {
                'status': 'success',
                'message': 'Balance fetched successfully',
                'data': {
                    'total_value': total_value,
                    'total_krw': total_krw,
                    'total_btc_value': total_btc_value,
                    'balances': processed_balances,
                    'last_update': self.last_update.isoformat() if self.last_update else None
                }
            }
            
        except Exception as e:
            print(f"❌ Error in get_balance: {e}")
            return {
                'status': 'error',
                'message': f'Error fetching balance: {str(e)}',
                'data': {
                    'total_value': 0,
                    'total_krw': 0,
                    'total_btc_value': 0,
                    'balances': []
                }
            }
    
    def get_transactions(self, limit: int = 20) -> Dict:
        """거래 내역 조회"""
        try:
            if not self.is_connected or not self.api_key or not self.secret_key:
                return {
                    'status': 'error',
                    'message': 'Not connected to Upbit API',
                    'data': []
                }
            
            upbit = pyupbit.Upbit(self.api_key, self.secret_key)
            orders = upbit.get_orders(state='done', limit=limit)
            
            if not orders:
                return {
                    'status': 'success',
                    'message': 'No transactions found',
                    'data': []
                }
            
            processed_orders = []
            for order in orders:
                processed_orders.append({
                    'uuid': order.get('uuid', ''),
                    'market': order.get('market', ''),
                    'side': order.get('side', ''),  # bid: 매수, ask: 매도
                    'price': float(order.get('price', 0)),
                    'volume': float(order.get('volume', 0)),
                    'executed_volume': float(order.get('executed_volume', 0)),
                    'executed_funds': float(order.get('executed_funds', 0)),
                    'state': order.get('state', ''),
                    'created_at': order.get('created_at', ''),
                    'updated_at': order.get('updated_at', '')
                })
            
            return {
                'status': 'success',
                'message': 'Transactions fetched successfully',
                'data': processed_orders
            }
            
        except Exception as e:
            print(f"❌ Error in get_transactions: {e}")
            return {
                'status': 'error',
                'message': f'Error fetching transactions: {str(e)}',
                'data': []
            }
    
    def test_connection(self) -> Dict:
        """API 연결 테스트"""
        try:
            if not self.api_key or not self.secret_key:
                return {
                    'status': 'error',
                    'message': 'API keys not set'
                }
            
            upbit = pyupbit.Upbit(self.api_key, self.secret_key)
            balance = upbit.get_balance("KRW")
            
            if balance is not None:
                return {
                    'status': 'success',
                    'message': 'API connection successful',
                    'balance': float(balance) if balance else 0
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to fetch balance - check API keys'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection failed: {str(e)}'
            }
    
    def get_status(self) -> Dict:
        """모듈 상태 조회"""
        return {
            'is_connected': self.is_connected,
            'has_api_keys': bool(self.api_key and self.secret_key),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cache_duration': self.cache_duration
        }

# 전역 인스턴스
wallet_module = WalletModule()
