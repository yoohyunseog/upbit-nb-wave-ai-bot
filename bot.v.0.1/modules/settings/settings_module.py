# ===== Settings Module - Python Backend =====

import json
import os
from datetime import datetime
from flask import jsonify, request

class SettingsModule:
    """Settings 모듈"""
    
    def __init__(self):
        self.settings_file = 'user_settings.json'
        self.default_settings = {
            'upbit': {
                'upbitAccessKey': '',
                'upbitSecretKey': '',
                'defaultKrwCoin': 'BTC',
                'test_mode': True
            },
            'trading': {
                'default_timeframe': 'minute1',
                'auto_rotation': False,
                'auto_rotation_interval': 5
            },
            'sound': {
                'master_volume': 0.5,
                'click_volume': 0.3,
                'success_volume': 0.4,
                'error_volume': 0.4,
                'type_volume': 0.2,
                'sequence_volume': 0.1,
                'enabled': True
            },
            'display': {
                'theme': 'starcraft',
                'language': 'en',
                'auto_refresh': True,
                'refresh_interval': 30
            },
            'system': {
                'background_monitoring': True,
                'data_cache_duration': 30,
                'log_level': 'info'
            }
        }
        self.current_settings = self.load_settings()
        
    def load_settings(self) -> dict:
        """설정 파일 로드"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    # 기본 설정과 병합
                    return self._merge_settings(self.default_settings, settings)
            else:
                # 기본 설정으로 파일 생성
                self.save_settings(self.default_settings)
                return self.default_settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self.default_settings
    
    def save_settings(self, settings: dict) -> bool:
        """설정 파일 저장"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            self.current_settings = settings
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def _merge_settings(self, default: dict, user: dict) -> dict:
        """설정 병합 (기본값 + 사용자 설정)"""
        result = default.copy()
        
        def merge_dict(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(result, user)
        return result
    
    def get_settings(self, section: str = None) -> dict:
        """설정 조회"""
        if section:
            return self.current_settings.get(section, {})
        return self.current_settings
    
    def update_settings(self, section: str, key: str, value) -> dict:
        """설정 업데이트"""
        try:
            if section not in self.current_settings:
                self.current_settings[section] = {}
            
            self.current_settings[section][key] = value
            
            # 파일에 저장
            if self.save_settings(self.current_settings):
                return {
                    'status': 'success',
                    'message': f'Setting {section}.{key} updated successfully',
                    'data': {section: {key: value}}
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to save settings'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating settings: {str(e)}'
            }
    
    def reset_settings(self, section: str = None) -> dict:
        """설정 초기화"""
        try:
            if section:
                if section in self.default_settings:
                    self.current_settings[section] = self.default_settings[section].copy()
                else:
                    return {
                        'status': 'error',
                        'message': f'Invalid section: {section}'
                    }
            else:
                self.current_settings = self.default_settings.copy()
            
            # 파일에 저장
            if self.save_settings(self.current_settings):
                return {
                    'status': 'success',
                    'message': f'Settings reset successfully',
                    'data': self.current_settings
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to save reset settings'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error resetting settings: {str(e)}'
            }
    
    def export_settings(self) -> dict:
        """설정 내보내기"""
        try:
            export_data = {
                'settings': self.current_settings,
                'export_time': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            return {
                'status': 'success',
                'data': export_data
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error exporting settings: {str(e)}'
            }
    
    def import_settings(self, settings_data: dict) -> dict:
        """설정 가져오기"""
        try:
            if 'settings' not in settings_data:
                return {
                    'status': 'error',
                    'message': 'Invalid settings data format'
                }
            
            imported_settings = settings_data['settings']
            
            # 기본 설정과 병합
            merged_settings = self._merge_settings(self.default_settings, imported_settings)
            
            # 파일에 저장
            if self.save_settings(merged_settings):
                return {
                    'status': 'success',
                    'message': 'Settings imported successfully',
                    'data': merged_settings
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to save imported settings'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error importing settings: {str(e)}'
            }
    
    def test_upbit_connection(self, api_key: str, secret_key: str) -> dict:
        """Upbit API 연결 테스트"""
        try:
            import pyupbit
            
            # API 키로 잔고 조회 테스트
            balance = pyupbit.get_balance("KRW", api_key=api_key, secret=secret_key)
            
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
    
    def get_sound_settings(self) -> dict:
        """사운드 설정 조회"""
        return self.current_settings.get('sound', {})
    
    def update_sound_settings(self, settings: dict) -> dict:
        """사운드 설정 업데이트"""
        try:
            current_sound = self.current_settings.get('sound', {})
            current_sound.update(settings)
            
            return self.update_settings('sound', 'all', current_sound)
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating sound settings: {str(e)}'
            }
    
    def get_status(self) -> dict:
        """모듈 상태 조회"""
        return {
            'settings_file': self.settings_file,
            'file_exists': os.path.exists(self.settings_file),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(self.settings_file)).isoformat() if os.path.exists(self.settings_file) else None,
            'sections': list(self.current_settings.keys())
        }

# 전역 인스턴스
settings_module = SettingsModule()
