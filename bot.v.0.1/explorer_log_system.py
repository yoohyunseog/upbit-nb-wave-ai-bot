#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
탐색원 이동 시스템 로그 관리자
탐색원들의 이동 상태와 문제 진단을 위한 로그 시스템
"""

import os
import json
import datetime
from typing import List, Dict, Any
from pathlib import Path

class ExplorerLogSystem:
    def __init__(self, max_log_messages: int = 100):
        """
        탐색원 로그 시스템 초기화
        
        Args:
            max_log_messages (int): 최대 로그 메시지 수 (기본값: 100)
        """
        self.log_messages = []
        self.max_log_messages = max_log_messages
        self.log_dir = Path("E:/Gif/www/hankookin.center/8BIT/bot.v.0.1/log")
        self.log_file = self.log_dir / "explorer_movement.log"
        
        # 로그 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 탐색원 로그 시스템 초기화 완료")
        print(f"📁 로그 디렉토리: {self.log_dir}")
        print(f"📄 로그 파일: {self.log_file}")
    
    def add_log_message(self, message: str) -> None:
        """
        로그 메시지 추가
        
        Args:
            message (str): 로그 메시지
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_messages.append(log_entry)
        
        # 최대 로그 메시지 수 제한
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop(0)
        
        # 로그 파일에 저장
        self.save_log_to_file()
        
        # 콘솔에도 출력
        print(log_entry)
    
    def save_log_to_file(self) -> None:
        """로그 파일에 저장"""
        try:
            log_content = "\n".join(self.log_messages)
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            print(f"💾 로그 파일 저장 완료: {self.log_file}")
            
        except Exception as e:
            print(f"❌ 로그 파일 저장 실패: {e}")
    
    def load_log_from_file(self) -> None:
        """로그 파일에서 읽기"""
        try:
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.log_messages = [line.strip() for line in content.split('\n') if line.strip()]
                print(f"📖 로그 파일 읽기 완료: {len(self.log_messages)}개 메시지")
            else:
                print(f"📄 로그 파일이 존재하지 않습니다: {self.log_file}")
                
        except Exception as e:
            print(f"❌ 로그 파일 읽기 실패: {e}")
    
    def get_log_content(self) -> str:
        """로그 내용 가져오기"""
        return "\n".join(self.log_messages)
    
    def clear_log(self) -> None:
        """로그 초기화"""
        self.log_messages = []
        self.save_log_to_file()
        self.add_log_message("🧹 로그 초기화 완료")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """로그 통계 정보"""
        return {
            "total_messages": len(self.log_messages),
            "max_messages": self.max_log_messages,
            "log_file": str(self.log_file),
            "log_dir": str(self.log_dir),
            "file_exists": self.log_file.exists(),
            "file_size": self.log_file.stat().st_size if self.log_file.exists() else 0
        }

class ExplorerMovementLogger:
    """탐색원 이동 전용 로거"""
    
    def __init__(self):
        self.log_system = ExplorerLogSystem()
        self.explorers = []
        self.is_initialized = False
    
    def initialize_system(self, scene_info: str = "", config_info: str = "") -> None:
        """시스템 초기화 로그"""
        if self.is_initialized:
            self.log_system.add_log_message("🔍 탐색원 이동 시스템: 이미 초기화됨")
            return
        
        self.is_initialized = True
        self.log_system.add_log_message(f"🔍 탐색원 이동 시스템 초기화 완료 (scene: {bool(scene_info)}, config: {bool(config_info)})")
    
    def register_explorers(self, explorer_models: List[Dict]) -> None:
        """탐색원 등록 로그"""
        if not explorer_models:
            self.log_system.add_log_message("❌ 탐색원 등록 실패: 탐색자 데이터가 없습니다.")
            return
        
        self.log_system.add_log_message(f"🔍 탐색원 등록 시작: {len(explorer_models)}개의 탐색자 모델")
        
        for i, explorer in enumerate(explorer_models):
            name = explorer.get('name', 'Unknown')
            target_x = explorer.get('targetX', 0)
            target_y = explorer.get('targetY', 0)
            
            self.log_system.add_log_message(f"🔍 탐색원 {i} 등록: {name} - 위치({target_x:.0f}, {target_y:.0f})")
        
        self.explorers = explorer_models
        self.log_system.add_log_message(f"🔍 {len(self.explorers)}명의 탐색원이 이동 시스템에 등록되었습니다.")
    
    def log_explorer_update(self, index: int, is_moving: bool, x: float, y: float) -> None:
        """탐색원 업데이트 로그"""
        status = "이동중" if is_moving else "정지"
        self.log_system.add_log_message(f"🔍 탐색원 {index} 업데이트: 상태={status}, 위치=({x:.0f}, {y:.0f})")
    
    def log_explorer_movement(self, index: int, x: float, y: float, remaining_distance: float) -> None:
        """탐색원 이동 로그"""
        self.log_system.add_log_message(f"🚶 탐색원 {index} 이동 중: ({x:.0f}, {y:.0f}) | 남은거리: {remaining_distance:.0f}px")
    
    def log_target_arrival(self, index: int, x: float, y: float, discovered_count: int) -> None:
        """목표 도달 로그"""
        self.log_system.add_log_message(f"🎯 탐색원 {index}: 새로운 좌표 발견! ({x:.0f}, {y:.0f}) - 총 발견: {discovered_count}개")
    
    def log_new_target(self, index: int, target_x: float, target_y: float) -> None:
        """새로운 목표 설정 로그"""
        self.log_system.add_log_message(f"🎯 탐색원 {index}: 새로운 목표 설정 ({target_x:.0f}, {target_y:.0f})")
    
    def log_movement_toggle(self, index: int, is_moving: bool) -> None:
        """이동 상태 변경 로그"""
        status = "재개" if is_moving else "일시정지"
        self.log_system.add_log_message(f"⏸️ 탐색원 {index} 이동 {status}")
    
    def log_all_movement_toggle(self, is_moving: bool) -> None:
        """모든 탐색원 이동 상태 변경 로그"""
        status = "재개" if is_moving else "일시정지"
        self.log_system.add_log_message(f"⏸️ 모든 탐색원 이동 {status}")
    
    def diagnose_explorer_issues(self, explorers_data: List[Dict]) -> None:
        """탐색원 문제 진단"""
        self.log_system.add_log_message("🔍 탐색원 문제 진단 시작")
        
        # 시스템 초기화 상태 확인
        self.log_system.add_log_message(f"🔍 시스템 초기화 상태: {self.is_initialized}")
        
        # 탐색원 등록 상태 확인
        self.log_system.add_log_message(f"🔍 등록된 탐색원 수: {len(explorers_data)}")
        
        # 각 탐색원의 상태 확인
        for i, explorer in enumerate(explorers_data):
            self.log_system.add_log_message(f"🔍 탐색원 {i} 상태:")
            self.log_system.add_log_message(f"  - 이름: {explorer.get('name', 'Unknown')}")
            self.log_system.add_log_message(f"  - 이동 상태: {'이동중' if explorer.get('isMoving', False) else '정지'}")
            self.log_system.add_log_message(f"  - 현재 위치: ({explorer.get('x', 0):.0f}, {explorer.get('y', 0):.0f})")
            self.log_system.add_log_message(f"  - 목표 위치: ({explorer.get('targetX', 0):.0f}, {explorer.get('targetY', 0):.0f})")
            self.log_system.add_log_message(f"  - 원 객체 존재: {bool(explorer.get('circle'))}")
            self.log_system.add_log_message(f"  - 이름 객체 존재: {bool(explorer.get('name'))}")
            self.log_system.add_log_message(f"  - 역할 객체 존재: {bool(explorer.get('role'))}")
        
        # 시스템 설정 확인
        self.log_system.add_log_message("🔍 시스템 설정:")
        self.log_system.add_log_message(f"  - 이동 속도: {0.1}")
        self.log_system.add_log_message(f"  - 도착 임계값: {25}")
        self.log_system.add_log_message(f"  - 탐색 범위: {15}")
        
        self.log_system.add_log_message("🔍 탐색원 문제 진단 완료")
    
    def force_move_test(self, explorers_data: List[Dict]) -> None:
        """강제 이동 테스트"""
        self.log_system.add_log_message("🧪 강제 이동 테스트 시작")
        
        for i, explorer in enumerate(explorers_data):
            if explorer.get('circle'):
                # 현재 위치에서 50픽셀 이동
                new_x = explorer.get('x', 0) + 50
                new_y = explorer.get('y', 0) + 50
                
                self.log_system.add_log_message(f"🧪 탐색원 {i} 강제 이동: ({new_x:.0f}, {new_y:.0f})")
        
        self.log_system.add_log_message("🧪 강제 이동 테스트 완료")
    
    def test_explorer_movement(self, explorers_data: List[Dict], config_width: int, config_height: int) -> None:
        """탐색원 이동 테스트"""
        self.log_system.add_log_message("🧪 탐색원 이동 테스트 시작")
        
        for i, explorer in enumerate(explorers_data):
            # 새로운 목표 설정
            margin = 40
            new_target_x = (config_width - 2 * margin) * 0.5 + margin  # 중앙 근처
            new_target_y = (config_height - 2 * margin) * 0.5 + margin
            
            self.log_system.add_log_message(f"🧪 탐색원 {i} 테스트 목표 설정: ({new_target_x:.0f}, {new_target_y:.0f})")
        
        self.log_system.add_log_message("🧪 탐색원 이동 테스트 완료")

# Flask API 엔드포인트 (기존 server.py에 추가할 수 있는 코드)
def create_explorer_log_routes(app):
    """탐색원 로그 관련 Flask 라우트 생성"""
    
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

# 사용 예시
if __name__ == "__main__":
    # 로그 시스템 테스트
    logger = ExplorerMovementLogger()
    
    # 시스템 초기화
    logger.initialize_system("test_scene", "test_config")
    
    # 탐색원 등록 테스트
    test_explorers = [
        {"name": "Explorer-1", "targetX": 100, "targetY": 100},
        {"name": "Explorer-2", "targetX": 200, "targetY": 200},
        {"name": "Explorer-3", "targetX": 300, "targetY": 300},
        {"name": "Explorer-4", "targetX": 400, "targetY": 400}
    ]
    logger.register_explorers(test_explorers)
    
    # 이동 로그 테스트
    logger.log_explorer_movement(0, 150, 150, 25.5)
    logger.log_target_arrival(1, 200, 200, 3)
    logger.log_new_target(2, 350, 350)
    
    # 문제 진단 테스트
    test_explorers_data = [
        {"name": "Explorer-1", "isMoving": True, "x": 150, "y": 150, "targetX": 100, "targetY": 100, "circle": True, "name": True, "role": True},
        {"name": "Explorer-2", "isMoving": False, "x": 200, "y": 200, "targetX": 200, "targetY": 200, "circle": True, "name": True, "role": True}
    ]
    logger.diagnose_explorer_issues(test_explorers_data)
    
    print("✅ 탐색원 로그 시스템 테스트 완료")
