#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
탐색원 로그 시스템 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from explorer_log_system import ExplorerMovementLogger

def test_explorer_log_system():
    """탐색원 로그 시스템 테스트"""
    print("🧪 탐색원 로그 시스템 테스트 시작")
    print("=" * 50)
    
    # 로거 초기화
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
    logger.log_movement_toggle(3, False)
    logger.log_all_movement_toggle(True)
    
    # 문제 진단 테스트
    test_explorers_data = [
        {
            "name": "Explorer-1", 
            "isMoving": True, 
            "x": 150, 
            "y": 150, 
            "targetX": 100, 
            "targetY": 100, 
            "circle": True, 
            "name": True, 
            "role": True
        },
        {
            "name": "Explorer-2", 
            "isMoving": False, 
            "x": 200, 
            "y": 200, 
            "targetX": 200, 
            "targetY": 200, 
            "circle": True, 
            "name": True, 
            "role": True
        }
    ]
    logger.diagnose_explorer_issues(test_explorers_data)
    
    # 강제 이동 테스트
    logger.force_move_test(test_explorers_data)
    
    # 이동 테스트
    logger.test_explorer_movement(test_explorers_data, 1086, 500)
    
    # 로그 통계 확인
    stats = logger.log_system.get_log_stats()
    print("\n📊 로그 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 탐색원 로그 시스템 테스트 완료")
    print(f"📄 로그 파일 위치: {logger.log_system.log_file}")

def test_log_file_operations():
    """로그 파일 작업 테스트"""
    print("\n🧪 로그 파일 작업 테스트")
    print("=" * 30)
    
    logger = ExplorerMovementLogger()
    
    # 로그 메시지 추가
    logger.log_system.add_log_message("테스트 메시지 1")
    logger.log_system.add_log_message("테스트 메시지 2")
    logger.log_system.add_log_message("테스트 메시지 3")
    
    # 로그 내용 확인
    content = logger.log_system.get_log_content()
    print(f"📄 로그 내용:\n{content}")
    
    # 로그 초기화
    logger.log_system.clear_log()
    
    print("✅ 로그 파일 작업 테스트 완료")

if __name__ == "__main__":
    try:
        test_explorer_log_system()
        test_log_file_operations()
        
        print("\n🎉 모든 테스트 완료!")
        print("📁 로그 파일을 확인하세요: E:/Gif/www/hankookin.center/8BIT/bot.v.0.1/log/explorer_movement.log")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
