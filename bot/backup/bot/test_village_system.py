#!/usr/bin/env python3
"""
8BIT 마을 시스템 테스트 스크립트
"""

import requests
import json
import time

# 서버 URL
BASE_URL = "http://localhost:5057"

def test_village_system():
    """8BIT 마을 시스템 테스트"""
    print("🏘️ 8BIT 마을 시스템 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 마을 상태 조회
        print("1. 마을 상태 조회...")
        response = requests.get(f"{BASE_URL}/api/village/status")
        if response.status_code == 200:
            village_status = response.json()
            print(f"✅ 마을 이름: {village_status['village_name']}")
            print(f"✅ 촌장: {village_status['mayor']}")
            print(f"✅ 마을 에너지: {village_status['village_energy']}")
            print(f"✅ 주민 수: {village_status['residents_count']}")
        else:
            print(f"❌ 마을 상태 조회 실패: {response.status_code}")
        
        print()
        
        # 2. 촌장 지침 조회
        print("2. 촌장 지침 조회...")
        response = requests.get(f"{BASE_URL}/api/village/mayor/guidance")
        if response.status_code == 200:
            guidance = response.json()
            print(f"✅ 지침 시간: {guidance['timestamp']}")
            print(f"✅ ML 모델 신뢰도: {guidance['trust_analysis']['ml_model_trust']}%")
            print(f"✅ N/B 길드 신뢰도: {guidance['trust_analysis']['nb_guild_trust']}%")
            print(f"✅ 공식 전략: {guidance['guidance']['official_strategy']}")
        else:
            print(f"❌ 촌장 지침 조회 실패: {response.status_code}")
        
        print()
        
        # 3. 마을 주민 조회
        print("3. 마을 주민 조회...")
        response = requests.get(f"{BASE_URL}/api/village/residents")
        if response.status_code == 200:
            residents = response.json()
            print(f"✅ 총 주민 수: {residents['total_count']}")
            for name, data in residents['residents'].items():
                print(f"   - {data['name']} ({data['role']}): {data['location']}")
        else:
            print(f"❌ 마을 주민 조회 실패: {response.status_code}")
        
        print()
        
        # 4. Scout 상태 조회
        print("4. Scout 상태 조회...")
        response = requests.get(f"{BASE_URL}/api/village/scout/status")
        if response.status_code == 200:
            scout_status = response.json()
            print(f"✅ 이름: {scout_status['status']['name']}")
            print(f"✅ 역할: {scout_status['status']['role']}")
            print(f"✅ 위치: {scout_status['status']['location']}")
            print(f"✅ 스킬 레벨: {scout_status['status']['skillLevel']}")
            print(f"✅ 현재 포지션: {scout_status['current_position']['pnl']}")
        else:
            print(f"❌ Scout 상태 조회 실패: {response.status_code}")
        
        print()
        
        # 5. 신뢰도 계산 테스트
        print("5. 신뢰도 계산 테스트...")
        test_data = {
            "personal_confidence": 100,
            "ml_trust": 40,
            "nb_guild_trust": 85
        }
        response = requests.post(f"{BASE_URL}/api/village/trust/calculate", json=test_data)
        if response.status_code == 200:
            trust_result = response.json()
            print(f"✅ 개인 확신: {trust_result['personal_confidence']}%")
            print(f"✅ ML 모델 신뢰도: {trust_result['ml_trust']}%")
            print(f"✅ N/B 길드 신뢰도: {trust_result['nb_guild_trust']}%")
            print(f"✅ 가중 평균: {trust_result['weighted_confidence']:.1f}%")
        else:
            print(f"❌ 신뢰도 계산 실패: {response.status_code}")
        
        print()
        
        # 6. 시스템 개요 조회
        print("6. 시스템 개요 조회...")
        response = requests.get(f"{BASE_URL}/api/village/system/overview")
        if response.status_code == 200:
            overview = response.json()
            print(f"✅ 시스템 이름: {overview['system_name']}")
            print(f"✅ 설명: {overview['description']}")
            print(f"✅ 활성 창고: {overview['current_status']['warehouses_active']}")
        else:
            print(f"❌ 시스템 개요 조회 실패: {response.status_code}")
        
        print()
        
        # 7. 거래 일지 시스템 테스트
        print("7. 거래 일지 시스템 테스트...")
        
        # Scout의 거래 일지 조회
        response = requests.get(f"{BASE_URL}/api/village/journal/scout/recent")
        if response.status_code == 200:
            journal = response.json()
            print(f"✅ Scout 최근 일지: {journal.get('count', 0)}개 항목")
        else:
            print(f"❌ Scout 일지 조회 실패: {response.status_code}")
        
        # 촌장 지침 일지 조회
        response = requests.get(f"{BASE_URL}/api/village/journal/scout/mayor-guidance")
        if response.status_code == 200:
            mayor_journal = response.json()
            print(f"✅ Scout 촌장 지침 일지: {mayor_journal.get('count', 0)}개 항목")
        else:
            print(f"❌ Scout 촌장 지침 일지 조회 실패: {response.status_code}")
        
        # ML 모델 판단 일지 조회
        response = requests.get(f"{BASE_URL}/api/village/journal/scout/ml-decisions")
        if response.status_code == 200:
            ml_journal = response.json()
            print(f"✅ Scout ML 모델 판단 일지: {ml_journal.get('count', 0)}개 항목")
        else:
            print(f"❌ Scout ML 모델 판단 일지 조회 실패: {response.status_code}")
        
        # 일지 요약 조회
        response = requests.get(f"{BASE_URL}/api/village/journal/summary")
        if response.status_code == 200:
            summary = response.json()
            print(f"✅ 전체 일지 요약: {summary.get('total_trainers', 0)}명의 트레이너")
            for trainer, data in summary.get('journal_summary', {}).items():
                print(f"   - {trainer}: 최근 {data.get('recent_entries', 0)}개, 촌장 지침 {data.get('mayor_guidance_entries', 0)}개")
        else:
            print(f"❌ 일지 요약 조회 실패: {response.status_code}")
        
        print()
        print("🎉 8BIT 마을 시스템 테스트 완료!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    test_village_system()
