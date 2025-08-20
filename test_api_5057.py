#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import json

def test_nb_wave_api():
    """N/B Wave API 테스트 (포트 5057)"""
    try:
        # API URL 구성
        base_url = "http://127.0.0.1:5057"
        endpoint = "/api/nb/wave-zones"
        params = {
            'interval': 'minute5',
            'count': '300',
            'window': '50'
        }
        
        # URL 인코딩
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}{endpoint}?{query_string}"
        
        print(f"🔍 테스트 URL: {url}")
        print("=" * 50)
        
        # API 호출
        with urllib.request.urlopen(url) as response:
            status_code = response.getcode()
            response_data = response.read().decode('utf-8')
            
            print(f"✅ 상태 코드: {status_code}")
            print(f"📄 응답 데이터:")
            print("-" * 30)
            
            # JSON 파싱 및 출력
            try:
                data = json.loads(response_data)
                print(f"성공 여부: {data.get('ok', False)}")
                print(f"현재 구역: {data.get('currentZone', 'N/A')}")
                print(f"현재 R값: {data.get('currentR', 0.0)}")
                print(f"ORANGE 구역 개수: {data.get('orangeCount', 0)}")
                print(f"BLUE 구역 개수: {data.get('blueCount', 0)}")
                print(f"총 구역 개수: {data.get('totalZones', 0)}")
                print(f"마지막 업데이트: {data.get('lastUpdate', 'N/A')}")
                
                # 샘플 데이터 출력
                orange_zones = data.get('orangeZones', [])
                blue_zones = data.get('blueZones', [])
                
                if orange_zones:
                    print(f"\n🟠 최근 ORANGE 구역 (최대 3개):")
                    for i, zone in enumerate(orange_zones[-3:], 1):
                        print(f"  {i}. 시간: {zone.get('time')}, 가격: {zone.get('close')}, R: {zone.get('r')}")
                
                if blue_zones:
                    print(f"\n🔵 최근 BLUE 구역 (최대 3개):")
                    for i, zone in enumerate(blue_zones[-3:], 1):
                        print(f"  {i}. 시간: {zone.get('time')}, 가격: {zone.get('close')}, R: {zone.get('r')}")
                
                print("\n✅ API 테스트 성공!")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"원본 응답: {response_data[:200]}...")
                
    except urllib.error.URLError as e:
        print(f"❌ 네트워크 오류: {e}")
        print("서버가 실행 중인지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

def test_other_apis():
    """다른 API들도 테스트 (포트 5057)"""
    apis_to_test = [
        "/api/nb/zone",
        "/api/nb/params",
        "/api/village/nb-zone-status"
    ]
    
    print("\n" + "=" * 50)
    print("🔍 다른 API 테스트 (포트 5057)")
    print("=" * 50)
    
    for api in apis_to_test:
        try:
            url = f"http://127.0.0.1:5057{api}"
            print(f"\n테스트 중: {api}")
            
            with urllib.request.urlopen(url) as response:
                status_code = response.getcode()
                print(f"  상태 코드: {status_code}")
                
                if status_code == 200:
                    print(f"  ✅ 성공")
                else:
                    print(f"  ⚠️ 예상과 다른 상태 코드")
                    
        except urllib.error.URLError as e:
            print(f"  ❌ 오류: {e}")
        except Exception as e:
            print(f"  ❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    print("🚀 N/B Wave API 테스트 시작 (포트 5057)")
    print("=" * 50)
    
    # 메인 API 테스트
    test_nb_wave_api()
    
    # 다른 API들 테스트
    test_other_apis()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print("=" * 50)
