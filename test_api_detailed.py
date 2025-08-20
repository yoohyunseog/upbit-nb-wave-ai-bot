#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import json

def test_nb_wave_api_detailed():
    """N/B Wave API 상세 테스트 - 응답 구조 확인"""
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
            print(f"📄 응답 데이터 구조:")
            print("-" * 30)
            
            # JSON 파싱 및 출력
            try:
                data = json.loads(response_data)
                
                # 기본 정보
                print(f"성공 여부: {data.get('ok', False)}")
                print(f"현재 구역: {data.get('currentZone', 'N/A')}")
                print(f"현재 R값: {data.get('currentR', 0.0)}")
                print(f"타임프레임: {data.get('interval', 'N/A')}")
                print(f"Window: {data.get('window', 'N/A')}")
                print(f"총 구역 개수: {data.get('totalZones', 0)}")
                print(f"ORANGE 구역 개수: {data.get('orangeCount', 0)}")
                print(f"BLUE 구역 개수: {data.get('blueCount', 0)}")
                print(f"마지막 업데이트: {data.get('lastUpdate', 'N/A')}")
                
                # 배열 데이터 확인
                print(f"\n📊 배열 데이터 확인:")
                print(f"  orangeZones 키 존재: {'orangeZones' in data}")
                print(f"  blueZones 키 존재: {'blueZones' in data}")
                print(f"  nbWaveColorArray 키 존재: {'nbWaveColorArray' in data}")
                
                orange_zones = data.get('orangeZones', [])
                blue_zones = data.get('blueZones', [])
                nb_wave_color_array = data.get('nbWaveColorArray', [])
                
                print(f"  orangeZones 길이: {len(orange_zones)}")
                print(f"  blueZones 길이: {len(blue_zones)}")
                print(f"  nbWaveColorArray 길이: {len(nb_wave_color_array)}")
                
                # 샘플 데이터 출력
                if orange_zones:
                    print(f"\n🟠 ORANGE 구역 샘플 (첫 3개):")
                    for i, zone in enumerate(orange_zones[:3], 1):
                        print(f"  {i}. {zone}")
                
                if blue_zones:
                    print(f"\n🔵 BLUE 구역 샘플 (첫 3개):")
                    for i, zone in enumerate(blue_zones[:3], 1):
                        print(f"  {i}. {zone}")
                
                if nb_wave_color_array:
                    print(f"\n🎨 N/B Wave Color Array 샘플 (첫 3개):")
                    for i, color_data in enumerate(nb_wave_color_array[:3], 1):
                        print(f"  {i}. {color_data}")
                
                # 모든 키 출력
                print(f"\n🔑 응답 데이터의 모든 키:")
                for key in data.keys():
                    value_type = type(data[key]).__name__
                    if isinstance(data[key], list):
                        print(f"  {key}: {value_type} (길이: {len(data[key])})")
                    else:
                        print(f"  {key}: {value_type}")
                
                print("\n✅ API 상세 테스트 성공!")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"원본 응답: {response_data[:500]}...")
                
    except urllib.error.URLError as e:
        print(f"❌ 네트워크 오류: {e}")
        print("서버가 실행 중인지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    print("🚀 N/B Wave API 상세 테스트")
    print("=" * 50)
    
    test_nb_wave_api_detailed()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print("=" * 50)
