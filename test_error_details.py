#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import json

def test_nb_wave_api_with_error_details():
    """N/B Wave API 테스트 - 오류 상세 정보 확인"""
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
            print(response_data)
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP 오류: {e.code} - {e.reason}")
        print(f"📄 오류 응답:")
        print("-" * 30)
        error_data = e.read().decode('utf-8')
        print(error_data)
        
        # JSON 파싱 시도
        try:
            error_json = json.loads(error_data)
            print(f"\n🔍 오류 상세 정보:")
            print(f"  오류 메시지: {error_json.get('error', 'N/A')}")
            print(f"  성공 여부: {error_json.get('ok', 'N/A')}")
        except:
            print("JSON 파싱 실패")
            
    except urllib.error.URLError as e:
        print(f"❌ 네트워크 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    print("🚀 N/B Wave API 오류 상세 테스트")
    print("=" * 50)
    
    test_nb_wave_api_with_error_details()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print("=" * 50)
