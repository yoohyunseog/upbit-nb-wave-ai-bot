#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import json
import time

def test_nb_wave_api_error():
    """N/B Wave API 500 에러 상세 분석"""
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
            print(response_data)
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP 에러: {e.code} - {e.reason}")
        print(f"📄 에러 응답:")
        error_data = e.read().decode('utf-8')
        print(error_data)
        
        # JSON 파싱 시도
        try:
            error_json = json.loads(error_data)
            print(f"\n🔍 에러 상세:")
            print(f"  성공 여부: {error_json.get('ok', False)}")
            print(f"  에러 메시지: {error_json.get('error', 'N/A')}")
        except:
            print("JSON 파싱 실패")
            
    except urllib.error.URLError as e:
        print(f"❌ 네트워크 오류: {e}")
        print("서버가 실행 중인지 확인해주세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    print("🚀 N/B Wave API 500 에러 분석")
    print("=" * 50)
    
    # 서버 시작 대기
    print("서버 시작 대기 중...")
    time.sleep(3)
    
    test_nb_wave_api_error()
    
    print("\n" + "=" * 50)
    print("🏁 테스트 완료")
    print("=" * 50)
