import requests
import json

def test_save_log_file():
    """save-log-file 엔드포인트 테스트"""
    
    # 테스트 데이터 (100줄을 초과하는 데이터로 테스트)
    test_lines = []
    for i in range(110):  # 110줄 생성 (100줄 초과)
        test_lines.append(f'[14:30:{i:02d}] 매도 전 예상 수익률 계산: {2.5 + i*0.1:.1f}% (테스트용 {i+1})\n')
    
    test_content = ''.join(test_lines)
    
    test_data = {
        'filename': 'log/trainer/sell-profit-rate-logs-2025-08-22.txt',
        'content': test_content,
        'logCount': len(test_lines)
    }
    
    try:
        # POST 요청 전송
        response = requests.post(
            'http://127.0.0.1:5057/save-log-file',
            headers={'Content-Type': 'application/json'},
            data=json.dumps(test_data)
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 로그 파일 저장 성공!")
                print(f"저장된 파일 경로: {result.get('filepath')}")
            else:
                print(f"❌ 로그 파일 저장 실패: {result.get('error')}")
        else:
            print(f"❌ HTTP 요청 실패: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    print("🧪 save-log-file 엔드포인트 테스트 시작...")
    test_save_log_file()
