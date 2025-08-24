#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
import datetime

class LogFileHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/save-log-file':
            self.handle_save_log_file()
        else:
            self.send_error(404, "Not Found")
    
    def handle_save_log_file(self):
        try:
            # 요청 본문 읽기
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # JSON 파싱
            data = json.loads(post_data.decode('utf-8'))
            filename = data.get('filename')
            content = data.get('content')
            log_count = data.get('logCount', 0)
            
            if not filename or not content:
                self.send_error(400, "Missing filename or content")
                return
            
            # 파일 경로 생성
            filepath = os.path.join(os.getcwd(), filename)
            
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 파일에 내용 쓰기
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 응답
            response = {
                'success': True,
                'filepath': filepath,
                'logCount': log_count,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
            print(f"💾 로그 파일 저장 완료: {filepath} ({log_count}개 로그)")
            
        except Exception as e:
            print(f"❌ 로그 파일 저장 실패: {str(e)}")
            response = {
                'success': False,
                'error': str(e)
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
    
    def do_OPTIONS(self):
        # CORS preflight 요청 처리
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server(port=8000):
    with socketserver.TCPServer(("", port), LogFileHandler) as httpd:
        print(f"🚀 로그 파일 저장 서버 시작: http://localhost:{port}")
        print(f"📁 현재 디렉토리: {os.getcwd()}")
        print("💾 /save-log-file 엔드포인트로 로그 파일 저장 가능")
        print("🛑 서버 중지: Ctrl+C")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 서버 중지됨")

if __name__ == "__main__":
    run_server(8000)
