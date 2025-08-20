import os
import sys
import time
import json
import random
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 게임 상태 데이터
game_state = {
    'village_energy': 150,  # 초기값을 150으로 증가
    'max_energy': 100,      # 표시용 최대값
    'energy_accumulated': 150,  # 누적 에너지 (무제한)
    'active_traders': 0,
    'village_reputation': '보통',
    'current_time': datetime.now().strftime('%H:%M:%S'),
    'characters': {
        'mayor': {
            'status': '지휘중',
            'hp': 100,
            'stamina': 100,
            'position': 'office',
            'level': 5.0,
            'profit': 0.0,
            'win_rate': 0.0,
            'role': 'N/B 길드 지점장',
            'authority': 100
        },
        'scout': {
            'status': '대기중',
            'hp': 85,
            'stamina': 70,
            'position': 'home',
            'level': 1.0,
            'profit': 0.0,
            'win_rate': 0.0,
            'role': '탐험가'
        },
        'guardian': {
            'status': '대기중',
            'hp': 95,
            'stamina': 80,
            'position': 'home',
            'level': 1.0,
            'profit': 0.0,
            'win_rate': 0.0,
            'role': '수호자'
        },
        'analyst': {
            'status': '대기중',
            'hp': 60,
            'stamina': 90,
            'position': 'home',
            'level': 1.0,
            'profit': 0.0,
            'win_rate': 0.0,
            'role': '전략가'
        },
        'elder': {
            'status': '대기중',
            'hp': 75,
            'stamina': 85,
            'position': 'home',
            'level': 1.0,
            'profit': 0.0,
            'win_rate': 0.0,
            'role': '어드바이저'
        }
    },
    'warehouse': {
        'mayor': {'coins': 0.0, 'entry_price': 0.0, 'ticks': 0},
        'scout': {'coins': 0.0, 'entry_price': 0.0, 'ticks': 0},
        'guardian': {'coins': 0.0, 'entry_price': 0.0, 'ticks': 0},
        'analyst': {'coins': 0.0, 'entry_price': 0.0, 'ticks': 0},
        'elder': {'coins': 0.0, 'entry_price': 0.0, 'ticks': 0}
    },
    'logs': [],
    'current_zone': 'BLUE',
    'current_price': 160000000,
    'mayor_announcements': [],
    'chart_changes': 0,
    'profitable_changes': 0
}

def add_log(message):
    """로그 추가"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'message': message
    }
    game_state['logs'].append(log_entry)
    if len(game_state['logs']) > 50:  # 최대 50개 로그 유지
        game_state['logs'] = game_state['logs'][-50:]

def simulate_village_activity():
    """마을 활동 시뮬레이션"""
    # 에너지 변화 - 무제한 누적
    energy_change = random.randint(-3, 8)  # 더 긍정적인 변화
    game_state['energy_accumulated'] = max(0, game_state['energy_accumulated'] + energy_change)
    game_state['village_energy'] = min(game_state['energy_accumulated'], 100)  # 표시용은 100% 제한
    
    # 차트 변경 추적
    if random.random() < 0.2:  # 20% 확률로 차트 변경
        game_state['chart_changes'] += 1
        if energy_change > 0:
            game_state['profitable_changes'] += 1
            add_log(f"📈 차트 변경으로 에너지 +{energy_change} (수익성 있는 변화)")
        else:
            add_log(f"📉 차트 변경으로 에너지 {energy_change} (손실 변화)")
    
    # 촌장의 지휘 활동
    mayor = game_state['characters']['mayor']
    if random.random() < 0.4:  # 40% 확률로 촌장이 지시
        zone_decision = random.choice(['BLUE', 'ORANGE'])
        strategy = random.choice(['BUY', 'SELL', 'HOLD'])
        
        announcement = f"🏛️ 촌장 지시: {zone_decision} 구역에서 {strategy} 전략 실행!"
        game_state['mayor_announcements'].append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'zone': zone_decision,
            'strategy': strategy,
            'message': announcement
        })
        
        add_log(announcement)
        
        # 촌장의 지시에 따라 구역 변경
        game_state['current_zone'] = zone_decision
    
    # 트레이너 캐릭터 활동 (촌장 제외)
    characters = ['scout', 'guardian', 'analyst', 'elder']
    random_char = random.choice(characters)
    
    if game_state['characters'][random_char]['status'] == '대기중' and game_state['energy_accumulated'] >= 10:
        # 거래 시작
        game_state['characters'][random_char]['status'] = '거래중'
        game_state['active_traders'] += 1
        game_state['energy_accumulated'] -= 10  # 누적 에너지에서 차감
        game_state['village_energy'] = min(game_state['energy_accumulated'], 100)  # 표시용 업데이트
        
        add_log(f"🤖 {random_char.upper()}가 거래를 시작했습니다. (에너지 소모: 10)")
        
        # 5초 후 거래 종료 시뮬레이션
        def end_trading():
            game_state['characters'][random_char]['status'] = '대기중'
            game_state['active_traders'] -= 1
            
            # 거래 결과 시뮬레이션
            profit = random.uniform(-2.0, 3.0)
            game_state['characters'][random_char]['profit'] += profit
            
            if profit > 0:
                add_log(f"✅ {random_char.upper()}의 거래 성공! +{profit:.2f}%")
            else:
                add_log(f"❌ {random_char.upper()}의 거래 실패! {profit:.2f}%")
        
        # 실제로는 타이머를 사용해야 하지만, 여기서는 간단히 처리
        import threading
        timer = threading.Timer(5.0, end_trading)
        timer.start()
    
    # 가격 변화
    price_change = random.uniform(-1000000, 1000000)
    game_state['current_price'] = max(100000000, game_state['current_price'] + price_change)
    
    # 구역 변화 (촌장 지시가 없을 때만)
    if random.random() < 0.2:  # 20% 확률로 구역 변경
        game_state['current_zone'] = random.choice(['BLUE', 'ORANGE'])
        add_log(f"🔄 구역 변경: {game_state['current_zone']}")

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'village.html')

@app.route('/api/game/state')
def get_game_state():
    """게임 상태 조회"""
    game_state['current_time'] = datetime.now().strftime('%H:%M:%S')
    return jsonify({
        'ok': True,
        'state': game_state
    })

@app.route('/api/game/energy/fill', methods=['POST'])
def fill_energy():
    """에너지 100% 채우기"""
    old_energy = game_state['energy_accumulated']
    game_state['energy_accumulated'] += 100  # 100 추가
    game_state['village_energy'] = min(game_state['energy_accumulated'], 100)
    add_log(f"⚡ 마을 에너지 100 추가: {old_energy} → {game_state['energy_accumulated']}")
    return jsonify({
        'ok': True,
        'previous_energy': old_energy,
        'new_energy': game_state['energy_accumulated']
    })

@app.route('/api/game/character/<char_name>/action', methods=['POST'])
def character_action(char_name):
    """캐릭터 액션"""
    if char_name not in game_state['characters']:
        return jsonify({'ok': False, 'error': 'Invalid character'}), 400
    
    char = game_state['characters'][char_name]
    
    if char['status'] == '대기중' and game_state['energy_accumulated'] >= 10:
        char['status'] = '거래중'
        game_state['active_traders'] += 1
        game_state['energy_accumulated'] -= 10
        game_state['village_energy'] = min(game_state['energy_accumulated'], 100)
        add_log(f"🎯 {char_name.upper()} 수동 거래 시작!")
        
        return jsonify({
            'ok': True,
            'character': char_name,
            'action': 'start_trading'
        })
    
    return jsonify({
        'ok': False,
        'error': 'Cannot start trading'
    }), 400

@app.route('/api/game/warehouse/<char_name>/modify', methods=['POST'])
def modify_warehouse(char_name):
    """창고 수정"""
    if char_name not in game_state['warehouse']:
        return jsonify({'ok': False, 'error': 'Invalid character'}), 400
    
    import flask
    data = flask.request.get_json(force=True)
    delta = data.get('delta', 0)
    
    warehouse = game_state['warehouse'][char_name]
    old_coins = warehouse['coins']
    warehouse['coins'] = max(0, warehouse['coins'] + delta)
    
    if delta > 0:
        add_log(f"📦 {char_name.upper()} 창고에 {delta:.8f} BTC 추가")
    else:
        add_log(f"📦 {char_name.upper()} 창고에서 {abs(delta):.8f} BTC 제거")
    
    return jsonify({
        'ok': True,
        'character': char_name,
        'old_coins': old_coins,
        'new_coins': warehouse['coins']
    })

@app.route('/api/game/logs')
def get_logs():
    """로그 조회"""
    return jsonify({
        'ok': True,
        'logs': game_state['logs']
    })

@app.route('/api/game/mayor/announce', methods=['POST'])
def mayor_announce():
    """촌장 공지사항"""
    import flask
    data = flask.request.get_json(force=True)
    zone = data.get('zone', 'BLUE')
    strategy = data.get('strategy', 'HOLD')
    
    announcement = f"🏛️ 촌장 지시: {zone} 구역에서 {strategy} 전략 실행!"
    game_state['mayor_announcements'].append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'zone': zone,
        'strategy': strategy,
        'message': announcement
    })
    
    game_state['current_zone'] = zone
    add_log(announcement)
    
    return jsonify({
        'ok': True,
        'announcement': announcement,
        'zone': zone,
        'strategy': strategy
    })

@app.route('/api/game/mayor/announcements')
def get_mayor_announcements():
    """촌장 공지사항 조회"""
    return jsonify({
        'ok': True,
        'announcements': game_state['mayor_announcements']
    })

@app.route('/api/game/mayor/command', methods=['POST'])
def mayor_command():
    """촌장 명령"""
    import flask
    data = flask.request.get_json(force=True)
    command = data.get('command', '')
    target = data.get('target', 'all')
    
    if command == 'start_trading':
        # 모든 트레이너에게 거래 시작 명령
        if target == 'all':
            for char in ['scout', 'guardian', 'analyst', 'elder']:
                if game_state['characters'][char]['status'] == '대기중' and game_state['energy_accumulated'] >= 10:
                    game_state['characters'][char]['status'] = '거래중'
                    game_state['active_traders'] += 1
                    game_state['energy_accumulated'] -= 10
                    game_state['village_energy'] = min(game_state['energy_accumulated'], 100)
                    add_log(f"🏛️ 촌장 명령: {char.upper()} 거래 시작!")
        else:
            if target in game_state['characters'] and game_state['characters'][target]['status'] == '대기중' and game_state['energy_accumulated'] >= 10:
                game_state['characters'][target]['status'] = '거래중'
                game_state['active_traders'] += 1
                game_state['energy_accumulated'] -= 10
                game_state['village_energy'] = min(game_state['energy_accumulated'], 100)
                add_log(f"🏛️ 촌장 명령: {target.upper()} 거래 시작!")
    
    elif command == 'stop_trading':
        # 모든 거래 중단
        for char in game_state['characters']:
            if game_state['characters'][char]['status'] == '거래중':
                game_state['characters'][char]['status'] = '대기중'
                game_state['active_traders'] -= 1
        add_log("🏛️ 촌장 명령: 모든 거래 중단!")
    
    elif command == 'fill_energy':
        # 에너지 100 추가
        old_energy = game_state['energy_accumulated']
        game_state['energy_accumulated'] += 100
        game_state['village_energy'] = min(game_state['energy_accumulated'], 100)
        add_log(f"🏛️ 촌장 명령: 마을 에너지 100 추가! ({old_energy} → {game_state['energy_accumulated']})")
    
    return jsonify({
        'ok': True,
        'command': command,
        'target': target
    })

def start_simulation():
    """시뮬레이션 시작"""
    add_log("🎮 8BIT 마을 시뮬레이터가 시작되었습니다!")
    add_log("🏘️ 주민들이 각자의 역할을 수행합니다.")
    
    # 주기적으로 마을 활동 시뮬레이션
    def simulation_loop():
        while True:
            time.sleep(3)  # 3초마다
            simulate_village_activity()
    
    import threading
    simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
    simulation_thread.start()

if __name__ == '__main__':
    print("🎮 8BIT 마을 시뮬레이터 서버 시작...")
    print("📍 접속 주소: http://localhost:5001")
    print("🎯 API 엔드포인트: http://localhost:5001/api/game/state")
    
    start_simulation()
    
    app.run(host='0.0.0.0', port=5001, debug=True)
