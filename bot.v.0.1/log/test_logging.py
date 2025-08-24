import os
import json
import datetime
import time
import urllib.request


BASE = os.path.dirname(os.path.abspath(__file__))
LEFT_LOG = os.path.join(BASE, 'left_panel.log')
SNAP_DIR = os.path.join(BASE, 'snapshots')
def day_path_from_ts(ts_ms: int) -> str:
    dt = datetime.datetime.fromtimestamp(int(ts_ms)/1000.0)
    return os.path.join(SNAP_DIR, dt.strftime('%Y-%m-%d') + '.log')


def post_json(url, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def tail_file(path, n=3):
    if not os.path.exists(path):
        return f'file not found: {path}'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return ''.join(lines[-n:]).rstrip('\n')
    except Exception as e:
        return f'error reading {path}: {e}'


def main():
    # Send status log
    try:
        r1 = post_json('http://127.0.0.1:5060/log', {
            'tf': '1m',
            'text': '테스트 로그',
            'ts': int(time.time()*1000),
            'mode': 'paper',
            'type': 'status'
        })
        print('status-log response:', r1)
    except Exception as e:
        print('status-log error:', e)

    # Show status log tail
    print('--- left_panel.log tail ---')
    print(tail_file(LEFT_LOG, 5))

    # Send snapshot
    try:
        ts_now = int(time.time()*1000)
        r2 = post_json('http://127.0.0.1:5061/snapshot', {
            'ts': ts_now,
            'currentTimeframe': '1m',
            'cards': []
        })
        print('snapshot response:', r2)
    except Exception as e:
        print('snapshot error:', e)

    # Show snapshot tail
    print('--- snapshot tail ---')
    snap_file = day_path_from_ts(ts_now)
    print(tail_file(snap_file, 5))


if __name__ == '__main__':
    main()


