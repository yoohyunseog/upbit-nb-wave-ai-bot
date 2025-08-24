import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS


APP = Flask(__name__)
CORS(APP)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'snapshots')
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
MAX_LINES = 100


def _day_path(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(int(ts_ms)/1000.0)
    return os.path.join(SNAPSHOT_DIR, dt.strftime('%Y-%m-%d') + '.log')


@APP.route('/snapshot', methods=['POST'])
def api_snapshot():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        ts = int(payload.get('ts') or 0) or int(datetime.utcnow().timestamp() * 1000)
        record = {
            'ts': ts,
            'currentTimeframe': payload.get('currentTimeframe'),
            'selected': payload.get('selected'),
            'cards': payload.get('cards') or [],
        }
        # write append to day file
        path = _day_path(ts)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        # enforce last MAX_LINES
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > MAX_LINES:
                keep = lines[-MAX_LINES:]
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(keep)
        except Exception:
            pass
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/snapshot', methods=['GET'])
def api_snapshot_get():
    try:
        day = request.args.get('day')  # YYYY-MM-DD
        if not day:
            day = datetime.utcnow().strftime('%Y-%m-%d')
        path = os.path.join(SNAPSHOT_DIR, f'{day}.log')
        if not os.path.exists(path):
            return jsonify({'ok': True, 'lines': []})
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f.readlines()[-MAX_LINES:]]
        return jsonify({'ok': True, 'lines': lines})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # default port 5061
    port = int(os.environ.get('LEFT_PANEL_SNAPSHOT_PORT', '5061'))
    APP.run(host='127.0.0.1', port=port)


