import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS


APP = Flask(__name__)
CORS(APP)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'left_panel.log')
MAX_LINES = 100


def ensure_file():
    try:
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w', encoding='utf-8') as f:
                f.write('')
    except Exception:
        pass


def tail_write_limit(line: str):
    ensure_file()
    try:
        # Append line
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line.rstrip('\n') + '\n')
        # Enforce last MAX_LINES
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            keep = lines[-MAX_LINES:]
            with open(LOG_PATH, 'w', encoding='utf-8') as f:
                f.writelines(keep)
    except Exception as e:
        raise e


@APP.route('/log', methods=['POST'])
def api_log():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        # expected: { tf, text, ts, mode, type }
        record = {
            'tf': payload.get('tf'),
            'text': payload.get('text'),
            'ts': int(payload.get('ts') or 0),
            'mode': payload.get('mode'),
            'type': payload.get('type') or 'status'
        }
        tail_write_limit(json.dumps(record, ensure_ascii=False))
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/log', methods=['GET'])
def api_log_get():
    try:
        ensure_file()
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            data = [line.rstrip('\n') for line in f.readlines()[-MAX_LINES:]]
        return jsonify({'ok': True, 'lines': data})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Run on 5060 by default
    port = int(os.environ.get('LEFT_PANEL_LOG_PORT', '5060'))
    APP.run(host='127.0.0.1', port=port)


