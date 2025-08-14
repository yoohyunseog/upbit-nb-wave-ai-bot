import os
import threading
import time
from collections import deque
from dataclasses import asdict
from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
import json
import pyupbit
import pandas as pd
import numpy as np
import joblib
import uuid
import requests

from main import load_config, get_candles
from dotenv import load_dotenv
from strategy import decide_signal
from trade import Trader, TradeConfig

app = Flask(__name__)
CORS(app)

@app.route("/")
def root():
    return (
        """
        <html>
          <head><meta charset="utf-8"><title>8BIT Bot Server</title></head>
          <body style="font-family:system-ui,Segoe UI,Arial; padding:20px;">
            <h2>8BIT PyUpbit Bot Server</h2>
            <p>Server is running. API endpoint: <a href="/api/state">/api/state</a></p>
            <div id="s">Loading...</div>
            <script>
              fetch('/api/state').then(r=>r.json()).then(j=>{
                document.getElementById('s').textContent = JSON.stringify(j);
              }).catch(()=>{ document.getElementById('s').textContent='Failed to load /api/state'; });
            </script>
          </body>
        </html>
        """,
        200,
        {"Content-Type": "text/html; charset=utf-8"}
    )


@app.route("/ui")
def serve_ui():
    # Serve the embedded chart UI from bot/static/ui.html
    return send_from_directory('static', 'ui.html')

@app.route('/static/<path:filename>')
def serve_static(filename: str):
    return send_from_directory('static', filename)

state = {
    "price": 0.0,
    "signal": "HOLD",
    "ema_fast": 10,
    "ema_slow": 30,
    "market": "KRW-BTC",
    "candle": "minute10",
    "history": deque(maxlen=200),  # (ts, price)
}

# ML training state
ml_state = {
    'train_count': 0,
}

# Grouped NB observations (time-bucketed)
GROUP_BUCKET_SEC = int(os.getenv('NB_GROUP_BUCKET_SEC', '60'))  # group by 1m default
GROUP_MIN_SIZE = int(os.getenv('NB_GROUP_MIN_SIZE', '25'))
_nb_groups: dict[int, list] = {}

def _bucket_ts(ts_ms: int | None = None, bucket_sec: int | None = None) -> int:
    try:
        b = int(bucket_sec or GROUP_BUCKET_SEC)
        t = int((ts_ms or int(time.time()*1000)) / 1000)
        return (t // b) * b
    except Exception:
        return int(time.time())

def _record_group_observation(interval: str, window: int, r_val: float,
                              pct_blue: float, pct_orange: float, ts_ms: int | None = None):
    try:
        bt = _bucket_ts(ts_ms, GROUP_BUCKET_SEC)
        row = {
            'ts': int(ts_ms or int(time.time()*1000)),
            'bucket': int(bt),
            'interval': str(interval),
            'window': int(window),
            'r': float(r_val),
            'pct_blue': float(pct_blue),
            'pct_orange': float(pct_orange),
        }
        _nb_groups.setdefault(bt, []).append(row)
        # trim old buckets to keep memory bounded
        if len(_nb_groups) > 1000:
            for k in sorted(list(_nb_groups.keys()))[:-900]:
                _nb_groups.pop(k, None)
    except Exception:
        pass

# In-memory order log for UI markers
orders = deque(maxlen=500)  # each item: {ts, side, price, size, paper, market}

# Bot controller for start/stop from UI
bot_ctrl = {
    'running': False,
    'thread': None,
    'last_signal': 'HOLD',
    'last_order': None,
    'nb_zone': None,  # 'BLUE' or 'ORANGE'
    'cfg_override': {  # values can be overridden via /api/bot/config
        'paper': None,
        'order_krw': None,
        'pnl_ratio': None,
        'pnl_profit_ratio': None,
        'pnl_loss_ratio': None,
        'ema_fast': None,
        'ema_slow': None,
        'candle': None,
        'market': None,
        'interval_sec': None,
        'require_ml': None,  # if true, require ML confirmation to place orders
        # runtime key injection (avoid restarting server)
        'access_key': None,
        'secret_key': None,
        'open_api_access_key': None,
        'open_api_secret_key': None,
    }
}

# ---------------- NB auto-tune persistence ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PARAMS_PATH = os.path.join(DATA_DIR, 'nb_params.json')

def _ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def load_nb_params():
    try:
        _ensure_data_dir()
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return { 'buy': 0.70, 'sell': 0.30, 'window': 50, 'updated_at': None }

def save_nb_params(params: dict):
    try:
        _ensure_data_dir()
        params = dict(params)
        params['updated_at'] = int(time.time()*1000)
        with open(PARAMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False)
        return True
    except Exception:
        return False

# ---------------- ML training/prediction (development) ----------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
ML_MODEL_PATH = os.path.join(MODELS_DIR, 'nb_ml.pkl')

def _model_path_for(interval: str) -> str:
    try:
        safe = str(interval or 'minute10').replace('/', '_')
    except Exception:
        safe = 'minute10'
    return os.path.join(MODELS_DIR, f'nb_ml_{safe}.pkl')

def _ensure_models_dir():
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception:
        pass

def _build_features(df: pd.DataFrame, window: int, ema_fast: int = 10, ema_slow: int = 30, horizon: int = 5) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['close'] = pd.to_numeric(df['close'], errors='coerce')
    out['high'] = pd.to_numeric(df['high'], errors='coerce')
    out['low'] = pd.to_numeric(df['low'], errors='coerce')
    # NB r
    r = _compute_r_from_ohlcv(df, window)
    out['r'] = r
    out['w'] = (out['high'].rolling(window).max() - out['low'].rolling(window).min()) / ((out['high'] + out['low'])/2).replace(0, np.nan)
    # EMA features
    out['ema_f'] = out['close'].ewm(span=ema_fast, adjust=False).mean()
    out['ema_s'] = out['close'].ewm(span=ema_slow, adjust=False).mean()
    out['ema_diff'] = out['ema_f'] - out['ema_s']
    # r smoothed and slopes
    out['r_ema3'] = out['r'].ewm(span=3, adjust=False).mean()
    out['r_ema5'] = out['r'].ewm(span=5, adjust=False).mean()
    out['dr'] = out['r'].diff()
    out['ret1'] = out['close'].pct_change(1)
    out['ret3'] = out['close'].pct_change(3)
    out['ret5'] = out['close'].pct_change(5)
    # Zone-aware helper features so model can learn BLUE/ORANGE context explicitly
    try:
        HIGH = float(os.getenv('NB_HIGH', '0.55'))
        LOW = float(os.getenv('NB_LOW', '0.45'))
    except Exception:
        HIGH, LOW = 0.55, 0.45
    rng = max(1e-9, HIGH - LOW)
    zone_flag = []  # +1=BLUE, -1=ORANGE
    dist_high = []  # max(0, r-HIGH)
    dist_low = []   # max(0, LOW-r)
    extreme_gap = []
    zone_conf = []  # confidence within current zone (0~1)
    # Zone extrema tracking (min/max r and corresponding prices within the current zone)
    zone_min_r_list = []
    zone_max_r_list = []
    zone_min_price_list = []
    zone_max_price_list = []
    zone_extreme_r_list = []     # r of current zone's defining extreme (min for BLUE, max for ORANGE)
    zone_extreme_price_list = [] # price at that extreme
    zone_extreme_age_list = []   # bars since that extreme was set/updated
    cur_zone = None
    cur_zone_min_r = None
    cur_zone_max_r = None
    cur_zone_min_idx = None
    cur_zone_max_idx = None
    cur_extreme_idx = None
    close_vals = out['close'].astype(float).fillna(method='bfill').fillna(method='ffill').fillna(0.0).values.tolist()
    r_vals = r.fillna(0.5).astype(float).values.tolist()
    for i, rv in enumerate(r_vals):
        if cur_zone not in ('BLUE','ORANGE'):
            cur_zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
        # update extremes per zone
        if cur_zone == 'BLUE' and rv >= HIGH:
            cur_zone = 'ORANGE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
        elif cur_zone == 'ORANGE' and rv <= LOW:
            cur_zone = 'BLUE'
            cur_zone_min_r = rv
            cur_zone_max_r = rv
            cur_zone_min_idx = i
            cur_zone_max_idx = i
            cur_extreme_idx = i
        # track within-zone min/max r and their indices
        cur_zone_min_r = rv if cur_zone_min_r is None else min(cur_zone_min_r, rv)
        cur_zone_max_r = rv if cur_zone_max_r is None else max(cur_zone_max_r, rv)
        if cur_zone_min_r == rv:
            cur_zone_min_idx = i
        if cur_zone_max_r == rv:
            cur_zone_max_idx = i
        if cur_zone == 'BLUE':
            cur_extreme_idx = cur_zone_min_idx if cur_zone_min_idx is not None else i
            zone_flag.append(1)
            zone_conf.append(max(0.0, (HIGH - rv) / rng))
        else:
            cur_extreme_idx = cur_zone_max_idx if cur_zone_max_idx is not None else i
            zone_flag.append(-1)
            zone_conf.append(max(0.0, (rv - LOW) / rng))
        dist_high.append(max(0.0, rv - HIGH))
        dist_low.append(max(0.0, LOW - rv))
        # current zone's defining extreme r
        cur_extreme_r = (cur_zone_min_r if cur_zone == 'BLUE' else cur_zone_max_r)
        extreme_gap.append(abs(rv - float(cur_extreme_r)))
        # append zone-wide extrema and their prices
        zone_min_r_list.append(float(cur_zone_min_r if cur_zone_min_r is not None else rv))
        zone_max_r_list.append(float(cur_zone_max_r if cur_zone_max_r is not None else rv))
        zmin_px = float(close_vals[cur_zone_min_idx]) if cur_zone_min_idx is not None else float(close_vals[i])
        zmax_px = float(close_vals[cur_zone_max_idx]) if cur_zone_max_idx is not None else float(close_vals[i])
        zone_min_price_list.append(zmin_px)
        zone_max_price_list.append(zmax_px)
        zone_extreme_r_list.append(float(cur_extreme_r))
        zext_px = float(close_vals[cur_extreme_idx]) if cur_extreme_idx is not None else float(close_vals[i])
        zone_extreme_price_list.append(zext_px)
        zone_extreme_age_list.append(int(i - (cur_extreme_idx if cur_extreme_idx is not None else i)))
    try:
        out['zone_flag'] = pd.Series(zone_flag, index=out.index)
        out['dist_high'] = pd.Series(dist_high, index=out.index)
        out['dist_low'] = pd.Series(dist_low, index=out.index)
        out['extreme_gap'] = pd.Series(extreme_gap, index=out.index)
        out['zone_conf'] = pd.Series(zone_conf, index=out.index)
        # new: zone extrema features (learning + insight)
        out['zone_min_r'] = pd.Series(zone_min_r_list, index=out.index)
        out['zone_max_r'] = pd.Series(zone_max_r_list, index=out.index)
        out['zone_min_price'] = pd.Series(zone_min_price_list, index=out.index)
        out['zone_max_price'] = pd.Series(zone_max_price_list, index=out.index)
        out['zone_extreme_r'] = pd.Series(zone_extreme_r_list, index=out.index)
        out['zone_extreme_price'] = pd.Series(zone_extreme_price_list, index=out.index)
        out['zone_extreme_age'] = pd.Series(zone_extreme_age_list, index=out.index)
    except Exception:
        pass
    # Time-of-day and weekly cycle features (help model learn time-localized BLUE/ORANGE behaviors)
    try:
        idx = out.index
        hours = pd.Index(getattr(idx, 'hour', pd.Series(idx).map(lambda x: getattr(x, 'hour', 0))))
        minutes = pd.Index(getattr(idx, 'minute', pd.Series(idx).map(lambda x: getattr(x, 'minute', 0))))
        tod_min = (hours.astype(int) * 60 + minutes.astype(int)).astype(float)
        out['tod_sin'] = np.sin(2 * np.pi * tod_min / (24*60))
        out['tod_cos'] = np.cos(2 * np.pi * tod_min / (24*60))
        # Day-of-week cyclic
        dows = pd.Index(getattr(idx, 'dayofweek', pd.Series(idx).map(lambda x: getattr(x, 'dayofweek', 0)))).astype(float)
        out['dow_sin'] = np.sin(2 * np.pi * dows / 7.0)
        out['dow_cos'] = np.cos(2 * np.pi * dows / 7.0)
        # Rough global sessions in KST: ASIA 09-17, EU 16-24, US 22-06
        h = hours.astype(int)
        out['sess_asia'] = ((h>=9) & (h<17)).astype(int)
        out['sess_eu'] = ((h>=16) | (h<0)).astype(int)  # 16~23
        out['sess_us'] = ((h>=22) | (h<6)).astype(int)
    except Exception:
        pass
    # forward return for labeling
    out['fwd'] = out['close'].shift(-horizon) / out['close'] - 1.0
    return out

def _train_ml(X: pd.DataFrame, y: np.ndarray):
    # Try scikit-learn; fall back to logistic regression if needed
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.utils.class_weight import compute_class_weight
        cls = GradientBoostingClassifier(random_state=42)
        # simple fit; for dev we skip CV heavy compute
        cls.fit(X, y)
        return cls
    except Exception as e:
        raise RuntimeError("scikit-learn이 필요합니다. pip install scikit-learn 실행 후 다시 시도하세요. 원인: %s" % e)

def _load_ml(interval: str | None = None):
    _ensure_models_dir()
    try:
        path = _model_path_for(interval or state.get('candle') or load_config().candle)
    except Exception:
        path = ML_MODEL_PATH
    if os.path.exists(path):
        return joblib.load(path)
    # Backward compatibility fallback
    if os.path.exists(ML_MODEL_PATH):
        return joblib.load(ML_MODEL_PATH)
    return None

def _simulate_pnl_from_preds(prices: pd.Series, preds: np.ndarray, fee_bps: float = 10.0) -> dict:
    pos = 0
    entry = 0.0
    pnl = 0.0
    wins = 0
    trades = 0
    for p, y in zip(prices.astype(float).values, preds.tolist()):
        if pos == 0 and y > 0:
            pos = 1
            entry = float(p)
            trades += 1
        elif pos == 1 and y < 0:
            ret = float(p) - entry
            ret -= abs(entry) * (fee_bps / 10000.0)
            ret -= abs(p) * (fee_bps / 10000.0)
            pnl += ret
            if ret > 0:
                wins += 1
            pos = 0
            entry = 0.0
    if pos == 1:
        p = float(prices.iloc[-1])
        ret = p - entry
        ret -= abs(entry) * (fee_bps / 10000.0)
        ret -= abs(p) * (fee_bps / 10000.0)
        pnl += ret
        if ret > 0:
            wins += 1
        pos = 0
    win_rate = (wins / trades * 100.0) if trades else 0.0
    return { 'pnl': float(pnl), 'trades': int(trades), 'wins': int(wins), 'win_rate': float(win_rate) }

@app.route('/api/ml/train', methods=['POST'])
def api_ml_train():
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        window = int(payload.get('window', load_nb_params().get('window', 50)))
        ema_fast = int(payload.get('ema_fast', 10))
        ema_slow = int(payload.get('ema_slow', 30))
        horizon = int(payload.get('horizon', 5))
        tau = float(payload.get('tau', 0.002))  # 0.2%
        count = int(payload.get('count', 1800))
        interval = payload.get('interval') or load_config().candle
        label_mode = str(payload.get('label_mode', 'nb_zone'))  # 'nb_zone' | 'fwd_return' | 'nb_extreme'
        # Optional: extreme-based labels tuning
        try:
            pullback_pct = float(payload.get('pullback_pct', os.getenv('NB_PULLBACK_PCT', '40')))
        except Exception:
            pullback_pct = 40.0
        try:
            confirm_bars = int(payload.get('confirm_bars', os.getenv('NB_CONFIRM_BARS', '2')))
        except Exception:
            confirm_bars = 2

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        # label: BUY(1), SELL(-1), HOLD(0) — strategy-aware by default
        if label_mode == 'fwd_return':
            fwd = feat['fwd']
            y = np.where(fwd >= tau, 1, np.where(fwd <= -tau, -1, 0))
        elif label_mode == 'nb_extreme':
            # Learn BLUE/ORANGE extremes with pullback confirmation; one BUY then one SELL
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            RANGE = max(1e-9, HIGH - LOW)
            pull_r = RANGE * (max(0.0, min(100.0, float(pullback_pct))) / 100.0)
            labels = np.zeros(len(df), dtype=int)
            zone = None
            zone_extreme = None
            prev_r = None
            confirm_up = 0
            confirm_dn = 0
            position = 'FLAT'
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                # init zone
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                # zone transitions reset extremes
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    zone_extreme = rv
                    confirm_up = 0; confirm_dn = 0
                # track extremes
                if zone == 'BLUE':
                    zone_extreme = min(zone_extreme, rv) if zone_extreme is not None else rv
                else:
                    zone_extreme = max(zone_extreme, rv) if zone_extreme is not None else rv
                # confirmations
                if prev_r is not None:
                    if rv > prev_r: confirm_up += 1
                    else: confirm_up = 0
                    if rv < prev_r: confirm_dn += 1
                    else: confirm_dn = 0
                prev_r = rv
                # decisions
                if position == 'FLAT' and zone == 'BLUE':
                    if (rv - zone_extreme) >= pull_r and confirm_up >= int(confirm_bars):
                        labels[i] = 1
                        position = 'LONG'
                        confirm_up = 0; confirm_dn = 0
                elif position == 'LONG' and zone == 'ORANGE':
                    if (zone_extreme - rv) >= pull_r and confirm_dn >= int(confirm_bars):
                        labels[i] = -1
                        position = 'FLAT'
                        confirm_up = 0; confirm_dn = 0
            # align labels to feature index
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        elif label_mode == 'nb_best_trade':
            # Build NB zone transitions, form BUY/SELL pairs, pick the single best PnL pair
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            zone = None
            signals = []  # (idx, side)
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE','ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    signals.append((i, -1))  # SELL
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    signals.append((i, 1))   # BUY
            # normalize to alternating BUY/SELL starting with BUY
            norm = []
            last = None
            for i, s in signals:
                if s == last:
                    continue
                norm.append((i, s))
                last = s
            while norm and norm[0][1] != 1:
                norm.pop(0)
            # pair and score
            prices = df['close'].astype(float).values.tolist()
            best = None
            for k in range(0, len(norm)-1, 2):
                bi, bs = norm[k]
                if k+1 >= len(norm):
                    break
                si, ss = norm[k+1]
                if bs != 1 or ss != -1:
                    continue
                if si <= bi or bi < 0 or si >= len(prices):
                    continue
                ret = float(prices[si]) - float(prices[bi])
                # approx fees: 0.1% in/out
                fee_bps = 10.0
                ret -= float(prices[bi]) * (fee_bps/10000.0)
                ret -= float(prices[si]) * (fee_bps/10000.0)
                if (best is None) or (ret > best['pnl']):
                    best = { 'buy_idx': bi, 'sell_idx': si, 'pnl': ret }
            labels = np.zeros(len(df), dtype=int)
            if best is not None:
                labels[best['buy_idx']] = 1
                labels[best['sell_idx']] = -1
            # align labels to feature index
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        else:
            # NB zone transition labels consistent with live trading loop
            r = _compute_r_from_ohlcv(df, window)
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
            labels = np.zeros(len(df), dtype=int)
            zone = None
            r_vals = r.values.tolist()
            for i in range(len(df)):
                rv = r_vals[i] if i < len(r_vals) else 0.5
                if zone not in ('BLUE', 'ORANGE'):
                    zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                sig = 0
                if zone == 'BLUE' and rv >= HIGH:
                    zone = 'ORANGE'
                    sig = -1  # SELL
                elif zone == 'ORANGE' and rv <= LOW:
                    zone = 'BLUE'
                    sig = 1   # BUY
                labels[i] = sig
            # align labels to feature frame
            idx_map = { ts: i for i, ts in enumerate(df.index) }
            y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
        X = feat[['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']]
        # Sample weights to reduce HOLD dominance
        total_n = len(X)
        c_neg = int((y==-1).sum()); c_zero = int((y==0).sum()); c_pos = int((y==1).sum())
        denom = max(1, c_neg) + max(1, c_zero) + max(1, c_pos)
        w_neg = float(total_n) / max(1, 3*c_neg)
        w_zero = float(total_n) / max(1, 3*c_zero)
        w_pos = float(total_n) / max(1, 3*c_pos)
        w = np.where(y==-1, w_neg, np.where(y==0, w_zero, w_pos)).astype(float)

        # Hyperparameter search with time-series CV (weighted)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        Xv = X.values
        tscv = TimeSeriesSplit(n_splits=3)
        grid = [
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 2},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 2},
            {'n_estimators': 150, 'learning_rate': 0.10, 'max_depth': 3},
        ]
        best_params = None
        best_score = -1e9
        best_pnl = -1e18
        # prices aligned to feature index
        prices = feat['close'].loc[X.index] if hasattr(X, 'index') else feat['close']
        for params in grid:
            accs=[]; f1s=[]; cms=None; pnl_sum=0.0
            for tr_idx, va_idx in tscv.split(Xv):
                cls = GradientBoostingClassifier(random_state=42, **params)
                cls.fit(Xv[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
                yp = cls.predict(Xv[va_idx])
                accs.append(accuracy_score(y[va_idx], yp))
                f1s.append(f1_score(y[va_idx], yp, average='macro', zero_division=0))
                cm = confusion_matrix(y[va_idx], yp, labels=[-1,0,1])
                cms = (cm if cms is None else (cms + cm))
                # pnl on validation slice
                try:
                    prices_va = prices.iloc[va_idx]
                    st = _simulate_pnl_from_preds(prices_va, yp)
                    pnl_sum += st['pnl']
                except Exception:
                    pass
            avg_f1 = float(np.mean(f1s)) if f1s else 0.0
            score = avg_f1
            if (score > best_score + 1e-9) or (abs(score - best_score) <= 1e-9 and pnl_sum > best_pnl):
                best_score = score
                best_params = params
                best_pnl = pnl_sum
        # Fit best model on all data with weights
        base = GradientBoostingClassifier(random_state=42, **(best_params or {}))
        base.fit(Xv, y, sample_weight=w)
        _ensure_models_dir()
        # compute reports
        yhat_in = base.predict(Xv)
        report_in = classification_report(y, yhat_in, output_dict=True, zero_division=0)
        cm_in = confusion_matrix(y, yhat_in, labels=[-1,0,1]).tolist()
        # summarize CV again for metrics payload
        metrics = {
            'in_sample': { 'report': report_in, 'confusion': cm_in },
            'cv': { 'f1_macro': float(best_score), 'pnl_sum': float(best_pnl) },
            'params': best_params,
        }
        # persist the exact feature order used for training
        try:
            feature_names = list(feat[['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']].columns)
        except Exception:
            feature_names = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
        pack = { 'model': base, 'window': window, 'ema_fast': ema_fast, 'ema_slow': ema_slow, 'horizon': horizon, 'tau': tau, 'interval': interval, 'metrics': metrics, 'trained_at': int(time.time()*1000), 'feature_names': feature_names }
        # save model per-interval
        try:
            joblib.dump(pack, _model_path_for(interval))
        except Exception:
            joblib.dump(pack, ML_MODEL_PATH)
        ml_state['train_count'] = int(ml_state.get('train_count', 0)) + 1
        classes = { '-1': int((y==-1).sum()), '0': int((y==0).sum()), '1': int((y==1).sum()) }
        return jsonify({'ok': True, 'classes': classes, 'report': report_in, 'cv': metrics['cv'], 'params': best_params, 'train_count': ml_state['train_count']})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/ml/predict', methods=['GET'])
def api_ml_predict():
    try:
        # load model for current interval
        cur_interval = state.get('candle') or load_config().candle
        pack = _load_ml(cur_interval)
        if not pack:
            return jsonify({'ok': False, 'error': 'model_not_trained'}), 400
        model = pack['model']
        window = int(pack.get('window', 50))
        ema_fast = int(pack.get('ema_fast', 10))
        ema_slow = int(pack.get('ema_slow', 30))
        horizon = int(pack.get('horizon', 5))
        cfg = load_config()
        df = get_candles(cfg.market, cur_interval, count=max(400, window*3))
        feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
        # IMPORTANT: Use same feature set and order as model was trained on
        base_cols = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
        ext_cols = ['zone_flag','dist_high','dist_low','extreme_gap','zone_conf','zone_min_r','zone_max_r','zone_extreme_r','zone_extreme_age']
        trained_cols = list(pack.get('feature_names') or [])
        if not trained_cols:
            # fallback: constrain to model.n_features_in_ if present
            cand = base_cols + [c for c in ext_cols if c in feat.columns]
            try:
                need = int(getattr(model, 'n_features_in_', len(cand)))
            except Exception:
                need = len(cand)
            trained_cols = cand[:need]
        X = feat[[c for c in trained_cols if c in feat.columns]]
        probs = None
        try:
            probs = model.predict_proba(X.values)[-1].tolist()
        except Exception:
            probs = []
        pred = int(model.predict(X.values)[-1])
        # Build insight payload from last feature row
        ins = {}
        try:
            last = feat.iloc[-1]
            zone_flag = int(round(float(last.get('zone_flag', 0))))
            zone = 'BLUE' if zone_flag == 1 else ('ORANGE' if zone_flag == -1 else 'UNKNOWN')
            # heuristic zone probabilities from r distances to thresholds
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            rv = float(last.get('r', 0.5))
            p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
            p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
            s0 = p_blue_raw + p_orange_raw
            if s0 > 0:
                p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
            # Trend-weighted adjustment using recent r trajectory
            try:
                trend_k = int(os.getenv('NB_TREND_K', '30'))
                trend_alpha = float(os.getenv('NB_TREND_ALPHA', '0.5'))
            except Exception:
                trend_k, trend_alpha = 30, 0.5
            try:
                r_series = _compute_r_from_ohlcv(df, window).astype(float)
                if len(r_series) >= trend_k*2:
                    tail_now = r_series.iloc[-trend_k:]
                    tail_prev = r_series.iloc[-trend_k*2:-trend_k]
                    zmax_now, zmax_prev = float(tail_now.max()), float(tail_prev.max())
                    zmin_now, zmin_prev = float(tail_now.min()), float(tail_prev.min())
                    # ORANGE weakening when recent peak < previous peak
                    trend_orange = max(0.0, (zmax_prev - zmax_now) / rng)
                    # BLUE weakening when recent trough > previous trough
                    trend_blue = max(0.0, (zmin_now - zmin_prev) / rng)
                    p_orange = max(0.0, min(1.0, p_orange_raw * (1.0 - trend_alpha * trend_orange)))
                    p_blue = max(0.0, min(1.0, p_blue_raw * (1.0 - trend_alpha * trend_blue)))
                    s = p_blue + p_orange
                    if s > 0:
                        p_blue, p_orange = p_blue/s, p_orange/s
                else:
                    p_blue, p_orange = p_blue_raw, p_orange_raw
            except Exception:
                p_blue, p_orange = p_blue_raw, p_orange_raw
            ins = {
                'r': rv,
                'zone_flag': zone_flag,
                'zone': zone,
                'zone_conf': float(last.get('zone_conf', 0.0)),
                'dist_high': float(last.get('dist_high', 0.0)),
                'dist_low': float(last.get('dist_low', 0.0)),
                'extreme_gap': float(last.get('extreme_gap', 0.0)),
                # expose zone extrema for UI insight
                'zone_min_r': float(last.get('zone_min_r', rv)),
                'zone_max_r': float(last.get('zone_max_r', rv)),
                'zone_extreme_r': float(last.get('zone_extreme_r', rv)),
                'zone_extreme_age': int(last.get('zone_extreme_age', 0)),
                'w': float(last.get('w', 0.0)),
                'ema_diff': float(last.get('ema_diff', 0.0)),
                'pct_blue_raw': float(p_blue_raw*100.0),
                'pct_orange_raw': float(p_orange_raw*100.0),
                'pct_blue': float(p_blue*100.0),
                'pct_orange': float(p_orange*100.0),
            }
            try:
                _record_group_observation(cur_interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
            except Exception:
                pass
        except Exception:
            ins = {}
        # Fallback insight if feature frame is empty
        if not ins:
            try:
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
            except Exception:
                HIGH, LOW = 0.55, 0.45
            rng = max(1e-9, HIGH - LOW)
            r_series = _compute_r_from_ohlcv(df, window)
            rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
            p_blue = max(0.0, min(1.0, (HIGH - rv) / rng))
            p_orange = max(0.0, min(1.0, (rv - LOW) / rng))
            s = p_blue + p_orange
            if s > 0:
                p_blue, p_orange = p_blue/s, p_orange/s
            zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
            ins = {
                'r': rv,
                'zone_flag': (-1 if zone=='ORANGE' else 1),
                'zone': zone,
                'zone_conf': float(max(0.0, (rv-LOW)/rng) if zone=='ORANGE' else max(0.0, (HIGH-rv)/rng)),
                'dist_high': float(max(0.0, rv - HIGH)),
                'dist_low': float(max(0.0, LOW - rv)),
                'extreme_gap': 0.0,
                'w': float(((df['high'].rolling(window).max() - df['low'].rolling(window).min()) / ((df['high'] + df['low'])/2).replace(0, np.nan)).iloc[-1]) if len(df) else 0.0,
                'ema_diff': float((df['close'].ewm(span=ema_fast, adjust=False).mean().iloc[-1] - df['close'].ewm(span=ema_slow, adjust=False).mean().iloc[-1])) if len(df) else 0.0,
                'pct_blue': float(p_blue*100.0),
                'pct_orange': float(p_orange*100.0),
            }
            try:
                _record_group_observation(cur_interval, window, rv, ins['pct_blue'], ins['pct_orange'], int(time.time()*1000))
            except Exception:
                pass
        action = 'HOLD'
        if pred > 0:
            action = 'BUY'
        elif pred < 0:
            action = 'SELL'
        return jsonify({'ok': True, 'action': action, 'pred': pred, 'probs': probs, 'train_count': ml_state.get('train_count', 0), 'insight': ins})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/ml/metrics', methods=['GET'])
def api_ml_metrics():
    try:
        cur_interval = state.get('candle') or load_config().candle
        pack = _load_ml(cur_interval)
        if not pack:
            return jsonify({'ok': False, 'error': 'model_not_trained'}), 400
        metrics = pack.get('metrics', {}) or {}
        # If metrics missing (old model), recompute lightweight metrics on recent data
        if not metrics or not metrics.get('in_sample'):
            try:
                model = pack['model']
                window = int(pack.get('window', 50))
                ema_fast = int(pack.get('ema_fast', 10))
                ema_slow = int(pack.get('ema_slow', 30))
                horizon = int(pack.get('horizon', 5))
                cfg = load_config()
                df = get_candles(cfg.market, cur_interval, count=max(800, window*3))
                feat = _build_features(df, window, ema_fast, ema_slow, horizon).dropna().copy()
                X = feat[['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']]
                # default NB zone labels for comparison
                r = _compute_r_from_ohlcv(df, window)
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
                labels = np.zeros(len(df), dtype=int)
                zone = None
                r_vals = r.values.tolist()
                for i in range(len(df)):
                    rv = r_vals[i] if i < len(r_vals) else 0.5
                    if zone not in ('BLUE','ORANGE'):
                        zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
                    sig = 0
                    if zone == 'BLUE' and rv >= HIGH:
                        zone = 'ORANGE'; sig = -1
                    elif zone == 'ORANGE' and rv <= LOW:
                        zone = 'BLUE'; sig = 1
                    labels[i] = sig
                idx_map = { ts: i for i, ts in enumerate(df.index) }
                y = np.array([ labels[idx_map.get(ts, 0)] for ts in feat.index ], dtype=int)
                from sklearn.metrics import classification_report, confusion_matrix, f1_score
                from sklearn.model_selection import TimeSeriesSplit
                yhat = model.predict(X.values)
                rep = classification_report(y, yhat, output_dict=True, zero_division=0)
                cm = confusion_matrix(y, yhat, labels=[-1,0,1]).tolist()
                # quick CV
                tscv = TimeSeriesSplit(n_splits=3)
                f1s=[]; pnl_sum=0.0
                for tr_idx, va_idx in tscv.split(X.values):
                    yp = model.predict(X.values[va_idx])
                    f1s.append(f1_score(y[va_idx], yp, average='macro', zero_division=0))
                    try:
                        prices_va = feat['close'].iloc[va_idx]
                        st = _simulate_pnl_from_preds(prices_va, yp)
                        pnl_sum += st['pnl']
                    except Exception:
                        pass
                metrics = {
                    'in_sample': { 'report': rep, 'confusion': cm },
                    'cv': { 'f1_macro': float(np.mean(f1s)) if f1s else 0.0, 'pnl_sum': float(pnl_sum) },
                    'params': None,
                }
                # persist back for faster future reads
                try:
                    pack['metrics'] = metrics
                    joblib.dump(pack, _model_path_for(cur_interval))
                except Exception:
                    pass
            except Exception:
                metrics = {}
        return jsonify({'ok': True, 'interval': pack.get('interval', cur_interval), 'metrics': metrics, 'params': metrics.get('params'), 'trained_at': pack.get('trained_at'), 'train_count': ml_state.get('train_count', 0)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def updater():
    cfg = load_config()
    state["ema_fast"] = cfg.ema_fast
    state["ema_slow"] = cfg.ema_slow
    state["market"] = cfg.market
    state["candle"] = cfg.candle
    # Initial seed with candles
    try:
        df = get_candles(cfg.market, cfg.candle, count=max(cfg.ema_slow + 60, 120))
        sig = decide_signal(df, cfg.ema_fast, cfg.ema_slow)
        tail = df.tail(60)
        for t, p in zip(tail.index, tail["close"].astype(float)):
            state["history"].append((int(t.timestamp()*1000), float(p)))
        state["price"] = float(tail["close"].iloc[-1])
        state["signal"] = sig
    except Exception:
        pass

    tick = 0
    tick_sec = int(os.getenv("UI_TICK_SEC", "1"))
    recalc_every = int(os.getenv("UI_RECALC_SEC", "30"))
    while True:
        try:
            # Live price via ticker
            cp = pyupbit.get_current_price(cfg.market)
            if cp:
                now_ms = int(time.time() * 1000)
                state["price"] = float(cp)
                state["history"].append((now_ms, float(cp)))
            # Periodic recalc of signal from candles
            if tick % max(recalc_every, 1) == 0:
                df = get_candles(cfg.market, cfg.candle, count=max(cfg.ema_slow + 5, 60))
                state["signal"] = decide_signal(df, cfg.ema_fast, cfg.ema_slow)
        except Exception:
            pass
        tick += tick_sec
        time.sleep(tick_sec)


def _resolve_config():
    base = load_config()
    ov = bot_ctrl['cfg_override']
    # merge overrides if present
    base.paper = base.paper if ov['paper'] is None else bool(ov['paper'])
    base.order_krw = base.order_krw if ov['order_krw'] is None else int(ov['order_krw'])
    # attach pnl_ratio dynamically to base for Trader
    try:
        base.pnl_ratio = float(ov['pnl_ratio']) if ov['pnl_ratio'] is not None else float(getattr(base, 'pnl_ratio', 0.0))
    except Exception:
        base.pnl_ratio = float(getattr(base, 'pnl_ratio', 0.0))
    # Attach new ratios for profit/loss mapping
    try:
        base.pnl_profit_ratio = float(ov['pnl_profit_ratio']) if ov['pnl_profit_ratio'] is not None else float(getattr(base, 'pnl_profit_ratio', 0.0))
    except Exception:
        base.pnl_profit_ratio = float(getattr(base, 'pnl_profit_ratio', 0.0))
    try:
        base.pnl_loss_ratio = float(ov['pnl_loss_ratio']) if ov['pnl_loss_ratio'] is not None else float(getattr(base, 'pnl_loss_ratio', 0.0))
    except Exception:
        base.pnl_loss_ratio = float(getattr(base, 'pnl_loss_ratio', 0.0))
    base.ema_fast = base.ema_fast if ov['ema_fast'] is None else int(ov['ema_fast'])
    base.ema_slow = base.ema_slow if ov['ema_slow'] is None else int(ov['ema_slow'])
    base.candle = base.candle if ov['candle'] is None else str(ov['candle'])
    base.market = base.market if ov['market'] is None else str(ov['market'])
    base.interval_sec = base.interval_sec if ov['interval_sec'] is None else int(ov['interval_sec'])
    # keys (if provided via API)
    base.access_key = base.access_key if ov['access_key'] is None else str(ov['access_key'])
    base.secret_key = base.secret_key if ov['secret_key'] is None else str(ov['secret_key'])
    return base

def _get_runtime_keys():
    """Return a tuple of (std_ak, std_sk, open_ak, open_sk) from overrides/env."""
    ov = bot_ctrl['cfg_override']
    std_ak = (ov.get('access_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_ACCESS_KEY')
    std_sk = (ov.get('secret_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_SECRET_KEY')
    open_ak = (ov.get('open_api_access_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_OPEN_API_ACCESS_KEY')
    open_sk = (ov.get('open_api_secret_key') if isinstance(ov, dict) else None) or os.getenv('UPBIT_OPEN_API_SECRET_KEY')
    return std_ak, std_sk, open_ak, open_sk

def _mask_key(v: str | None) -> str:
    if not v:
        return ''
    try:
        s = str(v)
        if len(s) <= 8:
            return s[:2] + ('*' * max(0, len(s) - 4)) + s[-2:]
        return s[:4] + ('*' * (len(s) - 8)) + s[-4:]
    except Exception:
        return '<?>'

def log_env_keys():
    std_ak, std_sk, open_ak, open_sk = _get_runtime_keys()
    print(f"[ENV] UPBIT_ACCESS_KEY={_mask_key(std_ak)} UPBIT_SECRET_KEY={_mask_key(std_sk)}")
    print(f"[ENV] UPBIT_OPEN_API_ACCESS_KEY={_mask_key(open_ak)} UPBIT_OPEN_API_SECRET_KEY={_mask_key(open_sk)}")

def _reload_env_vars() -> bool:
    try:
        # project root
        load_dotenv()
        load_dotenv("env.local", override=False)
        # bot dir (this file)
        base_dir = os.path.dirname(__file__)
        load_dotenv(os.path.join(base_dir, ".env"), override=True)
        load_dotenv(os.path.join(base_dir, "env.local"), override=True)
        return True
    except Exception:
        return False


def trade_loop():
    try:
        cfg = _resolve_config()
        upbit = None
        if not cfg.paper and cfg.access_key and cfg.secret_key:
            upbit = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
        trader = Trader(
            upbit,
            TradeConfig(
                market=cfg.market,
                order_krw=cfg.order_krw,
                paper=cfg.paper,
                pnl_ratio=float(getattr(cfg, 'pnl_ratio', 0.0)),
                pnl_profit_ratio=float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
                pnl_loss_ratio=float(getattr(cfg, 'pnl_loss_ratio', 0.0)),
            )
        )
        last_signal = 'HOLD'
        # ML model cache for confirmation
        ml_pack = None
        ml_interval = None
        while bot_ctrl['running']:
            try:
                cfg = _resolve_config()
                # Use NB wave zone transitions: one SELL when entering ORANGE, one BUY when entering BLUE
                df = get_candles(cfg.market, cfg.candle, count=max(120, cfg.ema_slow + 5))
                price = float(df['close'].iloc[-1])
                # Compute r in [0,1]
                try:
                    window = int(load_nb_params().get('window', 50))
                except Exception:
                    window = 50
                r = _compute_r_from_ohlcv(df, window)
                r_last = float(r.iloc[-1]) if len(r) else 0.5
                HIGH = float(os.getenv('NB_HIGH', '0.55'))
                LOW = float(os.getenv('NB_LOW', '0.45'))
                if bot_ctrl.get('nb_zone') not in ('BLUE','ORANGE'):
                    bot_ctrl['nb_zone'] = 'ORANGE' if r_last >= 0.5 else 'BLUE'
                sig = 'HOLD'
                if bot_ctrl['nb_zone'] == 'BLUE' and r_last >= HIGH:
                    bot_ctrl['nb_zone'] = 'ORANGE'
                    sig = 'SELL'
                elif bot_ctrl['nb_zone'] == 'ORANGE' and r_last <= LOW:
                    bot_ctrl['nb_zone'] = 'BLUE'
                    sig = 'BUY'
                state['signal'] = sig if sig != 'HOLD' else state.get('signal', 'HOLD')
                state['price'] = price
                if sig in ('BUY','SELL') and sig != last_signal:
                    # Optional: require ML confirmation
                    try:
                        require_ml = bool(bot_ctrl['cfg_override'].get('require_ml')) if bot_ctrl['cfg_override'].get('require_ml') is not None else (os.getenv('REQUIRE_ML_CONFIRM', 'false').lower()=='true')
                    except Exception:
                        require_ml = False
                    if require_ml:
                        try:
                            if ml_interval != cfg.candle or ml_pack is None:
                                ml_pack = _load_ml(cfg.candle)
                                ml_interval = cfg.candle
                            if ml_pack is not None:
                                model = ml_pack['model']
                                window = int(ml_pack.get('window', 50))
                                ema_fast = int(ml_pack.get('ema_fast', 10))
                                ema_slow = int(ml_pack.get('ema_slow', 30))
                                feat = _build_features(df, window, ema_fast, ema_slow, 5).dropna().copy()
                                # Respect trained feature order if available
                                trained_cols = list(ml_pack.get('feature_names') or [])
                                if not trained_cols:
                                    base_cols = ['r','w','ema_f','ema_s','ema_diff','r_ema3','r_ema5','dr','ret1','ret3','ret5']
                                    cols_ext = ['zone_flag','dist_high','dist_low','extreme_gap','zone_conf','zone_min_r','zone_max_r','zone_extreme_r','zone_extreme_age']
                                    cand = base_cols + [c for c in cols_ext if c in feat.columns]
                                    try:
                                        need = int(getattr(model, 'n_features_in_', len(cand)))
                                    except Exception:
                                        need = len(cand)
                                    trained_cols = cand[:need]
                                Xv = feat[[c for c in trained_cols if c in feat.columns]].values
                                ml_pred = int(model.predict(Xv)[-1]) if len(Xv) else 0
                                # Auto-sync server candle to ML model interval if they diverge
                                try:
                                    ml_used_interval = str(ml_pack.get('interval') or cfg.candle)
                                except Exception:
                                    ml_used_interval = cfg.candle
                                if ml_used_interval and ml_used_interval != cfg.candle:
                                    bot_ctrl['cfg_override']['candle'] = ml_used_interval
                                    state['candle'] = ml_used_interval
                                    # Skip this tick to reload with new interval
                                    last_signal = sig
                                    bot_ctrl['last_signal'] = sig
                                    time.sleep(max(1, _resolve_config().interval_sec))
                                    continue
                                if (ml_pred == 0) or (ml_pred == 1 and sig != 'BUY') or (ml_pred == -1 and sig != 'SELL'):
                                    last_signal = sig
                                    bot_ctrl['last_signal'] = sig
                                    time.sleep(max(1, _resolve_config().interval_sec))
                                    continue
                        except Exception:
                            pass
                    # Update trader's dynamic pnl_ratio before each order
                    try:
                        trader.cfg.pnl_ratio = float(getattr(cfg, 'pnl_ratio', 0.0))
                    except Exception:
                        trader.cfg.pnl_ratio = 0.0
                    o = trader.place(sig, price)
                    order = {
                        'ts': int(time.time()*1000),
                        'side': sig,
                        'price': price,
                        'size': (o.get('size') if isinstance(o, dict) else None) or 0,
                        'paper': cfg.paper,
                        'market': cfg.market,
                    }
                    orders.append(order)
                    bot_ctrl['last_order'] = order
                last_signal = sig
                bot_ctrl['last_signal'] = sig
            except Exception:
                pass
            time.sleep(max(1, _resolve_config().interval_sec))
    finally:
        bot_ctrl['running'] = False


@app.route('/api/stream')
def api_stream():
    def gen():
        last_ts = None
        last_order_ts = None
        while True:
            try:
                ts = state["history"][-1][0] if state["history"] else None
                if ts and ts != last_ts:
                    last_ts = ts
                    payload = {
                        "ts": ts,
                        "price": state.get("price", 0),
                        "signal": state.get("signal", "HOLD"),
                        "market": state.get("market"),
                        "candle": state.get("candle"),
                        "ema_fast": state.get("ema_fast"),
                        "ema_slow": state.get("ema_slow"),
                    }
                    # Include latest order only when there's a new one
                    if orders:
                        o = orders[-1]
                        if last_order_ts != o.get("ts"):
                            payload["order"] = o
                            last_order_ts = o.get("ts")
                    yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.5)
                continue
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    return Response(gen(), mimetype='text/event-stream', headers=headers)


@app.route("/api/state")
def api_state():
    return jsonify({
        "price": state["price"],
        "signal": state["signal"],
        "ema_fast": state["ema_fast"],
        "ema_slow": state["ema_slow"],
        "market": state["market"],
        "candle": state["candle"],
        "history": list(state["history"]),
    })


@app.route('/api/ohlcv')
def api_ohlcv():
    try:
        cfg = load_config()
        count = int((request.args.get('count') or 300))
        interval = request.args.get('interval') or cfg.candle
        df = get_candles(cfg.market, interval, count=count)
        out = []
        for idx, row in df.iterrows():
            out.append({
                'time': int(idx.timestamp()*1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0.0,
            })
        return jsonify({'market': state.get('market'), 'candle': state.get('candle'), 'data': out})
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@app.route('/api/orders', methods=['GET'])
def api_orders():
    """Return recent orders for plotting markers on the chart."""
    try:
        return jsonify({
            'market': state.get('market'),
            'data': list(orders),
        })
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@app.route('/api/order', methods=['POST'])
def api_order_create():
    """Accept order notifications from the trader (paper or live)."""
    try:
        if request.is_json:
            payload = request.get_json(force=True)
        else:
            payload = request.form.to_dict()
        # Normalize fields
        order = {
            'ts': int(payload.get('ts') or int(time.time() * 1000)),
            'side': str(payload.get('side', '')).upper(),
            'price': float(payload.get('price', 0) or 0),
            'size': float(payload.get('size', 0) or 0),
            'paper': bool(payload.get('paper', True) in (True, 'true', '1', 1, 'True')),
            'market': payload.get('market') or state.get('market'),
        }
        orders.append(order)
        return jsonify({'ok': True, 'order': order})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/orders/clear', methods=['POST'])
def api_orders_clear():
    """Clear in-memory order log and return ok."""
    try:
        orders.clear()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def _compute_r_from_ohlcv(df: pd.DataFrame, window: int) -> pd.Series:
    hi = df['high'].rolling(window=window, min_periods=window).max()
    lo = df['low'].rolling(window=window, min_periods=window).min()
    span = (hi - lo).replace(0, np.nan)
    r = (df['close'] - lo) / span
    # pandas 2.x: fillna(method=...) deprecated → use bfill()
    return r.clip(0, 1).bfill().fillna(0.5)


def _simulate_pnl_from_r(prices: pd.Series, r: pd.Series, buy_th: float, sell_th: float,
                         debounce: int = 0, fee_bps: float = 0.0) -> dict:
    pos = 0
    entry = 0.0
    pnl = 0.0
    wins = 0
    trades = 0
    peak = 0.0
    maxdd = 0.0
    last_sig_idx = -10**9
    for i, (p, rv) in enumerate(zip(prices.values, r.values)):
        if pos == 0 and rv >= buy_th and (i - last_sig_idx) >= debounce:
            pos = 1
            entry = float(p)
            trades += 1
            last_sig_idx = i
        elif pos == 1 and rv <= sell_th and (i - last_sig_idx) >= debounce:
            ret = float(p) - entry
            # apply fee (approx market in/out)
            ret -= abs(entry) * (fee_bps / 10000.0)
            ret -= abs(p) * (fee_bps / 10000.0)
            pnl += ret
            if ret > 0:
                wins += 1
            pos = 0
            entry = 0.0
            last_sig_idx = i
        peak = max(peak, pnl)
        maxdd = max(maxdd, peak - pnl)
    # close at last
    if pos == 1:
        p = float(prices.iloc[-1])
        ret = p - entry
        ret -= abs(entry) * (fee_bps / 10000.0)
        ret -= abs(p) * (fee_bps / 10000.0)
        pnl += ret
        if ret > 0:
            wins += 1
        pos = 0
    win_rate = (wins / trades * 100.0) if trades else 0.0
    return {
        'pnl': float(pnl),
        'trades': trades,
        'wins': wins,
        'win_rate': win_rate,
        'max_dd': float(maxdd),
    }


@app.route('/api/nb/optimize', methods=['POST'])
def api_nb_optimize():
    """Grid-search NB thresholds to maximize PnL on recent OHLCV.
    Body JSON: { window: int, buy: [start, stop, step], sell: [start, stop, step], debounce: int, fee_bps: float, count: int, interval: str }
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        window = int(payload.get('window', 50))
        buy_grid = payload.get('buy', [0.6, 0.85, 0.02])
        sell_grid = payload.get('sell', [0.15, 0.45, 0.02])
        debounce = int(payload.get('debounce', 6))
        fee_bps = float(payload.get('fee_bps', 10.0))  # 0.1%
        count = int(payload.get('count', 600))
        interval = payload.get('interval') or load_config().candle

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        if not {'open','high','low','close'}.issubset(df.columns):
            return jsonify({'ok': False, 'error': 'OHLCV missing', 'data': {}}), 400
        r = _compute_r_from_ohlcv(df, window)
        prices = df['close']

        b_start, b_stop, b_step = buy_grid
        s_start, s_stop, s_step = sell_grid
        best = None
        best_stats = None
        b = b_start
        while b <= b_stop + 1e-9:
            s = s_start
            while s <= s_stop + 1e-9:
                stats = _simulate_pnl_from_r(prices, r, b, s, debounce=debounce, fee_bps=fee_bps)
                if best is None or stats['pnl'] > best_stats['pnl']:
                    best = {'buy': round(b, 3), 'sell': round(s, 3)}
                    best_stats = stats
                s += s_step
            b += b_step

        # persist best and respond
        if best:
            save_nb_params({ 'buy': best['buy'], 'sell': best['sell'], 'window': window })
        return jsonify({'ok': True, 'best': best, 'stats': best_stats, 'saved': bool(best)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/zone')
def api_nb_zone():
    """Return current NB r and zone. Optional query params:
    - r: float (if provided, use this r directly)
    - interval: str (default: config.candle)
    - count: int (default: 300)
    - window: int (default: saved nb_params.window)
    """
    try:
        # thresholds: prefer env, else defaults
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH, LOW = 0.55, 0.45
        rng = max(1e-9, HIGH - LOW)

        q = request.args
        r_q = q.get('r')
        if r_q is not None:
            rv = float(r_q)
            interval = q.get('interval') or state.get('candle') or load_config().candle
            window = int(q.get('window') or load_nb_params().get('window', 50))
        else:
            cfg = load_config()
            interval = q.get('interval') or state.get('candle') or cfg.candle
            count = int(q.get('count') or 300)
            window = int(q.get('window') or load_nb_params().get('window', 50))
            df = get_candles(cfg.market, interval, count=count)
            r_series = _compute_r_from_ohlcv(df, window)
            rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
        p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
        p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
        s0 = p_blue_raw + p_orange_raw
        if s0 > 0:
            p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
        # Optional trend weighting when data available
        p_blue, p_orange = p_blue_raw, p_orange_raw
        try:
            trend_k = int(os.getenv('NB_TREND_K', '30'))
            trend_alpha = float(os.getenv('NB_TREND_ALPHA', '0.5'))
        except Exception:
            trend_k, trend_alpha = 30, 0.5
        if r_q is None:
            try:
                r_series = _compute_r_from_ohlcv(df, window).astype(float)
                if len(r_series) >= trend_k*2:
                    tail_now = r_series.iloc[-trend_k:]
                    tail_prev = r_series.iloc[-trend_k*2:-trend_k]
                    zmax_now, zmax_prev = float(tail_now.max()), float(tail_prev.max())
                    zmin_now, zmin_prev = float(tail_now.min()), float(tail_prev.min())
                    trend_orange = max(0.0, (zmax_prev - zmax_now) / rng)
                    trend_blue = max(0.0, (zmin_now - zmin_prev) / rng)
                    p_orange = max(0.0, min(1.0, p_orange_raw * (1.0 - trend_alpha * trend_orange)))
                    p_blue = max(0.0, min(1.0, p_blue_raw * (1.0 - trend_alpha * trend_blue)))
                    s = p_blue + p_orange
                    if s > 0:
                        p_blue, p_orange = p_blue/s, p_orange/s
            except Exception:
                pass
        zone = 'ORANGE' if rv >= 0.5 else 'BLUE'
        return jsonify({
            'ok': True,
            'interval': interval,
            'window': window,
            'r': float(rv),
            'zone': zone,
            'pct_blue_raw': float(p_blue_raw*100.0),
            'pct_orange_raw': float(p_orange_raw*100.0),
            'pct_blue': float(p_blue*100.0),
            'pct_orange': float(p_orange*100.0),
            'high': float(HIGH),
            'low': float(LOW),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/group', methods=['POST'])
def api_nb_group():
    """Group multiple intervals at the current time and return per-interval NB stats and a consensus.
    Body JSON (all optional):
      - intervals: ["minute1","minute3","minute5","minute10"]
      - window: int (default saved nb_params.window)
      - weights: { interval: number }
      - tolerance_sec: number (default: interval length in sec)
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        try:
            HIGH = float(os.getenv('NB_HIGH', '0.55'))
            LOW = float(os.getenv('NB_LOW', '0.45'))
        except Exception:
            HIGH, LOW = 0.55, 0.45
        rng = max(1e-9, HIGH - LOW)
        def interval_seconds(iv: str) -> int:
            if iv.startswith('minute'):
                try:
                    m = int(iv.replace('minute',''))
                except Exception:
                    m = 1
                return max(60, m*60)
            if iv == 'day':
                return 24*60*60
            if iv == 'minute60':
                return 60*60
            return 600
        cfg = load_config()
        intervals = payload.get('intervals') or ['minute1','minute3','minute5','minute10']
        base_window = int(payload.get('window', load_nb_params().get('window', 50)))
        weights = payload.get('weights') or { iv: max(1, interval_seconds(iv)//60) for iv in intervals }
        tol_sec = int(payload.get('tolerance_sec', 0))  # per-interval fallback below
        now = int(time.time())
        rows = []
        w_sum = 0.0
        blue_sum = 0.0
        orange_sum = 0.0
        for iv in intervals:
            try:
                sec = interval_seconds(iv)
                tol = tol_sec if tol_sec>0 else sec
                df = get_candles(cfg.market, iv, count=max(200, base_window*3))
                if df is None or df.empty:
                    continue
                ts_ms = int(df.index[-1].timestamp()*1000)
                ts_s = ts_ms//1000
                if abs(now - ts_s) > tol:
                    # skip very stale bars
                    continue
                r_series = _compute_r_from_ohlcv(df, base_window)
                rv = float(r_series.iloc[-1]) if len(r_series) else 0.5
                p_blue_raw = max(0.0, min(1.0, (HIGH - rv) / rng))
                p_orange_raw = max(0.0, min(1.0, (rv - LOW) / rng))
                s0 = p_blue_raw + p_orange_raw
                if s0>0:
                    p_blue_raw, p_orange_raw = p_blue_raw/s0, p_orange_raw/s0
                z = 'ORANGE' if rv >= 0.5 else 'BLUE'
                w = float(weights.get(iv, 1.0))
                w_sum += w
                blue_sum += w * p_blue_raw
                orange_sum += w * p_orange_raw
                rows.append({
                    'interval': iv,
                    'time_ms': ts_ms,
                    'r': rv,
                    'zone': z,
                    'pct_blue_raw': float(p_blue_raw*100.0),
                    'pct_orange_raw': float(p_orange_raw*100.0),
                    'weight': w,
                })
            except Exception:
                continue
        consensus = {
            'pct_blue': float(blue_sum/w_sum*100.0) if w_sum>0 else 0.0,
            'pct_orange': float(orange_sum/w_sum*100.0) if w_sum>0 else 0.0,
            'count': len(rows),
        }
        return jsonify({ 'ok': True, 'intervals': intervals, 'window': base_window, 'items': rows, 'consensus': consensus })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/api/nb/train', methods=['POST'])
def api_nb_train():
    """Auto period split training (grid search per segment) and persist best.
    Body JSON: { count: int(1800), segments: int(3), window: int, debounce: int, fee_bps: float, interval: str }
    """
    try:
        payload = request.get_json(force=True) if request.is_json else {}
        count = int(payload.get('count', 1800))
        segments = max(1, int(payload.get('segments', 3)))
        window = int(payload.get('window', load_nb_params().get('window', 50)))
        debounce = int(payload.get('debounce', 6))
        fee_bps = float(payload.get('fee_bps', 10.0))
        interval = payload.get('interval') or load_config().candle

        cfg = load_config()
        df = get_candles(cfg.market, interval, count=count)
        if len(df) < max(window*2, segments*50):
            return jsonify({'ok': False, 'error': 'Not enough data'}), 400
        r_all = _compute_r_from_ohlcv(df, window)
        prices_all = df['close']

        seg_len = len(df) // segments
        results = []
        def search_best(prices: pd.Series, r: pd.Series):
            best=None; best_stats=None
            b=0.6
            while b<=0.85+1e-9:
                s=0.15
                while s<=0.45+1e-9:
                    st = _simulate_pnl_from_r(prices, r, b, s, debounce=debounce, fee_bps=fee_bps)
                    if best is None or st['pnl']>best_stats['pnl']:
                        best={'buy':round(b,3),'sell':round(s,3)}; best_stats=st
                    s+=0.02
                b+=0.02
            return best, best_stats

        for i in range(segments):
            start = i*seg_len
            end = (i+1)*seg_len if i<segments-1 else len(df)
            r_seg = r_all.iloc[start:end]
            p_seg = prices_all.iloc[start:end]
            best, stats = search_best(p_seg, r_seg)
            results.append({'segment': i+1, 'start': int(df.index[start].timestamp()*1000), 'end': int(df.index[end-1].timestamp()*1000), 'best': best, 'stats': stats})

        # choose best by highest pnl; fallback to last segment if tie
        results_sorted = sorted(results, key=lambda x: x['stats']['pnl'], reverse=True)
        chosen = results_sorted[0]
        save_nb_params({ 'buy': chosen['best']['buy'], 'sell': chosen['best']['sell'], 'window': window })
        return jsonify({'ok': True, 'chosen': chosen, 'results': results, 'saved': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/nb/params', methods=['GET', 'POST'])
def api_nb_params():
    try:
        if request.method == 'GET':
            return jsonify({ 'ok': True, 'params': load_nb_params() })
        # POST to manually set/override
        payload = request.get_json(force=True)
        p = load_nb_params()
        for k in ('buy','sell','window'):
            if k in payload:
                p[k] = payload[k]
        ok = save_nb_params(p)
        return jsonify({ 'ok': ok, 'params': p })
    except Exception as e:
        return jsonify({ 'ok': False, 'error': str(e)}), 500


def nb_auto_opt_loop():
    """Background auto-optimizer: periodically updates NB parameters."""
    while True:
        try:
            cfg = load_config()
            # quick grid for development
            payload = {
                'window': load_nb_params().get('window', 50),
                'buy': [0.6, 0.85, 0.025],
                'sell': [0.15, 0.45, 0.025],
                'debounce': 6,
                'fee_bps': 10.0,
                'count': 800,
                'interval': state.get('candle') or cfg.candle,
            }
            # run optimize inline
            try:
                # reuse internal helpers
                df = get_candles(cfg.market, payload['interval'], count=payload['count'])
                r = _compute_r_from_ohlcv(df, payload['window'])
                prices = df['close']
                best=None; best_stats=None
                b=payload['buy'][0]
                while b <= payload['buy'][1] + 1e-9:
                    s=payload['sell'][0]
                    while s <= payload['sell'][1] + 1e-9:
                        stats = _simulate_pnl_from_r(prices, r, b, s, debounce=payload['debounce'], fee_bps=payload['fee_bps'])
                        if best is None or stats['pnl'] > best_stats['pnl']:
                            best={'buy': round(b,3), 'sell': round(s,3)}; best_stats=stats
                        s += payload['sell'][2]
                    b += payload['buy'][2]
                if best:
                    save_nb_params({ 'buy': best['buy'], 'sell': best['sell'], 'window': payload['window'] })
            except Exception:
                pass
        finally:
            # sleep (dev: 10 minutes; configurable via NB_OPT_MIN env)
            mins = int(os.getenv('NB_OPT_MIN', '10'))
            time.sleep(max(60, mins*60))

@app.route('/api/balance')
def api_balance():
    """Return Upbit balances (requires API keys and PAPER=false).
    Uses runtime-resolved config so UI Paper toggle takes effect.
    """
    try:
        cfg = _resolve_config()
        if cfg.paper:
            return jsonify({'ok': True, 'paper': True, 'balances': []})
        # Prefer standard keys from config; otherwise support UPBIT_OPEN_API_* env style (JWT direct call)
        bals = None
        std_ak, std_sk, open_ak, open_sk = _get_runtime_keys()
        if std_ak and std_sk:
            up = pyupbit.Upbit(cfg.access_key, cfg.secret_key)
            bals = up.get_balances()
        else:
            # Try JWT-based private API using env: UPBIT_OPEN_API_ACCESS_KEY, UPBIT_OPEN_API_SECRET_KEY
            ak = open_ak or std_ak
            sk = open_sk or std_sk
            server_url = os.getenv('UPBIT_OPEN_API_SERVER_URL', 'https://api.upbit.com')
            if not ak or not sk:
                return jsonify({'ok': False, 'error': 'missing_keys'}), 400
            try:
                # Lazy import PyJWT
                import jwt as pyjwt  # type: ignore
            except Exception:
                return jsonify({'ok': False, 'error': 'pyjwt_not_installed'}), 500
            payload = {
                'access_key': ak,
                'nonce': str(uuid.uuid4()),
            }
            token = pyjwt.encode(payload, sk, algorithm='HS256')
            headers = { 'Authorization': f'Bearer {token}', 'Accept': 'application/json' }
            resp = requests.get(server_url.rstrip('/') + '/v1/accounts', headers=headers, timeout=10)
            if resp.status_code >= 400:
                return jsonify({'ok': False, 'error': f'upbit_http_{resp.status_code}', 'body': resp.text[:200]}), 400
            try:
                bals = resp.json()
            except Exception as e:
                return jsonify({'ok': False, 'error': f'invalid_json: {e}', 'body': resp.text[:200]}), 500
        cleaned = []
        for b in (bals or []):
            try:
                cleaned.append({
                    'currency': b.get('currency'),
                    'balance': float(b.get('balance', 0) or 0),
                    'locked': float(b.get('locked', 0) or 0),
                    'avg_buy_price': float(b.get('avg_buy_price', 0) or 0),
                    'unit_currency': b.get('unit_currency', 'KRW'),
                })
            except Exception:
                continue
        # Enrich with current KRW price and asset_value
        out = []
        for row in cleaned:
            try:
                cur = (row.get('currency') or '').upper()
                bal = float(row.get('balance') or 0)
                if cur == 'KRW':
                    price = 1.0
                    asset_value = bal
                else:
                    try:
                        price = float(pyupbit.get_current_price(f"KRW-{cur}") or 0.0)
                    except Exception:
                        price = 0.0
                    asset_value = float(bal * price)
                row['price'] = price
                row['asset_value'] = asset_value
                out.append(row)
            except Exception:
                out.append(row)
        return jsonify({'ok': True, 'paper': False, 'balances': out})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/bot/config', methods=['POST'])
def api_bot_config():
    try:
        data = request.get_json(force=True)
        # Optional: reload env vars on demand
        if data.get('reload_env'):
            _reload_env_vars()
        ov = bot_ctrl['cfg_override']
        for k in ('paper','order_krw','pnl_ratio','pnl_profit_ratio','pnl_loss_ratio','ema_fast','ema_slow','candle','market','interval_sec','require_ml',
                  'access_key','secret_key','open_api_access_key','open_api_secret_key'):
            if k in data:
                ov[k] = data[k]
        # reflect into global state for UI
        cfg = _resolve_config()
        state['ema_fast'] = cfg.ema_fast
        state['ema_slow'] = cfg.ema_slow
        state['market'] = cfg.market
        state['candle'] = cfg.candle
        return jsonify({'ok': True, 'config': {
            'paper': cfg.paper,
            'order_krw': cfg.order_krw,
            'pnl_ratio': float(getattr(cfg, 'pnl_ratio', 0.0)),
            'ema_fast': cfg.ema_fast,
            'ema_slow': cfg.ema_slow,
            'candle': cfg.candle,
            'market': cfg.market,
            'interval_sec': cfg.interval_sec,
            'pnl_profit_ratio': float(getattr(cfg, 'pnl_profit_ratio', 0.0)),
            'pnl_loss_ratio': float(getattr(cfg, 'pnl_loss_ratio', 0.0)),
            'has_keys': bool((_get_runtime_keys()[0] and _get_runtime_keys()[1]) or (_get_runtime_keys()[2] and _get_runtime_keys()[3]))
        }})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    if bot_ctrl['running']:
        return jsonify({'ok': True, 'running': True})
    bot_ctrl['running'] = True
    t = threading.Thread(target=trade_loop, daemon=True)
    bot_ctrl['thread'] = t
    t.start()
    return jsonify({'ok': True, 'running': True})


@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    bot_ctrl['running'] = False
    return jsonify({'ok': True, 'running': False})


@app.route('/api/bot/status')
def api_bot_status():
    cfg = _resolve_config()
    # Log masked env keys on each status request for visibility
    try:
        log_env_keys()
    except Exception:
        pass
    return jsonify({
        'running': bot_ctrl['running'],
        'last_signal': bot_ctrl.get('last_signal', 'HOLD'),
        'last_order': bot_ctrl.get('last_order'),
        'config': {
            'paper': cfg.paper,
            'order_krw': cfg.order_krw,
            'pnl_ratio': float(getattr(cfg, 'pnl_ratio', 0.0)),
            'ema_fast': cfg.ema_fast,
            'ema_slow': cfg.ema_slow,
            'candle': cfg.candle,
            'market': cfg.market,
            'interval_sec': cfg.interval_sec,
            'has_keys': bool((_get_runtime_keys()[0] and _get_runtime_keys()[1]) or (_get_runtime_keys()[2] and _get_runtime_keys()[3]))
        }
    })


def run():
    threading.Thread(target=updater, daemon=True).start()
    threading.Thread(target=nb_auto_opt_loop, daemon=True).start()
    use_https = os.getenv("UI_HTTPS", "false").lower() == "true"
    ssl_ctx = 'adhoc' if use_https else None
    app.run(host="127.0.0.1", port=int(os.getenv("UI_PORT", "5057")), ssl_context=ssl_ctx)


if __name__ == "__main__":
    run()


