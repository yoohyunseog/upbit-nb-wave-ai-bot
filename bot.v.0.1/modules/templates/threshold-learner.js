// 간단한 온라인 임계치 학습기 (로지스틱 기반 확률 → 임계치 맵핑)
class ThresholdLearner {
    constructor(kind = 'buy') {
        this.kind = kind; // 'buy' | 'sell'
        this.weights = {}; // 특징 가중치
        this.bias = 0;
        this.learningRate = 0.05;
        this.enabled = true;
        // ORANGE/BLUE 변화율 계산용 최근 관측값
        this._lastObsTsMs = 0;
        this._lastOrangeVal = 0;
        this._lastBlueVal = 0;
    }

    // 특징 추출: DOM+게임 상태 기반 최소 특징들
    extractFeatures(context = {}) {
        const features = {};
        const priceChangeEl = document.getElementById('right-trading-price-change');
        const zoneStrengthEl = document.getElementById('right-trading-zone-strength');
        const majEl = document.getElementById('majority-zone');
        const orangeEl = document.getElementById('orange-sum');
        const blueEl = document.getElementById('blue-sum');

        const change = (() => {
            const t = priceChangeEl ? priceChangeEl.textContent || '' : '';
            const m = t.match(/-?[\d.]+/);
            return m ? parseFloat(m[0]) : 0;
        })();
        const strength = (() => {
            const t = zoneStrengthEl ? zoneStrengthEl.textContent || '' : '';
            const m = t.match(/\d+/);
            return m ? parseInt(m[0]) : 0;
        })();
        const majority = (majEl ? (majEl.textContent || '').trim().toUpperCase() : '');

        // ORANGE/BLUE 합계의 시간당 변화율(절대값) 계산
        // "체인지가 빠를수록 수익률이 낮아짐" 특징을 반영하기 위해 속도를 특징으로 추가
        const nowMs = Date.now();
        const parseIntSafe = (txt) => {
            if (!txt) return 0;
            const n = parseInt((txt || '0').replace(/[^\d]/g, ''), 10);
            return isNaN(n) ? 0 : n;
        };
        const orangeVal = parseIntSafe(orangeEl ? (orangeEl.textContent || '0') : '0');
        const blueVal = parseIntSafe(blueEl ? (blueEl.textContent || '0') : '0');
        let orangeRatePerSec = 0;
        let blueRatePerSec = 0;
        if (this._lastObsTsMs > 0) {
            const dtSec = Math.max(0.001, (nowMs - this._lastObsTsMs) / 1000);
            orangeRatePerSec = Math.abs(orangeVal - this._lastOrangeVal) / dtSec;
            blueRatePerSec = Math.abs(blueVal - this._lastBlueVal) / dtSec;
        }
        // 상태 갱신
        this._lastObsTsMs = nowMs;
        this._lastOrangeVal = orangeVal;
        this._lastBlueVal = blueVal;

        features.bias = 1;
        features.priceChange = change; // %
        features.zoneStrength = strength; // 0-200 등
        features.isBlue = majority.includes('BLUE') ? 1 : 0;
        features.isOrange = majority.includes('ORANGE') ? 1 : 0;
        if (typeof context.nbCoins === 'number') features.nbCoins = context.nbCoins;
        // 변화율 특징 추가 (값이 클수록 불리한 신호로 학습되도록 가중치가 음수로 수렴 가능)
        features.orangeChangeRate = orangeRatePerSec;
        features.blueChangeRate = blueRatePerSec;
        features.totalChangeRate = orangeRatePerSec + blueRatePerSec;
        return features;
    }

    // 시그모이드
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // 예측: 성공확률 p
    predict(context = {}) {
        const x = this.extractFeatures(context);
        let z = this.bias;
        for (const k of Object.keys(x)) {
            const w = this.weights[k] || 0;
            z += w * x[k];
        }
        return this.sigmoid(z);
    }

    // 온라인 학습: label ∈ {0,1}
    update(label, context = {}) {
        const x = this.extractFeatures(context);
        let z = this.bias;
        for (const k of Object.keys(x)) {
            const w = this.weights[k] || 0;
            z += w * x[k];
        }
        const p = this.sigmoid(z);
        const error = label - p; // 로지스틱 손실의 그라디언트 일부
        // 가중치 업데이트
        for (const k of Object.keys(x)) {
            const w = this.weights[k] || 0;
            this.weights[k] = w + this.learningRate * error * x[k];
        }
        this.bias += this.learningRate * error;
        return p;
    }

    // 임계치 맵핑: 확률 p → 임계치(%)
    // p↑ → 임계치↓ (더 쉽게 매수/매도함), p↓ → 임계치↑
    mapProbToThreshold(p) {
        // 0<=p<=1 → 0.1% ~ 무한대 범위로 맵핑 (p가 작을수록 높게)
        const minThreshold = 0.1; // 하한
        const base = 0.5;
        if (p <= 0.01) return base + 10; // 극저확률 시 매우 높게
        // 역비례형: base / p 형태, 단 하한 유지
        const thr = Math.max(minThreshold, base / p);
        return thr;
    }
}

window.ThresholdLearner = ThresholdLearner;

