// Trade Mode Decider
// - 분봉, 예상 수익률(임계치 대비), 불확실성 기준으로 모의전/실거래 결정
// - 기본 정책: 시간이 길고(상위 분봉) 수익률 불확실 → 모의전, 짧고 확실 → 실거래

(function(){
	class TradeModeDecider {
		constructor(options={}){
			this.baseRealBias = typeof options.baseRealBias === 'number' ? options.baseRealBias : 0.3; // 0~1
		}

		static normalizeTimeframe(tf){
			const map = { 'minute1':'1m','1m':'1m','minute3':'3m','3m':'3m','minute5':'5m','5m':'5m','minute10':'10m','10m':'10m','minute15':'15m','15m':'15m','minute30':'30m','30m':'30m','minute60':'1h','1h':'1h','day':'1D','1D':'1D' };
			return map[tf] || (tf || '1m');
		}

		_timeframeSpeedScore(tf){
			// 짧을수록 높음 (실거래 선호), 길수록 낮음 (모의전 선호)
			switch(TradeModeDecider.normalizeTimeframe(tf)){
				case '1m': return 1.0;
				case '3m': return 0.9;
				case '5m': return 0.8;
				case '10m': return 0.6;
				case '15m': return 0.5;
				case '30m': return 0.4;
				case '1h': return 0.3;
				case '1D': return 0.1;
				default: return 0.6;
			}
		}

		_decide(probMargin, speedScore, nbCoins){
			// probMargin: (예상 수익률 - 임계치), 양수 클수록 확실
			// speedScore: 짧은 분봉일수록 큼
			const coinBoost = (typeof nbCoins === 'number' && nbCoins > 0.005) ? 0.15 : ((nbCoins>0.001)?0.08:0);
			let realProb = this.baseRealBias + Math.max(0, probMargin) * 0.6 + speedScore * 0.25 + coinBoost;
			realProb = Math.max(0, Math.min(1, realProb));
			return realProb >= 0.6 ? { mode:'real', score: realProb } : { mode:'paper', score: realProb };
		}

		decideForAction(action, ctx={}){
			const tf = TradeModeDecider.normalizeTimeframe(ctx.timeframe || (window.currentDisplayTimeframe || '1m'));
			const speedScore = this._timeframeSpeedScore(tf);
			const nbCoins = (ctx.nbCoins != null) ? ctx.nbCoins : (window.gameInitializer?.gameData?.nbCoins);
			if (String(action).toUpperCase()==='BUY'){
				const pr = Number(ctx.buyProfitRate ?? window.gameInitializer?.gameData?.buyProfitRate ?? 0);
				const thr = Number(ctx.buyThreshold ?? window.gameInitializer?.gameData?.buyThresholdPercent ?? 0.5);
				const margin = pr - thr;
				const res = this._decide(margin, speedScore, nbCoins);
				return { ...res, reason: `BUY margin=${margin.toFixed(3)} speed=${speedScore.toFixed(2)}` };
			}
			if (String(action).toUpperCase()==='SELL'){
				const pr = Number(ctx.sellProfitRate ?? window.gameInitializer?.gameData?.sellProfitRate ?? 0);
				const thr = Number(ctx.sellThreshold ?? window.gameInitializer?.gameData?.sellThresholdPercent ?? 0.5);
				const margin = pr - thr;
				const res = this._decide(margin, speedScore, nbCoins);
				return { ...res, reason: `SELL margin=${margin.toFixed(3)} speed=${speedScore.toFixed(2)}` };
			}
			return { mode:'paper', score: this.baseRealBias, reason:'unknown action' };
		}
	}

	window.TradeModeDecider = TradeModeDecider;
	if (typeof module !== 'undefined' && module.exports){ module.exports = TradeModeDecider; }
})();


