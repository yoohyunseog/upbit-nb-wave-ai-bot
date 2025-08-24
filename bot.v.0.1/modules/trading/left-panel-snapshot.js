(function(){
	const DEFAULT_ENDPOINT = (window.LEFT_PANEL_SNAPSHOT_ENDPOINT
		|| ((window.location && window.location.origin) ? (window.location.origin + '/api/leftpanel/snapshot') : null)
		|| 'http://127.0.0.1:5061/api/leftpanel/snapshot');

	let lastSnapshotSignature = null;

	function getText(el){
		if (!el) return null;
		return (el.textContent || el.innerText || '').trim();
	}

	function collectLeftPanelData(){
		try{
			const currentTf = (function(){
				const ids = ['currentTimeframe','current-timeframe','current-timeframe-display'];
				for (const id of ids){ const el = document.getElementById(id); if (el){ const t = getText(el); if (t) return t; } }
				return null;
			})();
			const selectedCard = (function(){
				const list = document.querySelector('.left-panel .timeframe-card-list') || document.getElementById('timeframe-cards-container');
				if (!list) return null;
				const sel = list.querySelector('.selected') || null;
				if (!sel) return null;
				const tf = sel.getAttribute('data-timeframe');
				const label = getText(sel.querySelector('.fw-bold')) || tf || null;
				return { tf, label, id: sel.id };
			})();
			const perCards = (function(){
				const list = document.querySelector('.left-panel .timeframe-card-list') || document.getElementById('timeframe-cards-container');
				if (!list) return [];
				const items = [];
				list.querySelectorAll('[id^="timeframe-card-"]').forEach(card => {
					items.push({
						id: card.id,
						tf: card.getAttribute('data-timeframe'),
						label: getText(card.querySelector('.fw-bold')) || null,
						selected: card.classList.contains('selected')
					});
				});
				return items;
			})();

			// Market context from right panel
			function parseNumber(txt){
				if (!txt) return null; const m = String(txt).replace(/[^0-9+\-.]/g,''); if (m === '' || m === '-' || m === '+') return null; const v = Number(m); return isFinite(v)? v: null;
			}
			
			// Normalize zone string
			function normalizeZone(zoneStr) {
				if (!zoneStr || zoneStr === '') return null;
				const z = String(zoneStr).toUpperCase().trim();
				if (z.includes('ORANGE')) return 'ORANGE';
				if (z.includes('BLUE')) return 'BLUE';
				if (z === 'O') return 'ORANGE';
				if (z === 'B') return 'BLUE';
				return z;
			}
			
			const market = (function(){
				const symEl = document.getElementById('selected-coin-name');
				const priceEl = document.getElementById('right-trading-current-price') || document.getElementById('selected-coin-price');
				const chgEl = document.getElementById('right-trading-price-change');
				const zoneEl = document.getElementById('right-trading-current-zone');
				const strEl = document.getElementById('right-trading-zone-strength');
				const rawZone = (getText(zoneEl) || '').trim().toUpperCase() || null;
				return {
					symbol: getText(symEl) || null,
					currentPrice: parseNumber(getText(priceEl)),
					priceChangePct: parseNumber(getText(chgEl)),
					zone: normalizeZone(rawZone),
					zoneStrength: parseNumber(getText(strEl))
				};
			})();
			
			// Assets (must be defined before trade)
			const assets = (function(){
				const krw = parseNumber(getText(document.getElementById('krw-balance')));
				const btc = parseNumber(getText(document.getElementById('btc-balance')));
				const ratio = parseNumber(getText(document.getElementById('portfolio-ratio')));
				return { krwBalance: krw, btcBalance: btc, portfolioRatio: ratio };
			})();

			// Expected returns from game screen display
			function getExpectedReturns(){
				try {
					let buyProfitRate = 0;
					let sellProfitRate = 0;
					let basis = 'zoneStrength'; // 기본값
					
					// 매수 액션이 발생했는지 확인
					const hasBuyAction = window.lastBuyAction === true;
					
					// 매도 액션이 발생했는지 확인
					const hasSellAction = window.lastSellAction === true;
					
					// 1. Try to get from game screen display elements
					if (window.buyProfitRateDisplay && window.buyProfitRateDisplay.text) {
						const buyText = window.buyProfitRateDisplay.text;
						const buyMatch = buyText.match(/매수 전 예상 수익률:\s*([+-]?\d+\.?\d*)%/);
						if (buyMatch) {
							// 실제 매수가 발생했을 때만 매수 전 예상 수익률 반환
							if (hasBuyAction) {
								buyProfitRate = parseFloat(buyMatch[1]);
								basis = 'gameDisplay';
							}
						}
					}
					
					if (window.sellProfitRateDisplay && window.sellProfitRateDisplay.text) {
						const sellText = window.sellProfitRateDisplay.text;
						const sellMatch = sellText.match(/매도 전 예상 수익률:\s*([+-]?\d+\.?\d*)%/);
						if (sellMatch) {
							// 실제 매도가 발생했을 때만 매도 전 예상 수익률 반환
							if (hasSellAction) {
								sellProfitRate = parseFloat(sellMatch[1]);
								basis = 'gameDisplay';
							}
						}
					}
					
					// 2. Try to get from DOM elements if game display not available
					if (buyProfitRate === 0 && hasBuyAction) {
						const buyElement = document.querySelector('[id*="buy"][id*="profit"], [id*="매수"][id*="수익"]');
						if (buyElement) {
							const buyText = buyElement.textContent;
							const buyMatch = buyText.match(/[+-]?\d+\.?\d*%/);
							if (buyMatch) {
								buyProfitRate = parseFloat(buyMatch[0]);
								basis = 'domElement';
							}
						}
					}
					
					if (sellProfitRate === 0 && hasSellAction) {
						const sellElement = document.querySelector('[id*="sell"][id*="profit"], [id*="매도"][id*="수익"]');
						if (sellElement) {
							const sellText = sellElement.textContent;
							const sellMatch = sellText.match(/[+-]?\d+\.?\d*%/);
							if (sellMatch) {
								sellProfitRate = parseFloat(sellMatch[0]);
								basis = 'domElement';
							}
						}
					}
					
					// 3. Try to get from game modules as fallback
					if (buyProfitRate === 0 && hasBuyAction) {
						if (window.currentPriceManager && typeof window.currentPriceManager.calculateBuyProfitRate === 'function') {
							buyProfitRate = window.currentPriceManager.calculateBuyProfitRate();
							basis = 'currentPriceManager';
						}
					}
					
					if (sellProfitRate === 0 && hasSellAction) {
						if (window.currentPriceManager && typeof window.currentPriceManager.calculateSellProfitRate === 'function') {
							sellProfitRate = window.currentPriceManager.calculateSellProfitRate(window.buyPrice || 0);
							basis = 'currentPriceManager';
						}
					}
					
					return { buyProfitRate, sellProfitRate, basis };
				} catch (error) {
					console.error('Expected returns calculation error:', error);
					return { buyProfitRate: 0, sellProfitRate: 0, basis: 'error' };
				}
			}
			const expected = getExpectedReturns();

			// Trade summary from logger for current timeframe
			const trade = (function(){
				try{
					const tfKey = (selectedCard && selectedCard.tf) || currentTf || '1m';
					if (window.leftPanelTradeLogger && typeof window.leftPanelTradeLogger.getTimeframeSummary === 'function'){
						const sum = window.leftPanelTradeLogger.getTimeframeSummary(tfKey) || {};
						// Check if we have BTC balance (can sell) or active trade
						const hasBtcBalance = (assets && assets.btcBalance && assets.btcBalance > 0.00001);
						const hasActiveTrade = sum.hasActiveTrade || hasBtcBalance;
						const nextAction = hasActiveTrade ? 'SELL' : 'BUY';
						return { 
							...sum, 
							hasActiveTrade: hasActiveTrade,
							nextAction: nextAction,
							canSell: hasBtcBalance
						};
					}
				}catch(_){ }
				// Fallback: check BTC balance to determine if we can sell
				const hasBtcBalance = (assets && assets.btcBalance && assets.btcBalance > 0.00001);
				const hasActiveTrade = hasBtcBalance;
				const nextAction = hasActiveTrade ? 'SELL' : 'BUY';
				return { 
					timeframe: (selectedCard&&selectedCard.tf)||currentTf||'1m', 
					mode:'paper', 
					hasActiveTrade: hasActiveTrade, 
					nextAction: nextAction,
					canSell: hasBtcBalance
				};
			})();

			// N/B COIN 상태 추가
			const nbCoinStatus = (function(){
				const nbCoins = {};
				perCards.forEach(card => {
					const tf = card.tf;
					if (tf) {
						// 현재 BTC 보유 상태에 따라 N/B COIN 결정
						const hasBtc = (assets && assets.btcBalance && assets.btcBalance > 0.00001);
						nbCoins[tf] = hasBtc ? 1 : 0;
					}
				});
				return nbCoins;
			})();

			return {
				ts: Date.now(),
				currentTimeframe: currentTf,
				selected: selectedCard,
				cards: perCards,
				market: market,
				expected: expected,
				trade: trade,
				assets: assets,
				nbCoins: nbCoinStatus,
				barBucketTsSec: Date.now()
			};
		}catch(_){ return { ts: Date.now(), error: true }; }
	}

	function computeSignature(snapshot){
		try{
			if (!snapshot || snapshot.error) return 'err';
			const stable = {
				currentTimeframe: snapshot.currentTimeframe || null,
				selected: snapshot.selected ? { id: snapshot.selected.id || null, tf: snapshot.selected.tf || null } : null,
				cards: Array.isArray(snapshot.cards) ? snapshot.cards.map(c => ({ id: c.id || null, sel: !!c.selected })) : [],
				zone: (snapshot.market && snapshot.market.zone) || null,
				strength: (snapshot.market && Math.round(Number(snapshot.market.zoneStrength||0))) || 0,
				next: (snapshot.trade && snapshot.trade.nextAction) || null,
				mode: (snapshot.trade && snapshot.trade.mode) || null,
				has: (snapshot.trade && !!snapshot.trade.hasActiveTrade) || false,
				btc: (snapshot.assets && Math.round(Number(snapshot.assets.btcBalance||0)*10000)) || 0
			};
			return JSON.stringify(stable);
		}catch(_){ return 'err'; }
	}

	async function postSnapshot(endpoint, payload){
		try{
			await fetch(endpoint || DEFAULT_ENDPOINT, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
		}catch(_){ }
	}

	function start(intervalMs){
		const ms = Math.max(2000, Number(intervalMs || 10000));
		setInterval(()=>{
			const data = collectLeftPanelData();
			const sig = computeSignature(data);
			if (sig !== lastSnapshotSignature){
				postSnapshot(window.LEFT_PANEL_SNAPSHOT_ENDPOINT, data);
				lastSnapshotSignature = sig;
			}
		}, ms);
	}

	function boot(){
		try{ start(10000); }catch(_){ }
	}

	if (document.readyState === 'loading'){
		document.addEventListener('DOMContentLoaded', boot);
	} else {
		boot();
	}
})();


