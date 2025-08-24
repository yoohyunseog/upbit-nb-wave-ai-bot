(function(){
	function getSelectedCoinSymbol(){
		try{
			const el = document.getElementById('selected-coin-name');
			if (!el) return 'BTC';
			const txt = (el.textContent || el.innerText || '').trim();
			// Expected like 'BTC/KRW'
			if (txt.includes('/')) return txt.split('/')[0].replace(/[^A-Z]/g,'') || 'BTC';
			if (txt.startsWith('KRW-')) return txt.replace('KRW-','') || 'BTC';
			return txt || 'BTC';
		}catch(_){ return 'BTC'; }
	}

	function tfUiToApi(tf){
		switch(String(tf)){
			case '1m': return 'minute1';
			case '3m': return 'minute3';
			case '5m': return 'minute5';
			case '10m': return 'minute10';
			case '15m': return 'minute15';
			case '30m': return 'minute30';
			case '1h': return 'minute60';
			case '1D': return 'day';
			default: return 'minute10';
		}
	}

	function ensureInfoBox(card){
		let box = card.querySelector('.tf-mini');
		if (!box){
			box = document.createElement('div');
			box.className = 'tf-mini';
			box.style.fontSize = '11px';
			box.style.opacity = '0.9';
			box.style.margin = '6px 8px 4px 8px';
			box.style.lineHeight = '1.25';
			card.appendChild(box);
		}
		return box;
	}

	function computeExpectedReturn(zone, strength){
		// 실제 매수가 발생했을 때만 매수 전 예상 수익률 반환
		// 매수가 발생하지 않았으면 0 반환
		
		// 매수 액션이 발생했는지 확인
		const hasBuyAction = window.lastBuyAction === true;
		
		// 매수가 발생하지 않았으면 0 반환
		if (!hasBuyAction) {
			return 0;
		}
		
		// Try to get from game screen display first
		try {
			// Try to get from game screen display elements
			if (window.buyProfitRateDisplay && window.buyProfitRateDisplay.text) {
				const buyText = window.buyProfitRateDisplay.text;
				const buyMatch = buyText.match(/매수 전 예상 수익률:\s*([+-]?\d+\.?\d*)%/);
				if (buyMatch) {
					const rate = parseFloat(buyMatch[1]);
					if (rate !== 0) return rate;
				}
			}
			
			// Try to get from DOM elements if game display not available
			const buyElement = document.querySelector('[id*="buy"][id*="profit"], [id*="매수"][id*="수익"]');
			if (buyElement) {
				const buyText = buyElement.textContent;
				const buyMatch = buyText.match(/[+-]?\d+\.?\d*%/);
				if (buyMatch) {
					const rate = parseFloat(buyMatch[0]);
					if (rate !== 0) return rate;
				}
			}
			
			// Try to get from game modules as fallback
			if (window.currentPriceManager && typeof window.currentPriceManager.calculateBuyProfitRate === 'function') {
				const rate = window.currentPriceManager.calculateBuyProfitRate();
				if (rate !== 0) return rate;
			}
			
			if (window.profitRateCalculator && typeof window.profitRateCalculator.calculateBasicProfitRate === 'function') {
				const rate = window.profitRateCalculator.calculateBasicProfitRate();
				if (rate > 0) return rate;
			}
			
			if (window.learningSystem && typeof window.learningSystem.calculateAdvancedBuyProfitRate === 'function') {
				const currentPriceText = document.getElementById('right-trading-current-price')?.textContent || '';
				const advancedRate = window.learningSystem.calculateAdvancedBuyProfitRate(currentPriceText);
				if (advancedRate && advancedRate.advancedBuyProfitRate) {
					return advancedRate.advancedBuyProfitRate;
				}
			}
			
			if (window.trainerDecisionHandler && typeof window.trainerDecisionHandler.calculateBuyProfitRate === 'function') {
				const rate = window.trainerDecisionHandler.calculateBuyProfitRate();
				if (rate > 0) return rate;
			}
		} catch(_) { }
		
		// Fallback: Simple heuristic: ORANGE => +strength%, BLUE => -strength%, NEUTRAL => 0
		if (zone === 'ORANGE') {
			return Math.min(strength * 0.1, 5); // 최대 5%
		} else if (zone === 'BLUE') {
			return Math.max(-strength * 0.1, -5); // 최소 -5%
		}
		return 0;
	}

	async function fetchTfData(tfUi){
		const coin = getSelectedCoinSymbol();
		const tf = tfUiToApi(tfUi);
		const url = `/api/nb-wave?timeframe=${encodeURIComponent(tf)}&bars=120&coin=${encodeURIComponent(coin)}`;
		const res = await fetch(url);
		if (!res.ok) throw new Error('nb-wave fetch failed');
		const data = await res.json();
		
		// Add assets data
		try {
			const krw = parseFloat(document.getElementById('krw-balance')?.textContent?.replace(/[^0-9.]/g, '') || '0');
			const btc = parseFloat(document.getElementById('btc-balance')?.textContent?.replace(/[^0-9.]/g, '') || '0');
			const ratio = parseFloat(document.getElementById('portfolio-ratio')?.textContent?.replace(/[^0-9.]/g, '') || '0');
			data.assets = { krwBalance: krw, btcBalance: btc, portfolioRatio: ratio };
		} catch(_) {
			data.assets = { krwBalance: 0, btcBalance: 0, portfolioRatio: 0 };
		}
		
		return data;
	}

	// ===== 매수/매도 쿨다운 상태 관리 (좌측 패널 자동 저장) =====
	const STATE_KEY = 'leftPanelTradeState';
	function loadTradeState(){
		try{
			const raw = localStorage.getItem(STATE_KEY);
			if (!raw) return (window.leftPanelTradeState = { lastBuyTs: 0, lastSellTs: 0 });
			const parsed = JSON.parse(raw);
			if (!parsed || typeof parsed !== 'object') throw new Error('invalid');
			window.leftPanelTradeState = { lastBuyTs: Number(parsed.lastBuyTs)||0, lastSellTs: Number(parsed.lastSellTs)||0 };
			return window.leftPanelTradeState;
		}catch(_){ return (window.leftPanelTradeState = { lastBuyTs:0, lastSellTs:0 }); }
	}

	function saveTradeState(){
		try{ localStorage.setItem(STATE_KEY, JSON.stringify(window.leftPanelTradeState||{ lastBuyTs:0, lastSellTs:0 })); }catch(_){ }
	}

	// 매수 액션 시에만 저장
	function saveBuyAction(){
		if (!window.leftPanelTradeState) window.leftPanelTradeState = { lastBuyTs: 0, lastSellTs: 0 };
		window.leftPanelTradeState.lastBuyTs = Date.now();
		saveTradeState();
		console.log('💾 매수 액션 저장됨');
	}

	// 매도 액션 시에만 저장
	function saveSellAction(){
		if (!window.leftPanelTradeState) window.leftPanelTradeState = { lastBuyTs: 0, lastSellTs: 0 };
		window.leftPanelTradeState.lastSellTs = Date.now();
		saveTradeState();
		console.log('💾 매도 액션 저장됨');
	}

	function getTfMinutes(tfUi){
		switch(String(tfUi)){
			case '1m': return 1;
			case '3m': return 3;
			case '5m': return 5;
			case '10m': return 10;
			case '15m': return 15;
			case '30m': return 30;
			case '1h': return 60;
			case '1D': return 1440;
			default: return 1;
		}
	}

	function fmtMs(ms){
		const s = Math.max(0, Math.floor(ms/1000));
		const m = Math.floor(s/60), sec = s%60;
		return `${m}:${sec.toString().padStart(2,'0')}`;
	}

	function statusForCard(tfUi, nbCoin){
		const st = window.leftPanelTradeState || { lastBuyTs:0, lastSellTs:0 };
		const now = Date.now();
		const tfMin = getTfMinutes(tfUi);
		// 매도 가능: 마지막 매수 이후 tfMin 분 경과 + 코인 보유 중
		let sellAllowed = false, sellRemain = 0, sellElapsed = 0;
		if (st.lastBuyTs > 0){
			sellElapsed = now - st.lastBuyTs;
			const needMs = tfMin * 60 * 1000;
			sellRemain = Math.max(0, needMs - sellElapsed);
			sellAllowed = (sellRemain <= 0) && (nbCoin > 0);
		} else {
			sellAllowed = nbCoin > 0; // 과거 매수 기록 없으면 코인 보유 시 매도 허용
		}
		// 매수 가능: 마지막 매도 이후 1분 경과 + 코인 미보유
		let buyAllowed = false, buyRemain = 0, buyElapsed = 0;
		if (st.lastSellTs > 0){
			buyElapsed = now - st.lastSellTs;
			const needMs = 60 * 1000;
			buyRemain = Math.max(0, needMs - buyElapsed);
			buyAllowed = (buyRemain <= 0) && (nbCoin === 0);
		} else {
			buyAllowed = nbCoin === 0;
		}
		return {
			buy: { allowed: buyAllowed, remainMs: buyRemain, elapsedMs: buyElapsed },
			sell: { allowed: sellAllowed, remainMs: sellRemain, elapsedMs: sellElapsed }
		};
	}

	// N/B COIN 관리 함수
	function getNBCoinStatus(tfUi) {
		try {
			// 전역 N/B COIN 상태 관리
			if (!window.nbCoinStatus) {
				window.nbCoinStatus = {};
			}
			
			// 1) 카드 저장소 시스템에서 가져오기 (우선순위)
			if (window.cardStorageSystem && typeof window.cardStorageSystem.getCardStorage === 'function') {
				const storage = window.cardStorageSystem.getCardStorage(tfUi);
				window.nbCoinStatus[tfUi] = storage.nbCoins || 0;
			} else {
				// 2) 전역 상태에서 해당 분봉의 값 확인
				if (window.nbCoinStatus && window.nbCoinStatus[tfUi] !== undefined) {
					// 이미 설정된 값이 있으면 그대로 사용
					return window.nbCoinStatus[tfUi];
				} else {
					// 3) 대안: 현재 BTC 보유 상태로 추정 (실거래/외부 데이터 기반)
					const btcBalance = parseFloat(document.getElementById('btc-balance')?.textContent?.replace(/[^0-9.]/g, '') || '0');
					const hasBtc = btcBalance > 0.00001;
					window.nbCoinStatus[tfUi] = hasBtc ? 1 : 0;
				}
			}
			
			return window.nbCoinStatus[tfUi] || 0;
		} catch(_) {
			return 0;
		}
	}

	// N/B 미네랄 관리 함수
	function getNBMineralStatus(tfUi) {
		try {
			// 전역 N/B 미네랄 상태 관리
			if (!window.nbMineralStatus) {
				window.nbMineralStatus = {};
			}
			
			// 1) 카드 저장소 시스템에서 가져오기
			if (window.cardStorageSystem && typeof window.cardStorageSystem.getCardStorage === 'function') {
				const storage = window.cardStorageSystem.getCardStorage(tfUi);
				window.nbMineralStatus[tfUi] = storage.nbMinerals || 0.0;
			} else {
				// 2) 게임 중앙 상태가 있으면 사용
				if (window.gameInitializer && window.gameInitializer.gameData && typeof window.gameInitializer.gameData.nbMinerals === 'number') {
					window.nbMineralStatus[tfUi] = window.gameInitializer.gameData.nbMinerals;
				} else {
					window.nbMineralStatus[tfUi] = 0.0;
				}
			}
			
			return window.nbMineralStatus[tfUi] || 0.0;
		} catch(_) {
			return 0.0;
		}
	}

	function renderCard(card, tfUi, data, list){
		try{
			const box = ensureInfoBox(card);
			const summary = data && data.summary ? data.summary : {};
			const zones = data && Array.isArray(data.zones) ? data.zones : [];
			const last = zones.length ? zones[zones.length-1] : null;
			const zone = last ? (last.zone || '').toUpperCase() : '-';
			let strength = last ? Number(last.strength || 0) : 0;
			// 좌측 패널 강도는 우측 패널 DOM 값을 우선 적용
			try {
				const strengthElTxt = (document.getElementById('right-trading-zone-strength')?.textContent || '').trim();
				const m = strengthElTxt.match(/-?\d+/);
				if (m) strength = Number(m[0]);
			} catch(_) { /* ignore */ }
			const price = (summary.current_price != null) ? Number(summary.current_price) : (last ? Number(last.price||0) : 0);
			const exp = computeExpectedReturn(zone, Math.round(strength*100)/100);
			
			// N/B MAX COIN 상태 가져오기
			const nbCoin = getNBCoinStatus(tfUi);
			
			// 맨 위 카드인지 확인 (첫 번째 카드)
			const isTopCard = card === list.querySelector('[id^="timeframe-card-"]');
			
			// N/B MAX COIN 배지와 버튼 생성
			let nbCoinBadge;
			if (isTopCard) {
				// 맨 위 카드에는 +1, -1 버튼 추가
				const canAdd = nbCoin < 1; // 1보다 작을 때만 +1 가능
				const canRemove = nbCoin > 0; // 0보다 클 때만 -1 가능
				
				nbCoinBadge = `
					<div style="display: flex; align-items: center; gap: 4px;">
						${nbCoin > 0 ? `<span class="badge bg-success">N/B MAX COIN: ${nbCoin}</span>` : '<span class="badge bg-secondary">N/B MAX COIN: 0</span>'}
						<button onclick="addNBCoinToTopCard('${tfUi}')" 
							style="padding: 2px 6px; font-size: 10px; background: ${canAdd ? '#28a745' : '#6c757d'}; color: white; border: none; border-radius: 3px; cursor: ${canAdd ? 'pointer' : 'not-allowed'}; opacity: ${canAdd ? '1' : '0.5'};"
							${!canAdd ? 'disabled' : ''}>
							+1
						</button>
						<button onclick="removeNBCoinFromTopCard('${tfUi}')" 
							style="padding: 2px 6px; font-size: 10px; background: ${canRemove ? '#dc3545' : '#6c757d'}; color: white; border: none; border-radius: 3px; cursor: ${canRemove ? 'pointer' : 'not-allowed'}; opacity: ${canRemove ? '1' : '0.5'};"
							${!canRemove ? 'disabled' : ''}>
							-1
						</button>
					</div>
				`;
			} else {
				// 다른 카드들은 기존 배지만 표시
				nbCoinBadge = nbCoin > 0 ? `<span class="badge bg-success">N/B MAX COIN: ${nbCoin}</span>` : '<span class="badge bg-secondary">N/B MAX COIN: 0</span>';
			}
			
			// N/B 미네랄 상태 가져오기
			const nbMineral = getNBMineralStatus(tfUi);
			const nbMineralBadge = nbMineral > 0 ? `<span class="badge bg-warning text-dark">N/B 미네랄: ${nbMineral.toFixed(2)}%</span>` : '<span class="badge bg-secondary">N/B 미네랄: 0.00%</span>';

			// Majority / Orange Total / Blue Total from central DOM
			const majorityText = (document.getElementById('majority-zone')?.textContent || '').trim();
			const orangeSumTxt = (document.getElementById('orange-sum')?.textContent || '0');
			const blueSumTxt = (document.getElementById('blue-sum')?.textContent || '0');
			const orangeSumMatch = String(orangeSumTxt).match(/-?\d+/);
			const blueSumMatch = String(blueSumTxt).match(/-?\d+/);
			const orangeSum = orangeSumMatch ? parseInt(orangeSumMatch[0], 10) : 0;
			const blueSum = blueSumMatch ? parseInt(blueSumMatch[0], 10) : 0;
			
			// trade summary from logger
			let tradeInfo = { hasActiveTrade:false, mode:'paper', activeSide:null };
			try{
				if (window.leftPanelTradeLogger){
					const tfKey = ({'1m':'1m','3m':'3m','5m':'5m','10m':'10m','15m':'15m','30m':'30m','1h':'1h','1D':'1D'})[tfUi] || '1m';
					tradeInfo = window.leftPanelTradeLogger.getTimeframeSummary(tfKey) || tradeInfo;
				}
			}catch(_){ }
			// Check if we have BTC balance (can sell) or active trade
			const hasBtcBalance = (data && data.assets && data.assets.btcBalance && data.assets.btcBalance > 0.00001);
			const hasActiveTrade = tradeInfo.hasActiveTrade || hasBtcBalance;
			const needAction = hasActiveTrade ? 'SELL' : 'BUY';
			const modeBadge = tradeInfo.mode === 'real' ? '<span class="badge bg-danger">실거래</span>' : '<span class="badge bg-info text-dark">모의전</span>';
			const expBuy = (exp >= 0 ? exp : 0).toFixed(2);
			const expSell = (exp < 0 ? Math.abs(exp) : 0).toFixed(2);
			const ob = Number(summary.orange || 0);
			const bb = Number(summary.blue || 0);
			const pc24 = (summary.price_change_24h != null) ? Number(summary.price_change_24h).toFixed(2) : '0.00';
			// 쿨다운/가능 상태 계산
			const st = statusForCard(tfUi, nbCoin);
			const buyText = st.buy.allowed ? '매수 가능' : `매수 대기 (${fmtMs(st.buy.remainMs)})`;
			const sellText = st.sell.allowed ? '매도 가능' : `매도 대기 (${fmtMs(st.sell.remainMs)})`;

			box.innerHTML = `
				<div>가격: ₩${Math.round(price).toLocaleString()} <span style="opacity:.8">(${pc24}%)</span></div>
				<div>구역: <b>${zone}</b> | 강도: ${strength}</div>
				<div>예상수익: <b>${exp}%</b> (1 bar)</div>
				<div>액션: <b>${needAction}</b> (${modeBadge})</div>
				<div>매수전 예상: <b>${expBuy}%</b> | 매도전 예상: <b>${expSell}%</b></div>
				<div>학습요약: O:${ob} / B:${bb}</div>
				<div>Majority: <b>${majorityText || '-'}</b> | Orange Total: <b>${orangeSum}</b> | Blue Total: <b>${blueSum}</b></div>
				<div>상태: <span style="color:${st.buy.allowed?'lime':'#ffcc00'}">${buyText}</span> | <span style="color:${st.sell.allowed?'#ff8800':'#ffcc00'}">${sellText}</span></div>
				<div style="opacity:.85">경과: 매수후 ${st.sell.elapsedMs?fmtMs(st.sell.elapsedMs):'0:00'} | 매도후 ${st.buy.elapsedMs?fmtMs(st.buy.elapsedMs):'0:00'}</div>
				<div>${nbCoinBadge}</div>
				<div>${nbMineralBadge}</div>
			`;
		}catch(_){ }
	}

	async function updateAll(){
		try{
			const list = document.querySelector('.left-panel .timeframe-card-list');
			if (!list) return;
			const cards = Array.from(list.querySelectorAll('[id^="timeframe-card-"]'));
			if (!cards.length) return;
			const tasks = cards.map(async (card) => {
				const tfUi = card.getAttribute('data-timeframe');
				try{
					const data = await fetchTfData(tfUi);
					renderCard(card, tfUi, data, list);
				}catch(_){ /* ignore per-card errors */ }
			});
			await Promise.all(tasks);
		}catch(_){ }
	}

	// 매수/매도 이벤트 감지 및 N/B COIN 상태 업데이트
	function setupTradeEventListeners() {
		try {
			// 매수 버튼 클릭 이벤트 감지
			const buyButtons = document.querySelectorAll('[id*="buy"], [id*="매수"], .buy-btn, .buy-button');
			buyButtons.forEach(btn => {
				btn.addEventListener('click', () => {
					setTimeout(() => {
						// 글로벌 매수 타임스탬프 기록 및 저장
						saveBuyAction();
						// 매수 후 N/B COIN 상태를 1로 설정
						Object.keys(window.nbCoinStatus || {}).forEach(tfUi => {
							window.nbCoinStatus[tfUi] = 1;
						});
						console.log('🪙 매수 감지: N/B COIN 상태를 1로 설정');
						updateAll(); // UI 업데이트
					}, 1000); // 1초 후 상태 확인
				});
			});

			// 매도 버튼 클릭 이벤트 감지
			const sellButtons = document.querySelectorAll('[id*="sell"], [id*="매도"], .sell-btn, .sell-button');
			sellButtons.forEach(btn => {
				btn.addEventListener('click', () => {
					setTimeout(() => {
						// 글로벌 매도 타임스탬프 기록 및 저장
						saveSellAction();
						// 매도 후 N/B COIN 상태를 0으로 설정
						Object.keys(window.nbCoinStatus || {}).forEach(tfUi => {
							window.nbCoinStatus[tfUi] = 0;
						});
						console.log('🪙 매도 감지: N/B COIN 상태를 0으로 설정');
						updateAll(); // UI 업데이트
					}, 1000); // 1초 후 상태 확인
				});
			});

			// DOM 변경 감지 (동적으로 추가되는 버튼들)
			const observer = new MutationObserver((mutations) => {
				mutations.forEach((mutation) => {
					mutation.addedNodes.forEach((node) => {
						if (node.nodeType === 1) { // Element node
							// 새로 추가된 매수/매도 버튼들에 이벤트 리스너 추가
							const newBuyButtons = node.querySelectorAll && node.querySelectorAll('[id*="buy"], [id*="매수"], .buy-btn, .buy-button');
							const newSellButtons = node.querySelectorAll && node.querySelectorAll('[id*="sell"], [id*="매도"], .sell-btn, .sell-button');
							
							if (newBuyButtons) {
								newBuyButtons.forEach(btn => {
									btn.addEventListener('click', () => {
										setTimeout(() => {
											Object.keys(window.nbCoinStatus || {}).forEach(tfUi => {
												window.nbCoinStatus[tfUi] = 1;
											});
											console.log('🪙 매수 감지: N/B COIN 상태를 1로 설정');
											updateAll();
										}, 1000);
									});
								});
							}
							
							if (newSellButtons) {
								newSellButtons.forEach(btn => {
									btn.addEventListener('click', () => {
										setTimeout(() => {
											Object.keys(window.nbCoinStatus || {}).forEach(tfUi => {
												window.nbCoinStatus[tfUi] = 0;
											});
											console.log('🪙 매도 감지: N/B COIN 상태를 0으로 설정');
											updateAll();
										}, 1000);
									});
								});
							}
						}
					});
				});
			});

			// DOM 변경 감지 시작
			observer.observe(document.body, {
				childList: true,
				subtree: true
			});

		} catch(_) {
			console.warn('N/B COIN 이벤트 리스너 설정 실패');
		}
	}

	function boot(){
		// 상태 로드 및 주기 저장
		loadTradeState();
		updateAll();
		setInterval(() => { saveTradeState(); updateAll(); }, 12000);

		// 자동 거래(버튼 미사용) 시에도 시각 기록을 동기화하기 위한 감시자
		(function startNbCoinWatcher(){
			try{
				if (typeof window._prevNbCoinsForCooldown === 'undefined') {
					window._prevNbCoinsForCooldown = (window.gameInitializer?.gameData?.nbCoins || 0);
				}
				setInterval(() => {
					try{
						const current = window.gameInitializer?.gameData?.nbCoins || 0;
						const prev = window._prevNbCoinsForCooldown;
						if (current !== prev){
							loadTradeState();
							if (prev === 0 && current > 0){
								// 매수 발생
								window.leftPanelTradeState.lastBuyTs = Date.now();
								saveTradeState();
								updateAll();
							}else if (prev > 0 && current === 0){
								// 매도 발생
								window.leftPanelTradeState.lastSellTs = Date.now();
								saveTradeState();
								updateAll();
							}
							window._prevNbCoinsForCooldown = current;
						}
					}catch(_){ /* ignore one tick errors */ }
				}, 1000);
			}catch(_){ }
		})();
		
		// N/B COIN 이벤트 리스너 설정
		setupTradeEventListeners();
		
		// Refresh on timeframe change event if emitted by timeframe-cards
		try{
			document.addEventListener('timeframeChanged', ()=>{ try{ updateAll(); }catch(_){ } });
		}catch(_){ }
	}

	if (document.readyState === 'loading'){
		document.addEventListener('DOMContentLoaded', boot);
	} else {
		boot();
	}

	// N/B MAX COIN 버튼 함수들 (비동기 지원)
	window.addNBCoinToTopCard = async function(timeframe) {
		try {
			console.log(`🪙 맨 위 카드 ${timeframe}에 N/B MAX COIN +1 추가 시도`);
			
			// 현재 N/B MAX COIN 상태 확인
			const currentNbCoins = getNBCoinStatus(timeframe);
			
			// 1보다 작을 때만 추가 가능
			if (currentNbCoins >= 1) {
				console.log(`⚠️ ${timeframe} 분봉의 N/B MAX COIN이 이미 ${currentNbCoins}개입니다. 추가하지 않습니다.`);
				return;
			}
			
			// cardStorageSystem을 통해 N/B MAX COIN 추가
			if (window.cardStorageSystem && typeof window.cardStorageSystem.addNBCoin === 'function') {
				const newCount = await window.cardStorageSystem.addNBCoin(timeframe, 1);
				console.log(`✅ ${timeframe} 분봉 N/B MAX COIN +1 추가 완료 → 총 ${newCount}개`);
				
				// UI 업데이트
				updateAll();
			} else {
				console.error('❌ cardStorageSystem을 찾을 수 없습니다.');
			}
		} catch (error) {
			console.error('❌ N/B MAX COIN 추가 중 오류:', error);
		}
	};

	window.removeNBCoinFromTopCard = async function(timeframe) {
		try {
			console.log(`🪙 맨 위 카드 ${timeframe}에서 N/B MAX COIN -1 제거 시도`);
			
			// 현재 N/B MAX COIN 상태 확인
			const currentNbCoins = getNBCoinStatus(timeframe);
			
			// 0보다 클 때만 제거 가능
			if (currentNbCoins <= 0) {
				console.log(`⚠️ ${timeframe} 분봉의 N/B MAX COIN이 이미 ${currentNbCoins}개입니다. 제거하지 않습니다.`);
				return;
			}
			
			// cardStorageSystem을 통해 N/B MAX COIN 제거
			if (window.cardStorageSystem && typeof window.cardStorageSystem.removeNBCoin === 'function') {
				const newCount = await window.cardStorageSystem.removeNBCoin(timeframe, 1);
				console.log(`✅ ${timeframe} 분봉 N/B MAX COIN -1 제거 완료 → 총 ${newCount}개`);
				
				// UI 업데이트
				updateAll();
			} else {
				console.error('❌ cardStorageSystem을 찾을 수 없습니다.');
			}
		} catch (error) {
			console.error('❌ N/B MAX COIN 제거 중 오류:', error);
		}
	};
})();


