// Left Panel Trade Logger
// - Records all trades (real/paper) per timeframe in the left panel
// - Enforces one trade per bar per timeframe
// - Closes trade on SELL and summarizes PnL

(function(){
	class LeftPanelTradeLogger {
		constructor() {
			this.storageKey = 'left_panel_trade_logger_v1';
			this.perTimeframe = {}; // { tfKey: { activeTrade, history:[], lastBarTs:number, currentMode:'paper'|'real' } }
			this.currentTfKey = 'left_panel_trade_logger_current_tf';
			this.currentTf = this._loadCurrentTf();
			this._load();
		}

		// Normalize timeframe keys (supports 'minute1'|'1m' etc.)
		static normalizeTimeframe(tf) {
			if (!tf) return '1m';
			const map = {
				'minute1': '1m', '1m': '1m',
				'minute3': '3m', '3m': '3m',
				'minute5': '5m', '5m': '5m',
				'minute10': '10m', '10m': '10m',
				'minute15': '15m', '15m': '15m',
				'minute30': '30m', '30m': '30m',
				'minute60': '1h', '1h': '1h', 'hour1': '1h',
				'day': '1D', '1D': '1D'
			};
			return map[tf] || tf;
		}

		_getBucketTsMs(barTsSecOrMs) {
			if (!barTsSecOrMs) return null;
			const v = Number(barTsSecOrMs);
			if (!isFinite(v)) return null;
			return v < 2e10 ? Math.floor(v) * 1000 : Math.floor(v);
		}

		_tfToSec(tfKey) {
			const tf = LeftPanelTradeLogger.normalizeTimeframe(tfKey);
			switch (tf) {
				case '1m': return 60;
				case '3m': return 180;
				case '5m': return 300;
				case '10m': return 600;
				case '15m': return 900;
				case '30m': return 1800;
				case '1h': return 3600;
				case '1D': return 86400;
				default: return 60;
			}
		}

		_resolveBarTsMs(tfKey, barTsMaybe) {
			const explicit = this._getBucketTsMs(barTsMaybe);
			if (explicit) return explicit;
			const now = Date.now();
			const sec = this._tfToSec(tfKey);
			const bucket = Math.floor(now / (sec * 1000)) * (sec * 1000);
			return bucket;
		}

		_loadCurrentTf() {
			try{
				const v = localStorage.getItem(this.currentTfKey);
				return v ? LeftPanelTradeLogger.normalizeTimeframe(v) : '1m';
			}catch(_){ return '1m'; }
		}

		_saveCurrentTf(tf) {
			try{ localStorage.setItem(this.currentTfKey, LeftPanelTradeLogger.normalizeTimeframe(tf)); }catch(_){ }
		}

		setCurrentTimeframe(tf) {
			this.currentTf = LeftPanelTradeLogger.normalizeTimeframe(tf);
			this._saveCurrentTf(this.currentTf);
		}

		getCurrentTimeframe() {
			return this.currentTf || '1m';
		}

		_syncTfFromBadgeText(txt){
			if (!txt) return;
			const t = String(txt).trim();
			// Map display label to normalized
			const map = { '1m':'1m','3m':'3m','5m':'5m','10m':'10m','15m':'15m','30m':'30m','1h':'1h','1H':'1h','1D':'1D','1d':'1D' };
			const v = map[t] || t;
			this.setCurrentTimeframe(v);
		}

		syncCurrentTimeframeFromBadge(){
			try{
				const candidates = [
					document.getElementById('currentTimeframe'),
					document.getElementById('current-timeframe'),
					document.getElementById('current-timeframe-display')
				].filter(Boolean);
				if (!candidates.length) return;
				for (const badge of candidates){
					const txt = (badge.textContent || badge.innerText || '').trim();
					if (txt){ this._syncTfFromBadgeText(txt); break; }
				}
			}catch(_){ }
		}

		listenToTimeframeEvents(){
			try{
				document.addEventListener('timeframeChanged', (ev)=>{
					try{
						const tf = ev && ev.detail && (ev.detail.label || ev.detail.timeframe);
						if (tf) this._syncTfFromBadgeText(tf);
					}catch(_){ }
				});
			}catch(_){ }
		}

		startBadgeAutoSync(intervalMs=1500){
			try{
				if (this._badgeSyncTimer) return;
				this._badgeSyncTimer = setInterval(()=>{
					try{ this.syncCurrentTimeframeFromBadge(); }catch(_){ }
				}, Math.max(500, intervalMs));
			}catch(_){ }
		}

		_getTfState(tfKey) {
			const key = LeftPanelTradeLogger.normalizeTimeframe(tfKey);
			if (!this.perTimeframe[key]) {
				this.perTimeframe[key] = { activeTrade: null, history: [], lastBarTs: null, currentMode: 'paper' };
			}
			return this.perTimeframe[key];
		}

		// Public summary for UI consumption
		getTimeframeSummary(tfKey) {
			try{
				const key = LeftPanelTradeLogger.normalizeTimeframe(tfKey);
				const st = this._getTfState(key);
				const active = st.activeTrade;
				return {
					timeframe: key,
					mode: st.currentMode || 'paper',
					hasActiveTrade: !!(active && active.status === 'OPEN'),
					activeSide: active ? (active.side || 'BUY') : null,
					entryPrice: active ? Number(active.price || 0) : null,
					size: active ? Number(active.size || 0) : null
				};
			}catch(_){ return { timeframe: LeftPanelTradeLogger.normalizeTimeframe(tfKey), mode:'paper', hasActiveTrade:false }; }
		}

		// Set trade mode for a specific timeframe: 'paper' | 'real'
		setModeForTimeframe(tfKey, mode) {
			try{
				const key = LeftPanelTradeLogger.normalizeTimeframe(tfKey);
				const st = this._getTfState(key);
				st.currentMode = (mode === 'real') ? 'real' : 'paper';
				try{ localStorage.setItem(this.storageKey, JSON.stringify(this.perTimeframe)); }catch(_){ }
				this._renderCard(key);
				this._appendStatus(key, `⚙️ 모드 변경 → ${st.currentMode === 'real' ? '실거래' : '모의전'}`);
				return true;
			}catch(_){ return false; }
		}

		// Attempt to record a BUY. If a trade is already active for this TF, ignore.
		// opts: { timeframe, price, size, paper:boolean, barTs:number(sec|ms) }
		recordBuy(opts) {
			const tf = LeftPanelTradeLogger.normalizeTimeframe(opts?.timeframe || this.getCurrentTimeframe());
			const st = this._getTfState(tf);
			const barTsMs = this._resolveBarTsMs(tf, opts?.barTs);
			if (barTsMs && st.lastBarTs && st.lastBarTs === barTsMs) {
				this._appendStatus(tf, `⛔ 동일 분봉 재거래 차단`);
				return false;
			}
			if (st.activeTrade) {
				this._appendStatus(tf, `⏳ 진행 중 거래 존재`);
				return false;
			}
			st.activeTrade = {
				status: 'OPEN',
				side: 'BUY',
				price: Number(opts?.price) || 0,
				size: Number(opts?.size) || 0,
				paper: !!opts?.paper,
				openTs: Date.now(),
				barTs: barTsMs || null
			};
			st.lastBarTs = barTsMs || st.lastBarTs;
			st.currentMode = opts?.paper ? 'paper' : 'real';
			this._renderCard(tf);
			try{ localStorage.setItem(this.storageKey, JSON.stringify(this.perTimeframe)); }catch(_){ }
			return true;
		}

		// Close with SELL; computes pnl and pushes to history
		// opts: { timeframe, price, barTs:number(sec|ms) }
		recordSell(opts) {
			const tf = LeftPanelTradeLogger.normalizeTimeframe(opts?.timeframe || this.getCurrentTimeframe());
			const st = this._getTfState(tf);
			if (!st.activeTrade || st.activeTrade.status !== 'OPEN') {
				this._appendStatus(tf, `ℹ️ 종료할 거래 없음`);
				return false;
			}
			const barTsMs = this._resolveBarTsMs(tf, opts?.barTs);
			if (barTsMs && st.lastBarTs && st.lastBarTs === barTsMs) {
				this._appendStatus(tf, `⛔ 동일 분봉 재거래 차단 (매도)`);
				return false;
			}
			const exitPrice = Number(opts?.price) || 0;
			const entry = st.activeTrade.price;
			const size = st.activeTrade.size;
			const pnl = (exitPrice - entry) * size;
			const rec = {
				timeframe: tf,
				paper: !!st.activeTrade.paper,
				entryPrice: entry,
				exitPrice: exitPrice,
				size: size,
				pnl: pnl,
				openTs: st.activeTrade.openTs,
				closeTs: Date.now()
			};
			st.history.push(rec);
			st.activeTrade = null;
			st.lastBarTs = barTsMs || st.lastBarTs;
			this._renderCard(tf);
			try{ localStorage.setItem(this.storageKey, JSON.stringify(this.perTimeframe)); }catch(_){ }
			return true;
		}

		// Append a small status line below a card
		_appendStatus(tf, text) {
			const el = this._getCardEl(tf);
			if (!el) return;
			let box = el.querySelector('.tf-log');
			if (!box) {
				box = document.createElement('div');
				box.className = 'tf-log';
				box.style.fontSize = '11px';
				box.style.opacity = '0.85';
				el.appendChild(box);
			}
			const p = document.createElement('div');
			p.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
			box.prepend(p);
			// send to python log server (best-effort)
			try{
				const payload = {
					tf: tf,
					text: text,
					ts: Date.now(),
					mode: (this._getTfState(tf)?.currentMode || 'paper'),
					type: 'status'
				};
				const url = (window.LEFT_PANEL_LOG_ENDPOINT
					|| ((window.location && window.location.origin) ? (window.location.origin + '/api/leftpanel/log') : null)
					|| 'http://127.0.0.1:5057/api/leftpanel/log');
				fetch(url, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(payload)
				}).catch(()=>{});
			}catch(_){ }
		}

		_getCardEl(tf) {
			const tfIdMap = {
				'1m': 'timeframe-card-1m',
				'3m': 'timeframe-card-3m',
				'5m': 'timeframe-card-5m',
				'10m': 'timeframe-card-10m',
				'15m': 'timeframe-card-15m',
				'30m': 'timeframe-card-30m',
				'1h': 'timeframe-card-1h',
				'1D': 'timeframe-card-1D'
			};
			const id = tfIdMap[tf] || '';
			return id ? document.getElementById(id) : null;
		}

		_renderCard(tf) {
			const el = this._getCardEl(tf);
			if (!el) return;
			// Ensure content container exists
			let info = el.querySelector('.tf-info');
			if (!info) {
				info = document.createElement('div');
				info.className = 'tf-info';
				info.style.fontSize = '12px';
				info.style.marginTop = '6px';
				el.appendChild(info);
			}
			const st = this._getTfState(tf);
			const active = st.activeTrade;
			const modeBadge = (st.currentMode === 'real') ? '<span class="badge bg-danger">실거래</span>' : '<span class="badge bg-info text-dark">모의전</span>';
			if (active) {
				const mode = active.paper ? '모의전' : '실거래';
				info.innerHTML = `
					<div>상태: <span class="badge bg-warning text-dark">진행중</span> (${modeBadge})</div>
					<div>매수가: ₩${Math.round(active.price).toLocaleString()} | 수량: ${active.size.toFixed(8)} BTC</div>
				`;
				this._appendStatus(tf, `🟦 매수 체결 (${mode}) @ ₩${Math.round(active.price).toLocaleString()}`);
			} else {
				// Show last history summary if exists
				const last = st.history[st.history.length-1];
				if (last) {
					const mode = last.paper ? '모의전' : '실거래';
					const pnlTxt = `PnL ₩${Math.round(last.pnl).toLocaleString()}`;
					info.innerHTML = `
						<div>상태: <span class="badge bg-secondary">대기</span> (${modeBadge})</div>
						<div>마감: <b>${mode}</b> | ${pnlTxt}</div>
					`;
					this._appendStatus(tf, `🟧 매도 완료 (${mode}) @ ₩${Math.round(last.exitPrice).toLocaleString()} | ${pnlTxt}`);
				} else {
					info.innerHTML = `<div>상태: <span class=\"badge bg-secondary\">대기</span> (${modeBadge})</div>`;
				}
			}
		}

	}

	window.LeftPanelTradeLogger = LeftPanelTradeLogger;
	if (typeof module !== 'undefined' && module.exports) {
		module.exports = LeftPanelTradeLogger;
	}
})();


