(function(){
	function normalizeTf(tf){
		const map = { 'minute1':'1m','1m':'1m','minute3':'3m','3m':'3m','minute5':'5m','5m':'5m','minute10':'10m','10m':'10m','minute15':'15m','15m':'15m','minute30':'30m','30m':'30m','minute60':'1h','1h':'1h','day':'1D','1D':'1D' };
		return map[String(tf||'').trim()] || String(tf||'').trim();
	}

	function readBadgeTf(){
		const ids = ['currentTimeframe','current-timeframe','current-timeframe-display'];
		for (const id of ids){
			const el = document.getElementById(id);
			if (el){
				const txt = (el.textContent || el.innerText || '').trim();
				if (txt) return normalizeTf(txt);
			}
		}
		return null;
	}

	function syncLeftPanelSelection(tf){
		try{
			const list = document.querySelector('.left-panel .timeframe-card-list') || document.getElementById('timeframe-cards-container');
			if (!list) return;
			const cards = list.querySelectorAll('[id^="timeframe-card-"]');
			cards.forEach(c => {
				c.classList.remove('selected');
				const ind = c.querySelector('.this-indicator');
				if (ind) ind.remove();
			});
			const target = list.querySelector(`#timeframe-card-${tf}`);
			if (target) {
				target.classList.add('selected');
				// add a small "현재" indicator
				const tag = document.createElement('div');
				tag.className = 'this-indicator';
				tag.textContent = '현재';
				target.appendChild(tag);
				// move selected card to top using Masonry if available, else prepend
				try{
					if (window.Masonry){
						// ensure Masonry instance
						if (!list._masonry){
							list._masonry = new Masonry(list, { itemSelector: '.card', columnWidth: '.card', percentPosition: true, transitionDuration: '0.2s' });
						}
						list.insertBefore(target, list.firstChild);
						list._masonry.reloadItems();
						list._masonry.layout();
					}else{
						list.insertBefore(target, list.firstChild);
					}
				}catch(_){ list.insertBefore(target, list.firstChild); }
			}
			// update left panel badge text (create if missing)
			try{
				const panel = document.querySelector('.left-panel');
				if (panel){
					let badge = panel.querySelector('#left-panel-current-tf');
					if (!badge){
						badge = document.createElement('div');
						badge.id = 'left-panel-current-tf';
						badge.style.cssText = 'margin-bottom:8px; font-weight:bold; color:#00ff00;';
						panel.insertBefore(badge, panel.firstChild);
					}
					badge.textContent = `현재 분봉: ${tf}`;
				}
			}catch(_){ }
		}catch(_){ }
	}

	let lastTf = null;
	function tick(){
		try{
			const tf = readBadgeTf();
			if (!tf) return;
			if (tf !== lastTf){
				syncLeftPanelSelection(tf);
				try{ window.leftPanelTradeLogger && window.leftPanelTradeLogger.setCurrentTimeframe(tf); }catch(_){ }
				lastTf = tf;
			}
		}catch(_){ }
	}

	function start(){
		try{
			tick();
			setInterval(tick, 1000);
		}catch(_){ }
	}

	if (document.readyState === 'loading'){
		document.addEventListener('DOMContentLoaded', start);
	} else {
		start();
	}
})();


