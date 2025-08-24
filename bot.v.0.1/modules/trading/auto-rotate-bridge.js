(function(){
	function attach(){
		try{
			const btn = document.getElementById('btnAutoRotate');
			if (!btn) return;
			// 이미 TimeframeCards가 버튼 이벤트를 바인딩한다면, 중복 바인딩 방지
			if (window.timeframeCards && typeof window.timeframeCards.toggleAutoRotate === 'function'){
				// 초기 표시만 동기화하고, 클릭 리스너는 추가하지 않음
				try{
					if (!window.timeframeCards.isAutoRotating){
						btn.innerHTML = '<i class="fas fa-play"></i> 자동 순회';
						btn.className = 'btn btn-sm btn-outline-primary';
					}
				}catch(_){ }
				return;
			}

			// Fallback: TimeframeCards 미초기화 시에만 커스텀 순회 토글 바인딩
			if (btn.getAttribute('data-bridge-bound') === '1') return;
			btn.setAttribute('data-bridge-bound','1');
			btn.addEventListener('click', function(){
				try{
					// Fallback: rotate visible left-panel cards if module not initialized
						if (!window.leftPanelAutoRotator){ window.leftPanelAutoRotator = { timer:null, running:false, intervalMs:5000 }; }
						const rot = window.leftPanelAutoRotator;
						if (rot.running){
							try{ clearInterval(rot.timer); }catch(_){ }
							rot.timer = null; rot.running = false;
							btn.innerHTML = '<i class="fas fa-play"></i> 자동 순회';
							btn.className = 'btn btn-sm btn-outline-primary';
						} else {
							rot.timer = setInterval(()=>{
								try{
									const list = document.querySelector('.left-panel .timeframe-card-list');
									if (!list) return;
									const cards = Array.from(list.querySelectorAll('[id^="timeframe-card-"]'));
									if (!cards.length) return;
									const curIdx = Math.max(0, cards.findIndex(c => c.classList.contains('selected')));
									const next = cards[(curIdx + 1) % cards.length];
									if (next && typeof next.click === 'function') next.click();
								}catch(_){ }
							}, rot.intervalMs);
							rot.running = true;
							btn.innerHTML = '<i class="fas fa-pause"></i> 순회 중지';
							btn.className = 'btn btn-sm btn-outline-warning';
						}
				}catch(_){ }
			});
			// initialize button label to stopped state if module not initialized
			try{
				if (!(window.timeframeCards && window.timeframeCards.isAutoRotating)){
					btn.innerHTML = '<i class="fas fa-play"></i> 자동 순회';
					btn.className = 'btn btn-sm btn-outline-primary';
				}
			}catch(_){ }
		}catch(_){ }
	}

	if (document.readyState === 'loading'){
		document.addEventListener('DOMContentLoaded', attach);
	} else {
		attach();
	}
})();


