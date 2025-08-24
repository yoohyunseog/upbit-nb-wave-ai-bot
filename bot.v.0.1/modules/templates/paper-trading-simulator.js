(function () {
	class PaperTradingSimulator {
		constructor(gameInitializer) {
			this.gameInitializer = gameInitializer;
			this.isRunning = false;
			this.timerId = null;
			// 실제 거래 조건과 동일한 판정 주기
			this.intervalMs = 1000;
			this.tradeHistory = [];
			// 연속 매도 방지용 쿨다운 타이머
			this.lastSellAt = 0;
			// 실제 거래 조건과 동일한 매도 쿨다운
			this.sellCooldownMs = 1500;
			
			// 전역 변수로 설정하여 브라우저에서 접근 가능하게 함
			window.paperTradingDebug = this;
			
			// N/B 코인은 더이상 독립적으로 관리하지 않고 중앙 gameData를 참조
			// gameInitializer의 gameData가 단일 소스가 됨
			if (this.gameInitializer && this.gameInitializer.gameData) {
				// 기존 localStorage 백업 데이터가 있으면 한 번만 복원
				if (this.gameInitializer.gameData.nbCoins === 0) {
					try {
						const savedData = localStorage.getItem('paperTradingData');
						if (savedData) {
							const parsed = JSON.parse(savedData);
							if (parsed.nbCoins > 0) {
								this.gameInitializer.gameData.nbCoins = parsed.nbCoins;
								if (window.logManager) {
									window.logManager.addLog(`💾 Paper Trading: localStorage에서 N/B 코인 ${parsed.nbCoins}개 복원`);
								}
							}
						}
					} catch (e) {
						// localStorage 오류 시 무시
					}
				}
			}
		}

		start() {
			if (this.isRunning) {
				return;
			}
			this.isRunning = true;
			this.timerId = setInterval(() => this.step(), this.intervalMs);
			if (window.logManager) window.logManager.addLog('▶️ PaperTrading 시작');
			
			// 전역 변수로 설정하여 브라우저에서 접근 가능하게 함
			window.paperTradingRunning = true;
		}

		stop() {
			if (!this.isRunning) {
				return;
			}
			clearInterval(this.timerId);
			this.timerId = null;
			this.isRunning = false;
			if (window.logManager) window.logManager.addLog('⏸️ PaperTrading 정지');
			
			// 전역 변수 업데이트
			window.paperTradingRunning = false;
		}

		reset() {
			this.stop();
			this.nbCoins = 0;
			this.tradeHistory = [];
			if (window.logManager) window.logManager.addLog(`🔄 PaperTrading 리셋 (N/B 코인: 0개)`);
		}

		getPriceKrw() {
			const el = document.getElementById('right-trading-current-price') || document.getElementById('selected-coin-price');
			if (!el) return 0;
			const txt = el.textContent || '0';
			const num = parseInt(txt.replace(/[^\d]/g, ''), 10);
			return isNaN(num) ? 0 : num;
		}

		getMajority() {
			const el = document.getElementById('majority-zone');
			if (!el) {
				return '';
			}
			const majority = (el.textContent || '').trim().toUpperCase();
			return majority;
		}

		// 매 프레임 의사결정 → N/B 코인 거래 시뮬레이션
		step() {
			const gi = this.gameInitializer;
			if (!gi) {
				return;
			}
			const price = this.getPriceKrw();
			if (!price || price <= 0) {
				return;
			}
			const majority = this.getMajority();
			
			// 중앙 데이터에서 N/B 코인 개수 확인
			const currentNbCoins = this.gameInitializer?.gameData?.nbCoins || 0;
			
			// 매수 조건: BLUE 신호 + N/B 코인 0개 + 드랍 아이템 0개
			const dropItemsCount = this.gameInitializer?.gameData?.dropItemsCount || 0;
			if (majority.includes('BLUE') && currentNbCoins <= 0 && dropItemsCount <= 0) {
				if (window.logManager) {
					window.logManager.addLog(`🟦 Paper Trading: BLUE 신호 + N/B 코인 0개 + 드랍 아이템 0개 → 매수 실행`);
				}
				this.executeBuyNBCoin(price);
				return;
			} else if (majority.includes('BLUE') && currentNbCoins <= 0 && dropItemsCount > 0) {
				if (window.logManager) {
					window.logManager.addLog(`⏸️ Paper Trading: BLUE 신호 + N/B 코인 0개이지만 드랍 아이템 ${dropItemsCount}개 남음 → 매수 대기`);
				}
				return;
			}
			if (majority.includes('ORANGE') && currentNbCoins > 0) {
				// 연속 매도 방지: 쿨다운 적용
				const now = Date.now();
				if (now - this.lastSellAt < this.sellCooldownMs) {
					return;
				}
				if (window.logManager) {
					window.logManager.addLog(`🟧 Paper Trading: ORANGE 신호 + N/B 코인 ${currentNbCoins}개 → 매도 실행`);
				}
				this.executeSellNBCoin(price);
				return;
			}
		}

		executeBuyNBCoin(price) {
			// N/B MAX COIN을 좌측 패널의 맨 위 카드에만 추가
			if (this.gameInitializer && this.gameInitializer.gameData) {
				const currentNbCoins = this.gameInitializer.gameData.nbCoins;
				
				// 좌측 패널의 맨 위 카드에 N/B MAX COIN 추가
				if (window.gameInitializer && typeof window.gameInitializer.addNBCoinToTopTimeframeCard === 'function') {
					window.gameInitializer.addNBCoinToTopTimeframeCard();
				}
				
				this.tradeHistory.push({ side: 'BUY', price, nbCoins: 1, ts: Date.now() });
				if (window.logManager) window.logManager.addLog(`🟦 N/B MAX COIN 매수 | 가격 ₩${price.toLocaleString()} | 수량 1개 → 좌측 패널 맨 위 카드에 추가`);
				
				// N/B 코인 자동 저장
				this.gameInitializer.saveGameData();
				
				// localStorage에도 백업 저장
				try {
					localStorage.setItem('paperTradingData', JSON.stringify({
						nbCoins: currentNbCoins,
						lastUpdate: Date.now()
					}));
				} catch (e) {
					// localStorage 오류 무시
				}
				
				// N/B 코인 디스플레이 업데이트는 NB Coin Drop System에 위임
				if (window.nbCoinDropSystem && window.nbCoinDropSystem.updateNBCoinDisplay) {
					window.nbCoinDropSystem.updateNBCoinDisplay();
				}
				// 좌측 패널 동기화
				this.syncLeftPanelNbCoinStatus();
			}
		}

		executeSellNBCoin(price) {
			// 1. 좌측 패널의 N/B COIN 확인
			const leftPanelNbCoins = this.getLeftPanelNbCoins();
			
			// 2. 게임 속 트레이너의 N/B COIN 확인
			const gameNbCoins = this.gameInitializer?.gameData?.nbCoins || 0;
			
			// 3. 둘 다 1개 이상 있을 때만 매도 실행
			if (leftPanelNbCoins <= 0 || gameNbCoins <= 0) {
				if (window.logManager) {
					window.logManager.addLog(`⚠️ 매도 조건 불만족: 좌측패널 N/B 코인 ${leftPanelNbCoins}개, 게임 N/B 코인 ${gameNbCoins}개`);
				}
				return;
			}
			
			// 4. 1개씩 매도
			const soldCoins = 1;
			
			// 5. 좌측 패널 N/B COIN -1
			this.decreaseLeftPanelNbCoins();
			
			// 6. 게임 속 N/B MIN COIN -1
			if (this.gameInitializer && this.gameInitializer.gameData) {
				this.gameInitializer.gameData.nbCoins = gameNbCoins - 1;
			}
			
			// 7. 거래 기록 추가
			this.tradeHistory.push({ side: 'SELL', price, nbCoins: soldCoins, ts: Date.now() });
			
			// 8. N/B 미네랄 수익률 평균 반영
			this.addProfitToMineralAverage();
			
			if (window.logManager) {
				window.logManager.addLog(`🟧 N/B MAX COIN 매도 완료 | 가격 ₩${price.toLocaleString()} | 수량 ${soldCoins}개 → 좌측패널: ${leftPanelNbCoins - 1}개, N/B MIN 코인: ${gameNbCoins - 1}개`);
			}
			
			// 9. N/B 코인 자동 저장
			this.gameInitializer.saveGameData();
			
			// 10. localStorage에도 백업 저장
			try {
				localStorage.setItem('paperTradingData', JSON.stringify({
					nbCoins: this.gameInitializer.gameData.nbCoins,
					lastUpdate: Date.now()
				}));
			} catch (e) {
				// localStorage 오류 무시
			}
			
			// 11. N/B 코인 디스플레이 업데이트는 NB Coin Drop System에 위임
			if (window.nbCoinDropSystem && window.nbCoinDropSystem.updateNBCoinDisplay) {
				window.nbCoinDropSystem.updateNBCoinDisplay();
			}
			// 연속 매도 방지 타임스탬프 갱신
			this.lastSellAt = Date.now();
			// 좌측 패널 동기화
			this.syncLeftPanelNbCoinStatus();
		}
		
		// 좌측 패널의 N/B COIN 개수 가져오기
		getLeftPanelNbCoins() {
			try {
				// 좌측 패널에서 N/B 코인 개수 확인
				const nbCoinElement = document.getElementById('nb-coin-count') || document.getElementById('nb-coin-display');
				if (nbCoinElement) {
					const text = nbCoinElement.textContent || '';
					const match = text.match(/(\d+)/);
					return match ? parseInt(match[1]) : 0;
				}
				
				// 대안: N/B 코인 드롭 시스템에서 확인
				if (window.gameInitializer?.gameData?.nbCoins !== undefined) {
					return window.gameInitializer.gameData.nbCoins;
				}
				
				return 0;
			} catch (e) {
				return 0;
			}
		}
		
		// 좌측 패널 N/B COIN 감소
		decreaseLeftPanelNbCoins() {
			try {
				const nbCoinElement = document.getElementById('nb-coin-count') || document.getElementById('nb-coin-display');
				if (nbCoinElement) {
					const currentText = nbCoinElement.textContent || '';
					const match = currentText.match(/(\d+)/);
					if (match) {
						const currentCount = parseInt(match[1]);
						const newCount = Math.max(0, currentCount - 1);
						const newText = currentText.replace(/\d+/, newCount.toString());
						nbCoinElement.textContent = newText;
					}
				}
			} catch (e) {
				// 좌측 패널 업데이트 실패 시 무시
			}
		}
		
		// 좌측 패널의 N/B COIN 상태를 gameData 기준으로 동기화 (타임프레임 카드 새로고침)
		syncLeftPanelNbCoinStatus() {
			try {
				const coins = this.gameInitializer?.gameData?.nbCoins || 0;
				if (!window.nbCoinStatus) window.nbCoinStatus = {};
				Object.keys(window.nbCoinStatus).forEach(tfUi => {
					window.nbCoinStatus[tfUi] = coins > 0 ? 1 : 0;
				});
				// 좌측 패널 카드 강제 갱신 트리거
				try { document.dispatchEvent(new Event('timeframeChanged')); } catch(_) { /* ignore */ }
			} catch (_) { /* ignore */ }
		}
		
		// N/B 미네랄 평균에 수익률 추가 (합계/개수 관리)
		addProfitToMineralAverage() {
			try {
				// 좌측 패널의 수익률 가져오기
				const pnlElement = document.getElementById('selected-coin-pnl');
				if (pnlElement) {
					const pnlText = pnlElement.textContent || '';
					const pnlMatch = pnlText.match(/수익율:\s*([+-]?\d+\.?\d*)%/);
					if (pnlMatch) {
						const currentPnl = parseFloat(pnlMatch[1]);
						
						// N/B 미네랄 평균 계산을 위한 합계/개수 갱신
						if (window.gameInitializer && window.gameInitializer.gameData) {
							const sumPrev = window.gameInitializer.gameData.nbMineralsSum || 0;
							const cntPrev = window.gameInitializer.gameData.nbMineralsCount || 0;
							const sumNew = sumPrev + currentPnl;
							const cntNew = cntPrev + 1;
							window.gameInitializer.gameData.nbMineralsSum = sumNew;
							window.gameInitializer.gameData.nbMineralsCount = cntNew;
							window.gameInitializer.gameData.nbMinerals = sumNew / cntNew;
							
							if (window.logManager) {
								window.logManager.addLog(`💰 N/B 미네랄 평균 갱신: +${currentPnl.toFixed(2)}% → 평균 ${window.gameInitializer.gameData.nbMinerals.toFixed(2)}% (n=${cntNew})`);
							}
							
							// N/B 미네랄 표시 업데이트
							if (window.nbMineralDisplay) {
								const mineralText = `N/B 미네랄(평균): ${window.gameInitializer.gameData.nbMinerals.toFixed(2)}%`;
								window.nbMineralDisplay.setText(mineralText);
							}
						}
					}
				}
			} catch (e) {
				if (window.logManager) {
					window.logManager.addLog(`⚠️ N/B 미네랄 평균 갱신 실패: ${e.message}`);
				}
			}
		}

		// N/B 코인 디스플레이 업데이트 함수 (더이상 사용하지 않음 - 충돌 방지)
		// 모든 디스플레이 업데이트는 NB Coin Drop System에서 중앙 관리

		// 평가 자산 (N/B 코인 기준)
		getEquity(priceOverride) {
			const price = priceOverride || this.getPriceKrw();
			const currentNbCoins = this.gameInitializer?.gameData?.nbCoins || 0;
			return currentNbCoins * price; // N/B 코인 1개당 현재가
		}

		// 브라우저 콘솔에서 수동으로 시작할 수 있는 함수
		manualStart() {
			if (!this.isRunning) {
				this.start();
			}
		}

		// 현재 상태 확인 함수
		getStatus() {
			return {
				isRunning: this.isRunning,
				nbCoins: this.nbCoins,
				gameInitializer: !!this.gameInitializer,
				gameData: !!this.gameInitializer?.gameData,
				nbCoinDisplay: !!window.nbCoinDisplay
			};
		}
	}

	window.PaperTradingSimulator = PaperTradingSimulator;
	
	// 브라우저 콘솔에서 접근할 수 있는 전역 함수들
	window.startPaperTrading = () => {
		if (window.paperTrading) {
			window.paperTrading.manualStart();
			return 'PaperTradingSimulator 시작됨';
		} else {
			return 'PaperTradingSimulator 초기화되지 않음';
		}
	};
	
	window.getPaperTradingStatus = () => {
		if (window.paperTrading) {
			const status = window.paperTrading.getStatus();
			return status;
		} else {
			const error = { error: 'PaperTradingSimulator not initialized' };
			return error;
		}
	};
	
	window.forceNBCoinUpdate = () => {
		if (window.paperTrading) {
			window.paperTrading.updateNBCoinDisplay();
			return 'N/B MIN 코인 업데이트 완료';
		} else {
			return 'PaperTradingSimulator 초기화되지 않음';
		}
	};

	// 추가 디버깅 함수들
	window.debugPaperTrading = () => {
		const debugInfo = {};
		
		// 1. PaperTradingSimulator 존재 확인
		debugInfo.paperTradingClass = !!window.PaperTradingSimulator;
		
		// 2. paperTrading 인스턴스 확인
		debugInfo.paperTradingInstance = !!window.paperTrading;
		
		if (window.paperTrading) {
			debugInfo.paperTradingStatus = window.paperTrading.getStatus();
		}
		
		// 4. majority-zone 요소 확인
		const majorityEl = document.getElementById('majority-zone');
		debugInfo.majorityZoneExists = !!majorityEl;
		if (majorityEl) {
			debugInfo.majorityZoneText = majorityEl.textContent;
			debugInfo.majorityZoneHTML = majorityEl.innerHTML;
		}
		
		// 5. gameInitializer 확인
		debugInfo.gameInitializerExists = !!window.gameInitializer;
		if (window.gameInitializer) {
			debugInfo.gameDataExists = !!window.gameInitializer.gameData;
			debugInfo.nbCoins = window.gameInitializer.gameData?.nbCoins;
		}
		
		// 6. nbCoinDisplay 확인
		debugInfo.nbCoinDisplayExists = !!window.nbCoinDisplay;
		if (window.nbCoinDisplay) {
			debugInfo.nbCoinDisplayText = window.nbCoinDisplay.text;
			debugInfo.nbCoinDisplaySetText = typeof window.nbCoinDisplay.setText;
		}
		
		// 7. 가격 요소 확인
		const priceEl = document.getElementById('right-trading-current-price') || document.getElementById('selected-coin-price');
		debugInfo.priceElementExists = !!priceEl;
		if (priceEl) {
			debugInfo.priceText = priceEl.textContent;
		}
		
		return debugInfo;
	};

	window.forceCreatePaperTrading = () => {
		if (!window.PaperTradingSimulator) {
			return 'PaperTradingSimulator 클래스 없음';
		}
		
		if (!window.gameInitializer) {
			return 'gameInitializer 없음';
		}
		
		// 기존 인스턴스 제거
		if (window.paperTrading) {
			window.paperTrading.stop();
		}
		
		// 새 인스턴스 생성
		window.paperTrading = new window.PaperTradingSimulator(window.gameInitializer);
		
		// 즉시 시작
		window.paperTrading.start();
		return 'PaperTradingSimulator 생성 및 시작 완료';
	};

	window.testNBCoinTrade = () => {
		if (!window.paperTrading) {
			return 'PaperTradingSimulator 없음';
		}
		
		const price = 159000000; // 테스트 가격
		
		// 매수 테스트
		window.paperTrading.executeBuyNBCoin(price);
		
		// 2초 후 매도 테스트
		setTimeout(() => {
			window.paperTrading.executeSellNBCoin(price);
		}, 2000);
		
		return 'N/B 코인 거래 테스트 시작됨';
	};
})();


