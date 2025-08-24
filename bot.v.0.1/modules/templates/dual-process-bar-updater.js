(() => {
	const clamp01 = (v) => Math.min(1, Math.max(0, v));
  
	function getIntById(id) {
	  const el = document.getElementById(id);
	  const text = el?.textContent ?? '0';
	  const m = String(text).match(/-?\d+/);
	  const num = m ? parseInt(m[0], 10) : 0;
	  return Number.isNaN(num) ? 0 : num;
	}
  
	function setWidth(rect, width) {
	  if (!rect) return;
	  // Phaser Rectangle 객체는 width 속성을 직접 사용
	  rect.width = width;
	}

	function setColor(rect, color) {
		if (!rect) return;
		if (typeof rect.setFillStyle === 'function') {
			rect.setFillStyle(color);
		} else if ('fillColor' in rect) {
			rect.fillColor = color;
		}
	}

	// 2배 임계치 계산 함수
	function calculateThresholdRatio(orangeVal, blueVal) {
		// 2배 이상 차이나는지 확인
		const orangeIs2xMore = orangeVal >= blueVal * 2;
		const blueIs2xMore = blueVal >= orangeVal * 2;
		
		// ORANGE가 BLUE의 2배 이상인 경우
		if (orangeIs2xMore) {
			return {
				orangeRatio: 1.0,
				blueRatio: 0.0,
				majority: 'ORANGE',
				thresholdValue: blueVal * 2,
				hasNegative: blueVal < 0,
				is2xDominant: true
			};
		}
		// BLUE가 ORANGE의 2배 이상인 경우
		else if (blueIs2xMore) {
			return {
				orangeRatio: 0.0,
				blueRatio: 1.0,
				majority: 'BLUE',
				thresholdValue: orangeVal * 2,
				hasNegative: orangeVal < 0,
				is2xDominant: true
			};
		}
		
		// 2배 차이가 나지 않는 경우 - 기존 로직 적용
		const total = orangeVal + blueVal;
		const orangeRatio = orangeVal / total;
		const blueRatio = blueVal / total;
		
		// ORANGE가 다수인 경우
		if (orangeVal > blueVal) {
			// ORANGE가 BLUE의 2배일 때 100%가 되도록 계산
			const thresholdOrange = blueVal * 2;
			const thresholdTotal = thresholdOrange + blueVal;
			const actualRatio = orangeVal / thresholdTotal;
			return {
				orangeRatio: clamp01(actualRatio),
				blueRatio: clamp01(blueVal / thresholdTotal),
				majority: 'ORANGE',
				thresholdValue: thresholdOrange,
				hasNegative: false,
				is2xDominant: false
			};
		}
		// BLUE가 다수인 경우
		else if (blueVal > orangeVal) {
			// BLUE가 ORANGE의 2배일 때 100%가 되도록 계산
			const thresholdBlue = orangeVal * 2;
			const thresholdTotal = orangeVal + thresholdBlue;
			const actualRatio = blueVal / thresholdTotal;
			return {
				orangeRatio: clamp01(orangeVal / thresholdTotal),
				blueRatio: clamp01(actualRatio),
				majority: 'BLUE',
				thresholdValue: thresholdBlue,
				hasNegative: false,
				is2xDominant: false
			};
		}
		// 동일한 경우
		else {
			return {
				orangeRatio: 0.5,
				blueRatio: 0.5,
				majority: 'EQUAL',
				thresholdValue: orangeVal,
				hasNegative: false,
				is2xDominant: false
			};
		}
	}
  
	function update(orangeProcessBar, blueProcessBar, opts = {}) {
	  try {
		// 프로세스바 객체 확인
		if (!orangeProcessBar || !blueProcessBar) {
			console.log('❌ 프로세스바 객체 없음:', { orangeProcessBar: !!orangeProcessBar, blueProcessBar: !!blueProcessBar });
			return;
		}

		const { orangeId = 'orange-sum', blueId = 'blue-sum' } = opts;
		const orangeVal = getIntById(orangeId);
		const blueVal = getIntById(blueId);
		
		// 2배 임계치 기반 비율 계산
		const thresholdData = calculateThresholdRatio(orangeVal, blueVal);
		const { orangeRatio, blueRatio, majority, thresholdValue, hasNegative, is2xDominant } = thresholdData;

		// 디버깅: 임계치 계산 결과 확인
		console.log('📏 2배 임계치 계산:', {
			orangeVal, blueVal,
			orangeRatio, blueRatio,
			majority, thresholdValue, hasNegative, is2xDominant,
			orangeLeftWidth: orangeProcessBar.fillLeft?.width || 0,
			orangeRightWidth: orangeProcessBar.fillRight?.width || 0,
			blueWidth: blueProcessBar.fill?.width || 0
		});

		// 트레이너 학습 모델 컨트롤 상태 확인
		const signalCenterProcess = window.signalCenterProcess;
		const isTrainerControlled = signalCenterProcess?.takeover && signalCenterProcess?.followProfitRate;
		const trainerMode = signalCenterProcess?.mode || 'none';
		const trainerProgress = signalCenterProcess?.progress || 0;
		const trainerEnabled = signalCenterProcess?.enabled || false;

		console.log('🤖 트레이너 상태:', {
			isTrainerControlled,
			trainerMode,
			trainerProgress,
			trainerEnabled
		});

		// 트레이너 컨트롤 상태에 따른 색상 설정
		const normalOrangeColor = 0xff8800;
		const normalBlueColor = 0x0088ff;
		const trainerActiveColor = 0x00ff00; // 녹색으로 트레이너 활성화 표시
		const trainerInactiveColor = 0x666666; // 회색으로 비활성화 표시

		if (orangeProcessBar) {
		  // ORANGE bar 업데이트 (좌우 분할)
		  const targetW = orangeProcessBar.width * orangeRatio;
		  
		  // majority에 따라 좌우 배치 결정
		  if (majority === 'ORANGE') {
		  	setWidth(orangeProcessBar.fillLeft, targetW);
		  	setWidth(orangeProcessBar.fillRight, 0);
		  } else if (majority === 'BLUE') {
		  	setWidth(orangeProcessBar.fillLeft, 0);
		  	setWidth(orangeProcessBar.fillRight, targetW);
		  } else {
		  	// EQUAL인 경우 좌우 균등 분할
		  	setWidth(orangeProcessBar.fillLeft, targetW / 2);
		  	setWidth(orangeProcessBar.fillRight, targetW / 2);
		  }
		  
		  console.log('🟠 ORANGE 계산:', { 
		  	orangeRatio, targetW, 
		  	left: orangeProcessBar.fillLeft?.width || 0,
		  	right: orangeProcessBar.fillRight?.width || 0,
		  	majority 
		  });

		  // 트레이너 컨트롤 상태에 따른 색상 변경
		  if (isTrainerControlled && trainerEnabled) {
		  	setColor(orangeProcessBar.fillLeft, trainerActiveColor);
		  	setColor(orangeProcessBar.fillRight, trainerActiveColor);
		  } else if (isTrainerControlled && !trainerEnabled) {
		  	setColor(orangeProcessBar.fillLeft, trainerInactiveColor);
		  	setColor(orangeProcessBar.fillRight, trainerInactiveColor);
		  } else {
		  	setColor(orangeProcessBar.fillLeft, normalOrangeColor);
		  	setColor(orangeProcessBar.fillRight, normalOrangeColor);
		  }

		  // 라벨에 임계치 정보 표시
		  if (orangeProcessBar.label) {
		  	let labelText = 'ORANGE TOTAL →';
		  	if (is2xDominant && majority === 'ORANGE') {
		  		labelText += ` [100% | 2X DOMINANT]`;
		  	} else if (hasNegative) {
		  		labelText += ` [100% | NEGATIVE BLUE]`;
		  	} else if (majority === 'ORANGE') {
		  		labelText += ` [${Math.round(orangeRatio * 100)}% | 2x:${thresholdValue}]`;
		  	} else if (majority === 'BLUE') {
		  		labelText += ` [${Math.round(orangeRatio * 100)}%]`;
		  	}
		  	if (isTrainerControlled) {
		  		labelText += ` [${trainerMode.toUpperCase()}:${Math.round(trainerProgress * 100)}%]`;
		  	}
		  	orangeProcessBar.label.setText(labelText);
		  }
		}
  
		if (blueProcessBar) {
		  // BLUE bar 업데이트
		  const targetWBlue = blueProcessBar.width * blueRatio;
		  
		  console.log('🔵 BLUE 계산:', { blueRatio, targetWBlue, majority });
		  
		  setWidth(blueProcessBar.fill, targetWBlue);

		  // 트레이너 컨트롤 상태에 따른 색상 변경
		  if (isTrainerControlled && trainerEnabled) {
		  	setColor(blueProcessBar.fill, trainerActiveColor);
		  } else if (isTrainerControlled && !trainerEnabled) {
		  	setColor(blueProcessBar.fill, trainerInactiveColor);
		  } else {
		  	setColor(blueProcessBar.fill, normalBlueColor);
		  }

		  // 라벨에 임계치 정보 표시
		  if (blueProcessBar.label) {
		  	let labelText = '← BLUE TOTAL';
		  	if (is2xDominant && majority === 'BLUE') {
		  		labelText = `[100% | 2X DOMINANT] ← ${labelText}`;
		  	} else if (hasNegative) {
		  		labelText = `[0% | NEGATIVE] ← ${labelText}`;
		  	} else if (majority === 'BLUE') {
		  		labelText = `[${Math.round(blueRatio * 100)}% | 2x:${thresholdValue}] ← ${labelText}`;
		  	} else if (majority === 'ORANGE') {
		  		labelText = `[${Math.round(blueRatio * 100)}%] ← ${labelText}`;
		  	}
		  	if (isTrainerControlled) {
		  		labelText = `[${trainerMode.toUpperCase()}:${Math.round(trainerProgress * 100)}%] ← ${labelText}`;
		  	}
		  	blueProcessBar.label.setText(labelText);
		  }
		}

		// 중앙 라벨에 majority 정보 표시
		if (window.dualProcessCenterLabel) {
			let centerText = `<-> ${majority} <->`;
			if (is2xDominant) {
				if (majority === 'ORANGE') {
					centerText = `🟠 ${majority} DOMINANT [100% | 2X STRONG]`;
				} else {
					centerText = `🔵 ${majority} DOMINANT [100% | 2X STRONG]`;
				}
			} else if (hasNegative) {
				if (majority === 'ORANGE') {
					centerText = `🟠 ${majority} DOMINANT [100% | BLUE NEGATIVE]`;
				} else {
					centerText = `🔵 ${majority} DOMINANT [100% | ORANGE NEGATIVE]`;
				}
			} else if (majority === 'ORANGE') {
				centerText = `🟠 ${majority} DOMINANT [${Math.round(orangeRatio * 100)}%]`;
			} else if (majority === 'BLUE') {
				centerText = `🔵 ${majority} DOMINANT [${Math.round(blueRatio * 100)}%]`;
			} else {
				centerText = `⚖️ ${majority} [50%]`;
			}
			if (isTrainerControlled) {
				centerText = `🤖 TRAINER: ${trainerMode.toUpperCase()} ${Math.round(trainerProgress * 100)}%`;
			}
			window.dualProcessCenterLabel.setText(centerText);
		}

		console.log('✅ 프로세스바 업데이트 완료');
	  } catch (err) {
		console.error('❌ 프로세스바 업데이트 오류:', err);
	  }
	}
  
	window.dualProcessBarUpdater = { update };
  })();