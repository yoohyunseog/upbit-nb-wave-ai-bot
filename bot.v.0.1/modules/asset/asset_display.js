// ===== Asset Display Module =====
// 자산 표시 관련 기능들을 담당하는 모듈

class AssetDisplayManager {
    constructor() {
        this.btcElement = document.getElementById('btc-balance');
        this.krwElement = document.getElementById('krw-balance');
        this.portfolioElement = document.getElementById('portfolio-ratio');
        this.balanceList = document.querySelector('.balance-list');
        
        // 선택된 코인 초기화 (기본값: BTC)
        this.selectedCoin = window.selectedKrwCoin || 'BTC';
        
        //console.log('💰 Asset Display Manager initialized with selected coin:', this.selectedCoin);
    }
    
    // 공통 헬퍼 함수: 선택된 코인의 데이터 가져오기
    getSelectedCoinData() {
        const balanceItems = document.querySelectorAll('.balance-item');
        let coinBalance = 0;
        let avgBuyPrice = 0;
        
        balanceItems.forEach(item => {
            const currencyName = item.querySelector('.currency-name')?.textContent.trim();
            
            // 선택된 코인과 일치하고, KRW는 제외
            if (currencyName === this.selectedCoin && currencyName !== 'KRW') {
                const balanceRows = item.querySelectorAll('.balance-row');
                balanceRows.forEach(row => {
                    const label = row.querySelector('.label')?.textContent.trim();
                    const value = row.querySelector('.value')?.textContent.trim();
                    if (label && value) {
                        if (label.includes('보유 수량')) {
                            coinBalance = parseFloat(value.replace(/,/g, '')) || 0;
                        }
                        if (label.includes('평균 매수가')) {
                            avgBuyPrice = parseFloat(value.replace(/[₩,]/g, '')) || 0;
                        }
                    }
                });
            }
        });
        
        return { coinBalance, avgBuyPrice };
    }
    
    // 선택된 코인 업데이트 함수
    updateSelectedCoin(newCoin) {
        this.selectedCoin = newCoin;
        //console.log('🔄 Selected coin updated to:', this.selectedCoin);
        
        // 코인 이름 업데이트
        this.updateCoinDisplayName();
        
        // 코인 아이콘 업데이트
        this.updateCoinIcon();
    }
    
    // 코인 표시 이름 업데이트
    updateCoinDisplayName() {
        if (this.btcElement) {
            const currentText = this.btcElement.textContent;
            const balance = currentText.split(' ')[0]; // 숫자 부분만 추출
            this.btcElement.textContent = `${balance} ${this.selectedCoin}`;
            this.btcElement.title = `${this.selectedCoin} 보유량: ${balance}`;
        }
    }
    
    // 코인 아이콘 업데이트 함수
    updateCoinIcon() {
        const coinIcon = document.querySelector('.asset-item i.fas.fa-bitcoin');
        if (coinIcon) {
            // 코인별 아이콘 매핑
            const iconMap = {
                'BTC': 'fa-bitcoin',
                'ETH': 'fa-ethereum',
                'XRP': 'fa-coins',
                'ADA': 'fa-coins',
                'DOT': 'fa-coins',
                'LINK': 'fa-link',
                'LTC': 'fa-coins',
                'BCH': 'fa-coins',
                'XLM': 'fa-star',
                'EOS': 'fa-coins'
            };
            
            const iconClass = iconMap[this.selectedCoin] || 'fa-coins';
            
            // 기존 클래스 제거하고 새로운 클래스 추가
            coinIcon.className = `fas ${iconClass}`;
            coinIcon.title = `${this.selectedCoin} 보유량`;
            
            //console.log('🎨 Coin icon updated to:', iconClass, 'for', this.selectedCoin);
        }
    }
    
    // 자산 표시 업데이트 함수 (매개변수 기반)
    updateAssetDisplay(coinBalance = 0, krwBalance = 0, portfolioRatio = 0, avgBuyPrice = 0) {
        // 총 평가 금액
        const calculatedKrw = krwBalance || (coinBalance * avgBuyPrice);

        if (this.btcElement) {
            this.btcElement.textContent = `${coinBalance.toFixed(8)} ${this.selectedCoin}`;
            this.btcElement.title = `${this.selectedCoin} 보유량: ${coinBalance.toFixed(8)}`;
        }

        if (this.krwElement) {
            this.krwElement.textContent = `₩${calculatedKrw.toLocaleString()}`;
            this.krwElement.title = `KRW 평가 금액: ₩${calculatedKrw.toLocaleString()}`;
        }

        if (this.portfolioElement) {
            this.portfolioElement.textContent = `${portfolioRatio.toFixed(2)}%`;
            this.portfolioElement.title = `포트폴리오 비율: ${portfolioRatio.toFixed(2)}%`;
        }

        //console.log(`💰 Asset display updated: ${this.selectedCoin} ${coinBalance.toFixed(8)}, KRW ₩${calculatedKrw.toLocaleString()}, Portfolio ${portfolioRatio.toFixed(2)}%`);
    }
    
    // balance-item 업데이트 함수
    updateBalanceItem(coinBalance = 0, avgBuyPrice = 0) {
        // balance-list에서 선택된 코인의 보유 수량을 찾아서 업데이트
        const balanceItems = document.querySelectorAll('.balance-item');
        let targetBalanceItem = null;
        
        // 선택된 코인의 balance-item 찾기
        balanceItems.forEach(item => {
            const currencyName = item.querySelector('.currency-name');
            if (currencyName && currencyName.textContent === this.selectedCoin) {
                targetBalanceItem = item;
            }
        });
        
        if (targetBalanceItem) {
            const balanceRows = targetBalanceItem.querySelectorAll('.balance-row');
            balanceRows.forEach(row => {
                const label = row.querySelector('.label');
                const value = row.querySelector('.value');
                if (label && value) {
                    if (label.textContent.includes('보유 수량')) {
                        value.textContent = coinBalance.toFixed(8);
                    }
                    if (label.textContent.includes('평균 매수가')) {
                        value.textContent = `₩${avgBuyPrice.toLocaleString()}`;
                    }
                }
            });
        } else {
            console.warn(`⚠️ Balance item for ${this.selectedCoin} not found`);
        }
    }
    
    // balance-list 동적 업데이트 함수
    updateBalanceList(balanceData) {
        if (!this.balanceList) {
            console.warn('⚠️ Balance list container not found');
            return;
        }
        
        let balanceHtml = '';
        
        // KRW와 다른 코인들을 구분해서 처리
        balanceData.forEach(asset => {
            const isKRW = asset.currency === 'KRW';
            const currencyName = asset.currency;
            const amount = parseFloat(asset.amount) || 0;
            const avgBuyPrice = parseFloat(asset.avg_buy_price) || 0;
            
            balanceHtml += `
                <div class="balance-item">
                    <div class="currency-header">
                        <div class="currency-info">
                            <span class="currency-name">${currencyName}</span>
                        </div>
                    </div>
                    <div class="currency-details">
                        <div class="balance-row">
                            <span class="label">보유 수량:</span>
                            <span class="value">${isKRW ? amount.toLocaleString() : amount.toFixed(8)}</span>
                        </div>
                        <div class="balance-row">
                            <span class="label">평균 매수가:</span>
                            <span class="value">${isKRW ? '₩1' : `₩${avgBuyPrice.toLocaleString()}`}</span>
                        </div>
                    </div>
                </div>
            `;
        });
        
        this.balanceList.innerHTML = balanceHtml;
        //console.log('💰 Balance list updated with', balanceData.length, 'assets');
    }
    
    // KRW가 아닌 코인 찾기 함수
    findNonKRWAssets(balanceData) {
        return balanceData.filter(asset => asset.currency !== 'KRW');
    }
    
    // 설정된 코인 찾기 함수 (SETTINGS 값에 따라)
    findSelectedCoin(balanceData, selectedCoin = null) {
        const coinToFind = selectedCoin || this.selectedCoin;
        return balanceData.find(asset => asset.currency === coinToFind);
    }
    
    // 자산 데이터에서 asset-display 업데이트
    updateAssetDisplayFromBalanceList(balanceData) {
        const krwAsset = balanceData.find(asset => asset.currency === 'KRW');
        const coinAsset = balanceData.find(asset => asset.currency === this.selectedCoin);
        
        let krwBalance = 0;
        let coinBalance = 0;
        let portfolioRatio = 0;
        
        if (krwAsset) {
            krwBalance = parseFloat(krwAsset.amount) || 0;
        }
        
        if (coinAsset) {
            coinBalance = parseFloat(coinAsset.amount) || 0;
            const avgBuyPrice = parseFloat(coinAsset.avg_buy_price) || 0;
            
            // 포트폴리오 비율 계산 (코인 가치 / 전체 자산)
            const totalValue = krwBalance + (coinBalance * avgBuyPrice);
            if (totalValue > 0) {
                portfolioRatio = ((coinBalance * avgBuyPrice) / totalValue) * 100;
            }
        }
        
        // asset-display 업데이트
        this.updateAssetDisplay(coinBalance, krwBalance, portfolioRatio);
    }
    
    // 전체 자산 시스템 업데이트 함수
    updateAllAssets(balanceData) {
        // balance-list 업데이트
        this.updateBalanceList(balanceData);
        
        // asset-display 업데이트
        this.updateAssetDisplayFromBalanceList(balanceData);
        
        //console.log('🔄 All assets updated for', this.selectedCoin);
    }
    
    // 초기 자산 데이터 설정
    initializeAssetData() {
        // 공통 헬퍼 함수로 선택된 코인의 데이터 가져오기
        const { coinBalance: actualCoinBalance } = this.getSelectedCoinData();
        
        // KRW와 포트폴리오 비율은 기본값 사용 (실제 API에서 가져올 수 있음)
        const mockData = {
            coinBalance: actualCoinBalance,
            krwBalance: 5000000,
            portfolioRatio: 15.75
        };
        
        this.updateAssetDisplay(mockData.coinBalance, mockData.krwBalance, mockData.portfolioRatio);
    }
    
    // balance-item과 asset-display 동기화 함수
    syncBalanceWithAssetDisplay() {
        // 공통 헬퍼 함수로 선택된 코인의 데이터 가져오기
        const { coinBalance: actualCoinBalance, avgBuyPrice: coinPrice } = this.getSelectedCoinData();
        
        // KRW 계산 (코인 가격 * 코인 보유량)
        const calculatedKrw = actualCoinBalance * coinPrice;
        
        // 포트폴리오 비율 계산 (코인 가치 / 전체 자산)
        let portfolioRatio = 0;
        const totalValue = calculatedKrw + (actualCoinBalance * coinPrice);
        if (totalValue > 0) {
            portfolioRatio = ((actualCoinBalance * coinPrice) / totalValue) * 100;
        }
        
        // asset-display 업데이트
        this.updateAssetDisplay(actualCoinBalance, calculatedKrw, portfolioRatio);
        
        /** console.log('🔄 Balance synced with asset display for', this.selectedCoin, ':', {
            coinBalance: actualCoinBalance,
            krwValue: calculatedKrw,
            portfolioRatio: portfolioRatio
        });
		*/
    }
    
    
    // 테스트 함수들
    testAssetDisplayWithDemoData() {
        // 사용자가 제공한 샘플 데이터
        const demoData = {
            coinBalance: 0.00021472,
            avgBuyPrice: 163019086.083,
            portfolioRatio: 99.12
        };
        
        // balance-item 업데이트
        this.updateBalanceItem(demoData.coinBalance, demoData.avgBuyPrice);
        
        // asset-display 업데이트
        this.updateAssetDisplay(demoData.coinBalance, demoData.coinBalance * demoData.avgBuyPrice, demoData.portfolioRatio);
        
        //console.log('🎯 Demo data applied for', this.selectedCoin, ':', demoData);
    }
    
    testBalanceListSystem() {
        const testBalanceData = [
            {
                currency: 'KRW',
                amount: '300.762',
                avg_buy_price: '1'
            },
            {
                currency: this.selectedCoin,
                amount: '0.00021472',
                avg_buy_price: '163019086.083'
            }
        ];
        
        // 전체 자산 시스템 업데이트
        this.updateAllAssets(testBalanceData);
        
        // KRW가 아닌 자산 찾기
        const nonKRWAssets = this.findNonKRWAssets(testBalanceData);
        //console.log('🔍 Non-KRW assets:', nonKRWAssets);
        
        // 선택된 코인 찾기
        const selectedCoin = this.findSelectedCoin(testBalanceData);
        //console.log('🎯 Selected coin:', selectedCoin);
        
        //console.log('🧪 Balance list system test completed for', this.selectedCoin);
    }
}

// 자동 갱신을 위한 renderAssetDisplay 함수
function renderAssetDisplay() {
    try {
        // 잔고 개요 영역에서 값 가져오기
        const krwValue = document.querySelector('.balance-card.krw .balance-value')?.innerText || "₩0";
        const btcValue = document.querySelector('.balance-card.btc .balance-value')?.innerText || "₩0";
        const totalValue = document.querySelector('.balance-card.total .balance-value')?.innerText || "₩0";

        // 선택된 코인 보유량 (상세 데이터에서 가져오기)
        const selectedCoin = window.assetDisplayManager?.selectedCoin || 'BTC';
        const coinAmountElement = [...document.querySelectorAll('.balance-item')].find(item =>
            item.querySelector('.currency-name')?.innerText === selectedCoin
        );
        const coinAmount = coinAmountElement
            ? coinAmountElement.querySelector('.currency-details .value')?.innerText || "0"
            : "0";

        // 포트폴리오 비율 계산 (함수 스코프 상단에서 정의)
        let ratio = 0;
        const coinNumeric = parseFloat(btcValue.replace(/[₩,]/g, ""));
        const totalNumeric = parseFloat(totalValue.replace(/[₩,]/g, ""));
        if (totalNumeric > 0) {
            ratio = (coinNumeric / totalNumeric) * 100;
        }

        // 자산 표시 영역에 값 업데이트
        const btcElement = document.getElementById("btc-balance");
        if (btcElement) {
            btcElement.textContent = `${coinAmount} ${selectedCoin}`;
            btcElement.title = `${selectedCoin} 보유량: ${coinAmount}`;
        }

        const krwElement = document.getElementById("krw-balance");
        if (krwElement) {
            krwElement.textContent = krwValue;
            krwElement.title = `KRW 보유량: ${krwValue}`;
        }

        const portfolioElement = document.getElementById("portfolio-ratio");
        if (portfolioElement) {
            portfolioElement.textContent = `${ratio.toFixed(2)}%`;
            portfolioElement.title = `포트폴리오 비율: ${ratio.toFixed(2)}%`;
        }
        
        //console.log(`🔄 Asset display auto-updated: ${selectedCoin} ${coinAmount}, KRW ${krwValue}, Portfolio ${ratio.toFixed(2)}%`);
    } catch (err) {
        console.error("renderAssetDisplay 오류:", err);
    }
}

// 로컬 스토리지에 데이터 저장
function saveToLocalStorage(data) {
    try {
        // Trading Dashboard Info 데이터도 함께 저장
        const tradingData = {
            currentPrice: document.getElementById('right-trading-current-price')?.textContent || '₩0',
            priceChange: document.getElementById('right-trading-price-change')?.textContent || '0.00%',
            currentZone: document.getElementById('right-trading-current-zone')?.textContent || 'Loading...',
            zoneStrength: document.getElementById('right-trading-zone-strength')?.textContent || '강도: 0'
        };
        
        const storageData = {
            ...data,
            tradingData: tradingData,
            timestamp: new Date().toISOString(),
            lastUpdate: new Date().toLocaleTimeString("ko-KR")
        };
        localStorage.setItem('selectedCoinStatus', JSON.stringify(storageData));
        //console.log('💾 Data saved to localStorage:', storageData);
    } catch (err) {
        console.error("saveToLocalStorage 오류:", err);
    }
}

// 로컬 스토리지에서 데이터 복원
function loadFromLocalStorage() {
    try {
        const savedData = localStorage.getItem('selectedCoinStatus');
        if (savedData) {
            const data = JSON.parse(savedData);
            //console.log('📂 Data loaded from localStorage:', data);
            return data;
        }
    } catch (err) {
        console.error("loadFromLocalStorage 오류:", err);
    }
    return null;
}

// 선택된 코인 상태 업데이트 함수
function updateSelectedCoinStatus(data) {
    try {
        // data 예시:
        // {
        //   coin: "BTC/KRW",
        //   balance: 0.00021472,
        //   value: 33842.449,
        //   price: 157527000,
        //   pnl: -3.80,
        //   avgPrice: 163019086.083
        // }

        // 코인 이름
        const coinNameElement = document.getElementById("selected-coin-name");
        if (coinNameElement) {
            coinNameElement.textContent = data.coin;
        }

        // 보유 수량
        const balanceElement = document.getElementById("selected-coin-balance");
        if (balanceElement) {
            balanceElement.textContent = data.balance.toFixed(8);
        }

        // 코인 가치
        const valueElement = document.getElementById("selected-coin-value");
        if (valueElement) {
            valueElement.textContent = "₩" + data.value.toLocaleString();
        }

        // 현재 가격
        const priceElement = document.getElementById("selected-coin-price");
        if (priceElement) {
            priceElement.textContent = "₩" + data.price.toLocaleString();
        }

        // 수익률
        const pnlElement = document.getElementById("selected-coin-pnl");
        if (pnlElement) {
            pnlElement.textContent = `수익율: ${data.pnl.toFixed(2)}%`;
            pnlElement.style.color = data.pnl >= 0 ? "limegreen" : "red";
        }

        // 평균 단가
        const avgPriceElement = document.getElementById("selected-coin-avg-price");
        if (avgPriceElement) {
            avgPriceElement.textContent = "평균단가: ₩" + data.avgPrice.toLocaleString();
        }

        // 현재 시간 업데이트
        const timeElement = document.getElementById("current-time");
        if (timeElement) {
            const now = new Date();
            timeElement.textContent = now.toLocaleTimeString("ko-KR");
        }

        // Trading Dashboard Info 복원 (저장된 데이터가 있는 경우)
        if (data.tradingData) {
            const tradingData = data.tradingData;
            
            const currentPriceElement = document.getElementById('right-trading-current-price');
            if (currentPriceElement) {
                currentPriceElement.textContent = tradingData.currentPrice;
            }
            
            const priceChangeElement = document.getElementById('right-trading-price-change');
            if (priceChangeElement) {
                priceChangeElement.textContent = tradingData.priceChange;
                // 색상 설정
                if (tradingData.priceChange.includes('-')) {
                    priceChangeElement.className = 'price-change negative';
                } else {
                    priceChangeElement.className = 'price-change positive';
                }
            }
            
            const currentZoneElement = document.getElementById('right-trading-current-zone');
            if (currentZoneElement) {
                currentZoneElement.textContent = tradingData.currentZone;
                // 구역별 색상 설정
                if (tradingData.currentZone.includes('Blue')) {
                    currentZoneElement.className = 'current-zone zone-blue';
                } else if (tradingData.currentZone.includes('Orange')) {
                    currentZoneElement.className = 'current-zone zone-orange';
                }
            }
            
            const zoneStrengthElement = document.getElementById('right-trading-zone-strength');
            if (zoneStrengthElement) {
                zoneStrengthElement.textContent = tradingData.zoneStrength;
            }
        }

        // 로컬 스토리지에 저장
        saveToLocalStorage(data);

        //console.log(`📊 Selected coin status updated: ${data.coin}, Balance: ${data.balance}, PnL: ${data.pnl}%`);
    } catch (err) {
        console.error("updateSelectedCoinStatus 오류:", err);
    }
}

// 통합 자동 갱신 함수
function autoUpdateAllAssetData() {
    try {
        // 기본 자산 표시 업데이트
        renderAssetDisplay();
        
        // btc-balance에서 실제 값 가져오기
        const btcElement = document.getElementById('btc-balance');
        let actualBalance = 0;
        
        if (btcElement) {
            // title 속성에서 더 정확한 값 가져오기 (예: "BTC 보유량: 0.00021472")
            const titleText = btcElement.title;
            const titleMatch = titleText.match(/보유량:\s*(\d+\.?\d*)/);
            
            if (titleMatch) {
                actualBalance = parseFloat(titleMatch[1]);
            } else {
                // title이 없으면 textContent에서 추출
                const balanceText = btcElement.textContent;
                const balanceMatch = balanceText.match(/(\d+\.?\d*)/);
                if (balanceMatch) {
                    actualBalance = parseFloat(balanceMatch[1]);
                }
            }
        }
        
        // balance-card btc에서 실제 가치 가져오기
        const btcValueElement = document.querySelector('.balance-card.btc .balance-value');
        let actualValue = 0;
        
        if (btcValueElement) {
            const valueText = btcValueElement.textContent;
            // "₩33,864.78" 형태에서 숫자 부분만 추출
            const valueMatch = valueText.match(/₩([\d,]+\.?\d*)/);
            if (valueMatch) {
                actualValue = parseFloat(valueMatch[1].replace(/,/g, ''));
            }
        }
        
        // 현재 가격 계산 (가치 / 보유량)
        let currentPrice = 0;
        if (actualBalance > 0 && actualValue > 0) {
            currentPrice = actualValue / actualBalance;
        }
        
        // 선택된 코인 상태 업데이트 (실제 btc-balance 값, btc 가치, 평균 매수가, 수익률 사용)
        const selectedCoin = window.selectedKrwCoin || 'BTC';
        
        // balance-item에서 평균 매수가 가져오기
        const balanceItems = document.querySelectorAll('.balance-item');
        let actualAvgPrice = 0;
        
        balanceItems.forEach(item => {
            const currencyName = item.querySelector('.currency-name')?.textContent.trim();
            if (currencyName === selectedCoin) {
                const balanceRows = item.querySelectorAll('.balance-row');
                balanceRows.forEach(row => {
                    const label = row.querySelector('.label')?.textContent.trim();
                    const value = row.querySelector('.value')?.textContent.trim();
                    if (label && value && label.includes('평균 매수가')) {
                        // "₩163,019,086.083" 형태에서 숫자 추출
                        const avgPriceMatch = value.match(/₩([\d,]+\.?\d*)/);
                        if (avgPriceMatch) {
                            actualAvgPrice = parseFloat(avgPriceMatch[1].replace(/,/g, ''));
                        }
                    }
                });
            }
        });
        
        // 수익률 계산 (현재가격 - 평균매수가) / 평균매수가 × 100
        let pnl = 0;
        if (actualAvgPrice > 0 && currentPrice > 0) {
            pnl = ((currentPrice - actualAvgPrice) / actualAvgPrice) * 100;
        }
        const sampleData = {
            coin: `${selectedCoin}/KRW`,
            balance: actualBalance,
            value: actualValue,
            price: currentPrice,
            pnl: pnl,
            avgPrice: actualAvgPrice
        };
        
        updateSelectedCoinStatus(sampleData);
        
        /** console.log("🔄 All asset data auto-updated:", {
            balance: actualBalance,
            value: actualValue,
            price: currentPrice,
            avgPrice: actualAvgPrice,
            pnl: pnl
        });
		*/
    } catch (err) {
        console.error("autoUpdateAllAssetData 오류:", err);
    }
}

// 3초마다 자동 갱신 (통합 함수 사용)
setInterval(autoUpdateAllAssetData, 3000);

// 페이지 로드 시 저장된 데이터 복원 및 1회 실행
document.addEventListener("DOMContentLoaded", () => {
    // 저장된 데이터가 있으면 복원
    const savedData = loadFromLocalStorage();
    if (savedData) {
        //console.log('🔄 Restoring saved data on page load...');
        updateSelectedCoinStatus(savedData);
    }
    
    // 자동 업데이트 실행
    autoUpdateAllAssetData();
});

// 전역 인스턴스 생성
window.assetDisplayManager = new AssetDisplayManager();

// SETTINGS 변경 감지 및 자동 동기화
document.addEventListener('DOMContentLoaded', () => {
    // 초기 코인 표시 설정
    if (window.assetDisplayManager) {
        window.assetDisplayManager.updateCoinDisplayName();
        window.assetDisplayManager.updateCoinIcon();
    }
    
    // SETTINGS에서 코인 변경 감지
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-selected-coin') {
                const newCoin = mutation.target.getAttribute('data-selected-coin');
                if (newCoin && window.assetDisplayManager) {
                    window.assetDisplayManager.updateSelectedCoin(newCoin);
                }
            }
        });
    });
    
    // body 요소 관찰 시작
    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-selected-coin']
    });
    
    //console.log('🔄 Asset Display coin synchronization initialized');
});

// SETTINGS에서 직접 호출할 수 있는 함수
window.syncAssetDisplayWithSettings = (selectedCoin) => {
    if (window.assetDisplayManager) {
        window.assetDisplayManager.updateSelectedCoin(selectedCoin);
        //console.log('🔄 Asset Display synced with settings:', selectedCoin);
    }
};

// 기존 함수들과의 호환성을 위한 래퍼 함수들
window.updateAssetDisplay = (coinBalance, krwBalance, portfolioRatio) => {
    window.assetDisplayManager.updateAssetDisplay(coinBalance, krwBalance, portfolioRatio);
};

window.updateBalanceItem = (coinBalance, avgBuyPrice) => {
    window.assetDisplayManager.updateBalanceItem(coinBalance, avgBuyPrice);
};

window.updateBalanceList = (balanceData) => {
    window.assetDisplayManager.updateBalanceList(balanceData);
};

window.findNonKRWAssets = (balanceData) => {
    return window.assetDisplayManager.findNonKRWAssets(balanceData);
};

window.findSelectedCoin = (balanceData, selectedCoin) => {
    return window.assetDisplayManager.findSelectedCoin(balanceData, selectedCoin);
};

window.updateAssetDisplayFromBalanceList = (balanceData) => {
    window.assetDisplayManager.updateAssetDisplayFromBalanceList(balanceData);
};

window.updateAllAssets = (balanceData) => {
    window.assetDisplayManager.updateAllAssets(balanceData);
};

window.testAssetDisplayWithDemoData = () => {
    window.assetDisplayManager.testAssetDisplayWithDemoData();
};

window.testBalanceListSystem = () => {
    window.assetDisplayManager.testBalanceListSystem();
};

// 선택된 코인 업데이트 함수 추가
window.updateSelectedCoin = (newCoin) => {
    window.assetDisplayManager.updateSelectedCoin(newCoin);
};

// 초기화 및 동기화 함수들
window.initializeAssetData = () => {
    window.assetDisplayManager.initializeAssetData();
};

window.syncBalanceWithAssetDisplay = () => {
    window.assetDisplayManager.syncBalanceWithAssetDisplay();
};

// 새로운 전역 함수들
window.updateSelectedCoinStatus = updateSelectedCoinStatus;

window.autoUpdateAllAssetData = autoUpdateAllAssetData;

// 로컬 스토리지 관리 함수들
window.clearSavedData = () => {
    try {
        localStorage.removeItem('selectedCoinStatus');
        //console.log('🗑️ Saved data cleared from localStorage');
        return true;
    } catch (err) {
        console.error('❌ Failed to clear saved data:', err);
        return false;
    }
};

window.getSavedData = () => {
    return loadFromLocalStorage();
};

window.forceSaveCurrentData = () => {
    try {
        // 현재 화면의 데이터를 수집하여 저장
        const currentData = {
            coin: document.getElementById("selected-coin-name")?.textContent || "BTC/KRW",
            balance: parseFloat(document.getElementById("selected-coin-balance")?.textContent || "0"),
            value: parseFloat(document.getElementById("selected-coin-value")?.textContent.replace(/[₩,]/g, "") || "0"),
            price: parseFloat(document.getElementById("selected-coin-price")?.textContent.replace(/[₩,]/g, "") || "0"),
            pnl: parseFloat(document.getElementById("selected-coin-pnl")?.textContent.match(/-?\d+\.?\d*/)?.[0] || "0"),
            avgPrice: parseFloat(document.getElementById("selected-coin-avg-price")?.textContent.replace(/[₩,]/g, "") || "0")
        };
        
        saveToLocalStorage(currentData);
        //console.log('💾 Current data manually saved:', currentData);
        return true;
    } catch (err) {
        console.error('❌ Failed to save current data:', err);
        return false;
    }
};

// 수동으로 선택된 코인 상태 업데이트하는 함수
window.forceUpdateSelectedCoinStatus = (customData = null) => {
    const data = customData || {
        coin: "BTC/KRW",
        balance: 0.00021472,
        value: 33842.449,
        price: 157527000,
        pnl: -3.80,
        avgPrice: 163019086.083
    };
    
    //console.log('🚀 Force updating selected coin status...');
    updateSelectedCoinStatus(data);
    //console.log('✅ Selected coin status updated');
};

// 실시간 데이터로 업데이트하는 함수 (API 연동용)
window.updateWithRealTimeData = (apiData) => {
    try {
        // API 데이터를 updateSelectedCoinStatus 형식으로 변환
        const formattedData = {
            coin: apiData.currency || "BTC/KRW",
            balance: parseFloat(apiData.balance) || 0,
            value: parseFloat(apiData.value) || 0,
            price: parseFloat(apiData.current_price) || 0,
            pnl: parseFloat(apiData.pnl_percentage) || 0,
            avgPrice: parseFloat(apiData.avg_buy_price) || 0
        };
        
        updateSelectedCoinStatus(formattedData);
        //console.log('📡 Real-time data applied:', formattedData);
    } catch (error) {
        console.error('❌ Failed to update with real-time data:', error);
    }
};

// 디버그 및 테스트 함수
window.debugAssetDisplay = () => {
    //console.log('🔍 Asset Display Debug Info:');
    //console.log('- assetDisplayManager exists:', !!window.assetDisplayManager);
    
    if (window.assetDisplayManager) {
        //console.log('- selectedCoin:', window.assetDisplayManager.selectedCoin);
        //console.log('- btcElement exists:', !!window.assetDisplayManager.btcElement);
        //console.log('- krwElement exists:', !!window.assetDisplayManager.krwElement);
        //console.log('- portfolioElement exists:', !!window.assetDisplayManager.portfolioElement);
        
        // DOM 요소 확인
        const btcElement = document.getElementById('btc-balance');
        const krwElement = document.getElementById('krw-balance');
        const portfolioElement = document.getElementById('portfolio-ratio');
        
        //console.log('- DOM btc-balance exists:', !!btcElement);
        //console.log('- DOM krw-balance exists:', !!krwElement);
        //console.log('- DOM portfolio-ratio exists:', !!portfolioElement);
        
        //if (btcElement) console.log('- Current btc-balance text:', btcElement.textContent);
        //if (krwElement) console.log('- Current krw-balance text:', krwElement.textContent);
        //if (portfolioElement) console.log('- Current portfolio-ratio text:', portfolioElement.textContent);
    }
    
    // 테스트 업데이트 실행
    if (window.assetDisplayManager) {
        //console.log('🧪 Running test update...');
        window.assetDisplayManager.updateAssetDisplay(0.12345678, 5000000, 25.5);
        //console.log('✅ Test update completed');
    }
};

// 즉시 실행 테스트
window.testAssetDisplayImmediate = () => {
    //console.log('🚀 Immediate Asset Display Test');
    if (window.assetDisplayManager) {
        window.assetDisplayManager.updateAssetDisplay(1.23456789, 10000000, 50.0);
        //console.log('✅ Immediate test completed');
    } else {
        console.error('❌ assetDisplayManager not found');
    }
};

//console.log('✅ Asset Display Module loaded with coin synchronization');
//console.log('💡 Debug commands available:');
//console.log('  - debugAssetDisplay() - Show debug info');
//console.log('  - testAssetDisplayImmediate() - Run immediate test');

// 추가 디버그 함수들
window.checkAssetDisplayStatus = () => {
    //console.log('🔍 Asset Display Status Check:');
    
    // 1. DOM 요소 존재 확인
    const btcElement = document.getElementById('btc-balance');
    const krwElement = document.getElementById('krw-balance');
    const portfolioElement = document.getElementById('portfolio-ratio');
    
    //console.log('DOM Elements:');
    //console.log('- btc-balance exists:', !!btcElement);
    //console.log('- krw-balance exists:', !!krwElement);
    //console.log('- portfolio-ratio exists:', !!portfolioElement);
    
    if (btcElement) //console.log('- btc-balance current text:', btcElement.textContent);
    if (krwElement) //console.log('- krw-balance current text:', krwElement.textContent);
    if (portfolioElement) //console.log('- portfolio-ratio current text:', portfolioElement.textContent);
    
    // 2. AssetDisplayManager 인스턴스 확인
    //console.log('AssetDisplayManager:');
    //console.log('- Instance exists:', !!window.assetDisplayManager);
    if (window.assetDisplayManager) {
        //console.log('- selectedCoin:', window.assetDisplayManager.selectedCoin);
        //console.log('- btcElement reference:', !!window.assetDisplayManager.btcElement);
        //console.log('- krwElement reference:', !!window.assetDisplayManager.krwElement);
        //console.log('- portfolioElement reference:', !!window.assetDisplayManager.portfolioElement);
    }
    
    // 3. Balance 데이터 확인
    const balanceCards = document.querySelectorAll('.balance-card');
    //console.log('Balance Cards:');
    //console.log('- Found balance cards:', balanceCards.length);
    balanceCards.forEach((card, index) => {
        const value = card.querySelector('.balance-value');
        //console.log(`- Card ${index}:`, card.className, 'Value:', value?.textContent);
    });
    
    // 4. Balance items 확인
    const balanceItems = document.querySelectorAll('.balance-item');
    //console.log('Balance Items:');
    //console.log('- Found balance items:', balanceItems.length);
    balanceItems.forEach((item, index) => {
        const currencyName = item.querySelector('.currency-name');
        const value = item.querySelector('.value');
        //console.log(`- Item ${index}:`, currencyName?.textContent, 'Value:', value?.textContent);
    });
    
    // 5. 함수 존재 확인
    //console.log('Functions:');
    //console.log('- renderAssetDisplay exists:', typeof renderAssetDisplay === 'function');
    //console.log('- setInterval active:', !!window.assetDisplayInterval);
    
    return {
        domElements: { btcElement: !!btcElement, krwElement: !!krwElement, portfolioElement: !!portfolioElement },
        manager: !!window.assetDisplayManager,
        balanceCards: balanceCards.length,
        balanceItems: balanceItems.length,
        renderFunction: typeof renderAssetDisplay === 'function'
    };
};

// 수동으로 renderAssetDisplay 실행하는 함수
window.forceRenderAssetDisplay = () => {
    //console.log('🚀 Force executing renderAssetDisplay...');
    try {
        renderAssetDisplay();
        //console.log('✅ renderAssetDisplay executed successfully');
    } catch (error) {
        console.error('❌ renderAssetDisplay failed:', error);
    }
};

// 테스트 데이터로 업데이트하는 함수
window.testWithSampleData = () => {
    //console.log('🧪 Testing with sample data...');
    if (window.assetDisplayManager) {
        window.assetDisplayManager.updateAssetDisplay(0.12345678, 5000000, 25.5);
        //console.log('✅ Sample data applied');
    } else {
        console.error('❌ assetDisplayManager not found');
    }
};

// 초기화 상태 확인
document.addEventListener('DOMContentLoaded', () => {
    //console.log('📅 DOMContentLoaded event fired');
    setTimeout(() => {
        //console.log('🔍 Post-load status check:');
        window.checkAssetDisplayStatus();
    }, 1000);
});

// 페이지 로드 완료 후 추가 확인
window.addEventListener('load', () => {
    //console.log('📄 Window load event fired');
    setTimeout(() => {
        //console.log('🔍 Post-window-load status check:');
        window.checkAssetDisplayStatus();
    }, 2000);
});
