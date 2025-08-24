// 카드별 저장소 시스템 모듈
// 좌측 패널의 각 분봉 카드별 데이터 저장소를 관리 (Python 백엔드 API 사용)

class CardStorageSystem {
    constructor() {
        this.apiBaseUrl = 'http://127.0.0.1:5057/api/card-storage';
        this.cardStorages = {};
        this.isInitialized = false;
        this.isOnline = true;
        
        // 생성자에서 자동으로 초기화
        this.initialize();
    }

    // API 호출 헬퍼 함수
    async apiCall(endpoint, options = {}) {
        try {
            const url = `${this.apiBaseUrl}${endpoint}`;
            const defaultOptions = {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                ...options
            };

            console.log(`🌐 API 호출: ${defaultOptions.method} ${url}`);
            
            const response = await fetch(url, defaultOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'API 호출 실패');
            }
            
            return data;
        } catch (error) {
            console.error(`❌ API 호출 실패 (${endpoint}):`, error);
            this.isOnline = false;
            throw error;
        }
    }

    // 시스템 초기화
    async initialize() {
        if (this.isInitialized) {
            console.log('⚠️ 카드 저장소 시스템이 이미 초기화되었습니다.');
            return;
        }

        console.log('🚀 카드 저장소 시스템 초기화 시작...');
        await this.loadCardStorages();
        this.isInitialized = true;
        console.log('✅ 카드 저장소 시스템 초기화 완료');
        
        // 초기화 후 디버깅 정보 출력
        this.debugServerStatus();
    }

    // 카드별 저장소 생성
    createCardStorage(timeframe) {
        if (!this.cardStorages[timeframe]) {
            this.cardStorages[timeframe] = {
                timeframe: timeframe,
                nbCoins: 0,
                nbMinerals: 0.0,  // N/B 미네랄 추가
                buyCount: 0,
                sellCount: 0,
                totalProfit: 0,
                lastBuyPrice: 0,
                lastSellPrice: 0,
                lastBuyTime: 0,
                lastSellTime: 0,
                buyHistory: [],
                sellHistory: [],
                createdAt: Date.now(),
                lastUpdated: Date.now()
            };
            this.saveCardStorages();
            console.log(`📦 카드 저장소 생성: ${timeframe}`);
        }
        return this.cardStorages[timeframe];
    }

    // 카드 저장소 가져오기
    getCardStorage(timeframe) {
        if (!this.cardStorages[timeframe]) {
            return this.createCardStorage(timeframe);
        }
        return this.cardStorages[timeframe];
    }

    // N/B 코인 추가
    async addNBCoin(timeframe, count = 1) {
        try {
            const storage = this.getCardStorage(timeframe);
            const currentCoins = storage.nbCoins;
            
            // 서버에 추가 요청
            const response = await this.apiCall(`/${timeframe}/nb-coin`, {
                method: 'POST',
                body: JSON.stringify({
                    action: 'add',
                    count: count
                })
            });
            
            // 로컬 데이터 업데이트
            storage.nbCoins = response.nbCoins;
            storage.lastUpdated = Date.now();
            
            if (window.logManager) {
                window.logManager.addLog(`🪙 카드 ${timeframe} N/B MAX 코인 +${count} 추가 → 총 ${storage.nbCoins}개`);
            }
            
            console.log(`✅ ${timeframe} N/B 코인 추가 완료: ${currentCoins} → ${storage.nbCoins}`);
            return storage.nbCoins;
            
        } catch (error) {
            console.error(`❌ ${timeframe} N/B 코인 추가 실패:`, error);
            // 실패 시 로컬에서만 업데이트
            const storage = this.getCardStorage(timeframe);
            storage.nbCoins += count;
            storage.nbCoins = Math.max(0, storage.nbCoins);
            storage.lastUpdated = Date.now();
            return storage.nbCoins;
        }
    }

    // N/B 코인 제거
    async removeNBCoin(timeframe, count = 1) {
        try {
            const storage = this.getCardStorage(timeframe);
            const currentCoins = storage.nbCoins;
            
            if (storage.nbCoins < count) {
                console.log(`❌ 카드 ${timeframe} N/B MAX 코인이 부족합니다. 현재: ${storage.nbCoins}개, 요청: ${count}개`);
                return storage.nbCoins;
            }
            
            // 서버에 제거 요청
            const response = await this.apiCall(`/${timeframe}/nb-coin`, {
                method: 'POST',
                body: JSON.stringify({
                    action: 'remove',
                    count: count
                })
            });
            
            // 로컬 데이터 업데이트
            storage.nbCoins = response.nbCoins;
            storage.lastUpdated = Date.now();
            
            if (window.logManager) {
                window.logManager.addLog(`🪙 카드 ${timeframe} N/B MAX 코인 -${count} 제거 → 총 ${storage.nbCoins}개`);
            }
            
            console.log(`✅ ${timeframe} N/B 코인 제거 완료: ${currentCoins} → ${storage.nbCoins}`);
            return storage.nbCoins;
            
        } catch (error) {
            console.error(`❌ ${timeframe} N/B 코인 제거 실패:`, error);
            // 실패 시 로컬에서만 업데이트
            const storage = this.getCardStorage(timeframe);
            if (storage.nbCoins >= count) {
                storage.nbCoins -= count;
                storage.nbCoins = Math.max(0, storage.nbCoins);
                storage.lastUpdated = Date.now();
            }
            return storage.nbCoins;
        }
    }

    // N/B 미네랄 추가
    addNBMineral(timeframe, amount = 1.0) {
        const storage = this.getCardStorage(timeframe);
        storage.nbMinerals += amount;
        storage.lastUpdated = Date.now();
        this.saveCardStorages();
        
        if (window.logManager) {
            window.logManager.addLog(`💎 카드 ${timeframe} N/B 미네랄 +${amount.toFixed(2)}% 추가 → 총 ${storage.nbMinerals.toFixed(2)}%`);
        }
        
        return storage.nbMinerals;
    }

    // N/B 미네랄 제거
    removeNBMineral(timeframe, amount = 1.0) {
        const storage = this.getCardStorage(timeframe);
        if (storage.nbMinerals >= amount) {
            storage.nbMinerals -= amount;
            storage.lastUpdated = Date.now();
            this.saveCardStorages();
            
            if (window.logManager) {
                window.logManager.addLog(`💎 카드 ${timeframe} N/B 미네랄 -${amount.toFixed(2)}% 제거 → 총 ${storage.nbMinerals.toFixed(2)}%`);
            }
            
            return storage.nbMinerals;
        } else {
            console.log(`❌ 카드 ${timeframe} N/B 미네랄이 부족합니다. 현재: ${storage.nbMinerals.toFixed(2)}%, 요청: ${amount.toFixed(2)}%`);
            return storage.nbMinerals;
        }
    }

    // 매수 기록 추가
    addBuyRecord(timeframe, price, profitRate = 0) {
        const storage = this.getCardStorage(timeframe);
        const buyRecord = {
            price: price,
            profitRate: profitRate,
            timestamp: Date.now(),
            nbCoins: storage.nbCoins + 1,
            nbMinerals: storage.nbMinerals
        };
        
        storage.buyCount++;
        storage.lastBuyPrice = price;
        storage.lastBuyTime = Date.now();
        storage.buyHistory.push(buyRecord);
        storage.lastUpdated = Date.now();
        
        // 최근 10개만 유지
        if (storage.buyHistory.length > 10) {
            storage.buyHistory = storage.buyHistory.slice(-10);
        }
        
        this.saveCardStorages();
        
        if (window.logManager) {
            window.logManager.addLog(`📈 카드 ${timeframe} 매수 기록 추가: ₩${price.toLocaleString()}, 수익률: ${profitRate.toFixed(2)}%`);
        }
    }

    // 매도 기록 추가
    addSellRecord(timeframe, price, profitRate = 0) {
        const storage = this.getCardStorage(timeframe);
        const sellRecord = {
            price: price,
            profitRate: profitRate,
            timestamp: Date.now(),
            nbCoins: storage.nbCoins - 1,
            nbMinerals: storage.nbMinerals
        };
        
        storage.sellCount++;
        storage.lastSellPrice = price;
        storage.lastSellTime = Date.now();
        storage.sellHistory.push(sellRecord);
        storage.totalProfit += profitRate;
        storage.lastUpdated = Date.now();
        
        // 최근 10개만 유지
        if (storage.sellHistory.length > 10) {
            storage.sellHistory = storage.sellHistory.slice(-10);
        }
        
        this.saveCardStorages();
        
        if (window.logManager) {
            window.logManager.addLog(`📉 카드 ${timeframe} 매도 기록 추가: ₩${price.toLocaleString()}, 수익률: ${profitRate.toFixed(2)}%`);
        }
    }

    // 모든 카드 저장소 가져오기
    getAllCardStorages() {
        return this.cardStorages;
    }

    // 카드 저장소 통계
    getCardStatistics(timeframe) {
        const storage = this.getCardStorage(timeframe);
        const avgBuyPrice = storage.buyHistory.length > 0 
            ? storage.buyHistory.reduce((sum, record) => sum + record.price, 0) / storage.buyHistory.length 
            : 0;
        const avgSellPrice = storage.sellHistory.length > 0 
            ? storage.sellHistory.reduce((sum, record) => sum + record.price, 0) / storage.sellHistory.length 
            : 0;
        
        return {
            timeframe: timeframe,
            nbCoins: storage.nbCoins,
            nbMinerals: storage.nbMinerals,
            buyCount: storage.buyCount,
            sellCount: storage.sellCount,
            totalProfit: storage.totalProfit,
            avgBuyPrice: avgBuyPrice,
            avgSellPrice: avgSellPrice,
            lastBuyPrice: storage.lastBuyPrice,
            lastSellPrice: storage.lastSellPrice,
            lastBuyTime: storage.lastBuyTime,
            lastSellTime: storage.lastSellTime,
            createdAt: storage.createdAt,
            lastUpdated: storage.lastUpdated
        };
    }

    // 모든 카드 통계
    getAllStatistics() {
        const statistics = {};
        Object.keys(this.cardStorages).forEach(timeframe => {
            statistics[timeframe] = this.getCardStatistics(timeframe);
        });
        return statistics;
    }

    // 카드 저장소 저장 (서버에 동기화)
    async saveCardStorages() {
        try {
            console.log('💾 카드 저장소 데이터 서버 동기화 시작...');
            
            // 각 타임프레임별로 서버에 업데이트
            const updatePromises = Object.keys(this.cardStorages).map(async (timeframe) => {
                const storage = this.cardStorages[timeframe];
                const updateData = {
                    nbCoins: storage.nbCoins || 0,
                    nbMinerals: storage.nbMinerals || 0.0,
                    buyCount: storage.buyCount || 0,
                    sellCount: storage.sellCount || 0,
                    totalProfit: storage.totalProfit || 0,
                    lastBuyPrice: storage.lastBuyPrice || 0,
                    lastSellPrice: storage.lastSellPrice || 0,
                    lastBuyTime: storage.lastBuyTime || 0,
                    lastSellTime: storage.lastSellTime || 0
                };
                
                try {
                    await this.apiCall(`/${timeframe}`, {
                        method: 'POST',
                        body: JSON.stringify(updateData)
                    });
                    console.log(`✅ ${timeframe} 타임프레임 동기화 완료`);
                } catch (error) {
                    console.error(`❌ ${timeframe} 타임프레임 동기화 실패:`, error);
                }
            });
            
            await Promise.all(updatePromises);
            console.log('✅ 카드 저장소 데이터 서버 동기화 완료');
            
        } catch (error) {
            console.error('❌ 카드 저장소 서버 동기화 실패:', error);
            console.error('❌ 에러 상세:', error.message);
            console.log('⚠️ 서버 접근 실패, 메모리에만 저장됨');
        }
    }

    // 카드 저장소 로드
    async loadCardStorages() {
        try {
            console.log('📂 카드 저장소 데이터 로드 시작...');
            
            const response = await this.apiCall('');
            const data = response.data;
            
            if (data && Object.keys(data).length > 0) {
                console.log('📂 서버에서 데이터 로드');
                console.log('📂 로드된 타임프레임:', Object.keys(data));
                
                // 서버 데이터를 클라이언트 형식으로 변환
                this.cardStorages = {};
                Object.keys(data).forEach(timeframe => {
                    const serverData = data[timeframe];
                    this.cardStorages[timeframe] = {
                        timeframe: timeframe,
                        nbCoins: serverData.nbCoins || 0,
                        nbMinerals: serverData.nbMinerals || 0.0,
                        buyCount: serverData.buyCount || 0,
                        sellCount: serverData.sellCount || 0,
                        totalProfit: serverData.totalProfit || 0,
                        lastBuyPrice: serverData.lastBuyPrice || 0,
                        lastSellPrice: serverData.lastSellPrice || 0,
                        lastBuyTime: serverData.lastBuyTime || 0,
                        lastSellTime: serverData.lastSellTime || 0,
                        buyHistory: [],
                        sellHistory: [],
                        createdAt: serverData.createdAt ? new Date(serverData.createdAt).getTime() : Date.now(),
                        lastUpdated: serverData.lastUpdated ? new Date(serverData.lastUpdated).getTime() : Date.now()
                    };
                });
                
                console.log('✅ 카드 저장소 데이터 로드 완료');
                
                // 각 타임프레임의 N/B 코인 상태 확인
                Object.keys(this.cardStorages).forEach(tf => {
                    const storage = this.cardStorages[tf];
                    console.log(`📂 ${tf}: N/B 코인 ${storage.nbCoins}개, N/B 미네랄 ${storage.nbMinerals}%`);
                });
            } else {
                this.cardStorages = {};
                console.log('📂 새로운 카드 저장소 생성 (서버에 데이터 없음)');
            }
        } catch (error) {
            console.error('❌ 카드 저장소 로드 실패:', error);
            console.error('❌ 에러 상세:', error.message);
            this.cardStorages = {};
            console.log('⚠️ 서버 접근 실패, 빈 저장소로 시작');
        }
    }

    // 카드 저장소 초기화
    resetCardStorage(timeframe) {
        if (this.cardStorages[timeframe]) {
            delete this.cardStorages[timeframe];
            this.saveCardStorages();
            console.log(`🔄 카드 ${timeframe} 저장소 초기화 완료`);
        }
    }

    // 모든 카드 저장소 초기화
    resetAllCardStorages() {
        this.cardStorages = {};
        this.saveCardStorages();
        console.log('🔄 모든 카드 저장소 초기화 완료');
    }

    // 카드 저장소 백업
    exportCardStorages() {
        return JSON.stringify(this.cardStorages, null, 2);
    }

    // 카드 저장소 복원
    importCardStorages(data) {
        try {
            this.cardStorages = JSON.parse(data);
            this.saveCardStorages();
            console.log('📥 카드 저장소 데이터 복원 완료');
            return true;
        } catch (error) {
            console.error('❌ 카드 저장소 데이터 복원 실패:', error);
            return false;
        }
    }

    // 서버 상태 디버깅 정보
    async debugServerStatus() {
        console.log('🔍 서버 상태 디버깅 시작...');
        
        try {
            // 서버 연결 테스트
            const response = await this.apiCall('');
            console.log('✅ 서버 연결 성공');
            console.log('📊 서버 데이터:', response.data);
            console.log('📊 타임프레임 수:', Object.keys(response.data).length);
            
            // 각 타임프레임별 상세 정보
            Object.keys(response.data).forEach(timeframe => {
                const data = response.data[timeframe];
                console.log(`📊 ${timeframe}: N/B 코인 ${data.nbCoins}개, N/B 미네랄 ${data.nbMinerals}%`);
            });
            
        } catch (error) {
            console.error('❌ 서버 연결 실패:', error);
            console.log('⚠️ 오프라인 모드로 동작 중');
        }
    }

    // 서버 연결 상태 확인
    async checkServerConnection() {
        try {
            await this.apiCall('');
            this.isOnline = true;
            return true;
        } catch (error) {
            this.isOnline = false;
            return false;
        }
    }

    // 클라이언트 데이터를 서버에 강제 동기화
    async forceSyncToServer() {
        try {
            console.log('🔄 클라이언트 데이터를 서버에 강제 동기화 시작...');
            
            // 현재 클라이언트의 모든 데이터를 서버에 업데이트
            const syncPromises = Object.keys(this.cardStorages).map(async (timeframe) => {
                const storage = this.cardStorages[timeframe];
                const updateData = {
                    nbCoins: storage.nbCoins || 0,
                    nbMinerals: storage.nbMinerals || 0.0,
                    buyCount: storage.buyCount || 0,
                    sellCount: storage.sellCount || 0,
                    totalProfit: storage.totalProfit || 0,
                    lastBuyPrice: storage.lastBuyPrice || 0,
                    lastSellPrice: storage.lastSellPrice || 0,
                    lastBuyTime: storage.lastBuyTime || 0,
                    lastSellTime: storage.lastSellTime || 0
                };
                
                try {
                    await this.apiCall(`/${timeframe}`, {
                        method: 'POST',
                        body: JSON.stringify(updateData)
                    });
                    console.log(`✅ ${timeframe} 강제 동기화 완료: N/B 코인 ${storage.nbCoins}개`);
                } catch (error) {
                    console.error(`❌ ${timeframe} 강제 동기화 실패:`, error);
                }
            });
            
            await Promise.all(syncPromises);
            console.log('✅ 강제 동기화 완료');
            
        } catch (error) {
            console.error('❌ 강제 동기화 실패:', error);
        }
    }
}

// 전역 인스턴스 생성
window.cardStorageSystem = new CardStorageSystem();

// 전역 함수들 (비동기 지원)
window.createCardStorage = (timeframe) => window.cardStorageSystem.createCardStorage(timeframe);
window.getCardStorage = (timeframe) => window.cardStorageSystem.getCardStorage(timeframe);
window.addNBCoinToCard = async (timeframe, count) => await window.cardStorageSystem.addNBCoin(timeframe, count);
window.removeNBCoinFromCard = async (timeframe, count) => await window.cardStorageSystem.removeNBCoin(timeframe, count);
window.addNBMineralToCard = (timeframe, amount) => window.cardStorageSystem.addNBMineral(timeframe, amount);
window.removeNBMineralFromCard = (timeframe, amount) => window.cardStorageSystem.removeNBMineral(timeframe, amount);
window.addBuyRecordToCard = (timeframe, price, profitRate) => window.cardStorageSystem.addBuyRecord(timeframe, price, profitRate);
window.addSellRecordToCard = (timeframe, price, profitRate) => window.cardStorageSystem.addSellRecord(timeframe, price, profitRate);
window.getCardStatistics = (timeframe) => window.cardStorageSystem.getCardStatistics(timeframe);
window.getAllCardStatistics = () => window.cardStorageSystem.getAllCardStatistics();
window.resetCardStorage = (timeframe) => window.cardStorageSystem.resetCardStorage(timeframe);
window.resetAllCardStorages = () => window.cardStorageSystem.resetAllCardStorages();
window.exportCardStorages = () => window.cardStorageSystem.exportCardStorages();
window.importCardStorages = (data) => window.cardStorageSystem.importCardStorages(data);
window.debugCardStorage = async () => await window.cardStorageSystem.debugServerStatus();
window.checkServerConnection = async () => await window.cardStorageSystem.checkServerConnection();
window.forceSyncToServer = async () => await window.cardStorageSystem.forceSyncToServer();
