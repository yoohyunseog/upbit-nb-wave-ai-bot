// ===== Zone Strength Manager =====
// 구역 강도 관리 모듈

class ZoneStrengthManager {
    constructor() {
        this.isInitialized = false;
        this.currentZoneStrength = 0;
        this.lastZone = 'UNKNOWN';
        this.majorityZone = null;
        this.majorityZoneStrength = 0;
        
        //console.log('🔄 Zone Strength Manager initialized');
    }
    
    // 초기화
    initialize() {
        if (this.isInitialized) return;
        
        //console.log('🔄 Initializing Zone Strength Manager...');
        
        // 전역 변수 초기화
        window.currentZoneStrength = 0;
        window.lastZoneFromChart = 'UNKNOWN';
        window.currentMajorityZone = null;
        window.currentMajorityZoneStrength = 0;
        
        // 기본 구역 표시 설정
        this.setDefaultZoneDisplay();
        
        this.isInitialized = true;
        //console.log('✅ Zone Strength Manager initialized');
    }
    
    // 기본 구역 표시 설정
    setDefaultZoneDisplay() {
        const currentZoneElement = document.getElementById('trading-current-zone');
        const zoneStrengthElement = document.getElementById('trading-zone-strength');
        const rightCurrentZoneElement = document.getElementById('right-trading-current-zone');
        const rightZoneStrengthElement = document.getElementById('right-trading-zone-strength');
        
        if (currentZoneElement) {
            currentZoneElement.textContent = 'Neutral Zone';
            currentZoneElement.className = 'current-zone zone-neutral';
        }
        
        if (zoneStrengthElement) {
            zoneStrengthElement.textContent = '강도: 0';
        }
        
        // 우측 패널도 업데이트
        if (rightCurrentZoneElement) {
            rightCurrentZoneElement.textContent = 'Neutral Zone';
            rightCurrentZoneElement.className = 'current-zone zone-neutral';
        }
        
        if (rightZoneStrengthElement) {
            rightZoneStrengthElement.textContent = '강도: 0';
        }
        
        //console.log('✅ Default zone display set');
    }
    
    // 차트에서 구역 강도 계산
    calculateZoneStrengthFromChart(nbData) {
        //console.log('🔍 Calculating zone strength from chart data:', nbData);
        
        let currentZoneStrength = 0;
        let lastZone = 'UNKNOWN';
        
        if (nbData && nbData.zones && nbData.zones.length > 0) {
            // 마지막 zone 찾기
            const lastZoneObject = nbData.zones[nbData.zones.length - 1];
            
            if (lastZoneObject && lastZoneObject.zone) {
                lastZone = lastZoneObject.zone;
                
                // N/B WAVE MAP 값으로 강도 계산
                if (lastZone === 'BLUE') {
                    // N/B WAVE MAP의 BLUE - ORANGE 값 사용
                    const nbBlue = nbData && nbData.summary ? (nbData.summary.blue || 0) : 0;
                    const nbOrange = nbData && nbData.summary ? (nbData.summary.orange || 0) : 0;
                    currentZoneStrength = nbBlue - nbOrange;
                    //console.log('🔵 Blue Zone - N/B WAVE MAP 강도 계산:', { nbBlue, nbOrange, currentZoneStrength });
                } else if (lastZone === 'ORANGE') {
                    // N/B WAVE MAP의 ORANGE - BLUE 값 사용
                    const nbBlue = nbData && nbData.summary ? (nbData.summary.blue || 0) : 0;
                    const nbOrange = nbData && nbData.summary ? (nbData.summary.orange || 0) : 0;
                    currentZoneStrength = nbOrange - nbBlue;
                    //console.log('🟠 Orange Zone - N/B WAVE MAP 강도 계산:', { nbBlue, nbOrange, currentZoneStrength });
                }
            }
        }
        
        // 전역 변수에 저장
        this.currentZoneStrength = currentZoneStrength;
        this.lastZone = lastZone;
        window.currentZoneStrength = currentZoneStrength;
        window.lastZoneFromChart = lastZone;
        
        /*console.log('💪 Zone strength calculated:', {
            lastZone: lastZone,
            currentZoneStrength: currentZoneStrength
        });*/
        
        return {
            lastZone: lastZone,
            currentZoneStrength: currentZoneStrength
        };
    }
    
    // 차트에서 현재 구역 업데이트
    updateTradingCurrentZoneFromChart(nbData) {
        //console.log('🔍 updateTradingCurrentZoneFromChart called with nbData:', nbData);
        
        const currentZoneElement = document.getElementById('trading-current-zone');
        const zoneStrengthElement = document.getElementById('trading-zone-strength');
        const rightCurrentZoneElement = document.getElementById('right-trading-current-zone');
        const rightZoneStrengthElement = document.getElementById('right-trading-zone-strength');
        
        if (currentZoneElement && zoneStrengthElement) {
            // 차트의 마지막 구역 정보를 우선적으로 사용
            let currentZoneType = 'Neutral Zone';
            let chartZoneStrength = 0;
            
            if (nbData && nbData.zones && nbData.zones.length > 0) {
                // 차트의 마지막 구역 가져오기
                const lastZone = nbData.zones[nbData.zones.length - 1];
                //console.log('📊 Last zone from chart data:', lastZone);
                //console.log('🔍 Last zone strength value:', lastZone.strength);
                //console.log('🔍 Last zone change value:', lastZone.change);
                
                if (lastZone && lastZone.zone) {
                    // 실제 차트에서 계산된 strength 값을 사용
                    const chartZoneStrengths = window.chartZoneStrengths || { BLUE: 0, ORANGE: 0, NEUTRAL: 0 };
                    const chartCurrentZone = window.chartCurrentZone || 'NEUTRAL';
                    
                    //console.log('🔍 Zone strength calculation details:');
                    //console.log('  - Zone type:', lastZone.zone);
                    //console.log('  - Chart current zone:', chartCurrentZone);
                    //console.log('  - Raw strength value:', lastZone.strength);
                    //console.log('  - Chart zone strengths:', chartZoneStrengths);
                    //console.log('  - All available fields:', Object.keys(lastZone));
                    
                    if (lastZone.zone === 'BLUE') {
                        currentZoneType = 'Blue Zone';
                        // 실제 차트의 BLUE zone strength 값 사용 (이미 퍼센트로 변환됨)
                        chartZoneStrength = chartZoneStrengths.BLUE;
                        //console.log('🔵 Chart last zone is BLUE, using chart BLUE strength:', chartZoneStrength);
                        //console.log('🔵 Chart BLUE strength value:', chartZoneStrengths.BLUE);
                    } else if (lastZone.zone === 'ORANGE') {
                        currentZoneType = 'Orange Zone';
                        // 실제 차트의 ORANGE zone strength 값 사용 (이미 퍼센트로 변환됨)
                        chartZoneStrength = chartZoneStrengths.ORANGE;
                        //console.log('🟠 Chart last zone is ORANGE, using chart ORANGE strength:', chartZoneStrength);
                        //console.log('🟠 Chart ORANGE strength value:', chartZoneStrengths.ORANGE);
                    } else {
                        currentZoneType = 'Neutral Zone';
                        // 실제 차트의 NEUTRAL zone strength 값 사용 (이미 퍼센트로 변환됨)
                        chartZoneStrength = chartZoneStrengths.NEUTRAL;
                        //console.log('⚪ Chart last zone is NEUTRAL, using chart NEUTRAL strength:', chartZoneStrength);
                        //console.log('⚪ Chart NEUTRAL strength value:', chartZoneStrengths.NEUTRAL);
                    }
                }
            } else {
                //console.log('⚠️ No zones data available in nbData');
            }
            
            //console.log('🎯 Final zone type:', currentZoneType, 'Final chart strength:', chartZoneStrength);
            
            // 구역 표시 업데이트
            this.updateZoneDisplay(currentZoneElement, zoneStrengthElement, currentZoneType, chartZoneStrength);
            
            // 우측 패널 업데이트
            if (rightCurrentZoneElement && rightZoneStrengthElement) {
                this.updateZoneDisplay(rightCurrentZoneElement, rightZoneStrengthElement, currentZoneType, chartZoneStrength);
            }
            
            //console.log('🎯 Trading current zone updated from chart:', currentZoneType, 'Chart Strength:', chartZoneStrength);
            
            // 현재 구역 데이터를 전역으로 저장
            window.sharedTradingDashboardData = {
                ...window.sharedTradingDashboardData,
                currentZone: currentZoneType,
                zoneStrength: chartZoneStrength,
                last_update: new Date().toISOString(),
                timestamp: new Date().getTime()
            };
            
            //console.log('💾 Trading Dashboard zone data updated in global storage:', window.sharedTradingDashboardData);
        } else {
            //console.log('❌ Zone elements not found');
        }
    }
    
    // 현재 구역 업데이트 (기존 함수 - 호환성 유지)
    updateTradingCurrentZone(nbData) {
        //console.log('🔍 updateTradingCurrentZone called with nbData:', nbData);
        
        const currentZoneElement = document.getElementById('trading-current-zone');
        const zoneStrengthElement = document.getElementById('trading-zone-strength');
        const rightCurrentZoneElement = document.getElementById('right-trading-current-zone');
        const rightZoneStrengthElement = document.getElementById('right-trading-zone-strength');
        
        if (currentZoneElement && zoneStrengthElement) {
            // 전역 변수에서 마지막 구역 정보 가져오기 (우선순위)
            let currentZoneType = 'Neutral';
            let chartZoneStrength = 0; // 기본 강도
            
            // 과반수 구역 우선 사용
            if (window.currentMajorityZone) {
                if (window.currentMajorityZone === 'BLUE') {
                    currentZoneType = 'Blue Zone';
                    chartZoneStrength = window.currentMajorityZoneStrength || 0;
                    //console.log('🔵 Using BLUE majority zone, strength:', chartZoneStrength);
                } else if (window.currentMajorityZone === 'ORANGE') {
                    currentZoneType = 'Orange Zone';
                    chartZoneStrength = window.currentMajorityZoneStrength || 0;
                    //console.log('🟠 Using ORANGE majority zone, strength:', chartZoneStrength);
                } else {
                    currentZoneType = 'Neutral Zone';
                    chartZoneStrength = window.currentMajorityZoneStrength || 0;
                    //console.log('⚪ Using NEUTRAL majority zone, strength:', chartZoneStrength);
                }
            } else if (window.lastZoneFromChart) {
                // 차트에서 저장된 마지막 구역 정보 사용 (fallback)
                if (window.lastZoneFromChart === 'BLUE') {
                    currentZoneType = 'Blue Zone';
                    // 현재 구역 강도 변수 사용
                    chartZoneStrength = window.currentZoneStrength || 0;
                    //console.log('🔵 Using BLUE zone from chart, strength:', chartZoneStrength, 'currentZoneStrength:', window.currentZoneStrength);
                } else if (window.lastZoneFromChart === 'ORANGE') {
                    currentZoneType = 'Orange Zone';
                    // 현재 구역 강도 변수 사용
                    chartZoneStrength = window.currentZoneStrength || 0;
                    //console.log('🟠 Using ORANGE zone from chart, strength:', chartZoneStrength, 'currentZoneStrength:', window.currentZoneStrength);
                }
            } else if (nbData && nbData.zones && nbData.zones.length > 0) {
                // 전역 변수가 없으면 NB Wave 데이터에서 가져오기 (fallback)
                const lastZone = nbData.zones[nbData.zones.length - 1];
                //console.log('📊 Fallback to NB Wave data:', lastZone);
                
                if (lastZone.zone) {
                    if (lastZone.zone === 'BLUE') {
                        currentZoneType = 'Blue Zone';
                        // Strength 값을 직접 참조
                        chartZoneStrength = lastZone.strength ? Math.round(lastZone.strength * 100) : 0;
                        //console.log('🔵 Last point is BLUE zone (fallback), strength from lastZone.strength:', chartZoneStrength);
                    } else if (lastZone.zone === 'ORANGE') {
                        currentZoneType = 'Orange Zone';
                        // Strength 값을 직접 참조
                        chartZoneStrength = lastZone.strength ? Math.round(lastZone.strength * 100) : 0;
                        //console.log('🟠 Last point is ORANGE zone (fallback), strength from lastZone.strength:', chartZoneStrength);
                    } else {
                        // Strength 값을 직접 참조
                        chartZoneStrength = lastZone.strength ? Math.round(lastZone.strength * 100) : 0;
                        //console.log('⚪ Last point is NEUTRAL zone (fallback), strength from lastZone.strength:', chartZoneStrength);
                    }
                }
            } else {
                //console.log('❌ No zone information available');
                this.setDefaultZoneDisplay();
                return;
            }
            
            //console.log('🎯 Current zone type:', currentZoneType, 'from chart:', window.lastZoneFromChart, 'strength:', chartZoneStrength);
            
            // 구역 표시 업데이트
            this.updateZoneDisplay(currentZoneElement, zoneStrengthElement, currentZoneType, chartZoneStrength);
            
            // 우측 패널 업데이트
            if (rightCurrentZoneElement && rightZoneStrengthElement) {
                this.updateZoneDisplay(rightCurrentZoneElement, rightZoneStrengthElement, currentZoneType, chartZoneStrength);
            }
            
            //console.log('🎯 Trading current zone updated:', currentZoneType, 'Chart Strength:', chartZoneStrength);
        } else {
            //console.log('❌ Zone elements not found');
        }
    }
    
    // 구역 표시 업데이트 (공통 함수)
    updateZoneDisplay(currentZoneElement, zoneStrengthElement, currentZoneType, zoneStrength) {
        if (currentZoneElement) {
            currentZoneElement.textContent = currentZoneType;
            
            // 기존 zone 클래스들 제거
            currentZoneElement.classList.remove('zone-blue', 'zone-orange', 'zone-neutral');
            
            // 구역별 색상 적용
            if (currentZoneType === 'Blue Zone') {
                currentZoneElement.classList.add('zone-blue');
                //console.log('🔵 Applied zone-blue class to:', currentZoneElement.id || currentZoneElement.className);
            } else if (currentZoneType === 'Orange Zone') {
                currentZoneElement.classList.add('zone-orange');
                //console.log('🟠 Applied zone-orange class to:', currentZoneElement.id || currentZoneElement.className);
            } else {
                currentZoneElement.classList.add('zone-neutral');
                //console.log('⚪ Applied zone-neutral class to:', currentZoneElement.id || currentZoneElement.className);
            }
        }
        
        if (zoneStrengthElement) {
            zoneStrengthElement.textContent = `강도: ${zoneStrength}`;
        }
    }
    
    // 과반수 구역 설정
    setMajorityZone(zone, strength) {
        this.majorityZone = zone;
        this.majorityZoneStrength = strength;
        window.currentMajorityZone = zone;
        window.currentMajorityZoneStrength = strength;
        
        //console.log('🏆 Majority zone set:', { zone, strength });
    }
    
    // 현재 구역 정보 가져오기
    getCurrentZoneInfo() {
        return {
            currentZoneStrength: this.currentZoneStrength,
            lastZone: this.lastZone,
            majorityZone: this.majorityZone,
            majorityZoneStrength: this.majorityZoneStrength
        };
    }
    
    // 강도 값 설정
    setZoneStrength(strength) {
        this.currentZoneStrength = strength;
        window.currentZoneStrength = strength;
        //console.log('💪 Zone strength set to:', strength);
    }
    
    // 마지막 구역 설정
    setLastZone(zone) {
        this.lastZone = zone;
        window.lastZoneFromChart = zone;
        //console.log('📊 Last zone set to:', zone);
    }
    
    // 정리
    destroy() {
        this.isInitialized = false;
        //console.log('🗑️ Zone Strength Manager destroyed');
    }
}

// 전역 인스턴스 생성
window.zoneStrengthManager = new ZoneStrengthManager();

// 전역 함수들 (기존 코드와의 호환성)
window.updateTradingCurrentZoneFromChart = (nbData) => {
    window.zoneStrengthManager.updateTradingCurrentZoneFromChart(nbData);
};

window.updateTradingCurrentZone = (nbData) => {
    window.zoneStrengthManager.updateTradingCurrentZone(nbData);
};

window.setDefaultZoneDisplay = () => {
    window.zoneStrengthManager.setDefaultZoneDisplay();
};

window.calculateZoneStrengthFromChart = (nbData) => {
    return window.zoneStrengthManager.calculateZoneStrengthFromChart(nbData);
};

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    if (window.zoneStrengthManager) {
        window.zoneStrengthManager.initialize();
    }
});

//console.log('✅ Zone Strength Manager Module loaded');
