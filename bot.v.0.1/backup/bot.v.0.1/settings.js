// ===== 8BIT Trading System - Settings Manager =====

class SettingsManager {
    constructor() {
        this.settings = {
            upbit: {
                accessKey: '',
                secretKey: '',
                useTestnet: false
            },
            chart: {
                defaultTimeframe: 'minute1',
                availableTimeframes: [
                    'minute1', 'minute3', 'minute5', 'minute10', 
                    'minute15', 'minute30', 'minute60', 'minute240', 'day'
                ],
                autoRefresh: true,
                refreshInterval: 30 // seconds
            },
            display: {
                theme: 'dark',
                language: 'ko'
            }
        };
        this.init();
    }

    init() {
        this.loadSettings();
        this.createSettingsUI();
        this.bindEvents();
    }

    // 설정 로드
    loadSettings() {
        try {
            const saved = localStorage.getItem('8bit_settings');
            if (saved) {
                const parsed = JSON.parse(saved);
                this.settings = { ...this.settings, ...parsed };
            }
        } catch (e) {
            console.error('설정 로드 실패:', e);
        }
    }

    // 설정 저장
    saveSettings() {
        try {
            localStorage.setItem('8bit_settings', JSON.stringify(this.settings));
            console.log('설정이 저장되었습니다.');
            return true;
        } catch (e) {
            console.error('설정 저장 실패:', e);
            return false;
        }
    }

    // 설정 가져오기
    getSetting(path) {
        const keys = path.split('.');
        let value = this.settings;
        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                return null;
            }
        }
        return value;
    }

    // 설정 업데이트
    updateSetting(path, value) {
        const keys = path.split('.');
        let current = this.settings;
        
        for (let i = 0; i < keys.length - 1; i++) {
            if (!(keys[i] in current)) {
                current[keys[i]] = {};
            }
            current = current[keys[i]];
        }
        
        current[keys[keys.length - 1]] = value;
        this.saveSettings();
    }

    // 설정 UI 생성
    createSettingsUI() {
        const settingsHTML = `
            <div class="settings-panel" id="settings-panel">
                <div class="settings-header">
                    <h3>⚙️ 시스템 설정</h3>
                    <button class="close-btn" onclick="settingsManager.toggleSettings()">×</button>
                </div>
                <div class="settings-content">
                    <!-- 업비트 API 설정 -->
                    <div class="setting-section">
                        <h4>🔑 업비트 API 설정</h4>
                        <div class="setting-group">
                            <label>Access Key:</label>
                            <input type="password" id="upbit-access-key" placeholder="업비트 Access Key 입력">
                        </div>
                        <div class="setting-group">
                            <label>Secret Key:</label>
                            <input type="password" id="upbit-secret-key" placeholder="업비트 Secret Key 입력">
                        </div>
                        <div class="setting-group">
                            <label>
                                <input type="checkbox" id="upbit-testnet">
                                테스트넷 사용 (개발용)
                            </label>
                        </div>
                        <button class="test-api-btn" onclick="settingsManager.testUpbitAPI()">API 연결 테스트</button>
                    </div>

                    <!-- 차트 설정 -->
                    <div class="setting-section">
                        <h4>📊 차트 설정</h4>
                        <div class="setting-group">
                            <label>기본 시간대:</label>
                            <select id="default-timeframe">
                                <option value="minute1">1분봉</option>
                                <option value="minute3">3분봉</option>
                                <option value="minute5">5분봉</option>
                                <option value="minute10">10분봉</option>
                                <option value="minute15">15분봉</option>
                                <option value="minute30">30분봉</option>
                                <option value="minute60">1시간봉</option>
                                <option value="minute240">4시간봉</option>
                                <option value="day">1일봉</option>
                            </select>
                        </div>
                        <div class="setting-group">
                            <label>
                                <input type="checkbox" id="auto-refresh">
                                자동 새로고침
                            </label>
                        </div>
                        <div class="setting-group">
                            <label>새로고침 간격 (초):</label>
                            <input type="number" id="refresh-interval" min="5" max="300" value="30">
                        </div>
                    </div>

                    <!-- 표시 설정 -->
                    <div class="setting-section">
                        <h4>🎨 표시 설정</h4>
                        <div class="setting-group">
                            <label>테마:</label>
                            <select id="theme">
                                <option value="dark">다크 테마</option>
                                <option value="light">라이트 테마</option>
                            </select>
                        </div>
                        <div class="setting-group">
                            <label>언어:</label>
                            <select id="language">
                                <option value="ko">한국어</option>
                                <option value="en">English</option>
                            </select>
                        </div>
                    </div>

                    <!-- 설정 버튼 -->
                    <div class="setting-buttons">
                        <button class="save-btn" onclick="settingsManager.saveAllSettings()">설정 저장</button>
                        <button class="reset-btn" onclick="settingsManager.resetSettings()">기본값으로 복원</button>
                        <button class="export-btn" onclick="settingsManager.exportSettings()">설정 내보내기</button>
                        <button class="import-btn" onclick="settingsManager.importSettings()">설정 가져오기</button>
                    </div>
                </div>
            </div>
        `;

        // 기존 설정 패널이 있으면 제거
        const existingPanel = document.getElementById('settings-panel');
        if (existingPanel) {
            existingPanel.remove();
        }

        // 새 설정 패널 추가
        document.body.insertAdjacentHTML('beforeend', settingsHTML);
        this.updateSettingsUI();
    }

    // 설정 UI 업데이트
    updateSettingsUI() {
        // 업비트 API 설정
        const accessKeyInput = document.getElementById('upbit-access-key');
        const secretKeyInput = document.getElementById('upbit-secret-key');
        const testnetCheckbox = document.getElementById('upbit-testnet');

        if (accessKeyInput) accessKeyInput.value = this.settings.upbit.accessKey;
        if (secretKeyInput) secretKeyInput.value = this.settings.upbit.secretKey;
        if (testnetCheckbox) testnetCheckbox.checked = this.settings.upbit.useTestnet;

        // 차트 설정
        const timeframeSelect = document.getElementById('default-timeframe');
        const autoRefreshCheckbox = document.getElementById('auto-refresh');
        const refreshIntervalInput = document.getElementById('refresh-interval');

        if (timeframeSelect) timeframeSelect.value = this.settings.chart.defaultTimeframe;
        if (autoRefreshCheckbox) autoRefreshCheckbox.checked = this.settings.chart.autoRefresh;
        if (refreshIntervalInput) refreshIntervalInput.value = this.settings.chart.refreshInterval;

        // 표시 설정
        const themeSelect = document.getElementById('theme');
        const languageSelect = document.getElementById('language');

        if (themeSelect) themeSelect.value = this.settings.display.theme;
        if (languageSelect) languageSelect.value = this.settings.display.language;
    }

    // 이벤트 바인딩
    bindEvents() {
        // 자동 저장을 위한 이벤트 리스너
        const inputs = document.querySelectorAll('#settings-panel input, #settings-panel select');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                this.saveAllSettings();
            });
        });
    }

    // 설정 토글
    toggleSettings() {
        const panel = document.getElementById('settings-panel');
        if (panel) {
            panel.classList.toggle('show');
            if (panel.classList.contains('show')) {
                this.updateSettingsUI();
            }
        }
    }

    // 모든 설정 저장
    saveAllSettings() {
        // 업비트 API 설정
        const accessKeyInput = document.getElementById('upbit-access-key');
        const secretKeyInput = document.getElementById('upbit-secret-key');
        const testnetCheckbox = document.getElementById('upbit-testnet');

        if (accessKeyInput) this.settings.upbit.accessKey = accessKeyInput.value;
        if (secretKeyInput) this.settings.upbit.secretKey = secretKeyInput.value;
        if (testnetCheckbox) this.settings.upbit.useTestnet = testnetCheckbox.checked;

        // 차트 설정
        const timeframeSelect = document.getElementById('default-timeframe');
        const autoRefreshCheckbox = document.getElementById('auto-refresh');
        const refreshIntervalInput = document.getElementById('refresh-interval');

        if (timeframeSelect) this.settings.chart.defaultTimeframe = timeframeSelect.value;
        if (autoRefreshCheckbox) this.settings.chart.autoRefresh = autoRefreshCheckbox.checked;
        if (refreshIntervalInput) this.settings.chart.refreshInterval = parseInt(refreshIntervalInput.value);

        // 표시 설정
        const themeSelect = document.getElementById('theme');
        const languageSelect = document.getElementById('language');

        if (themeSelect) this.settings.display.theme = themeSelect.value;
        if (languageSelect) this.settings.display.language = languageSelect.value;

        // 저장
        if (this.saveSettings()) {
            this.showMessage('설정이 저장되었습니다.', 'success');
            // 차트 새로고침
            if (window.refreshCharts) {
                window.refreshCharts();
            }
        } else {
            this.showMessage('설정 저장에 실패했습니다.', 'error');
        }
    }

    // 설정 리셋
    resetSettings() {
        this.showMessage('설정을 기본값으로 복원하시겠습니까? (Y/N)', 'info');
        
        // 키보드 이벤트 리스너 추가
        const handleKeyPress = (e) => {
            if (e.key.toLowerCase() === 'y') {
                localStorage.removeItem('8bit_settings');
                this.settings = {
                    upbit: {
                        accessKey: '',
                        secretKey: '',
                        useTestnet: false
                    },
                    chart: {
                        defaultTimeframe: 'minute1',
                        availableTimeframes: [
                            'minute1', 'minute3', 'minute5', 'minute10', 
                            'minute15', 'minute30', 'minute60', 'minute240', 'day'
                        ],
                        autoRefresh: true,
                        refreshInterval: 30
                    },
                    display: {
                        theme: 'dark',
                        language: 'ko'
                    }
                };
                this.updateSettingsUI();
                this.showMessage('설정이 기본값으로 복원되었습니다.', 'success');
                document.removeEventListener('keydown', handleKeyPress);
            } else if (e.key.toLowerCase() === 'n') {
                this.showMessage('설정 복원이 취소되었습니다.', 'info');
                document.removeEventListener('keydown', handleKeyPress);
            }
        };
        
        document.addEventListener('keydown', handleKeyPress);
        
        // 10초 후 자동으로 리스너 제거
        setTimeout(() => {
            document.removeEventListener('keydown', handleKeyPress);
        }, 10000);
    }

    // 업비트 API 테스트
    async testUpbitAPI() {
        const accessKey = document.getElementById('upbit-access-key')?.value;
        const secretKey = document.getElementById('upbit-secret-key')?.value;

        if (!accessKey || !secretKey) {
            this.showMessage('Access Key와 Secret Key를 모두 입력해주세요.', 'error');
            return;
        }

        try {
            this.showMessage('API 연결을 테스트하는 중...', 'info');
            
            const response = await fetch('/api/test-upbit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    accessKey: accessKey,
                    secretKey: secretKey
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.showMessage('API 연결 성공! 잔고: ' + result.balance + ' KRW', 'success');
            } else {
                this.showMessage('API 연결 실패: ' + result.error, 'error');
            }
        } catch (e) {
            this.showMessage('API 테스트 중 오류가 발생했습니다: ' + e.message, 'error');
        }
    }

    // 설정 내보내기
    exportSettings() {
        try {
            const dataStr = JSON.stringify(this.settings, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = '8bit_settings.json';
            link.click();
            
            URL.revokeObjectURL(url);
            this.showMessage('설정이 내보내기되었습니다.', 'success');
        } catch (e) {
            this.showMessage('설정 내보내기에 실패했습니다.', 'error');
        }
    }

    // 설정 가져오기
    importSettings() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const imported = JSON.parse(e.target.result);
                        this.settings = { ...this.settings, ...imported };
                        this.updateSettingsUI();
                        this.saveSettings();
                        this.showMessage('설정이 가져와졌습니다.', 'success');
                    } catch (e) {
                        this.showMessage('설정 파일 형식이 올바르지 않습니다.', 'error');
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }

    // 메시지 표시
    showMessage(message, type = 'info') {
        if (window.updateStatusMessage) {
            window.updateStatusMessage(message);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// 전역 설정 매니저 인스턴스
let settingsManager = null;

// 전역 함수들
window.toggleSettings = function() { 
    if (settingsManager) settingsManager.toggleSettings(); 
};

window.testUpbitAPI = function() { 
    if (settingsManager) settingsManager.testUpbitAPI(); 
};

window.saveAllSettings = function() { 
    if (settingsManager) settingsManager.saveAllSettings(); 
};

window.resetSettings = function() { 
    if (settingsManager) settingsManager.resetSettings(); 
};

window.exportSettings = function() { 
    if (settingsManager) settingsManager.exportSettings(); 
};

window.importSettings = function() { 
    if (settingsManager) settingsManager.importSettings(); 
};

// DOM 로드 시 설정 매니저 초기화
document.addEventListener('DOMContentLoaded', function() {
    console.log('⚙️ Settings Manager Initializing...');
    settingsManager = new SettingsManager();
    window.settingsManager = settingsManager;
    console.log('⚙️ Settings Manager Ready');
});
