// ===== Settings Module - JavaScript Frontend =====

class SettingsFrontend {
    constructor() {
        this.isInitialized = false;
        this.currentSettings = {};
        this.soundSettings = {};
    }
    
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('⚙️ Initializing Settings Frontend...');
        
        // 설정 로드
        await this.loadSettings();
        
        // 이벤트 리스너 등록
        this.registerEventListeners();
        
        // UI 업데이트
        this.updateSettingsUI();
        
        this.isInitialized = true;
        console.log('✅ Settings Frontend initialized');
    }
    
    registerEventListeners() {
        // 설정 저장 버튼
        const saveBtn = document.getElementById('settings-save-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                this.saveSettings();
            });
        }
        
        // 설정 리셋 버튼
        const resetBtn = document.getElementById('settings-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetSettings();
            });
        }
        
        // 설정 내보내기 버튼
        const exportBtn = document.getElementById('settings-export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportSettings();
            });
        }
        
        // 설정 가져오기 버튼
        const importBtn = document.getElementById('settings-import-btn');
        if (importBtn) {
            importBtn.addEventListener('click', () => {
                this.importSettings();
            });
        }
        
        // Upbit API 테스트 버튼
        const testApiBtn = document.getElementById('test-api-btn');
        if (testApiBtn) {
            testApiBtn.addEventListener('click', () => {
                this.testUpbitConnection();
            });
        }
        
        // 사운드 설정 토글
        const soundToggle = document.getElementById('sound-toggle');
        if (soundToggle) {
            soundToggle.addEventListener('change', (e) => {
                this.updateSoundSetting('enabled', e.target.checked);
            });
        }
        
        // 볼륨 슬라이더들
        const volumeSliders = document.querySelectorAll('.volume-slider');
        volumeSliders.forEach(slider => {
            slider.addEventListener('input', (e) => {
                const setting = e.target.dataset.setting;
                const value = parseFloat(e.target.value);
                this.updateSoundSetting(setting, value);
            });
        });
    }
    
    async loadSettings() {
        try {
            const response = await fetch('/api/settings/get');
            const result = await response.json();
            
            if (result.status === 'success') {
                this.currentSettings = result.data;
                this.soundSettings = this.currentSettings.sound || {};
                console.log('Settings loaded:', this.currentSettings);
            } else {
                console.error('Failed to load settings:', result.message);
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }
    
    updateSettingsUI() {
        // Upbit API 설정
        const apiKeyInput = document.getElementById('upbit-api-key');
        const secretKeyInput = document.getElementById('upbit-secret-key');
        
        if (apiKeyInput) {
            apiKeyInput.value = this.currentSettings.upbit?.api_key || '';
        }
        if (secretKeyInput) {
            secretKeyInput.value = this.currentSettings.upbit?.secret_key || '';
        }
        
        // 기본 KRW 코인 설정
        const defaultKrwCoinSelect = document.getElementById('default-krw-coin');
        if (defaultKrwCoinSelect) {
            defaultKrwCoinSelect.value = this.currentSettings.upbit?.defaultKrwCoin || 'BTC';
        }
        
        // Trading 설정
        const timeframeSelect = document.getElementById('default-timeframe');
        if (timeframeSelect) {
            timeframeSelect.value = this.currentSettings.trading?.default_timeframe || 'minute1';
        }
        
        const autoRotationCheckbox = document.getElementById('auto-rotation');
        if (autoRotationCheckbox) {
            autoRotationCheckbox.checked = this.currentSettings.trading?.auto_rotation !== false; // 기본값 true
        }
        
        // 분봉 카드 자동 순회 설정
        const timeframeAutoRotateCheckbox = document.getElementById('timeframe-auto-rotate');
        if (timeframeAutoRotateCheckbox) {
            timeframeAutoRotateCheckbox.checked = this.currentSettings.trading?.timeframe_auto_rotate !== false; // 기본값 true
        }
        
        // 사운드 설정
        const soundToggle = document.getElementById('sound-toggle');
        if (soundToggle) {
            soundToggle.checked = this.soundSettings.enabled !== false;
        }
        
        // 볼륨 슬라이더들
        const volumeSettings = [
            'master_volume',
            'click_volume',
            'success_volume',
            'error_volume',
            'type_volume',
            'sequence_volume'
        ];
        
        volumeSettings.forEach(setting => {
            const slider = document.querySelector(`[data-setting="${setting}"]`);
            if (slider) {
                slider.value = this.soundSettings[setting] || 0.5;
            }
        });
    }
    
    async saveSettings() {
        try {
            // UI에서 현재 설정값들을 수집
            const settings = this.collectSettingsFromUI();
            
            // 각 설정을 서버에 저장
            for (const [section, sectionData] of Object.entries(settings)) {
                for (const [key, value] of Object.entries(sectionData)) {
                    await this.updateSetting(section, key, value);
                }
            }
            
            this.showSuccess('Settings saved successfully! ✅');
            
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showError('Failed to save settings');
        }
    }
    
    collectSettingsFromUI() {
        const settings = {
            upbit: {},
            trading: {},
            sound: {}
        };
        
        // Upbit 설정
        const apiKeyInput = document.getElementById('upbit-access-key');
        const secretKeyInput = document.getElementById('upbit-secret-key');
        
        if (apiKeyInput) settings.upbit.upbitAccessKey = apiKeyInput.value;
        if (secretKeyInput) settings.upbit.upbitSecretKey = secretKeyInput.value;
        
        // 기본 KRW 코인 설정
        const defaultKrwCoinSelect = document.getElementById('default-krw-coin');
        if (defaultKrwCoinSelect) settings.upbit.defaultKrwCoin = defaultKrwCoinSelect.value;
        
        // Trading 설정
        const timeframeSelect = document.getElementById('default-timeframe');
        const autoRotationCheckbox = document.getElementById('auto-rotation');
        const timeframeAutoRotateCheckbox = document.getElementById('timeframe-auto-rotate');
        
        if (timeframeSelect) settings.trading.default_timeframe = timeframeSelect.value;
        if (autoRotationCheckbox) settings.trading.auto_rotation = autoRotationCheckbox.checked;
        if (timeframeAutoRotateCheckbox) settings.trading.timeframe_auto_rotate = timeframeAutoRotateCheckbox.checked;
        
        // 사운드 설정
        const soundToggle = document.getElementById('sound-toggle');
        if (soundToggle) settings.sound.enabled = soundToggle.checked;
        
        // 볼륨 설정
        const volumeSliders = document.querySelectorAll('.volume-slider');
        volumeSliders.forEach(slider => {
            const setting = slider.dataset.setting;
            settings.sound[setting] = parseFloat(slider.value);
        });
        
        return settings;
    }
    
    async updateSetting(section, key, value) {
        try {
            const response = await fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    section: section,
                    key: key,
                    value: value
                })
            });
            
            const result = await response.json();
            
            if (result.status !== 'success') {
                throw new Error(result.message);
            }
            
        } catch (error) {
            console.error(`Error updating setting ${section}.${key}:`, error);
            throw error;
        }
    }
    
    async resetSettings() {
        if (!confirm('Are you sure you want to reset all settings to default values?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/settings/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.currentSettings = result.data;
                this.updateSettingsUI();
                this.showSuccess('Settings reset to default values! ✅');
            } else {
                this.showError(result.message);
            }
            
        } catch (error) {
            console.error('Error resetting settings:', error);
            this.showError('Failed to reset settings');
        }
    }
    
    async exportSettings() {
        try {
            const response = await fetch('/api/settings/export');
            const result = await response.json();
            
            if (result.status === 'success') {
                // JSON 파일로 다운로드
                const dataStr = JSON.stringify(result.data, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                
                const link = document.createElement('a');
                link.href = URL.createObjectURL(dataBlob);
                link.download = `settings_${new Date().toISOString().split('T')[0]}.json`;
                link.click();
                
                this.showSuccess('Settings exported successfully! 📁');
            } else {
                this.showError(result.message);
            }
            
        } catch (error) {
            console.error('Error exporting settings:', error);
            this.showError('Failed to export settings');
        }
    }
    
    async importSettings() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            try {
                const text = await file.text();
                const settingsData = JSON.parse(text);
                
                const response = await fetch('/api/settings/import', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(settingsData)
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    this.currentSettings = result.data;
                    this.updateSettingsUI();
                    this.showSuccess('Settings imported successfully! 📂');
                } else {
                    this.showError(result.message);
                }
                
            } catch (error) {
                console.error('Error importing settings:', error);
                this.showError('Failed to import settings');
            }
        };
        
        input.click();
    }
    
    async testUpbitConnection() {
        const apiKeyInput = document.getElementById('upbit-api-key');
        const secretKeyInput = document.getElementById('upbit-secret-key');
        
        if (!apiKeyInput || !secretKeyInput) {
            this.showError('API key inputs not found');
            return;
        }
        
        const apiKey = apiKeyInput.value;
        const secretKey = secretKeyInput.value;
        
        if (!apiKey || !secretKey) {
            this.showError('Please enter both API key and secret key');
            return;
        }
        
        try {
            const response = await fetch('/api/settings/test-upbit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    secret_key: secretKey
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSuccess(`API connection successful! 💰 Balance: ₩${result.balance?.toLocaleString() || '0'}`);
            } else {
                this.showError(result.message);
            }
            
        } catch (error) {
            console.error('Error testing API connection:', error);
            this.showError('Failed to test API connection');
        }
    }
    
    async updateSoundSetting(setting, value) {
        try {
            const response = await fetch('/api/settings/sound', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    [setting]: value
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.soundSettings[setting] = value;
                console.log(`Sound setting updated: ${setting} = ${value}`);
            } else {
                console.error('Failed to update sound setting:', result.message);
            }
            
        } catch (error) {
            console.error('Error updating sound setting:', error);
        }
    }
    
    showSuccess(message) {
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 3000);
    }
    
    showError(message) {
        const notification = document.createElement('div');
        notification.className = 'notification error';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 5000);
    }
    
    destroy() {
        this.isInitialized = false;
    }
}

// 전역 인스턴스
window.settingsFrontend = new SettingsFrontend();
