// ===== 8BIT Trading System - Audio System (Starcraft Sounds) =====

class AudioSystem {
    constructor() {
        this.soundEnabled = true;
        this.initialized = false;
        this.isPlaying = false;
        this.userInteracted = false;
        this.init();
    }

    // 소리 재생 상태 업데이트
    updateSoundStatus(isPlaying) {
        this.isPlaying = isPlaying;
        const statusElement = document.getElementById('sound-status');
        const statusIcon = document.getElementById('sound-status-icon');
        
        if (statusElement && statusIcon) {
            if (isPlaying) {
                statusElement.className = 'sound-status playing';
                statusIcon.className = 'fas fa-play';
            } else {
                statusElement.className = 'sound-status stopped';
                statusIcon.className = 'fas fa-pause';
            }
        }
    }

    // 오디오 시스템 초기화
    init() {
        console.log('🎵 Audio System Initializing...');
        this.setupUserInteraction();
        this.initializeSettings(); // 사운드 설정 초기화
        this.initialized = true;
        console.log('🎵 Audio System Ready');
        
        // 초기 상태 설정
        this.updateSoundStatus(false);
        
        // Startup 사운드 비활성화 (설정에 따라 재생)
        // this.playStartupSound();
    }

    // Startup 사운드 1회 재생
    playStartupSound() {
        // 3초 후 Startup 사운드 재생
        setTimeout(() => {
            if (this.userInteracted && this.soundEnabled) {
                console.log('🎵 Playing startup sound (one time only)...');
                this.playWithReducedVolume('type', 0.2); // Startup 사운드 재생
            }
        }, 3000);
    }

    // 연속 재생 기능 (스타크래프트 사운드)
    playSoundSequence() {
        console.log('🎵 Playing Starcraft sound sequence...');
        
        const sounds = ['success', 'click', 'success']; // Startup 사운드 제거
        let currentIndex = 0;
        
        // 전체 재생 시작 시 상태 업데이트
        this.updateSoundStatus(true);
        
        const playNextSound = () => {
            if (currentIndex < sounds.length) {
                const soundType = sounds[currentIndex];
                console.log(`🎵 Playing sequence sound ${currentIndex + 1}/${sounds.length}: ${soundType}`);
                
                // 연속 재생 시 볼륨을 조정 (스타크래프트 사운드에 맞게)
                this.playWithReducedVolume(soundType, 0.15);
                
                currentIndex++;
                setTimeout(playNextSound, 1200); // 스타크래프트 사운드에 맞게 간격 조정
            } else {
                console.log('🎵 Starcraft sound sequence completed!');
                // 전체 재생 완료 후 상태 해제
                setTimeout(() => {
                    this.updateSoundStatus(false);
                }, 1000);
            }
        };
        
        // 즉시 시작
        playNextSound();
    }

    // 볼륨이 줄어든 사운드 재생
    playWithReducedVolume(soundType, volume = 0.1) {
        if (!this.soundEnabled) {
            console.log('🎵 Sound disabled, skipping play');
            return;
        }
        
        if (!this.initialized) {
            console.log('🎵 Audio system not initialized yet');
            return;
        }
        
        // 사용자 상호작용이 없었다면 재생하지 않음
        if (!this.userInteracted) {
            console.log('🎵 No user interaction yet, skipping sound play');
            return;
        }
        
        try {
            console.log(`🎵 Attempting to play ${soundType} sound with reduced volume (${volume})...`);
            
            // 재생 상태 업데이트 (하단 표시)
            this.updateSoundStatus(true);
            
            // 파이썬 서버 API 호출 (볼륨 정보 포함)
            console.log('🎵 Calling Python server sound API with reduced volume...');
            this.callPythonServerAPIWithVolume(soundType, volume);
            
            // 300ms 후 재생 상태 해제
            setTimeout(() => {
                this.updateSoundStatus(false);
            }, 300);
            
        } catch (e) {
            console.error('🎵 Audio play error:', e);
            this.updateSoundStatus(false);
        }
    }

    // 사용자 상호작용 감지
    setupUserInteraction() {
        const activateAudio = () => {
            if (!this.userInteracted) {
                this.userInteracted = true;
                console.log('🎵 User interaction detected - audio system activated');
                
                // 이벤트 리스너 제거 (한 번만 실행)
                document.removeEventListener('click', activateAudio);
                document.removeEventListener('touchstart', activateAudio);
                document.removeEventListener('keydown', activateAudio);
            }
        };
        
        // 다양한 사용자 상호작용 이벤트 리스너 추가
        document.addEventListener('click', activateAudio);
        document.addEventListener('touchstart', activateAudio);
        document.addEventListener('keydown', activateAudio);
    }

    // 사운드 설정 토글
    toggleSettings() {
        const panel = document.getElementById('sound-settings-panel');
        if (panel) {
            panel.classList.toggle('show');
        }
    }

    // 마스터 볼륨 업데이트
    updateMasterVolume() {
        const slider = document.getElementById('master-volume');
        const value = document.getElementById('master-volume-value');
        if (slider && value) {
            const volume = slider.value / 100;
            value.textContent = slider.value + '%';
            this.masterVolume = volume;
            console.log(`🎵 Master volume set to: ${volume}`);
        }
    }

    // 개별 사운드 볼륨 업데이트
    updateSoundVolume(soundType) {
        const slider = document.getElementById(`${soundType}-volume`);
        const value = document.getElementById(`${soundType}-volume-value`);
        if (slider && value) {
            const volume = slider.value / 100;
            value.textContent = slider.value + '%';
            this.soundVolumes[soundType] = volume;
            console.log(`🎵 ${soundType} volume set to: ${volume}`);
        }
    }

    // 사운드 효과 토글
    toggleSoundEffect(soundType) {
        const checkbox = document.getElementById(`enable-${soundType}`);
        if (checkbox) {
            this.soundEnabled[soundType] = checkbox.checked;
            console.log(`🎵 ${soundType} sound ${checkbox.checked ? 'enabled' : 'disabled'}`);
        }
    }

    // 설정 초기화 (스타크래프트 사운드에 맞게 조정)
    initializeSettings() {
        // localStorage에서 설정 불러오기
        const savedSettings = localStorage.getItem('8bit-settings');
        if (savedSettings) {
            try {
                const settings = JSON.parse(savedSettings);
                
                // 마스터 볼륨 설정
                this.masterVolume = settings.masterVolume ? settings.masterVolume / 100 : 0.5;
                
                // 사운드 활성화 설정
                this.soundEnabled = settings.soundEffects !== false; // 기본값 true
                
                console.log(`🎵 Loaded settings - Volume: ${this.masterVolume * 100}%, Sound: ${this.soundEnabled ? 'ON' : 'OFF'}`);
            } catch (e) {
                console.error('🎵 Failed to load settings:', e);
                this.setDefaultSettings();
            }
        } else {
            this.setDefaultSettings();
        }

        // UI 업데이트
        this.updateSettingsUI();
    }
    
    // 기본 설정
    setDefaultSettings() {
        this.masterVolume = 0.5; // 마스터 볼륨을 50%로 설정
        this.soundEnabled = true; // 사운드 활성화
        console.log('🎵 Using default settings');
    }

    // 설정 UI 업데이트
    updateSettingsUI() {
        // 마스터 볼륨
        const masterSlider = document.getElementById('master-volume');
        const masterValue = document.getElementById('master-volume-value');
        if (masterSlider && masterValue) {
            masterSlider.value = this.masterVolume * 100;
            masterValue.textContent = Math.round(this.masterVolume * 100) + '%';
        }

        // 사운드 효과 체크박스
        const soundEffectsCheckbox = document.getElementById('sound-effects');
        if (soundEffectsCheckbox) {
            soundEffectsCheckbox.checked = this.soundEnabled;
        }
    }

    // 설정 리셋
    resetSettings() {
        this.initializeSettings();
        this.updateSettingsUI();
        console.log('🎵 Sound settings reset to default');
    }

    // 볼륨 계산 (마스터 볼륨 적용)
    calculateVolume(soundType) {
        const masterVol = this.masterVolume || 0.5;
        return masterVol; // 마스터 볼륨만 사용
    }

    // 사운드 재생 (개선된 버전 - 설정 적용)
    play(soundType) {
        if (!this.soundEnabled) {
            console.log('🎵 Sound disabled, skipping play');
            return;
        }
        
        if (!this.initialized) {
            console.log('🎵 Audio system not initialized yet');
            return;
        }
        
        // 사용자 상호작용이 없었다면 재생하지 않음
        if (!this.userInteracted) {
            console.log('🎵 No user interaction yet, skipping sound play');
            return;
        }
        
        try {
            console.log(`🎵 Attempting to play ${soundType} sound...`);
            
            // 재생 상태 업데이트 (하단 표시)
            this.updateSoundStatus(true);
            
            // 계산된 볼륨으로 파이썬 서버 API 호출
            const volume = this.calculateVolume(soundType);
            console.log(`🎵 Calling Python server sound API with volume: ${volume}`);
            this.callPythonServerAPIWithVolume(soundType, volume);
            
            // 300ms 후 재생 상태 해제
            setTimeout(() => {
                this.updateSoundStatus(false);
            }, 300);
            
        } catch (e) {
            console.error('🎵 Audio play error:', e);
            this.updateSoundStatus(false);
        }
    }

    // 볼륨 정보를 포함한 파이썬 서버 API 호출
    callPythonServerAPIWithVolume(soundType, volume = 0.1) {
        try {
            console.log(`🎵 Sending request to Python server for ${soundType} sound with volume ${volume}...`);
            
            fetch('/api/play-sound', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: soundType,
                    volume: volume
                })
            })
            .then(response => {
                console.log(`🎵 Python server response status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                console.log(`🎵 Python server response:`, data);
                if (data.success) {
                    console.log(`🎵 Python server sound success: ${data.message}`);
                } else {
                    console.error(`🎵 Python server sound failed: ${data.error}`);
                }
            })
            .catch(error => {
                console.error(`🎵 Python server API request failed:`, error);
            });
            
        } catch (e) {
            console.error(`🎵 Python server API error:`, e);
        }
    }

    // 사운드 토글
    toggle() {
        this.soundEnabled = !this.soundEnabled;
        const soundIcon = document.getElementById('sound-icon');
        
        if (soundIcon) {
            if (this.soundEnabled) {
                soundIcon.className = 'fas fa-volume-up';
                this.showMessage('Sound enabled');
            } else {
                soundIcon.className = 'fas fa-volume-mute';
                this.showMessage('Sound disabled');
            }
        }
        
        this.play('click');
    }

    // 모든 사운드 테스트
    testAll() {
        if (!this.soundEnabled) {
            this.showMessage('Sound is disabled. Enable sound first.');
            return;
        }
        
        this.showMessage('Testing all Starcraft sounds...');
        
        const sounds = ['click', 'success', 'error']; // Startup 사운드 제거
        let currentIndex = 0;
        
        const playNextSound = () => {
            if (currentIndex < sounds.length) {
                const soundType = sounds[currentIndex];
                this.play(soundType);
                this.showMessage(`Testing ${soundType} sound...`);
                currentIndex++;
                setTimeout(playNextSound, 1200); // 스타크래프트 사운드에 맞게 조정
            } else {
                this.showMessage('Starcraft sound test completed!');
            }
        };
        
        playNextSound();
    }

    // 메시지 표시 (외부 함수와 연동)
    showMessage(message) {
        if (window.updateStatusMessage) {
            window.updateStatusMessage(message);
        } else {
            console.log('🎵', message);
        }
    }

    // 사운드 활성화 상태 확인
    isEnabled() {
        return this.soundEnabled;
    }

    // 사운드 비활성화
    disable() {
        this.soundEnabled = false;
        console.log('🎵 Sound disabled');
        
        // UI 업데이트
        const soundEffectsCheckbox = document.getElementById('sound-effects');
        if (soundEffectsCheckbox) {
            soundEffectsCheckbox.checked = false;
        }
    }

    // 사운드 활성화
    enable() {
        this.soundEnabled = true;
        console.log('🎵 Sound enabled');
        
        // UI 업데이트
        const soundEffectsCheckbox = document.getElementById('sound-effects');
        if (soundEffectsCheckbox) {
            soundEffectsCheckbox.checked = true;
        }
    }

    // 볼륨 설정
    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume)); // 0-1 범위로 제한
        this.masterVolume = this.volume; // 마스터 볼륨도 함께 설정
        console.log(`🎵 Volume set to: ${this.volume}`);
        
        // UI 업데이트
        const masterSlider = document.getElementById('master-volume');
        const masterValue = document.getElementById('master-volume-value');
        if (masterSlider && masterValue) {
            masterSlider.value = this.volume * 100;
            masterValue.textContent = Math.round(this.volume * 100) + '%';
        }
        
        // HTML5 오디오 요소들에 볼륨 적용
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            audio.volume = this.volume;
        });
        
        // Web Audio API 컨텍스트에 볼륨 적용
        if (window.audioContext && window.audioContext.gainNode) {
            window.audioContext.gainNode.gain.value = this.volume;
        }
    }

    // 현재 볼륨 가져오기
    getVolume() {
        return this.volume || 0.5;
    }

    // 디버그 정보 출력
    debug() {
        console.log('🎵 Audio System Debug Info:');
        console.log('- Sound enabled:', this.soundEnabled);
        console.log('- Initialized:', this.initialized);
        console.log('- User interacted:', this.userInteracted);
        console.log('- Volume:', this.getVolume());
        console.log('- User Agent:', navigator.userAgent);
    }
}

// 전역 오디오 시스템 인스턴스
let audioSystem = null;

// 전역 함수들 (기존 코드와 호환성 유지)
window.playSimpleSound = function(type) {
    if (audioSystem) {
        audioSystem.play(type);
    }
};

window.toggleSound = function() {
    if (audioSystem) {
        audioSystem.toggle();
    }
};

window.testAllSounds = function() {
    if (audioSystem) {
        audioSystem.testAll();
    }
};

window.playSoundSequence = function() {
    if (audioSystem) {
        audioSystem.playSoundSequence();
    }
};

// 디버그 함수 추가
window.debugAudio = function() {
    if (audioSystem) {
        audioSystem.debug();
    }
};

window.toggleSoundSettings = function() {
    if (audioSystem) {
        audioSystem.toggleSettings();
    }
};

window.updateMasterVolume = function() {
    if (audioSystem) {
        audioSystem.updateMasterVolume();
    }
};

window.updateClickVolume = function() {
    if (audioSystem) {
        audioSystem.updateSoundVolume('click');
    }
};

window.updateSuccessVolume = function() {
    if (audioSystem) {
        audioSystem.updateSoundVolume('success');
    }
};

window.updateErrorVolume = function() {
    if (audioSystem) {
        audioSystem.updateSoundVolume('error');
    }
};

window.updateTypeVolume = function() {
    if (audioSystem) {
        audioSystem.updateSoundVolume('type');
    }
};

window.updateSequenceVolume = function() {
    if (audioSystem) {
        audioSystem.updateSoundVolume('sequence');
    }
};

window.toggleSoundEffect = function(soundType) {
    if (audioSystem) {
        audioSystem.toggleSoundEffect(soundType);
    }
};

window.resetSoundSettings = function() {
    if (audioSystem) {
        audioSystem.resetSettings();
    }
};

// 즉시 오디오 시스템 초기화
console.log('🎵 Initializing Audio System...');
audioSystem = new AudioSystem();

// 전역으로 오디오 시스템 노출
window.audioSystem = audioSystem;

// DOM 로드 시 추가 초기화
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎵 Audio System DOM ready');
    
    // 3초 후 디버그 정보 출력
    setTimeout(() => {
        if (audioSystem) {
            audioSystem.debug();
        }
    }, 3000);
});
