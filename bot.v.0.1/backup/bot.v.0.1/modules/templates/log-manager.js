class LogManager {
    constructor() {
        this.logHistory = [];
        this.logFileName = `ai-trainer-logs-${new Date().toISOString().slice(0, 10)}.txt`;
        this.lastSaveTime = 0; // 마지막 저장 시간
        this.saveInterval = 1000; // 저장 간격 (1초로 단축)
        this.maxLogs = 100; // 최대 로그 개수 (1000개로 증가)
        this.isSaving = false; // 저장 중 플래그
    }

    // 로그 추가
    addLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = `[${timestamp}] ${message}`;
        
        // 로그 히스토리에 추가
        this.logHistory.push(logEntry);

        // 최대 로그 개수 제한 (오래된 로그 제거)
        if (this.logHistory.length > this.maxLogs) {
            this.logHistory = this.logHistory.slice(-this.maxLogs);
        }

        // 콘솔에 출력
        console.log(logEntry);

        // 1초마다 서버에 저장 (저장 중이 아닐 때만)
        const currentTime = Date.now();
        if (currentTime - this.lastSaveTime >= this.saveInterval && !this.isSaving) {
            this.saveToServer();
            this.lastSaveTime = currentTime;
        }
    }

    // 서버에 저장
    saveToServer() {
        if (this.isSaving) {
            console.log('⚠️ 이미 저장 중입니다. 건너뜁니다.');
            return;
        }

        this.isSaving = true;
        
        try {
            const logText = this.logHistory.join('\n');
            
            console.log('📝 로그 저장 시도 중...');
            console.log('📊 로그 데이터 길이:', logText.length);
            console.log('📁 파일명:', this.logFileName);
            console.log('📈 현재 로그 개수:', this.logHistory.length);
            
            // Python 서버를 통해 저장 (append 모드로 변경)
            fetch('/api/save-log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    logData: logText,
                    fileName: this.logFileName,
                    append: true // 추가 모드 플래그
                })
            })
            .then(response => {
                console.log('🌐 Python 서버 응답 상태:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('✅ 로그 저장 성공:', data);
                this.isSaving = false;
            })
            .catch(error => {
                console.error('❌ 로그 저장 실패:', error);
                this.isSaving = false;
                
                // 저장 실패 시 로컬 스토리지에 백업
                try {
                    localStorage.setItem('logBackup', logText);
                    localStorage.setItem('logBackupTime', new Date().toISOString());
                    console.log('💾 로컬 스토리지에 로그 백업 완료');
                } catch (backupError) {
                    console.error('❌ 로그 백업도 실패:', backupError);
                }
            });
        } catch (error) {
            console.error('❌ 로그 저장 중 예외 발생:', error);
            this.isSaving = false;
        }
    }

    // 현재 로그 개수 반환
    getLogCount() {
        return this.logHistory.length;
    }

    // 로그 히스토리 전체 반환
    getLogHistory() {
        return this.logHistory;
    }

    // 로그 히스토리 초기화
    clearLogs() {
        this.logHistory = [];
        console.log('🧹 로그 히스토리가 초기화되었습니다.');
    }

    // 로그 저장 강제 실행
    forceSave() {
        console.log('🔄 로그 강제 저장 실행...');
        this.isSaving = false; // 강제 저장 시 플래그 리셋
        this.saveToServer();
    }

    // 백업된 로그 복원
    restoreFromBackup() {
        try {
            const backupData = localStorage.getItem('logBackup');
            const backupTime = localStorage.getItem('logBackupTime');
            if (backupData) {
                console.log('🔄 백업된 로그 복원 중...', backupTime);
                const backupLines = backupData.split('\n');
                this.logHistory = backupLines;
                console.log('✅ 백업 로그 복원 완료:', this.logHistory.length, '개');
            }
        } catch (error) {
            console.error('❌ 백업 로그 복원 실패:', error);
        }
    }
}

// 전역 로그 매니저 인스턴스 생성
window.logManager = new LogManager();

// 백업된 로그가 있으면 복원
window.logManager.restoreFromBackup();

// 페이지 로드 시 로그 매니저 초기화 메시지
window.logManager.addLog('🔧 AI 트레이너 로그 시스템 초기화 완료 (최대 1000개 메시지 유지, 1초마다 저장)');
