// ===== Central Hub Phaser Ball System =====
/**
 * Central Hub Phaser Ball System
 * 
 * - 빈 화면에 공 1개만 표시
 * - 클릭으로 공을 움직일 수 있음
 * - 설정값과 실시간 동기화
 * - 간단하고 깔끔한 디자인
 */

// 게임 상태 관리
let game = null;
let ball = null;
let isDragging = false;
let targetX = null;
let targetY = null;
let isMovingToTarget = false;

// 중추 시스템 모듈 로드 (통합 뷰)
async function loadCentralSystem() {
    return `
        <div class="central-hub-layout">
            <div class="central-header">
                <h2><i class="fas fa-brain"></i> Central Hub - Integrated View</h2>
                <p>System Overview with Trading Dashboard & Wallet</p>
            </div>
            
            <!-- Trading Dashboard Section -->
            <div class="trading-section">
                <h3><i class="fas fa-chart-line"></i> Trading Dashboard</h3>
                <div id="trading-dashboard-container">
                    <!-- 기존 Trading Dashboard가 여기에 로드됩니다 -->
                </div>
            </div>
            
            <!-- Wallet Section -->
            <div class="wallet-section">
                <h3><i class="fas fa-wallet"></i> Wallet Status</h3>
                <div id="wallet-dashboard-container">
                    <!-- 기존 Wallet Dashboard가 여기에 로드됩니다 -->
                </div>
            </div>
            
            <!-- System Status Section -->
            <div class="system-status-section">
                <h3><i class="fas fa-server"></i> System Status</h3>
                <div class="status-grid">
                    <div class="status-item">
                        <i class="fas fa-server"></i>
                        <span>Server: Online</span>
                    </div>
                    <div class="status-item">
                        <i class="fas fa-database"></i>
                        <span>Database: Connected</span>
                    </div>
                    <div class="status-item">
                        <i class="fas fa-wifi"></i>
                        <span>Network: Stable</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Phaser 게임 설정 함수
function getGameConfig() {
    return {
        type: Phaser.AUTO,
        width: 800,
        height: 600,
        parent: 'phaser-game',
        backgroundColor: '#1a1a1a',
        physics: {
            default: 'arcade',
            arcade: {
                gravity: { y: 0 },
                debug: false
            }
        },
        scene: {
            preload: preload,
            create: create,
            update: update
        },
        render: {
            pixelArt: false,
            antialias: true
        },
        scale: {
            mode: Phaser.Scale.RESIZE,
            autoCenter: Phaser.Scale.CENTER_BOTH
        }
    };
}

// 게임 리소스 로드
function preload() {
    console.log('📦 Preloading game resources...');
    
    // 공 텍스처 생성 (원형, 더 크게)
    const graphics = this.add.graphics();
    graphics.fillStyle(0x3498db);
    graphics.fillCircle(30, 30, 30);
    graphics.generateTexture('ball', 60, 60);
    graphics.destroy();
    
    console.log('✅ Ball texture created');
}

// 게임 생성
function create() {
    console.log('🎮 Creating Phaser game...');
    
    // 화면 크기 가져오기
    const screenWidth = this.cameras.main.width;
    const screenHeight = this.cameras.main.height;
    
    console.log(`📐 Screen size: ${screenWidth}x${screenHeight}`);
    
    // 간단한 배경 그리드 생성
    const graphics = this.add.graphics();
    graphics.lineStyle(1, 0x2c3e50, 0.3);
    
    // 그리드 그리기 (50x50 픽셀 간격으로 더 크게)
    for (let x = 0; x < screenWidth; x += 50) {
        graphics.moveTo(x, 0);
        graphics.lineTo(x, screenHeight);
    }
    for (let y = 0; y < screenHeight; y += 50) {
        graphics.moveTo(0, y);
        graphics.lineTo(screenWidth, y);
    }
    
    // 맵 경계선 그리기
    graphics.lineStyle(3, 0x3498db, 0.8);
    graphics.strokeRect(0, 0, screenWidth, screenHeight * 0.85);
    
    // 공 생성 (화면 중앙)
    ball = this.add.sprite(screenWidth / 2, screenHeight / 2, 'ball');
    ball.setInteractive();
    
    console.log(`🔵 Ball created at: (${screenWidth / 2}, ${screenHeight / 2})`);
    
    // 물리 바디 추가
    this.physics.add.existing(ball);
    ball.body.setCollideWorldBounds(true);
    ball.body.setBounce(0.8, 0.8);
    ball.body.setDrag(200, 200);
    
    // 공 크기 설정 (더 크게)
    ball.setScale(1.0);
    
    // 전역 변수로 노출
    window.ball = ball;
    window.game = this;
    
    // 클릭 이벤트 설정
    this.input.on('pointerdown', (pointer) => {
        console.log(`🖱️ Click at: (${pointer.x}, ${pointer.y})`);
        
        // 멤버 카드 영역 클릭 방지
        if (pointer.y > screenHeight * 0.85) {
            return;
        }
        
        // 목표 위치 설정
        targetX = pointer.x;
        targetY = pointer.y;
        isMovingToTarget = true;
        
        if (ball.getBounds().contains(pointer.x, pointer.y)) {
            ball.setTint(0xffeb3b);
            setTimeout(() => ball.clearTint(), 200);
        }
    });
    
    console.log('✅ Phaser Ball created successfully');
}

// 게임 업데이트
function update() {
    // 목표 위치로 계속 이동
    if (isMovingToTarget && targetX !== null && targetY !== null && ball) {
        // 현재 위치에서 목표 위치로의 방향과 거리 계산
        const dx = targetX - ball.x;
        const dy = targetY - ball.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // 목표에 가까워지면 정지
        if (distance < 10) {
            ball.body.setVelocity(0, 0);
            isMovingToTarget = false;
            targetX = null;
            targetY = null;
        } else {
            // 속도 설정 (거리에 비례, 최소 속도 보장)
            const speed = Math.max(distance * 0.3, 100);
            
            // 물리 속도 적용
            ball.body.setVelocity(
                (dx / distance) * speed,
                (dy / distance) * speed
            );
        }
    }
    
    // 맵 경계 내에서만 이동 가능하도록 제한
    if (ball) {
        const screenHeight = this.cameras.main.height;
        const maxY = screenHeight * 0.85; // 멤버 카드 영역 제외
        
        if (ball.y > maxY - ball.height / 2) {
            ball.y = maxY - ball.height / 2;
            ball.body.setVelocityY(0);
        }
    }
}

// 공 리셋 함수 (더 이상 사용되지 않음)
function resetBall() {
    // 함수는 유지하되 내부 로직 제거 (버튼이 없어졌으므로)
}

// 게임 토글 함수 (더 이상 사용되지 않음)
function toggleGame() {
    // 함수는 유지하되 내부 로직 제거 (버튼이 없어졌으므로)
}

// Phaser 내장 캡처 기능 (더 이상 사용되지 않음)
function takePhaserScreenshot() {
    // 함수는 유지하되 내부 로직 제거 (버튼이 없어졌으므로)
}

// Phaser 스크린샷 다운로드 (더 이상 사용되지 않음)
function downloadPhaserScreenshot(canvas) {
    // 함수는 유지하되 내부 로직 제거 (버튼이 없어졌으므로)
}

// 중추 시스템 초기화
function initializeCentralSystem() {
    console.log('🏛️ Initializing Central Hub (Phaser Ball)...');
    
    // Phaser 라이브러리 로드 확인
    if (typeof Phaser === 'undefined') {
        console.debug('⚠️ Phaser library not loaded yet - waiting for CDN...');
        // 1초 후 다시 시도
        setTimeout(() => {
            if (typeof Phaser !== 'undefined') {
                initializeCentralSystem();
            } else {
                console.error('❌ Phaser library failed to load from CDN');
            }
        }, 1000);
        return;
    }
    
    // 기존 게임이 있다면 제거
    if (game) {
        game.destroy(true);
        game = null;
    }
    
    // 게임 시작
    try {
        game = new Phaser.Game(getGameConfig());
        console.log('✅ Central Hub Phaser Ball initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize Phaser game:', error);
    }
}

// 중추 시스템 상태 업데이트
function updateCentralSystemStatus() {
    if (ball) {
        console.log(`🏛️ Ball position: (${Math.round(ball.x)}, ${Math.round(ball.y)})`);
    }
}

// 이벤트 리스트에 이벤트 추가
function addEventToList(eventText) {
    console.log('🏛️ Central Hub Event:', eventText);
}

// 시스템 상태 새로고침
async function refreshSystemStatus() {
    console.log('🏛️ Central Hub status refreshed');
    if (game && game.scene.isActive('default')) {
        console.log('✅ Phaser game is running');
    } else {
        console.log('⚠️ Phaser game is paused or stopped');
    }
}

// 시스템 통계 표시
function showSystemStats() {
    const ballPos = ball ? `(${Math.round(ball.x)}, ${Math.round(ball.y)})` : 'N/A';
    const gameStatus = game && game.scene.isActive('default') ? 'Running' : 'Paused';
    
    alert(`Central Hub 통계:

시스템 상태: ${gameStatus}
모듈: Central Hub Phaser Ball
공 위치: ${ballPos}
드래그 상태: ${isDragging ? '활성' : '비활성'}

Phaser 게임이 실행 중입니다.`);
}

// 중추 시스템 상태 조회
function getCentralSystemState() {
    return {
        currentModule: 'central-hub-phaser',
        activeModules: ['phaser-ball-game'],
        systemStatus: {
            server: 'Online',
            api: 'Connected',
            database: 'Connected',
            phaser: game ? 'Running' : 'Stopped'
        },
        events: [],
        status: 'Phaser Ball Game Active',
        ballPosition: ball ? { x: Math.round(ball.x), y: Math.round(ball.y) } : null
    };
}

// 중추 시스템 통계 조회
function getCentralSystemStats() {
    return {
        totalEvents: 0,
        activeModules: 1,
        currentModule: 'central-hub-phaser',
        systemUptime: new Date() - new Date(),
        status: 'Phaser Ball Game',
        ballPosition: ball ? { x: Math.round(ball.x), y: Math.round(ball.y) } : null,
        isDragging: isDragging
    };
}

// 중추 시스템 리셋
function resetCentralSystem() {
    if (confirm('Central Hub를 리셋하시겠습니까?')) {
        console.log('🏛️ Central Hub reset');
        resetBall();
    }
}

// 모듈 전환
function switchModule(moduleName) {
    console.log(`🏛️ Module switched to: ${moduleName}`);
}

// 이벤트 리스너 추가
function addEventListener(eventName, callback) {
    console.log(`🏛️ Event listener added for: ${eventName}`);
}

// 이벤트 발생
function emitEvent(eventName, data) {
    console.log(`🏛️ Event emitted: ${eventName}`, data);
}

// 전역 함수로 노출
window.loadCentralSystem = loadCentralSystem;
window.initializeCentralSystem = initializeCentralSystem;
window.updateCentralSystemStatus = updateCentralSystemStatus;
window.addEventToList = addEventToList;
window.refreshSystemStatus = refreshSystemStatus;
window.showSystemStats = showSystemStats;
window.getCentralSystemState = getCentralSystemState;
window.getCentralSystemStats = getCentralSystemStats;
window.resetCentralSystem = resetCentralSystem;
window.switchModule = switchModule;
window.emitEvent = emitEvent;
window.resetBall = resetBall;
window.toggleGame = toggleGame;
window.takePhaserScreenshot = takePhaserScreenshot;

// 스타일 추가
const phaserStyles = `
    .central-hub-layout {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        position: relative;
    }
    
    .phaser-container {
        flex: 1;
        padding: 0;
        margin: 0;
        width: 100%;
        height: 85%;
    }
    
    #phaser-game {
        margin: 0;
        padding: 0;
        width: 100% !important;
        height: 100% !important;
        display: block !important;
        visibility: visible !important;
    }
    
    #phaser-game canvas {
        width: 100% !important;
        height: 100% !important;
    }
    
    .member-cards-container {
        width: 100%;
        height: 15%;
        display: flex;
        flex-direction: row;
        background: rgba(44, 62, 80, 0.95);
        border-top: 2px solid #34495e;
        overflow-x: auto;
        padding: 10px;
        gap: 10px;
    }
    
    .member-card {
        display: flex;
        align-items: center;
        padding: 10px;
        margin: 0;
        background: rgba(52, 73, 94, 0.8);
        border: 1px solid #34495e;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        flex: 1;
        min-width: 0;
    }
    
    .member-card:hover {
        background: rgba(52, 152, 219, 0.3);
        border-color: #3498db;
        transform: translateY(-5px);
    }
    
    .member-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #3498db;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-right: 10px;
        border: 2px solid #2980b9;
    }
    
    .member-info {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
    
    .member-name {
        font-weight: bold;
        color: #ecf0f1;
        font-size: 14px;
        margin-bottom: 2px;
    }
    
    .member-status {
        font-size: 12px;
        padding: 2px 6px;
        border-radius: 10px;
        text-align: center;
        width: fit-content;
    }
    
    .member-status.online {
        background: rgba(46, 204, 113, 0.3);
        color: #2ecc71;
        border: 1px solid #27ae60;
    }
    
    .member-status.offline {
        background: rgba(231, 76, 60, 0.3);
        color: #e74c3c;
        border: 1px solid #c0392b;
    }
    
    .member-status.busy {
        background: rgba(243, 156, 18, 0.3);
        color: #f39c12;
        border: 1px solid #e67e22;
    }
`;

// 스타일 추가
if (!document.getElementById('phaser-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'phaser-styles';
    styleElement.textContent = phaserStyles;
    document.head.appendChild(styleElement);
}

console.log('🏛️ Central Hub (Phaser Ball) module loaded');
