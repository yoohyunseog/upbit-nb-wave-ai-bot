// ===== N/B 길드 엔진 시스템 =====

// 게임 상태
let gameRunning = true;
let gameObjects = [];
let bitcoinSquare = null;
let nbGuildPolygon = null;
let animationId;

// 캔버스 설정
let canvas, ctx;

function initializeCanvas() {
    canvas = document.getElementById('gameCanvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return false;
    }
    ctx = canvas.getContext('2d');
    return true;
}

// 게임 객체 클래스들
class GameObject {
  constructor(x, y, radius, color, name, type) {
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.color = color;
    this.name = name;
    this.type = type;
    this.vx = 0;
    this.vy = 0;
    this.targetX = x;
    this.targetY = y;
    this.status = '대기';
    this.energy = 100;
    this.pulse = 0;
    this.cards = 0;
    this.isDragging = false;
    this.dragOffsetX = 0;
    this.dragOffsetY = 0;
  }
  
  // 비트코인 4각형 클래스
  static createBitcoinSquare(x, y, size, color) {
    return {
      x: x,
      y: y,
      size: size,
      color: color,
      rotation: 0,
      pulse: 0,
      isActive: false,
      draw(ctx) {
        ctx.save();
        
        // 맥박 효과
        this.pulse += 0.05;
        const pulseScale = 1 + Math.sin(this.pulse) * 0.1;
        const scaledSize = this.size * pulseScale;
        
        // 회전
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);
        
        // 그림자
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 15;
        
        // 4각형 그리기 (다이아몬드 모양)
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.moveTo(0, -scaledSize);
        ctx.lineTo(scaledSize, 0);
        ctx.lineTo(0, scaledSize);
        ctx.lineTo(-scaledSize, 0);
        ctx.closePath();
        ctx.fill();
        
        // 테두리
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // 비트코인 로고 (₿)
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('₿', 0, 0);
        
        ctx.restore();
      },
      
      update() {
        // 천천히 회전
        this.rotation += 0.01;
      }
    };
  }
  
  // N/B 길드 다각형 클래스
  static createNBGuildPolygon(x, y, size, color) {
    return {
      x: x,
      y: y,
      size: size,
      color: color,
      rotation: 0,
      pulse: 0,
      sides: 6, // 6각형
      isActive: false,
      draw(ctx) {
        ctx.save();
        
        // 맥박 효과
        this.pulse += 0.03;
        const pulseScale = 1 + Math.sin(this.pulse) * 0.15;
        const scaledSize = this.size * pulseScale;
        
        // 회전
        ctx.translate(this.x, this.y);
        ctx.rotate(this.rotation);
        
        // 그림자
        ctx.shadowColor = this.color;
        ctx.shadowBlur = 20;
        
        // 6각형 그리기
        ctx.fillStyle = this.color;
        ctx.beginPath();
        for (let i = 0; i < this.sides; i++) {
          const angle = (i * 2 * Math.PI) / this.sides;
          const px = Math.cos(angle) * scaledSize;
          const py = Math.sin(angle) * scaledSize;
          if (i === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        }
        ctx.closePath();
        ctx.fill();
        
        // 테두리
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // N/B 로고
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('N/B', 0, 0);
        
        ctx.restore();
      },
      
      update() {
        // 반대 방향으로 회전
        this.rotation -= 0.015;
      }
    };
  }
  
  update() {
    // 부드러운 이동
    const dx = this.targetX - this.x;
    const dy = this.targetY - this.y;
    this.x += dx * 0.1;
    this.y += dy * 0.1;
    
    // 마을 이동 기능
    if (this.type === 'mayor' && this.village && this.village.isMoving) {
      // 자동 이동 (원형 경로)
      const time = Date.now() * 0.001;
      const radius = 50;
      const centerX = 400;
      const centerY = 300;
      
      this.targetX = centerX + Math.cos(time * this.village.moveSpeed) * radius;
      this.targetY = centerY + Math.sin(time * this.village.moveSpeed) * radius;
    }
    
    // 맥박 효과
    this.pulse += 0.1;
    
    // 에너지 감소
    if (this.energy > 0) {
      this.energy -= 0.1;
    }
  }
  
  draw() {
    ctx.save();
    
    // 맥박 효과
    const pulseScale = 1 + Math.sin(this.pulse) * 0.1;
    const scaledRadius = this.radius * pulseScale;
    
    // 그림자
    ctx.shadowColor = this.color;
    ctx.shadowBlur = 20;
    
    // 구슬 그리기
    const gradient = ctx.createRadialGradient(
      this.x - scaledRadius * 0.3, this.y - scaledRadius * 0.3, 0,
      this.x, this.y, scaledRadius
    );
    gradient.addColorStop(0, this.color);
    gradient.addColorStop(1, this.darkenColor(this.color, 0.3));
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(this.x, this.y, scaledRadius, 0, Math.PI * 2);
    ctx.fill();
    
    // 테두리
    ctx.strokeStyle = this.lightenColor(this.color, 0.5);
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 이름
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(this.name, this.x, this.y + this.radius + 20);
    
    // 상태
    ctx.fillStyle = this.getStatusColor();
    ctx.font = '10px Arial';
    ctx.fillText(this.status, this.x, this.y + this.radius + 35);
    
    // 카드 수
    if (this.cards > 0) {
      ctx.fillStyle = '#ffd700';
      ctx.font = 'bold 12px Arial';
      ctx.fillText(`${this.cards}`, this.x, this.y - this.radius - 10);
    }
    
    ctx.restore();
  }
  
  getStatusColor() {
    switch (this.status) {
      case '활성': return '#00ff00';
      case '분석중': return '#00d1ff';
      case '거래중': return '#ff6b6b';
      case '완료': return '#4ecdc4';
      case '실패': return '#ff4757';
      default: return '#888888';
    }
  }
  
  lightenColor(color, amount) {
    const num = parseInt(color.replace("#", ""), 16);
    const amt = Math.round(2.55 * amount * 100);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
      (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
      (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
  }
  
  darkenColor(color, amount) {
    const num = parseInt(color.replace("#", ""), 16);
    const amt = Math.round(2.55 * amount * 100);
    const R = (num >> 16) - amt;
    const G = (num >> 8 & 0x00FF) - amt;
    const B = (num & 0x0000FF) - amt;
    return "#" + (0x1000000 + (R > 255 ? 255 : R < 0 ? 0 : R) * 0x10000 +
      (G > 255 ? 255 : G < 0 ? 0 : G) * 0x100 +
      (B > 255 ? 255 : B < 0 ? 0 : B)).toString(16).slice(1);
  }
}

// 게임 초기화
function initGame() {
  gameObjects = [];
  
  // 길드장 (Mayor)
  const mayor = new GameObject(400, 300, 40, '#ffd700', 'Mayor', 'mayor');
  mayor.status = '활성';
  mayor.village = {
    isMoving: false,
    moveSpeed: 0.5
  };
  mayor.warehouse = {
    isOpen: true,
    cards: 0,
    capacity: 100
  };
  gameObjects.push(mayor);
  
  // 주민들 (Residents)
  const residents = [
    { name: 'Scout', color: '#00d1ff', x: 200, y: 200 },
    { name: 'Analyst', color: '#ff6b6b', x: 600, y: 200 },
    { name: 'Guardian', color: '#4ecdc4', x: 200, y: 400 },
    { name: 'Elder', color: '#a8e6cf', x: 600, y: 400 }
  ];
  
  residents.forEach((resident, index) => {
    const obj = new GameObject(resident.x, resident.y, 25, resident.color, resident.name, 'resident');
    obj.status = '대기';
    gameObjects.push(obj);
  });
  
  // 비트코인 센터
  bitcoinSquare = GameObject.createBitcoinSquare(800, 300, 60, '#f7931a');
  
  // N/B 길드
  nbGuildPolygon = GameObject.createNBGuildPolygon(100, 300, 50, '#ffb703');
  
  // 통계 초기화
  updateGameStats();
}

// 게임 루프
function gameLoop() {
  if (!gameRunning) return;
  
  // 캔버스 클리어
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // 배경 그라데이션
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
  gradient.addColorStop(0, '#0b1220');
  gradient.addColorStop(1, '#1e2329');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // 게임 객체 업데이트 및 그리기
  gameObjects.forEach(obj => {
    obj.update();
    obj.draw();
  });
  
  // 비트코인 센터 그리기
  if (bitcoinSquare) {
    bitcoinSquare.update();
    bitcoinSquare.draw(ctx);
  }
  
  // N/B 길드 그리기
  if (nbGuildPolygon) {
    nbGuildPolygon.update();
    nbGuildPolygon.draw(ctx);
  }
  
  // UI 그리기
  drawUI();
  
  animationId = requestAnimationFrame(gameLoop);
}

// UI 그리기
function drawUI() {
  ctx.save();
  
  // 제목
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 24px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('🏛️ N/B Guild Members Card System', canvas.width / 2, 30);
  
  // 통계
  ctx.fillStyle = '#cccccc';
  ctx.font = '16px Arial';
  ctx.textAlign = 'left';
  ctx.fillText(`Total Cards: ${getTotalCards()}`, 20, canvas.height - 80);
  ctx.fillText(`Active: ${getActiveCards()}`, 20, canvas.height - 60);
  ctx.fillText(`Completed: ${getCompletedCards()}`, 20, canvas.height - 40);
  ctx.fillText(`Failed: ${getFailedCards()}`, 20, canvas.height - 20);
  
  ctx.restore();
}

// 통계 함수들
function getTotalCards() {
  return gameObjects.reduce((total, obj) => total + obj.cards, 0);
}

function getActiveCards() {
  return gameObjects.filter(obj => obj.status === '분석중' || obj.status === '거래중').length;
}

function getCompletedCards() {
  return gameObjects.filter(obj => obj.status === '완료').length;
}

function getFailedCards() {
  return gameObjects.filter(obj => obj.status === '실패').length;
}

function updateGameStats() {
  const totalCards = getTotalCards();
  const activeCards = getActiveCards();
  const completedCards = getCompletedCards();
  const failedCards = getFailedCards();
  
  // DOM 업데이트 (안전하게 처리)
  const elements = {
    'totalCards': totalCards,
    'activeCards': activeCards,
    'completedCards': completedCards,
    'failedCards': failedCards
  };
  
  Object.keys(elements).forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = elements[id];
    }
  });
  
  // 길드원 상태 업데이트 (안전하게 처리)
  const mayor = gameObjects.find(obj => obj.type === 'mayor');
  const residents = gameObjects.filter(obj => obj.type === 'resident');
  
  if (mayor) {
    const mayorElement = document.getElementById('mayorStatus');
    if (mayorElement) {
      mayorElement.textContent = mayor.status;
    }
  }
  
  residents.forEach((resident, index) => {
    const elementId = `resident${index + 1}Status`;
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = resident.status;
    }
  });
}

// 시뮬레이션 함수들
function simulateCardCreation() {
  const residents = gameObjects.filter(obj => obj.type === 'resident');
  const randomResident = residents[Math.floor(Math.random() * residents.length)];
  
  if (randomResident && randomResident.status === '대기') {
    randomResident.cards++;
    randomResident.status = '분석중';
    
    // 3초 후 상태 변경
    setTimeout(() => {
      if (Math.random() > 0.3) {
        randomResident.status = '완료';
        randomResident.cards--;
        
        // 성공한 카드를 창고에 추가
        const mayor = gameObjects.find(obj => obj.type === 'mayor');
        if (mayor && mayor.warehouse && mayor.warehouse.isOpen) {
          if (mayor.warehouse.cards < mayor.warehouse.capacity) {
            mayor.warehouse.cards++;
          }
        }
      } else {
        randomResident.status = '실패';
        randomResident.cards--;
      }
      
      // 2초 후 대기 상태로 복귀
      setTimeout(() => {
        randomResident.status = '대기';
      }, 2000);
    }, 3000);
  }
}

// 주기적 시뮬레이션
setInterval(simulateCardCreation, 5000);

// 게임 초기화 함수
function initGameSystem() {
  if (!initializeCanvas()) {
    console.error('Canvas initialization failed');
    return;
  }
  
  // 게임 시작
  initGame();
  gameLoop();
  
  // 주기적 통계 업데이트
  setInterval(updateGameStats, 2000);
  
  console.log('🃏 Guild Members Card System - N/B 길드 엔진 완료');
  
  // 3초 후 마을 에너지 자동 충전
  setTimeout(() => {
    chargeGameEnergy();
  }, 3000);
}

// 게임 내 마을 에너지 충전 함수
function chargeGameEnergy() {
  try {
    console.log('⚡ Auto-charging game village energy...');
    
    // 게임 객체들에서 마을(Mayor) 찾기
    const mayor = gameObjects.find(obj => obj.type === 'mayor');
    if (mayor) {
      mayor.energy = 100;
      console.log('✅ 게임 마을 에너지 충전됨 (100)');
    }
    
    // 마을 이동 활성화
    if (mayor && mayor.village) {
      mayor.village.isMoving = true;
      console.log('✅ 게임 마을 이동 활성화됨');
    }
    
    console.log('⚡ 게임 마을 에너지 자동 충전 완료');
  } catch (error) {
    console.error('❌ 게임 마을 에너지 충전 중 오류:', error);
  }
}

// 전역 함수로 노출
window.initGameSystem = initGameSystem;
