// ========================================
// 트레이너 시스템 (N/B Guild NPC Control)
// ========================================

// 🏗️ 기존 트레이너 창고 시스템 제거됨
async function modifyTrainerStorage(trainer, amount) {
  console.log(`🏗️ 기존 창고 시스템 제거됨: ${trainer} ${amount > 0 ? '+' : ''}${amount.toFixed(8)} BTC`);
  return null;
}

// Function to reset trainer storage average price
async function resetTrainerStoragePrice(trainer) {
  try {
    console.log(`Resetting average price for: ${trainer}`);
    
    const response = await fetch('/api/trainer/storage/reset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        trainer: trainer
      })
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`Average price reset for: ${trainer}`);
      return result;
    } else {
      const result = await response.json();
      console.error('Failed to reset average price:', result.error);
      return null;
    }
  } catch (error) {
    console.error('Error resetting average price:', error);
    return null;
  }
}

// 🏗️ 기존 트레이너 창고 시스템 제거됨
async function modifyTrainerTicks(trainer, delta) {
  console.log(`🏗️ 기존 창고 시스템 제거됨: ${trainer} 틱 ${delta > 0 ? '+' : ''}${delta}`);
  return null;
}

// Trainer message (EN) builder
function buildTrainerMessage(iv, side, coinCount, reasons, extra) {
  const chosen = extra?.chosen || 'N/A';
  const intent = extra?.intent || 'N/A';
  const feasTxt = extra?.feasTxt || '';
  
  let message = `Interval: ${iv} | Side: ${side} | Coins: ${coinCount.toFixed(8)} BTC`;
  
  if (reasons && reasons.length > 0) {
    message += ` | Reasons: ${reasons.join(', ')}`;
  }
  
  if (chosen !== 'N/A') {
    message += ` | Chosen: ${chosen}`;
  }
  
  if (intent !== 'N/A') {
    message += ` | Intent: ${intent}`;
  }
  
  if (feasTxt) {
    message += ` | ${feasTxt}`;
  }
  
  return message;
}

// Helper function to append lines to trainer diagnostics box
function appendTrainerDiagnosticsLine(text) {
  const diagnosticsBox = document.getElementById('trainerDiagnosticsBox');
  if (diagnosticsBox) {
    const timestamp = new Date().toLocaleTimeString();
    const line = `[${timestamp}] ${text}`;
    diagnosticsBox.textContent += line + '\n';
    diagnosticsBox.scrollTop = diagnosticsBox.scrollHeight;
  }
}

// Trainer System Diagnostics Function
async function runTrainerDiagnostics() {
  try {
    appendTrainerDiagnosticsLine('Trainer System Diagnostics 시작...');
    
    // 🏗️ 기존 트레이너 창고 시스템 제거됨
    appendTrainerDiagnosticsLine('🏗️ 기존 창고 시스템 제거됨');
    
    // Test trainer suggestions
    const intervals = ['minute1', 'minute3', 'minute5', 'minute10', 'minute15', 'minute30', 'minute60', 'day'];
    for (const interval of intervals) {
      try {
        const suggestRes = await fetch(`/api/trainer/suggest?interval=${encodeURIComponent(interval)}`);
        if (suggestRes.ok) {
          const suggestData = await suggestRes.json();
          appendTrainerDiagnosticsLine(`${interval} 트레이너 제안: ${suggestData.side || 'N/A'}, ${suggestData.coin_count?.toFixed(8) || 'N/A'} BTC`);
        } else {
          appendTrainerDiagnosticsLine(`${interval} 트레이너 제안 실패`);
        }
      } catch (e) {
        appendTrainerDiagnosticsLine(`${interval} 트레이너 제안 오류: ${e.message}`);
      }
    }
    
    appendTrainerDiagnosticsLine('Trainer System Diagnostics 완료');
    
  } catch (error) {
    appendTrainerDiagnosticsLine(`Trainer System Diagnostics 오류: ${error.message}`);
  }
}

// Trainer Learning System - moved to village-learning-system.js

// Trainer Grants: simulate random BTC distribution among trainers
function distributeTrainerGrants() {
  const trainerGrantsBox = document.getElementById('trainerGrantsBox');
  if (!trainerGrantsBox) return;
  
  const now = new Date();
  const text = `[${now.toLocaleTimeString()}] Trainer Grants distributed`;
  
  const prev = String(trainerGrantsBox.textContent || '').trim();
  trainerGrantsBox.textContent = prev && prev !== '-' ? `${text}\n${prev}` : text;
}

// Split BTC portion randomly among trainers
function splitBTCAmongTrainers(btcAmount, trainers) {
  if (!trainers || trainers.length === 0) return {};
  
  const distribution = {};
  const remainingBTC = btcAmount;
  
  // Randomly distribute BTC among trainers
  trainers.forEach(trainer => {
    const randomPortion = Math.random() * remainingBTC * 0.1; // Max 10% of remaining
    distribution[trainer] = randomPortion;
  });
  
  return distribution;
}

// Get trainer storage data
async function getTrainerStorageData() {
  try {
    let trainerStorageData = {};
    
    try {
      const storageRes = await fetch('/api/trainer/storage');
      if (storageRes.ok) {
        const result = await storageRes.json();
        trainerStorageData = result.storage;
      }
    } catch (e) {
      console.error('Failed to fetch trainer storage data:', e);
    }
    
    return trainerStorageData;
  } catch (error) {
    console.error('Error getting trainer storage data:', error);
    return {};
  }
}

// Check for open position using trainer storage data
function checkTrainerPosition(member, trainerStorageData) {
  const trainerData = trainerStorageData[member.name];
  if (trainerData && trainerData.coins > 0) {
    return {
      hasPosition: true,
      coins: trainerData.coins,
      avgPrice: trainerData.avg_price,
      ticks: trainerData.ticks
    };
  }
  return { hasPosition: false };
}

// 🏗️ 기존 트레이너 창고 시스템 제거됨
function createTrainerStorageHTML(trainerStorageData, currentPrice) {
  return `
    <div style="font-weight: bold; margin-bottom: 4px; color: #ff9800;">🏗️ 기존 창고 시스템 제거됨</div>
    <div style="font-size: 12px; color: #ff9800;">기존 창고 시스템이 완전히 제거되었습니다.</div>
  `;
}

// Initialize trainer system
function initializeTrainerSystem() {
  // Set up event listeners for trainer grants
  const btnClearGrants = document.getElementById('btnClearGrants');
  if (btnClearGrants) {
    btnClearGrants.addEventListener('click', () => {
      const trainerGrantsBox = document.getElementById('trainerGrantsBox');
      if (trainerGrantsBox) trainerGrantsBox.textContent = '-';
    });
  }
  
  // Expose functions globally
  window.modifyTrainerStorage = modifyTrainerStorage;
  window.resetTrainerStoragePrice = resetTrainerStoragePrice;
  window.modifyTrainerTicks = modifyTrainerTicks;
  window.buildTrainerMessage = buildTrainerMessage;
  window.runTrainerDiagnostics = runTrainerDiagnostics;
  // window.trainerLearningSystem = trainerLearningSystem; // moved to village-learning-system.js
  window.distributeTrainerGrants = distributeTrainerGrants;
  window.getTrainerStorageData = getTrainerStorageData;
  window.checkTrainerPosition = checkTrainerPosition;
  window.createTrainerStorageHTML = createTrainerStorageHTML;
  
  console.log('Trainer System initialized');
}

// Export functions for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    modifyTrainerStorage,
    resetTrainerStoragePrice,
    modifyTrainerTicks,
    buildTrainerMessage,
    runTrainerDiagnostics,
         // trainerLearningSystem, // moved to village-learning-system.js
    distributeTrainerGrants,
    getTrainerStorageData,
    checkTrainerPosition,
    createTrainerStorageHTML,
    initializeTrainerSystem
  };
}
