// Lightweight Charts UI (pan only) + order markers using bot server APIs
(function(){
  const container = document.getElementById('tvChart');
  if (!container) return;
  const tfEl = document.getElementById('timeframe');
  const getInterval = () => (tfEl ? tfEl.value : 'minute10');

  // Use same-origin base to avoid mixed-content/host issues
  const base = '';
  const startBtn = document.getElementById('botStart');
  const stopBtn = document.getElementById('botStop');
  const shotBtn = document.getElementById('btnShot');
  const btBtn = document.getElementById('btnBacktest');
  const clearBtn = document.getElementById('btnClearOrders');
  const ordersToggle = document.getElementById('ordersToggle');
  const optBtn = document.getElementById('btnOptimize');
  const trainBtn = document.getElementById('btnTrain');
  const mlTrainBtn = document.getElementById('btnMlTrain');
  const mlPredictBtn = document.getElementById('btnMlPredict');
  const miZone = document.getElementById('miZone');
  const miText = document.getElementById('miText');

  function updateModelInsight(j){
    try{
      const ins = j && j.insight ? j.insight : {};
      if (miZone){ miZone.textContent = String(ins.zone||'-'); miZone.className = 'badge bg-white text-dark'; }
      if (miText){
        const blueAdj = (ins.pct_blue||0);
        const orangeAdj = (ins.pct_orange||0);
        const blueRaw = (ins.pct_blue_raw!=null? ins.pct_blue_raw : blueAdj);
        const orangeRaw = (ins.pct_orange_raw!=null? ins.pct_orange_raw : orangeAdj);
        miText.innerHTML = `r=${(ins.r||0).toFixed(3)} | BLUE(raw)=${Number(blueRaw).toFixed(1)}% | ORANGE(raw)=${Number(orangeRaw).toFixed(1)}% | BLUE=${Number(blueAdj).toFixed(1)}% | ORANGE=${Number(orangeAdj).toFixed(1)}% | conf=${(ins.zone_conf||0).toFixed(3)} | w=${(ins.w||0).toFixed(3)}<br/>`+
          `dist_high=${(ins.dist_high||0).toFixed(3)} | dist_low=${(ins.dist_low||0).toFixed(3)} | gap=${(ins.extreme_gap||0).toFixed(3)} | ema_diff=${(ins.ema_diff||0).toFixed(1)}<br/>`+
          `zone_min_r=${(ins.zone_min_r!=null? ins.zone_min_r: ins.r||0).toFixed(3)} | zone_max_r=${(ins.zone_max_r!=null? ins.zone_max_r: ins.r||0).toFixed(3)} | zone_extreme_r=${(ins.zone_extreme_r!=null? ins.zone_extreme_r: ins.r||0).toFixed(3)} | zone_extreme_age=${Number(ins.zone_extreme_age||0)}`;
      }
    }catch(_){ }
  }
  const mlMetricsBtn = document.getElementById('btnMlMetrics');
  const mlRandomBtn = document.getElementById('btnMlRandom');
  const mlRandNEl = document.getElementById('mlRandN');
  const mlAutoToggle = document.getElementById('mlAuto');
  let mlAutoTimer = null;
  const loadBalBtn = document.getElementById('btnLoadBalance');
  // Top assets UI
  const assetsBox = null; // legacy removed
  const assetsMeta = document.getElementById('assetsMeta');
  const assetsRefresh = document.getElementById('assetsRefresh');
  const assetsAutoToggle = document.getElementById('assetsAuto');
  // new assets design elements
  const assetTotalEl = document.getElementById('assetTotal');
  const assetBuyableEl = document.getElementById('assetBuyable');
  const assetSellableEl = document.getElementById('assetSellable');
  const assetsBars = document.getElementById('assetsBars');
  let assetsTimer = null;
  const assetsSummary = null;
  const mlCountEl = document.getElementById('mlCount');
  const trainCountEl = document.getElementById('trainCount');
  const trainSegEl = document.getElementById('trainSeg');
  const autoBtToggle = document.getElementById('autoBtToggle');
  const autoBtSecEl = document.getElementById('autoBtSec');
  let autoBtTimer = null;
  const logBox = document.getElementById('logBox');
  const logAuto = document.getElementById('logAutoscroll');
  const logClearBtn = document.getElementById('btnClearLog');
  const LOG_MAX_LINES = 50;
  // Orders bottom log elements
  const orderLog = document.getElementById('orderLog');
  const orderClearBtn = document.getElementById('btnOrderClear');
  const orderExportBtn = document.getElementById('btnOrderExport');
  let orderKeys = new Set();
  const mlMetricsBox = document.getElementById('mlMetricsBox');
  const emaFilterEl = document.getElementById('emaFilter');
  const nbFromEmaEl = document.getElementById('nbFromEma');
  const nbEmaPeriodEl = document.getElementById('nbEmaPeriod');
  const nbDebounceEl = document.getElementById('nbDebounce');
  const nbBuyThEl = document.getElementById('nbBuyTh');
  const nbSellThEl = document.getElementById('nbSellTh');
  const showSMAEl = document.getElementById('showSMA');
  const sma50El = document.getElementById('sma50');
  const sma100El = document.getElementById('sma100');
  const sma200El = document.getElementById('sma200');
  const showEMA9El = document.getElementById('showEMA9');
  const showIchimokuEl = document.getElementById('showIchimoku');
  const ichiTenkanEl = document.getElementById('ichiTenkan');
  const ichiKijunEl = document.getElementById('ichiKijun');
  function uiLog(msg, data){
    try{
      const ts = new Date().toISOString();
      const detail = data? (typeof data==='string'? data: JSON.stringify(data)) : '';
      const line = `[${ts}] ${msg}${detail? ' ' + detail: ''}`;
      if (logBox){
        // append
        const prevTop = logBox.scrollTop;
        const prevHeight = logBox.scrollHeight;
        logBox.textContent += (line + "\n");
        // trim to last LOG_MAX_LINES
        try{
          const parts = logBox.textContent.split('\n');
          if (parts.length > LOG_MAX_LINES+1){
            logBox.textContent = parts.slice(-LOG_MAX_LINES-1).join('\n');
          }
        }catch(_){ }
        // autoscroll only if explicit toggle exists and is ON
        const shouldScroll = !!(logAuto && logAuto.checked);
        if (shouldScroll){
          logBox.scrollTop = logBox.scrollHeight;
        } else {
          // restore previous scroll position
          try{ logBox.scrollTop = prevTop; }catch(_){ }
        }
      }
      console.log(line);
    }catch(_){ }
  }
  async function optimizeNb(){
    try{
      const payload = {
        window: parseInt(nbWindowEl?.value||'50',10),
        buy: [0.6, 0.85, 0.02],
        sell: [0.15, 0.45, 0.02],
        debounce: parseInt(nbDebounceEl?.value||'6',10),
        fee_bps: 10.0,
        count: 800,
        interval: getInterval(),
      };
      const r = await fetch('/api/nb/optimize', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const j = await r.json();
      if (j && j.ok && j.best){
        if (nbBuyThEl) nbBuyThEl.value = String(j.best.buy);
        if (nbSellThEl) nbSellThEl.value = String(j.best.sell);
        updateNB();
      }
    }catch(_){ }
  }
  const paperEl = document.getElementById('paperMode');
  const orderEl = document.getElementById('orderKrw');
  const emaFastEl = document.getElementById('emaFast');
  const emaSlowEl = document.getElementById('emaSlow');
  // Forecast controls
  const fcWindowEl = document.getElementById('fcWindow');
  const fcHorizonEl = document.getElementById('fcHorizon');
  const fcToggleEl = document.getElementById('fcToggle');

  const sTicker = document.getElementById('s_ticker');
  const sPrice = document.getElementById('s_price');
  const sSignal = document.getElementById('s_signal');
  const sEma = document.getElementById('s_ema');
  const sBot = document.getElementById('s_bot');
  const sInterval = document.getElementById('s_interval');
  const sEntry = document.getElementById('nb_entry');
  const sPnl = document.getElementById('nb_pnl');
  // Top PnL slider elements
  const pnlLeft = document.getElementById('pnlLeftBar');
  const pnlRight = document.getElementById('pnlRightBar');
  const pnlLeftLabel = document.getElementById('pnlLeftLabel');
  const pnlRightLabel = document.getElementById('pnlRightLabel');
  let lastAggPct = 0;
  // Track last live BUY to compute realized PnL on SELL
  let liveLastBuyPrice = 0;
  function updateTopPnlSlider(pnl, winRate){
    if (!pnlLeft || !pnlRight) return;
    // Normalize 0~100; draw from center: LeftBar extends left, RightBar extends right
    const isProfit = pnl >= 0;
    const wr = Math.max(0, Math.min(100, isFinite(winRate)? winRate : (isProfit? 60:40)));
    const profPct = isProfit ? wr : 0;
    const lossPct = isProfit ? (100-wr) : wr;
    pnlLeft.style.width = `${profPct/2}%`; // half track to the left
    pnlRight.style.width = `${lossPct/2}%`; // half track to the right
    if (pnlLeftLabel) pnlLeftLabel.textContent = `Profit ${profPct.toFixed(1)}%`;
    if (pnlRightLabel) pnlRightLabel.textContent = `Loss ${lossPct.toFixed(1)}%`;
    // glow animation cue
    try{
      (isProfit? pnlLeft : pnlRight).classList.remove(isProfit? 'glow-green':'glow-red');
      void (isProfit? pnlLeft : pnlRight).offsetWidth; // reflow to restart animation
      (isProfit? pnlLeft : pnlRight).classList.add(isProfit? 'glow-green':'glow-red');
    }catch(_){ }
  }

  // Aggregate last up to 9 win buttons
  function updateTopPnlFromList(){
    if (!winListEl || !pnlLeft || !pnlRight) return;
    const items = Array.from(winListEl.children).slice(0,9);
    const total = items.length;
    const positives = items.filter(el=> el.classList.contains('positive')).length;
    const profitPct = total ? (positives/total*100) : 0;
    const lossPct = 100 - profitPct;
    pnlLeft.style.width = `${profitPct/2}%`;
    pnlRight.style.width = `${lossPct/2}%`;
    if (pnlLeftLabel) pnlLeftLabel.textContent = `Profit ${profitPct.toFixed(1)}%`;
    if (pnlRightLabel) pnlRightLabel.textContent = `Loss ${lossPct.toFixed(1)}%`;
    // Push profit/loss ratios to server for order sizing
    try {
      postJson('/api/bot/config', { pnl_profit_ratio: profitPct, pnl_loss_ratio: lossPct }).catch(()=>{});
    } catch(_) { }
    // glow on direction
    try{
      if (profitPct >= lastAggPct){ pnlLeft.classList.remove('glow-green'); void pnlLeft.offsetWidth; pnlLeft.classList.add('glow-green'); }
      else { pnlRight.classList.remove('glow-red'); void pnlRight.offsetWidth; pnlRight.classList.add('glow-red'); }
      lastAggPct = profitPct;
    }catch(_){ }
  }
  // ---- Rolling win list (left panel) ----
  const winListEl = document.getElementById('winList');
  const winClearBtn = document.getElementById('winClear');
  let winKeys = new Set();
  const makeWinKey = (pnl, winRate)=> `${Math.round(pnl)}|${Number(winRate).toFixed(1)}`;
  function pushWinItem({ ts, pnl, winRate }){
    if (!winListEl) return;
    const key = makeWinKey(pnl, winRate);
    // If duplicate exists, refresh its content and move to top
    const dup = Array.from(winListEl.children).find(el=> el.dataset && el.dataset.key === key);
    if (dup){
      dup.classList.remove('positive','negative');
      dup.classList.add(pnl>=0? 'positive':'negative');
      const timeStr = new Date(ts).toLocaleTimeString();
      const meta = dup.querySelector('.meta'); if (meta) meta.textContent = timeStr;
      const sign = pnl>=0? '+':'-'; const abs = Math.abs(pnl);
      const wrSign = pnl>=0? '+':'-';
      const full = `PnL ${sign}${abs.toLocaleString()} | Win% ${wrSign}${Number(winRate).toFixed(1)}%`;
      const val = dup.querySelector('.val'); if (val) val.textContent = full;
      winListEl.prepend(dup);
      updateTopPnlFromList();
      return;
    }
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'win-chip btn btn-sm ' + (pnl>=0? 'positive':'negative');
    const sign = pnl>=0? '+':'-';
    const abs = Math.abs(pnl);
    const wrSign = pnl>=0? '+':'-';
    const timeStr = new Date(ts).toLocaleTimeString();
    const full = `PnL ${sign}${abs.toLocaleString()} | Win% ${wrSign}${winRate.toFixed(1)}%`;
    item.title = `${timeStr}  ${full}`;
    item.innerHTML = `<div class='meta'>${timeStr}</div><div class='val'>${full}</div>`;
    item.dataset.key = key;
    winListEl.prepend(item);
    // keep last 9
    while (winListEl.childElementCount>9){ const last = winListEl.lastElementChild; if (last && last.dataset && last.dataset.key) winKeys.delete(last.dataset.key); winListEl.removeChild(last); }
    winKeys.add(key);
    // refresh top aggregate slider
    updateTopPnlFromList();
  }
  if (winClearBtn) winClearBtn.addEventListener('click', ()=>{ if (winListEl) winListEl.innerHTML=''; winKeys.clear(); updateTopPnlFromList(); });

  // periodic top slider refresh to ensure UI remains in sync
  let topPnlTimer = setInterval(()=>{ try{ updateTopPnlFromList(); }catch(_){} }, 2000);

  // ---- Local storage for options ----
  const LS_KEY = 'eightbit_ui_opts_v1';
  function readOpts(){
    try{ return JSON.parse(localStorage.getItem(LS_KEY)||'{}'); }catch(_){ return {}; }
  }
  function writeOpts(partial){
    const cur = readOpts();
    const next = { ...cur, ...partial };
    try{ localStorage.setItem(LS_KEY, JSON.stringify(next)); }catch(_){ }
  }
  const saveOpts = ()=>{ try{ writeOpts(snapshotOpts()); }catch(_){ } };
  function snapshotOpts(){
    return {
      timeframe: tfEl ? tfEl.value : undefined,
      paper: paperEl ? paperEl.value : undefined,
      order_krw: orderEl ? orderEl.value : undefined,
      ema_fast: emaFastEl ? emaFastEl.value : undefined,
      ema_slow: emaSlowEl ? emaSlowEl.value : undefined,
      nb_window: nbWindowEl ? nbWindowEl.value : undefined,
      nb_show: nbToggleEl ? !!nbToggleEl.checked : undefined,
      nb_buy_th: nbBuyThEl ? nbBuyThEl.value : undefined,
      nb_sell_th: nbSellThEl ? nbSellThEl.value : undefined,
      nb_debounce: nbDebounceEl ? nbDebounceEl.value : undefined,
      ema_filter: emaFilterEl ? !!emaFilterEl.checked : undefined,
      nb_from_ema: nbFromEmaEl ? !!nbFromEmaEl.checked : undefined,
      nb_ema_period: nbEmaPeriodEl ? nbEmaPeriodEl.value : undefined,
      fc_window: (typeof fcWindowEl !== 'undefined' && fcWindowEl) ? fcWindowEl.value : undefined,
      fc_horizon: (typeof fcHorizonEl !== 'undefined' && fcHorizonEl) ? fcHorizonEl.value : undefined,
      fc_show: (typeof fcToggleEl !== 'undefined' && fcToggleEl) ? !!fcToggleEl.checked : undefined,
      show_orders: (typeof ordersToggle !== 'undefined' && ordersToggle) ? !!ordersToggle.checked : undefined,
      auto_bt: (typeof autoBtToggle !== 'undefined' && autoBtToggle) ? !!autoBtToggle.checked : undefined,
      auto_bt_sec: (typeof autoBtSecEl !== 'undefined' && autoBtSecEl) ? autoBtSecEl.value : undefined,
      show_sma: showSMAEl ? !!showSMAEl.checked : undefined,
      sma50: sma50El ? sma50El.value : undefined,
      sma100: sma100El ? sma100El.value : undefined,
      sma200: sma200El ? sma200El.value : undefined,
      show_ema9: showEMA9El ? !!showEMA9El.checked : undefined,
      show_ichimoku: showIchimokuEl ? !!showIchimokuEl.checked : undefined,
      ichi_tenkan: ichiTenkanEl ? ichiTenkanEl.value : undefined,
      ichi_kijun: ichiKijunEl ? ichiKijunEl.value : undefined,
      train_count: (typeof trainCountEl !== 'undefined' && trainCountEl) ? trainCountEl.value : undefined,
      train_seg: (typeof trainSegEl !== 'undefined' && trainSegEl) ? trainSegEl.value : undefined,
    };
  }

  const postJson = (path, data) => fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data || {})
  }).then(r=>r.json()).catch((e)=>{ console.error('POST fail', path, e); return { ok:false, error:String(e) }; });

  async function fetchJsonStrict(path, init){
    const r = await fetch(path, init);
    const ct = (r.headers.get('content-type')||'').toLowerCase();
    const text = await r.text();
    if (!ct.includes('application/json')){
      throw new Error('API 응답이 JSON이 아닙니다. Flask UI에서 여세요: http://127.0.0.1:5057/ui');
    }
    try{ return JSON.parse(text); }catch(_){ throw new Error('JSON 파싱 실패: ' + text.slice(0,120)); }
  }

  const sleep = (ms)=> new Promise(res=>setTimeout(res, ms));

  async function waitForNbSignals(maxMs=5000){
    const start = Date.now();
    while (Date.now()-start < maxMs){
      try{
        updateNB();
        const data = candle.data();
        const hasSignals = (Array.isArray(nbMarkers) && nbMarkers.length > 0) || (Array.isArray(window.lastNbSignals) && window.lastNbSignals.length>0);
        if (data && data.length >= 50 && hasSignals){ return true; }
      }catch(_){ }
      await sleep(200);
    }
    return false;
  }

  async function backtestAfterReady(maxMs=6000){
    try{ await waitForNbSignals(maxMs); runBacktest(); }catch(_){ }
  }

  function collectConfig(){
    return {
      paper: (paperEl && paperEl.value === 'true'),
      order_krw: orderEl ? parseInt(orderEl.value||'5000',10) : 5000,
      ema_fast: emaFastEl ? parseInt(emaFastEl.value||'10',10) : 10,
      ema_slow: emaSlowEl ? parseInt(emaSlowEl.value||'30',10) : 30,
      candle: getInterval(),
    };
  }

  async function pushConfig(){
    const cfg = collectConfig();
    await postJson('/api/bot/config', cfg);
    if (sEma) sEma.textContent = `${cfg.ema_fast}/${cfg.ema_slow}`;
    if (sInterval) sInterval.textContent = cfg.candle;
    writeOpts(snapshotOpts());
  }
  const chart = LightweightCharts.createChart(container, {
    layout: { background: { type: 'solid', color: '#0b1220' }, textColor: '#e6eefc' },
    grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
    rightPriceScale: { borderColor: 'rgba(255,255,255,0.08)' },
    timeScale: { borderColor: 'rgba(255,255,255,0.08)' },
    crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { mouseWheel: false, pinch: false, axisPressedMouseMove: false },
    autoSize: true,
  });

  const candle = chart.addCandlestickSeries({ upColor:'#0ecb81', downColor:'#f6465d', wickUpColor:'#0ecb81', wickDownColor:'#f6465d', borderVisible:false });
  const emaF = chart.addLineSeries({ color:'rgba(14,203,129,0.9)', lineWidth:2 });
  const emaS = chart.addLineSeries({ color:'rgba(246,70,93,0.9)', lineWidth:2 });
  const sma50Series = chart.addLineSeries({ color:'#9aa0a6', lineWidth:1, priceLineVisible:false });
  const sma100Series = chart.addLineSeries({ color:'#c7cbd1', lineWidth:1, priceLineVisible:false });
  const sma200Series = chart.addLineSeries({ color:'#e0e3e7', lineWidth:1, priceLineVisible:false });
  const ema9Series = chart.addLineSeries({ color:'#ffd166', lineWidth:1, priceLineVisible:false });
  const ema12Series = chart.addLineSeries({ color:'#fca311', lineWidth:1, priceLineVisible:false });
  const ema26Series = chart.addLineSeries({ color:'#fb8500', lineWidth:1, priceLineVisible:false });
  const ichiTenkanSeries = chart.addLineSeries({ color:'#00d1ff', lineWidth:1, priceLineVisible:false });
  const ichiKijunSeries = chart.addLineSeries({ color:'#ff006e', lineWidth:1, priceLineVisible:false });

  // ---- NB helpers (using user's BIT_* functions) ----
  function initializeArraysBIT(length){
    return {
      BIT_START_A50: new Array(length).fill(0),
      BIT_START_A100: new Array(length).fill(0),
      BIT_START_B50: new Array(length).fill(0),
      BIT_START_B100: new Array(length).fill(0),
      BIT_START_NBA100: new Array(length).fill(0),
    };
  }
  function calculateBit(nb, bit=5.5, reverse=false){
    if (!nb || nb.length < 2) return bit/100;
    const BIT_NB = bit;
    const max = Math.max(...nb);
    const min = Math.min(...nb);
    const COUNT = 50;
    const rangeN = min < 0 ? Math.abs(min) : 0;
    const rangeP = max > 0 ? max : 0;
    const incN = rangeN / (COUNT * nb.length - 1 || 1);
    const incP = rangeP / (COUNT * nb.length - 1 || 1);
    const arrays = initializeArraysBIT(COUNT * nb.length);
    let count = 0; let NB50 = 0;
    for (const value of nb){
      for (let i=0;i<COUNT;i++){
        const A50 = value < 0 ? (min + incN * (count+1)) : (min + incP * (count+1));
        const A100 = (count+1) * BIT_NB / (COUNT * nb.length);
        const B50 = value < 0 ? (A50 - incN * 2) : (A50 - incP * 2);
        const B100 = value < 0 ? (A50 + incN) : (A50 + incP);
        const NBA100 = A100 / (nb.length - 1);
        arrays.BIT_START_A50[count] = A50;
        arrays.BIT_START_A100[count] = A100;
        arrays.BIT_START_B50[count] = B50;
        arrays.BIT_START_B100[count] = B100;
        arrays.BIT_START_NBA100[count] = NBA100;
        count++;
      }
    }
    if (reverse) arrays.BIT_START_NBA100.reverse();
    for (const value of nb){
      for (let a=0;a<arrays.BIT_START_NBA100.length;a++){
        if (arrays.BIT_START_B50[a] <= value && arrays.BIT_START_B100[a] >= value){
          NB50 += arrays.BIT_START_NBA100[Math.min(a, arrays.BIT_START_NBA100.length-1)];
          break;
        }
      }
    }
    if (nb.length === 2) return bit - NB50;
    return NB50;
  }
  let SUPER_BIT = 0;
  function updateSuperBit(v){ SUPER_BIT = v; }
  function BIT_MAX_NB(nb, bit=5.5){
    let r = calculateBit(nb, bit, false);
    if (!isFinite(r) || isNaN(r) || r > 100 || r < -100) return SUPER_BIT; else { updateSuperBit(r); return r; }
  }
  function BIT_MIN_NB(nb, bit=5.5){
    let r = calculateBit(nb, bit, true);
    if (!isFinite(r) || isNaN(r) || r > 100 || r < -100) return SUPER_BIT; else { updateSuperBit(r); return r; }
  }

  // NB UI controls
  const nbWindowEl = document.getElementById('nbWindow');
  const nbToggleEl = document.getElementById('nbToggle');
  const sNbMax = document.getElementById('s_nbMax');
  const sNbMin = document.getElementById('s_nbMin');
  const sNbState = document.getElementById('s_nbState');
  const nbMaxSeries = chart.addAreaSeries({
    topColor: 'rgba(255,183,3,0.55)',
    bottomColor: 'rgba(255,183,3,0.20)',
    lineColor: '#ffb703', lineWidth: 3,
    lastValueVisible: true, priceLineVisible: true, priceLineColor: '#ffb703'
  });
  const nbMinSeries = chart.addAreaSeries({
    topColor: 'rgba(0,209,255,0.55)',
    bottomColor: 'rgba(0,209,255,0.20)',
    lineColor: '#00d1ff', lineWidth: 3,
    lastValueVisible: true, priceLineVisible: true, priceLineColor: '#00d1ff'
  });
  // Baseline wave series (visual emphasis)
  const nbWaveSeries = chart.addBaselineSeries({
    baseValue: { type: 'price', price: 0 },
    topFillColor1: 'rgba(255,183,3,0.50)',
    topFillColor2: 'rgba(255,183,3,0.20)',
    topLineColor: '#ffb703',
    bottomFillColor1: 'rgba(0,209,255,0.50)',
    bottomFillColor2: 'rgba(0,209,255,0.20)',
    bottomLineColor: '#00d1ff',
    lineWidth: 4,
  });
  function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
  let nbMaxPriceLine = null; let nbMinPriceLine = null;
  function updateNB(){
    try{
      const n = parseInt((nbWindowEl && nbWindowEl.value) || '100', 10);
      const data = candle.data(); if (!data || data.length < Math.max(5,n)) { nbMaxSeries.setData([]); nbMinSeries.setData([]); return; }
      if (nbToggleEl && !nbToggleEl.checked){ nbMaxSeries.setData([]); nbMinSeries.setData([]); if (sNbMax) sNbMax.textContent='-'; if (sNbMin) sNbMin.textContent='-'; return; }
      const outMax=[]; const outMin=[]; const outWave=[];
      for (let i=n-1;i<data.length;i++){
        const win = data.slice(i-n+1, i+1);
        let highs, lows, closes;
        if (nbFromEmaEl && nbFromEmaEl.checked){
          const period = parseInt(nbEmaPeriodEl?.value||'10',10);
          const emaVals = ema(data.slice(0,i+1).map(d=>d.close), period);
          const emaWin = emaVals.slice(-win.length);
          highs = emaWin; lows = emaWin; closes = emaWin.map(v=>({value:v}));
          // for hi/lo span, use small buffer around EMA within window
          const hiVal = Math.max(...emaWin); const loVal = Math.min(...emaWin);
          // override below using computed hi/lo
          const hiArr = Array(win.length).fill(hiVal); const loArr = Array(win.length).fill(loVal);
          highs = hiArr; lows = loArr;
        } else {
          highs = win.map(d=>d.high); lows = win.map(d=>d.low); closes = win.map(d=>d.close);
        }
        const hi = Math.max(...highs); const lo = Math.min(...lows); const span = Math.max(hi-lo, 1e-9);
        const closeArr = (nbFromEmaEl && nbFromEmaEl.checked) ? highs.map((_,idx)=> (win[idx]?.close ?? highs[idx])) : closes;
        const changes = [];
        for (let k=1;k<closeArr.length;k++){ const prev=Number(closeArr[k-1]); const cur=Number(closeArr[k]); changes.push(((cur-prev) / (prev||1)) * 100); }
        if (changes.length < 2) continue;
        const scoreMax = clamp(BIT_MAX_NB(changes), 0, 100);
        const scoreMin = clamp(BIT_MIN_NB(changes), 0, 100);
        const priceMax = lo + span * (scoreMax/100);
        const priceMin = lo + span * (scoreMin/100);
        const t = data[i].time;
        const ratio = (scoreMax + scoreMin) > 0 ? (scoreMax / (scoreMax + scoreMin)) : 0.5;
        const waveVal = lo + span * ratio;
        outMax.push({ time:t, value: priceMax });
        outMin.push({ time:t, value: priceMin });
        outWave.push({ time:t, value: waveVal });
      }
      nbMaxSeries.setData([]); // hide standalone bands when wave is enabled
      nbMinSeries.setData([]);
      // Simulated wave using baseline around dynamic middle
      const lastWin = data.slice(Math.max(0, data.length - n), data.length);
      if (lastWin.length){
        const mid = (Math.max(...lastWin.map(d=>d.high)) + Math.min(...lastWin.map(d=>d.low))) / 2;
        nbWaveSeries && nbWaveSeries.applyOptions({ baseValue: { type: 'price', price: mid } });
        nbWaveSeries && nbWaveSeries.setData(outWave);
        const mxL = (outMax[outMax.length-1]?.value ?? 0);
        const mnL = (outMin[outMin.length-1]?.value ?? 0);
        const hiL = Math.max(mxL, mnL);
        const loL = Math.min(mxL, mnL);
        const denomL = (hiL - loL) || 1;
        let rLast = ((outWave[outWave.length-1]?.value ?? loL) - loL) / denomL;
        rLast = clamp(rLast, 0, 1);
        uiLog('NB 업데이트', `윈도우=${n}, r(마지막)=${(rLast||0).toFixed(3)}`);
        // Backfill signals with hysteresis: only BUY in BLUE zone, only SELL in ORANGE zone
        nbMarkers = [];
        window.lastNbSignals = [];
        const rArr = outWave.map((w, i)=>{
          const mx = outMax[Math.min(i, outMax.length-1)].value;
          const mn = outMin[Math.min(i, outMin.length-1)].value;
          const hi = Math.max(mx, mn);
          const lo = Math.min(mx, mn);
          const denom = (hi - lo);
          const rRaw = denom !== 0 ? (w.value - lo) / denom : 0.5;
          return clamp(rRaw, 0, 1);
        });
        const HIGH = 0.55, LOW = 0.45; // hysteresis to avoid chattering
        let zone = null; // 'BLUE'|'ORANGE'
        for (let i=0;i<outWave.length;i++){
          const r = rArr[i] ?? 0.5;
          const tm = outWave[i].time;
          // EMA filter: require EMA fast>slow for BUY, < for SELL
          let emaOkBuy = true, emaOkSell = true;
          if (emaFilterEl && emaFilterEl.checked){
            const data = candle.data();
            if (data && data.length>i){
              const closes = data.slice(0, i+1).map(d=>d.close);
              const ef = Number(emaFastEl?.value||10); const es = Number(emaSlowEl?.value||30);
              const emaFastArr = ema(closes, ef); const emaSlowArr = ema(closes, es);
              const efv = emaFastArr[emaFastArr.length-1]; const esv = emaSlowArr[emaSlowArr.length-1];
              emaOkBuy = (efv >= esv); emaOkSell = (efv <= esv);
            }
          }
          // decide zone using hysteresis
          if (zone === null){ zone = (r >= 0.5) ? 'ORANGE' : 'BLUE'; }
          if (zone === 'BLUE' && r >= HIGH && emaOkSell){
            zone = 'ORANGE';
            pushNBSignal(tm, 'SELL');
            try{ window.lastNbSignals.push({ time: tm, side: 'SELL' }); }catch(_){ }
            uiLog('SELL 신호', `구간전환: BLUE→ORANGE, r=${r.toFixed(3)} (상단 우세 구간으로 전환되어 매도)`);
          } else if (zone === 'ORANGE' && r <= LOW && emaOkBuy){
            zone = 'BLUE';
            pushNBSignal(tm, 'BUY');
            try{ window.lastNbSignals.push({ time: tm, side: 'BUY' }); }catch(_){ }
            uiLog('BUY 신호', `구간전환: ORANGE→BLUE, r=${r.toFixed(3)} (하단 우세 구간으로 전환되어 매수)`);
          }
        }
        // update live PnL display
        if (sEntry) sEntry.textContent = liveEntry? liveEntry.toLocaleString(): '-';
        if (sPnl) sPnl.textContent = livePnl.toLocaleString();
      }
      nbMaxOutline.setData(outMax);
      nbMinOutline.setData(outMin);
      if (outMax.length){ if (sNbMax) sNbMax.textContent = Number(outMax[outMax.length-1].value).toLocaleString(); }
      if (outMin.length){ if (sNbMin) sNbMin.textContent = Number(outMin[outMin.length-1].value).toLocaleString(); }
      if (sNbState && outMax.length && outMin.length){
        const mx = outMax[outMax.length-1].value;
        const mn = outMin[outMin.length-1].value;
        const hi = Math.max(mx, mn);
        const lo = Math.min(mx, mn);
        const crossed = mn > mx;
        sNbState.textContent = crossed
          ? `Zone crossover (Min>Max): Hi ${hi.toLocaleString()} / Lo ${lo.toLocaleString()}`
          : `Hi/Lo: ${hi.toLocaleString()} / ${lo.toLocaleString()}`;
        sNbState.className = crossed ? 'badge bg-info' : 'badge bg-secondary';
      }
      // labeled price lines disabled in wave-only mode
    }catch(e){ /* ignore */ }
  }

  // -------- Forecast (gray dashed) ---------
  const forecastSeries = chart.addLineSeries({ color:'rgba(200,200,200,0.95)', lineStyle: 2, lineWidth: 3 });
  function updateForecast(){
    try{
      if (!fcToggleEl || !fcToggleEl.checked){ forecastSeries.setData([]); return; }
      const w = parseInt((fcWindowEl && fcWindowEl.value) || '120', 10);
      const h = parseInt((fcHorizonEl && fcHorizonEl.value) || '30', 10);
      const data = candle.data(); if (!data || data.length < w+2){ forecastSeries.setData([]); return; }
      const win = data.slice(-w);
      const xs = win.map((_,i)=>i);
      const ys = win.map(p=>p.close ?? p.value ?? p.open ?? p.high ?? p.low);
      // Quadratic regression y = a2*x^2 + a1*x + a0 (captures curvature)
      const n = xs.length;
      let s1=0,s2=0,s3=0,s4=0, sy=0, sxy=0, sx2y=0;
      for (let i=0;i<n;i++){
        const x=xs[i]; const x2=x*x; const x3=x2*x; const x4=x2*x2; const y=ys[i];
        s1 += x; s2 += x2; s3 += x3; s4 += x4; sy += y; sxy += x*y; sx2y += x2*y;
      }
      // Solve normal equations
      // | n   s1   s2 | |a0|   | sy  |
      // | s1  s2   s3 |*|a1| = | sxy |
      // | s2  s3   s4 | |a2|   | sx2y|
      function det3(a,b,c,d,e,f,g,h,i){ return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g); }
      const D  = det3(n, s1, s2,  s1, s2, s3,  s2, s3, s4) || 1;
      const D0 = det3(sy, s1, s2,  sxy, s2, s3,  sx2y, s3, s4);
      const D1 = det3(n, sy, s2,  s1, sxy, s3,  s2, sx2y, s4);
      const D2 = det3(n, s1, sy,  s1, s2, sxy,  s2, s3, sx2y);
      const a0 = D0/D, a1 = D1/D, a2 = D2/D;
      const startT = win[0].time;
      const step = (win[win.length-1].time - startT) / (win.length-1 || 1);
      const proj = [];
      const lo = Math.min(...win.map(p=>p.low ?? p.value ?? p.close));
      const hi = Math.max(...win.map(p=>p.high ?? p.value ?? p.close));
      const span = Math.max(hi-lo, 1e-9);
      for (let i=0;i<w+h;i++){
        const t = startT + i*step;
        let v = a2*i*i + a1*i + a0;
        // clamp to reasonable band to avoid explosions
        const minV = lo - 0.25*span, maxV = hi + 0.25*span;
        if (v < minV) v = minV; if (v > maxV) v = maxV;
        proj.push({ time: Math.round(t), value: v });
      }
      forecastSeries.setData(proj);
    }catch(_){ forecastSeries.setData([]); }
  }

  function ema(values, period){
    if (!values.length) return [];
    const k = 2/(period+1); const out=[]; let prev = values[0];
    for (let i=0;i<values.length;i++){ const v=(i? values[i]*k + prev*(1-k) : values[0]); out.push(v); prev=v; }
    return out;
  }
  const msToSec = (ms)=> Math.floor(ms/1000);
  function bucketTs(tsMs, interval){
    if (interval.startsWith('minute')){ const m = parseInt(interval.replace('minute',''),10)||1; return Math.floor(tsMs/(m*60*1000))*(m*60*1000); }
    if (interval==='minute60'){ return Math.floor(tsMs/(60*60*1000))*(60*60*1000); }
    if (interval==='day'){ const d=new Date(tsMs); d.setHours(0,0,0,0); return d.getTime(); }
    return tsMs;
  }
  async function retrainLatest(){
    try{
      // Use current UI options to retrain briefly with recent data
      const interval = getInterval();
      const window = parseInt((nbWindowEl && nbWindowEl.value) || '50', 10);
      const ema_fast = parseInt((emaFastEl && emaFastEl.value) || '10', 10);
      const ema_slow = parseInt((emaSlowEl && emaSlowEl.value) || '30', 10);
      const payload = { window, ema_fast, ema_slow, horizon: 5, tau: 0.002, count: Math.max(600, window*12), interval };
      const t = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const pred = await fetchJsonStrict('/api/ml/predict');
      if (mlCountEl && t && t.ok) mlCountEl.textContent = `(train# ${t.train_count||0})`;
      if (pred && pred.ok){ uiLog('ML Auto 예측', `action=${pred.action}, pred=${pred.pred}`); }
    }catch(_){ }
  }

  let baseMarkers = [];
  let nbMarkers = [];
  // NB wave based lightweight signals on the UI
  let nbPosition = 'FLAT';
  let nbPeakRatio = 0; // highest ratio seen while LONG
  const NB_UP_TH = 0.7; // buy threshold
  const NB_DN_TH = 0.3; // sell floor
  function pushNBSignal(timeSec, side){
    const isBuy = side==='BUY';
    nbMarkers.push({ time: timeSec, position: isBuy?'belowBar':'aboveBar', color: isBuy?'#ffd166':'#00d1ff', shape:'circle', text:`NB ${side}` });
    if (nbMarkers.length>500) nbMarkers = nbMarkers.slice(-500);
    candle.setMarkers([...baseMarkers, ...nbMarkers]);
  }
  function pushOrderMarker(o, interval){
    if (!o||!o.ts) return;
    const sideStr = String(o.side||'').toUpperCase();
    const key = `${Number(o.ts)||0}|${sideStr}|${Math.round(Number(o.price||0))}|${o.paper?1:0}`;
    if (orderKeys.has(key)){
      return;
    }
    orderKeys.add(key);
    if (orderKeys.size>2000){
      // prune oldest by reconstructing from current log length
      try{ orderKeys = new Set(Array.from(orderKeys).slice(-1500)); }catch(_){ }
    }
    const sec = msToSec(bucketTs(Number(o.ts), interval||getInterval()));
    const isBuy = sideStr==='BUY';
    baseMarkers.push({
      time: sec,
      position: isBuy ? 'belowBar' : 'aboveBar',
      color: isBuy ? '#0ecb81' : '#f6465d',
      shape: isBuy ? 'arrowUp' : 'arrowDown',
      text: `${isBuy?'B':'S'} @${Number(o.price||0).toLocaleString()} ${o.size? '('+Number(o.size).toFixed(6)+')':''}`,
    });
    if (baseMarkers.length>500) baseMarkers = baseMarkers.slice(-500);
    candle.setMarkers([...baseMarkers, ...nbMarkers]);
    // Append to bottom order log
    try{
      if (orderLog){
        const ts = new Date(Number(o.ts)).toLocaleString();
        const line = `[${ts}] ${isBuy? 'BUY':'SELL'} @${Number(o.price||0).toLocaleString()} ${o.size? '('+Number(o.size).toFixed(6)+')':''} ${o.paper? '[PAPER]':''}`;
        const div = document.createElement('div');
        div.textContent = line;
        orderLog.prepend(div);
        // keep last 200
        while (orderLog.childElementCount>200){ orderLog.removeChild(orderLog.lastElementChild); }
      }
    }catch(_){ }
  }

  function seed(interval){
    fetch(`${base}/api/ohlcv?interval=${interval}&count=300`).then(r=>r.json()).then(res=>{
      const rows=res.data||[];
      const cs = rows.map(r=>({ time: msToSec(r.time), open:r.open, high:r.high, low:r.low, close:r.close }));
      candle.setData(cs);
      const closes = rows.map(r=>r.close); const times = rows.map(r=>msToSec(r.time));
      const ef = Number(emaFastEl?.value||10), es = Number(emaSlowEl?.value||30);
      emaF.setData(ema(closes,ef).map((y,i)=>({ time: times[i], value:y })));
      emaS.setData(ema(closes,es).map((y,i)=>({ time: times[i], value:y })));
      // SMA
      function sma(arr, n){ const out=[]; let sum=0; for(let i=0;i<arr.length;i++){ sum+=arr[i]; if(i>=n) sum-=arr[i-n]; out.push(i>=n-1? sum/n : arr[i]); } return out; }
      const sma50 = sma(closes, Number(sma50El?.value||50)).map((v,i)=>({ time: times[i], value:v }));
      const sma100 = sma(closes, Number(sma100El?.value||100)).map((v,i)=>({ time: times[i], value:v }));
      const sma200 = sma(closes, Number(sma200El?.value||200)).map((v,i)=>({ time: times[i], value:v }));
      if (showSMAEl && showSMAEl.checked){ sma50Series.setData(sma50); sma100Series.setData(sma100); sma200Series.setData(sma200); }
      else { sma50Series.setData([]); sma100Series.setData([]); sma200Series.setData([]); }
      // EMA 9/12/26
      const e9 = ema(closes,9).map((v,i)=>({ time: times[i], value:v }));
      const e12 = ema(closes,12).map((v,i)=>({ time: times[i], value:v }));
      const e26 = ema(closes,26).map((v,i)=>({ time: times[i], value:v }));
      if (showEMA9El && showEMA9El.checked){ ema9Series.setData(e9); ema12Series.setData(e12); ema26Series.setData(e26); }
      else { ema9Series.setData([]); ema12Series.setData([]); ema26Series.setData([]); }
      // Ichimoku Tenkan/Kijun (단순 고저 평균)
      function highLowAvg(rowsArr, period){ const out=[]; for(let i=0;i<rowsArr.length;i++){ const start=Math.max(0,i-period+1); let hi=-Infinity, lo=Infinity; for(let j=start;j<=i;j++){ hi=Math.max(hi, rowsArr[j].high); lo=Math.min(lo, rowsArr[j].low); } out.push((hi+lo)/2); } return out; }
      try {
        const tenkanN = Number(ichiTenkanEl?.value||9), kijunN = Number(ichiKijunEl?.value||26);
        const tenkan = highLowAvg(rows, tenkanN).map((v,i)=>({ time: times[i], value:v }));
        const kijun = highLowAvg(rows, kijunN).map((v,i)=>({ time: times[i], value:v }));
        if (showIchimokuEl && showIchimokuEl.checked){ ichiTenkanSeries.setData(tenkan); ichiKijunSeries.setData(kijun); }
        else { ichiTenkanSeries.setData([]); ichiKijunSeries.setData([]); }
      } catch(_){ ichiTenkanSeries.setData([]); ichiKijunSeries.setData([]); }
      updateNB();
      updateForecast();
    }).then(()=>{
      // load existing orders (only when toggle on)
      if (ordersToggle && !ordersToggle.checked){ markers=[]; candle.setMarkers([...nbMarkers]); return; }
      return fetch(`${base}/api/orders`).then(r=>r.json()).then(or=>{
        markers=[]; (or.data||[]).forEach(o=>pushOrderMarker(o, interval));
      });
    }).catch(()=>{});
  }

  // Restore saved options
  (function restore(){
    const o = readOpts();
    if (tfEl && o.timeframe) tfEl.value = o.timeframe;
    if (paperEl && typeof o.paper !== 'undefined') paperEl.value = String(o.paper);
    if (orderEl && o.order_krw) orderEl.value = o.order_krw;
    if (emaFastEl && o.ema_fast) emaFastEl.value = o.ema_fast;
    if (emaSlowEl && o.ema_slow) emaSlowEl.value = o.ema_slow;
    if (nbWindowEl && o.nb_window) nbWindowEl.value = o.nb_window;
    if (nbToggleEl && typeof o.nb_show !== 'undefined') nbToggleEl.checked = !!o.nb_show;
    if (nbBuyThEl && o.nb_buy_th) nbBuyThEl.value = o.nb_buy_th;
    if (nbSellThEl && o.nb_sell_th) nbSellThEl.value = o.nb_sell_th;
    if (nbDebounceEl && o.nb_debounce) nbDebounceEl.value = o.nb_debounce;
    if (emaFilterEl && typeof o.ema_filter !== 'undefined') emaFilterEl.checked = !!o.ema_filter;
    if (nbFromEmaEl && typeof o.nb_from_ema !== 'undefined') nbFromEmaEl.checked = !!o.nb_from_ema;
    if (nbEmaPeriodEl && o.nb_ema_period) nbEmaPeriodEl.value = o.nb_ema_period;
    if (typeof fcWindowEl !== 'undefined' && fcWindowEl && o.fc_window) fcWindowEl.value = o.fc_window;
    if (typeof fcHorizonEl !== 'undefined' && fcHorizonEl && o.fc_horizon) fcHorizonEl.value = o.fc_horizon;
    if (typeof fcToggleEl !== 'undefined' && fcToggleEl && typeof o.fc_show !== 'undefined') fcToggleEl.checked = !!o.fc_show;
    if (typeof ordersToggle !== 'undefined' && ordersToggle && typeof o.show_orders !== 'undefined') ordersToggle.checked = !!o.show_orders;
    if (typeof autoBtToggle !== 'undefined' && autoBtToggle && typeof o.auto_bt !== 'undefined') autoBtToggle.checked = !!o.auto_bt;
    if (typeof autoBtSecEl !== 'undefined' && autoBtSecEl && o.auto_bt_sec) autoBtSecEl.value = o.auto_bt_sec;
    if (showSMAEl && typeof o.show_sma !== 'undefined') showSMAEl.checked = !!o.show_sma;
    if (sma50El && o.sma50) sma50El.value = o.sma50;
    if (sma100El && o.sma100) sma100El.value = o.sma100;
    if (sma200El && o.sma200) sma200El.value = o.sma200;
    if (showEMA9El && typeof o.show_ema9 !== 'undefined') showEMA9El.checked = !!o.show_ema9;
    if (showIchimokuEl && typeof o.show_ichimoku !== 'undefined') showIchimokuEl.checked = !!o.show_ichimoku;
    if (ichiTenkanEl && o.ichi_tenkan) ichiTenkanEl.value = o.ichi_tenkan;
    if (ichiKijunEl && o.ichi_kijun) ichiKijunEl.value = o.ichi_kijun;
    if (typeof trainCountEl !== 'undefined' && trainCountEl && o.train_count) trainCountEl.value = o.train_count;
    if (typeof trainSegEl !== 'undefined' && trainSegEl && o.train_seg) trainSegEl.value = o.train_seg;
    // push restored config to server and persist again
    pushConfig().catch(()=>{});
    // fetch persisted NB params from server and apply
    fetch('/api/nb/params').then(r=>r.json()).then(j=>{
      if (j && j.ok && j.params){
        if (nbBuyThEl && j.params.buy) nbBuyThEl.value = String(j.params.buy);
        if (nbSellThEl && j.params.sell) nbSellThEl.value = String(j.params.sell);
        if (nbWindowEl && j.params.window) nbWindowEl.value = String(j.params.window);
        updateNB();
      }
    }).catch(()=>{});
    // re-arm auto BT if enabled
    if (autoBtToggle && autoBtToggle.checked){ autoBtToggle.dispatchEvent(new Event('change')); }
  })();

  seed(getInterval());
  if (tfEl) tfEl.addEventListener('change', ()=>{ seed(getInterval()); pushConfig(); });
  if (nbWindowEl) nbWindowEl.addEventListener('change', ()=>{ updateNB(); updateForecast(); saveOpts(); });
  if (nbToggleEl) nbToggleEl.addEventListener('change', ()=>{ updateNB(); updateForecast(); saveOpts(); });
  if (nbBuyThEl) nbBuyThEl.addEventListener('change', saveOpts);
  if (nbSellThEl) nbSellThEl.addEventListener('change', saveOpts);
  if (nbDebounceEl) nbDebounceEl.addEventListener('change', saveOpts);
  if (emaFilterEl) emaFilterEl.addEventListener('change', ()=>{ saveOpts(); updateNB(); });
  if (nbFromEmaEl) nbFromEmaEl.addEventListener('change', ()=>{ saveOpts(); updateNB(); });
  if (nbEmaPeriodEl) nbEmaPeriodEl.addEventListener('change', ()=>{ saveOpts(); updateNB(); });
  if (typeof fcWindowEl !== 'undefined' && fcWindowEl) fcWindowEl.addEventListener('change', ()=>{ updateForecast(); saveOpts(); });
  if (typeof fcHorizonEl !== 'undefined' && fcHorizonEl) fcHorizonEl.addEventListener('change', ()=>{ updateForecast(); saveOpts(); });
  if (typeof fcToggleEl !== 'undefined' && fcToggleEl) fcToggleEl.addEventListener('change', ()=>{ updateForecast(); saveOpts(); });
  if (ordersToggle) ordersToggle.addEventListener('change', saveOpts);
  if (autoBtSecEl) autoBtSecEl.addEventListener('change', saveOpts);
  if (showSMAEl) showSMAEl.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (sma50El) sma50El.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (sma100El) sma100El.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (sma200El) sma200El.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (showEMA9El) showEMA9El.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (showIchimokuEl) showIchimokuEl.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (ichiTenkanEl) ichiTenkanEl.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (ichiKijunEl) ichiKijunEl.addEventListener('change', ()=>{ saveOpts(); seed(getInterval()); });
  if (trainCountEl) trainCountEl.addEventListener('change', saveOpts);
  if (trainSegEl) trainSegEl.addEventListener('change', saveOpts);
  window.addEventListener('beforeunload', saveOpts);

  // Bind config change handlers
  [paperEl, orderEl, emaFastEl, emaSlowEl].forEach(el=>{
    if (!el) return;
    el.addEventListener('change', ()=>{ pushConfig(); });
  });

  // Start/Stop bot
  if (startBtn) startBtn.addEventListener('click', async ()=>{
    await pushConfig();
    await postJson('/api/bot/start', {});
    if (sBot) sBot.textContent = 'running';
  });
  if (stopBtn) stopBtn.addEventListener('click', async ()=>{
    await postJson('/api/bot/stop', {});
    if (sBot) sBot.textContent = 'stopped';
  });

  try{
    const es = new EventSource(`/api/stream`);
    es.onmessage = (e)=>{
      try{
        const j = JSON.parse(e.data);
        const meta = document.getElementById('meta'); if (meta) meta.textContent = `${j.market} ${j.candle} | ${j.signal} | EMA ${j.ema_fast}/${j.ema_slow}`;
        if (sTicker) sTicker.textContent = j.market || '-';
        if (sPrice) sPrice.textContent = (j.price||0).toLocaleString();
        if (sSignal){ sSignal.textContent=j.signal; sSignal.className = (j.signal==='BUY'?'buy':'sell'); }
        const itv = getInterval(); const bMs = bucketTs(j.ts, itv); const bSec = msToSec(bMs);
        const data = candle.data(); const last = data[data.length-1];
        if (last && last.time === bSec){
          candle.update({ ...last, close:j.price, high:Math.max(last.high,j.price), low:Math.min(last.low,j.price) });
        } else {
          const prev = last ? last.close : j.price;
          candle.update({ time:bSec, open:prev, high:j.price, low:j.price, close:j.price });
        }
        const closes = candle.data().map(d=>d.close); const times = candle.data().map(d=>d.time);
        emaF.setData(ema(closes, j.ema_fast).map((y,i)=>({ time: times[i], value:y })));
        emaS.setData(ema(closes, j.ema_slow).map((y,i)=>({ time: times[i], value:y })));
        if (j.order){
          pushOrderMarker(j.order, itv);
          try{
            const side = String(j.order.side||'').toUpperCase();
            const op = Number(j.order.price||0);
            if (side === 'BUY' && op>0){ liveLastBuyPrice = op; }
            else if (side === 'SELL' && op>0 && liveLastBuyPrice>0){
              const pnl = op - liveLastBuyPrice;
              const wr = pnl>0 ? 100 : 0;
              pushWinItem({ ts: Number(j.order.ts)||Date.now(), pnl, winRate: wr });
              updateTopPnlSlider(pnl, wr);
              liveLastBuyPrice = 0;
            }
          }catch(_){ }
        }
        updateNB();
        // (removed) incremental retrain on bar. ML Auto uses random trainer on timer.
      }catch(_){ }
    };
  }catch(_){ }
  // ML Auto: automatic random training on interval
  if (mlAutoToggle) mlAutoToggle.addEventListener('change', ()=>{
    try{ if (mlAutoTimer){ clearInterval(mlAutoTimer); mlAutoTimer=null; } }catch(_){ }
    if (mlAutoToggle.checked){
      const run = async ()=>{
        try{
          const minsArr = [1,3,5,10,15,30,60];
          const mins = minsArr[Math.floor(Math.random()*minsArr.length)];
          const interval = mins===60 ? 'minute60' : `minute${mins}`;
          const window = Math.floor(20 + Math.random()*100);
          const ema_fast = Math.floor(5 + Math.random()*20);
          const ema_slow = Math.max(ema_fast+5, Math.floor(20 + Math.random()*60));
          if (tfEl){ tfEl.value = interval; tfEl.dispatchEvent(new Event('change')); }
          if (emaFastEl){ emaFastEl.value = String(ema_fast); emaFastEl.dispatchEvent(new Event('change')); }
          if (emaSlowEl){ emaSlowEl.value = String(ema_slow); emaSlowEl.dispatchEvent(new Event('change')); }
          if (typeof nbWindowEl !== 'undefined' && nbWindowEl){ nbWindowEl.value = String(window); nbWindowEl.dispatchEvent(new Event('change')); }
          await sleep(400);
          const payload = { window, ema_fast, ema_slow, horizon: 5, tau: 0.002, count: 1200, interval };
          const j = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
          if (!(j && j.ok)) { uiLog('ML Auto 랜덤 학습 실패', JSON.stringify(j)); return; }
          if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
          await backtestAfterReady(6000);
          await sleep(800);
          await backtestAfterReady(3000);
          const pred = await fetchJsonStrict('/api/ml/predict');
          if (pred && pred.ok){
            uiLog('ML Auto 랜덤 예측', `action=${pred.action}, pred=${pred.pred}`);
            if (mlCountEl) mlCountEl.textContent = `(train# ${pred.train_count||0})`;
            updateModelInsight(pred);
          }
        }catch(_){ }
      };
      const sec = Math.max(5, parseInt(autoBtSecEl?.value||'15',10));
      uiLog('ML Auto 랜덤 켜짐', `주기=${sec}s`);
      // 자동 매매: ML Auto 시작 시 봇도 자동 시작
      try{ postJson('/api/bot/start', {}).then(()=>{ if (sBot) sBot.textContent='running'; }).catch(()=>{}); }catch(_){ }
      run();
      mlAutoTimer = setInterval(run, sec*1000);
    } else {
      uiLog('ML Auto 랜덤 꺼짐');
    }
  });

  // Initial status fetch
  fetch(`/api/bot/status`).then(r=>r.json()).then(st=>{
    try{
      if (sBot) sBot.textContent = st.running ? 'running' : 'stopped';
      if (sEma && st.config) sEma.textContent = `${st.config.ema_fast}/${st.config.ema_slow}`;
      if (sInterval && st.config) sInterval.textContent = st.config.candle;
      if (sTicker && st.config) sTicker.textContent = st.config.market;
    }catch(_){ }
  }).catch((e)=>{ console.error('status fail', e); });

  // Screenshot -> clipboard (fallback download)
  if (shotBtn) shotBtn.addEventListener('click', async ()=>{
    try{
      const card = container.closest('.card') || container;
      const canvas = await html2canvas(card, { backgroundColor: '#0b1220', scale: 2, useCORS: true });
      const tryClipboard = async (blob)=>{
        if (navigator.clipboard && window.ClipboardItem){
          try {
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
            console.log('Screenshot copied to clipboard');
            return true;
          } catch (e) {
            console.warn('Clipboard write failed', e);
          }
        }
        return false;
      };
      if (canvas.toBlob){
        canvas.toBlob(async (blob)=>{
          const ok = await tryClipboard(blob);
          if (!ok){
            const ts = new Date().toISOString().replace(/[:.]/g,'-');
            const link = document.createElement('a');
            link.download = `8bit-chart-${ts}.png`;
            link.href = URL.createObjectURL(blob);
            link.click();
          }
        }, 'image/png');
      } else {
        const dataUrl = canvas.toDataURL('image/png');
        const blob = await (await fetch(dataUrl)).blob();
        const ok = await tryClipboard(blob);
        if (!ok){
          const ts = new Date().toISOString().replace(/[:.]/g,'-');
          const link = document.createElement('a');
          link.download = `8bit-chart-${ts}.png`;
          link.href = dataUrl;
          link.click();
        }
      }
    }catch(e){ console.error('screenshot failed', e); }
  });

  if (mlMetricsBtn) mlMetricsBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/ml/metrics');
      if (!(j && j.ok)){ uiLog('ML Metrics 실패', JSON.stringify(j)); return; }
      const cv = j.metrics?.cv || {}; const inr = j.metrics?.in_sample || {};
      const acc = inr.report?.accuracy ? (inr.report.accuracy*100).toFixed(1)+'%' : '-';
      const f1 = cv.f1_macro ? (cv.f1_macro*100).toFixed(1)+'%' : '-';
      const pnl = (cv.pnl_sum||0).toLocaleString(undefined,{maximumFractionDigits:0});
      const params = j.params || j.metrics?.params || {};
      const trainedAt = j.trained_at ? new Date(j.trained_at).toLocaleString() : '-';
      const html = `
        <div class="card border-secondary rounded-3 p-2 mt-2">
          <div><strong>ML Metrics</strong> <span class="text-muted">(${j.interval})</span></div>
          <div class="kv"><span>Accuracy(in-sample)</span><span>${acc}</span></div>
          <div class="kv"><span>F1-macro(CV)</span><span>${f1}</span></div>
          <div class="kv"><span>CV PnL Sum</span><span>${pnl}</span></div>
          <div class="kv"><span>Params</span><span>${JSON.stringify(params)}</span></div>
          <div class="kv"><span>Trained At</span><span>${trainedAt}</span></div>
        </div>`;
      if (mlMetricsBox) mlMetricsBox.innerHTML = html;
      uiLog('ML Metrics', `acc=${acc}, f1=${f1}, pnl=${pnl}`);
    }catch(e){ uiLog('ML Metrics 에러', String(e)); }
  });

  // -------- Backtest using NB signals on current chart data --------
  function runBacktest(){
    try{
      const data = candle.data();
      if (!data || data.length < 50) return;
      let raw = (Array.isArray(nbMarkers)? nbMarkers: []).slice().sort((a,b)=>a.time-b.time);
      if (!raw.length){
        // Fallback: EMA 크로스 기반 신호 생성 (NB 신호 없을 때)
        try{
          const closes = data.map(d=>d.close);
          const ef = Number(emaFastEl?.value||10); const es = Number(emaSlowEl?.value||30);
          const efArr = ema(closes, ef); const esArr = ema(closes, es);
          const sigs=[];
          let prev = efArr[0] - esArr[0];
          for (let i=1;i<closes.length;i++){
            const diff = (efArr[i] - esArr[i]);
            if (prev<=0 && diff>0) sigs.push({ time: data[i].time, text: 'NB BUY' });
            else if (prev>=0 && diff<0) sigs.push({ time: data[i].time, text: 'NB SELL' });
            prev = diff;
          }
          raw = sigs;
          if (!raw.length){ uiLog('백테스트 취소', '신호 없음'); return; }
          uiLog('NB 신호 없음 → EMA 크로스 백테스트로 대체');
        }catch(_){ uiLog('백테스트 취소', '신호 없음'); return; }
      }
      // 1) 중복/연속 동일 신호 제거하여 BUY/SELL 교대로 정규화
      const norm=[]; let lastSide=null;
      for(const m of raw){ const side = m.text.includes('BUY')?'BUY':(m.text.includes('SELL')?'SELL':null); if(!side) continue; if(side===lastSide) continue; norm.push({time:m.time, side}); lastSide=side; }
      // 선행 SELL 제거
      while (norm.length && norm[0].side==='SELL') norm.shift();
      if (norm.length<2){ uiLog('백테스트 취소', '유효 신호 부족'); return; }
      // 2) 페어링하여 수익/승률 계산
      let trades=0, wins=0; let pnl=0; let peak=0, dd=0; let entry=0;
      for (let i=0;i<norm.length-1;i+=2){
        const buy = norm[i]; const sell = norm[i+1]; if(!buy||!sell) break;
        const buyBar = data.find(d=>d.time===buy.time) || data.reduce((p,c)=> Math.abs(c.time-buy.time)<Math.abs((p?.time||0)-buy.time)? c : p, null);
        const sellBar = data.find(d=>d.time===sell.time) || data.reduce((p,c)=> Math.abs(c.time-sell.time)<Math.abs((p?.time||0)-sell.time)? c : p, null);
        if (!buyBar || !sellBar) continue;
        trades++;
        entry = buyBar.close;
        const ret = (sellBar.close - entry);
        pnl += ret;
        if (ret>0) wins++;
        peak = Math.max(peak, pnl);
        dd = Math.max(dd, peak - pnl);
      }
      const winRate = trades? (wins/trades*100):0;
      const sTrades = document.getElementById('bt_trades'); if (sTrades) sTrades.textContent = String(trades);
      const sPnl = document.getElementById('bt_pnl'); if (sPnl){ const sign = pnl>=0? '+' : '-'; sPnl.textContent = `${sign}${Math.abs(pnl).toLocaleString(undefined,{maximumFractionDigits:0})}`; }
      const sWin = document.getElementById('bt_win'); if (sWin){ const sign = pnl>=0? '+' : '-'; sWin.textContent = `${sign}${winRate.toFixed(1)}%`; }
      const sDd = document.getElementById('bt_dd'); if (sDd) sDd.textContent = dd.toLocaleString(undefined,{maximumFractionDigits:0});
      const wl = document.getElementById('bt_wl'); if (wl) wl.textContent = `${wins}/${Math.max(0,trades-wins)}`;
      uiLog('백테스트 완료', `거래수=${trades}, 승수=${wins}, 손익=${pnl.toFixed(0)}원, 승률=${winRate.toFixed(1)}%, 최대낙폭=${dd.toFixed(0)}원`);
      // push rolling item
      pushWinItem({ ts: Date.now(), pnl, winRate });
      // update top slider
      updateTopPnlSlider(pnl, winRate);
    }catch(_){ }
  }
  if (btBtn) btBtn.addEventListener('click', runBacktest);

  if (clearBtn) clearBtn.addEventListener('click', async ()=>{
    try{
      await fetch('/api/orders/clear', { method:'POST' });
      baseMarkers = []; candle.setMarkers([...baseMarkers, ...nbMarkers]);
      if (orderLog) orderLog.innerHTML='';
      orderKeys.clear();
    }catch(_){ }
  });

  if (ordersToggle) ordersToggle.addEventListener('change', ()=>{
    if (!ordersToggle.checked){ baseMarkers=[]; candle.setMarkers([...nbMarkers]); }
    else { seed(getInterval()); }
  });

  // Orders bottom log: clear & export
  if (orderClearBtn) orderClearBtn.addEventListener('click', async ()=>{
    try{ await fetch('/api/orders/clear', { method:'POST' }); if (orderLog) orderLog.innerHTML=''; orderKeys.clear(); }catch(_){ }
  });
  if (orderExportBtn) orderExportBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/orders');
      const rows = (j && j.data) ? j.data : [];
      const header = ['ts','side','price','size','paper','market'];
      const csv = [header.join(',')].concat(rows.map(r=>[
        r.ts, r.side, r.price, r.size, r.paper, r.market
      ].join(','))).join('\n');
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = `orders-${Date.now()}.csv`; a.click();
      setTimeout(()=>URL.revokeObjectURL(url), 1000);
    }catch(_){ }
  });

  if (optBtn) optBtn.addEventListener('click', ()=>{ optimizeNb(); });
  if (trainBtn) trainBtn.addEventListener('click', async ()=>{
    try{
      const payload = { count: parseInt(trainCountEl?.value||'1800',10), segments: parseInt(trainSegEl?.value||'3',10), window: parseInt(nbWindowEl?.value||'50',10), debounce: parseInt(nbDebounceEl?.value||'6',10), fee_bps: 10.0, interval: getInterval() };
      uiLog('훈련 시작', `자동 분할: ${payload.segments}구간, 캔들=${payload.interval}, 데이터=${payload.count}`);
      const r = await fetch('/api/nb/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const j = await r.json();
      if (j && j.ok){
        uiLog('훈련 완료', `선택 세그먼트=${j.chosen.segment}, PnL=${j.chosen.stats.pnl.toFixed(0)}, BUY=${j.chosen.best.buy}, SELL=${j.chosen.best.sell}`);
        if (nbBuyThEl) nbBuyThEl.value = String(j.chosen.best.buy);
        if (nbSellThEl) nbSellThEl.value = String(j.chosen.best.sell);
        updateNB();
      } else { uiLog('훈련 실패', JSON.stringify(j)); }
    }catch(e){ uiLog('훈련 에러', String(e)); }
  });
  if (autoBtToggle) autoBtToggle.addEventListener('change', ()=>{
    if (autoBtToggle.checked){
      const run = ()=>{ if (btBtn) btBtn.click(); };
      const sec = Math.max(10, parseInt(autoBtSecEl?.value||'60',10));
      run();
      autoBtTimer = setInterval(run, sec*1000);
      uiLog('자동 백테스트 시작', `주기=${sec}s`);
    } else {
      if (autoBtTimer) clearInterval(autoBtTimer); autoBtTimer=null;
      uiLog('자동 백테스트 중지');
    }
  });
  if (mlTrainBtn) mlTrainBtn.addEventListener('click', async ()=>{
    try{
      uiLog('ML 학습 시작', 'LightGBM/GBDT (sklearn) 베이스라인');
      const payload = { window: parseInt(nbWindowEl?.value||'50',10), ema_fast: parseInt(emaFastEl?.value||'10',10), ema_slow: parseInt(emaSlowEl?.value||'30',10), horizon: 5, tau: 0.002, count: 1800, interval: getInterval() };
      const j = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (j && j.ok){ uiLog('ML 학습 완료', `라벨 개수: BUY=${j.classes['1']}, HOLD=${j.classes['0']}, SELL=${j.classes['-1']}`); if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`; }
      else { uiLog('ML 학습 실패', JSON.stringify(j)); }
    }catch(e){ uiLog('ML 학습 에러', String(e)); }
  });
  if (mlPredictBtn) mlPredictBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/ml/predict');
      if (j && j.ok){
        uiLog('ML 예측', `action=${j.action}, pred=${j.pred}`);
        if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
        updateModelInsight(j);
      }
      else { uiLog('ML 예측 실패', JSON.stringify(j)); }
    }catch(e){ uiLog('ML 예측 에러', String(e)); }
  });
  if (mlRandomBtn) mlRandomBtn.addEventListener('click', async ()=>{
    try{
      const n = Math.max(1, parseInt(mlRandNEl?.value||'10',10));
      uiLog('ML 랜덤 학습 시작', `시도 횟수=${n}`);
      for (let i=0;i<n;i++){
        const mins = [1,3,5,10,15,30,60][Math.floor(Math.random()*7)];
        const interval = mins===60 ? 'minute60' : `minute${mins}`;
        const window = Math.floor(20 + Math.random()*100); // 20~120
        const ema_fast = Math.floor(5 + Math.random()*20); // 5~25
        const ema_slow = Math.max(ema_fast+5, Math.floor(20 + Math.random()*60));
        // Reflect random options on UI so 사용자가 볼 수 있게 함
        try{
          if (tfEl){ tfEl.value = interval; tfEl.dispatchEvent(new Event('change')); }
          if (emaFastEl){ emaFastEl.value = String(ema_fast); emaFastEl.dispatchEvent(new Event('change')); }
          if (emaSlowEl){ emaSlowEl.value = String(ema_slow); emaSlowEl.dispatchEvent(new Event('change')); }
          if (typeof nbWindowEl !== 'undefined' && nbWindowEl){ nbWindowEl.value = String(window); nbWindowEl.dispatchEvent(new Event('change')); }
          // 잠깐 대기하여 차트/지표 갱신이 화면에 반영되도록
          await sleep(400);
        }catch(_){ }
        const payload = { window, ema_fast, ema_slow, horizon: 5, tau: 0.002, count: 1200, interval };
        uiLog('ML 랜덤 학습', JSON.stringify(payload));
        const j = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        if (!(j && j.ok)) { uiLog('학습 실패, 시도 건너뜀', JSON.stringify(j)); continue; }
        if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
        // 각 랜덤 학습 시도 후: 차트/NB 신호 준비 대기 → 백테스트 실행
        try{
          // 여러 번 시도하여 비동기 지연을 흡수
          await backtestAfterReady(6000);
          await sleep(1200); await backtestAfterReady(3000);
        }catch(_){ }
      }
      const pred = await fetchJsonStrict('/api/ml/predict');
      if (pred && pred.ok){ uiLog('ML 예측(랜덤 후)', `action=${pred.action}, pred=${pred.pred}`); if (mlCountEl) mlCountEl.textContent = `(train# ${pred.train_count||0})`; }
      else { uiLog('ML 예측 실패(랜덤 후)', JSON.stringify(pred)); }
      // 마지막으로 한 번 더 백테스트 갱신
      try{
        await backtestAfterReady(4000);
        await sleep(1200); await backtestAfterReady(3000);
      }catch(_){ }
    }catch(e){ uiLog('ML 랜덤 에러', String(e)); }
  });
  if (loadBalBtn) loadBalBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/balance');
      const box = document.getElementById('balanceBox');
      if (!box) return;
      if (!j.ok){ box.textContent = `에러: ${j.error||'unknown'}`; return; }
      if (j.paper){ box.textContent = 'PAPER 모드 (실자산 없음)'; return; }
      const rows = (j.balances||[]);
      const lines = rows.map(b=>`${b.currency}: 수량=${b.balance} 잠김=${b.locked} 평균매수가=${b.avg_buy_price}`);
      box.textContent = lines.length? lines.join('\n') : '잔고가 없습니다';
    }catch(e){ const box = document.getElementById('balanceBox'); if (box) box.textContent = String(e); }
  });
  // --- Top assets auto loader ---
  async function refreshAssets(){
    try{
      const j = await fetchJsonStrict('/api/balance');
      if (!j.ok){ if (assetsMeta) assetsMeta.textContent = `(에러: ${j.error||'unknown'})`; return; }
      if (j.paper){ if (assetsMeta) assetsMeta.textContent = '(PAPER 모드)'; return; }
      const rows = (j.balances||[]);
      // show KRW first, then others sorted by balance desc
      const krw = rows.filter(b=>b.currency==='KRW');
      const rest = rows.filter(b=>b.currency!=='KRW').sort((a,b)=> (b.balance||0) - (a.balance||0));
      const all = [...krw, ...rest];
      // Stats cards
      const totalValue = all.reduce((s,b)=> s + Number(b.asset_value||0), 0);
      const krwVal = Number((krw[0]?.asset_value)||0);
      const sellables = rest.filter(b=> Number(b.asset_value||0) > 0).map(b=> b.currency).slice(0, 20);
      if (assetTotalEl) assetTotalEl.textContent = Math.round(totalValue).toLocaleString();
      if (assetBuyableEl) assetBuyableEl.textContent = Math.round(krwVal).toLocaleString();
      if (assetSellableEl) assetSellableEl.innerHTML = sellables.length? sellables.map(s=>`<span class='chip'>${s}</span>`).join(' ') : '<span class="chip">-</span>';

      // Bars by KRW value proportions (top 10 including KRW)
      if (assetsBars){
        assetsBars.innerHTML = '';
        const top = [{ currency:'KRW', asset_value: krwVal }, ...rest].filter(b=> (b.asset_value||0)>0).slice(0, 10);
        const sum = top.reduce((s,b)=> s + Number(b.asset_value||0), 0) || 1;
        top.forEach(b=>{
          const pct = Math.max(1, Math.round((Number(b.asset_value||0)/sum)*100));
          const row = document.createElement('div');
          row.className = 'asset-bar' + (b.currency==='KRW'?' krw':'');
          row.innerHTML = `<div class='top'><div class='label'>${b.currency}</div><div class='muted'>${Math.round(b.asset_value||0).toLocaleString()} KRW (${pct}%)</div></div>
            <div class='meter'><div class='fill' style='width:${pct}%;'></div></div>`;
          assetsBars.appendChild(row);
        });
      }
      if (assetsMeta) assetsMeta.textContent = `(${new Date().toLocaleTimeString()})`;
    }catch(e){ if (assetsBox) assetsBox.textContent = String(e); }
  }
  if (assetsRefresh) assetsRefresh.addEventListener('click', refreshAssets);
  if (assetsAutoToggle) assetsAutoToggle.addEventListener('change', ()=>{
    if (assetsAutoToggle.checked){
      refreshAssets();
      assetsTimer = setInterval(refreshAssets, 30*1000);
    } else {
      if (assetsTimer) clearInterval(assetsTimer), assetsTimer=null;
    }
  });
  // kick off initial load
  refreshAssets().catch(()=>{});
  if (assetsAutoToggle && assetsAutoToggle.checked){ assetsTimer = setInterval(refreshAssets, 30*1000); }
  if (logClearBtn) logClearBtn.addEventListener('click', ()=>{ if (logBox) logBox.textContent=''; });
})();


