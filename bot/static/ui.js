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
  const ordersToggle = null;
  const optBtn = document.getElementById('btnOptimize');
  const trainBtn = document.getElementById('btnTrain');
  const mlTrainBtn = document.getElementById('btnMlTrain');
  const mlPredictBtn = document.getElementById('btnMlPredict');
  const miZone = document.getElementById('miZone');
  const miText = document.getElementById('miText');

  function updateModelInsight(j){
    try{
      const ins = j && j.insight ? j.insight : {};
      try{ window.lastInsight = ins; }catch(_){ }
      // Show model's zone-aware intent summary
      try{
        const zact = j && j.zone_actions ? j.zone_actions : {};
        const badge = document.getElementById('miZone');
        if (badge){
          const hint = (zact.buy_in_blue ? 'BUY@BLUE' : (zact.sell_in_orange ? 'SELL@ORANGE' : '-'));
          badge.textContent = hint || String(ins.zone||'-');
          badge.className = 'badge bg-white text-dark';
        }
      }catch(_){ }
      if (miZone){ miZone.textContent = String(ins.zone||'-'); miZone.className = 'badge bg-white text-dark'; }
      // reflect current zone majority on Win% card header and background
      try{
        const winZoneNow = document.getElementById('winZoneNow');
        const winCard = document.getElementById('winCard');
        if (winZoneNow){ winZoneNow.textContent = String(ins.zone||'-'); winZoneNow.className = 'badge bg-white text-dark'; }
        if (winCard){
          winCard.classList.remove('win-card-blue','win-card-orange');
          const pb = Number(ins.pct_blue||0), po = Number(ins.pct_orange||0);
          if (po >= pb && po>0){ winCard.classList.add('win-card-orange'); }
          else if (pb > po && pb>0){ winCard.classList.add('win-card-blue'); }
        }
      }catch(_){ }
      if (miText){
        const blueAdj = (ins.pct_blue||0);
        const orangeAdj = (ins.pct_orange||0);
        const blueRaw = (ins.pct_blue_raw!=null? ins.pct_blue_raw : blueAdj);
        const orangeRaw = (ins.pct_orange_raw!=null? ins.pct_orange_raw : orangeAdj);
        let slopeLine = '';
        try{
          const st = j && j.steep ? j.steep : null;
          if (st && (st.blue_up_slope!=null || st.orange_down_slope!=null)){
            const up = st.blue_up_slope!=null ? Number(st.blue_up_slope*10000).toFixed(2) : '-';
            const dn = st.orange_down_slope!=null ? Number(st.orange_down_slope*10000).toFixed(2) : '-';
            slopeLine = ` | upSlope@BLUE=${up}bp/bar | downSlope@ORANGE=${dn}bp/bar`;
          }
        }catch(_){ }
        miText.innerHTML = `r=${(ins.r||0).toFixed(3)} | BLUE(raw)=${Number(blueRaw).toFixed(1)}% | ORANGE(raw)=${Number(orangeRaw).toFixed(1)}% | BLUE=${Number(blueAdj).toFixed(1)}% | ORANGE=${Number(orangeAdj).toFixed(1)}% | zone=${String(ins.zone||'-')} | conf=${(ins.zone_conf||0).toFixed(3)} | age=${Number(ins.zone_extreme_age||0)} | w=${(ins.w||0).toFixed(3)}${slopeLine}<br/>`+
          `dist_high=${(ins.dist_high||0).toFixed(3)} | dist_low=${(ins.dist_low||0).toFixed(3)} | gap=${(ins.extreme_gap||0).toFixed(3)} | ema_diff=${(ins.ema_diff||0).toFixed(1)}<br/>`+
          `zone_min_r=${(ins.zone_min_r!=null? ins.zone_min_r: ins.r||0).toFixed(3)} | zone_max_r=${(ins.zone_max_r!=null? ins.zone_max_r: ins.r||0).toFixed(3)} | zone_extreme_r=${(ins.zone_extreme_r!=null? ins.zone_extreme_r: ins.r||0).toFixed(3)}<br/>`+
          `blue_min_cur=${(ins.blue_min_cur!=null? ins.blue_min_cur: ins.zone_min_r||0).toFixed(3)} | blue_min_last=${(ins.blue_min_last!=null? ins.blue_min_last: ins.zone_min_r||0).toFixed(3)} | orange_max_cur=${(ins.orange_max_cur!=null? ins.orange_max_cur: ins.zone_max_r||0).toFixed(3)} | orange_max_last=${(ins.orange_max_last!=null? ins.orange_max_last: ins.zone_max_r||0).toFixed(3)}`;
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
  const enforceZoneSideEl = document.getElementById('enforceZoneSide');
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
  const btnBuy = document.getElementById('btnBuy');
  const btnSell = document.getElementById('btnSell');
  const tradeReadyMeta = document.getElementById('tradeReadyMeta');
  const miniWinZone = document.getElementById('miniWinZone');
  const miniWinBaseBar = document.getElementById('miniWinBaseBar');
  const miniWinOverlayBar = document.getElementById('miniWinOverlayBar');
  const autoPending = document.getElementById('autoPending');
  const autoPendingBar = document.getElementById('autoPendingBar');
  const btnCancelPending = document.getElementById('btnCancelPending');
  const autoTradeToggle = document.getElementById('autoTradeToggle');
  // Additional toggles
  let mlOnlyToggle = null;
  let autoPendingTimer = null;
  const btnPreflight = document.getElementById('btnPreflight');
  const tradeReadyBox = document.getElementById('tradeReadyBox');
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
        // append without forcing scroll
        const prevTop = logBox.scrollTop;
        logBox.textContent += (line + "\n");
        // trim to last LOG_MAX_LINES
        try{
          const parts = logBox.textContent.split('\n');
          if (parts.length > LOG_MAX_LINES+1){
            logBox.textContent = parts.slice(-LOG_MAX_LINES-1).join('\n');
          }
        }catch(_){ }
        // No auto-scroll: always keep previous position
        try{ logBox.scrollTop = prevTop; }catch(_){ }
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
  const autoGaugeBar = document.getElementById('autoGaugeBar');
  const autoGaugeText = document.getElementById('autoGaugeText');
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

  // Aggregate last up to 25 win buttons
  function updateTopPnlFromList(){
    if (!winListEl || !pnlLeft || !pnlRight) return;
    const items = Array.from(winListEl.children).slice(0,25);
    const total = items.length;
    const positives = items.filter(el=> el.classList.contains('positive')).length;
    const profitPct = total ? (positives/total*100) : 0;
    const lossPct = 100 - profitPct;
    pnlLeft.style.width = `${profitPct/2}%`;
    pnlRight.style.width = `${lossPct/2}%`;
    if (pnlLeftLabel) pnlLeftLabel.textContent = `Profit ${profitPct.toFixed(1)}%`;
    if (pnlRightLabel) pnlRightLabel.textContent = `Loss ${lossPct.toFixed(1)}%`;
    // compute majority zone from buttons' text
    try{
      let blue=0, orange=0;
      for (const el of items){
        const txt = (el.textContent||'').toUpperCase();
        if (txt.includes('BLUE')) blue++;
        else if (txt.includes('ORANGE')) orange++;
      }
      const maj = (orange>=blue && orange>0)? 'ORANGE' : (blue>orange? 'BLUE' : '-');
      const winMajor = document.getElementById('winMajor');
      if (winMajor){ winMajor.textContent = maj; winMajor.className = 'badge bg-white text-dark'; }
    }catch(_){ }
    // also update local fill bar on periodic refresh (1%..100%)
    try{
      const bar = document.getElementById('winFillBar');
      if (bar){
        const n = Math.min(25, (winListEl?.childElementCount||0));
        const pct = Math.max(1, Math.round((n/25)*100));
        bar.style.width = `${pct}%`;
      }
    }catch(_){ }
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
  function pushWinItem({ ts, pnl, winRate, zone }){
    if (!winListEl) return;
    const key = makeWinKey(pnl, winRate);
    // If duplicate exists, refresh its content and move to top
    const dup = Array.from(winListEl.children).find(el=> el.dataset && el.dataset.key === key);
    if (dup){
      const timeStr = new Date(ts).toLocaleTimeString();
      const zDup = (dup.dataset && dup.dataset.zone) || (zone) || '-';
      const meta = dup.querySelector('.meta'); if (meta) meta.textContent = `${timeStr} ${String(zDup).toUpperCase()}`;
      const val = dup.querySelector('.val'); if (val) try{ val.remove(); }catch(_){ }
      winListEl.prepend(dup);
      updateTopPnlFromList();
      return;
    }
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'win-chip btn btn-sm';
    const timeStr = new Date(ts).toLocaleTimeString();
    const z = (zone || (typeof window !== 'undefined' && window.zoneNow) || (window.lastInsight && window.lastInsight.zone)) || '-';
    item.title = `${timeStr}  ${z && z!=='-'? `Zone ${String(z).toUpperCase()}`:'-'}`;
    item.innerHTML = `<div class='meta'>${timeStr} ${String(z).toUpperCase()}</div>`;
    item.dataset.key = key;
    item.dataset.zone = String(z).toUpperCase();
    winListEl.prepend(item);
    // keep last 25
    while (winListEl.childElementCount>25){ const last = winListEl.lastElementChild; if (last && last.dataset && last.dataset.key) winKeys.delete(last.dataset.key); winListEl.removeChild(last); }
    winKeys.add(key);
    // refresh top aggregate slider
    updateTopPnlFromList();
    // update local fill bar (1%..100% while filling up to 25)
    try{
      const bar = document.getElementById('winFillBar');
      if (bar){
        const n = Math.min(25, winListEl.childElementCount||0);
        const pct = Math.max(1, Math.round((n/25)*100));
        bar.style.width = `${pct}%`;
      }
    }catch(_){ }
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
  const saveOpts = ()=>{ try{ const o = snapshotOpts(); if (o.opt_auto_save === undefined || o.opt_auto_save){ writeOpts(o); } }catch(_){ } };
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
      show_orders: undefined,
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
      enforce_zone_side: enforceZoneSideEl ? !!enforceZoneSideEl.checked : undefined,
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
      throw new Error('API response is not JSON. Open the Flask UI at: http://127.0.0.1:5057/ui');
    }
    try{ return JSON.parse(text); }catch(_){ throw new Error('Failed to parse JSON: ' + text.slice(0,120)); }
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
      nb_window: nbWindowEl ? parseInt(nbWindowEl.value||'50',10) : undefined,
      enforce_zone_side: enforceZoneSideEl ? !!enforceZoneSideEl.checked : undefined,
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
        uiLog('NB update', `window=${n}, r(last)=${(rLast||0).toFixed(3)}`);
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
        let lastReady = 0; // readiness percentage
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
            uiLog('SELL signal', `zone switch: BLUE→ORANGE, r=${r.toFixed(3)} (switched to top-dominant zone)`);
          } else if (zone === 'ORANGE' && r <= LOW && emaOkBuy){
            zone = 'BLUE';
            pushNBSignal(tm, 'BUY');
            try{ window.lastNbSignals.push({ time: tm, side: 'BUY' }); }catch(_){ }
            uiLog('BUY signal', `zone switch: ORANGE→BLUE, r=${r.toFixed(3)} (switched to bottom-dominant zone)`);
          }
          // readiness (simple): distance to threshold within current zone
          if (zone==='BLUE'){
            const d = Math.max(0, Math.min(1, (HIGH - r) / Math.max(1e-6, HIGH-LOW)));
            lastReady = Math.round((1-d)*100);
          }else{
            const d = Math.max(0, Math.min(1, (r - LOW) / Math.max(1e-6, HIGH-LOW)));
            lastReady = Math.round((d)*100);
          }
        }
        // expose latest chart-derived zone for other UI (e.g., Win buttons)
        try{ window.zoneNow = zone; }catch(_){ }
        // reflect readiness gauge
        try{
          if (autoGaugeBar){ autoGaugeBar.style.width = `${Math.max(0, Math.min(100, lastReady))}%`; }
          if (autoGaugeText){ autoGaugeText.textContent = `${Math.max(0, Math.min(100, lastReady))}%`; autoGaugeText.className = 'badge ' + (lastReady>=99? 'bg-success': 'bg-secondary'); }
        }catch(_){ }
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
  // Predicted path series
  const predSeries = chart.addLineSeries({ color:'#ffffff', lineStyle: 0, lineWidth: 2 });
  const predMarkerSeries = chart.addLineSeries({ color:'rgba(0,0,0,0)', lineWidth: 0, priceLineVisible:false });
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

  async function drawPredictedPath(){
    try{
      const j = await fetchJsonStrict('/api/ml/predict');
      if (!j || !j.ok) {
        // Always show narrative even if prediction not available
        predSeries.setData([]);
        try{
          const box = document.getElementById('nbNarrative');
          const badge = document.getElementById('nbNarrativeBadge');
          if (box){
            const zone = (typeof window!=='undefined' && window.zoneNow) ? String(window.zoneNow).toUpperCase() : '-';
            const line = `Current zone: ${zone}. Model prediction not available yet. Waiting for training/prediction...`;
            box.textContent = line;
            if (badge) { badge.textContent = zone; badge.className = 'badge bg-white text-dark'; }
          }
        }catch(_){ }
        return;
      }
      const steep = j.steep || {};
      const ins = j.insight || {};
      const data = candle.data(); if (!data || data.length < 5){ predSeries.setData([]); return; }
      const last = data[data.length-1];
      const times = data.map(d=>d.time);
      const closeNow = last.close ?? last.value;
      const interval = j.interval || getInterval();
      const curIv = getInterval();
      const sameIv = (String(interval) === String(curIv));
      const horizon = Math.max(1, Number(j.horizon||5));
      const bpPerBar = (ins.zone==='BLUE' ? steep.blue_up_slope : steep.orange_down_slope);
      let v = closeNow;
      if (bpPerBar==null){
        // No slope yet → keep path empty but still update narrative below
        predSeries.setData([]);
      } else {
        // bp/bar → fractional slope per bar
        const k = Number(bpPerBar)/10000.0;
        const dt = (times[times.length-1] - times[times.length-2]) || 60; // seconds
        const step = dt; // seconds per bar
        const proj = [{ time: last.time, value: closeNow }];
        for (let i=1;i<=horizon;i++){
          v = v * (1 + k); // geometric per bar
          proj.push({ time: last.time + i*step, value: v });
        }
        predSeries.setData(proj);
      }
      // NB signal marker at predicted flip time
      try{
        const nb = j.pred_nb;
        if (sameIv && nb && nb.ts){
          const m = [{ time: msToSec(nb.ts), value: v }];
          predMarkerSeries.setData(m);
          candle.setMarkers([
            { time: msToSec(nb.ts), position: nb.side==='BUY'?'belowBar':'aboveBar', color: nb.side==='BUY'?'#ffd166':'#00d1ff', shape:'circle', text:`ML NB ${nb.side}` }
          ]);
        }
      }catch(_){}
      // English narrative using current NB/zone and predicted path
      try{
        const box = document.getElementById('nbNarrative');
        const badge = document.getElementById('nbNarrativeBadge');
        if (box){
          const zone = (j.insight?.zone||'-').toUpperCase();
          const slope = (j.steep && (j.steep.blue_up_slope!=null ? j.steep.blue_up_slope : j.steep.orange_down_slope));
          const slopeBp = (slope!=null) ? (slope*10000).toFixed(2) : '-';
          const nb = j.pred_nb || null;
          const nbTxt = (nb && nb.side) ? `${nb.side} in ~${nb.bars} bars` : 'no flip expected soon';
          const line = `Current zone: ${zone}. Model projects a ${slope!=null ? (zone==='ORANGE'?'down':'up') : 'flat'} slope of ${slopeBp} bp/bar. Expected NB flip: ${nbTxt}.`;
          box.textContent = line;
          if (badge) { badge.textContent = zone; badge.className = 'badge bg-white text-dark'; }
        }
      }catch(_){ }
      // Place one ML signal per zone segment at its extreme if certain, and persist to server
      try{
        const zone = String(ins.zone||'-').toUpperCase();
        if (mlSegPrevZone !== zone){ mlSegPrevZone = zone; mlSegPlaced = false; }
        const barSec = (times[times.length-1] - times[times.length-2]) || 60; // seconds per bar
        const age = Number(ins.zone_extreme_age||0);
        const extremeTime = last.time - Math.max(0, age)*barSec;
        const extremePrice = (ins.zone_extreme_price!=null) ? Number(ins.zone_extreme_price) : closeNow;
        const pb = Number(ins.pct_blue||ins.pct_blue_raw||0);
        const po = Number(ins.pct_orange||ins.pct_orange_raw||0);
        const pctMajor = Math.max(pb, po);
        const slope = (j.steep && (j.steep.blue_up_slope!=null ? j.steep.blue_up_slope : j.steep.orange_down_slope));
        const slopeBp = (slope!=null) ? (slope*10000) : 0;
        const predOk = !!(j.pred_nb && j.pred_nb.ts);
        const confTh = 99.95, minBp = 1.0, minAge = 3;
        const gated = (pctMajor >= confTh) && (Math.abs(slopeBp) >= minBp) && predOk && (age >= minAge);
        const extreme = (zone==='ORANGE') ? 'TOP' : (zone==='BLUE' ? 'BOTTOM' : '-');
        const sideBuy = (zone==='BLUE');
        const key = `${interval}|${zone}|${extreme}|${Math.floor(extremeTime)}`;
        if (sameIv && zone!=='-' && extreme!=='-' && gated && !mlSegPlaced && !mlSignalKeys.has(key)){
          const marker = { time: Math.floor(extremeTime), position: sideBuy?'belowBar':'aboveBar', color: sideBuy?'#ffd166':'#00d1ff', shape:'circle', text:`ML ${sideBuy?'BUY':'SELL'}` };
          baseMarkers.push(marker);
          if (baseMarkers.length>500) baseMarkers = baseMarkers.slice(-500);
          candle.setMarkers([...baseMarkers, ...nbMarkers]);
          mlSegPlaced = true;
          mlSignalKeys.add(key);
          const body = { ts: Math.floor(extremeTime*1000), zone, extreme, price: extremePrice, pct_major: pctMajor, slope_bp: slopeBp, horizon, pred_nb: j.pred_nb||null, interval, score0: Number(j.score0||0) };
          postJson('/api/signal/log', body).catch(()=>{});
        }
      }catch(_){ }
    }catch(_){ predSeries.setData([]); }
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
      if (pred && pred.ok){ uiLog('ML Auto predict', `action=${pred.action}, pred=${pred.pred}`); }
    }catch(_){ }
  }

  let baseMarkers = [];
  let nbMarkers = [];
  // ML segment state and logged keys to avoid duplicate markers per segment
  let mlSegPrevZone = null;
  let mlSegPlaced = false;
  let mlSignalKeys = new Set();

  // Helper: show order markers only if there is an ML signal near the order time on this timeframe
  function hasMlSignalNear(orderTimeSec, interval){
    try{
      const curIv = interval || getInterval();
      const data = candle.data();
      if (!data || data.length < 2) return false;
      const barSec = (data[data.length-1].time - data[data.length-2].time) || 60;
      const maxDelta = barSec * 2; // within ±2 bars
      // baseMarkers contains both ML signal markers (text starts with 'ML') and order markers
      for (const m of baseMarkers){
        try{
          if (!m || !m.text) continue;
          if (String(m.text).startsWith('ML')){
            if (Math.abs(Number(m.time) - Number(orderTimeSec)) <= maxDelta) return true;
          }
        }catch(_){ }
      }
      return false;
    }catch(_){ return false; }
  }
  // NB wave based lightweight signals on the UI
  let nbPosition = 'FLAT';
  let nbPeakRatio = 0; // highest ratio seen while LONG
  const NB_UP_TH = 0.7; // buy threshold
  const NB_DN_TH = 0.3; // sell floor
  function pushNBSignal(timeSec, side){
    // Disabled NB system markers; showing ML NB markers only
    return;
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
    const curIv = getInterval();
    const orderIv = String(o.interval||'');
    // Show orders only when they belong to current chart interval
    if (orderIv && String(orderIv) !== String(curIv)) return;
    const sec = msToSec(bucketTs(Number(o.ts), curIv));
    // Skip if there is no ML signal near this order time
    if (!hasMlSignalNear(sec, curIv)) return;
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
        let line = `[${ts}] ${isBuy? 'BUY':'SELL'} @${Number(o.price||0).toLocaleString()} ${o.size? '('+Number(o.size).toFixed(6)+')':''} ${o.paper? '[PAPER]':''}`;
        // Append model insight snapshot if present
        try{
          const ins = o.insight || (typeof window!=='undefined' && window.lastInsight ? window.lastInsight : {});
          if (ins && (typeof ins === 'object')){
            const r = isFinite(ins.r)? Number(ins.r).toFixed(3) : '-';
            const zone = String(ins.zone||'-');
            const cb = isFinite(ins.pct_blue)? Number(ins.pct_blue).toFixed(1) : (isFinite(ins.pct_blue_raw)? Number(ins.pct_blue_raw).toFixed(1) : '-');
            const co = isFinite(ins.pct_orange)? Number(ins.pct_orange).toFixed(1) : (isFinite(ins.pct_orange_raw)? Number(ins.pct_orange_raw).toFixed(1) : '-');
            const minr = isFinite(ins.zone_min_r)? Number(ins.zone_min_r).toFixed(3) : '-';
            const maxr = isFinite(ins.zone_max_r)? Number(ins.zone_max_r).toFixed(3) : '-';
            const exr = isFinite(ins.zone_extreme_r)? Number(ins.zone_extreme_r).toFixed(3) : '-';
            const age = isFinite(ins.zone_extreme_age)? Number(ins.zone_extreme_age) : '-';
            line += ` | r=${r} | zone=${zone} | BLUE=${cb}% | ORANGE=${co}% | min_r=${minr} | max_r=${maxr} | ex_r=${exr} | age=${age}`;
          }
        }catch(_){ }
        // NB trade signal context
        try{
          const nbSig = String(o.nb_signal||'').toUpperCase();
          const nbWin = Number(o.nb_window||0);
          const nbR = (o.nb_r!=null) ? Number(o.nb_r).toFixed(3) : undefined;
          if (nbSig){ line += ` | NB=${nbSig}${nbWin? ' w='+nbWin:''}${(nbR!==undefined)? ' r='+nbR:''}`; }
        }catch(_){ }
        const div = document.createElement('div');
        // Only log when an actual order happened: in paper mode always, in live only if o.live_ok
        try{
          const liveOk = (!o.paper) ? !!o.live_ok : true;
          if (liveOk){ div.textContent = line; orderLog.prepend(div); }
        }catch(_){ div.textContent = line; orderLog.prepend(div); }
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
      // Ichimoku Tenkan/Kijun (simple high-low average)
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
      // load existing orders; show only when order interval matches current chart interval
      const curIv = getInterval();
      return fetch(`${base}/api/orders`).then(r=>r.json()).then(or=>{
        markers=[]; (or.data||[]).forEach(o=>{
          try{
            const ok = !o.interval || String(o.interval)===String(curIv);
            if (ok) pushOrderMarker(o, interval);
          }catch(_){ pushOrderMarker(o, interval); }
        });
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
    // ordersToggle removed
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
    // extras
    try{
      const enforceZoneSideEl = document.getElementById('enforceZoneSide');
      if (enforceZoneSideEl && typeof o.enforce_zone_side !== 'undefined') enforceZoneSideEl.checked = !!o.enforce_zone_side;
      const optAutoSaveEl = document.getElementById('optAutoSave');
      if (optAutoSaveEl && typeof o.opt_auto_save !== 'undefined') optAutoSaveEl.checked = !!o.opt_auto_save;
    }catch(_){ }
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
    // Do not auto-start anything here
  })();

  seed(getInterval());
  // periodic prediction path
  setInterval(()=>{ drawPredictedPath(); }, 3000);
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
  // ordersToggle removed
  if (autoBtSecEl) autoBtSecEl.addEventListener('change', saveOpts);
  try{
    const enforceZoneSideEl2 = document.getElementById('enforceZoneSide');
    const assetsAutoToggle2 = document.getElementById('assetsAuto');
    const optAutoSaveEl2 = document.getElementById('optAutoSave');
    if (enforceZoneSideEl2) enforceZoneSideEl2.addEventListener('change', ()=>{ saveOpts(); pushConfig(); });
    if (assetsAutoToggle2) assetsAutoToggle2.addEventListener('change', saveOpts);
    if (optAutoSaveEl2) optAutoSaveEl2.addEventListener('change', ()=>{ writeOpts({ opt_auto_save: !!optAutoSaveEl2.checked }); });
  }catch(_){ }
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
    // Start bot only via explicit Auto Trade toggle; here we do nothing to avoid accidental starts
    uiLog('Hint', 'Use Auto Trade toggle to start the bot');
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
          if (!(j && j.ok)) { uiLog('ML Auto random train failed', JSON.stringify(j)); }
          // Ensure narrative renders even during training gaps
          try{ await drawPredictedPath(); }catch(_){ }
          if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
          await backtestAfterReady(6000);
          await sleep(800);
          await backtestAfterReady(3000);
          const pred = await fetchJsonStrict('/api/ml/predict');
          if (pred && pred.ok){
            uiLog('ML Auto random predict', `action=${pred.action}, pred=${pred.pred}`);
            if (mlCountEl) mlCountEl.textContent = `(train# ${pred.train_count||0})`;
            updateModelInsight(pred);
          }
          // Update narrative regardless
          try{ await drawPredictedPath(); }catch(_){ }
        }catch(_){ }
      };
      const sec = Math.max(5, parseInt(autoBtSecEl?.value||'15',10));
      uiLog('ML Auto random ON', `interval=${sec}s`);
      run();
      mlAutoTimer = setInterval(run, sec*1000);
    } else {
      uiLog('ML Auto random OFF');
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
      if (!(j && j.ok)){ uiLog('ML Metrics failed', JSON.stringify(j)); return; }
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
    }catch(e){ uiLog('ML Metrics error', String(e)); }
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
          if (!raw.length){ uiLog('Backtest canceled', 'no signal'); return; }
          uiLog('No NB signal → fallback to EMA cross backtest');
        }catch(_){ uiLog('Backtest canceled', 'no signal'); return; }
      }
      // 1) De-duplicate to alternate BUY/SELL
      const norm=[]; let lastSide=null;
      for(const m of raw){ const side = m.text.includes('BUY')?'BUY':(m.text.includes('SELL')?'SELL':null); if(!side) continue; if(side===lastSide) continue; norm.push({time:m.time, side}); lastSide=side; }
      // Drop leading SELL
      while (norm.length && norm[0].side==='SELL') norm.shift();
      if (norm.length<2){ uiLog('Backtest canceled', 'insufficient signals'); return; }
      // 2) Pair trades and compute PnL/Win%
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
      uiLog('Backtest done', `trades=${trades}, wins=${wins}, pnl=${pnl.toFixed(0)}, win%=${winRate.toFixed(1)}%, maxDD=${dd.toFixed(0)}`);
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

  // ordersToggle removed

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

  // Trade readiness panel (buyable/sellable)
  async function refreshTradeReady(){
    try{
      const j = await fetchJsonStrict('/api/trade/preflight');
      if (!j || !j.ok){ if (tradeReadyBox) tradeReadyBox.textContent = 'Preflight error'; return; }
      const p = j.preflight||{};
      const price = Number(p.price||0);
      const krw = Number(p.krw||0);
      const coinBal = Number(p.coin_balance||0);
      const buyKrw = Number(p.planned_buy_krw||0);
      const sellSize = Number(p.planned_sell_size||0);
      const sym = (p.market||'KRW-COIN').split('-')[1]||'';
      const buyRemain = Math.max(0, krw - buyKrw);
      const sellRemain = Math.max(0, coinBal - sellSize);
      const buyLine = p.can_buy
        ? `after BUY: ${buyRemain.toLocaleString()} KRW left (spend ${buyKrw.toLocaleString()} KRW)`
        : `need ≥ 5,000 KRW (KRW=${krw.toLocaleString()})`;
      const sellLine = p.can_sell
        ? `after SELL: ${sellRemain.toFixed(8)} ${sym} left (sell ${sellSize.toFixed(8)} ≈ ${Math.round(sellSize*price).toLocaleString()} KRW)`
        : `need ≥ 5,000 KRW notional (bal=${coinBal.toFixed(8)} ${sym})`;
      if (tradeReadyBox){
        // Fetch current N/B COIN status
        let coinTxt = '-';
        try{
          const cs = await fetchJsonStrict(`/api/nb/coin?interval=${encodeURIComponent(getInterval())}`);
          if (cs && cs.ok){
            const c = cs.current;
            if (c && c.side){ coinTxt = c.side; }
          }
        }catch(_){ }
        const minSellSize = price>0 ? (5000/price) : 0;
        tradeReadyBox.innerHTML = `
          <div>Price: <b>${price? price.toLocaleString(): '-'}</b></div>
          <div>N/B COIN (this bar): <b id="nbCoinNowInline">${coinTxt}</b></div>
          <div>Buy: <b>${buyLine}</b></div>
          <div>Sell: <b>${sellLine}</b></div>
          <div>Min SELL size (~5,000 KRW): <b>${minSellSize>0? minSellSize.toFixed(8): '-'}</b> ${sym}</div>
          <div>Keys: ${p.has_keys} | Paper: ${p.paper}</div>
        `;
        if (tradeReadyMeta){ tradeReadyMeta.textContent = `(${new Date().toLocaleTimeString()})`; }
      }
    }catch(_){ if (tradeReadyBox) tradeReadyBox.textContent = 'Preflight error'; }
  }
  refreshTradeReady().catch(()=>{});
  setInterval(()=>{ refreshTradeReady(); }, 15000);
  if (assetsRefresh) assetsRefresh.addEventListener('click', ()=>{ refreshTradeReady(); });

  // N/B COIN strip renderer
  async function refreshNbCoinStrip(){
    try{
      const strip = document.getElementById('nbCoinStrip');
      const nowBadge = document.getElementById('nbCoinNow');
      const nowInline = document.getElementById('nbCoinNowInline');
      if (!strip && !nowBadge && !nowInline) return;
      let cs = null; let cur = null; let recent = [];
      try{
        cs = await fetchJsonStrict(`/api/nb/coin?interval=${encodeURIComponent(getInterval())}&n=50`);
        if (cs && cs.ok){ cur = cs.current||null; recent = cs.recent||[]; }
      }catch(_){ }
      const label = cur && cur.side ? cur.side : '-';
      if (nowBadge){ nowBadge.textContent = label; }
      if (nowInline){ nowInline.textContent = label; }
      if (strip){
        strip.innerHTML = '';
        // fallback placeholders when no data
        if (!recent || recent.length===0){
          recent = Array.from({length:50}).map((_,i)=>({ bucket: 0, side:'NONE' }));
        }
        // left older → right newer
        recent.reverse().forEach(c=>{
          const el = document.createElement('div');
          el.style.height = '8px'; el.style.flex = '1 1 auto'; el.style.margin = '0 1px'; el.title = `${new Date((c.bucket||0)*1000).toLocaleTimeString()} ${c.side||'NONE'}`;
          const side = String(c.side||'NONE').toUpperCase();
          el.style.background = side==='BUY' ? '#0ecb81' : (side==='SELL' ? '#f6465d' : '#2b3139');
          strip.appendChild(el);
        });
        // If current bar has diagnostics, append a compact reason list below the strip
        try{
          const curCoin = cur || null;
          const reasons = (curCoin && Array.isArray(curCoin.reasons)) ? curCoin.reasons.slice(-5) : [];
          if (reasons.length){
            const diag = document.createElement('div');
            diag.className = 'text-muted';
            diag.style.fontSize = '11px';
            diag.style.marginTop = '4px';
            diag.textContent = `Why no trade: ${reasons.join(', ')}`;
            strip.parentElement?.appendChild(diag);
          }
        }catch(_){ }
      }
    }catch(_){ }
  }
  // initial and periodic refresh for N/B COIN
  refreshNbCoinStrip().catch(()=>{});
  setInterval(()=>{ refreshNbCoinStrip(); }, 8000);

  // Zone Win% mini gauge updater (from winMajor)
  function refreshMiniWinGaugeFromWinMajor(){
    try{
      const winMajorEl = document.getElementById('winMajor');
      if (!winMajorEl) return;
      const txt = (winMajorEl.textContent||'').toUpperCase().trim();
      if (!(txt==='BLUE' || txt==='ORANGE')) return;
      // pct: 미니 게이지는 100%로 고정(요구사항: mini가 winMajor 값을 그대로 사용)
      const isBlueMajor = (txt==='BLUE');
      const pct = 100;
      if (miniWinZone) miniWinZone.textContent = `${txt} ${pct}%`;
      if (miniWinBaseBar) miniWinBaseBar.style.background = isBlueMajor ? '#ffb703' : '#00d1ff';
      if (miniWinOverlayBar){ miniWinOverlayBar.style.background = isBlueMajor ? '#00d1ff' : '#ffb703'; miniWinOverlayBar.style.width = `${pct}%`; }
    }catch(_){ }
  }
  // Wrap updateModelInsight to also drive mini gauge if present
  try{
    const _prevUpdateModelInsight = updateModelInsight;
    updateModelInsight = function(j){
      try{ _prevUpdateModelInsight(j); }catch(_){ }
      try{ refreshMiniWinGaugeFromWinMajor(); }catch(_){ }
    }
  }catch(_){ }

  // Manual trade buttons
  if (btnBuy) btnBuy.addEventListener('click', async ()=>{
    try{
      // Arm auto order with 5-sec cancel window
      armAutoPending(async ()=>{
        const j = await postJson('/api/trade/buy', {});
        if (j && j.ok && j.order){ pushOrderMarker(j.order); uiLog('Manual BUY', JSON.stringify({ price:j.order.price, size:j.order.size, paper:j.order.paper })); }
        else { uiLog('Manual BUY failed', JSON.stringify(j)); }
        try{ refreshTradeReady(); }catch(_){ }
      });
    }catch(e){ uiLog('Manual BUY error', String(e)); }
  });
  if (btnSell) btnSell.addEventListener('click', async ()=>{
    try{
      armAutoPending(async ()=>{
        const j = await postJson('/api/trade/sell', {});
        if (j && j.ok && j.order){ pushOrderMarker(j.order); uiLog('Manual SELL', JSON.stringify({ price:j.order.price, size:j.order.size, paper:j.order.paper })); }
        else { uiLog('Manual SELL failed', JSON.stringify(j)); }
        try{ refreshTradeReady(); }catch(_){ }
      });
    }catch(e){ uiLog('Manual SELL error', String(e)); }
  });

  function armAutoPending(executeFn){
    try{
      if (!autoPending || !autoPendingBar){ executeFn(); return; }
      // Reset UI
      autoPending.style.display = '';
      autoPendingBar.style.width = '0%';
      let ms = 5000; const step = 100;
      if (autoPendingTimer) { clearInterval(autoPendingTimer); autoPendingTimer=null; }
      autoPendingTimer = setInterval(()=>{
        ms -= step; const pct = Math.max(0, Math.min(100, Math.round(((5000-ms)/5000)*100)));
        autoPendingBar.style.width = pct + '%';
        if (ms <= 0){ clearInterval(autoPendingTimer); autoPendingTimer=null; autoPending.style.display='none'; executeFn(); }
      }, step);
      if (btnCancelPending){
        btnCancelPending.onclick = ()=>{
          try{ if (autoPendingTimer) clearInterval(autoPendingTimer); }catch(_){ }
          autoPendingTimer = null; autoPending.style.display='none'; uiLog('Auto order cancelled within 5s');
        };
      }
    }catch(_){ executeFn(); }
  }

  // Live Trade Preflight test
  if (btnPreflight) btnPreflight.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/trade/preflight');
      if (!j.ok){ uiLog('Preflight failed', JSON.stringify(j)); return; }
      const p = j.preflight || {};
      const lines = [
        `paper=${p.paper} keys=${p.has_keys} market=${p.market} price=${Number(p.price||0).toLocaleString()}`,
        `KRW=${Number(p.krw||0).toLocaleString()} coin_bal=${p.coin_balance}`,
        `BUY_KRW=${Number(p.planned_buy_krw||0).toLocaleString()} (>=5000 → ${p.can_buy})`,
        `SELL_SIZE=${p.planned_sell_size} (>=5000KRW → ${p.can_sell})`,
      ];
      uiLog('Preflight', lines.join(' | '));
    }catch(e){ uiLog('Preflight error', String(e)); }
  });

  // Auto Trade toggle: start/stop server trade loop
  if (autoTradeToggle){
    autoTradeToggle.addEventListener('change', async ()=>{
      try{
        if (autoTradeToggle.checked){
          await postJson('/api/bot/start', {});
          uiLog('Auto Trade', 'started');
        } else {
          await postJson('/api/bot/stop', {});
          uiLog('Auto Trade', 'stopped');
        }
      }catch(e){ uiLog('Auto Trade toggle error', String(e)); }
    });
  }
  // Inject ML-only/ML-seg-only toggles next to Auto Trade (runtime only)
  try{
    const parent = document.getElementById('autoTradeToggle')?.closest('.card');
    const holder = document.getElementById('tradeReadyBox')?.parentElement;
    if (holder){
      const wrap = document.createElement('div');
      wrap.className = 'mt-2';
      wrap.innerHTML = `<div class=\"form-check form-switch\"><input class=\"form-check-input\" type=\"checkbox\" id=\"mlOnlyToggle\"><label class=\"form-check-label text-muted\" for=\"mlOnlyToggle\">ML-only Auto Trade</label></div>
      <div class=\"form-check form-switch mt-1\"><input class=\"form-check-input\" type=\"checkbox\" id=\"mlSegOnlyToggle\"><label class=\"form-check-label text-muted\" for=\"mlSegOnlyToggle\">ML segment-only (extreme only)</label></div>`;
      holder.appendChild(wrap);
      mlOnlyToggle = document.getElementById('mlOnlyToggle');
      mlOnlyToggle.addEventListener('change', async ()=>{
        try{ await postJson('/api/bot/config', { ml_only: !!mlOnlyToggle.checked }); uiLog('Config', `ml_only=${mlOnlyToggle.checked}`); }catch(_){ }
      });
      mlSegOnlyToggle = document.getElementById('mlSegOnlyToggle');
      mlSegOnlyToggle && mlSegOnlyToggle.addEventListener('change', async ()=>{
        try{ await postJson('/api/bot/config', { ml_seg_only: !!mlSegOnlyToggle.checked }); uiLog('Config', `ml_seg_only=${mlSegOnlyToggle.checked}`); }catch(_){ }
      });
    }
  }catch(_){ }

  if (optBtn) optBtn.addEventListener('click', ()=>{ optimizeNb(); });
  if (trainBtn) trainBtn.addEventListener('click', async ()=>{
    try{
      const payload = { count: parseInt(trainCountEl?.value||'1800',10), segments: parseInt(trainSegEl?.value||'3',10), window: parseInt(nbWindowEl?.value||'50',10), debounce: parseInt(nbDebounceEl?.value||'6',10), fee_bps: 10.0, interval: getInterval() };
      uiLog('NB Train start', `auto split: ${payload.segments} segments, candle=${payload.interval}, count=${payload.count}`);
      const r = await fetch('/api/nb/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const j = await r.json();
      if (j && j.ok){
        uiLog('NB Train done', `chosen seg=${j.chosen.segment}, PnL=${j.chosen.stats.pnl.toFixed(0)}, BUY=${j.chosen.best.buy}, SELL=${j.chosen.best.sell}`);
        if (nbBuyThEl) nbBuyThEl.value = String(j.chosen.best.buy);
        if (nbSellThEl) nbSellThEl.value = String(j.chosen.best.sell);
        updateNB();
      } else { uiLog('NB Train failed', JSON.stringify(j)); }
    }catch(e){ uiLog('NB Train error', String(e)); }
  });
  if (autoBtToggle) autoBtToggle.addEventListener('change', ()=>{
    if (autoBtToggle.checked){
      const run = ()=>{ if (btBtn) btBtn.click(); };
      const sec = Math.max(10, parseInt(autoBtSecEl?.value||'60',10));
      run();
      autoBtTimer = setInterval(run, sec*1000);
      uiLog('Auto backtest start', `interval=${sec}s`);
    } else {
      if (autoBtTimer) clearInterval(autoBtTimer); autoBtTimer=null;
      uiLog('Auto backtest stop');
    }
  });
  if (mlTrainBtn) mlTrainBtn.addEventListener('click', async ()=>{
    try{
      uiLog('ML Train start', 'LightGBM/GBDT (sklearn) baseline');
      const payload = { window: parseInt(nbWindowEl?.value||'50',10), ema_fast: parseInt(emaFastEl?.value||'10',10), ema_slow: parseInt(emaSlowEl?.value||'30',10), horizon: 5, tau: 0.002, count: 1800, interval: getInterval() };
      const j = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (j && j.ok){ uiLog('ML Train done', `labels: BUY=${j.classes['1']}, HOLD=${j.classes['0']}, SELL=${j.classes['-1']}`); if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`; }
      else { uiLog('ML Train failed', JSON.stringify(j)); }
    }catch(e){ uiLog('ML Train error', String(e)); }
  });
  if (mlPredictBtn) mlPredictBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/ml/predict');
      if (j && j.ok){
        uiLog('ML Predict', `action=${j.action}, pred=${j.pred}`);
        if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
        updateModelInsight(j);
      }
      else { uiLog('ML Predict failed', JSON.stringify(j)); }
    }catch(e){ uiLog('ML Predict error', String(e)); }
  });
  if (mlRandomBtn) mlRandomBtn.addEventListener('click', async ()=>{
    try{
      const n = Math.max(1, parseInt(mlRandNEl?.value||'10',10));
      uiLog('ML Random Train start', `trials=${n}`);
      for (let i=0;i<n;i++){
        const mins = [1,3,5,10,15,30,60][Math.floor(Math.random()*7)];
        const interval = mins===60 ? 'minute60' : `minute${mins}`;
        const window = Math.floor(20 + Math.random()*100); // 20~120
        const ema_fast = Math.floor(5 + Math.random()*20); // 5~25
        const ema_slow = Math.max(ema_fast+5, Math.floor(20 + Math.random()*60));
        // Reflect random options on UI so user can see
        try{
          if (tfEl){ tfEl.value = interval; tfEl.dispatchEvent(new Event('change')); }
          if (emaFastEl){ emaFastEl.value = String(ema_fast); emaFastEl.dispatchEvent(new Event('change')); }
          if (emaSlowEl){ emaSlowEl.value = String(ema_slow); emaSlowEl.dispatchEvent(new Event('change')); }
          if (typeof nbWindowEl !== 'undefined' && nbWindowEl){ nbWindowEl.value = String(window); nbWindowEl.dispatchEvent(new Event('change')); }
          // short wait so chart/indicators update
          await sleep(400);
        }catch(_){ }
        const payload = { window, ema_fast, ema_slow, horizon: 5, tau: 0.002, count: 1200, interval };
        uiLog('ML Random Train', JSON.stringify(payload));
        const j = await fetchJsonStrict('/api/ml/train', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
        if (!(j && j.ok)) { uiLog('Train failed, skipping attempt', JSON.stringify(j)); continue; }
        if (mlCountEl) mlCountEl.textContent = `(train# ${j.train_count||0})`;
        // After each random trial: wait NB signals ready → run backtest
        try{
          // Retry several times to absorb async delay
          await backtestAfterReady(6000);
          await sleep(1200); await backtestAfterReady(3000);
        }catch(_){ }
      }
      const pred = await fetchJsonStrict('/api/ml/predict');
      if (pred && pred.ok){ uiLog('ML Predict(after random)', `action=${pred.action}, pred=${pred.pred}`); if (mlCountEl) mlCountEl.textContent = `(train# ${pred.train_count||0})`; }
      else { uiLog('ML Predict failed(after random)', JSON.stringify(pred)); }
      // 마지막으로 한 번 더 백테스트 갱신
      try{
        await backtestAfterReady(4000);
        await sleep(1200); await backtestAfterReady(3000);
      }catch(_){ }
    }catch(e){ uiLog('ML Random error', String(e)); }
  });
  if (loadBalBtn) loadBalBtn.addEventListener('click', async ()=>{
    try{
      const j = await fetchJsonStrict('/api/balance');
      const box = document.getElementById('balanceBox');
      if (!box) return;
      if (!j.ok){ box.textContent = `Error: ${j.error||'unknown'}`; return; }
      if (j.paper){ box.textContent = 'PAPER mode (no live assets)'; return; }
      const rows = (j.balances||[]);
      const lines = rows.map(b=>`${b.currency}: balance=${b.balance} locked=${b.locked} avg_buy=${b.avg_buy_price}`);
      box.textContent = lines.length? lines.join('\n') : 'No balances';
    }catch(e){ const box = document.getElementById('balanceBox'); if (box) box.textContent = String(e); }
  });
  // --- Top assets auto loader ---
  async function refreshAssets(){
    try{
      const j = await fetchJsonStrict('/api/balance');
      if (!j.ok){ if (assetsMeta) assetsMeta.textContent = `(error: ${j.error||'unknown'})`; return; }
      if (j.paper){ if (assetsMeta) assetsMeta.textContent = '(PAPER mode)'; return; }
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


