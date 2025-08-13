// TradingView Lightweight Charts integration
// Pan-only (no zoom); real-time update via SSE; EMA overlays; optional volume histogram

(function(){
  const container = document.getElementById('tvChart');
  if (!container) return;

  const proto = (location.protocol === 'https:') ? 'https' : 'http';
  const base = `${proto}://127.0.0.1:5057`;

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

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#0ecb81', downColor: '#f6465d', wickUpColor: '#0ecb81', wickDownColor: '#f6465d', borderVisible: false,
  });
  const emaFastSeries = chart.addLineSeries({ color: 'rgba(14,203,129,0.9)', lineWidth: 2 });
  const emaSlowSeries = chart.addLineSeries({ color: 'rgba(246,70,93,0.9)', lineWidth: 2 });
  const volumeSeries = chart.addHistogramSeries({ priceScaleId: 'left', color: 'rgba(76,201,240,0.5)', lineWidth: 1 });
  chart.priceScale('left').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 }, borderColor: 'rgba(255,255,255,0.08)' });

  // Order markers storage
  let markers = [];
  const pushOrderMarker = (o, interval) => {
    if (!o || !o.ts) return;
    const sec = msToSec(bucketTs(Number(o.ts), interval || 'minute10'));
    const isBuy = (String(o.side).toUpperCase() === 'BUY');
    markers.push({
      time: sec,
      position: isBuy ? 'belowBar' : 'aboveBar',
      color: isBuy ? '#0ecb81' : '#f6465d',
      shape: isBuy ? 'arrowUp' : 'arrowDown',
      text: `${isBuy ? 'B' : 'S'} @${Number(o.price||0).toLocaleString()} (${(o.size||0).toFixed ? (o.size||0).toFixed(6) : o.size||''})`,
    });
    if (markers.length > 300) markers = markers.slice(-300);
    candleSeries.setMarkers(markers);
  };

  function ema(values, period){
    if (!values || values.length === 0) return [];
    const k = 2/(period+1); const out=[]; let prev = values[0];
    for (let i=0;i<values.length;i++){
      const v = (i===0) ? values[0] : values[i]*k + prev*(1-k); out.push(v); prev=v;
    }
    return out;
  }

  function msToSec(ms){ return Math.floor(ms/1000); }
  function bucketTs(tsMs, interval){
    if (interval.startsWith('minute')){ const m = parseInt(interval.replace('minute',''),10)||1; return Math.floor(tsMs/(m*60*1000))*(m*60*1000); }
    if (interval === 'minute60'){ return Math.floor(tsMs/(60*60*1000))*(60*60*1000); }
    if (interval === 'day'){ const d = new Date(tsMs); d.setHours(0,0,0,0); return d.getTime(); }
    return tsMs;
  }

  function seed(interval){
    fetch(`${base}/api/ohlcv?interval=${interval}&count=300`).then(r=>r.json()).then(res => {
      const rows = res.data || [];
      const candles = rows.map(r => ({ time: msToSec(r.time), open: r.open, high: r.high, low: r.low, close: r.close }));
      candleSeries.setData(candles);
      const closes = rows.map(r => r.close);
      const times = rows.map(r => msToSec(r.time));
      const f = ema(closes, 10).map((y,i)=> ({ time: times[i], value: y }));
      const s = ema(closes, 30).map((y,i)=> ({ time: times[i], value: y }));
      emaFastSeries.setData(f); emaSlowSeries.setData(s);
      const vols = rows.map((r,i)=> ({ time: msToSec(r.time), value: r.volume||0, color: (i>0 ? (rows[i].close >= rows[i-1].close) : (r.close>=r.open)) ? 'rgba(14,203,129,0.5)' : 'rgba(246,70,93,0.5)' }));
      volumeSeries.setData(vols);
      // load existing orders for markers
      fetch(`${base}/api/orders`).then(r=>r.json()).then(or => {
        if (Array.isArray(or.data)){
          markers = [];
          or.data.forEach(o => pushOrderMarker(o, interval));
        }
      }).catch(()=>{});
    }).catch(()=>{});
  }

  // initial
  const tfSelect = document.getElementById('timeframe');
  const getInterval = () => (tfSelect ? tfSelect.value : 'minute10');
  seed(getInterval());
  if (tfSelect) tfSelect.addEventListener('change', () => seed(getInterval()));

  // Live updates via SSE
  try{
    const es = new EventSource(`${base}/api/stream`);
    es.onmessage = (e) => {
      try{
        const j = JSON.parse(e.data);
        const meta = document.getElementById('meta'); if (meta) meta.textContent = `${j.market} ${j.candle} | ${j.signal} | EMA ${j.ema_fast}/${j.ema_slow}`;
        const priceEl = document.getElementById('s_price'); if (priceEl) priceEl.textContent = (j.price||0).toLocaleString();
        const sigEl = document.getElementById('s_signal'); if (sigEl){ sigEl.textContent=j.signal; sigEl.className = (j.signal==='BUY'?'buy':'sell'); }
        const tickMs = j.ts; const interval = getInterval(); const bMs = bucketTs(tickMs, interval); const bSec = msToSec(bMs);
        const last = candleSeries.dataByIndex(candleSeries.data().length-1, LightweightCharts.MismatchDirection.None);
        if (last && last.time === bSec){
          const updated = { ...last, close: j.price, high: Math.max(last.high, j.price), low: Math.min(last.low, j.price) };
          candleSeries.update(updated);
        } else {
          const prev = last ? last.close : j.price;
          candleSeries.update({ time: bSec, open: prev, high: j.price, low: j.price, close: j.price });
        }
        // update EMAs incrementally: recompute from current series (simple approach)
        const data = candleSeries.data(); const closes = data.map(d=> d.close); const times = data.map(d=> d.time);
        const f = ema(closes, j.ema_fast).map((y,i)=> ({ time: times[i], value: y }));
        const s = ema(closes, j.ema_slow).map((y,i)=> ({ time: times[i], value: y }));
        emaFastSeries.setData(f); emaSlowSeries.setData(s);
        // new order marker via SSE
        if (j.order){ pushOrderMarker(j.order, interval); }
      }catch(_){ }
    };
  }catch(_){ }

  // Toolbar actions: paper Buy/Sell to add markers; Reset to fit content
  const btnBuy = document.getElementById('btnBuy');
  const btnSell = document.getElementById('btnSell');
  const btnReset = document.getElementById('btnReset');
  const paperSel = document.getElementById('paperMode');
  const orderKrw = document.getElementById('orderKrw');
  const emaFast = document.getElementById('emaFast');
  const emaSlow = document.getElementById('emaSlow');
  const startBtn = document.getElementById('botStart');
  const stopBtn = document.getElementById('botStop');
  function postPaper(side){
    const last = candleSeries.dataByIndex(candleSeries.data().length-1, LightweightCharts.MismatchDirection.None);
    if (!last) return;
    const payload = { ts: (last.time*1000), side, price: last.close, size: 0.0001, paper: true };
    fetch(`${base}/api/order`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) }).catch(()=>{});
  }
  if (btnBuy) btnBuy.addEventListener('click', ()=> postPaper('BUY'));
  if (btnSell) btnSell.addEventListener('click', ()=> postPaper('SELL'));
  if (btnReset) btnReset.addEventListener('click', ()=> { chart.timeScale().fitContent(); });

  function pushCfg(){
    const body = {
      paper: paperSel ? (paperSel.value === 'true') : undefined,
      order_krw: orderKrw ? Number(orderKrw.value||5000) : undefined,
      ema_fast: emaFast ? Number(emaFast.value||10) : undefined,
      ema_slow: emaSlow ? Number(emaSlow.value||30) : undefined,
      candle: getInterval(),
    };
    Object.keys(body).forEach(k=> body[k] === undefined && delete body[k]);
    if (Object.keys(body).length === 0) return Promise.resolve();
    return fetch(`${base}/api/bot/config`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  }
  if (paperSel) paperSel.addEventListener('change', ()=>pushCfg());
  if (orderKrw) orderKrw.addEventListener('change', ()=>pushCfg());
  if (emaFast) emaFast.addEventListener('change', ()=>pushCfg());
  if (emaSlow) emaSlow.addEventListener('change', ()=>pushCfg());
  if (tfSelect) tfSelect.addEventListener('change', ()=>pushCfg());
  if (startBtn) startBtn.addEventListener('click', ()=>{ pushCfg().finally(()=>fetch(`${base}/api/bot/start`,{method:'POST'})); });
  if (stopBtn) stopBtn.addEventListener('click', ()=>{ fetch(`${base}/api/bot/stop`,{method:'POST'}); });
})();


