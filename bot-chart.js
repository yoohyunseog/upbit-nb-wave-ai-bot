$(function(){
  const ctx = document.getElementById('fullChart');
  const vctx = document.getElementById('volChart');
  const chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Price', data: [], borderColor: '#4cc9f0', borderWidth: 2, pointRadius: 0 },
      { label: 'EMA Fast', data: [], borderColor: 'rgba(14,203,129,0.9)', borderWidth: 1.5, pointRadius: 0 },
      { label: 'EMA Slow', data: [], borderColor: 'rgba(246,70,93,0.9)', borderWidth: 1.5, pointRadius: 0 },
    ]},
    options: {
      animation: false,
      plugins: { 
        legend: { labels: { color: '#e6eefc' } }, 
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(27,32,39,0.95)',
          titleColor: '#e6eefc',
          bodyColor: '#e6eefc',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => {
              const dsLabel = ctx.dataset.label || '';
              const v = typeof ctx.parsed.y === 'number' ? ctx.parsed.y : ctx.raw?.y;
              if (v != null) return `${dsLabel}: ${v.toLocaleString()}`;
              return dsLabel;
            }
          }
        },
        annotation: {
          annotations: {
            lastPrice: {
              type: 'line',
              yMin: 0, yMax: 0,
              borderColor: '#4cc9f0', borderWidth: 1, borderDash: [4,4],
              label: { enabled: true, content: '—', position: 'end', backgroundColor: '#2b3139', color: '#e6eefc' }
            }
          }
        },
        crosshair: {
          line: { color: 'rgba(255,255,255,0.3)', width: 1 },
          sync: { enabled: false },
          zoom: { enabled: false },
          callbacks: {
            beforeZoom: () => false,
            afterZoom: () => {}
          }
        },
        zoom: { 
          zoom: { 
            wheel: { enabled: false },     // 휠 줌 비활성화
            pinch: { enabled: false },     // 핀치 줌 비활성화
            drag: { enabled: false },      // 드래그 줌 비활성화
            mode: 'x' 
          }, 
          pan: { enabled: true, mode: 'x', modifierKey: null, overScaleMode: 'x', threshold: 0 } // 클릭 드래그로 좌우 이동만
        } 
      },
      scales: {
        x: { type: 'time', time: { unit: 'minute' }, ticks: { color: '#9bb0d1' } },
        y: { position: 'right', ticks: { color: '#9bb0d1' } }
      },
      parsing: false
    }
  });
  // 볼륨 차트 숨김 (라인 전용)
  $('#volChart').hide();

  const proto = (location.protocol === 'https:') ? 'https' : 'http';
  const base = `${proto}://127.0.0.1:5057`;
  function bucketTs(tsMs, interval){
    const d = new Date(tsMs);
    if (interval.startsWith('minute')){
      const m = parseInt(interval.replace('minute',''), 10) || 1;
      const bucket = Math.floor(tsMs / (m*60*1000)) * (m*60*1000);
      return bucket;
    }
    if (interval === 'day'){
      d.setHours(0,0,0,0); return d.getTime();
    }
    if (interval === 'minute60'){
      const bucket = Math.floor(tsMs / (60*60*1000)) * (60*60*1000);
      return bucket;
    }
    return tsMs;
  }
  function recomputeEMAs(periodFast, periodSlow){
    const ohlc = chart.data.datasets[0].data;
    if (!ohlc.length) return;
    const labels = ohlc.map(c=> c.x);
    const closes = ohlc.map(c=> c.c);
    function emaSeries(values, period){
      const k = 2/(period+1); const out=[]; let prev = values[0];
      for (let i=0;i<values.length;i++){
        const v = (i===0)? values[0] : values[i]*k + prev*(1-k); out.push(v); prev=v;
      }
      return out;
    }
    const f = emaSeries(closes, periodFast);
    const s = emaSeries(closes, periodSlow);
    chart.data.datasets[1].data = f.map((v,i)=> ({x: labels[i], y: v}));
    chart.data.datasets[2].data = s.map((v,i)=> ({x: labels[i], y: v}));
  }
  // Seed once
  function ema(arr, period){ const k=2/(period+1); const out=[]; let prev=arr[0]; for(let i=0;i<arr.length;i++){ const v=(i===0)?arr[0]:arr[i]*k+prev*(1-k); out.push(v); prev=v;} return out; }
  function seed(interval){
    $.getJSON(`${base}/api/ohlcv?interval=${interval}&count=300`).done(res => {
      const data = res.data || [];
      const labels = data.map(d=> new Date(d.time));
      chart.data.labels = labels;
      const closes = data.map(d=> d.close);
      chart.data.datasets[0].data = closes.map((c,i)=> ({x:labels[i], y:c}));
      chart.data.datasets[1].data = closes.map((c,i)=> ({x:labels[i], y: ema(closes,10)[i]}));
      chart.data.datasets[2].data = closes.map((c,i)=> ({x:labels[i], y: ema(closes,30)[i]}));
      // Last price annotation
      const last = closes[closes.length-1];
      chart.options.plugins.annotation.annotations.lastPrice.yMin = last;
      chart.options.plugins.annotation.annotations.lastPrice.yMax = last;
      chart.options.plugins.annotation.annotations.lastPrice.label.content = last.toLocaleString();
      chart.update();
    });
  }
  // initial seed
  seed('minute10');

  // Live via SSE
  const es = new EventSource(`${base}/api/stream`);
  es.onmessage = (e) => {
    try{
      const j = JSON.parse(e.data);
      $('#meta').text(`${j.market} ${j.candle} | signal: ${j.signal} | EMA ${j.ema_fast}/${j.ema_slow}`);
      $('#s_price').text(j.price.toLocaleString());
      $('#s_signal').text(j.signal).attr('class', j.signal==='BUY'?'buy':'sell');
      $('#s_ticker').text(j.market);
      $('#s_interval').text(j.candle);
      $('#s_ema').text(`${j.ema_fast}/${j.ema_slow}`);
      // update last-price line
      chart.options.plugins.annotation.annotations.lastPrice.yMin = j.price;
      chart.options.plugins.annotation.annotations.lastPrice.yMax = j.price;
      chart.options.plugins.annotation.annotations.lastPrice.label.content = j.price.toLocaleString();
      // append line point
      const ds = chart.data.datasets[0].data;
      ds.push({ x: new Date(j.ts), y: j.price });
      if (ds.length > 1000) ds.shift();
      recomputeEMAs(j.ema_fast, j.ema_slow);
      chart.update('none');
    }catch(err){}
  };

  // Toolbar controls
  $('#btnReset').on('click', ()=>{ chart.resetZoom(); });
  $('#btnZoomIn').on('click', ()=>{ chart.zoom(1.2); });
  $('#btnZoomOut').on('click', ()=>{ chart.zoom(0.8); });
  $('#timeframe').on('change', function(){ seed(this.value); });
});


