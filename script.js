$(document).ready(function() {
    const minute = 10;
    const coin = 'KRW-BTC';
    // Use server proxy to avoid CORS and mitigate 429
    const apiUrl = `./proxy_upbit.php?endpoint=candles&unit=${minute}`;
    const count = 200;
    let chartInitialized = false;
    let chart;
    let previousTrend = null;
    let supportLevel = null; // 지지선을 저장할 변수
    // audio removed

    $('#title').text(`${coin} ${minute}분봉 차트`);

    // Theme toggle
    $('#themeToggle').on('click', function(){
        if (document.documentElement.classList.contains('light-theme')) {
            document.documentElement.classList.remove('light-theme');
        } else {
            document.documentElement.classList.add('light-theme');
        }
    });

    // Compact toggle: hide heavy tables for quick glance
    $('#compactToggle').on('click', function(){
        if (document.documentElement.classList.contains('compact')) {
            document.documentElement.classList.remove('compact');
            $(this).text('Compact');
        } else {
            document.documentElement.classList.add('compact');
            $(this).text('Expanded');
        }
    });

    function processDataAndUpdateChart(emaData, dataProcessingFunction, priceChart, tradeName, color, type, opacity) {
        const { xValues, yValues } = separateXYProperties(emaData);
        const processedPoints = dataProcessingFunction(xValues, yValues);
        const dateArray = processedPoints['risingTimes'] || processedPoints['minimumTimes'];
        const valueArray = processedPoints['arrValue'];
        let combinedArray = {
            data: valueArray,
            times: dateArray
        };
        const movingAverage = combinedArray.data;
        const timeAverage = combinedArray.times;
        updateOrAddPoints(priceChart, tradeName, timeAverage.map((time, index) => ({ x: time, y: movingAverage[index] })), color, type, opacity);
        return {
            movingAverage, 
            timeAverage
        }
    }

    function separateXYProperties(dataArray) {
        let xValues = [];
        let yValues = [];
        for (let i = 0; i < dataArray.length; i++) {
            xValues.push(dataArray[i].x);
            yValues.push(dataArray[i].y);
        }
        return { xValues, yValues };
    }

    function updateOrAddPoints(chart, label, data, color, type, opacity) {
        let dataset = chart.data.datasets.find(ds => ds.label === label);
        if (!dataset) {
            dataset = {
                label: label,
                data: data,
                borderColor: color,
                borderWidth: 2,
                pointRadius: 3,
                fill: false,
                tension: 0.1,
                type: type,
                showLine: true,
                hidden: false,
                backgroundColor: color,
                pointBackgroundColor: color,
                pointBorderColor: color
            };
            chart.data.datasets.push(dataset);
        } else {
            dataset.data = data;
            dataset.backgroundColor = color;
            dataset.borderColor = color;
            dataset.borderWidth = 2;
            dataset.pointBackgroundColor = color;
            dataset.pointBorderColor = color;
        }
        chart.update();
    }

    function initializeChart(labels) {
        const ctx = document.getElementById('krwBtcChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: []
            },
            options: {
                animation: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            tooltipFormat: 'yyyy-MM-dd HH:mm',
                            displayFormats: {
                                minute: 'MMM d, h:mm a'
                            }
                        },
                        title: {
                            display: true,
                            color: "#CCCCCC",
                            text: '시간'
                        },
                        ticks: {
                            autoSkip: true,
                            color: "#CCCCCC",
                            maxTicksLimit: 20
                        }
                    },
                    y: {
                        type: 'linear',
                        ticks: {
                            color: "#CCCCCC"
                        },
                        title: {
                            display: true,
                            text: '가격',
                            color: "#CCCCCC"
                        },
                        position: 'right'
                    }
                },
                plugins: {
                    legend: {
                        onClick(e, legendItem) {
                            const index = legendItem.datasetIndex;
                            const ci = this.chart;
                            const meta = ci.getDatasetMeta(index);
                            meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                            ci.update();
                        },
                        labels: {
                            color: '#FFFFFF'
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'xy',
                            onPan({ chart }) { /* console.log(`Panning!`); */ }
                        },
                        zoom: {
                            enabled: true,
                            mode: 'xy',
                            wheel: { enabled: true },
                            pinch: { enabled: true },
                            onZoom({ chart }) { /* console.log(`Zooming!`); */ }
                        }
                    },
                    tooltip: {
                        titleColor: '#CCCCCC',
                        bodyColor: '#CCCCCC',
                        footerColor: '#CCCCCC',
                        enabled: false,
                        mode: 'nearest',
                        intersect: false
                    }
                }
            }
        });
    }

    function addLineToChart(label, data, color, hidden = false) {
        chart.data.datasets.push({
            label,
            data,
            borderColor: color,
            borderWidth: 2,
            pointRadius: 3,
            fill: false,
            tension: 0.1,
            type: 'line',
            showLine: true,
            hidden
        });
        chart.update();
    }

    function updateLineInChart(index, data) {
        chart.data.datasets[index].data = data;
        chart.update();
    }

    function calculateEMA(prices, period) {
        const k = 2 / (period + 1);
        let emaArray = [prices[0]];
        prices.reduce((prev, curr) => {
            const ema = curr * k + prev * (1 - k);
            emaArray.push(ema);
            return ema;
        });
        return emaArray;
    }

    function filterSparseData(dataArray) {
        let count = 0;
        for (let i = 0; i < dataArray.length; i++) {
            if (dataArray[i] !== null) {
                count++;
                if (count <= 2) {
                    dataArray[i] = null;
                }
            } else {
                count = 0;
            }
        }
    }

    function findKeyPoints(data, labels) {
        const nonNullData = data.filter(price => price !== null);
        const nonNullLabels = labels.filter((label, index) => data[index] !== null);
        if (nonNullData.length === 0) return { points: [], times: [] };
        const maxPrice = Math.max(...nonNullData);
        const startPoint = nonNullData.findIndex(price => price === maxPrice);
        const endPoint = 0;
        const halfMidPoint = Math.round(((endPoint - startPoint) / 2) + startPoint);
        const points = [nonNullData[startPoint], nonNullData[halfMidPoint], nonNullData[endPoint]];
        const times = [nonNullLabels[startPoint], nonNullLabels[halfMidPoint], nonNullLabels[endPoint]];
        return { points, times };
    }

    function findKeyPointsLowestStart(data, labels) {
        const nonNullData = data.filter(price => price !== null);
        const nonNullLabels = labels.filter((label, index) => data[index] !== null);
        if (nonNullData.length === 0) return { points: [], times: [] };
        const minPrice = Math.min(...nonNullData);
        const startPoint = nonNullData.findIndex(price => price === minPrice);
        const endPoint = 0;
        const halfMidPoint = Math.round(((endPoint - startPoint) / 2) + startPoint);
        const points = [nonNullData[startPoint], nonNullData[halfMidPoint], nonNullData[endPoint]];
        const times = [nonNullLabels[startPoint], nonNullLabels[halfMidPoint], nonNullLabels[endPoint]];
        return { points, times };
    }

    function lastcalculateSpecialSectionLengthWithTimeIntervalsRevised(dates, values) {
        let risingTimes = [];
        let arrValue = [];
        let array = [];
        let count = values.length - 1;
        for (let i = values.length - 1; i >= 0; i--) {
            if (values[i] > values[i + 1]) {
                if (Math.max(...array) < values[i]) {
                    risingTimes.push(dates[i]);
                    arrValue.push(values[i]);
                    count = i;
                }
                array.push(values[i]);
            }
        }
        for (var a = 0; a < 50; a++) {
            if (count > 0) {
                array = [];
                for (let i = count; i >= 0; i--) {
                    if (values[i] < values[i + 1]) {
                        if (Math.min(...array) > values[i]) {
                            risingTimes = [];
                            arrValue = [];
                            count = i;
                        }
                        array.push(values[i]);
                    }
                }
                array = [];
                for (let i = count; i >= 0; i--) {
                    if (values[i] > values[i + 1]) {
                        if (Math.max(...array) < values[i]) {
                            risingTimes.push(dates[i]);
                            arrValue.push(values[i]);
                            count = i;
                        }
                        array.push(values[i]);
                    }
                }
            }
            if (count <= 0) {
                break;
            }
        }
        const uniqueRisingTimes = [...new Set(risingTimes)];
        const uniqueArrValue = [...new Set(arrValue)];
        return {
            risingTimes: uniqueRisingTimes,
            arrValue: uniqueArrValue
        };
    }

    function calculateLastFallSectionWithRateOfChange(dates, values) {
        let minimumTimes = [];
        let arrValue = [];
        let array = [];
        var count = values.length - 1;
        for (let i = values.length - 1; i >= 0; i--) {
            if (values[i] > values[i + 1]) {
                if (Math.max(...array) < values[i]) {
                    count = i;
                }
                array.push(values[i]);
            }
        }
        array = [];
        for (let i = count; i >= 0; i--) {
            if (values[i] < values[i + 1]) {
                if (Math.min(...array) > values[i]) {
                    minimumTimes.push(dates[i]);
                    arrValue.push(values[i]);
                    count = i;
                }
                array.push(values[i]);
            }
        }
        for (var a = 0; a < 50; a++) {
            array = [];
            for (let i = count; i >= 0; i--) {
                if (values[i] > values[i + 1]) {
                    if (Math.max(...array) < values[i]) {
                        minimumTimes = [];
                        arrValue = [];
                        count = i;
                    }
                    array.push(values[i]);
                }
            }
            array = [];
            for (let i = count; i >= 0; i--) {
                if (values[i] < values[i + 1]) {
                    if (Math.min(...array) > values[i]) {
                        minimumTimes.push(dates[i]);
                        arrValue.push(values[i]);
                        count = i;
                    }
                    array.push(values[i]);
                }
            }
            if (count <= 0) {
                break;
            }
        }
        const uniqueMinimumTimes = [...new Set(minimumTimes)];
        const uniqueArrValue = [...new Set(arrValue)];
        return {
            minimumTimes: uniqueMinimumTimes,
            arrValue: uniqueArrValue
        };
    }

    function createXYArray(ema, labels) {
        return ema.map((value, index) => ({
            x: labels[index],
            y: value
        }));
    }

    function fetchDataAndUpdateChart() {
        $.ajax({
            url: apiUrl,
            type: "GET",
            async: false,
            data: { market: coin, count },
            success: function(data) {
                const labels = data.map(item => new Date(item.timestamp));
                const prices = data.map(item => item.trade_price);
                // KPI update
                const latest = prices[0];
                $('#kpiCurrentPrice').text(latest ? latest.toLocaleString() : '-');
                const volumes = data.map(item => item.candle_acc_trade_volume);
                let previousPrice = prices[0];
                const risingPrices = [];
                const fallingPrices = [];
                const risingTimes = [];
                const fallingTimes = [];
                prices.forEach((price, index) => {
                    if (price >= previousPrice) {
                        risingPrices.push(price);
                        risingTimes.push(labels[index]);
                        fallingPrices.push(null);
                        fallingTimes.push(null);
                    } else {
                        risingPrices.push(null);
                        risingTimes.push(null);
                        fallingPrices.push(price);
                        fallingTimes.push(labels[index]);
                    }
                    previousPrice = price;
                });
                filterSparseData(risingPrices);
                filterSparseData(fallingPrices);
              
                const nonNullVolumes = volumes.filter(vol => vol !== null);

                const volumeNbMax = BIT_MAX_NB(nonNullVolumes, BIT_MIN_NB(nonNullVolumes, 5.5)) ;
                const volumeNbMin = BIT_MIN_NB(nonNullVolumes, BIT_MAX_NB(nonNullVolumes, 5.5)) * getSliderValues() / 2;
                const volumeUpDown = volumeNbMax > volumeNbMin ? 'Up' : 'Down';

                // 비율 계산
                const totalVolume = volumeNbMax + volumeNbMin;
                const volumeNbMaxRatio = totalVolume > 0 ? (volumeNbMax / totalVolume * 100).toFixed(2) + '%' : 'N/A';
                const volumeNbMinRatio = totalVolume > 0 ? (volumeNbMin / totalVolume * 100).toFixed(2) + '%' : 'N/A';

                $('#volumeNbMax').text(volumeNbMax);
                $('#volumeNbMin').text(volumeNbMin);
                $('#volumeUpDown').text(volumeUpDown);

                // 비율 업데이트
                $('#volumeNbMaxRatio').text(volumeNbMaxRatio);
                $('#volumeNbMinRatio').text(volumeNbMinRatio);
              
                $('#volumeNbMaxRatio').text(volumeNbMaxRatio);
                $('#volumeNbMinRatio').text(volumeNbMinRatio);

                      
                // Extract the text values
                const totalBuyRatioText = $('#totalBuyRatio').text();
                const totalSellRatioText = $('#totalSellRatio').text();

                // Remove the '%' symbol and convert to float
                const totalBuyRatio = parseFloat(totalBuyRatioText.replace('%', ''));
                const totalSellRatio = parseFloat(totalSellRatioText.replace('%', ''));
              
                let Ratio = 0;
                let volumeNbMax100 = 0;
                let volumeNbMin100 = 0;
                let volumeUpDown100 = false;
              
                if(totalBuyRatio > totalSellRatio) {
                  Ratio = totalBuyRatio - totalSellRatio;
                  volumeNbMax100 = (volumeNbMax * Ratio);
                  volumeNbMin100 = volumeNbMin;
                } else {
                  Ratio = totalSellRatio - totalBuyRatio;
                  volumeNbMin100 = (volumeNbMin * Ratio);
                  volumeNbMax100 = volumeNbMax;
                }
              
                if(volumeNbMax100 > volumeNbMin100) {
                  volumeUpDown100 = 'Up';
                } else {
                  volumeUpDown100 = 'Down';
                }
                
                // 볼륨 값이 숫자가 아닌 경우 기본 자리 표시자로 설정
                if (isNaN(volumeNbMax100) || isNaN(volumeNbMin100)) {
                    volumeNbMax100 = '...';
                    volumeNbMin100 = '...';
                    volumeUpDown100 = '...';
                    // 콘솔에 변수 출력
                    console.log('totalBuyRatio:', totalBuyRatio);
                    console.log('totalSellRatio:', totalSellRatio);
                    console.log('Ratio:', Ratio);
                    console.log('100volumeNbMax:', volumeNbMax);
                    console.log('100volumeNbMin:', volumeNbMin);
                    console.log('100volumeUpDown:', volumeUpDown);
                    
                }
              
                  $('#100volumeNbMax').text(volumeNbMax);
                  $('#100volumeNbMin').text(volumeNbMin);
                  $('#100volumeUpDown').text(volumeUpDown);
                  $('#100volumeNbMaxRatio').text(volumeNbMaxRatio);
                  $('#100volumeNbMinRatio').text(volumeNbMinRatio);
              
                const ema10 = calculateEMA(prices, 10);
                const ema30 = calculateEMA(prices, 30);
                const ema60 = calculateEMA(prices, 60);
                if (!chartInitialized) {
                    initializeChart(labels);
                    addLineToChart('상승 가격', risingPrices.map((price, index) => ({ x: risingTimes[index], y: price })), 'rgba(0, 255, 0, 1)');
                    addLineToChart('하락 가격', fallingPrices.map((price, index) => ({ x: fallingTimes[index], y: price })), 'rgba(255, 0, 0, 1)');
                    addLineToChart('종가', prices.map((price, index) => ({ x: labels[index], y: price })), 'rgb(75, 192, 192)', true);
                    addLineToChart('EMA (10)', ema10.map((price, index) => ({ x: labels[index], y: price })), 'rgba(255, 120, 0, 0.1)');
                    addLineToChart('EMA (30)', ema30.map((price, index) => ({ x: labels[index], y: price })), 'rgba(255, 110, 0, 0.1)');
                    addLineToChart('EMA (60)', ema60.map((price, index) => ({ x: labels[index], y: price })), 'rgba(255, 100, 0, 0.1)');
                    chartInitialized = true;
                } else {
                    chart.data.labels = labels;
                    updateLineInChart(0, risingPrices.map((price, index) => ({ x: risingTimes[index], y: price })));
                    updateLineInChart(1, fallingPrices.map((price, index) => ({ x: fallingTimes[index], y: price })));
                    updateLineInChart(2, prices.map((price, index) => ({ x: labels[index], y: price })));
                    updateLineInChart(3, ema10.map((price, index) => ({ x: labels[index], y: price })));
                    updateLineInChart(4, ema30.map((price, index) => ({ x: labels[index], y: price })));
                    updateLineInChart(5, ema60.map((price, index) => ({ x: labels[index], y: price })));
                }
                const keyPoints = findKeyPoints(risingPrices, labels);
                const additionalLine1 = keyPoints.points;
                const additionalLabels = keyPoints.times;
                if (!chart.data.datasets[6]) {
                    chart.data.datasets.push({
                        label: '추가 라인 1',
                        data: additionalLine1.map((price, index) => ({ x: additionalLabels[index], y: price })),
                        borderColor: 'Coral',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: false,
                        tension: 0.1,
                        type: 'line',
                        showLine: true
                    });
                } else {
                    updateLineInChart(6, additionalLine1.map((price, index) => ({ x: additionalLabels[index], y: price })));
                }
                const keyPoints2 = findKeyPointsLowestStart(fallingPrices, labels);
                const additionalLine2 = keyPoints2.points;
                const additionalLabels2 = keyPoints2.times;
                if (!chart.data.datasets[7]) {
                    chart.data.datasets.push({
                        label: '추가 라인 2',
                        data: additionalLine2.map((price, index) => ({ x: additionalLabels2[index], y: price })),
                        borderColor: 'Pink',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: false,
                        tension: 0.1,
                        type: 'line',
                        showLine: true
                    });
                } else {
                    updateLineInChart(7, additionalLine2.map((price, index) => ({ x: additionalLabels2[index], y: price })));
                }
                const ema10XY = createXYArray(ema10, labels);
                const ema30XY = createXYArray(ema30, labels);
                const ema60XY = createXYArray(ema60, labels);
                
                // 기존 라인 3 ~ 8 데이터 처리 및 테이블 업데이트
                const line3Data = processDataAndUpdateChart(ema10XY, calculateLastFallSectionWithRateOfChange, chart, '추가 라인 3 (EMA10)', 'Lavender', 'line', 2.5);
                const line4Data = processDataAndUpdateChart(ema30XY, calculateLastFallSectionWithRateOfChange, chart, '추가 라인 4 (EMA30)', 'Lavender', 'line', 2.5);
                const line5Data = processDataAndUpdateChart(ema60XY, calculateLastFallSectionWithRateOfChange, chart, '추가 라인 5 (EMA60)', 'Lavender', 'line', 2.5);
                const line6Data = processDataAndUpdateChart(ema10XY, lastcalculateSpecialSectionLengthWithTimeIntervalsRevised, chart, '추가 라인 6 (EMA10)', 'Purple', 'line', 2.5);
                const line7Data = processDataAndUpdateChart(ema30XY, lastcalculateSpecialSectionLengthWithTimeIntervalsRevised, chart, '추가 라인 7 (EMA30)', 'Purple', 'line', 2.5);
                const line8Data = processDataAndUpdateChart(ema60XY, lastcalculateSpecialSectionLengthWithTimeIntervalsRevised, chart, '추가 라인 8 (EMA60)', 'Purple', 'line', 2.5);
                
                updateTableForLine(3, line3Data.movingAverage);
                updateTableForLine(4, line4Data.movingAverage);
                updateTableForLine(5, line5Data.movingAverage);
                updateTableForLine(6, line6Data.movingAverage);
                updateTableForLine(7, line7Data.movingAverage);
                updateTableForLine(8, line8Data.movingAverage);
                function updateTableData(lineIndex, max, min, trend, resistanceCount, goldenCross, deadCross) {
                    $(`.additionalLine${lineIndex}-nb-max`).text(max);
                    $(`.additionalLine${lineIndex}-nb-min`).text(min);
                    $(`.additionalLine${lineIndex}-trend`).text(trend);
                    $(`.additionalLine${lineIndex}-resistance`).text(resistanceCount);
                    $(`.additionalLine${lineIndex}-golden-cross`).text(goldenCross);
                    $(`.additionalLine${lineIndex}-dead-cross`).text(deadCross);
                }

                function analyzeLineData(data, higherValue) {
                    const nonNullData = data.filter(point => point.y !== null);
                    if (nonNullData.length === 0) {
                        return { max: '-', min: '-', trend: '-', resistanceCount: '-', goldenCross: '-', deadCross: '-' };
                    }

                    const max = Math.max(...nonNullData.map(point => point.y));
                    const min = Math.min(...nonNullData.map(point => point.y));
                    
                    const NB_MAX = BIT_MAX_NB(nonNullData.map(point => point.y), 5.5);
                    const NB_MIN = BIT_MIN_NB(nonNullData.map(point => point.y), 5.5);
                  
                    const trend = NB_MAX > NB_MIN ? '상승' : '하락';

                    const resistanceCount = calculateResistanceCount(nonNullData);
                    const { goldenCross, deadCross } = calculateCrosses(nonNullData);

                    return { max, min, trend, resistanceCount, goldenCross, deadCross };
                }

                function calculateResistanceCount(data) {
                    let intervals = [];
                    for (let i = 1; i < data.length; i++) {
                        intervals.push(new Date(data[i].x) - new Date(data[i - 1].x));
                    }

                    if (intervals.length < 1) {
                        return 0;
                    }

                    let totalInterval = intervals.reduce((sum, interval) => sum + interval, 0);
                    let averageInterval = totalInterval / intervals.length;
                    let threshold = averageInterval * 2;

                    let resistancePoints = 0;
                    for (let i = 0; i < intervals.length; i++) {
                        let intervalBefore = intervals[i];
                        let intervalAfter = i < intervals.length - 1 ? intervals[i + 1] : null;

                        if (intervalBefore > threshold) {
                            resistancePoints++;
                        }
                    }

                    return resistancePoints;
                }

                function calculateCrosses(data) {
                    let goldenCross = 0;
                    let deadCross = 0;
                    for (let i = 1; i < data.length; i++) {
                        if (data[i - 1].y < data[i].y && data[i - 2] && data[i - 2].y > data[i - 1].y) {
                            goldenCross++;
                        }
                        if (data[i - 1].y > data[i].y && data[i - 2] && data[i - 2].y < data[i - 1].y) {
                            deadCross++;
                        }
                    }
                    return { goldenCross, deadCross };
                }


                function updateTableForLine(lineIndex, lineData) {
                    const max = BIT_MAX_NB(lineData, bit = 5.5);
                    const min = BIT_MIN_NB(lineData, bit = 5.5);
                    const higherValue = max > min ? "Up" : "Down";

                    $(`.additionalLine${lineIndex}-nb-max-value`).text(max);
                    $(`.additionalLine${lineIndex}-nb-min-value`).text(min);
                    $(`.additionalLine${lineIndex}-higher-value`).text(higherValue);
                }

                for (let i = 1; i <= 8; i++) {
                    const lineData = chart.data.datasets[5 + i].data;
                    const { max, min, trend, resistanceCount, goldenCross, deadCross } = analyzeLineData(lineData);
                    updateTableData(i, max, min, trend, resistanceCount, goldenCross, deadCross);
                }

                // 자동 지지선 계산 및 추가
                supportLevel = calculateSupportLevel(prices);
                updateSupportLineTable(supportLevel);
                addSupportLineToChart(supportLevel, labels);
              
                // 저항선 돌파 여부 계산 및 업데이트
                analyzeResistanceBreakthrough(parseFloat($('#currentPrice').text()));
              
                // 함수 호출
                supportLevel = addBtcAvgBuyPriceSupportLine();
                addAvgPriceLineToChart(supportLevel, labels);
              
                // 함수 호출
                updateTradingStrategyTable();

                const tableData = extractDataFromTable();

                const analyzedData = analyzeData(tableData, volumeNbMax, volumeNbMin, volumeUpDown);
                generateSummary(analyzedData, supportLevel, analyzedData.volumeNbMax, analyzedData.volumeNbMin, analyzedData.volumeUpDown);
                // KPI mirrored values + Overall snapshot row
                $('#kpiBuyRatio').text(analyzedData.totalBuyRatio + '%');
                $('#kpiSellRatio').text(analyzedData.totalSellRatio + '%');
                $('#kpiResistanceCount').text(analyzedData.totalResistanceCount);
                const trendTxt = $('#trend3to8').text() || '-';
                const badge = $('#kpiTrendBadge');
                badge.text(trendTxt);
                badge.css('background', trendTxt === '매수세' ? 'linear-gradient(180deg,#2dce89,#28a745)' : 'linear-gradient(180deg,#ff7b7b,#dc3545)');
                $('#lastUpdateTs').text(new Date().toLocaleTimeString());

                // Simple property summary table
                 $('#simpleCurrentPrice').text(latest ? latest.toLocaleString() : '-');
                $('#simpleBuyRatio').text(analyzedData.totalBuyRatio + '%');
                $('#simpleSellRatio').text(analyzedData.totalSellRatio + '%');
                $('#simpleResistanceCount').text(analyzedData.totalResistanceCount);
                $('#simpleTrend').text(trendTxt);
                calculateAndAppendSum()
                // Update Trade System live metrics
                updateTradeSystemLive(latest);
              
            },
            error: function(xhr, status, error) {
                console.error("데이터를 가져오는 중 오류 발생:", status, error);
            }
        });
    }

    // 지지선 테이블 업데이트 함수
    function updateSupportLineTable(supportLevel) {
        $('#supportLevel').text(supportLevel);
    }
  
    // 자동 지지선 계산 함수
    function calculateSupportLevel(prices) {
        const recentPrices = prices.slice(-50); // 최근 50개 가격 데이터만 사용
        const minPrice = Math.min(...recentPrices);
        return minPrice;
    }

    // 차트에 지지선 추가 함수
    function addSupportLineToChart(supportLevel, labels) {
        const supportLineData = labels.map(label => ({ x: label, y: supportLevel }));
        const supportLineIndex = chart.data.datasets.findIndex(dataset => dataset.label === '지지선');

        if (supportLineIndex === -1) {
            chart.data.datasets.push({
                label: '지지선',
                data: supportLineData,
                borderColor: 'yellow',
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0,
                type: 'line',
                showLine: true
            });
        } else {
            updateLineInChart(supportLineIndex, supportLineData);
        }
        chart.update();
    }

    function addAvgPriceLineToChart(avgPriceLevel, labels) {
        const avgPriceLineData = labels.map(label => ({ x: label, y: avgPriceLevel }));
        const avgPriceLineIndex = chart.data.datasets.findIndex(dataset => dataset.label === '매수 평균가');

        if (avgPriceLineIndex === -1) {
            chart.data.datasets.push({
                label: '매수 평균가',
                data: avgPriceLineData,
                borderColor: 'green',
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0,
                type: 'line',
                showLine: true
            });
        } else {
            updateLineInChart(avgPriceLineIndex, avgPriceLineData);
        }
        chart.update();
    }

  function addBtcAvgBuyPriceSupportLine() {
    if (!chart || !chart.data.labels) {
      console.error("차트가 초기화되지 않았거나 데이터 라벨을 사용할 수 없습니다.");
      return;
    }

    // 테이블에서 BTC 평균 매수가를 가져오는 부분
    const btcRow = $('#cryptoAssetsTable tbody tr').filter((index, row) => {
      return $(row).find('td').eq(1).text().trim() === 'BTC';
    });

    const btcAvgBuyPrice = parseFloat(btcRow.find('td').eq(4).text().trim());
    return btcAvgBuyPrice;
  }


    fetchDataAndUpdateChart();
    setInterval(fetchDataAndUpdateChart, 60000);
});

document.addEventListener('DOMContentLoaded', function () {
  const container = document.getElementById('slider-container');
  if (!container) return;
  // 단일 민감도 슬라이더
  const wrapper = document.createElement('div');
  wrapper.classList.add('slider-wrapper');
  const label = document.createElement('label');
  label.textContent = 'Volume Sensitivity:';
  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = 0; slider.max = 20; slider.value = 20; slider.id = 'slider-1';
  const output = document.createElement('span'); output.id = 'value-1'; output.textContent = slider.value;
  slider.addEventListener('input', () => { output.textContent = slider.value; });
  const minusButton = document.createElement('button'); minusButton.textContent = '-'; minusButton.onclick = () => { if (slider.value > slider.min) { slider.value--; output.textContent = slider.value; } };
  const plusButton = document.createElement('button'); plusButton.textContent = '+'; plusButton.onclick = () => { if (slider.value < slider.max) { slider.value++; output.textContent = slider.value; } };
  wrapper.appendChild(label); wrapper.appendChild(minusButton); wrapper.appendChild(slider); wrapper.appendChild(plusButton); wrapper.appendChild(output);
  container.appendChild(wrapper);
});

// 슬라이더 값 반환 함수
function getSliderValues() {
  const slider = document.getElementById('slider-1');
  if (!slider) return 20;
  return parseInt(slider.value, 10);
}

const historicalData = JSON.parse(localStorage.getItem('historicalData')) || [];
var previousTrend = null; // 전역 변수로 정의

function recordHistoricalData(totalBuyRatio, totalSellRatio, totalResistanceCount, currentPrice) {
    const currentTime = new Date().toLocaleTimeString();

    const record = {
        time: currentTime,
        buy: totalBuyRatio,
        sell: totalSellRatio,
        resistanceCount: totalResistanceCount,
        currentPrice: currentPrice // 현재 가격 추가
    };

    historicalData.push(record);

    // historicalData 길이가 200을 초과하면 첫 번째 요소 제거
    if (historicalData.length > 200) {
        historicalData.splice(0, historicalData.length - 200);
    }

    // localStorage에 historicalData 저장
    localStorage.setItem('historicalData', JSON.stringify(historicalData));

    // historicalDataTable 초기화
    $('#historicalDataTable').empty();

    // historicalDataTable 데이터 추가 (최신 데이터가 맨 위로)
    historicalData.slice().reverse().forEach(data => {
        const row = `<tr>
            <td>${data.time}</td>
            <td>${data.buy}%</td>
            <td>${data.sell}%</td>
            <td>${data.resistanceCount}</td>
            <td>${data.currentPrice}</td> <!-- 현재 가격 추가 -->
        </tr>`;
        $('#historicalDataTable').append(row);
    });

    // 지지선 계산 및 업데이트
    calculateAndUpdateSupportLine();
}

function calculateAndUpdateSupportLine() {
    const prices = historicalData.map(data => data.currentPrice);
    const supportLevel = calculateSupportLevel(prices);
    $('#supportLevel').text(supportLevel.toFixed(2));
}

// 자동 지지선 계산 함수
function calculateSupportLevel(prices) {
    const recentPrices = prices.slice(-50); // 최근 50개 가격 데이터만 사용
    const minPrice = Math.min(...recentPrices);
    return minPrice;
}

function updateTradingStrategyTable() {
    // lineAnalysisTable 데이터를 가져오기
    const lineAnalysisData = [];
    $('#lineAnalysisTable tr').each(function(index) {
        if (index >= 2 && index <= 9) { // 추가 라인 3 ~ 8까지의 데이터만 처리
            const cells = $(this).find('td');
            if (cells.length > 0) {
                let maxPrice = parseFloat(cells.eq(1).text().trim());
                lineAnalysisData.push({
                    line: cells.eq(0).text().trim(),
                    maxPrice: parseFloat(cells.eq(1).text().trim()),
                    minPrice: parseFloat(cells.eq(2).text().trim()),
                    maxValue: parseFloat(cells.eq(7).text().trim()), // N/B Max
                    minValue: parseFloat(cells.eq(8).text().trim()), // N/B Min
                    resistanceCount: parseInt(cells.eq(4).text().trim()) || 0, // 저항선 개수 (NaN일 경우 0)
                    higherValue: cells.eq(9).text().trim(), // 더 높은 값 (Up 또는 Down)
                    currentPrice: parseFloat($('#currentPrice').text().trim()) // 현재 가격 추가
                });

                // maxPrice가 null인 경우 higherValue를 0으로 설정
                if (isNaN(lineAnalysisData[lineAnalysisData.length - 1].maxPrice) || isNaN(lineAnalysisData[lineAnalysisData.length - 1].minPrice)) {
                    lineAnalysisData[lineAnalysisData.length - 1].higherValue = 0;
                }
            }
        }
    });

    // 현재 추세를 계산하는 함수
    function analyzeTrend(higherValue) {
        if (higherValue === 'Up') {
            return '상승';
        } else if (higherValue === 'Down') {
            return '하락';
        } else {
            return '알 수 없음'; // 예외 처리
        }
    }

    // 분석하여 매수와 매도 비율을 결정
    const strategyData = lineAnalysisData.map((data, index) => {
        const total = data.maxValue + data.minValue;
        let buyPercentage = (data.maxValue / total) * 100;
        let sellPercentage = (data.minValue / total) * 100;

        // 저항선이 있을 경우 비율 조정
        if (data.resistanceCount > 0) {
            const adjustmentFactor = data.resistanceCount * 0.1; // 저항선 개수에 따른 조정 비율 (10%씩 조정)

            if (index >= 0 && index <= 2) { // 추가 라인 3 ~ 5 (매도세 저항선)
                sellPercentage *= 1 + adjustmentFactor; // 매도 비율을 저항선 개수에 비례하여 증가
                buyPercentage *= 1 - adjustmentFactor; // 매수 비율을 저항선 개수에 비례하여 감소
            } else if (index >= 3 && index <= 5) { // 추가 라인 6 ~ 8 (매수세 저항선)
                buyPercentage *= 1 + adjustmentFactor; // 매수 비율을 저항선 개수에 비례하여 증가
                sellPercentage *= 1 - adjustmentFactor; // 매도 비율을 저항선 개수에 비례하여 감소
            }
        }

        const trend = analyzeTrend(data.higherValue); // 현재 추세 분석

        return {
            condition: data.line,
            buy: buyPercentage.toFixed(2), // 소수점 둘째 자리까지 반올림
            sell: sellPercentage.toFixed(2), // 소수점 둘째 자리까지 반올림
            resistanceCount: data.resistanceCount,
            trend: trend, // 분석한 추세
            currentPrice: data.currentPrice // 현재 가격 추가
        };
    });

    // tradingStrategyTable 데이터 추가
    $('#tradingStrategyTable').empty();

    // tradingStrategyTable 데이터 추가
    strategyData.forEach(strategy => {
        const row = `<tr>
            <td>${strategy.condition}</td>
            <td>${strategy.buy}%</td>
            <td>${strategy.sell}%</td>
            <td>${strategy.resistanceCount}</td>
            <td>${strategy.currentPrice}</td> <!-- 현재 가격 추가 -->
        </tr>`;
        $('#tradingStrategyTable').append(row);
    });

    // 전체 비율 계산
    const totalBuy = strategyData.reduce((sum, strategy) => sum + parseFloat(strategy.buy), 0);
    const totalSell = strategyData.reduce((sum, strategy) => sum + parseFloat(strategy.sell), 0);
    const totalResistanceCount = strategyData.reduce((sum, strategy) => sum + strategy.resistanceCount, 0);

    const totalBuyRatio = (totalBuy / strategyData.length).toFixed(2);
    const totalSellRatio = (totalSell / strategyData.length).toFixed(2);
    const currentPrice = strategyData[0].currentPrice;

    // overallStrategyTable 업데이트
    $('#totalBuyRatio').text(`${totalBuyRatio}%`);
    $('#totalSellRatio').text(`${totalSellRatio}%`);
    $('#totalResistanceCount').text(totalResistanceCount);

    // 기록 함수 호출
    recordHistoricalData(totalBuyRatio, totalSellRatio, totalResistanceCount, currentPrice);

    // 라인 3 ~ 8 추세 분석
    const trendCounts3to5 = analyzeTrendCounts(strategyData.slice(0, 3), '매도세');
    const trendCounts6to8 = analyzeTrendCounts(strategyData.slice(3, 6), '매수세');
    const overallTrendCounts = analyzeOverallTrend(trendCounts3to5, trendCounts6to8);

    const tdr = (overallTrendCounts.buyRisingCount + overallTrendCounts.buyFallingCount) > 
                (overallTrendCounts.sellRisingCount + overallTrendCounts.sellFallingCount) ? '매수세' : '매도세';

    $('#trend3to8').text(tdr);

    // Check if trend has changed and reset the prices accordingly
    if (previousTrend && previousTrend !== tdr) {
        if (tdr === '매수세') {
            highestBuyPrice = 0; // Reset highest buy price
        } else if (tdr === '매도세') {
            lowestSellPrice = 0; // Reset lowest sell price
        }
    }

    previousTrend = tdr; // Update previous trend

    $('#buyRisingCount').text(overallTrendCounts.buyRisingCount);
    $('#buyFallingCount').text(overallTrendCounts.buyFallingCount);
    $('#sellRisingCount').text(overallTrendCounts.sellRisingCount);
    $('#sellFallingCount').text(overallTrendCounts.sellFallingCount);

    let resistanceStatus = '';
    if (tdr === '매수세') {
        resistanceStatus = overallTrendCounts.buyRisingCount > overallTrendCounts.buyFallingCount ? '저항선 형성' : '저항선 붕괴';
    } else {
        resistanceStatus = overallTrendCounts.sellRisingCount > overallTrendCounts.sellFallingCount ? '저항선 형성' : '저항선 붕괴';
    }
    $('#resistanceStatus').text(resistanceStatus);
}


function analyzeTrendCounts(trendData, type) {
    let buyRisingCount = 0;
    let buyFallingCount = 0;
    let sellRisingCount = 0;
    let sellFallingCount = 0;
    trendData.forEach(data => {
        if (data.trend === '상승') {
            if (type === '매수세') {
                buyRisingCount++;
            } else if (type === '매도세') {
                sellRisingCount++;
            }
        } else if (data.trend === '하락') {
            if (type === '매수세') {
                buyFallingCount++;
            } else if (type === '매도세') {
                sellFallingCount++;
            }
        }
    });
    return { buyRisingCount, buyFallingCount, sellRisingCount, sellFallingCount };
}

function analyzeOverallTrend(trendCounts3to5, trendCounts6to8) {
    const totalBuyRisingCount = trendCounts6to8.buyRisingCount;
    const totalBuyFallingCount = trendCounts6to8.buyFallingCount;
    const totalSellRisingCount = trendCounts3to5.sellRisingCount;
    const totalSellFallingCount = trendCounts3to5.sellFallingCount;

    const totalTrend = (totalBuyRisingCount + totalBuyFallingCount) > (totalSellRisingCount + totalSellFallingCount) ? '상승' : '하락';

    return {
        totalTrend,
        buyRisingCount: totalBuyRisingCount,
        buyFallingCount: totalBuyFallingCount,
        sellRisingCount: totalSellRisingCount,
        sellFallingCount: totalSellFallingCount
    };
}

function calculateBreakthroughStatus() {
    const recentData = historicalData.slice(-10); // 가장 최근의 10개 데이터 가져오기

    const totalBuy = recentData.reduce((sum, data) => sum + parseFloat(data.buy), 0);
    const totalSell = recentData.reduce((sum, data) => sum + parseFloat(data.sell), 0);
    const totalResistanceCount = recentData.reduce((sum, data) => sum + parseInt(data.resistanceCount), 0);

    const averageBuy = (totalBuy / recentData.length).toFixed(2);
    const averageSell = (totalSell / recentData.length).toFixed(2);
    const averageResistanceCount = (totalResistanceCount / recentData.length).toFixed(2);

    let breakthroughStatus = '';
    if (averageBuy > averageSell) {
        breakthroughStatus = '매수세 저항선 돌파';
    } else {
        breakthroughStatus = '매도세 저항선 돌파';
    }

    // breakthroughTable 업데이트
    $('#breakthroughStatus').text(breakthroughStatus);
    
    // 통합 메시지 div 생성 및 업데이트
    updateSummaryMessage(breakthroughStatus, averageBuy, averageSell, averageResistanceCount);
}

let highestBuyPrice = 0;
let lowestSellPrice = 0;

// 초기 예상 수익률 설정
let fallProfitPercent = 0;
let riseProfitPercent = 0;

function analyzeResistanceBreakthrough(currentPrice) {
    const recentData = historicalData.slice(-10); // 최근 10개의 데이터 가져오기

    const avgBuy = recentData.reduce((sum, data) => sum + parseFloat(data.buy), 0) / recentData.length;
    const avgSell = recentData.reduce((sum, data) => sum + parseFloat(data.sell), 0) / recentData.length;
    const avgResistanceCount = recentData.reduce((sum, data) => sum + parseInt(data.resistanceCount), 0) / recentData.length;

    let breakthroughStatus = '';
    // 전체 추세는 요약 테이블의 '라인 3 ~ 8 전체 추세' 값을 사용
    const overallTrend = $('#trend3to8').text().trim();
    const totalResistanceCount = Number($('#totalResistanceCount').text());
  
    // 테이블 값 가져오기
    const buyNbMaxSum_3_5 = parseFloat($('#buyNbMaxSum_3_5').text());
    const buyNbMinSum_3_5 = parseFloat($('#buyNbMinSum_3_5').text());
    const sellNbMaxSum_6_8 = parseFloat($('#sellNbMaxSum_6_8').text());
    const sellNbMinSum_6_8 = parseFloat($('#sellNbMinSum_6_8').text()); 
    const stateNonZero = $('#NonZero').text(); 
  
    // 각 값들의 합 계산
    const buySum = buyNbMaxSum_3_5 + buyNbMinSum_3_5;
    const sellSum = sellNbMaxSum_6_8 + sellNbMinSum_6_8;

    // 값들 중 하나라도 0이거나 합이 25를 넘는 경우 false를 반환하는 변수
    const allNonZeroAndWithinLimit = (buyNbMaxSum_3_5 !== 0 && buyNbMinSum_3_5 !== 0 && sellNbMaxSum_6_8 !== 0 && sellNbMinSum_6_8 !== 0) &&
                                     (buySum <= 25) && (sellSum <= 25) && stateNonZero === 'true';
  
    const volumeUpDown = $('#volumeUpDown').text();
    const volumeUpDown100 = $('#100volumeUpDown').text();
    
    if (overallTrend === '매수') {
        if (highestBuyPrice === 0 || currentPrice > highestBuyPrice) {
            highestBuyPrice = currentPrice;
            $('#highestBuyPrice').text(currentPrice);
            breakthroughStatus = '매수세 저항선 돌파';
            addBreakthroughTime('매수세 저항선 돌파', new Date().toLocaleString());
            fallProfitPercent = -0.01;
            riseProfitPercent += 0.01;
          
            if(allNonZeroAndWithinLimit) {
                if(buyNbMaxSum_3_5 > buyNbMinSum_3_5 && sellNbMaxSum_6_8 < sellNbMinSum_6_8 && volumeUpDown100 === 'Up') {
                  $('#aiDecision').text('매수 시도');
                } else {
                  $('#aiDecision').text('매수 실패');
                }
              } else {
                  $('#aiDecision').text('매수 조건 부적합');
              }
        } else {
            $('#highestBuyPrice').text(highestBuyPrice); // 갱신되지 않았더라도 표시
            breakthroughStatus = '매수세 저항선 저지';
            addBreakthroughTime('매수세 저항선 저지', new Date().toLocaleString());
            fallProfitPercent += 0.005;
            riseProfitPercent = -0.005;
        }
    } else if (overallTrend === '매도') {
        if (lowestSellPrice === 0 || currentPrice < lowestSellPrice) {
            lowestSellPrice = currentPrice;
            $('#lowestSellPrice').text(currentPrice);
            breakthroughStatus = '매도세 저항선 돌파';
            addBreakthroughTime('매도세 저항선 돌파', new Date().toLocaleString());
            fallProfitPercent += 0.01;
            riseProfitPercent = -0.01;
          
            if(allNonZeroAndWithinLimit) {
              if(buyNbMaxSum_3_5 < buyNbMinSum_3_5 && sellNbMaxSum_6_8 > sellNbMinSum_6_8 && volumeUpDown100 === 'Down') {
                $('#aiDecision').text('매도 시도');
              } else {
                $('#aiDecision').text('매도 실패');
              }
            } else {
                $('#aiDecision').text('매도 조건 부적합');
            }
        } else {
            $('#lowestSellPrice').text(lowestSellPrice); // 갱신되지 않았더라도 표시
            breakthroughStatus = '매도세 저항선 저지';
            addBreakthroughTime('매도세 저항선 저지', new Date().toLocaleString());
            fallProfitPercent = -0.005;
            riseProfitPercent += 0.005;
        }
    } else {
        breakthroughStatus = '저항선 돌파 없음';
    }
  
    $('#fallProfitPercent').text((fallProfitPercent).toFixed(5) + '%');
    $('#riseProfitPercent').text((riseProfitPercent).toFixed(5) + '%');
  
    // 돌파 여부 테이블 업데이트
    $('#breakthroughStatus').text(breakthroughStatus);
    
    // 통합 메시지 div 생성 및 업데이트
    updateSummaryMessage(breakthroughStatus, avgBuy.toFixed(2), avgSell.toFixed(2), avgResistanceCount.toFixed(2));

    return { avgBuy, avgSell, avgResistanceCount, breakthroughStatus };
}  

function updateSummaryMessage(breakthroughStatus, avgBuy, avgSell, avgResistanceCount) {
    const summaryMessage = `
        돌파 상태: ${breakthroughStatus}
        평균 매수: ${avgBuy}
        평균 매도: ${avgSell}
        평균 저항선 개수: ${avgResistanceCount}
    `;

    // 결과를 출력할 div 생성 또는 업데이트
    let resultDiv = $('#resultDiv');
    if (resultDiv.length === 0) {
        resultDiv = $('<div id="resultDiv"></div>');
        $('body').append(resultDiv);
    }
    resultDiv.html(summaryMessage.replace(/\n/g, '<br>'));
}


// 저항선 돌파 시점을 기록하는 함수
function addBreakthroughTime(breakthroughType, timestamp) {
    const row = `<tr>
        <td>${breakthroughType}</td>
        <td>${timestamp}</td>
    </tr>`;
    $('#breakthroughTimesTable').prepend(row);

    // 행 개수에 따라 div 높이 조절
    const tableRows = $('#breakthroughTimesTable tr').length;
    if (tableRows > 3) {
        $('#breakthroughTimesContainer').css('overflow-y', 'auto');
    } else {
        $('#breakthroughTimesContainer').css('overflow-y', 'visible');
    }
}

function executeSellRequest() {
    $.ajax({
        url: 'https://참소식.com/_8비트/UpbitAssetsFetcher_Sell3.php',
        type: 'POST',
        async: false,
        success: function(response) {
            console.log('매도세 저항선 돌파 시 AJAX 요청 성공:', response);
        },
        error: function(error) {
            console.error('매도세 저항선 돌파 시 AJAX 요청 실패:', error);
        }
    });
}

function executeBuyRequest(param) {
    $.ajax({
        url: 'https://참소식.com/_8비트/UpbitAssetsFetcher_Buy2.php',
        type: 'POST',
        async: false,
        success: function(response) {
            console.log('매수세 저항선 돌파 시 AJAX 요청 성공:', response);
        },
        error: function(error) {
            console.error('매수세 저항선 돌파 시 AJAX 요청 실패:', error);
        }
    });
}

// 주기적으로 데이터를 수집하고 저장
function collectAndSaveData() {
    const lineAnalysisData = [];
    $('#lineAnalysisTable tr').each(function(index) {
        if (index >= 2 && index <= 9) { // 추가 라인 3 ~ 8까지의 데이터만 처리
            const cells = $(this).find('td');
            if (cells.length > 0) {
                lineAnalysisData.push({
                    buy: parseFloat(cells.eq(7).text().trim()), // N/B Max
                    sell: parseFloat(cells.eq(8).text().trim()), // N/B Min
                    resistanceCount: parseInt(cells.eq(4).text().trim()) || 0 // 저항선 개수 (NaN일 경우 0)
                });
            }
        }
    });

    if (lineAnalysisData.length > 0) {
        historicalData.push(...lineAnalysisData);
        if (historicalData.length > 200) {
            historicalData.splice(0, historicalData.length - 200); // 최신 200개의 데이터만 유지
        }
        localStorage.setItem('historicalData', JSON.stringify(historicalData));
    }
}

// 테이블 데이터를 추출하는 함수
function extractDataFromTable() {
    const lineAnalysisData = [];
    $('#lineAnalysisTable tr').each(function(index) {
        if (index >= 2 && index <= 9) { // 추가 라인 3 ~ 8까지의 데이터만 처리
            const cells = $(this).find('td');
            if (cells.length > 0) {
                lineAnalysisData.push({
                    line: cells.eq(0).text().trim(),
                    maxPrice: parseFloat(cells.eq(1).text().trim()) || null,
                    minPrice: parseFloat(cells.eq(2).text().trim()) || null,
                    trend: cells.eq(3).text().trim(),
                    resistanceCount: parseInt(cells.eq(4).text().trim()) || 0,
                    goldenCross: parseInt(cells.eq(5).text().trim()) || 0,
                    deadCross: parseInt(cells.eq(6).text().trim()) || 0,
                    nbMax: parseFloat(cells.eq(7).text().trim()) || null,
                    nbMin: parseFloat(cells.eq(8).text().trim()) || null,
                    higherValue: cells.eq(9).text().trim()
                });
            }
        }
    });
    return lineAnalysisData;
}

// 데이터를 분석하는 함수
function analyzeData(lineAnalysisData, volumeNbMax, volumeNbMin, volumeUpDown) {
    let totalBuy = 0;
    let totalSell = 0;
    let totalResistanceCount = 0;
    let buyRisingCount = 0;
    let buyFallingCount = 0;
    let sellRisingCount = 0;
    let sellFallingCount = 0;

    lineAnalysisData.forEach(data => {
        const total = data.nbMax + data.nbMin;
        const buyPercentage = (data.nbMax / total) * 100;
        const sellPercentage = (data.nbMin / total) * 100;
        totalBuy += buyPercentage;
        totalSell += sellPercentage;
        totalResistanceCount += data.resistanceCount;

        if (data.trend === '상승') {
            buyRisingCount++;
        } else if (data.trend === '하락') {
            sellFallingCount++;
        }
    });

    const totalBuyRatio = (totalBuy / lineAnalysisData.length).toFixed(2);
    const totalSellRatio = (totalSell / lineAnalysisData.length).toFixed(2);
    const overallTrendCounts = {
        buyRisingCount,
        buyFallingCount,
        sellRisingCount,
        sellFallingCount
    };
  
    return {
        totalBuyRatio,
        totalSellRatio,
        totalResistanceCount,
        trend3to8: $('#trend3to8').text(),
        overallTrendCounts,
        volumeNbMax,
        volumeNbMin,
        volumeUpDown
    };
}

// 요약 내용을 생성하는 함수
function generateSummary(analyzedData, supportLevel, volumeNbMax, volumeNbMin, volumeUpDown) {
    const { totalBuyRatio, totalSellRatio, totalResistanceCount, trend3to8, overallTrendCounts } = analyzedData;

    // 비율 계산
    const totalVolume = volumeNbMax + volumeNbMin;
    const volumeNbMaxRatio = totalVolume > 0 ? (volumeNbMax / totalVolume * 100).toFixed(2) + '%' : 'N/A';
    const volumeNbMinRatio = totalVolume > 0 ? (volumeNbMin / totalVolume * 100).toFixed(2) + '%' : 'N/A';

    // 수익 계산 정보 가져오기
    const currentPrice = parseFloat($('#currentPrice').text());
    const averagePrice = parseFloat($('#averagePrice').text());
    const buyAmount = parseFloat($('#buyAmount').text());
    const fallProfitPercent = ((averagePrice - currentPrice) / averagePrice * 100).toFixed(2) + '%';
    const riseProfitPercent = ((currentPrice - averagePrice) / averagePrice * 100).toFixed(2) + '%';
    const priceStatus = currentPrice > averagePrice ? 'Up' : 'Down';
    const volatilityLevel = parseInt($('#volatilityLevel').text());

    let aiDecision = 'Hold';

    // 매수/매도 결정 조건
    if (trend3to8 === '매수세' && totalBuyRatio > totalSellRatio && volumeUpDown === 'Up') {
        aiDecision = '매수';
    } else if (trend3to8 === '매도세' && totalSellRatio > totalBuyRatio && volumeUpDown === 'Down') {
        aiDecision = '매도';
    } else if (trend3to8 === '매도세' && totalResistanceCount > 10 && volumeUpDown === 'Down') {
        aiDecision = '매도';
    } else if (trend3to8 === '매수세' && totalResistanceCount <= 10 && volumeUpDown === 'Up') {
        aiDecision = '매수';
    } else {
        aiDecision = '매수 조건 부적합';
    }

    let summary = `현재 비트코인 (KRW-BTC)은 전반적으로 `;
    summary += trend3to8 === '매도세' ? '매도세가 강하며' : '매수세가 강하며';

    // 추가 라인에서 상승 추세 확인 여부
    if (overallTrendCounts.buyRisingCount > 0) {
        summary += `, 추가 라인에서 상승 추세가 확인됩니다. `;
    } else if (overallTrendCounts.sellRisingCount > 0) {
        summary += `, 추가 라인에서 상승 추세는 없고 매도세가 강합니다. `;
    } else {
        summary += `, 추가 라인에서 뚜렷한 추세가 확인되지 않습니다. `;
    }

    // 매수 비율과 매도 비율 비교
    summary += `전체적으로 매도 비율(${totalSellRatio}%)이 매수 비율(${totalBuyRatio}%)보다 `;
    if (totalSellRatio > totalBuyRatio) {
        summary += `높고, `;
    } else if (totalSellRatio < totalBuyRatio) {
        summary += `낮고, `;
    } else {
        summary += `같으며, `;
    }
    
    // 저항선 개수에 따른 분석
    if (totalResistanceCount > 10) {
        summary += `저항선 개수가 ${totalResistanceCount}개로 많아 돌파가 어려운 상황입니다. `;
    } else if (totalResistanceCount > 0) {
        summary += `저항선 개수가 ${totalResistanceCount}개로 돌파 가능성이 존재합니다. `;
    } else {
        summary += `저항선이 거의 없으므로 상대적으로 돌파가 용이할 수 있습니다. `;
    }

    summary += `저항선 상태는 저항선 형성으로 분석됩니다.\n\n`;

    // 지지선 분석
    if (supportLevel) {
        summary += `현재 지지선은 ${supportLevel}입니다. `;
        summary += `현재 가격이 지지선 아래에 있다면 추가적인 하락 가능성을 염두에 두고 대비할 필요가 있습니다. `;
        summary += `지지선을 유지하면 반등의 기회를 노릴 수 있습니다.\n\n`;
    } else {
        summary += `현재 지지선 정보가 충분하지 않습니다.\n\n`;
    }

    // 볼륨 분석 추가
    summary += `볼륨 분석 결과, 현재 N/B Max는 ${volumeNbMax} (${volumeNbMaxRatio})이며 N/B Min은 ${volumeNbMin} (${volumeNbMinRatio})입니다. `;
    summary += `따라서 볼륨 트렌드는 ${volumeUpDown}입니다. `;

    // 추세에 따른 전략 제시
    summary += `이러한 상황을 고려할 때, 현재는 `;
    if (trend3to8 === '매도세' && volumeUpDown === 'Down') {
        summary += `매도세가 매우 강해 추가적인 하락을 대비한 전략이 필요해 보입니다. `;
        if (overallTrendCounts.sellFallingCount > 0) {
            summary += `특히, 하락세가 이어질 가능성이 높아 보입니다. `;
        }
    } else if (trend3to8 === '매수세' && volumeUpDown === 'Up') {
        summary += `매수세가 매우 강해 추가적인 상승을 대비한 전략이 필요해 보입니다. `;
        if (overallTrendCounts.buyRisingCount > 0) {
            summary += `상승세가 지속될 가능성이 높아 보입니다. `;
        }
    } else if (trend3to8 === '매도세' && volumeUpDown === 'Up') {
        summary += `가격은 하락세이나 거래대금이 증가하고 있어, 변동성에 대비한 신중한 접근이 필요합니다. `;
    } else if (trend3to8 === '매수세' && volumeUpDown === 'Down') {
        summary += `가격은 상승세이나 거래대금이 감소하고 있어, 조정 가능성에 유의해야 합니다. `;
    } else {
        summary += `현재 추세를 신중하게 지켜보며 전략을 수립해야 합니다. `;
    }

    summary += `특히, 강력한 저항선 돌파 여부를 주의 깊게 관찰해야 합니다.`;

    // 텍스트 요약 내용 출력
    const resultDiv = $('#resultDiv');
    if (resultDiv.length === 0) {
        $('<div id="resultDiv"></div>').appendTo('#analysisSummary');
    }
    $('#resultDiv').html(summary.replace(/\n/g, '<br>'));

    // 테이블 내용 추가
    const summaryTable = $('#summaryTable tbody');
    summaryTable.empty(); // 기존 테이블 내용 삭제
    const rows = [
        { item: '전체 추세', content: trend3to8 },
        { item: '매수 비율', content: totalBuyRatio + '%' },
        { item: '매도 비율', content: totalSellRatio + '%' },
        { item: '저항선 개수', content: totalResistanceCount },
        { item: '지지선', content: supportLevel || '정보 없음' },
        { item: 'N/B Max', content: volumeNbMax },
        { item: 'N/B Min', content: volumeNbMin },
        { item: 'N/B Max 비율', content: volumeNbMaxRatio },
        { item: 'N/B Min 비율', content: volumeNbMinRatio },
        { item: '볼륨 트렌드', content: volumeUpDown },
        { item: '현재 가격', content: currentPrice },
        { item: '평균 가격', content: averagePrice },
        { item: '매수 수량 (5,000원 기준)', content: buyAmount + ' BTC' },
        { item: '하락 시 예상 수익 (%)', content: fallProfitPercent },
        { item: '상승 시 예상 수익 (%)', content: riseProfitPercent },
        { item: '현재 가격 상태', content: priceStatus },
        { item: '변동률 등급', content: volatilityLevel },
        { item: '매수/매도 결정', content: aiDecision, id: 'aiDecisionTd' } // 고유 ID 추가
    ];

    rows.forEach(row => {
        const tr = $('<tr></tr>');
        const tdItem = $(`<td style="border: 1px solid #ffffff; padding: 8px;">${row.item}</td>`);
        const tdContent = $(`<td style="border: 1px solid #ffffff; padding: 8px;" id="${row.id || ''}">${row.content}</td>`);
        tr.append(tdItem).append(tdContent);
        summaryTable.append(tr);
    });
  
    return false;
}

function calculateAndAppendSum() {
    var buyNbMaxSum_3_5 = 0;
    var buyNbMinSum_3_5 = 0;
    var sellNbMaxSum_6_8 = 0;
    var sellNbMinSum_6_8 = 0;

    $('#lineAnalysisTable tr').each(function(index) {
        var trend = $(this).find('td').eq(3).text().trim();
        var nbMax = parseFloat($(this).find('td').eq(7).text().trim());
        var nbMin = parseFloat($(this).find('td').eq(8).text().trim());

        if (index >= 2 && index <= 4 && !isNaN(nbMax) && !isNaN(nbMin)) {
            if (trend === '상승' || trend === '하락') {
                buyNbMaxSum_3_5 += nbMax;
                buyNbMinSum_3_5 += nbMin;
            }
        } else if (index >= 5 && index <= 7 && !isNaN(nbMax) && !isNaN(nbMin)) {
            if (trend === '상승' || trend === '하락') {
                sellNbMaxSum_6_8 += nbMax;
                sellNbMinSum_6_8 += nbMin;
            }
        }
    });

    // 기존 테이블 값 가져오기
    const buyNbMaxSum_3_5_val = parseFloat($('#buyNbMaxSum_3_5').text()) || 0;
    const buyNbMinSum_3_5_val = parseFloat($('#buyNbMinSum_3_5').text()) || 0;
    const sellNbMaxSum_6_8_val = parseFloat($('#sellNbMaxSum_6_8').text()) || 0;
    const sellNbMinSum_6_8_val = parseFloat($('#sellNbMinSum_6_8').text()) || 0;

    // 계산된 값이 0이 아닌 경우에만 기존 값을 갱신
    const newBuyNbMaxSum_3_5 = buyNbMaxSum_3_5 !== 0 ? buyNbMaxSum_3_5 : buyNbMaxSum_3_5_val;
    const newBuyNbMinSum_3_5 = buyNbMinSum_3_5 !== 0 ? buyNbMinSum_3_5 : buyNbMinSum_3_5_val;
    const newSellNbMaxSum_6_8 = sellNbMaxSum_6_8 !== 0 ? sellNbMaxSum_6_8 : sellNbMaxSum_6_8_val;
    const newSellNbMinSum_6_8 = sellNbMinSum_6_8 !== 0 ? sellNbMinSum_6_8 : sellNbMinSum_6_8_val;

    // 각 값들의 합 계산
    const buySum = newBuyNbMaxSum_3_5 + newBuyNbMinSum_3_5;
    const sellSum = newSellNbMaxSum_6_8 + newSellNbMinSum_6_8;

    // 값들 중 하나라도 0이거나 합이 25를 넘는 경우 false를 반환하는 변수
    const allNonZeroAndWithinLimit = (newBuyNbMaxSum_3_5 !== 0 && newBuyNbMinSum_3_5 !== 0 && newSellNbMaxSum_6_8 !== 0 && newSellNbMinSum_6_8 !== 0) &&
                                     (buySum <= 31) && (sellSum <= 31);
    
    console.log('All values are non-zero and within limit:', allNonZeroAndWithinLimit);
  
    var newRow = '<tr>' +
        '<td id="buyNbMaxSum_3_5">' + newBuyNbMaxSum_3_5 + '</td>' +
        '<td id="buyNbMinSum_3_5">' + newBuyNbMinSum_3_5 + '</td>' +
        '<td id="sellNbMaxSum_6_8">' + newSellNbMaxSum_6_8 + '</td>' +
        '<td id="sellNbMinSum_6_8">' + newSellNbMinSum_6_8 + '</td>' +
        '<td id="NonZero">' + allNonZeroAndWithinLimit + '</td>' +
        '</tr>';
    $('#calculatedTable tbody').html(newRow);
}


function initializeArrays(count) {
  const arrays = ['BIT_START_A50', 'BIT_START_A100', 'BIT_START_B50', 'BIT_START_B100', 'BIT_START_NBA100'];
  const initializedArrays = {};
  arrays.forEach(array => {
    initializedArrays[array] = new Array(count);
  });
  return initializedArrays;
}

function calculateBit(nb, bit = 5.5, reverse = false) {
  if (nb.length < 2) {
    nb = [1, 2, 3, 4, 5, 6, 7, 8];
  }

  const BIT_NB = bit;
  const max = Math.max(...nb);
  const min = Math.min(...nb);
  const COUNT = 500;
  const CONT = 40;
  const range = max - min;
  const increment = range / (COUNT * nb.length - 1);
  const VIEW = BIT_NB / CONT;

  const arrays = initializeArrays(COUNT * nb.length);
  let count = 0;
  let totalSum = 0;

  for (let value of nb) {
    for (let i = 0; i < COUNT; i++) {
      const BIT_END = 1;
      const A50 = min + increment * (count + 1);
      const A100 = (count + 1) * BIT_NB / (COUNT * nb.length);
      const B50 = A50 - increment * 2;
      const B100 = A50 + increment;
      const NBA100 = A100 / (nb.length - BIT_END);

      arrays.BIT_START_A50[count] = A50;
      arrays.BIT_START_A100[count] = A100;
      arrays.BIT_START_B50[count] = B50;
      arrays.BIT_START_B100[count] = B100;
      arrays.BIT_START_NBA100[count] = NBA100;
      count++;
    }
    totalSum += value;
  }

  if (reverse) {
    arrays.BIT_START_NBA100.reverse();
  }

  let NB50 = 0;
  for (let value of nb) {
    for (let a = 0; a < arrays.BIT_START_NBA100.length; a++) {
      if (arrays.BIT_START_B50[a] < value && arrays.BIT_START_B100[a] > value) {
        NB50 += arrays.BIT_START_NBA100[Math.min(a, arrays.BIT_START_NBA100.length - 1)];
        break;
      }
    }
  }

  const BIT = Math.max((10 - nb.length) * 10, 1);
  const averageRatio = (totalSum / (nb.length * max)) * 100;
  NB50 = Math.min((NB50 / 100) * averageRatio, BIT_NB);

  return NB50;
}

function BIT_MAX_NB(nb, bit = 5.5) {
  return calculateBit(nb, bit, false);
}

function BIT_MIN_NB(nb, bit = 5.5) {
  return calculateBit(nb, bit, true);
}

$(document).ready(function() {
    const MINIMUM_BUY_PRICE = 5000; // 최소 매수 단위
    const INTERVAL = 30000; // 10초를 10000ms로 설정
    const API_URL = "./proxy_upbit.php?endpoint=ticker&market=KRW-BTC";
    let priceHistory = [];
    let volatilityHistory = [];

    // 로컬 스토리지에서 volatilityHistory 불러오기
    function loadVolatilityHistory() {
        const storedHistory = localStorage.getItem('volatilityHistory');
        if (storedHistory) {
            volatilityHistory = JSON.parse(storedHistory);
        }
    }

    // 로컬 스토리지에 volatilityHistory 저장하기
    function saveVolatilityHistory() {
        localStorage.setItem('volatilityHistory', JSON.stringify(volatilityHistory));
    }

    // 실제 가격을 가져오는 함수
    function fetchCurrentPrice() {
        return $.ajax({
            url: API_URL,
            method: "GET",
            async: false,
            dataType: "json"
        });
    }

    // 가격 히스토리 평균 계산 함수
    function calculateAveragePrice() {
        const sum = priceHistory.reduce((a, b) => a + b, 0);
        return sum / priceHistory.length;
    }

    // 변동률 계산 함수
    function calculateVolatility() {
        if (priceHistory.length < 2) return 0;
        const changes = [];
        for (let i = 1; i < priceHistory.length; i++) {
            changes.push(Math.abs(priceHistory[i] - priceHistory[i - 1]));
        }
        const totalChange = changes.reduce((a, b) => a + b, 0);
        return totalChange / (priceHistory.length - 1);
    }

    // 회귀 모델을 사용하여 변동률 등급 예측 함수
    function predictVolatilityGrade(volatility) {
        if (volatilityHistory.length >= 255) {
            volatilityHistory.shift(); // 255개를 초과하면 가장 오래된 값을 제거
        }
        volatilityHistory.push(volatility);
        saveVolatilityHistory(); // 저장

        // 학습 데이터
        const X = volatilityHistory.map((v, i) => [i, v]);
        const Y = volatilityHistory.map((v, i) => i);

        // 단순 선형 회귀 모델 학습
        const regression = ss.linearRegression(X);
        const slope = regression.m;
        const intercept = regression.b;

        // 새로운 변동률 등급 예측
        const predictedGrade = Math.round(slope * volatility + intercept);

        // 등급이 1과 10 사이에 있도록 제한
        const grade = Math.max(1, Math.min(10, predictedGrade));

        return grade;
    }

    // 매수/매도 결정 함수
    function aiTradingDecision(currentPrice, averagePrice, volatilityGrade, resistanceStatus) {
        const fallProfitPercent = (currentPrice - averagePrice) / averagePrice * 100;
        const riseProfitPercent = (averagePrice - currentPrice) / currentPrice * 100;
        const priceStatus = currentPrice < averagePrice ? 'Down' : 'Up';

        // 변동률 등급에 따라 임계값 설정
        const sellThreshold = volatilityGrade >= 8 ? -5 : -3;
        const buyThreshold = volatilityGrade >= 8 ? 5 : 3;

        if (fallProfitPercent <= sellThreshold && priceStatus === 'Down') {
            return 'Sell';
        } else if (riseProfitPercent >= buyThreshold && priceStatus === 'Up') {
            // 저항선 상태가 붕괴 상태이면 매수하지 않음
            if (resistanceStatus === '저항선 붕괴') {
                return 'Hold';
            }
            return 'Buy';
        } else {
            return 'Hold';
        }
    }

// 현재 가격 업데이트 함수
function updateCurrentPrice() {
    fetchCurrentPrice().done(function(data) {
        const currentPrice = data[0].trade_price;
        $('#currentPrice').text(currentPrice);

        // 가격 히스토리 업데이트
        if (priceHistory.length >= 255) {
            priceHistory.shift(); // 255개를 초과하면 가장 오래된 값을 제거
        }
        priceHistory.push(currentPrice);

        // 매수 수량 계산 및 업데이트
        const buyAmount = MINIMUM_BUY_PRICE / currentPrice;
        $('#buyAmount').text(buyAmount.toFixed(6) + ' BTC');
        // mirror to trade system price
        updateTradeSystemLive(currentPrice);

        // 가격 히스토리 평균 계산
        const averagePrice = calculateAveragePrice();
        $('#averagePrice').text(averagePrice.toFixed(2));

        // 평균 가격과 비교하여 가격 상태 업데이트
        if (currentPrice < averagePrice) {
            $('#priceStatus').text('Down');
        } else if (currentPrice > averagePrice) {
            $('#priceStatus').text('Up');
        } else {
            $('#priceStatus').text('Neutral');
        }

        // 변동률 계산 및 등급 업데이트
        const volatility = calculateVolatility();
        const volatilityGrade = predictVolatilityGrade(volatility);
        $('#volatilityLevel').text(volatilityGrade);

        // 매수/매도 결정
        const aiDecision = aiTradingDecision(currentPrice, averagePrice, volatilityGrade);
        //$('#aiDecision').text(aiDecision);
    });
}

    // 로컬 스토리지에서 데이터 불러오기
    loadVolatilityHistory();

    // 주기적으로 현재 가격 업데이트
    setInterval(updateCurrentPrice, 10000);
});

// audio/visualizer removed


// ---------------------------
// Bot UI Fetch + Chart (Chart.js already loaded globally)
// ---------------------------
(function(){
  let botChart;
  function ensureChart(){
    const ctx = document.getElementById('botChart');
    if (!ctx || botChart) return;
    botChart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Bot Price', data: [], borderColor: '#4cc9f0', borderWidth: 2, pointRadius: 0 }] },
      options: { animation:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{display:true, ticks:{color:'#ccc'}}} }
    });
  }
  function updateBotUI(){
    const botProto = (location.protocol === 'https:') ? 'https' : 'http';
    const botBase = `${botProto}://127.0.0.1:5057`;
    $.getJSON(`${botBase}/api/state`)
      .done(function(s){
        $('#botPrice').text(s.price ? s.price.toLocaleString() : '-');
        $('#botSignal').text(s.signal || '-');
        $('#botMarket').text(s.market || 'KRW-BTC');
        $('#botCandle').text(s.candle || 'minute10');
        $('#botEma').text(`${s.ema_fast}/${s.ema_slow}`);
        ensureChart();
        if (botChart && s.history && s.history.length){
          const labels = s.history.map(p=> new Date(p[0]));
          const data = s.history.map(p=> p[1]);
          botChart.data.labels = labels;
          botChart.data.datasets[0].data = data;
          botChart.update();
        }
      })
      .fail(function(){ /* ignore if server not running */ });
  }
  setInterval(updateBotUI, 5000);
  updateBotUI();
})();

// ---------------------------
// Trade System (Spot-like)
// ---------------------------
(function(){
  const storeKey = 'tradeSystemState_v1';

  function loadState() {
    try {
      const raw = localStorage.getItem(storeKey);
      if (!raw) return { balanceKrw: 1000000, balanceBtc: 0, avgPrice: 0, positionSide: 'FLAT', orders: [] };
      const parsed = JSON.parse(raw);
      return Object.assign({ balanceKrw: 1000000, balanceBtc: 0, avgPrice: 0, positionSide: 'FLAT', orders: [] }, parsed);
    } catch (e) {
      return { balanceKrw: 1000000, balanceBtc: 0, avgPrice: 0, positionSide: 'FLAT', orders: [] };
    }
  }

  function saveState(state) { localStorage.setItem(storeKey, JSON.stringify(state)); }

  function fmt(n, d=0) { if (Number.isNaN(n) || n === null || n === undefined) return '-'; return Number(n).toLocaleString(undefined, { maximumFractionDigits: d }); }
  function fmtBtc(n) { if (!n && n !== 0) return '-'; return Number(n).toFixed(8); }

  function render(state, markPrice) {
    $('#balanceKrw').text(fmt(state.balanceKrw));
    $('#balanceBtc').text(fmtBtc(state.balanceBtc));
    $('#avgPrice').text(state.avgPrice ? fmt(state.avgPrice) : '-');
    $('#positionSide').text(state.positionSide);

    const unrealized = state.balanceBtc > 0 && markPrice
      ? (markPrice - state.avgPrice) * state.balanceBtc
      : 0;
    const pnlEl = $('#unrealizedPnl');
    pnlEl.text(fmt(unrealized));
    pnlEl.removeClass('pnl-positive pnl-negative');
    if (unrealized > 0) pnlEl.addClass('pnl-positive');
    if (unrealized < 0) pnlEl.addClass('pnl-negative');

    const list = $('#orderHistory');
    list.empty();
    state.orders.slice().reverse().forEach(o => {
      const row = $('<div class="d-flex justify-content-between"/>');
      row.append(`<span>${o.side}</span>`);
      row.append(`<span>${fmt(o.price)}</span>`);
      row.append(`<span>${fmtBtc(o.size)}</span>`);
      row.append(`<span>${o.time}</span>`);
      list.append(row);
    });
  }

  let state = loadState();
  render(state);

  function nowStr(){ return new Date().toLocaleString(); }

  function placeOrder(side, amountKrw, markPrice) {
    if (!markPrice || markPrice <= 0) return;
    if (side === 'BUY') {
      const cost = Math.max(0, amountKrw);
      if (state.balanceKrw < cost) return;
      const size = cost / markPrice;
      // average price update
      const newPositionSize = state.balanceBtc + size;
      const newAvg = state.balanceBtc > 0 ? ((state.avgPrice * state.balanceBtc) + (markPrice * size)) / newPositionSize : markPrice;
      state.balanceKrw -= cost;
      state.balanceBtc += size;
      state.avgPrice = newAvg;
      state.positionSide = state.balanceBtc > 0 ? 'LONG' : 'FLAT';
      state.orders.push({ side: 'BUY', price: markPrice, size, krw: cost, time: nowStr() });
    } else if (side === 'SELL') {
      // sell up to available BTC equivalent of amountKrw
      const sizeToSell = Math.min(state.balanceBtc, amountKrw / markPrice);
      if (sizeToSell <= 0) return;
      const proceeds = sizeToSell * markPrice;
      state.balanceBtc -= sizeToSell;
      state.balanceKrw += proceeds;
      if (state.balanceBtc <= 1e-12) {
        state.balanceBtc = 0;
        state.avgPrice = 0;
        state.positionSide = 'FLAT';
      }
      state.orders.push({ side: 'SELL', price: markPrice, size: sizeToSell, krw: proceeds, time: nowStr() });
    }
    saveState(state);
    render(state, markPrice);
  }

  function closePosition(markPrice) {
    if (state.balanceBtc <= 0 || !markPrice) return;
    const size = state.balanceBtc;
    const proceeds = size * markPrice;
    state.balanceBtc = 0;
    state.balanceKrw += proceeds;
    state.avgPrice = 0;
    state.positionSide = 'FLAT';
    state.orders.push({ side: 'CLOSE', price: markPrice, size, krw: proceeds, time: nowStr() });
    saveState(state);
    render(state, markPrice);
  }

  // Wire buttons
  $(document).on('click', '#btnBuy', function(){
    const amount = parseFloat($('#orderAmount').val());
    const price = parseFloat($('#currentPrice').text());
    placeOrder('BUY', isNaN(amount)?0:amount, price);
  });
  $(document).on('click', '#btnSell', function(){
    const amount = parseFloat($('#orderAmount').val());
    const price = parseFloat($('#currentPrice').text());
    placeOrder('SELL', isNaN(amount)?0:amount, price);
  });
  $(document).on('click', '#btnClose', function(){
    const price = parseFloat($('#currentPrice').text());
    closePosition(price);
  });

  // Keep UI synced with live price
  window.updateTradeSystemLive = function(markPrice){
    if (!markPrice) return;
    render(state, markPrice);
  }
})();
