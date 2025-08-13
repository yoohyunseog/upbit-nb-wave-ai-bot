$(document).ready(function() {
    const minute = 10;
    const coin = 'KRW-BTC';
    const apiUrl = `./proxy_upbit.php?endpoint=candles&unit=${minute}`;
    const count = 200;
    let chartInitialized = false;
    let chart;
    let previousTrend = null;
    let supportLevel = null; // 지지선을 저장할 변수
    let isFirstPlay = true;
    let currentAudioIndex = 0;

    $('#title').text(`${coin} ${minute}분봉 차트`);

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

                const volumeNbMax = BIT_MAX_NB(nonNullVolumes, 5.5);
                const volumeNbMin = BIT_MIN_NB(nonNullVolumes, 5.5);
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
                analyzeResistanceBreakthrough($('#currentPrice').text());
              
                // 함수 호출
                supportLevel = addBtcAvgBuyPriceSupportLine();
                addAvgPriceLineToChart(supportLevel, labels);
              
                // 함수 호출
                updateTradingStrategyTable();

                const tableData = extractDataFromTable();

                const analyzedData = analyzeData(tableData, volumeNbMax, volumeNbMin, volumeUpDown);
                                     generateSummary(analyzedData, supportLevel, analyzedData.volumeNbMax, analyzedData.volumeNbMin, analyzedData.volumeUpDown);
                calculateAndAppendSum()
              
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
    const overallTrend = $('#aiDecisionTd').text().trim();
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
        url: 'https://www.xn--9l4b4xi9r.com/UpbitAssetsFetcher_Sell3.php',
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
        url: 'https://www.xn--9l4b4xi9r.com/UpbitAssetsFetcher_Buy2.php',
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
    const INTERVAL = 10000; // 10초를 10000ms로 설정
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

// 코드의 나머지 부분 (추가된 IndexedDB, audio, visualizer 관련 부분 포함)

function initIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("frequencyDataDB", 1);
    request.onupgradeneeded = function(event) {
      const db = event.target.result;
      db.createObjectStore("frequencyData", { keyPath: "url" });
    };
    request.onsuccess = function(event) {
      resolve(event.target.result);
    };
    request.onerror = function(event) {
      reject(event.target.error);
    };
  });
}

function saveFrequencyData(db, url, data) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(["frequencyData"], "readwrite");
    const store = transaction.objectStore("frequencyData");
    const request = store.put({ url, data });
    request.onsuccess = function() {
      resolve();
    };
    request.onerror = function(event) {
      reject(event.target.error);
    };
  });
}

function loadFrequencyData(db, url) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(["frequencyData"], "readonly");
    const store = transaction.objectStore("frequencyData");
    const request = store.get(url);
    request.onsuccess = function(event) {
      resolve(event.target.result ? event.target.result.data : null);
    };
    request.onerror = function(event) {
      reject(event.target.error);
    };
  });
}

$(document).ready(function() {
    var audioPlayer = $('#audioPlayer')[0];
    var audioTitle = $('#audioTitle');
    audioPlayer.volume = 1.0; // 볼륨을 최대값으로 설정
    var audioFiles = [];
    var imageFiles = [];
    var currentTrend = $('#trend3to8').text();
    var audioHref = 'https://www.xn--9l4b4xi9r.com/8bit/';
    var imageHref = 'https://www.xn--9l4b4xi9r.com/8bit/';
    var audioFilesUpdated = false;
    var imageFilesUpdated = false;
    var updatingImage = false; // 이미지 업데이트 상태
    var currentFrequencyData = [];
    var dataIndex = 0;
    var visualizationInterval;
    var animationId;
    var isFirstPlay = true;
    var currentAudioIndex = 0;
    var lastAudioIndex = -1; // 마지막 재생된 오디오 인덱스를 저장

    function updateFiles(trend, callback) {
        var folder = trend === '매도세' ? 'Down' : 'Up';

        // 오디오 파일 로드
        $.getJSON(audioHref + 'audio-files.php?folder=' + folder, function(response) {
            if (typeof response === 'object') {
                audioFiles = response.map(file => ({
                    file: decodeURIComponent(file.file),
                    title: decodeURIComponent(file.file).split('/').pop().replace(/\.[^/.]+$/, ""), // 파일명에서 제목 추출
                    spectrumDataFile: audioHref + 'frequency-file.php?file=' + encodeURIComponent('spectrum_' + decodeURIComponent(file.file).split('/').pop().replace(/\.[^/.]+$/, "") + ".txt")
                }));
                audioFilesUpdated = true;
                if (callback) callback();
            } else {
                console.error('Invalid JSON response:', response);
            }
        }).fail(function(jqxhr, textStatus, error) {
            console.error('오디오 파일 로드 오류:', textStatus, error);
        });

        // 이미지 파일 로드
        $.getJSON(imageHref + 'image-files.php?folder=' + folder, function(response) {
            if (typeof response === 'object') {
                imageFiles = response
                    .filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file)) // jpg, jpeg, png, gif 파일만 포함
                    .map(file => imageHref + decodeURIComponent(file));
                imageFilesUpdated = true;
                if (imageFilesUpdated && !updatingImage) {
                    updateRandomBackgroundImage();
                }
            } else {
                console.error('Invalid JSON response:', response);
            }
        }).fail(function(jqxhr, textStatus, error) {
            console.error('이미지 파일 로드 오류:', textStatus, error);
        });

        isFirstPlay = true; // 트렌드가 변할 때마다 isFirstPlay를 true로 설정
    }

    function getRandomIndex(length) {
        return Math.floor(Math.random() * length);
    }

    function playNextAudio() {
        // 모든 데이터 초기화
        clearInterval(visualizationInterval);
        cancelAnimationFrame(animationId);
        // 초기화 함수를 처음 한 번만 호출
        initializeCanvas();

        currentFrequencyData = [];
        dataIndex = 0;

        if (audioFiles.length > 0) {
            var index = -1;
            if (isFirstPlay) {
                currentAudioIndex = getRandomIndex(audioFiles.length);
                isFirstPlay = false;
            } else {
                do {
                    currentAudioIndex = getRandomIndex(audioFiles.length);
                } while (currentAudioIndex === lastAudioIndex || audioPlayer.src === audioHref + audioFiles[currentAudioIndex].file);
            }

            lastAudioIndex = currentAudioIndex;

            var index = currentAudioIndex;
            if (index >= 0 && index < audioFiles.length) {
                var audioSrc = audioHref + audioFiles[index].file;
                var spectrumDataFile = audioFiles[index].spectrumDataFile;
                var title = audioFiles[index].title;
                audioTitle.text(title); // 제목 업데이트

                audioPlayer.src = audioSrc;

                audioPlayer.play().then(() => {
                    // 주파수 데이터 시각화
                    initIndexedDB().then(db => {
                        loadFrequencyData(db, spectrumDataFile).then(cachedData => {
                            if (cachedData) {
                                currentFrequencyData = cachedData;
                                dataIndex = 0;
                                startVisualization();
                            } else {
                                $.get(spectrumDataFile, function(response) {
                                    currentFrequencyData = response.map(item => item.map(value => parseFloat(value)));
                                    saveFrequencyData(db, spectrumDataFile, currentFrequencyData).then(() => {
                                        dataIndex = 0;
                                        startVisualization();
                                    }).catch(error => {
                                        console.error('주파수 데이터 저장 오류:', error);
                                    });
                                }).fail(function() {
                                    console.error('주파수 데이터 로드 오류:', spectrumDataFile);
                                });
                            }
                        }).catch(error => {
                            console.error('주파수 데이터 로드 오류:', error);
                        });
                    }).catch(error => {
                        console.error('IndexedDB 초기화 오류:', error);
                    });
                }).catch(error => {
                    console.error('오디오 재생 오류:', error);
                });
            } else {
                console.error('잘못된 인덱스 접근:', index);
            }
        } else {
            console.warn('오디오 파일이 없습니다.');
        }
    }

    function startVisualization() {
        clearInterval(visualizationInterval);
        cancelAnimationFrame(animationId);
        const frameDuration = audioPlayer.duration / currentFrequencyData.length * 1000; // 프레임 간격 (밀리초 단위)
        visualizationInterval = setInterval(() => {
            if (audioPlayer.paused) {
                clearInterval(visualizationInterval);
                cancelAnimationFrame(animationId);
                return;
            }
            visualizeFrequencyData();
        }, frameDuration);
    }

    function visualizeFrequencyData() {
        if (dataIndex < currentFrequencyData.length) {
            drawFrequencyData(currentFrequencyData[dataIndex]);
            dataIndex++;
        } else {
            clearInterval(visualizationInterval); // 데이터가 끝나면 인터벌 정지
            cancelAnimationFrame(animationId);
        }
    }

    let canvas, ctx, stage, parent, centerX, centerY, lights, circle;
    let width, height;
    let initialized = false;
    let lastUpdateTime = 0;
    const updateInterval = 10000; // 10초마다 초기화

    function initializeCanvas() {
        canvas = document.getElementById('frequencyVisualizer');
        ctx = canvas.getContext('2d');
        stage = document.querySelector('.stage');
        parent = document.querySelector('.custom-chart-container');
        width = canvas.width = parent.clientWidth;
        height = canvas.height = parent.clientHeight;
        centerX = width / 2;
        centerY = height / 2;
        lights = document.getElementsByClassName('light');
        circle = document.querySelector('.circle');

        const numBars = 30; // 조명의 수 설정
        while (lights.length < numBars) {
            const light = document.createElement('div');
            light.className = 'light';
            stage.appendChild(light);
        }
        while (lights.length > numBars) {
            stage.removeChild(lights[lights.length - 1]);
        }

        initialized = true;
    }

    function clearAnimation() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
    function getRandomPosition(range) {
        const randomX = Math.random() * width;
        const randomY = Math.random() * range; // Y 좌표를 상단 부분으로 제한
        return { x: randomX, y: randomY };
    }

function drawFrequencyData(frequencyData) {
    if (!initialized) {
        initializeCanvas();
    }

    // 차트 캔버스 요소를 가져오기
    const chartCanvas = document.getElementById('krwBtcChart');
    const chartCanvasRect = chartCanvas.getBoundingClientRect();

    ctx.clearRect(0, 0, width, height);

    const numBars = 30; // 조명의 수 설정
    const notes = ['도', '레', '미', '파', '솔', '라', '시']; // 음계 배열
    const avgValue = frequencyData.reduce((acc, val) => acc + val, 0) / frequencyData.length;
    const range = Math.max(200, Math.min(width, height) * (avgValue / 255) * 6); // 리듬에 따라 위치 범위 6배로 조정
    const speed = (avgValue / 255) * 1.2 + 0.3; // 리듬에 따라 속도 조정 (0.3초 ~ 1.5초 범위)

    // Set the transition speed dynamically based on the average value
    for (let i = 0; i < numBars; i++) {
        lights[i].style.transitionDuration = `${speed}s`;
        lights[i].style.willChange = 'transform, width, height, background-color'; // 성능 최적화
    }
    circle.style.transitionDuration = `${speed}s`;
    circle.style.willChange = 'transform, width, height, background-color'; // 성능 최적화

    function getRandomPosition(range) {
        const randomX = Math.random() * width;
        const randomY = Math.random() * range; // Y 좌표를 상단 부분으로 제한
        return { x: randomX, y: randomY };
    }

    function updateLights() {
        const currentTime = Date.now();
        if (currentTime - lastUpdateTime > updateInterval) {
            // 일정 시간 간격으로 초기화
            initializeCanvas();
            lastUpdateTime = currentTime;
        }

        ctx.clearRect(0, 0, width, height); // 캔버스를 초기화

        const trend = $('#trend3to8').text(); // 현재 트렌드를 가져옴

        for (let i = 0; i < numBars; i++) {
            const value = frequencyData[i * Math.floor(frequencyData.length / numBars)];
            const size = 20 + (value / 255) * 80; // 조명의 크기를 20~100px 범위로 조정

            let randomPosition;
            do {
                randomPosition = getRandomPosition(range);
            } while (randomPosition.x > chartCanvasRect.left && randomPosition.x < chartCanvasRect.right &&
                     randomPosition.y > chartCanvasRect.top && randomPosition.y < chartCanvasRect.bottom);

            const offsetX = randomPosition.x;
            const offsetY = height - randomPosition.y - size; // Y 좌표를 상단 부분으로 이동

            lights[i].style.width = `${size}px`;
            lights[i].style.height = `${size}px`;
            lights[i].style.transform = `translate(${offsetX - size / 2}px, ${offsetY - size / 2}px)`; // 중앙에 맞추기 위해 크기의 절반을 빼줌

            const note = notes[i % notes.length];
            lights[i].innerText = note;

            switch (note) {
                case '도':
                    lights[i].style.backgroundColor = `rgba(255, 0, 0, 0.8)`; // 빨간색
                    break;
                case '레':
                    lights[i].style.backgroundColor = `rgba(255, 165, 0, 0.8)`; // 밝은 주황색
                    break;
                case '미':
                    lights[i].style.backgroundColor = `rgba(255, 255, 0, 0.8)`; // 밝은 노란색
                    break;
                case '파':
                    lights[i].style.backgroundColor = `rgba(144, 238, 144, 0.8)`; // 밝은 연두색
                    break;
                case '솔':
                    lights[i].style.backgroundColor = `rgba(0, 255, 255, 0.8)`; // 밝은 청록색
                    break;
                case '라':
                    lights[i].style.backgroundColor = `rgba(173, 216, 230, 0.8)`; // 밝은 하늘색
                    break;
                case '시':
                    lights[i].style.backgroundColor = `rgba(255, 192, 203, 0.8)`; // 밝은 분홍색
                    break;
            }

            lights[i].style.animation = `fall ${speed}s linear infinite`;
        }

        const circleSize = 100 + (avgValue / 255) * 200; // 원의 크기를 100~300px 범위로 조정
        let randomCirclePosition;
        do {
            randomCirclePosition = getRandomPosition(range);
        } while (randomCirclePosition.x > chartCanvasRect.left && randomCirclePosition.x < chartCanvasRect.right &&
                 randomCirclePosition.y > chartCanvasRect.top && randomCirclePosition.y < chartCanvasRect.bottom);

        const randomCircleX = randomCirclePosition.x;
        const randomCircleY = height - randomCirclePosition.y - circleSize; // Y 좌표를 상단 부분으로 이동

        circle.style.width = `${circleSize}px`;
        circle.style.height = `${circleSize}px`;

        circle.style.transform = `translate(${randomCircleX - circleSize / 2}px, ${randomCircleY - circleSize / 2}px) scale(${1 + avgValue / 255})`; // 중심 위치 유지 및 크기 변환

        circle.innerText = '도';
        circle.style.background = `radial-gradient(circle, rgba(255, 0, 0, 0.8) 0%, rgba(0, 255, 0, 0.5) 70%)`; // 빨간색과 녹색 그라디언트

        circle.style.animation = `fall ${speed}s linear infinite`;

        animationId = requestAnimationFrame(updateLights);
    }

    clearAnimation();
    animationId = requestAnimationFrame(updateLights);

    const brightness = 50 + (avgValue / 255) * 50; // 밝기 값을 50~100% 범위로 조정
    const blur = (avgValue / 255) * 10; // 블러 값을 0~10px 범위로 조정

    stage.style.filter = `brightness(${brightness}%) blur(${blur}px)`;
}

    function updateRandomBackgroundImage() {
        if (imageFiles.length > 0) {
            updatingImage = true; // 이미지 업데이트 시작
            var randomIndex = getRandomIndex(imageFiles.length);
            var imageUrl = imageFiles[randomIndex];
            $('#imgContainer').css('background-image', 'url("' + imageUrl + '")');
            setTimeout(function() {
                updatingImage = false; // 이미지 업데이트 종료
                updateRandomBackgroundImage();
            }, 10000); // 10초마다 이미지 변경
        }
    }

    $('#playAudioButton').click(function() {
        updateFiles(currentTrend, function() {
            playNextAudio();
        });
        setInterval(function() {
            var newTrend = $('#trend3to8').text();
            if (newTrend !== currentTrend) {
                currentTrend = newTrend;
                updateFiles(currentTrend);
            }
        }, 1000); // 1초마다 trend3to8 값을 확인
        setInterval(function() {
            updateFiles(currentTrend);
        }, 60000); // 1분마다 파일 목록 업데이트
    });

    audioPlayer.addEventListener('pause', function() {
        clearInterval(visualizationInterval);
        cancelAnimationFrame(animationId);
    });

    audioPlayer.addEventListener('play', function() {
        startVisualization();
    });

    audioPlayer.addEventListener('ended', function() {
        playNextAudio(); // 현재 곡이 끝나면 새로운 곡 재생
    });

    audioPlayer.addEventListener('seeked', function() {
        const currentTime = audioPlayer.currentTime;
        dataIndex = Math.floor((currentTime / audioPlayer.duration) * currentFrequencyData.length);
    });

    window.addEventListener('resize', function() {
        clearInterval(visualizationInterval);
        cancelAnimationFrame(animationId);
        if (currentFrequencyData.length > 0) {
            startVisualization();
        }
    });
});
