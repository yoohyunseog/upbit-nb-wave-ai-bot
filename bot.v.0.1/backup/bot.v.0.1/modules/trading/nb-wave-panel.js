// ===== NB Wave Panel Module =====
/**
 * NB Wave Panel Module
 * 
 * - N/B Wave 차트 그리기
 * - 실시간 데이터 업데이트
 * - 구역별 색상 표시
 * - 거래량 라인 표시
 */

class NbWavePanel {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.currentData = null;
        this.isInitialized = false;
    }

    // 초기화
    init() {
        this.canvas = document.getElementById('nb-wave-chart');
        if (!this.canvas) {
            console.error('NB Wave chart canvas not found');
            return false;
        }
        
        // 캔버스 크기를 컨테이너에 맞게 동적으로 설정
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        this.canvas.width = containerWidth;
        this.canvas.height = 200;
        
        this.ctx = this.canvas.getContext('2d');
        this.isInitialized = true;
        console.log('NB Wave Panel initialized with width:', containerWidth);
        return true;
    }

    // HTML 생성
    static generateHTML() {
        return `
            <div class="nb-wave-panel" style="margin-top:14px;">
                <h3>N/B Wave Map</h3>
                <canvas id="nb-wave-chart" height="200"></canvas>
            </div>
        `;
    }

    // 차트 그리기
    drawChart(nbData) {
        if (!this.isInitialized) {
            if (!this.init()) {
                return;
            }
        }
        
        if (!nbData) {
            console.warn('No NB wave data available');
            return;
        }

        this.currentData = nbData;
        
        // N/B Wave 데이터를 전역으로 저장 (300개 zone)
        window.sharedNbWaveData = {
            zones: nbData.zones || [],
            summary: nbData.summary || {},
            labels: nbData.labels || [],
            last_update: new Date().toISOString(),
            timestamp: new Date().getTime()
        };
        
        console.log('💾 N/B Wave data saved to global storage:', window.sharedNbWaveData);
        
        // 분봉별 구역 업데이트
        if (typeof updateTimeframeZones === 'function') {
            updateTimeframeZones();
        }
        
        // 캔버스 크기를 다시 설정 (반응형 대응)
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        this.canvas.width = containerWidth;
        this.canvas.height = 200;
        
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // 배경 클리어
        ctx.fillStyle = 'rgba(0, 0, 0, 0.95)';
        ctx.fillRect(0, 0, width, height);

        const labels = nbData.labels || [];
        const zones = nbData.zones || [];
        
        if (!labels.length || !zones.length) {
            console.warn('No labels or zones data');
            return;
        }

        console.log('Drawing NB wave chart with zones:', zones.length);

        const stepX = width / zones.length;
        const chartHeight = height - 80; // 상단 제목과 하단 정보 공간 확보 (높이 증가)
        const volumeHeight = 100; // 거래량 차트 높이 (높이 증가)

        // 격자 그리기
        this.drawGrid(width, height, volumeHeight);

        // 여러 라인 그리기 제거됨 (모든 라인 제거)
        // this.drawMultipleLines(zones, stepX, chartHeight, height);

        // 거래량 라인 그리기 제거됨 (모든 라인 제거)
        // this.drawVolumeLine(zones, stepX, height, volumeHeight);

        // N/B Wave 점 그리기 및 연결선
        this.drawWaveDots(zones, stepX, chartHeight, height);

        // 시간 라벨 그리기
        this.drawTimeLabels(labels, stepX, height);

        // 통계 정보 표시
        this.drawStatistics(nbData, width, height, volumeHeight);

        // 차트 제목
        this.drawTitle(width);
    }

    // 격자 그리기
    drawGrid(width, height, volumeHeight) {
        const ctx = this.ctx;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            const x = (width / 10) * i;
            ctx.beginPath();
            ctx.moveTo(x, 40);
            ctx.lineTo(x, height - volumeHeight);
            ctx.stroke();
        }
    }

    // 여러 라인 그리기 (비활성화됨 - 모든 라인 제거)
    drawMultipleLines(zones, stepX, chartHeight, height) {
        // 모든 라인 제거됨
        return;
    }

    // 강도 라인 그리기
    drawStrengthLine(zones, stepX, chartHeight, height) {
        const ctx = this.ctx;
        const strengths = zones.map(z => z.strength || 0.5);
        const maxStrength = Math.max(...strengths);
        const minStrength = Math.min(...strengths);
        
        ctx.strokeStyle = 'rgba(255, 0, 255, 0.6)'; // 마젠타
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        
        for (let i = 0; i < strengths.length; i++) {
            const x = i * stepX + stepX / 2;
            const strengthRatio = (strengths[i] - minStrength) / (maxStrength - minStrength);
            const y = 40 + (1 - strengthRatio) * (chartHeight - 20);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 구역 변화 라인 그리기
    drawZoneChangeLine(zones, stepX, chartHeight, height) {
        const ctx = this.ctx;
        const zoneValues = zones.map(z => {
            if (z.zone === 'ORANGE') return 1;
            if (z.zone === 'BLUE') return -1;
            return 0;
        });
        
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.7)'; // 시안
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < zoneValues.length; i++) {
            const x = i * stepX + stepX / 2;
            const y = 40 + (chartHeight / 2) + (zoneValues[i] * (chartHeight / 4));
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
    }

    // 이동평균 라인 그리기
    drawMovingAverageLine(zones, stepX, chartHeight, height) {
        const ctx = this.ctx;
        const period = 5; // 5기간 이동평균
        const strengths = zones.map(z => z.strength || 0.5);
        
        if (strengths.length < period) return;
        
        const maValues = [];
        for (let i = period - 1; i < strengths.length; i++) {
            const sum = strengths.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            maValues.push(sum / period);
        }
        
        const maxStrength = Math.max(...strengths);
        const minStrength = Math.min(...strengths);
        
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)'; // 노란색
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 5]);
        ctx.beginPath();
        
        for (let i = 0; i < maValues.length; i++) {
            const x = (i + period - 1) * stepX + stepX / 2;
            const maRatio = (maValues[i] - minStrength) / (maxStrength - minStrength);
            const y = 40 + (1 - maRatio) * (chartHeight - 20);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 추세 라인 그리기
    drawTrendLine(zones, stepX, chartHeight, height) {
        const ctx = this.ctx;
        const strengths = zones.map(z => z.strength || 0.5);
        
        if (strengths.length < 10) return;
        
        // 선형 회귀로 추세선 계산
        const n = strengths.length;
        const xValues = Array.from({length: n}, (_, i) => i);
        const sumX = xValues.reduce((a, b) => a + b, 0);
        const sumY = strengths.reduce((a, b) => a + b, 0);
        const sumXY = xValues.reduce((sum, x, i) => sum + x * strengths[i], 0);
        const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        ctx.strokeStyle = 'rgba(255, 165, 0, 0.9)'; // 주황색
        ctx.lineWidth = 3;
        ctx.setLineDash([15, 10]);
        ctx.beginPath();
        
        for (let i = 0; i < n; i++) {
            const x = i * stepX + stepX / 2;
            const trendY = slope * i + intercept;
            const maxStrength = Math.max(...strengths);
            const minStrength = Math.min(...strengths);
            const trendRatio = (trendY - minStrength) / (maxStrength - minStrength);
            const y = 40 + (1 - trendRatio) * (chartHeight - 20);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // 거래량 라인 그리기 (비활성화됨 - 모든 라인 제거)
    drawVolumeLine(zones, stepX, height, volumeHeight) {
        // 거래량 라인 제거됨
        return;
    }

    // N/B Wave 점 그리기 (원본 데이터 사용)
    drawWaveDots(zones, stepX, chartHeight, height) {
        const ctx = this.ctx;
        console.log('Drawing NB wave dots:', zones.length);
        
        // 원본 데이터 사용 (스무딩 제거)
        for (let i = 0; i < zones.length; i++) {
            const z = zones[i];
            const x = i * stepX + stepX / 2;
            
            // r_value를 기반으로 Y 위치 계산 (위쪽으로 더 분산)
            const rValue = z.r_value || 0.0;
            const y = 20 + (1 - rValue) * (chartHeight - 40);
            
            // 점 크기 (겹침 방지를 위해 작게)
            const dotSize = 2 + (z.strength || 0.5) * 2;
            
            // 구역별 색상 설정 (투명도 높여서 겹침 방지)
            this.setZoneColor(z.zone, true);
            
            // 점 그리기 (겹침 방지를 위해 얇게)
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.arc(x, y, dotSize, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // 연결선 그리기 (원본 데이터로)
            if (i > 0) {
                this.drawConnectionLine(zones, i, stepX, chartHeight);
            }
        }
    }
    
    // 구역 데이터 스무딩 (비활성화됨 - 원본 데이터 사용)
    smoothZones(zones) {
        // 스무딩 비활성화 - 원본 데이터 그대로 반환
        return zones;
    }
    
    // 최종 스무딩 (비활성화됨 - 원본 데이터 사용)
    finalSmoothing(zones) {
        // 스무딩 비활성화 - 원본 데이터 그대로 반환
        return zones;
    }
    
    // 스무딩된 연결선 그리기
    drawSmoothedConnectionLine(zones, i, stepX, chartHeight) {
        const ctx = this.ctx;
        const prevR = zones[i-1].r_value || 0.5;
        const currentR = zones[i].r_value || 0.5;
        
        const prevX = (i - 1) * stepX + stepX / 2;
        const prevY = 40 + (1 - prevR) * (chartHeight - 20);
        const x = i * stepX + stepX / 2;
        const y = 40 + (1 - currentR) * (chartHeight - 20);
        
        // 구역별 연결선 색상 (매우 투명하게)
        const prevZone = zones[i-1].zone;
        const currentZone = zones[i].zone;
        let lineColor;
        
        if (currentZone === 'ORANGE' || prevZone === 'ORANGE') {
            lineColor = 'rgba(255, 183, 3, 0.3)'; // 주황색 연결선 (매우 투명)
        } else if (currentZone === 'BLUE' || prevZone === 'BLUE') {
            lineColor = 'rgba(0, 209, 255, 0.3)'; // 파란색 연결선 (매우 투명)
        } else {
            lineColor = 'rgba(128, 128, 128, 0.2)'; // 회색 연결선 (매우 투명)
        }
        
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 0.5; // 선 두께 매우 얇게
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(x, y);
        ctx.stroke();
    }

    // 구역별 색상 설정
    setZoneColor(zone, lowOpacity = false) {
        const ctx = this.ctx;
        const opacity = lowOpacity ? 0.4 : 0.9;
        
        if (zone === 'ORANGE') {
            ctx.fillStyle = `rgba(255, 183, 3, ${opacity})`; // 주황색
            ctx.strokeStyle = `rgba(255, 140, 0, ${opacity})`;
        } else if (zone === 'BLUE') {
            ctx.fillStyle = `rgba(0, 209, 255, ${opacity})`; // 파란색
            ctx.strokeStyle = `rgba(0, 102, 204, ${opacity})`;
        } else {
            ctx.fillStyle = `rgba(128, 128, 128, ${opacity * 0.8})`; // 회색 (중립)
            ctx.strokeStyle = `rgba(100, 100, 100, ${opacity})`;
        }
    }

    // 연결선 그리기 (원본 데이터 사용)
    drawConnectionLine(zones, i, stepX, chartHeight) {
        const ctx = this.ctx;
        const prevZone = zones[i - 1];
        const currZone = zones[i];
        
        if (!prevZone || !currZone) return;
        
        const prevX = (i - 1) * stepX + stepX / 2;
        const currX = i * stepX + stepX / 2;
        
        const prevRValue = prevZone.r_value || 0.0;
        const currRValue = currZone.r_value || 0.0;
        
        const prevY = 20 + (1 - prevRValue) * (chartHeight - 40);
        const currY = 20 + (1 - currRValue) * (chartHeight - 40);
        
        // 구역별 색상으로 연결선 그리기
        this.setZoneColor(currZone.zone, false);
        ctx.lineWidth = 0.8;
        ctx.globalAlpha = 0.4;
        
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(currX, currY);
        ctx.stroke();
        
        ctx.globalAlpha = 1.0;
    }

    // 시간 라벨 그리기
    drawTimeLabels(labels, stepX, height) {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
        ctx.font = 'bold 10px Courier New';
        ctx.textAlign = 'center';
        
        // 라벨 개수 제한 (너무 많으면 겹침)
        const labelInterval = Math.max(1, Math.floor(labels.length / 6));
        for (let i = 0; i < labels.length; i += labelInterval) {
            const x = i * stepX + stepX / 2;
            ctx.fillText(labels[i], x, height - 5);
        }
    }

    // 통계 정보 표시
    drawStatistics(nbData, width, height, volumeHeight) {
        const ctx = this.ctx;
        const zones = nbData.zones || [];

        ctx.font = 'bold 14px Courier New';
        ctx.textAlign = 'left';
        
        // BLUE/ORANGE 개수 직접 계산
        const orangeCount = zones.filter(z => z.zone === 'ORANGE').length;
        const blueCount = zones.filter(z => z.zone === 'BLUE').length;
        const totalCount = zones.length;
        
        // ORANGE 구역 정보 (더 큰 폰트)
        ctx.fillStyle = 'rgba(255, 183, 3, 0.9)';
        ctx.fillText(`ORANGE: ${orangeCount} (${((orangeCount/totalCount)*100).toFixed(1)}%)`, 10, height - 60);
        
        // BLUE 구역 정보 (더 큰 폰트)
        ctx.fillStyle = 'rgba(0, 209, 255, 0.9)';
        ctx.fillText(`BLUE: ${blueCount} (${((blueCount/totalCount)*100).toFixed(1)}%)`, 10, height - 40);
        
        // 총 개수 정보
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fillText(`TOTAL: ${totalCount} zones`, 10, height - 20);
        
        // 현재 가격 정보 (summary에서 가져오기)
        if (nbData.summary && nbData.summary.current_price) {
            ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
            ctx.fillText(`Price: ${nbData.summary.current_price.toLocaleString()}`, width - 200, height - 40);
        }
        
        // 거래량 정보
        const volumes = zones.map(z => z.volume || 0);
        const maxVolume = Math.max(...volumes);
        if (maxVolume > 0) {
            ctx.fillStyle = 'rgba(255, 255, 0, 0.9)';
            ctx.fillText(`Max Volume: ${maxVolume.toFixed(2)}`, width - 200, height - 20);
        }
    }

    // 라인 범례 표시
    drawLegend(width, height, volumeHeight) {
        const ctx = this.ctx;
        const legendY = height - volumeHeight - 50;
        const legendX = width - 200;
        
        ctx.font = 'bold 10px Courier New';
        ctx.textAlign = 'left';
        
        // 강도 라인 범례
        ctx.strokeStyle = 'rgba(255, 0, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(legendX, legendY);
        ctx.lineTo(legendX + 20, legendY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 0, 255, 0.9)';
        ctx.fillText('Strength', legendX + 25, legendY + 3);
        
        // 구역 변화 라인 범례
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.7)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX, legendY + 15);
        ctx.lineTo(legendX + 20, legendY + 15);
        ctx.stroke();
        ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
        ctx.fillText('Zone Change', legendX + 25, legendY + 18);
        
        // 이동평균 라인 범례
        ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 5]);
        ctx.beginPath();
        ctx.moveTo(legendX, legendY + 30);
        ctx.lineTo(legendX + 20, legendY + 30);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 255, 0, 0.9)';
        ctx.fillText('MA(5)', legendX + 25, legendY + 33);
        
        // 추세 라인 범례
        ctx.strokeStyle = 'rgba(255, 165, 0, 0.9)';
        ctx.lineWidth = 3;
        ctx.setLineDash([15, 10]);
        ctx.beginPath();
        ctx.moveTo(legendX, legendY + 45);
        ctx.lineTo(legendX + 20, legendY + 45);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255, 165, 0, 0.9)';
        ctx.fillText('Trend', legendX + 25, legendY + 48);
    }

    // 차트 제목 그리기
    drawTitle(width) {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = 'bold 14px Courier New';
        ctx.textAlign = 'center';
        ctx.fillText('N/B Wave Analysis (Multi-Line)', width / 2, 25);
    }

    // 데이터 업데이트
    async updateData(timeframe = 'minute1', bars = 300) {
        try {
            // 전역 저장된 데이터가 있으면 사용, 없으면 API 호출
            if (window.sharedNbWaveData && window.sharedNbWaveData.timeframe === timeframe) {
                console.log('📊 Using cached NB wave data for timeframe:', timeframe);
                this.drawChart(window.sharedNbWaveData);
                return window.sharedNbWaveData;
            }
            
            const response = await fetch(`http://localhost:5057/api/nb-wave?timeframe=${timeframe}&bars=${bars}`);
            if (!response.ok) {
                throw new Error(`NB wave API error: ${response.status}`);
            }
            
            const data = await response.json();
            this.drawChart(data);
            return data;
        } catch (error) {
            console.error('Failed to update NB wave data:', error);
            throw error;
        }
    }

    // 차트 크기 조정
    resize(width, height) {
        if (this.canvas) {
            this.canvas.width = width;
            this.canvas.height = height;
            if (this.currentData) {
                this.drawChart(this.currentData);
            }
        }
    }

    // 정리
    destroy() {
        this.canvas = null;
        this.ctx = null;
        this.currentData = null;
        this.isInitialized = false;
    }
}

// 전역 인스턴스 생성
window.nbWavePanel = new NbWavePanel();

// 모듈 내보내기
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NbWavePanel;
}
