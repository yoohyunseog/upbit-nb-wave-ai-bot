$(function(){
    const maxItems = 30;
    let records = [];
    let debounceTimer = null; // 디바운싱을 위한 타이머

    // localStorage에서 히스토리 데이터 로드
    function loadHistoryFromStorage() {
        const savedHistory = localStorage.getItem('nbHistoryData');
        if (savedHistory) {
            try {
                records = JSON.parse(savedHistory);
                //console.log('📂 히스토리 데이터 로드 완료:', records.length + '개 항목');
                
                // 저장된 데이터로 UI 재구성
                $("#historyWrapper").empty();
                records.forEach(record => {
                    renderHistoryItem(record.zone, record.strength);
                });
                
                // Summary 업데이트
                updateSummary();
            } catch (error) {
                console.error('❌ 히스토리 데이터 로드 실패:', error);
                records = [];
            }
        }
    }

    // localStorage에 히스토리 데이터 저장
    function saveHistoryToStorage() {
        try {
            localStorage.setItem('nbHistoryData', JSON.stringify(records));
            //console.log('💾 히스토리 데이터 저장 완료:', records.length + '개 항목');
        } catch (error) {
            console.error('❌ 히스토리 데이터 저장 실패:', error);
        }
    }

    // UI 렌더링 함수 (분리)
    function renderHistoryItem(zone, strength) {
        // 원 크기 (strength 값에 비례, 최대 70px 정사각형)
        const size = Math.min((strength / 100) * 70, 70);

        // col-6 col-6 구조
        const $colCircle = $("<div>").addClass("col-6 circle-container")
            .append(
              $("<div>")
                .addClass("circle " + (zone==="BLUE" ? "blue" : "orange"))
                .css({
                    width: size+"px",
                    height: size+"px"
                })
            );

        const $colEmpty = $("<div>").addClass("col-6"); // 오른쪽 비워둠

        const $container = $("<div>").addClass("row gx-1")
            .append($colCircle)
            .append($colEmpty);

        $("#historyWrapper").append($container);
        if($("#historyWrapper .row").length > maxItems) {
            $("#historyWrapper .row").first().remove();
        }
    }

    // Summary 업데이트 함수 (분리)
    function updateSummary() {
        let orangeCount=0, blueCount=0, orangeSum=0, blueSum=0;
        records.forEach(r=>{
            if(r.zone==="ORANGE"){ orangeCount++; orangeSum+=r.strength; }
            else { blueCount++; blueSum+=r.strength; }
        });

        // 이전 majority 값 가져오기
        let majority = $("#majority-zone").text();

        if(orangeCount > blueCount) {
            majority = "ORANGE";
        } else if(blueCount > orangeCount) {
            majority = "BLUE";
        }
        // 같으면 (TIE)일 경우 → 이전 majority 유지

        $("#majority-zone").text(majority).css("color", 
            majority==="ORANGE" ? "orange" : (majority==="BLUE" ? "dodgerblue" : "black")
        );
        $("#orange-sum").text(orangeSum);
        $("#blue-sum").text(blueSum);
    }

    function addHistory(zone, strength) {
        // 기록 저장
        records.push({zone, strength, timestamp: Date.now()});
        if(records.length > maxItems) records.shift();

        // UI 렌더링
        renderHistoryItem(zone, strength);
        
        // Summary 업데이트
        updateSummary();
        
        // localStorage에 저장
        saveHistoryToStorage();
    }

    // 현재 timeframe display에서 데이터를 읽어서 히스토리에 추가 (디바운싱 적용)
    function addCurrentTimeframeToHistory() {
        // 기존 타이머가 있으면 취소
        if (debounceTimer) {
            clearTimeout(debounceTimer);
        }
        
        // 새로운 타이머 설정 (200ms 후 실행)
        debounceTimer = setTimeout(() => {
            const currentZone = $("#current-timeframe-zone").text();
            const currentStrengthText = $("#right-trading-zone-strength").text();
            const currentStrength = parseInt(currentStrengthText.replace('강도: ', ''));
            
            if (currentZone && !isNaN(currentStrength)) {
                // 마지막 히스토리와 같은 값인지 확인 (중복 방지)
                const lastRecord = records[records.length - 1];
                if (!lastRecord || lastRecord.zone !== currentZone || lastRecord.strength !== currentStrength) {
                    addHistory(currentZone, currentStrength);
                    //console.log(`📝 Added current timeframe to history: ${currentZone} (${currentStrength}%)`);
                } else {
                    //console.log(`⏭️ Skipped duplicate: ${currentZone} (${currentStrength}%)`);
                }
            }
            debounceTimer = null;
        }, 200);
    }

    // MutationObserver로 current-timeframe-display 변화 감지
    function setupTimeframeObserver() {
        const targetNode = document.getElementById('current-timeframe-zone');
        const strengthNode = document.getElementById('right-trading-zone-strength');
        
        if (targetNode && strengthNode) {
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' || mutation.type === 'characterData') {
                        // 디바운싱된 함수 호출
                        addCurrentTimeframeToHistory();
                    }
                });
            });

            // zone과 strength 모두 감시
            observer.observe(targetNode, { 
                childList: true, 
                characterData: true, 
                subtree: true 
            });
            observer.observe(strengthNode, { 
                childList: true, 
                characterData: true, 
                subtree: true 
            });

            //console.log('👁️ Timeframe display observer setup complete');
        } else {
            console.warn('⚠️ Timeframe display elements not found');
        }
    }

    // 실제 시스템과 연동하는 함수
    window.addHistoryItem = (timeframe, zone, strength) => {
        addHistory(zone, strength);
    };

    // 현재 timeframe을 히스토리에 추가하는 함수
    window.addCurrentTimeframeToHistory = addCurrentTimeframeToHistory;

    // 히스토리 초기화 함수
    window.clearHistory = () => {
        records = [];
        $("#historyWrapper").empty();
        $("#majority-zone").text("-").css("color", "black");
        $("#orange-sum").text("0");
        $("#blue-sum").text("0");
        
        // localStorage에서도 삭제
        localStorage.removeItem('nbHistoryData');
        //console.log('🗑️ 히스토리 데이터 완전 삭제 완료');
    };

    // 히스토리 데이터 조회 함수
    window.getHistoryData = () => {
        return records;
    };

    // DOM이 로드된 후 observer 설정 및 저장된 히스토리 로드
    $(document).ready(function() {
        // 저장된 히스토리 데이터 로드
        loadHistoryFromStorage();
        
        // observer 설정
        setupTimeframeObserver();
        
        //console.log('🚀 히스토리 시스템 초기화 완료');
    });

    // 예시: 0.2초마다 랜덤 zone, strength로 호출
    // setInterval(function(){
    //     const zones = ["BLUE","ORANGE"];
    //     const zone = zones[Math.floor(Math.random() * zones.length)];
    //     const strength = Math.floor(Math.random() * 100) + 1;
    //     addHistory(zone, strength);   // ← 함수 호출
    // }, 200);

    // 추가: 1초마다 현재 timeframe을 히스토리에 추가 (선택사항)
    // setInterval(addCurrentTimeframeToHistory, 1000);
});
