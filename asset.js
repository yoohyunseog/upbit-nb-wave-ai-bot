$(document).ready(function() {
    let chartInitialized = false;
    let chart;
    const isFileScheme = location.protocol === 'file:';

    // 데이터를 가져오는 함수
    function fetchData() {
        if (isFileScheme) {
            // Skip in file:// development to avoid CORS errors
            console.info('Skipping fetch_assets.php on file:// (CORS). Run a local PHP server to enable.');
            return;
        }
        $.ajax({
            url: './fetch_assets.php',
            method: 'GET',
            dataType: 'json',
            success: function(response) {
                if (response && response.balances) {
                    populateTable(response.balances);
                    displayProfitDetails(response.profitDetails);
                } else {
                    console.error("Invalid response format", response);
                }
            },
            error: function(xhr, status, error) {
                console.warn("Error fetching data (assets)", status);
            }
        });
    }

    // 테이블을 업데이트하는 함수
    function populateTable(balances) {
        var tableBody = $('#cryptoAssetsTable tbody');
        tableBody.empty(); // 기존 데이터를 지웁니다.

        balances.forEach(function(balance) {
            var row = `
                <tr>
                    <td>보유자산</td>
                    <td>${balance.currency}</td>
                    <td>${balance.balance}</td>
                    <td>${balance.locked}</td>
                    <td>${balance.avg_buy_price}</td>
                    <td>${balance.asset_value}</td>
                </tr>
            `;
            tableBody.append(row);
        });
    }

    // 수익 세부 정보를 표시하는 함수
    function displayProfitDetails(profitDetails) {
        var profitContainer = $('#profitDetails');
        profitContainer.empty(); // 기존 데이터를 지웁니다.

        var profitHtml = `
            <p>수익 실현 %: ${profitDetails.profit}</p>
            <p>BTC 자산의 총 가치: ${profitDetails.btc_asset_value}</p>
            <p>BTC 자산의 ${profitDetails.profit}% 가치: ${profitDetails.btc_25_percent_value}</p>
        `;
        profitContainer.append(profitHtml);
    }

    // 초기 데이터 가져오기
    fetchData();

    // 주기적 업데이트 (HTTP/HTTPS에서만 실동작)
    setInterval(fetchData, 20000);
});
