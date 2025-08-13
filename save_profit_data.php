<?php
// CORS 헤더 설정
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST, GET, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");

// 예비 요청 처리
if ($_SERVER['REQUEST_METHOD'] == 'OPTIONS') {
    http_response_code(200);
    exit();
}

// POST 요청 처리
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $currentPrice = isset($_POST['currentPrice']) ? floatval($_POST['currentPrice']) : 0;
    $averagePrice = isset($_POST['averagePrice']) ? floatval($_POST['averagePrice']) : 0;
    $fallProfitPercent = isset($_POST['fallProfitPercent']) ? floatval($_POST['fallProfitPercent']) : 0;
    $riseProfitPercent = isset($_POST['riseProfitPercent']) ? floatval($_POST['riseProfitPercent']) : 0;
    
    // 데이터를 저장할 파일 경로
    $filePath = __DIR__ . '/profit_data.txt';
    
    // 파일 내용 생성
    $content = "Current Price: $currentPrice\n";
    $content .= "Average Price: $averagePrice\n";
    $content .= "Fall Profit Percent: $fallProfitPercent\n";
    $content .= "Rise Profit Percent: $riseProfitPercent\n";
    
    // 데이터를 파일에 저장
    if (file_put_contents($filePath, $content) !== false) {
        echo json_encode(['message' => 'Data saved successfully']);
    } else {
        http_response_code(500);
        echo json_encode(['message' => 'Failed to save data']);
    }
} else {
    http_response_code(405);
    echo json_encode(['message' => 'Method not allowed']);
}
?>
