<?php
header('Access-Control-Allow-Origin: *'); // CORS 문제 해결을 위한 헤더 추가
header('Content-Type: application/json');

// 오류 메시지 표시
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if (!isset($_GET['file'])) {
    http_response_code(400);
    echo json_encode(['error' => 'File parameter is missing']);
    exit;
}

$file = urldecode($_GET['file']); // 파일 이름 디코딩
$fileBaseName = basename($file); // 기본 파일 이름 추출
$filePath = 'output_data/' . $fileBaseName;

if (!file_exists($filePath)) {
    http_response_code(404);
    echo json_encode(['error' => 'File not found', 'path' => $filePath]);
    error_log('File not found: ' . $filePath);
    exit;
}

$content = @file_get_contents($filePath);
if ($content === false) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to read file', 'path' => $filePath]);
    error_log('Failed to read file: ' . $filePath);
    exit;
}

$lines = explode("\n", $content);
$data = array_map('str_getcsv', $lines); // CSV 형식으로 파싱

echo json_encode($data);
?>
