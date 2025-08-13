<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // CORS 문제 해결을 위한 헤더 추가

// 오류 메시지 표시
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$folder = isset($_GET['folder']) ? $_GET['folder'] : 'Up'; // 기본 폴더는 'Up'
$imageDir = 'img/' . $folder; // 요청된 폴더로 설정
$imageFiles = [];

if (is_dir($imageDir)) {
    if ($dh = opendir($imageDir)) {
        while (($file = readdir($dh)) !== false) {
            $filePath = $imageDir . '/' . $file;
            if ($file != '.' && $file != '..') {
                $extension = pathinfo($file, PATHINFO_EXTENSION);
                if (in_array($extension, ['JPG', 'jpg', 'jpeg', 'png', 'gif'])) {
                    $imageFiles[] = rawurlencode($filePath);
                } else {
                    // 디버깅: 지원되지 않는 파일 형식
                    error_log("Unsupported file format: $filePath");
                }
            } else {
                // 디버깅: 읽은 항목이 디렉터리 표시자일 경우
                error_log("Skipping directory entry: $file");
            }
        }
        closedir($dh);
    } else {
        // 디버깅: 디렉터리 열기에 실패한 경우
        error_log("Failed to open directory: $imageDir");
        echo json_encode(["error" => "Failed to open directory: $imageDir"]);
        exit;
    }
} else {
    // 디버깅: 디렉터리가 존재하지 않는 경우
    error_log("Not a directory: $imageDir");
    echo json_encode(["error" => "Not a directory: $imageDir"]);
    exit;
}

if (empty($imageFiles)) {
    // 디버깅: 파일이 없는 경우
    error_log("No image files found in directory: $imageDir");
} else {
    // 디버깅: 찾은 파일 출력
    error_log("Found image files: " . implode(', ', $imageFiles));
}

echo json_encode($imageFiles);
?>
