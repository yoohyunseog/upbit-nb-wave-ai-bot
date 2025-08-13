<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // CORS 문제 해결을 위한 헤더 추가

// 오류 메시지 표시
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$folder = isset($_GET['folder']) ? $_GET['folder'] : 'Up'; // 기본 폴더는 'Up'
$audioDir = 'music/' . $folder; // 요청된 폴더로 설정
$audioFiles = [];

if (is_dir($audioDir)) {
    if ($dh = opendir($audioDir)) {
        while (($file = readdir($dh)) !== false) {
            $filePath = $audioDir . '/' . $file;
            if ($file != '.' && $file != '..') {
                $extension = pathinfo($file, PATHINFO_EXTENSION);
                if ($extension == 'mp3') {
                    $audioFiles[] = rawurlencode($filePath);
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
        error_log("Failed to open directory: $audioDir");
        echo json_encode(["error" => "Failed to open directory: $audioDir"]);
        exit;
    }
} else {
    // 디버깅: 디렉터리가 존재하지 않는 경우
    error_log("Not a directory: $audioDir");
    echo json_encode(["error" => "Not a directory: $audioDir"]);
    exit;
}

if (empty($audioFiles)) {
    // 디버깅: 파일이 없는 경우
    error_log("No audio files found in directory: $audioDir");
} else {
    // 디버깅: 찾은 파일 출력
    error_log("Found audio files: " . implode(', ', $audioFiles));
}

echo json_encode($audioFiles);
?>
