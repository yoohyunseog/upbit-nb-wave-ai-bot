<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // CORS 문제 해결을 위한 헤더 추가

// 오류 메시지 표시
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$audioFile = isset($_GET['audioFile']) ? urldecode($_GET['audioFile']) : null;

if (!$audioFile) {
    echo json_encode(["error" => "No audio file specified"]);
    exit;
}

if (!file_exists($audioFile)) {
    echo json_encode(["error" => "Audio file not found: $audioFile"]);
    exit;
}

// 임시 파일 경로
$tempImage = tempnam(sys_get_temp_dir(), 'wave') . '.png';

// ffmpeg 명령어를 사용하여 오디오 데이터를 추출하여 이미지로 저장
$command = escapeshellcmd("ffmpeg -i '$audioFile' -filter_complex 'showwavespic=s=640x120' -frames:v 1 -f image2 '$tempImage'");
$output = shell_exec($command . " 2>&1");

if (!file_exists($tempImage)) {
    echo json_encode(["error" => "Failed to execute ffmpeg command", "command" => $command, "output" => $output]);
    exit;
}

// 이미지 데이터를 읽고 주파수 데이터로 변환
$image = imagecreatefrompng($tempImage);
if (!$image) {
    echo json_encode(["error" => "Failed to create image from $tempImage"]);
    exit;
}

$soundData = [];
$width = imagesx($image);
$height = imagesy($image);

for ($x = 0; $x < $width; $x++) {
    $sum = 0;
    for ($y = 0; $y < $height; $y++) {
        $rgb = imagecolorat($image, $x, $y);
        $r = ($rgb >> 16) & 0xFF;
        $g = ($rgb >> 8) & 0xFF;
        $b = $rgb & 0xFF;
        $gray = ($r + $g + $b) / 3;
        $sum += $gray;
    }
    $avg = $sum / $height;
    $soundData[] = 255 - $avg; // 파형의 밝기 값을 주파수 데이터로 변환
}

imagedestroy($image);
unlink($tempImage);

echo json_encode($soundData);
?>
