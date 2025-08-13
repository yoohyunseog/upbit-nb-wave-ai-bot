<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');

// 오류 메시지 표시
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$ffmpegBinary = 'C:\\ffmpeg\\bin\\ffmpeg.exe'; // FFmpeg 실행 파일 경로
$outputDir = 'output_data'; // 출력 파일을 저장할 폴더

function executeFFmpeg($inputFile, $outputPngFile) {
    global $ffmpegBinary;
    // FFmpeg 명령어: 주파수 스펙트럼 이미지를 생성
    $command = "\"$ffmpegBinary\" -i \"$inputFile\" -lavfi showspectrumpic=s=1024x512:legend=disabled \"$outputPngFile\" -y";
    error_log("Executing FFmpeg command: $command");
    shell_exec($command);
    if (!file_exists($outputPngFile)) {
        error_log("Failed to create spectrum image: $outputPngFile");
    }
}

function analyzeSpectrogram($pngFile) {
    $data = [];
    if (!file_exists($pngFile)) {
        error_log("File not found: $pngFile");
        return $data;
    }
    
    $image = imagecreatefrompng($pngFile);
    if (!$image) {
        error_log("Failed to create image from: $pngFile");
        return $data;
    }
    
    $width = imagesx($image);
    $height = imagesy($image);
    
    for ($x = 0; $x < $width; $x++) {
        $columnData = [];
        for ($y = 0; $y < $height; $y++) {
            $rgb = imagecolorat($image, $x, $y);
            $r = ($rgb >> 16) & 0xFF;
            $g = ($rgb >> 8) & 0xFF;
            $b = $rgb & 0xFF;
            $brightness = ($r + $g + $b) / 3;
            $columnData[] = $brightness;
        }
        $data[] = $columnData;
    }
    
    imagedestroy($image);
    return $data;
}

function saveSpectrumDataToFile($spectrumData, $outputTxtFile, $duration) {
    $file = fopen($outputTxtFile, 'w');
    if (!$file) {
        error_log("Failed to open file for writing: $outputTxtFile");
        return;
    }
    
    $frameCount = count($spectrumData);
    if ($frameCount == 0) {
        error_log("No spectrum data to save for: $outputTxtFile");
        fclose($file);
        return;
    }
    
    $frameDuration = $duration / $frameCount; // 각 프레임의 지속 시간 (초 단위)
    foreach ($spectrumData as $index => $column) {
        $time = $index * $frameDuration;
        fputcsv($file, array_merge([$time], $column)); // 시간 정보를 추가
    }
    fclose($file);
}

function getDuration($filePath) {
    global $ffmpegBinary;
    $command = "\"$ffmpegBinary\" -i \"$filePath\" 2>&1";
    error_log("Executing command for duration: $command");
    $output = shell_exec($command);
    error_log("Command output: $output");
    
    if (preg_match('/Duration: (\d+):(\d+):(\d+\.\d+)/', $output, $matches)) {
        $hours = $matches[1];
        $minutes = $matches[2];
        $seconds = $matches[3];
        $duration = ($hours * 3600) + ($minutes * 60) + $seconds;
        error_log("Calculated duration: $duration seconds");
        return $duration;
    } else {
        error_log("Failed to parse duration from output: $output");
        return null;
    }
}

$folder = isset($_GET['folder']) ? $_GET['folder'] : 'Up'; // 기본 폴더는 'Up'
$audioDir = 'music/' . $folder; // 요청된 폴더로 설정
$audioFiles = [];

// 출력 디렉토리가 없는 경우 생성
if (!is_dir($outputDir)) {
    if (!mkdir($outputDir, 0777, true)) {
        error_log("Failed to create output directory: $outputDir");
        echo json_encode(["error" => "Failed to create output directory: $outputDir"]);
        exit;
    }
}

if (is_dir($audioDir)) {
    if ($dh = opendir($audioDir)) {
        while (($file = readdir($dh)) !== false) {
            $filePath = $audioDir . '/' . $file;
            if ($file != '.' && $file != '..') {
                $extension = pathinfo($file, PATHINFO_EXTENSION);
                if ($extension == 'mp4') {
                    $audioFiles[] = rawurlencode($filePath);
                } else {
                    error_log("Unsupported file format: $filePath");
                }
            } else {
                error_log("Skipping directory entry: $file");
            }
        }
        closedir($dh);
    } else {
        error_log("Failed to open directory: $audioDir");
        echo json_encode(["error" => "Failed to open directory: $audioDir"]);
        exit;
    }
} else {
    error_log("Not a directory: $audioDir");
    echo json_encode(["error" => "Not a directory: $audioDir"]);
    exit;
}

$audioData = [];

foreach ($audioFiles as $file) {
    $filePath = rawurldecode($file);
    $uniqueId = pathinfo($filePath, PATHINFO_FILENAME);
    $outputPngFile = "$outputDir/spectrum_$uniqueId.png";
    $outputTxtFile = "$outputDir/spectrum_$uniqueId.txt";
    
    if (!file_exists($outputPngFile)) {
        executeFFmpeg($filePath, $outputPngFile);
    }
    
    $duration = getDuration($filePath);
    if ($duration === null) {
        continue;
    }
    
    if (!file_exists($outputTxtFile)) {
        $spectrumData = analyzeSpectrogram($outputPngFile);
        saveSpectrumDataToFile($spectrumData, $outputTxtFile, $duration);
    }
    
    $audioData[] = [
        'file' => $file,
        'spectrumDataFile' => $outputTxtFile
    ];
}

echo json_encode($audioData);
?>
