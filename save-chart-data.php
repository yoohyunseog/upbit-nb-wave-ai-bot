<?php
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight requests
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit();
}

try {
    // Get JSON input
    $input = json_decode(file_get_contents('php://input'), true);
    
    if (!$input || !isset($input['filename']) || !isset($input['data'])) {
        http_response_code(400);
        echo json_encode(['error' => 'Invalid input data']);
        exit();
    }
    
    $filename = $input['filename'];
    $data = $input['data'];
    
    // Validate filename
    if (!preg_match('/^chart_data_[a-zA-Z0-9_-]+\.json$/', $filename)) {
        http_response_code(400);
        echo json_encode(['error' => 'Invalid filename']);
        exit();
    }
    
    // Create data directory if it doesn't exist
    $dataDir = 'data/chart_data';
    if (!is_dir($dataDir)) {
        mkdir($dataDir, 0755, true);
    }
    
    // Create subdirectories by date
    $dateDir = $dataDir . '/' . date('Y-m-d');
    if (!is_dir($dateDir)) {
        mkdir($dateDir, 0755, true);
    }
    
    // Create subdirectories by interval
    $interval = $data['interval'] ?? 'unknown';
    $intervalDir = $dateDir . '/' . $interval;
    if (!is_dir($intervalDir)) {
        mkdir($intervalDir, 0755, true);
    }
    
    // Full file path
    $filepath = $intervalDir . '/' . $filename;
    
    // Convert data to JSON with pretty formatting
    $jsonData = json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
    
    if ($jsonData === false) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to encode JSON data']);
        exit();
    }
    
    // Write to file
    $bytesWritten = file_put_contents($filepath, $jsonData);
    
    if ($bytesWritten === false) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to write file']);
        exit();
    }
    
    // Return success response
    echo json_encode([
        'success' => true,
        'filename' => $filename,
        'filepath' => $filepath,
        'fileSize' => $bytesWritten,
        'totalCandles' => $data['totalCandles'] ?? 0,
        'totalZones' => $data['totalZones'] ?? 0,
        'totalNbWave' => $data['totalNbWave'] ?? 0,
        'timestamp' => date('Y-m-d H:i:s')
    ]);
    
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode([
        'error' => 'Server error',
        'message' => $e->getMessage()
    ]);
}
?>
