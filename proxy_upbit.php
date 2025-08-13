<?php
// Simple Upbit proxy to avoid browser CORS and ease rate limits with short caching
header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
  http_response_code(200);
  exit;
}

$endpoint = $_GET['endpoint'] ?? '';
$market = $_GET['market'] ?? 'KRW-BTC';

// Build Upbit URL
if ($endpoint === 'candles') {
  $unit = (int)($_GET['unit'] ?? 1);
  $count = (int)($_GET['count'] ?? 200);
  if ($unit < 1) { $unit = 1; }
  if ($count < 1 || $count > 200) { $count = 200; }
  $upbitUrl = sprintf('https://api.upbit.com/v1/candles/minutes/%d?market=%s&count=%d', $unit, urlencode($market), $count);
} elseif ($endpoint === 'ticker') {
  $upbitUrl = 'https://api.upbit.com/v1/ticker?markets=' . urlencode($market);
} else {
  http_response_code(400);
  echo json_encode(['error' => 'Invalid endpoint']);
  exit;
}

// Cache settings (seconds)
$ttl = 8; // short cache to mitigate 429
$cacheKey = 'upbit_' . md5($upbitUrl);
$cacheDir = sys_get_temp_dir();
$cacheFile = rtrim($cacheDir, DIRECTORY_SEPARATOR) . DIRECTORY_SEPARATOR . $cacheKey . '.json';

// Serve from cache if fresh
if (file_exists($cacheFile) && (time() - filemtime($cacheFile)) < $ttl) {
  readfile($cacheFile);
  exit;
}

// Fetch from Upbit
$ch = curl_init($upbitUrl);
curl_setopt_array($ch, [
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_TIMEOUT => 8,
  CURLOPT_USERAGENT => '8BIT-Proxy/1.0',
]);
$resp = curl_exec($ch);
$errno = curl_errno($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($errno !== 0 || $httpCode >= 400 || !$resp) {
  // Fallback to stale cache if available
  if (file_exists($cacheFile)) {
    readfile($cacheFile);
    exit;
  }
  // Generate safe fallback payload to avoid breaking UI
  if ($endpoint === 'ticker') {
    $fallback = [ [ 'trade_price' => 0 ] ];
    echo json_encode($fallback);
    exit;
  }
  if ($endpoint === 'candles') {
    $now = round(microtime(true) * 1000);
    $base = 100000000; // 100M KRW default
    $out = [];
    for ($i = 0; $i < 30; $i++) {
      $out[] = [
        'timestamp' => $now - ($i * 60000),
        'trade_price' => $base,
        'candle_acc_trade_volume' => 0,
      ];
    }
    echo json_encode($out);
    exit;
  }
}

// Validate JSON
$json = json_decode($resp, true);
if ($json === null) {
  if (file_exists($cacheFile)) {
    readfile($cacheFile);
    exit;
  }
  // Return minimal valid JSON
  echo ($endpoint === 'ticker') ? json_encode([[ 'trade_price' => 0 ]]) : json_encode([]);
  exit;
}

// Save cache and return
@file_put_contents($cacheFile, json_encode($json));
echo json_encode($json);
?>


