<?php
// Upbit private balances fetcher with IP whitelist and CORS
header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type, Authorization');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
  http_response_code(200);
  exit;
}

// ------------ Configuration ------------
// Move secrets to environment or a separate config file for security.
$ACCESS_KEY = getenv('UPBIT_ACCESS_KEY') ?: '';
$SECRET_KEY = getenv('UPBIT_SECRET_KEY') ?: '';

// Allowed client IPs (requests to this PHP). Upbit itself also validates the server's egress IP.
$ALLOWED_IPS = [
  '203.245.9.72',
  '222.234.171.87',
  // local dev
  '127.0.0.1', '::1'
];

// ------------ Guards ------------
$clientIp = $_SERVER['REMOTE_ADDR'] ?? '';
if (!in_array($clientIp, $ALLOWED_IPS, true)) {
  http_response_code(403);
  echo json_encode(['error' => 'Forbidden', 'ip' => $clientIp]);
  exit;
}

// Dev-friendly fallback: if credentials are missing, return mock data
if (!$ACCESS_KEY || !$SECRET_KEY) {
  $mockBalances = [
    [ 'currency' => 'KRW', 'balance' => 1000000, 'locked' => 0, 'avg_buy_price' => 0, 'asset_value' => 0 ],
    [ 'currency' => 'BTC', 'balance' => 0.00123456, 'locked' => 0, 'avg_buy_price' => 98000000, 'asset_value' => 0 ],
  ];
  $ticker = public_ticker('KRW-BTC');
  $price = (!empty($ticker) && isset($ticker[0]['trade_price'])) ? (float)$ticker[0]['trade_price'] : 0;
  $btcAssetValue = $price * 0.00123456;
  echo json_encode([
    'balances' => $mockBalances,
    'profitDetails' => [
      'profit' => 0.902,
      'btc_asset_value' => $btcAssetValue,
      'btc_25_percent_value' => $btcAssetValue * 0.00902,
    ],
  ]);
  exit;
}

// ------------ Helpers ------------
function uuid4(): string {
  $data = random_bytes(16);
  $data[6] = chr((ord($data[6]) & 0x0f) | 0x40); // version 4
  $data[8] = chr((ord($data[8]) & 0x3f) | 0x80); // variant
  return vsprintf('%s%s-%s-%s-%s-%s%s%s', str_split(bin2hex($data), 4));
}

function jwt_encode(array $payload, string $secret): string {
  $header = ['alg' => 'HS256', 'typ' => 'JWT'];
  $base64UrlHeader = rtrim(strtr(base64_encode(json_encode($header)), '+/', '-_'), '=');
  $base64UrlPayload = rtrim(strtr(base64_encode(json_encode($payload)), '+/', '-_'), '=');
  $signature = hash_hmac('sha256', $base64UrlHeader . '.' . $base64UrlPayload, $secret, true);
  $base64UrlSignature = rtrim(strtr(base64_encode($signature), '+/', '-_'), '=');
  return $base64UrlHeader . '.' . $base64UrlPayload . '.' . $base64UrlSignature;
}

function upbit_get(string $endpoint, string $accessKey, string $secretKey, array $query = []): array {
  $nonce = uuid4();
  $payload = [
    'access_key' => $accessKey,
    'nonce' => $nonce,
  ];

  if (!empty($query)) {
    ksort($query);
    $qs = http_build_query($query, '', '&');
    $queryHash = hash('sha512', $qs);
    $payload['query_hash'] = $queryHash;
    $payload['query_hash_alg'] = 'SHA512';
  }

  $jwt = jwt_encode($payload, $secretKey);
  $authorization = 'Bearer ' . $jwt;

  $url = 'https://api.upbit.com' . $endpoint;
  if (!empty($query)) {
    $url .= '?' . http_build_query($query, '', '&');
  }

  $ch = curl_init($url);
  curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
      'Content-Type: application/json',
      'Authorization: ' . $authorization,
    ],
    CURLOPT_TIMEOUT => 10,
  ]);
  $resp = curl_exec($ch);
  $errno = curl_errno($ch);
  $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
  curl_close($ch);

  if ($errno !== 0) {
    throw new Exception('cURL Error: ' . $errno);
  }
  $json = json_decode($resp, true);
  if ($json === null) {
    throw new Exception('Invalid JSON from Upbit. HTTP ' . $httpCode . ' Body: ' . substr((string)$resp, 0, 200));
  }
  if ($httpCode >= 400) {
    throw new Exception('Upbit API error: HTTP ' . $httpCode . ' ' . json_encode($json));
  }
  return $json;
}

function public_ticker(string $market): array {
  $url = 'https://api.upbit.com/v1/ticker?markets=' . urlencode($market);
  $ch = curl_init($url);
  curl_setopt_array($ch, [CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 10]);
  $resp = curl_exec($ch);
  $errno = curl_errno($ch);
  $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
  curl_close($ch);
  if ($errno !== 0) {
    return [];
  }
  $json = json_decode($resp, true);
  return is_array($json) ? $json : [];
}

// ------------ Fetch balances ------------
try {
  $accounts = upbit_get('/v1/accounts', $ACCESS_KEY, $SECRET_KEY);

  // Compute profit summary based on BTC position (if present)
  $balances = [];
  $btcAssetValue = 0.0;
  $btcBalance = 0.0;
  $btcAvg = 0.0;

  foreach ($accounts as $acc) {
    $currency = $acc['currency'];
    $balance = (float)$acc['balance'];
    $locked = (float)$acc['locked'];
    $avgBuyPrice = isset($acc['avg_buy_price']) ? (float)$acc['avg_buy_price'] : 0.0;
    $balances[] = [
      'currency' => $currency,
      'balance' => $balance,
      'locked' => $locked,
      'avg_buy_price' => $avgBuyPrice,
      'asset_value' => 0,
    ];

    if ($currency === 'BTC') {
      $btcBalance = $balance;
      $btcAvg = $avgBuyPrice;
    }
  }

  if ($btcBalance > 0) {
    $ticker = public_ticker('KRW-BTC');
    if (!empty($ticker) && isset($ticker[0]['trade_price'])) {
      $price = (float)$ticker[0]['trade_price'];
      $btcAssetValue = $btcBalance * $price;
    }
  }

  $profitPercent = 0.902; // placeholder from UI
  $profitValue = $btcAssetValue * ($profitPercent / 100);

  echo json_encode([
    'balances' => $balances,
    'profitDetails' => [
      'profit' => $profitPercent,
      'btc_asset_value' => $btcAssetValue,
      'btc_25_percent_value' => $profitValue,
    ],
  ]);
} catch (Throwable $e) {
  http_response_code(500);
  echo json_encode(['error' => $e->getMessage()]);
}
?>

