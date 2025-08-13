<?php
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $filename = isset($_POST['filename']) ? $_POST['filename'] : 'default.txt';
    $content = isset($_POST['content']) ? $_POST['content'] : '';

    $filePath = __DIR__ . '/files/' . $filename;

    if (file_put_contents($filePath, $content) !== false) {
        echo json_encode(['message' => 'File saved successfully', 'filePath' => $filePath]);
    } else {
        http_response_code(500);
        echo json_encode(['message' => 'Failed to save file']);
    }
} else {
    http_response_code(405);
    echo json_encode(['message' => 'Method not allowed']);
}
?>
