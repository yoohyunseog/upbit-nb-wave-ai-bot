# ===== 8BIT Trading System - Audio Policy Fix Script =====
# 이 스크립트는 브라우저의 자동재생 정책을 안전하게 완화합니다.
# 관리자 권한으로 실행해야 합니다.

param(
    [switch]$Backup,
    [switch]$Restore,
    [switch]$Apply
)

# 관리자 권한 확인
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# 백업 생성
function Backup-Registry {
    Write-Host "🔧 레지스트리 백업 생성 중..." -ForegroundColor Yellow
    
    $backupPath = ".\audio-policy-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').reg"
    
    $registryPaths = @(
        "HKCU:\SOFTWARE\Policies\Microsoft\Edge",
        "HKCU:\SOFTWARE\Policies\Google\Chrome", 
        "HKCU:\SOFTWARE\Policies\Mozilla\Firefox",
        "HKLM:\SOFTWARE\Policies\Microsoft\Edge",
        "HKLM:\SOFTWARE\Policies\Google\Chrome",
        "HKLM:\SOFTWARE\Policies\Mozilla\Firefox"
    )
    
    $backupContent = "Windows Registry Editor Version 5.00`n`n"
    
    foreach ($path in $registryPaths) {
        if (Test-Path $path) {
            $backupContent += "; $path`n"
            $backupContent += "[$path]`n"
            
            Get-ItemProperty $path | ForEach-Object {
                $_.PSObject.Properties | Where-Object { $_.Name -notlike "PS*" } | ForEach-Object {
                    if ($_.PropertyType -eq "DWord") {
                        $backupContent += "`"$($_.Name)`"=dword:$($_.Value.ToString('X8'))`n"
                    } else {
                        $backupContent += "`"$($_.Name)`"=`"$($_.Value)`"`n"
                    }
                }
            }
            $backupContent += "`n"
        }
    }
    
    $backupContent | Out-File -FilePath $backupPath -Encoding ASCII
    Write-Host "✅ 백업 완료: $backupPath" -ForegroundColor Green
}

# 정책 적용
function Apply-AudioPolicy {
    Write-Host "🎵 오디오 자동재생 정책 적용 중..." -ForegroundColor Yellow
    
    # Edge 브라우저 정책
    Write-Host "  - Edge 브라우저 정책 설정..." -ForegroundColor Cyan
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Microsoft\Edge" -Name "AutoplayAllowed" -Value 1 -PropertyType DWord -Force | Out-Null
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Microsoft\Edge" -Name "AutoplayAllow" -Value 1 -PropertyType DWord -Force | Out-Null
    
    # Chrome 브라우저 정책
    Write-Host "  - Chrome 브라우저 정책 설정..." -ForegroundColor Cyan
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Google\Chrome" -Name "AutoplayAllowed" -Value 1 -PropertyType DWord -Force | Out-Null
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Google\Chrome" -Name "AutoplayAllow" -Value 1 -PropertyType DWord -Force | Out-Null
    
    # Firefox 브라우저 정책
    Write-Host "  - Firefox 브라우저 정책 설정..." -ForegroundColor Cyan
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Mozilla\Firefox" -Name "AutoplayAllowed" -Value 1 -PropertyType DWord -Force | Out-Null
    New-ItemProperty -Path "HKCU:\SOFTWARE\Policies\Mozilla\Firefox" -Name "AutoplayAllow" -Value 1 -PropertyType DWord -Force | Out-Null
    
    # Windows 미디어 정책
    Write-Host "  - Windows 미디어 정책 설정..." -ForegroundColor Cyan
    New-ItemProperty -Path "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer" -Name "NoAutoplayfornonVolume" -Value 0 -PropertyType DWord -Force | Out-Null
    
    Write-Host "✅ 오디오 정책 적용 완료!" -ForegroundColor Green
    Write-Host "🔄 브라우저를 재시작해주세요." -ForegroundColor Yellow
}

# 정책 확인
function Test-AudioPolicy {
    Write-Host "🔍 현재 오디오 정책 상태 확인..." -ForegroundColor Yellow
    
    $policies = @(
        @{Path="HKCU:\SOFTWARE\Policies\Microsoft\Edge"; Name="AutoplayAllowed"},
        @{Path="HKCU:\SOFTWARE\Policies\Google\Chrome"; Name="AutoplayAllowed"},
        @{Path="HKCU:\SOFTWARE\Policies\Mozilla\Firefox"; Name="AutoplayAllowed"}
    )
    
    foreach ($policy in $policies) {
        if (Test-Path $policy.Path) {
            $value = Get-ItemProperty -Path $policy.Path -Name $policy.Name -ErrorAction SilentlyContinue
            if ($value) {
                Write-Host "  ✅ $($policy.Path) - $($policy.Name): $($value.$($policy.Name))" -ForegroundColor Green
            } else {
                Write-Host "  ❌ $($policy.Path) - $($policy.Name): 설정되지 않음" -ForegroundColor Red
            }
        } else {
            Write-Host "  ⚠️  $($policy.Path): 경로가 존재하지 않음" -ForegroundColor Yellow
        }
    }
}

# 메인 실행
Write-Host "🎵 8BIT Trading System - Audio Policy Fix" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta

if (-not (Test-Administrator)) {
    Write-Host "❌ 이 스크립트는 관리자 권한으로 실행해야 합니다." -ForegroundColor Red
    Write-Host "   PowerShell을 관리자 권한으로 실행한 후 다시 시도해주세요." -ForegroundColor Yellow
    exit 1
}

if ($Backup) {
    Backup-Registry
} elseif ($Restore) {
    Write-Host "🔄 백업 파일에서 복원하려면 백업 파일을 직접 실행하세요." -ForegroundColor Yellow
} elseif ($Apply) {
    Backup-Registry
    Apply-AudioPolicy
    Test-AudioPolicy
} else {
    Write-Host "사용법:" -ForegroundColor Cyan
    Write-Host "  -Backup    : 현재 설정 백업" -ForegroundColor White
    Write-Host "  -Apply     : 정책 적용 (백업 포함)" -ForegroundColor White
    Write-Host "  -Restore   : 복원 안내" -ForegroundColor White
    Write-Host ""
    Write-Host "예시: .\fix-audio-policy.ps1 -Apply" -ForegroundColor Yellow
}

Write-Host "================================================" -ForegroundColor Magenta
