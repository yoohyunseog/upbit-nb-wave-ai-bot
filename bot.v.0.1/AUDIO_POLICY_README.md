# 🎵 8BIT Trading System - Audio Policy Fix

브라우저의 자동재생 정책으로 인해 소리가 나오지 않는 문제를 해결하는 도구입니다.

## 📋 문제 상황

- 첫 번째 소리만 나오고 그 이후로 안 나옴
- 브라우저에서 "사용자 상호작용 없이 오디오 재생 불가" 오류
- 시스템 소리 정책이 제한되어 있음

## 🔧 해결 방법

### 방법 1: PowerShell 스크립트 사용 (권장)

#### 1단계: 관리자 권한으로 PowerShell 실행
```
Windows 키 + X → "Windows PowerShell (관리자)"
```

#### 2단계: 스크립트 실행
```powershell
# 현재 디렉토리로 이동
cd "E:\Gif\www\hankookin.center\8BIT\bot.v.0.1"

# 정책 적용 (백업 포함)
.\fix-audio-policy.ps1 -Apply

# 또는 백업만 생성
.\fix-audio-policy.ps1 -Backup

# 현재 상태 확인
.\fix-audio-policy.ps1
```

#### 3단계: 브라우저 재시작
모든 브라우저를 완전히 종료하고 다시 실행

### 방법 2: 레지스트리 파일 직접 실행

#### 1단계: 레지스트리 파일 실행
```
registry-fix.reg 파일을 더블클릭
```

#### 2단계: 확인 메시지에서 "예" 클릭

#### 3단계: 브라우저 재시작

## 📁 파일 설명

- `fix-audio-policy.ps1` - PowerShell 스크립트 (안전한 방법)
- `registry-fix.reg` - 레지스트리 파일 (직접 실행)
- `AUDIO_POLICY_README.md` - 이 파일

## ⚠️ 주의사항

1. **관리자 권한 필요**: 레지스트리 변경은 관리자 권한이 필요합니다.
2. **백업 권장**: 변경 전에 반드시 백업을 만드세요.
3. **브라우저 재시작**: 변경 후 모든 브라우저를 재시작해야 합니다.
4. **보안 정책**: 회사 정책이 있는 경우 IT 관리자와 상의하세요.

## 🔍 적용된 정책

### Edge 브라우저
```
HKCU\SOFTWARE\Policies\Microsoft\Edge
- AutoplayAllowed = 1
- AutoplayAllow = 1
```

### Chrome 브라우저
```
HKCU\SOFTWARE\Policies\Google\Chrome
- AutoplayAllowed = 1
- AutoplayAllow = 1
```

### Firefox 브라우저
```
HKCU\SOFTWARE\Policies\Mozilla\Firefox
- AutoplayAllowed = 1
- AutoplayAllow = 1
```

### Windows 미디어
```
HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\Explorer
- NoAutoplayfornonVolume = 0
```

## 🚀 테스트 방법

1. 정책 적용 후 브라우저 재시작
2. `http://127.0.0.1:5057/ui` 접속
3. 사운드 테스트 버튼 클릭
4. 모든 사운드가 정상 재생되는지 확인

## 🔄 복원 방법

만약 문제가 발생하면 백업 파일을 실행하여 원래 상태로 복원할 수 있습니다:

```powershell
# 백업 파일 실행
.\audio-policy-backup-YYYYMMDD-HHMMSS.reg
```

## 📞 문제 해결

여전히 소리가 나오지 않는다면:

1. 브라우저 설정에서 사이트 권한 확인
2. Windows 사운드 설정 확인
3. 브라우저 확장 프로그램 비활성화
4. 다른 브라우저로 테스트

## ✅ 성공 확인

정책이 제대로 적용되면:
- 모든 브라우저에서 사운드가 정상 재생
- 첫 번째 소리 이후에도 계속 재생
- 타이핑 효과음도 정상 작동

---

**🎵 이제 8BIT Trading System의 모든 사운드가 정상적으로 작동할 것입니다!**
