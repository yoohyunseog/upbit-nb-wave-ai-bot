import numpy as np
import wave
import struct

def create_beep_sound(frequency=800, duration=0.1, sample_rate=44100, volume=0.3):
    """간단한 비프음 생성"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t) * volume
    
    # 페이드 인/아웃 적용
    fade_samples = int(0.01 * sample_rate)  # 10ms 페이드
    if len(tone) > 2 * fade_samples:
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return tone

def save_wav(filename, audio_data, sample_rate=44100):
    """오디오 데이터를 WAV 파일로 저장"""
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # 모노
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # 16-bit 정수로 변환
        audio_int = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_int.tobytes())

def main():
    """각종 사운드 효과 생성"""
    print("오디오 파일 생성 중...")
    
    # 클릭 사운드 (800Hz, 100ms)
    click_sound = create_beep_sound(800, 0.1, volume=0.2)
    save_wav('click.wav', click_sound)
    print("click.wav 생성 완료")
    
    # 성공 사운드 (1000Hz, 200ms)
    success_sound = create_beep_sound(1000, 0.2, volume=0.25)
    save_wav('success.wav', success_sound)
    print("success.wav 생성 완료")
    
    # 에러 사운드 (400Hz, 300ms)
    error_sound = create_beep_sound(400, 0.3, volume=0.3)
    save_wav('error.wav', error_sound)
    print("error.wav 생성 완료")
    
    # 타이핑 사운드 (600Hz, 50ms)
    type_sound = create_beep_sound(600, 0.05, volume=0.15)
    save_wav('type.wav', type_sound)
    print("type.wav 생성 완료")
    
    print("모든 오디오 파일 생성 완료!")

if __name__ == "__main__":
    main()
