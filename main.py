import matplotlib
matplotlib.use('TkAgg')  # 독립 실행형 창 사용

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PyAudio 초기화
p = pyaudio.PyAudio()

# 오디오 스트림 설정
stream = p.open(format=pyaudio.paInt16,  # 16비트 오디오
                channels=1,  # 모노 채널
                rate=44100,  # 샘플링 레이트 (Hz)
                input=True,  # 입력 장치 사용
                frames_per_buffer=1024,# 버퍼 크기
                input_device_index=1 # 장치 인덱스
                )

print("FFT Started...")

# 실시간 그래프 생성
fig, ax = plt.subplots()
x_data = np.fft.rfftfreq(1024, d=1 / 44100)
y_data = np.zeros(len(x_data))
line, = ax.plot(x_data, y_data)

# 그래프 업데이트 함수
def update_FFT(frame):
    data = stream.read(1024, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)
    fft_data = np.fft.rfft(audio_data)
    fft_magnitude = np.log1p(np.abs(fft_data))

    # y_data 업데이트
    line.set_ydata(fft_magnitude)

    # 플롯의 축 범위를 명시적으로 설정
    ax.set_ylim(0, np.max(fft_magnitude) * 1.1)  # y축 최대값 설정
    plt.xlabel("frequency")
    plt.ylabel("Amplitude(log scale)")
    return line,

# 애니메이션 설정
ani = FuncAnimation(fig, update_FFT, interval=50, blit=True, cache_frame_data=False)
plt.show()

try:
    while True:
        pass
except KeyboardInterrupt:
    print("프로세스를 종료합니다.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
