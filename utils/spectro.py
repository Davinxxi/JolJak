import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 폴더 경로 설정
noisy_folder = "C:/Users/user/Desktop/CRNN_1/Data/cv/Noisy_test"
enhancements_folder = "C:/Users/user/Desktop/CRNN_1/output/train_230217_1939/enhancements/best_checkpoint_36_epoch"
clean_folder = "C:/Users/user/Desktop/CRNN_1/Data/cv/Clean_test"

# 파일 목록 가져오기
noisy_files = os.listdir(noisy_folder)[:20]  # 폴더에서 처음부터 20개 파일만 선택
enhancements_files = os.listdir(enhancements_folder)[:20]
clean_files = os.listdir(clean_folder)[:20]

# 20개의 Figure 생성
for i in range(20):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Spectrogram 그리기
    def plot_spectrogram(ax, wav_file_path, title):
        y, sr = librosa.load(wav_file_path)
        spectrogram = np.abs(librosa.stft(y))
        ax.imshow(librosa.amplitude_to_db(spectrogram, ref=np.max), cmap='viridis', origin='lower', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

    # Noisy 파일 그리기
    wav_file_path = os.path.join(noisy_folder, noisy_files[i])
    plot_spectrogram(axs[0], wav_file_path, "Noise")

    # Enhancements 파일 그리기
    wav_file_path = os.path.join(enhancements_folder, enhancements_files[i])
    plot_spectrogram(axs[1], wav_file_path, "Enhancements")

    # Clean 파일 그리기
    wav_file_path = os.path.join(clean_folder, clean_files[i])
    plot_spectrogram(axs[2], wav_file_path, "Clean")

    plt.tight_layout()
    plt.show()
