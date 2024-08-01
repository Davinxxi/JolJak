import soundfile as sf
from pesq import pesq
from pathlib import Path
import os
import numpy as np

clean_dir = Path("./Data/cv/Clean_test")
enhanced_dir = Path("./output/train_240704_1006/enhancements/best_checkpoint_46_epoch")
pesq_file = open("./output/train_240704_1006/enhancements/pesq_file2.txt", 'w')

noisy_files = os.listdir(enhanced_dir)
num_of_files = len(noisy_files)

avg_pesq = np.zeros(5, dtype='float32')
num_of_dB = np.zeros(5)

for noisy in noisy_files:
    n = noisy.split('_')
    clean = "".join(["clean_", n[1], ".wav"])

    ref, sr = sf.read(clean_dir / clean)
    deg, sr = sf.read(enhanced_dir / noisy)

    score = pesq(sr, ref, deg, 'wb')
    pesq_file.write(" ".join([noisy, str(score), '\n']))


    if n[-1] == "-5db.wav":
        avg_pesq[0] += score
        num_of_dB[0] += 1
    elif n[-1] == "0db.wav":
        avg_pesq[1] += score
        num_of_dB[1] += 1
    elif n[-1] == "5db.wav":
        avg_pesq[2] += score
        num_of_dB[2] += 1
    elif n[-1] == "10db.wav":
        avg_pesq[3] += score
        num_of_dB[3] += 1
    elif n[-1] == "15db.wav":
        avg_pesq[4] += score
        num_of_dB[4] += 1
    else:
        print("$$$$")

print(num_of_dB)
avg_pesq /= num_of_dB

pesq_file.write(str(avg_pesq))

pesq_file.close()