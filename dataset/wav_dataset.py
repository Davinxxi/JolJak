import os

import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


class WavDataset(Dataset):
    """
    Define train dataset
    """

    def __init__(self,
                 mixture_dataset,
                 clean_dataset,
                 limit=None,
                 offset=0,
                 ):
        """
        Construct train dataset
        Args:
            mixture_dataset (str): mixture dir (wav format files)
            clean_dataset (str): clean dir (wav format files)
            limit (int): the limit of the dataset
            offset (int): the offset of the dataset
        """
        mixture_dataset = os.path.abspath(mixture_dataset) 
        clean_dataset = os.path.abspath(os.path.expanduser(clean_dataset)) 
        
        print(mixture_dataset)
        print(clean_dataset)
        
        assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)

        print("Search datasets...")
        mixture_wav_files = librosa.util.find_files(mixture_dataset, ext="wav", limit=limit, offset=offset)
        clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
        mixture_wav_files.sort()
        clean_wav_files.sort()

        print(f"\t Original length: {len(mixture_wav_files)}")

        self.length = len(mixture_wav_files)
        self.mixture_wav_files = mixture_wav_files  # string list
        self.clean_wav_files = clean_wav_files      # string list

        print(f"\t Offset: {offset}")
        print(f"\t Limit: {limit}")
        print(f"\t Final length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):


        mixture_path = self.mixture_wav_files[item]
        clean_path = self.clean_wav_files[item]
        noisy_name = os.path.splitext(os.path.basename(mixture_path))[0]
        noisy_cnt = noisy_name.split('_')[1]

        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
        clean_cnt = clean_name.split('_')[1]

        assert noisy_cnt == clean_cnt

        mixture, sr_mixture = sf.read(mixture_path, dtype="float32")
        clean, sr_clean = sf.read(clean_path, dtype="float32")

        # Check and resample if necessary
        if sr_mixture != 16000:
            mixture = librosa.resample(mixture, orig_sr=sr_mixture, target_sr=16000)
        if sr_clean != 16000:
            clean = librosa.resample(clean, orig_sr=sr_clean, target_sr=16000)

        assert mixture.shape == clean.shape, f"Shape mismatch: {mixture.shape} vs {clean.shape}"

        n_frames = (len(mixture) - 320) // 160 + 1

        return mixture, clean, n_frames, clean_name
