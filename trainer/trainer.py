import librosa
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt

import numpy as np
import librosa.display
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader,
                 device="cpu"):  # 기본적으로 device를 cpu로 설정
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.root_dir = (Path(config["save_location"]) / config["experiment_name"]).expanduser().absolute()
        self.checkpoints_dir = self.root_dir / "checkpoints"

    def _train_epoch(self, epoch):
        loss_total = 0.0
        
        for mixture, clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):

            self.optimizer.zero_grad()

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)

            clean_D = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

            enhanced_mag = self.model(mixture_mag)

            loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list, self.device)
            loss.backward()

            self.optimizer.step()

            loss_total += float(loss)

        save_path = "model_epoch%d" % epoch
        torch.save(self.model, self.checkpoints_dir / save_path)
        dataloader_len = len(self.train_dataloader)
        print("train loss = ", loss_total / dataloader_len)

        return loss_total / dataloader_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0

        for mixture, clean, n_frames_list, _ in tqdm(self.validation_dataloader, desc="Validation"):

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)

            clean_D = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

            enhanced_mag = self.model(mixture_mag)

            loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list, self.device)
            loss_total += float(loss)

        dataloader_len = len(self.validation_dataloader)
        print("validation loss = ", loss_total / dataloader_len)

        return loss_total / dataloader_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        mixture_mean = None
        mixture_std = None

        for mixture, clean, n_frames_list, _ in tqdm(self.validation_dataloader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D = self.stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)

            clean_D = self.stft.transform(clean)
            clean_real = clean_D[:, :, :, 0]
            clean_imag = clean_D[:, :, :, 1]
            clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

            enhanced_mag = self.model(mixture_mag)

            loss = self.loss_function(enhanced_mag, clean_mag, n_frames_list, self.device)
            loss_total += float(loss)  # validation loss

            # if self.z_score:
            #     enhanced_mag = reverse_z_score(enhanced_mag, mixture_mean, mixture_std)

        dataloader_len = len(self.validation_dataloader)
        # self.writer.add_scalar("Loss/Validation", loss_total / dataloader_len, epoch)
        print("validation loss = ", loss_total / dataloader_len)

        return loss_total / dataloader_len

