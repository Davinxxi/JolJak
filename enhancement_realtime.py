import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stft import STFT
from utils.utils import initialize_config


def main(config, epoch):
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements"
    checkpoints_dir = root_dir / "checkpoints"

    """==========================="""
    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
    )

    model = initialize_config(config["model"])

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    stft = STFT(
        filter_length=320,
        hop_length=160
    ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()

    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"real_time_{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"real_time_checkpoint_{epoch}_epoch"

    results_dir.mkdir(parents=True, exist_ok=True)

    filter_length = 320
    hop_length = 160
    min_frame = 15
    mixture_mag = []
    enhanced = []

    for i, (mixture, _, _, names) in enumerate(dataloader):
        print(f"Enhance {i + 1}th speech")
        name = names[0]

        # print("####", mixture.shape)  # torch.Size([1, 153536])
        # print("####", names)          # ('noisy_00001_noise_43_10dB',)

        total_frame = mixture.shape[1]//hop_length-1
        mixture = mixture[:, :(total_frame+1)*hop_length]
        min_frame_len = (min_frame + 1) * hop_length

        for i in range(70):
            # mixture_seg = mixture[:, i*hop_length: i*hop_length+min_frame_len]
            mixture_seg = mixture[:, i * (min_frame_len-320): i * (min_frame_len-320) + min_frame_len]

            # Mixture mag and Clean mag
            # print("\tSTFT...")

            mixture_D = stft.transform(mixture_seg)

            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)  # [1, T, F]

            # if i == 0:
            #     mixture_real_cat = mixture_real
            #     mixture_imag_cat = mixture_imag
            #     mixture_mag_cat = mixture_mag
            # else:
            #     mixture_real_cat = torch.cat((mixture_real_cat[:, :, :], mixture_real[:, -1:, :]), 1)
            #     mixture_imag_cat = torch.cat((mixture_imag_cat[:, :, :], mixture_imag[:, -1:, :]), 1)
            #     mixture_mag_cat = torch.cat((mixture_mag_cat.squeeze(1)[:, :, :], mixture_mag.squeeze(1)[:, -1:, :]), 1)

            # mixture_D = stft.transform(mixture)
            # mixture_real = mixture_D[:, :, :, 0]
            # mixture_imag = mixture_D[:, :, :, 1]
            # mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2)  # [1, T, F]

            # print("\tEnhancement...")
            # enhanced_mag = model(mixture_mag_chunk).detach().cpu().unsqueeze(0)  # [1, T, F]
            # enhanced_mag = model(mixture_mag).detach().cpu().unsqueeze(0)  # [1, T, F]
            enhanced_mag = model(mixture_mag).detach().cpu().unsqueeze(0)  # [1, T, F]

            # if i==0:
            #     test_mag = enhanced_mag
            # else:
            #     test_mag = torch.cat((test_mag[:, :, :], enhanced_mag[:, 4:5, :]), 1)

            # enhanced_mag = enhanced_mag.detach().cpu().data.numpy()
            # mixture_mag = mixture_mag.cpu()

        # enhanced_real = test_mag * mixture_real_cat[:, :test_mag.size(1), :] / mixture_mag_cat.unsqueeze(1)[:, :test_mag.size(1), :]
        # enhanced_imag = test_mag * mixture_imag_cat[:, :test_mag.size(1), :] / mixture_mag_cat.unsqueeze(1)[:, :test_mag.size(1), :]

            enhanced_real = enhanced_mag * mixture_real[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            enhanced_imag = enhanced_mag * mixture_imag[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            enhanced_D = torch.stack([enhanced_real, enhanced_imag], 3).squeeze(0).permute(0, 1, 3, 2)

            # enhanced = stft.inverse(enhanced_D)
            enhanced_seg = stft.inverse(enhanced_D)

            if i == 0:
                enhanced = enhanced_seg
                # enhanced = enhanced_seg[:, 0:160*9]
            else:
                enhanced = torch.cat((enhanced, enhanced_seg[:, 160:-160]), 1)
                # enhanced = torch.cat((enhanced, enhanced_seg[:, 160*8:160*9]), 1)

        enhanced_wav = enhanced.detach().cpu().squeeze().numpy()


        # sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)
        sf.write(f"{results_dir}/{name}.wav", enhanced_wav, 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    parser.add_argument("-C", "--config", default="config/enhancement/enhancement.json", type=str,
                        help="Specify the configuration file for enhancement (*.json).")
    parser.add_argument("-E", "--epoch", default="best",
                        help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    main(config, args.epoch)
