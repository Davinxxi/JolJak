import argparse
import json
import os

parser = argparse.ArgumentParser(description='EHNET')
parser.add_argument("-C", "--config", default="config/train/train.json", type=str,
                    help="Specify the configuration file for training (*.json).")
parser.add_argument('-D', '--device', default=None, type=str,
                    help="Specify the GPU visible in the experiment, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true",
                    help="Whether to resume training from a recent breakpoint.")
args = parser.parse_args()

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.trainer import Trainer
from utils.utils import initialize_config
from torch.nn.utils.rnn import pad_sequence

from datetime import datetime

def pad_to_longest(batch):
        mixture_list = []
        clean_list = []
        names = []
        n_frames_list = []

        for mixture, clean, n_frames, name in batch:
            mixture_list.append(torch.tensor(mixture).reshape(-1, 1))
            clean_list.append(torch.tensor(clean).reshape(-1, 1))
            n_frames_list.append(n_frames)
            names.append(name)


        mixture_list = pad_sequence(mixture_list).squeeze(2).permute(1, 0)
        clean_list = pad_sequence(clean_list).squeeze(2).permute(1, 0)

        return mixture_list, clean_list, n_frames_list, names

def main(config, resume):

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_dataset = initialize_config(config["train_dataset"])
    train_data_loader = DataLoader(
        shuffle=config["train_dataloader"]["shuffle"],
        dataset=train_dataset,
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        collate_fn=pad_to_longest,
        drop_last=True
    )

    validation_dataset = initialize_config(config["validation_dataset"])
    valid_data_loader = DataLoader(
        shuffle=config["validation_dataloader"]["shuffle"],
        dataset=validation_dataset,
        num_workers=config["validation_dataloader"]["num_workers"],
        batch_size=config["validation_dataloader"]["batch_size"],
        collate_fn=pad_to_longest,
    )

    model = initialize_config(config["model"])


    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer = Trainer(
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_dataloader=train_data_loader,
        validation_dataloader=valid_data_loader,
        device="cpu"
    )

    trainer.train()


if __name__ == '__main__':

    config = json.load(open(args.config))
    now = datetime.now()
    config["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]+now.strftime('_%y%m%d_%H%M')
    main(config, resume=args.resume)



