#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2025/01/06 22:21:42
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import yaml

import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from rVAE import rVAE
from torch.optim import Adam
from utils import imlocal, seed_everything, get_device, setup_logger


def load_dynamic_transition_data(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from the given path.
    """
    STEM_real = np.load(os.path.join(data_path, "3DStack13-1-exp.npy"))
    decoded_imgs = np.load(os.path.join(data_path, "3DStack13-1-dec.npy"))
    lattice_coord = np.load(
        os.path.join(data_path, "3DStack13-1-coord.npy"), allow_pickle=True
    )[()]
    return STEM_real, decoded_imgs, lattice_coord


def train_epoch(
    rvae: torch.nn.Module,
    train_loader: DataLoader,
    current_epoch: int,
    logger: logging.Logger,
) -> None:
    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0
    for _, data in enumerate(train_loader):
        rvae.optG.zero_grad()
        data = data[0].to(rvae.device)
        elbo, log_p_x_g_z, kl_div = rvae(data)
        loss = -elbo
        loss.backward()
        rvae.optG.step()

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        b = data.size(0)
        c += b
        delta = b * (gen_loss - gen_loss_accum)
        gen_loss_accum += delta / c

        delta = b * (elbo - elbo_accum)
        elbo_accum += delta / c

        delta = b * (kl_loss - kl_loss_accum)
        kl_loss_accum += delta / c

        template = "# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}"
        line = template.format(
            current_epoch + 1,
            rvae.epoch,
            c / len(train_loader.dataset),
            elbo_accum,
            gen_loss_accum,
            kl_loss_accum,
        )
        print(line, end="\r", file=sys.stderr)

    # logger.info(
    #     f"[{current_epoch + 1}/{rvae.epoch}], ELBO={elbo_accum:.5f}, Error={gen_loss_accum:.5f}, KL={kl_loss_accum:.5f}"
    # )

    return elbo_accum, gen_loss_accum, kl_loss_accum


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="rotation-invarient VAE experiment")
    parser.add_argument(
        "--config", type=str, default="config/rVAE.yaml", help="path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config["logging"]["wandb"]:
        wandb.init(
            project="VAE_NF",
            config=config,
        )
    seed_everything(config["train"]["seed"])

    ## logging
    curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    current_exp_save_path = os.path.join(
        config["data"]["save_path"], "rVAE", config["data"]["phase"], curr_time
    )
    logger = setup_logger(
        config["logging"]["logger_name"],
        current_exp_save_path,
        config["data"]["phase"],
        level=logging.INFO,
        screen=True,
    )

    ## Load data
    STEM_real, decoded_imgs, lattice_coord = load_dynamic_transition_data(
        config["data"]["data_path"]
    )

    C_cnt = 0
    for i in lattice_coord.keys():
        coord_arr = lattice_coord[i]
        inner_cnt = len(coord_arr) - np.count_nonzero(coord_arr[:, -1])
        C_cnt += inner_cnt

    print(f"There are {C_cnt} carbon atoms in all STEM frames")

    s = imlocal(
        np.sum(decoded_imgs[..., :-1], -1)[..., None],
        lattice_coord,
        24,
        0,
    )
    imgstack, imgstack_com, imgstack_frm = s.imgstack, s.imgstack_com, s.imgstack_frames

    ## DataLoader
    imstack_train, imstack_test = train_test_split(
        imgstack,
        test_size=0.15,
        shuffle=True,
        random_state=42,
    )

    train_dataset, val_dataset = TensorDataset(
        torch.from_numpy(
            imstack_train.transpose(0, 3, 1, 2).reshape(len(imstack_train), -1)
        )
    ), TensorDataset(
        torch.from_numpy(
            imstack_test.transpose(0, 3, 1, 2).reshape(len(imstack_test), -1)
        ),
    )
    logger.info("Initial Dataset Finish")
    logger.info(f"train dataset length: {len(train_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )

    ## Model
    rvae = rVAE(
        in_dim=train_dataset[0][0].size(0),
        z_dim=config["model"]["z_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        init_lr=config["train"]["learning_rate"],
        translation=config["prior"][
            "translation"
        ],  ## this requires explicitly set to be true
        theta_prior=config["prior"]["theta_prior"],
        dx_prior=config["prior"]["dx_prior"],
        activation=config["model"]["activation"],
        device=get_device(),
    )
    rvae.get_optim()
    logger.info("Model Initialization Finish")

    ## train
    for e in range(config["train"]["epochs"]):
        elbo_accum, gen_loss_accum, kl_loss_accum = train_epoch(
            rvae, train_loader, e, logger
        )
        if config["logging"]["wandb"]:
            wandb.log(
                {
                    "ELBO": elbo_accum,
                    "Error": gen_loss_accum,
                    "KL": kl_loss_accum,
                },
                step=e + 1,
            )
        if config["data"]["save_freq"] and (e + 1) % config["data"]["save_freq"] == 0:
            rvae.save_network(e, current_exp_save_path)
            logger.info("Saving models and training states at epoch {}".format(e + 1))
    rvae.save_network(config["train"]["epochs"], current_exp_save_path)
    logger.info("End of training")


if __name__ == "__main__":
    main()
