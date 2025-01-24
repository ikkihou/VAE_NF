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
from utils.utils import imlocal, seed_everything, get_device, setup_logger


def load_MNIST(data_path):
    pass


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
    # 初始化累积值和计数器
    rvae.train()
    running_elbo = 0
    running_gen_loss = 0
    running_kl_div = 0
    num_samples = 0

    for _, data in enumerate(train_loader):
        data = data[0].to(rvae.device)
        elbo, log_p_x_g_z, kl_div = rvae(data)
        loss = -elbo
        loss.backward()
        rvae.optG.step()
        rvae.optG.zero_grad()

        # 获取当前批次的损失值
        batch_size = data.size(0)
        num_samples += batch_size

        # 直接累加损失值
        running_elbo += elbo.item() * batch_size
        running_gen_loss += (-log_p_x_g_z.item()) * batch_size
        running_kl_div += kl_div.item() * batch_size

        # Calculate running averages
        avg_elbo = running_elbo / num_samples
        avg_gen_loss = running_gen_loss / num_samples
        avg_kl_div = running_kl_div / num_samples

        # Print on same line with \r
        template = "\rTrain: [{}/{}] {:.1%}, ELBO={:.5f}, Recon_Loss={:.5f}, KL={:.5f}"
        print(
            template.format(
                current_epoch + 1,
                rvae.epoch,
                num_samples / len(train_loader.dataset),
                avg_elbo,
                avg_gen_loss,
                avg_kl_div,
            ),
            end="",
            file=sys.stderr,
        )

    # Add newline only at the end of epoch
    print(file=sys.stderr)
    return avg_elbo, avg_gen_loss, avg_kl_div


def eval_epoch(
    rvae: torch.nn.Module,
    val_loader: DataLoader,
    current_epoch: int,
    logger: logging.Logger,
) -> None:
    rvae.eval()
    running_elbo = 0
    running_gen_loss = 0
    running_kl_div = 0
    num_samples = 0

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            data = data[0].to(rvae.device)
            elbo, log_p_x_g_z, kl_div = rvae(data)
            loss = -elbo

            batch_size = data.size(0)
            num_samples += batch_size

            running_elbo += elbo.item() * batch_size
            running_gen_loss += (-log_p_x_g_z.item()) * batch_size
            running_kl_div += kl_div.item() * batch_size

            # Calculate running averages
            avg_elbo = running_elbo / num_samples
            avg_gen_loss = running_gen_loss / num_samples
            avg_kl_div = running_kl_div / num_samples

            # Print on same line with \r
            template = (
                "\rEval:  [{}/{}] {:.1%}, ELBO={:.5f}, Recon_Loss={:.5f}, KL={:.5f}"
            )
            print(
                template.format(
                    current_epoch + 1,
                    rvae.epoch,
                    num_samples / len(val_loader.dataset),
                    avg_elbo,
                    avg_gen_loss,
                    avg_kl_div,
                ),
                end="",
                file=sys.stderr,
            )

    # Add newline only at the end of epoch
    print(file=sys.stderr)
    return avg_elbo, avg_gen_loss, avg_kl_div


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

    # imgstack = (imgstack - imgstack.min()) / np.ptp(imgstack)
    
    ## DataLoader
    imstack_train, imstack_test = train_test_split(
        imgstack,
        test_size=0.15,
        shuffle=True,
        random_state=config["train"]["seed"],
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
        in_dim=config["model"]["in_dim"],
        z_dim=config["model"]["z_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        translation=config["prior"][
            "translation"
        ],  ## this requires explicitly set to be true
        dx_prior=config["prior"]["dx_prior"],
        theta_prior=config["prior"]["theta_prior"],
        optim=config["train"]["optim_type"],
        epoch=config["train"]["epochs"],
        init_lr=config["train"]["learning_rate"],
        activation=config["model"]["activation"],
        loss_type=config["train"]["loss_type"],
        device=get_device(),
    )

    # 自定义一个衰减函数，按 epoch 衰减
    def lr_lambda(epoch):
        return 0.98**epoch  # 每个 epoch 衰减 5%

    rvae.get_optim()
    scheduler = torch.optim.lr_scheduler.LambdaLR(rvae.optG, lr_lambda)
    logger.info("Model Initialization Finish")

    ## train
    for e in range(config["train"]["epochs"]):
        avg_elbo, avg_gen_loss, avg_kl_div = train_epoch(rvae, train_loader, e, logger)
        scheduler.step()
        if config["logging"]["wandb"]:
            wandb.log(
                {
                    "train_ELBO": avg_elbo,
                    "train_Recon_Error": avg_gen_loss,
                    "train_KL": avg_kl_div,
                },
                step=e + 1,
            )
        avg_elbo, avg_gen_loss, avg_kl_div = eval_epoch(rvae, val_loader, e, logger)
        if config["logging"]["wandb"]:
            wandb.log(
                {
                    "eval_ELBO": avg_elbo,
                    "eval_Recon_Error": avg_gen_loss,
                    "eval_KL": avg_kl_div,
                },
                step=e + 1,
            )
        if config["data"]["save_freq"] and (e + 1) % config["data"]["save_freq"] == 0:
            rvae.save_network(e + 1, current_exp_save_path)
            logger.info("Saving models and training states at epoch {}".format(e + 1))
    rvae.save_network(config["train"]["epochs"], current_exp_save_path)
    logger.info("End of training")


if __name__ == "__main__":
    main()
