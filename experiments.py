#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   experiments.py
@Time    :   2025/01/06 22:12:41
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import os

import torch
from torch import optim
from rVAE import BaseVAE
from rVAE.types_ import *

# from utils import data_loader
import pytorch_lightning as pl
import torchvision.utils as vutils
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import platform


def get_device():
    if platform.system() == "Darwin":
        # macOS 操作系统
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device on macOS.")
        else:
            device = torch.device("cpu")
            print("MPS device not available on macOS. Falling back to CPU.")
    else:
        # 其他操作系统
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            device = torch.device("cpu")
            print("CUDA device not available. Falling back to CPU.")
    return device


DEVICE = get_device()


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = DEVICE
        self.hold_graph = False
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img = batch[0]
        self.curr_device = self.curr_device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
            # optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img = batch[0]
        self.curr_device = self.curr_device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    def on_validation_end(self) -> None:
        self.sample_images()

    def manifold2d(self, **kwargs: Union[int, List, str, bool]) -> None:
        self.model.manifold2d(**kwargs)

    def sample_images(self):
        # Get sample reconstruction image
        test_input = next(iter(self.trainer.val_dataloaders))[0]
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
            normalize=True,
            nrow=12,
        )

        try:
            samples = self.model.sample(144, self.curr_device)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir,
                    "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png",
                ),
                normalize=True,
                nrow=12,
            )
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params["LR_2"] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.params["submodel"]).parameters(),
                    lr=self.params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params["scheduler_gamma"] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.params["scheduler_gamma"]
                )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params["scheduler_gamma_2"] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.params["scheduler_gamma_2"]
                        )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
