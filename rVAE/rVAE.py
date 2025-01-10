#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   rVAE.py
@Time    :   2025/01/06 20:44:51
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import math
import torch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from .types_ import Tensor, Variable, Union, List
from .base import BaseVAE
from .modules import SpatialGenerator, InferenceNetwork


class rVAE(BaseVAE):
    def __init__(
        self,
        in_dim: int = 576,  ## 24*24
        z_dim: int = 2,
        hidden_dim: int = 512,
        num_layers: int = 1,
        translation: bool = False,
        dx_prior: float = 1.0,
        theta_prior: float = 1.0,
        optim: str = "Adam",
        epoch: int = 100,
        init_lr: float = 1e-3,
        activation: str = "tanh",
        loss_type: str = "bce_logits",  ## "mse" or "bce_logits"
        device: str = "cpu",
    ):
        """rVAE model
        Args:
            in_dim (int, optional): input image dimension. Defaults to 576.
            hidden_dim (int, optional): hidden dimension. Defaults to 512.
            num_layers (int, optional): number of layers. Defaults to 1.
            translation (bool, optional): whether to use translation. Defaults to False.
            dx_prior (float, optional): translation prior. Defaults to 1.0.
            theta_prior (float, optional): rotation prior. Defaults to 1.0.
            optim (str, optional): optimizer. Defaults to "Adam". options: "Adam", "SGD"
            epoch (int, optional): number of epochs. Defaults to 100.
            init_lr (float, optional): learning rate. Defaults to 1e-3.
            activation (str, optional): activation function. Defaults to "tanh". options: "tanh", "relu"
            loss_type (str, optional): loss type. Defaults to "bce_logits". options: "mse", "bce_logits".
            device (str, optional): device. Defaults to "cpu". options: "cpu", "mps", "cuda".
        """
        super(rVAE, self).__init__()

        self.in_dim = in_dim
        self.z_dim = z_dim

        inf_dim = z_dim + 1  ## rotation
        if translation:  ## translation
            inf_dim += 2

        self.translation = translation
        self.theta_prior = theta_prior
        self.dx_prior = dx_prior

        self.optim_type = optim
        self.init_lr = init_lr
        self.epoch = epoch
        self.loss_type = loss_type
        self.device = device
        self.optG = None  # 初始化优化器变量为 None, 训练时必须显式调用

        self.encoder = InferenceNetwork(
            n=in_dim,
            latent_dim=inf_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            act_func=activation,
        ).to(self.device)

        self.decoder = SpatialGenerator(
            z_dim,
            hidden_dim,
            num_layers,
            act_func=activation,
        ).to(self.device)

    def get_optim(self):
        """初始化优化器并保存为类成员变量"""
        if self.optG is None:  # 如果优化器还未创建
            if self.optim_type == "Adam":
                self.optG = Adam(self.parameters(), lr=self.init_lr, weight_decay=1e-5)
            elif self.optim_type == "SGD":
                self.optG = SGD(self.parameters(), lr=self.init_lr, weight_decay=1e-5)
            else:
                raise ValueError(f"Optimizer {self.optim_type} not supported")
        return self.optG

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)  ## [z_mu, z_logstd], both (batch, inf_dim)

    def decode(self, x_coord: Tensor, z: Tensor) -> Tensor:
        return self.decoder(x_coord, z)  ## (batch, n*m)

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        coord = self.img2coord().to(self.device)
        coord = coord.expand(num_samples, -1, -1)
        samples = self.decode(coord.contiguous(), z)  ## (num_samples, n*m)
        return samples

    def reparameterize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        return torch.randn_like(logstd) * torch.exp(logstd) + mu

    def generate(self, x_test: Tensor, **kwargs) -> Tensor:
        recons = self.forward(x_test)
        return recons

    def forward(self, y: Tensor) -> Tensor:  ## x_coord(n*m, 2)
        b = y.size(0)
        x_coord = self.img2coord().to(self.device)
        x_coord_ = x_coord.expand(b, *x_coord.size())

        z_mu, z_logstd = self.encode(y)
        z_std = torch.exp(z_logstd)

        z = self.reparameterize(z_mu, z_logstd)

        kl_div = 0
        theta_mu = z_mu[:, 0]  ## rotation
        theta_std = z_std[:, 0]
        theta_logstd = z_logstd[:, 0]
        theta = z[:, 0]
        z = z[:, 1:]
        z_mu = z_mu[:, 1:]
        z_std = z_std[:, 1:]
        z_logstd = z_logstd[:, 1:]

        rot = Variable(theta.data.new(b, 2, 2).zero_())
        rot[:, 0, 0] = torch.cos(theta)
        rot[:, 0, 1] = torch.sin(theta)
        rot[:, 1, 0] = -torch.sin(theta)
        rot[:, 1, 1] = torch.cos(theta)

        x_coord_ = torch.bmm(x_coord_, rot)

        ## Calculate the rotation KL divergence term
        sigma = torch.tensor(self.theta_prior).to(self.device)
        kl_div = -theta_logstd + torch.log(sigma) + (theta_std**2) / 2 / sigma**2 - 0.5

        if self.translation:
            # dx_mu = z_mu[:, :2]
            # dx_std = z_std[:, :2]
            # dx_logstd = z_logstd[:, :2]
            dx = z[:, :2] * self.dx_prior
            dx = dx.unsqueeze(1)
            z = z[:, 2:]

            x_coord_ = x_coord_ + dx

        y_hat = self.decode(x_coord_.contiguous(), z)
        y_hat = y_hat.view(b, -1)

        if self.loss_type == "bce_logits":
            size = y.size(1)
            log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y) * size
        elif self.loss_type == "mse":
            log_p_x_g_z = -(
                0.5 * torch.sum((y_hat - y) ** 2, 1)
            ).mean()  ## likelihood, not recon_loss
        else:
            raise ValueError(f"Loss type {self.loss_type} not supported")

        z_kl = -z_logstd + 0.5 * z_std**2 + 0.5 * z_mu**2 - 0.5
        kl_div = kl_div + torch.sum(z_kl, 1)
        kl_div = kl_div.mean()

        elbo = log_p_x_g_z - kl_div

        return elbo, log_p_x_g_z, kl_div

    def loss_function(
        self, elbo: Tensor, log_p_x_g_z: Tensor, kl_div: Tensor
    ) -> Tensor:
        return {
            "loss": -elbo,
            "Reconstruction_Loss": log_p_x_g_z,
            "KLD": kl_div,
        }

    def img2coord(self) -> Tensor:
        m = n = int(math.sqrt(self.in_dim))
        xgrid = np.linspace(-1, 1, m)
        ygrid = np.linspace(1, -1, n)
        x0, x1 = np.meshgrid(xgrid, ygrid)
        x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
        x_coord = torch.from_numpy(x_coord).float()
        return x_coord

    def manifold2d(self, **kwargs: Union[int, List, str, bool]) -> None:
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        in_dim = (int(math.sqrt(self.in_dim)), int(math.sqrt(self.in_dim)))
        figure = np.zeros((in_dim[0] * d, in_dim[1] * d))

        grid_x = norm.ppf(np.linspace(0.95, 0.05, d))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = np.array([xi, yi])
                x_coord = self.img2coord()
                imdec = self.decode(
                    x_coord.contiguous(),
                    torch.tensor(z_sample.astype(np.float32)).unsqueeze(0),
                )
                figure[
                    i * in_dim[0] : (i + 1) * in_dim[0],
                    j * in_dim[1] : (j + 1) * in_dim[1],
                ] = (
                    imdec.detach().cpu().numpy().reshape(in_dim[0], in_dim[0])
                )
        if figure.min() < 0:
            figure = (figure - figure.min()) / figure.ptp()

        ## plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            figure,
            cmap=cmap,
            origin=kwargs.get("origin", "lower"),
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        )
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        draw_grid = kwargs.get("draw_grid")
        if draw_grid:
            major_ticks_x = np.arange(0, d * in_dim[0], in_dim[0])
            major_ticks_y = np.arange(0, d * in_dim[1], in_dim[1])
            ax.set_xticks(major_ticks_x)
            ax.set_yticks(major_ticks_y)
            ax.grid(which="major", alpha=0.6)
        for item in (
            [ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(18)

        if not kwargs.get("savefig"):
            plt.show()
        else:
            savedir = kwargs.get("savedir", "./vae_learning/")
            fname = kwargs.get("filename", "manifold_2d")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            fig.savefig(os.path.join(savedir, "{}.png".format(fname)))
            plt.close(fig)
        return figure

    def save_network(self, current_step: int, save_dir: str):
        gen_path = os.path.join(save_dir, "E{}_gen.pth".format(current_step))
        opt_path = os.path.join(save_dir, "E{}_opt.pth".format(current_step))
        torch.save(self.state_dict(), gen_path)
        opt_state = {
            "epoch": self.epoch,
            "iter": current_step,
            "scheduler": None,
            "optimizer": None,
        }
        opt_state["optimizer"] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
