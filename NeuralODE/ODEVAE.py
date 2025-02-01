#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   ODEVAE.py
@Time    :   2025/01/19 00:51:10
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint

from utils.coords import imcoordgrid

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from typing import Union, List


def activation_func(act: str = "tanh"):
    if act == "tanh":
        return nn.Tanh()
    elif act == "relu":
        return nn.ReLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "swish":
        return nn.SiLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation function {act} not supported")


class NeuralODE(nn.Module):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.odefunc = func

    def forward(self, y0, t):
        return odeint(
            self.odefunc,
            y0,
            t,
            method="rk4",
            options={"step_size": 0.01, "dtype": torch.float32},
        )


class NNODEF(nn.Module):
    def __init__(self, in_dim, hid_dim, time_invariant=False):
        super(NNODEF, self).__init__()
        self.time_invariant = time_invariant

        if time_invariant:
            self.lin1 = nn.Linear(in_dim, hid_dim)
        else:
            self.lin1 = nn.Linear(in_dim + 1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t, x):
        if not self.time_invariant:
            x = torch.cat((x, t.reshape(1, 1)), dim=-1)

        h = self.relu(self.lin1(x))
        h = self.relu(self.lin2(h))
        out = self.lin3(h)
        return out


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, act: str = "tanh"):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.act = activation_func(act)

        self.rnn = nn.GRU(self.input_dim + 1, self.hidden_dim, num_layers=2)
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )

    def forward(self, x, t):
        # Concatenate time to input
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.0
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.layers(h0[0])
        z0_mean = z0[:, : self.latent_dim]
        z0_log_var = z0[:, self.latent_dim :]
        return z0_mean, z0_log_var  ## both (n_timestamps, n_samples, latent_dim)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, act: str = "tanh"):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.act = activation_func(act)

        # 将 GRU 替换为 LSTM
        self.rnn = nn.LSTM(self.input_dim + 1, self.hidden_dim, num_layers=1)

        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.act,
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )

    def forward(self, x, t):
        # 计算时间间隔
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.0
        xt = torch.cat((x, t), dim=-1)  # 拼接时间信息

        _, (h_n, c_n) = self.rnn(xt.flip((0,)))  # LSTM 额外返回 c_n，但我们不使用它
        z0 = self.layers(h_n[0])  # 仅使用 h_n

        z0_mean = z0[:, : self.latent_dim]
        z0_log_var = z0[:, self.latent_dim :]
        return z0_mean, z0_log_var  # 输出形状 (n_timestamps, n_samples, latent_dim)


class CoordBoost(nn.Module):  ## 处理的对象是x_coord和z
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_out: int = 1,
        act: bool = False,
    ):
        super(CoordBoost, self).__init__()
        # self.fc_coord = nn.Linear(2, hidden_dim)  ## for coord matrix
        # self.fc_latent = nn.Linear(
        #     latent_dim, hidden_dim, bias=False
        # )  ## for image content
        # self.activation = nn.Tanh() if act else None

        self.fc_coord = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.Tanh()  # 或者尝试 Swish

        layers = []
        for _ in range(3):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, coord, z):  ## z 仅是content，不包含rotation和translatioon
        sq_len = coord.size(0)
        b = coord.size(1)
        n = coord.size(2)
        coord = coord.view(sq_len * b * n, -1)

        h_x = self.fc_coord(coord)
        h_x = h_x.view(sq_len, b, n, -1)

        h_z = 0
        if hasattr(self, "latent_linear"):
            z = z.view(sq_len * b, -1)
            h_z = self.fc_latent(z)
            h_z = h_z.view(sq_len, b, 1, -1)

        h = h_z + h_x
        h = h.view(sq_len * b * n, -1)

        y = self.layers(h)
        y = y.view(sq_len, b, -1)

        return y


class NeuralODEDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        coord,
        act: str = "tanh",
        **kwargs,
    ):
        super(NeuralODEDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.coord = coord
        self.act = activation_func(act)

        self.dx_prior = Tensor(kwargs.get("dx_prior", [1.0]))

        func = NNODEF(latent_dim + coord, hidden_dim, time_invariant=False)
        self.ode = NeuralODE(func)

        self.decode_net = (
            CoordBoost(latent_dim, hidden_dim)
            if coord > 0
            else nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, self.output_dim),
                nn.Sigmoid(),
            )
        )

    def forward(self, z0, t):
        zs = self.ode(z0, t)  ## (n_timestamps, batch_size, latent)

        if self.coord > 0:
            dx = self.calc_dx(zs, self.dx_prior.to(zs))
            coord = self.transform_coordinate(zs, dx)

            xs = self.decode_net(coord, zs[:, :, 3:])
        else:
            xs = self.decode_net(zs)

        return xs, zs

    def calc_dx(self, zs, dx_prior):
        z_dx = zs[:, :, 1:3]
        dx = z_dx * dx_prior
        return dx

    def _make_grid_stack(self, zs):
        sq_len = zs.size(0)
        b = zs.size(1)
        single_grid = imcoordgrid(self.input_dim)
        grid_stack = single_grid.view(1, 1, self.input_dim, 2).repeat(sq_len, b, 1, 1)
        return grid_stack

    def transform_coordinate(self, zs, dx):
        grid_stack = self._make_grid_stack(zs).to(zs)
        sq_len = grid_stack.size(0)
        b = grid_stack.size(1)

        theta = zs[:, :, 0]
        rotmat_r1 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        rotmat_r2 = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
        rotmat = torch.stack([rotmat_r1, rotmat_r2], dim=-1)

        grid_stack_reshaped = grid_stack.view(-1, self.input_dim, 2)
        rotmat_reshaped = rotmat.view(-1, 2, 2)

        coord_stack_reshape = torch.bmm(grid_stack_reshaped, rotmat_reshaped)
        coord_stack = coord_stack_reshape.view(sq_len, b, self.input_dim, 2)
        return coord_stack + dx.unsqueeze(2)  ## (sq_len, b, input_dim, 2)


class ODEVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 64,
        latent_dim: int = 2,
        coord: int = 3,
        act: str = "tanh",
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        super(ODEVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.coord = coord
        self.act = activation_func(act)
        self.theta_prior = torch.tensor(kwargs.get("theta_prior", 0.1)).to(device)

        self.encoder = LSTMEncoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim + coord,
            act="relu",
        )

        self.decoder = NeuralODEDecoder(
            self.input_dim,
            self.hidden_dim,
            self.latent_dim,
            coord,
            act="relu",
            **kwargs,
        )

        self.device = device
        self.to(device)

    def forward(self, x, t, beta, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)  ## (batch_size, latent_dim+3)

        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)

        if self.coord > 0:
            theta_std = torch.exp(z_log_var[:, 0] * 0.5)
            theta_logstd = 0.5 * z_log_var[:, 0]
            kl_div_rot = (
                -theta_logstd
                + torch.log(self.theta_prior)
                + (theta_std**2) / 2 / self.theta_prior**2
                - 0.5
            )

        z_mu_content = z_mean[:, 1:] if self.coord > 0 else z_mean
        z_std_content = (
            torch.exp(0.5 * z_log_var)[:, 1:]
            if self.coord > 0
            else torch.exp(0.5 * z_log_var)
        )
        z_logstd_content = 0.5 * z_log_var[:, 1:] if self.coord > 0 else 0.5 * z_log_var

        z_div_kl = (
            -z_logstd_content + 0.5 * z_std_content**2 + 0.5 * z_mu_content**2 - 0.5
        )

        kl_div = (
            torch.sum(z_div_kl, 1) + kl_div_rot
            if self.coord > 0
            else torch.sum(z_div_kl, 1)
        )
        kl_div = kl_div.mean()

        x_p, zs = self.decoder(z, t[:, 0].view(-1))

        log_p_x_g_z = -(
            0.5
            * torch.sum(
                (x.view(-1, self.input_dim) - x_p.view(-1, self.input_dim)) ** 2, 1
            )
        ).mean()

        elbo = log_p_x_g_z - beta * kl_div

        return x_p, elbo, log_p_x_g_z, kl_div, zs

    def _loss_func(self):
        pass

    def manifold2d(self, **kwargs: Union[int, List, str, bool]):
        d = kwargs.get("d", 9)
        cmap = kwargs.get("cmap", "gnuplot")
        in_dim = (int(math.sqrt(self.in_dim)), int(math.sqrt(self.in_dim)))
        figure = np.zeros((in_dim[0] * d, in_dim[1] * d))

        grid_x = norm.ppf(np.linspace(0.95, 0.05, d))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, d))

        for i, xi in enumerate(grid_x):
            for j, yj in enumerate(grid_y):
                z_sample = np.array([xi, yj])

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t[:, 0].view(-1))
        return x_p


if __name__ == "__main__":
    vae = ODEVAE()
    x = torch.randn(100, 10, 2)
    t = torch.randn(100, 10, 1)
    t, _ = torch.sort(t, dim=0)
    out = vae(x, t)
