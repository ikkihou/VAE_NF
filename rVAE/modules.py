#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   modules.py
@Time    :   2025/01/06 20:42:07
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidLinear(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(ResidLinear, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class InferenceNetwork(nn.Module):  ## q(z, \theta, \Delta_X | X)
    def __init__(
        self,
        n: int,
        latent_dim: int,
        num_layers: int,
        hidden_dim: int,
        act_func: str = "tanh",
        resid: bool = False,
    ):
        """Inference network for approximating p(\theta) p(z) p(\Delta_X)

        Args:
            n (int): number of pixels in the image
            latent_dim (int): dimension of the input vector
            num_layers (int): number of layers
            hidden_dim (list[int]): list of hidden dimensions
            act_func (str, optional): activation function
            resid (bool, optional): whether to use residual connections
        """
        super(InferenceNetwork, self).__init__()

        self.latent_dim = latent_dim  ## z_dim + 1(if rotate) + 2(if translate)
        self.n = n  ## h * w

        activation = nn.Tanh if act_func == "tanh" else nn.ReLU

        layers = [
            nn.Linear(n, hidden_dim),
            activation(),
        ]
        for _ in range(1, num_layers):
            if resid:
                layers.append(
                    ResidLinear(hidden_dim, hidden_dim, activation=activation)
                )
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())

        layers.append(nn.Linear(hidden_dim, 2 * latent_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x is (batch,num_coords)
        z = self.layers(x)  ## NOTE: (batch, 2*latent_dim)

        ld = self.latent_dim
        z_mu = z[:, :ld]
        z_logstd = z[:, ld:]

        return z_mu, z_logstd


class SpatialGenerator(nn.Module):  ## p(X | z, \theta, \Delta_X)
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_out: int = 1,
        act_func: str = "tanh",
        softplus: bool = False,
        resid: bool = False,
        expand_coords: bool = False,
        bilinear=False,
    ):
        """Spatial generator for generating the spatial coordinates of the image

        Args:
            latent_dim (int): dimension of the latent vector
            hidden_dim (int): hidden dimension
            num_layers (int): number of layers
            n_out (int, optional): number of output dimensions
            act_func (str, optional): activation function
            act_func (str, optional): _description_. Defaults to "tanh".
            softplus (bool, optional): _description_. Defaults to False.
            resid (bool, optional): _description_. Defaults to False.
            expand_coords (bool, optional): _description_. Defaults to False.
            bilinear (bool, optional): _description_. Defaults to False.
        """
        super(SpatialGenerator, self).__init__()

        activation = nn.Tanh if act_func == "tanh" else nn.ReLU

        self.softplus = softplus
        self.expand_coords = expand_coords

        in_dim = 2
        if expand_coords:
            in_dim = 5  # include squares of coordinates as inputs

        self.coord_linear = nn.Linear(in_dim, hidden_dim)
        self.latent_dim = latent_dim
        if latent_dim > 0:
            self.latent_linear = nn.Linear(latent_dim, hidden_dim, bias=False)

        if (
            latent_dim > 0 and bilinear
        ):  # include bilinear layer on latent and coordinates
            self.bilinear = nn.Bilinear(in_dim, latent_dim, hidden_dim, bias=False)

        layers = [activation()]
        for _ in range(1, num_layers):
            if resid:
                layers.append(
                    ResidLinear(hidden_dim, hidden_dim, activation=activation)
                )
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation())
        layers.append(nn.Linear(hidden_dim, n_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, z):  ## NOTE: x包含图像中每个像素的rotation和translate的信息
        # x is (batch, num_coords, 2)
        # z is (batch, latent_dim)

        if len(x.size()) < 3:
            x = x.unsqueeze(0)
        b = x.size(0)
        n = x.size(1)
        x = x.view(b * n, -1)
        if self.expand_coords:
            x2 = x**2
            xx = x[:, 0] * x[:, 1]
            x = torch.cat([x, x2, xx.unsqueeze(1)], 1)

        h_x = self.coord_linear(x)
        h_x = h_x.view(b, n, -1)  ## NOTE: (batch, num_coords, hidden_dim)

        h_z = 0
        if hasattr(self, "latent_linear"):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            h_z = self.latent_linear(z)
            h_z = h_z.unsqueeze(1)  ## NOTE: (batch, 1, hidden_dim)

        h_bi = 0
        if hasattr(self, "bilinear"):
            if len(z.size()) < 2:
                z = z.unsqueeze(0)
            z = z.unsqueeze(1)  # broadcast over coordinates
            x = x.view(b, n, -1)
            z = z.expand(b, x.size(1), z.size(2)).contiguous()
            h_bi = self.bilinear(x, z)

        h = h_x + h_z + h_bi  # (batch, num_coords, hidden_dim)
        h = h.view(b * n, -1)

        y = self.layers(h)  # (batch*num_coords, nout)
        y = y.view(b, n, -1)

        if self.softplus:  # only apply softplus to first output
            y = torch.cat([F.softplus(y[:, :, :1]), y[:, :, 1:]], 2)

        return y