#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   coords.py
@Time    :   2025/01/21 15:10:51
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import math
import torch
import numpy as np
from typing import Tuple, Union


def grid2xy(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Maps (M, N) grid to (M*N, 2) xy coordinates
    """
    X = torch.cat((X1[None], X2[None]), 0)
    d0, d1 = X.shape[0], X.shape[1] * X.shape[2]
    X = X.reshape(d0, d1).T
    return X


def imcoordgrid(in_dim: Union[int, Tuple]) -> torch.Tensor:
    if isinstance(in_dim, Tuple):
        xx = torch.linspace(-1, 1, in_dim[0])
        yy = torch.linspace(1, -1, in_dim[1])
    elif isinstance(in_dim, int):
        sqrt_dim = int(math.sqrt(in_dim))
        xx = torch.linspace(-1, 1, sqrt_dim)
        yy = torch.linspace(1, -1, sqrt_dim)
    else:
        return None  # Handle other cases explicitly if needed

    # Update torch.meshgrid with explicit indexing
    x0, x1 = torch.meshgrid(xx, yy, indexing="ij")
    return grid2xy(x0, x1)




def transform_coordinates(
    coord: torch.Tensor,
    phi: float,
    coord_dx: torch.Tensor = 0,
) -> torch.Tensor:

    rotmat_r1 = torch.stack([torch.cos(phi), torch.sin(phi)], 1)
    rotmat_r2 = torch.stack([-torch.sin(phi), torch.cos(phi)], 1)
    rotmat = torch.stack([rotmat_r1, rotmat_r2], axis=1)
    coord = torch.bmm(coord, rotmat)

    return coord + coord_dx
