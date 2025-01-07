#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   base.py
@Time    :   2025/01/06 20:39:00
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
from .types_ import *
from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def reparameterize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
