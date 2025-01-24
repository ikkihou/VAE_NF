#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2025/01/07 15:00:22
@Author  :   Paul_Bao
@Version :   1.0
@Contact :   paulbao@mail.ecust.edu.cn
"""

# here put the import lib
import numpy as np
from typing import Dict, Tuple, Union
import sys
import logging
import os


class imlocal:
    """
    STEM image local crystallography class
    highly ported from atomai https://github.com/pycroscopy/atomai/tree/master
    """

    def __init__(
        self,
        network_output: np.ndarray,
        coord_class_dict_all: Dict[int, np.ndarray],
        window_size: int = None,
        coord_class: int = 0,
    ) -> None:
        self.network_output = network_output
        self.nb_classes = network_output.shape[-1]
        self.coord_all = coord_class_dict_all
        self.coord_class = float(coord_class)
        self.r = window_size
        (self.imgstack, self.imgstack_com, self.imgstack_frames) = (
            self.extract_subimages_()
        )
        self.d0, self.d1, self.d2, self.d3 = self.imgstack.shape

    def get_imgstack(
        self,
        imgdata: np.ndarray,
        coord: np.ndarray,
        r: int,
    ) -> Tuple[np.ndarray]:

        img_cr_all = []
        com = []
        for c in coord:
            cx = int(np.around(c[0]))
            cy = int(np.around(c[1]))
            if r % 2 != 0:
                img_cr = np.copy(
                    imgdata[
                        cx - r // 2 : cx + r // 2 + 1, cy - r // 2 : cy + r // 2 + 1
                    ]
                )
            else:
                img_cr = np.copy(
                    imgdata[cx - r // 2 : cx + r // 2, cy - r // 2 : cy + r // 2]
                )
            if img_cr.shape[0:2] == (int(r), int(r)) and not np.isnan(img_cr).any():
                img_cr_all.append(img_cr[None, ...])
                com.append(c[None, ...])
        if len(img_cr_all) == 0:
            return None, None
        img_cr_all = np.concatenate(img_cr_all, axis=0)
        com = np.concatenate(com, axis=0)
        return img_cr_all, com

    def extract_subimages(
        self,
        imgdata: np.ndarray,
        coordinates: Union[Dict[int, np.ndarray], np.ndarray],
        window_size: int,
        coord_class: int = 0,
    ) -> Tuple[np.ndarray]:

        if isinstance(coordinates, np.ndarray):
            coordinates = np.concatenate(
                (coordinates, np.zeros((coordinates.shape[0], 1))), axis=-1
            )
            coordinates = {0: coordinates}
        if np.ndim(imgdata) == 2:
            imgdata = imgdata[None, ..., None]
        subimages_all, com_all, frames_all = [], [], []
        for i, (img, coord) in enumerate(zip(imgdata, coordinates.values())):
            coord_i = coord[np.where(coord[:, 2] == coord_class)][:, :2]
            stack_i, com_i = self.get_imgstack(img, coord_i, window_size)
            if stack_i is None:
                continue
            subimages_all.append(stack_i)
            com_all.append(com_i)
            frames_all.append(np.ones(len(com_i), int) * i)
        if len(subimages_all) > 0:
            subimages_all = np.concatenate(subimages_all, axis=0)
            com_all = np.concatenate(com_all, axis=0)
            frames_all = np.concatenate(frames_all, axis=0)
        return subimages_all, com_all, frames_all

    def extract_subimages_(self) -> Tuple[np.ndarray]:

        imgstack, imgstack_com, imgstack_frames = self.extract_subimages(
            self.network_output, self.coord_all, self.r, self.coord_class
        )
        return imgstack, imgstack_com, imgstack_frames


import platform
import torch


def is_mps():
    return platform.system() == "Darwin" and torch.backends.mps.is_available()


def is_cuda():
    return torch.cuda.is_available()


def seed_everything(seed):
    torch.manual_seed(seed)
    if is_mps():
        torch.mps.manual_seed(seed)
    elif is_cuda():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(0)
    else:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)


def get_device():
    if is_mps():
        return torch.device("mps")
    elif is_cuda():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_logger(
    logger_name: str, root: str, phase: str, level=logging.INFO, screen=False
):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    log_file = os.path.join(root, "{}.log".format(phase))
    if not os.path.exists(root):
        os.makedirs(root)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l