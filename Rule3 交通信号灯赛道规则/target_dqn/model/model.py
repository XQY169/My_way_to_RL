#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from target_dqn.config import Config


class Model(nn.Module):
    """
    Basic neural network implemented by PyTorch
    Can choose whether to output softmax in the last layer based on the configuration Config.SOFTMAX

    由pytorch实现的基础神经网络
    可以根据配置Config.SOFTMAX 选择是否在最后一层输出softmax
    """

    def __init__(self, device=None):
        super().__init__()
        action_shape = [Config.DIM_OF_ACTION_PHASE, Config.DIM_OF_ACTION_DURATION]
        self.device = device

        modules = []
        all_dims = [Config.DIM_OF_OBSERVATION] + [16, 32] + [16]

        for i in range(len(all_dims) - 1):
            modules.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                modules.append(nn.LayerNorm(all_dims[i + 1]))
                modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules).to(self.device)

        num_heads = len(action_shape)
        self.heads = nn.ModuleList([nn.Linear(16, np.prod(action_shape[i])) for i in range(num_heads)]).to(self.device)

    def forward(self, s, state=None, info=None):
        if info is None:
            info = {}

        if not isinstance(s, torch.Tensor):
            s = torch.tensor(
                np.array(s, dtype=np.float32),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            s = s.to(torch.float32)

        batch = s.shape[0]
        s = self.model(s)

        if Config.SOFTMAX:
            return [F.softmax(head(s), dim=-1) for head in self.heads], state
        else:
            return [head(s) for head in self.heads], state
