#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from diy.config import Config

ObsData = create_cls("ObsData", feature=None, sub_action_mask=None)

ActData = create_cls("ActData", junction_id=None, phase_index=None, duration=None)

SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
    sub_action_mask=None,
    _sub_action_mask=None,
)


@attached
def observation_process(raw_obs, env_info=None):
    pass


@attached
def action_process(act_data):
    pass


@attached
def sample_process(list_game_data):
    pass


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    pass


@attached
def NumpyData2SampleData(s_data):
    pass
