#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwudrl.interface.array_spec import ArraySpec
from kaiwudrl.common.algorithms.distribution import CategoricalDist
from kaiwudrl.interface.action import Action, ActionSpec
from kaiwudrl.common.config.config_control import CONFIG

try:
    config_module = __import__(f"{CONFIG.algo}.config", fromlist=["Config"])
    Config = getattr(config_module, "Config")
except ModuleNotFoundError:
    raise NotImplementedError(f"The algorithm '{CONFIG.algo}' is not yet implemented")
except AttributeError:
    raise ImportError(f"The module '{CONFIG.algo}.config' does not have a 'Config' class")


class TrafficAction(Action):
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {"a": self.a}

    @staticmethod
    def action_space():
        action_phase_space = CONFIG.DIM_OF_ACTION_PHASE
        action_duration_space = CONFIG.DIM_OF_ACTION_DURATION
        return {
            "a": ActionSpec(
                ArraySpec(((action_phase_space,), (action_duration_space,)), np.int32),
                pdclass=CategoricalDist,
            )
        }

    def __str__(self):
        return str(self.a)
