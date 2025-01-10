#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:
    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for Target-DQN is 1128,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中target_dqn的维度是1128
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 58

    # Size of observation
    # observation的维度，
    DIM_OF_OBSERVATION = 25
    DIM_OF_ACTION_PHASE = 4
    DIM_OF_ACTION_DURATION = 20
    DIM_SUB_ACTION_MASK = 24

    SOFTMAX = False

    # Algorithm Config
    # 算法的配置
    GAMMA = 0.9
    EPSILON = 0.1
    LR = 1e-3

    START_EPSILON_GREEDY = 1.0
    END_EPSILON_GREEDY = 0.3
    EPSILON_DECAY = 0.995
    LAMBDA = 0.95
    NUMB_HEAD = 2
    TARGET_UPDATE_FREQ = 500

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
    LINES3 = [[0,1,2,8,9,10],[3,11],[5,6,12],[7,13]]
    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 45000
    LEGAL_ACTION_SHAPE = 2
