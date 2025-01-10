#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from target_dqn.feature.definition import *
from target_dqn.feature.traffic_info import TrafficInfo


class TrafficHandler:
    """
    Reinforcement learning traffic signal control tools
    """

    """
    强化学习信号控制工具类
    """

    def __init__(self, logger) -> None:
        self.last_action = [0, 0, 0]
        # The initial decision frame has a 3-second delay
        # 初始决策帧有3秒的延迟
        self.dec_time = 3000
        # Yellow light interval time is 5 seconds
        # 黄灯间隔时间为5秒
        self.yellow_duration = 5000
        self.old_phase = 0
        self.phase = 0
        self.update_dec = False
        self.traffic_info = TrafficInfo(logger)
        self.logger = logger
        self.current_step = 0

    def __need_to_predict(self, raw_obs):
        # Determine whether this frame needs prediction
        # 确定该帧是否需要预测
        frame_state = raw_obs[1].framestate
        frame_time = frame_state.frame_time
        action = self.last_action
        phase = action[1]
        duration = action[2]

        # If the current time is greater than or equal to the decision time,
        # then the prediction is performed and the decision time is updated
        # 若当前时刻大于等于应决策时刻, 则执行预测并更新决策时间
        if self.update_dec:
            # Situations that require switching to a yellow light
            # 需要切黄灯的情况
            if self.phase != phase:
                self.dec_time += duration + self.yellow_duration
            # Situations where a yellow light is not needed
            # 不需要加黄灯的情况
            else:
                self.dec_time += duration
            self.update_dec = False
            # Used to correctly call env.step to transfer actions after
            # the predicted frame has given actions and updated dec_time
            # 用于在预测帧已给出动作并更新dec_time后, 正确调用env.step传输动作
            return True

        # 用户自定义部分, 可每帧对交通信息进行记录或更新
        self.traffic_info.update_traffic_info(raw_obs)

        # Used to determine whether the prediction frame has been
        # reached to jump out of the while not need_to_predict loop
        # 用于判断是否到达预测帧以跳出while not need_to_predict循环
        if frame_time >= self.dec_time:
            self.update_dec = True
            return True
        else:
            return False

    def update(self, env, obs, last_action=None):
        if last_action is None:
            last_action = [0, 0, 0]

        self.last_action = last_action
        # Determine whether this frame needs prediction.
        # 确定该帧是否需要预测
        need_to_predict = self.__need_to_predict(obs)
        if need_to_predict:
            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, obs, score, terminated, truncated, env_info = env.step(
                [last_action[0], last_action[1], last_action[2]]
            )
            self.current_step += 1

        # After the frame is predicted, transmit [None, None, None] to advance to the next prediction frame,
        # with the duration being the phase duration predicted for the current frame.
        # Since no further prediction is needed during this phase duration,
        # a null action is returned to advance the environment.
        # 该帧预测后，空传【None, None, None】至下一次预测帧，持续时长为该帧预测的相位持续时间，因为在这段相位持续时间里不需要再次预测，通过返回空来推进环境
        need_to_predict = False
        while not need_to_predict:
            frame_no, obs, score, terminated, truncated, env_info = env.step([None, None, None])
            self.current_step += 1
            if terminated or truncated:
                return frame_no, obs, score, terminated, truncated, env_info

            # Disaster recovery
            # 容灾
            if obs is None:
                return None, None, None, None, True, None
            need_to_predict = self.__need_to_predict(obs)

        # Initialize old_phase with last_action[1], and the value of old_phase in the first frame is 0.
        # 初始old_phase = last_action[1], old_phase在第1帧的值为0
        self.old_phase = self.phase
        self.phase = self.last_action[1]

        return frame_no, obs, score, terminated, truncated, env_info
