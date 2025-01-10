#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from kaiwu_agent.utils.common_func import attached
from target_dqn.model.model import Model
from target_dqn.feature.definition import ActData
from target_dqn.feature.definition import *
import numpy as np
from copy import deepcopy
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from target_dqn.config import Config
import torch.nn.functional as F


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.model = Model(device=device)

        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=Config.LR)
        self._eps = Config.START_EPSILON_GREEDY
        self.end_eps = Config.END_EPSILON_GREEDY
        self.eps_decay = Config.EPSILON_DECAY
        self._lambda = Config.LAMBDA
        self.num_head = Config.NUMB_HEAD
        self.head_dim = [
            Config.DIM_OF_ACTION_PHASE,
            Config.DIM_OF_ACTION_DURATION,
        ]
        self.device = device

        self._gamma = Config.GAMMA
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.epsilon = Config.EPSILON
        self.logger = logger
        self.monitor = monitor
        self.last_report_monitor_time = 0
        self.last_action = -1

        super().__init__(agent_type, device, logger, monitor)

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        feature = [obs_data.feature for obs_data in list_obs_data]
        junction_id = 0
        list_junction = [
            junction_id,
        ] * len(list_obs_data)
        obs_feature = feature[0]
        if self.last_action == -1:
            list_phase = [0]
            self.last_action = list_phase[0]

        else:
            if obs_feature[self.last_action+16] <= 0.5 or obs_feature[self.last_action] <= 0.5 or obs_feature[24] >= 40:
                list_phase = [obs_feature[16:20].index(max(obs_feature[16:20]))]
                self.last_action = list_phase[0]
            else:
                list_phase = [self.last_action] 

        # print('agent_count:',obs_feature[24])
        # phase = Config.LINES3[self.last_action]
        # datas_position = []
        # datas_speed = []
        # for i in phase:
        #     datas_position += obs_feature[25+i*20:45+i*20]
        #     datas_speed += obs_feature[305+i*20:325+i*20]
        # self.logger.info(
        #         f"observation[0:8]:{obs_feature[0:8]},observation[16:24]: {obs_feature[16:24]},action: {list_phase[0]},'sum_vehicle:{obs_feature[24]}"
        #     )
        # self.logger.info(
        #         f"total_speed:{obs_feature[8:16]},"
        #     )
        # for i in phase:
        #     self.logger.info(
        #         f"line:{i},position:{datas_position[phase.index(i)*20:phase.index(i)*20+20]},speed:{datas_speed[phase.index(i)*20:phase.index(i)*20+20]}"
        #     )
        return [
            ActData(
                junction_id=list_junction[i],
                phase_index=list_phase[i],
                duration=1,
            )
            for i in range(len(list_obs_data))
        ]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        t_data = list_sample_data

        obs = torch.tensor(np.array([frame.obs for frame in t_data]), dtype=torch.float32).to(self.device)
        action = (
            torch.LongTensor(np.array([frame.act if not np.any(np.isinf(frame.act)) else 0 for frame in t_data]))
            .long()
            .to(self.model.device)
        )
        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.model.device)
        _obs = torch.tensor(np.array([frame._obs for frame in t_data]), dtype=torch.float32).to(self.device)
        not_done = torch.tensor(
            np.array([frame.done for frame in t_data]),
            dtype=torch.float32,
            device=self.device,
        )

        # Main implementation of the multi-head output Target_DQN algorithm
        # 多头输出target_dqn算法的主要实现
        self.target_model.eval()

        with torch.no_grad():
            # Calculate the target Q-values for each head
            # 计算各个头的目标q值
            q_targets = []
            for head_idx in range(self.num_head):
                q_targets_head = (
                    rew[:, head_idx].unsqueeze(1)
                    + self._gamma * (self.target_model(_obs)[0][head_idx]).max(1)[0].unsqueeze(1) * not_done[:, None]
                )
                q_targets.append(q_targets_head)
            q_targets = torch.cat(q_targets, dim=1)

        # Calculate the Q-values for each head
        # 计算各个头的q值
        self.model.train()
        q_values = []
        for head_idx in range(self.num_head):
            q_values_head = self.model(obs)[0][head_idx].gather(1, action[:, head_idx + 1].unsqueeze(1))
            q_values.append(q_values_head)
        q_values = torch.cat(q_values, dim=1)

        self.optim.zero_grad()
        loss = F.mse_loss(q_targets.float(), q_values.float())
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        self.optim.step()
        self.train_step += 1

        if self.train_step % Config.TARGET_UPDATE_FREQ == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        target_q_value = q_targets.mean().detach().item()
        q_value = q_values.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "target_q_value": target_q_value,
                "q_value": q_value,
                "model_grad_norm": model_grad_norm,
            }
            self.monitor.put_data({os.getpid(): monitor_data})
            self.logger.info(
                f"value_loss: {value_loss}, target_q_value: {target_q_value},\
                                q_value: {q_value},\
                                model_grad_norm: {model_grad_norm}"
            )
            self.last_report_monitor_time = now

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict = self.model.state_dict()
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))

        self.logger.info(f"load model {model_file_path} successfully")
