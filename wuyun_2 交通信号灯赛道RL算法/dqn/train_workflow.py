#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
from dqn.feature.definition import *
from kaiwu_agent.utils.common_func import Frame, attached
from conf.usr_conf import read_usr_conf, check_usr_conf
from dqn.feature.traffic_handler import TrafficHandler


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    g_data_truncat = 8
    last_save_model_time = 0

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("conf/usr_conf.yaml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check conf/usr_conf.yaml")
        return
    # check_usr_conf is a tool to check whether the environment configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查环境配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check conf/usr_conf.yaml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0

        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger):
            data_length += len(g_data)
            total_rew = []
            for data in g_data:
                total_rew.append(data.rew[0])

            total_rew = sum(total_rew)
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        # Reporting training progress
        # 上报训练进度
        if now - last_report_monitor_time > 60:
            monitor_data["reward"] = avg_step_reward
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
                last_report_monitor_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger):
    for episode in range(n_episode):
        traffic_handler = TrafficHandler(logger)
        collector = list()

        # At the start of each environment, support loading the latest model file
        # 每次对局开始时, 支持加载最新model文件
        agent.load_model(id="latest")

        # Reset the environment and get the initial state
        # 重置环境, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            logger.info(f"obs is None, so continue")
            continue

        # Interact with the environment, execute actions, get the next state
        # 与环境交互, 执行动作, 获取下一步的状态
        frame_no, obs, score, terminated, truncated, env_info = traffic_handler.update(env, obs)
        # Disaster recovery
        # 容灾
        if obs is None:
            logger.info(f"obs is None, so continue")
            continue

        # Feature processing
        # 特征处理
        obs_data = observation_process(obs, traffic_handler, env_info)

        done = False
        count = 0
        while not done:
            # Agent makes a prediction to get the next frame's action
            # Agent 进行推理, 获取下一帧的预测动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpack ActData into actions
            # ActData 解包成动作
            act = action_process(act_data)

            # If the frame does not require prediction, advance the environment until a frame that requires prediction.
            # 如果是不需要预测的帧，则推进env直到需要预测的帧
            (
                frame_no,
                _obs,
                score,
                terminated,
                truncated,
                _env_info,
            ) = traffic_handler.update(env, obs, act)
            count += 1
            logger.info(f"current step is {count}")

            # Disaster recovery
            # 容灾
            if _obs is None:
                logger.info(f"_obs is None, so break")
                break

            # Feature processing
            # 特征处理
            _obs_data = observation_process(_obs, traffic_handler, _env_info)

            if truncated:
                logger.info(f"truncated is True, frame_no is {frame_no}, so this episode timeout")
            elif terminated:
                logger.info(f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

            # Calculate reward Rewards include phase_reward and duration_reward
            # 奖励有phase_reward和duration_reward
            reward = reward_shaping(_obs, traffic_handler, _env_info)

            # Determine if the environment is over and update the number of wins
            # 判断环境结束, 并更新胜利次数
            done = terminated or truncated

            # Construct environment frames to prepare for sample construction
            # 构造环境帧，为构造样本做准备
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                sub_action_mask=obs_data.sub_action_mask,
                _sub_action_mask=_obs_data.sub_action_mask,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            # Some frames do not require prediction and thus do not need to send samples
            # 存在有些帧不需要预测, 同时就不需要进行样本发送
            if len(collector) % g_data_truncat == 0 and len(collector) > 1:
                collector = sample_process(collector)
                yield collector

            if len(collector) % 20 == 0:
                agent.load_model(id="latest")

            if done:
                if len(collector) > 1:
                    collector = sample_process(collector)
                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            env_info = _env_info
