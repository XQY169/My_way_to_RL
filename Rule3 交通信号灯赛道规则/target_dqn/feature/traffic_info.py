#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
import copy
from target_dqn.feature.traffic_utils import *


class TrafficInfo:
    """
    Update traffic information and perform feature processing
    """

    """
    更新交通信息并进行特征处理
    """

    def __init__(self, logger):
        self.logger = logger
        # Store road structure and other relatively fixed dictionary-type variables
        # 存储道路结构等相对固定的字典型变量
        self.junction_dict = {}
        self.edge_dict = {}
        self.lane_dict = {}
        self.l_id_to_index = {}
        self.vehicle_configs_dict = {}

        # Store dictionary-type variables for dynamic traffic information in traffic scenarios,
        # and update data after each communication
        # 存储交通场景中动态交通信息的字典型变量, 在每次通信后进行数据更新
        self.vehicle_prev_junction = {}
        self.vehicle_prev_position = {}
        self.vehicle_distance_store = {}
        self.last_waiting_moment = {}
        self.waiting_time_store = {}
        self.enter_lane_time = {}

        self.last_record_waiting_vehicle = {}
        self.current_record_waiting_vehicle = {}

        # Stores variables that can be used to calculate rewards
        # 存储可用于计算奖励的变量
        self.lane_volume = {}

        # User-defined variable
        # 用户自定义变量
        self.old_waiting_time = 0
        self.old_obs = [0.0] * 25
        self.change_flag = 0
        self.count = -1
    def init_road_info(self, start_info):
        """
        Obtain fixed variables such as road structure
        """
        """
        获取道路结构等信息固定的变量
        """
        junctions, signals, edges = (
            start_info.junctions,
            start_info.signals,
            start_info.edges,
        )
        lane_configs, vehicle_configs = (
            start_info.lane_configs,
            start_info.vehicle_configs,
        )
        # Store road structure information in various variables
        # 将道路结构信息存储到各个变量
        for junction in junctions:
            self.junction_dict[junction.j_id] = junction
            self.l_id_to_index[junction.j_id] = {}

            index = 0
            for approaching_edges in junction.enter_lanes_on_directions:
                for lane in approaching_edges.lanes:
                    self.l_id_to_index[junction.j_id][lane] = index
                    index += 1

            for edge in edges:
                self.edge_dict[edge.e_id] = edge
            for lane in lane_configs:
                self.lane_dict[lane.l_id] = lane
            for vehicle_config in vehicle_configs:
                self.vehicle_configs_dict[vehicle_config.v_config_id] = vehicle_config
            for lane in lane_configs:
                self.lane_volume[lane.l_id] = []

    def update_traffic_info(self, raw_obs):
        """
        Update vehicle history information and calculate various dynamic traffic variables
        """
        """
        更新车辆历史信息, 计算各项动态交通变量
        """
        frame_state = raw_obs[1].framestate
        game_info = raw_obs[1].gameinfo
        frame_no = frame_state.frame_no
        frame_time, vehicles = frame_state.frame_time, frame_state.vehicles

        if frame_no <= 1:
            # Initial frame loads road structure information
            # 初始帧载入道路结构信息
            self.init_road_info(game_info)

        for vehicle in vehicles:
            # If the vehicle appears for the first time, initialize the vehicle's historical intersection information
            # 如果车辆第一次出现，则初始化车辆的历史交叉口信息
            if vehicle.v_id not in self.vehicle_prev_junction:
                self.vehicle_prev_junction[vehicle.v_id] = vehicle.junction
            # For vehicles that appear for the first time, if they are on the lane, record their appearance time
            # 对于首次出现的车辆, 若在车道上则记录其出现时间
            if (
                self.vehicle_prev_junction[vehicle.v_id] == -1
                and on_enter_lane(vehicle)
                and vehicle.v_id not in self.enter_lane_time
            ):
                self.enter_lane_time[vehicle.v_id] = frame_time
            # When a vehicle enters another entrance lane from the intersection, recalculate its appearance time
            # 当车辆从交叉口驶入另一进口车道时, 重新统计其出现时间
            elif self.vehicle_prev_junction[vehicle.v_id] != vehicle.junction:
                if self.vehicle_prev_junction[vehicle.v_id] != -1 and on_enter_lane(vehicle):
                    self.enter_lane_time[vehicle.v_id] = frame_time

            self.cal_waiting_time(frame_time, vehicle)
            self.cal_travel_distance(vehicle)
            self.cal_v_num_in_lane(vehicle)

    def cal_waiting_time(self, frame_time, vehicle):
        """
        Calculate the waiting time of the vehicle. When the vehicle is on the enter lane,
        count the accumulated time when its speed is <= 0.1m/s as its waiting time when driving at the intersection
        """
        """
        计算车辆等待时间, 当车辆处于进口车道上时, 统计其车速<=0.1m/s的累计时长作为其在该交叉口行驶时的等待时间
        """
        waiting_time = 0
        # Determine whether the vehicle in the lane approaching the intersection is in a waiting state,
        # and calculate the waiting time
        # 对处于车道驶向交叉口的车辆判断是否处于等待状态, 计算等待时间
        if on_enter_lane(vehicle):
            # Determine whether the vehicle is in a waiting state.
            # The determination condition is that the vehicle speed is <= 0.1m/s
            # 判断车辆是否处于等待状态, 判定条件为车辆速度<=0.1m/s
            if vehicle.speed <= 0.1 * 1000:
                if vehicle.v_id not in self.last_waiting_moment:
                    # Record the starting moment of each time the vehicle enters the waiting state
                    # 记录车辆在每次进入等待状态的起始时刻
                    self.last_waiting_moment[vehicle.v_id] = frame_time
                    # When the vehicle is in the waiting state for the first time,
                    # initialize its accumulated waiting time
                    # 车辆首次处于等待状态则初始化车辆累计等待时间
                    if vehicle.v_id not in self.waiting_time_store:
                        self.waiting_time_store[vehicle.v_id] = 0
                else:
                    # When a vehicle enters the waiting state on a lane,
                    # waiting_time records the duration of the current waiting state
                    # 车辆在一条道路上进入等待状态, waiting_time记录本次等待状态已持续的时间
                    waiting_time = frame_time - self.last_waiting_moment[vehicle.v_id]
                    self.waiting_time_store[vehicle.v_id] += waiting_time
                    self.last_waiting_moment[vehicle.v_id] = frame_time
            else:
                if vehicle.v_id in self.last_waiting_moment:
                    del self.last_waiting_moment[vehicle.v_id]
        else:
            # Prevent repeated del when the vehicle is generated for the first time or at an intersection,
            # v_id is not stored in self.waiting_time_store
            # 防止车辆首次生成或位于交叉口时反复del, v_id未储存在self.waiting_time_store内
            if vehicle.v_id in self.waiting_time_store:
                del self.waiting_time_store[vehicle.v_id]
            if vehicle.v_id in self.last_waiting_moment:
                del self.last_waiting_moment[vehicle.v_id]

    def cal_travel_distance(self, vehicle):
        """
        Calculate the travel distance. When the vehicle is on the enter lane,
        count the total distance it travels at the intersection
        """
        """
        计算旅行路程, 当车辆处于进口车道上时, 统计其在该交叉口行驶时的总路程
        """
        # When the vehicle is on the lane, calculate the cumulative distance
        # 当车辆处于车道上时, 计算累计路程
        if on_enter_lane(vehicle):
            # When the vehicle enters the lane from inside the intersection for the second or subsequent time,
            # clear the cumulative distance and prepare to calculate the distance of this entry into the inlane
            # 车辆非首次从交叉口内部驶入车道时, 清空累计路程, 准备计算此次进入进口车道的路程
            if self.vehicle_prev_junction[vehicle.v_id] != -1 and vehicle.v_id in self.vehicle_distance_store:
                del self.vehicle_distance_store[vehicle.v_id]
            if vehicle.v_id not in self.vehicle_distance_store:
                self.vehicle_distance_store[vehicle.v_id] = 0
                self.vehicle_prev_position[vehicle.v_id] = [
                    vehicle.position_in_lane.x,
                    vehicle.position_in_lane.y,
                ]
            else:
                if vehicle.v_id in self.vehicle_distance_store and vehicle.v_id in self.vehicle_prev_position:
                    try:
                        # Calculate Euclidean distance
                        # 计算欧氏距离
                        self.vehicle_distance_store[vehicle.v_id] += math.sqrt(
                            math.pow(
                                vehicle.position_in_lane.x - self.vehicle_prev_position[vehicle.v_id][0],
                                2,
                            )
                            + math.pow(
                                vehicle.position_in_lane.y - self.vehicle_prev_position[vehicle.v_id][1],
                                2,
                            )
                        )
                    except Exception:
                        raise ValueError
            # Update the vehicle's historical position after each distance calculation
            # 每次计算距离后更新车辆历史位置
            self.vehicle_prev_position[vehicle.v_id] = [
                vehicle.position_in_lane.x,
                vehicle.position_in_lane.y,
            ]
        else:
            # When the vehicle enters the intersection,
            # delete the historical location information to avoid calculating the driving distance
            # based on the last departure position when entering the lane next time
            # 当车辆驶入交叉口, 删除历史位置信息, 避免下次进入车道时按上一次离开路口位置计算行驶距离
            if vehicle.v_id in self.vehicle_prev_position:
                del self.vehicle_prev_position[vehicle.v_id]

    def cal_v_num_in_lane(self, vehicle):
        """
        Calculate the number of vehicles on the lane.
        When a vehicle is in the import lane, the number of vehicles on the enter lane increases
        """
        """
            计算车道上的车辆数, 当车辆处于进口车道上时, 则该进口车道上车辆数增加
            """
        # Update the number of vehicles on each lane
        # 更新每条车道上的车辆数量
        if on_enter_lane(vehicle):
            if vehicle.v_id not in self.lane_volume[vehicle.lane]:
                self.lane_volume[vehicle.lane].append(vehicle.v_id)

        # Update the vehicle's historical intersection information
        # 更新车辆的历史所在交叉口信息
        self.vehicle_prev_junction[vehicle.v_id] = vehicle.junction

    def record_last_waiting_vehicle(self):
        self.last_record_waiting_vehicle = copy.deepcopy(self.current_record_waiting_vehicle)
    
    def record_current_waiting_vehicle(self):
        self.current_record_waiting_vehicle = copy.deepcopy(self.waiting_time_store)
    
    def get_all_junction_waiting_time(self, vehicles: list):
        """
        This function obtain a dict of waiting_time by junction

        Args:
            - vehicles (list): input list of Vehicle
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        Returns:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        """
        此函数获取交叉口车辆等待时间的字典

        参数:
            - vehicles (list): 车辆的输入列表
            - vehicle_waiting_time (list): input key = v_id (uint32), value = vehicle_waiting_time (list)

        返回:
            - dict: key = vehicle.junction (uint32), value = junction waiting time
        """
        res = {}
        v_num = {}
        for junction_id in self.junction_dict:
            res[junction_id] = 0
            v_num[junction_id] = 0
        for vehicle in vehicles:
            if vehicle.junction != -1 or vehicle.target_junction == -1:
                continue
            if vehicle.v_id in self.waiting_time_store:
                t = self.waiting_time_store[vehicle.v_id]
            else:
                t = 0
            res[vehicle.target_junction] += t
            v_num[vehicle.target_junction] += 1
        # Calculate the average waiting time of all vehicles in the scene
        # 计算场景内所有车辆的平均等待时间
        for junction_id in self.junction_dict:
            if v_num[junction_id] != 0:
                res[junction_id] /= v_num[junction_id]
        return res
