#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


def on_enter_lane(vehicle):
    """
    This function determines whether the vehicle is located on the enter lane

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于进口车道上

    参数:
        - vehicle
    """
    lane_id = vehicle.lane
    inlane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    if lane_id in inlane_code and vehicle.target_junction != -1:
        return True
    else:
        return False


def in_junction(vehicle):
    """
    This function determines whether the vehicle is located in the junction

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于交叉口中

    参数:
        - vehicle
    """
    junction = vehicle.junction
    target_junction = vehicle.target_junction
    if junction != -1:
        return True
    else:
        return False


def on_depart_lane(vehicle):
    """
    This function determines whether the vehicle is located on the depart lane

    Args:
        - vehicle
    """
    """
    此函数判断车辆是否位于出口车道上

    参数:
        - vehicle
    """
    junction = vehicle.junction
    target_junction = vehicle.target_junction
    # Prevent vehicles in the right turn lane from being judged as being in the exit lane
    # 避免车辆在右转车道被判定为在出口车道上
    if (on_enter_lane(vehicle) or in_junction(vehicle)) or (junction == -1 and target_junction != -1):
        return False
    else:
        return True


def get_lane_code(vehicle):
    """
    This function divides each import lane into a different number of grids according to
    different rules and classifies them

    Args:
        - lane_id: The ID of the lane where the vehicle is located

    Returns:
        - inlane_code: The number assigned to the lane according to the division rule
    """
    """
    此函数将各进口车道按不同规则划分为不同数量的栅格, 并对其进行分类

    参数:
        - lane_id: 车辆所处车道的id

    返回:
        - lane_code: 根据划分规则分配给该车道的编号
    """
    lane_id = vehicle.lane
    lane_code = {
        11: 0,
        10: 1,
        9: 2,
        8: 3,
        129: 4,
        128: 5,
        127: 6,
        126: 7,
        23: 8,
        22: 9,
        21: 10,
        20: 11,
        163: 12,
        162: 13,
    }
    return lane_code.get(lane_id)


def get_webster_lane_group():
    """
    Classify according to the green light phase corresponding to each import lane,
    such as "1" corresponding to the [8, 20] lane corresponding to the north-south left turn phase
    """
    """
    根据各进口车道所对应的绿灯通行相位进行分类, 如"1"对应的[8, 20]车道对应南北左转相位
    """
    lane_group = {
        "0": [11, 10, 9, 23, 22, 21],
        "1": [8, 20],
        "2": [129, 128, 127, 163],
        "3": [126, 162],
    }
    return lane_group
