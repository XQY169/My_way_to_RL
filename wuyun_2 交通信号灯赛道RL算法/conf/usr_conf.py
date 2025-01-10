#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import yaml


def read_usr_conf(usr_conf_file, logger):
    """
    read usr conf
    读取配置文件
    """
    if not usr_conf_file or not os.path.exists(usr_conf_file):
        return None

    try:
        with open(usr_conf_file, "r") as file:
            loaded_usr_conf = yaml.safe_load(file)
    except yaml.YAMLError as e:
        logger.error(f"yaml.safe_load failed, {usr_conf_file} please check")
        return None

    try:
        if not loaded_usr_conf["traffic_control"]["custom_configuration"]:
            traffic_control = []
        else:
            traffic_control = [
                [
                    item["lane"]["direction"],
                    int(item["lane"]["index"]),
                    int(item["start_time"]),
                    int(item["end_time"]),
                ]
                for item in loaded_usr_conf["traffic_control"]["custom_configuration"]
            ]

        traffic_control_random_count = int(loaded_usr_conf["traffic_control"]["random_count"])

        if not loaded_usr_conf["traffic_accidents"]["custom_configuration"]:
            traffic_accidents = []
        else:
            traffic_accidents = [
                [
                    item["lane"]["direction"],
                    int(item["lane"]["index"]),
                    int(item["start_time"]),
                    int(item["end_time"]),
                ]
                for item in loaded_usr_conf["traffic_accidents"]["custom_configuration"]
            ]
        traffic_accidents_random_count = int(loaded_usr_conf["traffic_accidents"]["random_count"])

        usr_conf = {
            "diy": {
                "car_max_speed": int(loaded_usr_conf["speed_limit"]),
                "weather_id": int(loaded_usr_conf["weather"]),
                "car_accident": traffic_accidents,
                "traffic_accidents_random_count": traffic_accidents_random_count,
                "lanes_disable": traffic_control,
                "traffic_control_random_count": traffic_control_random_count,
                "rush_hour": int(loaded_usr_conf["rush_hour"]),
                "random_seed_for_over_speed_cars": int(loaded_usr_conf["speeding_cars_rate"]),
                "max_waiting_cars": int(loaded_usr_conf["max_waiting_cars"]),
                "max_waiting_duration_for_cars": int(loaded_usr_conf["max_waiting_cars_duration"]),
                "max_step": int(loaded_usr_conf["max_step"]),
            }
        }

        return usr_conf
    except Exception as e:
        logger.error(f"read_usr_conf failed, {usr_conf_file}, {str(e)}, please check")
        return None


def check_usr_conf(usr_conf, logger):
    """
    check usr conf

    检测输入的配置项是否合理
    """
    try:
        car_max_speed = usr_conf["diy"]["car_max_speed"]
        weather_id = usr_conf["diy"]["weather_id"]
        car_accident = usr_conf["diy"]["car_accident"]
        traffic_accidents_random_count = usr_conf["diy"]["traffic_accidents_random_count"]
        lanes_disable = usr_conf["diy"]["lanes_disable"]
        traffic_control_random_count = usr_conf["diy"]["traffic_control_random_count"]
        rush_hour = usr_conf["diy"]["rush_hour"]
        random_seed_for_over_speed_cars = usr_conf["diy"]["random_seed_for_over_speed_cars"]
        max_waiting_cars = usr_conf["diy"]["max_waiting_cars"]
        max_waiting_duration_for_cars = usr_conf["diy"]["max_waiting_duration_for_cars"]
        max_step = usr_conf["diy"]["max_step"]

        if weather_id not in [-1, 0, 1, 2, 3]:
            logger.error(f"Environment Config Error: weather_id should between [0, 1, 2, 3] or -1")
            return False

        if rush_hour not in [-1, 0, 1]:
            logger.error(f"Environment Config Error: rush_hour should between [0, 1] or -1")
            return False

        if not (isinstance(car_accident, list) and isinstance(lanes_disable, list)):
            logger.error(
                f"Traffic accidents (car_accident) or traffic control (lanes_disable): expandable arrays, "
                f"each element represents an accident, each accident contains four elements"
            )
            return False

        if len(car_accident) > 4 or len(lanes_disable) > 4:
            logger.error(
                f"Traffic accidents (car_accident) or traffic control (lanes_disable): expandable arrays, "
                f"length between [0, 4]"
            )
            return False

        if max_step < 1 or max_step > 3600:
            logger.error(f"Environment Config Error: max_step should between [1, 3600]")
            return False

        for each_car_accident_or_lanes_disable in car_accident + lanes_disable:
            if not isinstance(each_car_accident_or_lanes_disable, list) or len(each_car_accident_or_lanes_disable) != 4:
                logger.error(
                    f"Traffic accidents (car_accident) or traffic control (lanes_disable): "
                    f"expandable array, each element represents an accident, each accident contains four elements"
                )
                return False
            direction, index, start_time, end_time = each_car_accident_or_lanes_disable

            if direction not in ["west", "east", "north", "south"]:
                logger.error(f"Environment Config Error: direction should between ['west', 'east', 'north', 'south']")
                return False
            if direction == "west" and index not in [0, 1]:
                logger.error(f"Environment Config Error: direction {direction} index {index} should between [0, 1]")
                return False
            elif direction in ["east", "north", "south"] and index not in [0, 1, 2, 3]:
                logger.error(
                    f"Environment Config Error: direction {direction} index {index} should between [0, 1, 2, 3]"
                )
                return False
            if start_time < 0 or start_time > max_step:
                logger.error(f"Environment Config Error: start_time should between [0, {max_step}]")
                return False
            if end_time <= start_time or end_time > max_step:
                logger.error(f"Environment Config Error: end_time should between ({start_time}, {max_step}]")
                return False

        if (car_max_speed < 20 or car_max_speed > 60) and car_max_speed != -1:
            logger.error(f"Environment Config Error: car_max_speed should between [20, 60] or -1")
            return False
        if (
            random_seed_for_over_speed_cars < 0 or random_seed_for_over_speed_cars > 4
        ) and random_seed_for_over_speed_cars != -1:
            logger.error(f"Environment Config Error: random_seed_for_over_speed_cars should between [0, 4] or -1")
            return False

        if (max_waiting_cars < 150 or max_waiting_cars > 450) and max_waiting_cars != -1:
            logger.error(f"Environment Config Error: max_waiting_cars should between [150, 450] or -1")
            return False

        if (
            max_waiting_duration_for_cars < 80 or max_waiting_duration_for_cars > 200
        ) and max_waiting_duration_for_cars != -1:
            logger.error(f"Environment Config Error: max_waiting_duration_for_cars should between [80, 200] or -1")
            return False

        if traffic_accidents_random_count > 4 or traffic_accidents_random_count < 0:
            logger.error(f"Environment Config Error: traffic_accidents_random_count should between [0, 4]")
            return False
        if traffic_control_random_count > 4 or traffic_control_random_count < 0:
            logger.error(f"Environment Config Error: traffic_control_random_count should between [0, 4]")
            return False

        return True

    except Exception as e:
        logger.error(f"check_usr_conf failed, {str(e)}, please check")
        return False
