import os
from enum import Enum
import numpy as np
import time
from gymnasium import spaces
import pybullet as p
from BaseAviary import BaseAviary
from enums import DroneModel,Physics
import math

class My_RL_env(BaseAviary):

    def __init__(self,
                 drone_model:DroneModel=DroneModel.CF2X,
                 initial_xyzs = None,
                 initial_rpys = None,
                 physics:Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq:int = 120,
                 gui = False,
                 record = False,
                 user_debug_gui = False,
                 target_pos = [5,0,0],
                 ):
        super().__init__(drone_model=drone_model,
                         num_drones= 1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui = gui,
                         record = record,
                         obstacles=True,
                         user_debug_gui=user_debug_gui,
                         vision_attributes = False,
        );

        self.target_pos = target_pos;
        self.expect_line = [-0.57,2.3,4];
        self.EPISODE_LEN_SEC = 6;
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,physicsClientId = self.CLIENT)


    def _actionSpace(self):
        return spaces.Box(low = -1*np.ones(4), high = 1*np.ones(4),dtype=np.float32);

    def _preprocessAction(self,
                          action
                          ):

         return np.array(self.HOVER_RPM*(1+0.05*action));
        # return np.array(action)
    def _observationSpace(self):
        return spaces.Box(low = np.array([-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                          high = np.array([1,1,1,1,1,1,1,1,1,1,1,1],dtype=np.float32),
                          dtype=np.float32
                          );
    #
    def _addObstacles(self):
        # robot_id = p.loadURDF('r2d2.urdf',useMaximalCoordinates=True,basePosition=[1,0,1],physicsClientId=self.CLIENT);
        #旋转角[x,y,z]表示绕x轴、y轴、z轴（自身坐标系）旋转的角度，x为正则后仰，y为正则右侧躺，z为正则左转身
        extents = [[0.5,2.5,1],[2.5,0.5,1],[0.5,2.5,1],[2.5,0.5,1]];
        posis = [[3,0,1],[0,3,1],[-3,0,1],[0,-3,1]];
        self.wall_id = [];
        for i in range(4):
            visual_shape_id = p.createVisualShape(
                shapeType = p.GEOM_BOX,
                halfExtents = extents[i],
                physicsClientId = self.CLIENT
            )
            collision_shape_id = p.createCollisionShape(
                shapeType = p.GEOM_BOX,
                halfExtents = extents[i],
                physicsClientId = self.CLIENT
            )
            self.wall_id.append(p.createMultiBody(
                baseMass = 0,
                baseCollisionShapeIndex = collision_shape_id,
                baseVisualShapeIndex = visual_shape_id,
                basePosition = posis[i],
                baseOrientation = p.getQuaternionFromEuler([0,0,0]),
                physicsClientId = self.CLIENT
            ));
        # p.changeVisualShape(self.wall_id,-1,rgbaColor=[1,1,1,1],physicsClientId=self.CLIENT)

    def _computeObs(self):
        obs = self._getDroneStateVector(self.follow_num);
        # print(obs[12])
        obs = self._clipAndNormalizeState(obs);
        # np.array(self.HOVER_RPM * (1 + 0.05 * action))
        action = (obs[16:20]/self.HOVER_RPM - 1)/0.05
        # print(action)
        ret = np.hstack((obs[0:3], obs[7:10],obs[10:13],obs[13:16],action,self.step_counter/(self.CTRL_FREQ*self.EPISODE_LEN_SEC))).reshape(17,)
        return ret.astype('float32');

    def _clipAndNormalizeState(self,obs):
        MAX_LIN_VEL_XY = 10.0;
        MAX_LIN_VEL_Z = 10.0;
        MAX_XY = 10.0;
        MAX_Z = 50.0;
        MAX_PIITCH_ROLL = np.pi;
        clipped_pos_xy = np.clip(obs[0:2],-MAX_XY,MAX_XY);
        clipped_pos_z = np.clip(obs[2],0,MAX_Z);
        clipped_rp = np.clip(obs[7:9],-MAX_PIITCH_ROLL,MAX_PIITCH_ROLL);
        clipped_vel_xy = np.clip(obs[10:12],-MAX_LIN_VEL_XY,MAX_LIN_VEL_XY);
        clipped_vel_z = np.clip(obs[12],-MAX_LIN_VEL_Z,MAX_LIN_VEL_Z);

        normalized_pos_xy = clipped_pos_xy/MAX_XY;
        normalized_pos_z = clipped_pos_z/MAX_Z;
        normalized_rp = clipped_rp/MAX_PIITCH_ROLL;
        normalized_y = obs[9]/np.pi;
        normalized_vel_xy = clipped_vel_xy/MAX_LIN_VEL_XY;
        normalized_vel_z = clipped_vel_z/MAX_LIN_VEL_Z;
        normalized_ang_vel = obs[13:16] / np.linalg.norm(obs[13:16]) if np.linalg.norm(obs[13:16]) != 0 else obs[13:16];

        norm_and_clipped = np.hstack([
            normalized_pos_xy, normalized_pos_z,
            obs[3:7],
            normalized_rp,normalized_y,
            normalized_vel_xy, normalized_vel_z,
            normalized_ang_vel,
            obs[16:20]
        ]).reshape(20,);
        return norm_and_clipped;

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return -1 * np.linalg.norm(np.array([0, 0, 2])-state[0:3])**2

    def _computeTerminated(self):
        if self.step_counter/self.CTRL_FREQ > self.EPISODE_LEN_SEC:
            return True;
        else:
            return False;

    def _computeTruncated(self):
        return 0;

    def _computeInfo(self):
        return {"answer": 22}

