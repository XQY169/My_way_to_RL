# My_way_to_RL
将研一一年的强化学习的学习经历、经验分享于此。

## A new Framework
构建了一个联合训练框架， 主要功能有：
- 首先使用envpool并行开启多个环境，提高采样速率。
- 训练过程与main函数相独立，将保存模型训练文件、开启tensorboard、关闭tensorboard等代码集成在一个Training类中。

## HER_SAC
手动修改环境奖励，将环境变成一个稀疏奖励环境，使用HER采样，对比了不同的采样方式，并绘图观察其性能。

## POMDP
手动修改Hopper-v4环境，使其变成一个POMDP环境，并在Actor和Critic中集成lstm模块，用来处理序列输入。序列输入长度可调。

## Q-learning
一个简单的Q-learning算法，以及自制的简单RL环境。

## State Abstraction
针对PPO算法和SAC算法设计了两种集成策略，抽象方法采用深度双模拟抽象，并搭配构建重建损失防止抽象退化成简单抽象。

## Imitation_Learning_clone
模仿学习中行为克隆算法的尝试。使用的RL是SAC算法。

## ray调参框架
envpool添加环境方法较为底层，不甚方便。同时，由于大规模并行实验的需要，开发此调参框架。同时可以开启多个trial，大大提高了实验效率。使用的强化学习算法为PPO，这里引入了lstm模块用来处理POMDP问题。有两种采样方式，基于envpool的采样和基于ray.remote函数的采样，因此在使用envpool中本身就有的环境时，envpool采样效率要更高一些。而在处理envpool中没有的环境时，使用remote采样，采样阶段使用cpu，而在训练阶段使用gpu，加速训练。

## version_1_2_4_ppo
基于gym-pybullet-drone项目中的BaseAviary.py基类，扩充改写成一个能够进行人机交互的强化学习环境。本项目采用多进程，一个进程用于drone环境采样，一个进程用于人机交互界面的显示。训练模式分为有界面模式和无界面模式，有界面模式可以实时看到采样过程，无界面模式采样和训练速度更快。同时ppo训练得分会实时显示在人机交互界面面板上。
