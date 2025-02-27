import numpy as np
import torch
import torch.nn as nn
from model import PPO

kwargs = {'state_dim': 3, 'neu_size': 56, 'action_dim': 3, 'data_size': 1200, 'entropy_coef_decay': 0.9, 'gamma': 0.95,
          'lambd': 0.85, 'batch_size': 128, 'epochs': 100, 'clip_rate': 0.1, 'entropy_coef': 0.08, 'l2_reg': 0.0001,
          'lr': 0.01}
model = PPO(**kwargs);
model.actor.load_state_dict(torch.load('ppo_test2_actor.pth'));
model.critic.load_state_dict(torch.load('ppo_test2_critic.pth'));
# a = np.array([1,2,3,4,5]);
# b = np.array([a[1],a[3]]);
# print(id(a))
# a = exchange(a);
# print(a)
# print(id(a))
# s = np.array([0.0,0.0,0.1,0.0,0.0,0.])
s = np.array([0.05,0.00,0.1]);
s = torch.from_numpy(s).float();
print(0%5)