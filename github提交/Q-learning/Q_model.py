# if __name__ == '__main__':
#     import sys;
#     from IPython import get_ipython;
#
#     sys.modules['pydev_umd'] = None;
#     get_ipython().run_line_magic('reset', '-f');

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_model():
    def __init__(self,states_num,actions_num,alpha,gamma,epsilon):
        # self.cell_num = cell_num;
        self.q_table = np.zeros((states_num, actions_num), dtype=np.float32);
        # self.lr = 0.001;
        # self.sac_alpha = 0.4
        # self.pi_table = torch.zeros((states_num,actions_num),dtype = torch.float32,requires_grad=True);
        # self.optimizer = torch.optim.Adam([self.pi_table],lr = self.lr)
        self.states_num = states_num;
        self.actions_num = actions_num;
        self.alpha = alpha;
        self.gamma = gamma;
        self.epsilon = epsilon;

    def choose_action(self,state_mode_num):
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.actions_num);
        else:
            action_idx = np.argmax(self.q_table[state_mode_num]);
        # with torch.no_grad():
        #     softmax_row = F.softmax(self.pi_table[state_mode_num],dim=0)
        #     distribution = torch.distributions.Categorical(softmax_row)
        #     action_idx = distribution.sample().item()
        return action_idx;


    def update_q_table(self,state_mode_num,next_state_mode_num,action,reward,ret):
        q_val = self.q_table[state_mode_num, action];

        # with torch.no_grad():
            # softmax_row = F.softmax(self.pi_table[next_state_mode_num], dim=0)
            # next_log_prob = torch.log(softmax_row+1e-6);
            # q_row = torch.from_numpy(self.q_table[next_state_mode_num]);
        if ret == -1:
            updated_q_val = (1 - self.alpha) * q_val + self.alpha * (reward);
        else:
            updated_q_val = (1 - self.alpha) * q_val + self.alpha * (
                        reward + self.gamma * np.max(self.q_table[next_state_mode_num]));

        self.q_table[state_mode_num, action] = updated_q_val;

        # probs = F.softmax(self.pi_table[state_mode_num], dim=0);
        # log_probs = torch.log(probs + 1e-6);
        # a_loss = torch.sum(probs * (self.sac_alpha*log_probs - torch.from_numpy(self.q_table[state_mode_num])));
        # self.optimizer.zero_grad()
        # a_loss.backward();
        # self.optimizer.step();



