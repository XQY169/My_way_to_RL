# if __name__ == '__main__':
#     from IPython import get_ipython;
#     get_ipython().run_line_magic('reset', '-f');
#     import sys;
#     sys.modules['pydev_umd'] = None;

import numpy as np

from Myenv import Myenv
from Q_model import Q_model
import torch.nn.functional as F
if __name__ == '__main__':
    episodes = 20000;
    cell_num =5;
    alpha = 1;
    gamma = 0.99;
    epsilon = 0.1;
    env = Myenv(cell_num,is_trea=0);
    # model = Q_model(cell_num*(cell_num-1)*(cell_num-2)+cell_num*(cell_num-1),2,alpha,gamma,epsilon);
    model = Q_model(cell_num * (cell_num - 1), 2, alpha, gamma, epsilon);
    rewards = [];
    for i in range(episodes):
        cur_state,mode_num = env.init_state();
        print('----------------------------------init_state---------------------------------------');
        # print('init_state:',cur_state);
        ret = 1;
        total_reward = 0;
        while ret  == 1:

            action = model.choose_action(mode_num);
            print('cur_state:', cur_state,'action:',action);
            ret, reward,mode_num2, cur_state = env.step(action);
            model.update_q_table(mode_num,mode_num2,action,reward,ret);
            mode_num = mode_num2;

            total_reward = total_reward + reward;

        rewards.append(total_reward);
    print(model.q_table)
    print(F.softmax(model.pi_table,dim = 1))
    # print(rewards)