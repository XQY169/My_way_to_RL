import numpy as np

from Q_model import Q_model
import torch.nn.functional as F
from matplotlib import pyplot as plt
if __name__ == '__main__':
    alpha = 1;
    gamma = 0.99;
    epsilon = 0.1;
    total_grids = 10
    model = Q_model(total_grids,2,alpha,gamma,epsilon)
    state = 0
    rewardes = []
    for k in range(100):
        rewards = 0
        state = 0
        for i in range(100):

            action = model.choose_action(state)
            next_state = min(max(state + int((action - 0.5)*2),0),total_grids - 1)
            reward = 0.99*(total_grids - next_state) - (total_grids - state)
            rewards += reward
            if next_state == total_grids :
                done = -1
            else:
                done = 0
            # print('state,next_state,action,reward,',state,next_state,action,reward,)
            model.update_q_table(state,next_state,action,reward,done)
            if done == -1:
                break
            state = next_state
        print(rewards,state)
        rewardes.append(rewards)
    plt.plot(range(len(rewardes)),rewardes)
    plt.show()
    print(model.q_table)




