import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
from torch.distributions import Normal
import gym
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_size, action_size, neu_size,mode):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.mu = nn.Linear(neu_size, action_size)
        self.log_std = nn.Linear(neu_size, action_size)
        self.mode = mode;
        self.MAX = 2;
        self.MIN = -20;

    def forward(self, state,deterministic,with_logprob):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x);
        std = torch.exp(log_std);

        dist = Normal(mu, std);
        if deterministic:
            u = mu;
        else:
            if self.mode:
                u = dist.rsample();
            else:
                u = dist.sample();

        a = 2*torch.tanh(u);

        if with_logprob:
            log_prob_y = (dist.log_prob(u) - torch.log(1-a.pow(2)/4+1e-6)-torch.log(torch.tensor([2.0],dtype=torch.float))).sum(axis = 1,keepdim = True);
        else:
            log_prob_y = None;
        return a,log_prob_y

    def back_for_log_prob(self,state,action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x);
        std = torch.exp(log_std);
        dist = Normal(mu, std);
        u = np.arctanh(action/2.0)
        return dist.log_prob(u)

class attention_net(nn.Module):
    def __init__(self, state_size, neu_size):
        super(attention_net, self).__init__()
        self.fc1 = nn.Linear(state_size, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.fc3 = nn.Linear(neu_size, state_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        attention = (torch.tanh(self.fc3(x)) + 1)/2.0
        return attention

class Critic(nn.Module):
    def __init__(self, state_size, action_size, neu_size,):
        super(Critic, self).__init__()
        self.f11 = nn.Linear(state_size+action_size, neu_size)
        self.f12 = nn.Linear(neu_size, neu_size)
        self.f13 = nn.Linear(neu_size, 1);
        self.f21 = nn.Linear(state_size+action_size, neu_size)
        self.f22 = nn.Linear(neu_size, neu_size)
        self.f23 = nn.Linear(neu_size, 1);

    def forward(self, state,action):
        sa = torch.cat([state,action],1);
        x = torch.relu(self.f11(sa))
        x = torch.relu(self.f12(x))
        q1 = self.f13(x);
        x = torch.relu(self.f21(sa))
        x = torch.relu(self.f22(x))
        q2 = self.f23(x);
        return q1,q2;

# class Episode_Cache():
#     def __init__(self, state_size, action_size, max_size,device):
#         self.device = device
#         self.max_size = max_size
#         self.ptr = 0;
#         self.size = 0;
#         self.s = torch.zeros((max_size,state_size),dtype=torch.float,device = self.device);
#         self.a = torch.zeros((max_size,action_size),dtype=torch.float,device = self.device);
#         self.r = torch.zeros((max_size,1),dtype=torch.float,device = self.device);
#         self.s_next = torch.zeros((max_size,state_size),dtype=torch.float,device = self.device);
#         self.dw = torch.zeros((max_size,1),dtype = torch.bool,device = self.device);
#


class ReplayBuffer():
    def __init__(self, state_size, action_size, max_size,device):
        self.device = device
        self.max_size = max_size
        self.ptr = 0;
        self.size = 0;
        self.s = torch.zeros((max_size,state_size),dtype=torch.float,device = self.device);
        self.a = torch.zeros((max_size,action_size),dtype=torch.float,device = self.device);
        self.r = torch.zeros((max_size,1),dtype=torch.float,device = self.device);
        self.s_next = torch.zeros((max_size,state_size),dtype=torch.float,device = self.device);
        self.dw = torch.zeros((max_size,1),dtype = torch.bool,device = self.device);

    def add(self, state, action, reward, next_state, done):
        self.s[self.ptr] = torch.from_numpy(state).to(self.device);
        self.a[self.ptr] = torch.from_numpy(action).to(self.device);
        self.r[self.ptr] = reward;
        self.s_next[self.ptr] = torch.from_numpy(next_state).to(self.device);
        self.dw[self.ptr] = done;
        # print('s_inf',torch.isinf(self.s[self.ptr]).any(),'a_inf',torch.isinf(self.a[self.ptr]).any(),'r_inf',torch.isinf(self.r[self.ptr]).any(),'s__inf',torch.isinf(self.s_next[self.ptr]).any())
        self.ptr = (self.ptr + 1) % self.max_size;
        self.size = min((self.size+1),self.max_size);


    def sample(self, batch_size):
        ind = torch.randint(0,self.size,device = self.device,size = (min(self.size,batch_size),));
        return self.s[ind],self.a[ind],self.r[ind],self.s_next[ind],self.dw[ind]

# class HER_Replay():
#     def __init__(self,capacity,her_k = 4):
#         self.capacity = capacity
#         self.her_k = her_k
#         self.memory = deque(maxlen=capacity)
class SAC():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tau = 0.005;
        self.maxaction = 2.0;
        self.actor = Actor(self.state_size,self.action_size,self.neu_size,mode = 1).to(self.device);
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.lr);
        # self.attention = attention_net(self.state_size,self.neu_size).to(self.device)
        # self.attention_optimizer = torch.optim.Adam(self.attention_net.parameters(),lr = self.lr)
        self.critic = Critic(self.state_size,self.action_size,self.neu_size).to(self.device);
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.lr);
        self.critic_target = copy.deepcopy(self.critic);

        for p in self.critic_target.parameters():
            p.requires_grad = False;
        # self.target_entropy = torch.tensor(self.action_size,dtype=float,device=self.device);
        # self.log_alpha = torch.tensor(np.log(self.alpha),dtype = float,requires_grad=True,device = self.device);
        # self.alpha_optim = torch.optim.Adam([self.log_alpha],lr = self.lr);

        self.replay_buffer = ReplayBuffer(self.state_size,self.action_size,max_size = int(1e6),device = self.device)


    def act(self, state,deterministic,with_logprob):
        # print(state)
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.device);
            a,_ = self.actor(state,deterministic,with_logprob);
        return a.cpu().numpy()[0];

    def train(self):
        s,a,r,s_next,dw = self.replay_buffer.sample(self.batch_size);
        with torch.no_grad():#这个语句包括的所有新增变量都不会带梯度
            a_next,log_prob_a_next= self.actor(s_next,deterministic = False,with_logprob = True);
            q1,q2 = self.critic_target(s_next,a_next);
            target_q = torch.min(q1,q2);
            target_q = r +(~dw)*self.gamma*(target_q - self.alpha*log_prob_a_next);
            # target_q = r +(~dw)*self.gamma*target_q
        current_q1,current_q2 = self.critic(s,a);
        q_loss = F.mse_loss(current_q1,target_q) + F.mse_loss(current_q2,target_q);
        self.critic_optimizer.zero_grad();
        q_loss.backward();
        self.critic_optimizer.step();
        for p in self.critic.parameters():
            p.requires_grad = False;

        a,log_prob_a_y = self.actor(s,deterministic = False,with_logprob = True);

        current_q1,current_q2 = self.critic(s,a);
        Q = torch.min(current_q1,current_q2);
        a_loss = (self.alpha*log_prob_a_y - Q).mean();
        # a_loss = -Q.mean()
        self.actor_optimizer.zero_grad();

        a_loss.backward();
        self.actor_optimizer.step();

        for p in self.critic.parameters():
            p.requires_grad = True;

        for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
            target_param.data.copy_(self.tau*param+(1-self.tau)*target_param);

def moving_average(data,window_size):
    return np.array([np.sum(data[i:i + window_size])/window_size for i in range(len(data) - window_size + 1)])

def save_results(rewards_average,rewards,model,name,plt,**kwargs,):

    new_folder_path = os.path.join(os.getcwd(),name);
    print(new_folder_path)
    os.makedirs(new_folder_path,exist_ok=True)
    rewards = np.array(rewards);
    rewards_average = np.array(rewards_average);
    rewards = rewards[:,np.newaxis];
    rewards_average = rewards_average[:,np.newaxis];
    np.savetxt(os.path.join(new_folder_path,'rewards.csv'),rewards);
    np.savetxt(os.path.join(new_folder_path,'rewards_average.csv'),rewards_average);
    file_path = os.path.join(new_folder_path, 'hyperparameters.txt')
    with open(file_path,'w') as f:
        for key,value in kwargs.items():
            f.write(f'{key}:{value}\n')
    torch.save(model.actor.state_dict(), os.path.join(new_folder_path,'actor.pth'));
    torch.save(model.critic_target.state_dict(), os.path.join(new_folder_path,'critic_target.pth'));
    plt.savefig(os.path.join(new_folder_path,'plot.png'));

def calcu_reward(goal,state):
    tolerance = 0.01
    goal_cos,goal_sin,goal_thdot = goal[0],goal[1],goal[2]
    cos_th,sin_th,thdot = state[0],state[1],state[2]
    costs = (goal_cos - cos_th)**2 + (goal_sin - sin_th)**2 + 0.1*(goal_thdot - thdot)**2
    reward = 0 if costs < tolerance else -1
    return reward;

def get_new_goals(episode_cache,num):
    new_goals = random.sample(episode_cache,num)
    return [np.array(new_goal[3][:3]) for new_goal in new_goals]