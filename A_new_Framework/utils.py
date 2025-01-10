import numpy as np
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import os
import torch


# class GaussianActor_musigma(nn.Module):
#     def __init__(self, state_dim, action_dim, net_width):
#         super(GaussianActor_musigma, self).__init__()
#
#         self.l1 = nn.Linear(state_dim, net_width)
#         self.l2 = nn.Linear(net_width, net_width)
#         self.mu_head = nn.Linear(net_width, action_dim)
#         self.sigma_head = nn.Linear(net_width, action_dim)
#
#     def forward(self, state):
#         a = torch.relu(self.l1(state))
#         a = torch.relu(self.l2(a))
#         mu = 2.0*torch.tanh(self.mu_head(a))
#         sigma = F.softplus( self.sigma_head(a) )
#         return mu,sigma
#
#     def get_dist(self, state):
#         mu,sigma = self.forward(state)
#         dist = Normal(mu,sigma)
#         return dist
#
#     def deterministic_act(self, state):
#         mu, sigma = self.forward(state)
#         return mu
class Actor(nn.Module):
    def __init__(self, state_size, action_size, neu_size,mode,action_expand):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.mu = nn.Linear(neu_size, action_size)
        self.log_std = nn.Linear(neu_size, action_size)
        self.mode = mode;
        # self.MAX = 2;
        # self.MIN = -20;
        self.expand = action_expand

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

        a = self.expand*torch.tanh(u);

        if with_logprob:
            log_prob_y = (dist.log_prob(u) - torch.log(1-a.pow(2)/(self.expand**2)+1e-6)-torch.log(torch.tensor([self.expand],dtype=torch.float))).sum(axis = 1,keepdim = True);
        else:
            log_prob_y = None;
        return a,log_prob_y

# class Critic2(nn.Module):
#     def __init__(self, state_dim,net_width):
#         super(Critic2, self).__init__()
#
#         self.C1 = nn.Linear(state_dim, net_width)
#         self.C2 = nn.Linear(net_width, net_width)
#         self.C3 = nn.Linear(net_width, 1)
#
#     def forward(self, state):
#         v = torch.tanh(self.C1(state))
#         v = torch.tanh(self.C2(v))
#         v = self.C3(v)
#         return v
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

class ReplayBuffer():
    def __init__(self, state_size, action_size, max_size,num_envs,device):
        self.device = device
        self.max_size = max_size
        self.num_envs = num_envs
        self.state_size = state_size
        self.action_size = action_size
        self.ptr = 0;
        self.size = 0;
        self.s = torch.zeros((max_size,num_envs,state_size),dtype=torch.float,device = self.device);
        self.a = torch.zeros((max_size,num_envs,action_size),dtype=torch.float,device = self.device);
        self.r = torch.zeros((max_size,num_envs,1),dtype=torch.float,device = self.device);
        self.s_next = torch.zeros((max_size,num_envs,state_size),dtype=torch.float,device = self.device);
        self.dw = torch.zeros((max_size,num_envs,1),dtype = torch.bool,device = self.device);

    def add(self, state, action, reward, next_state, done):
        self.s[self.ptr] = torch.from_numpy(state).to(self.device);
        self.a[self.ptr] = torch.from_numpy(action).to(self.device);
        self.r[self.ptr] = torch.from_numpy(reward[:,np.newaxis]).to(self.device);
        self.s_next[self.ptr] = torch.from_numpy(next_state).to(self.device);
        self.dw[self.ptr] = torch.from_numpy(done[:,np.newaxis]).to(self.device);
        self.ptr = (self.ptr + 1) % self.max_size;
        self.size = min((self.size+1),self.max_size);

    def sample(self, batch_size):
        ind = torch.randint(0,self.size*self.num_envs,device = self.device,size = (min(self.size*self.num_envs,batch_size),));
        first_item = ind // self.num_envs
        second_item = ind % self.num_envs
        return (self.s[first_item,second_item],self.a[first_item,second_item],
                self.r[first_item,second_item],self.s_next[first_item,second_item],self.dw[first_item,second_item])
def moving_average(data,window_size):
    return np.array([np.sum(data[i:i + window_size])/window_size for i in range(len(data) - window_size + 1)])

def save_results(rewards_average,rewards,model,name,plt,**kwargs,):
    new_folder_path = os.path.join(os.getcwd(),name);
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