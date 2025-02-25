import copy

import numpy as np
import torch.nn as nn
from torch.distributions import Normal,Beta
import torch.nn.functional as F
import os
import torch

class PPO_Actor(nn.Module):
    def __init__(self, state_size, action_size, neu_size):
        super(PPO_Actor, self).__init__()
        self.state_size = state_size
        self.l1 = nn.Linear(state_size, neu_size)
        self.l2 = nn.Linear(neu_size, neu_size)

        self.alpha_head = nn.Linear(neu_size, action_size)
        self.beta_head = nn.Linear(neu_size, action_size)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def get_act(self, state,deterministic,with_logprob):
        alpha, beta = self.forward(state)
        if deterministic:
            mode = (alpha) / (alpha + beta)
            return mode
        else:
            dist = Beta(alpha, beta)
            a = dist.sample()
            if with_logprob:
                return a, dist.log_prob(a)
            else:
                return a

class V_Critic(nn.Module):
    def __init__(self, state_size,net_width):
        super(V_Critic, self).__init__()
        self.state_size = state_size
        self.C1 = nn.Linear(state_size, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v


class PPO_Actor_lstm(nn.Module):
    def __init__(self, state_size, action_size, neu_size):
        super(PPO_Actor_lstm, self).__init__()
        self.state_size = state_size
        self.l1 = nn.Linear(state_size + action_size, neu_size)
        self.l2 = nn.Linear(state_size, neu_size)
        self.lstm_layer = nn.LSTM(neu_size, neu_size, batch_first=True)

        self.alpha_head = nn.Linear(neu_size * 2, action_size)
        self.beta_head = nn.Linear(neu_size * 2, action_size)

    def forward(self, state):
        combine_state = state[:, :-1, :]
        last_state = state[:, -1, :].squeeze(1)
        last_state = last_state[:, :self.state_size]
        a = torch.tanh(self.l1(combine_state))
        _, (out_hx, out_cx) = self.lstm_layer(a)
        out_hx = out_hx.squeeze(0)

        a = torch.tanh(out_hx)
        b = torch.tanh(self.l2(last_state))
        c = torch.cat((a, b), dim=1)
        alpha = F.softplus(self.alpha_head(c)) + 1.0
        beta = F.softplus(self.beta_head(c)) + 1.0

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)

        dist = Beta(alpha, beta)
        return dist

    def get_act(self, state, deterministic, with_logprob):
        alpha, beta = self.forward(state)
        alpha, beta = alpha.squeeze(0), beta.squeeze(0)
        if deterministic:
            mode = (alpha) / (alpha + beta)
            return mode
        else:
            dist = Beta(alpha, beta)
            a = dist.sample()
            if with_logprob:
                return a, dist.log_prob(a)
            else:
                return a


class V_Critic_lstm(nn.Module):
    def __init__(self, state_size, action_size, net_width):
        super(V_Critic_lstm, self).__init__()
        self.state_size = state_size
        self.C1 = nn.Linear(state_size + action_size, net_width)
        self.lstm_layer = nn.LSTM(net_width, net_width, batch_first=True)
        self.C2 = nn.Linear(state_size, net_width)
        self.C3 = nn.Linear(net_width * 2, 1)

    def forward(self, state):
        combine_state = state[:, :-1, :]
        last_state = state[:, -1, :].squeeze(1)
        last_state = last_state[:, :self.state_size]
        v = torch.tanh(self.C1(combine_state))
        _, (out_hx, out_cx) = self.lstm_layer(v)
        out_hx = out_hx.squeeze(0)
        v = torch.tanh(out_hx)
        v_2 = torch.tanh(self.C2(last_state))
        v_3 = torch.cat((v, v_2), 1)
        v_4 = self.C3(v_3)
        return v_4
class PPO_ReplayBuffer():
    def __init__(self, state_size,his_length, action_size, max_steps, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.ptr_step = 0

        self.s = np.zeros((max_steps,his_length, state_size), dtype=np.float32);
        self.a= np.zeros((max_steps, action_size), dtype= np.float32);
        self.logprob_a = np.zeros((max_steps,action_size),dtype = np.float32);
        self.r= np.zeros((max_steps, 1), dtype=np.float32);
        self.s_next= np.zeros((max_steps,his_length, state_size), dtype=np.float32,);
        self.done = np.zeros((max_steps, 1), dtype=np.bool_);
        self.dw= np.zeros((max_steps, 1), dtype=np.bool_);


    def add(self, state, action,logprob, reward, next_state, done,dw):
        valid_index = np.arange(self.ptr_step,min(self.ptr_step+len(state),self.max_steps))
        self.s[valid_index,:,:] = state[:len(valid_index)]
        self.a[valid_index,:] = action[:len(valid_index)]
        self.logprob_a[valid_index,:] = logprob[:len(valid_index)]
        self.r[valid_index,:] = reward[:len(valid_index), np.newaxis]
        self.s_next[valid_index,:,:] = next_state[:len(valid_index)]
        self.done[valid_index,:] = done[:len(valid_index), np.newaxis]
        new_dw = np.logical_or(done,dw)
        self.dw[valid_index,:] = new_dw[:len(valid_index), np.newaxis]
        self.ptr_step = min(self.ptr_step+len(state),self.max_steps)

    def get_data(self):
        merge_s = torch.from_numpy(self.s).to(self.device)
        merge_a= torch.from_numpy(self.a).to(self.device)
        merge_logprob_a = torch.from_numpy(self.logprob_a).to(self.device)
        merge_r= torch.from_numpy(self.r).to(self.device)
        merge_s_next= torch.from_numpy(self.s_next).to(self.device)
        merge_done = torch.from_numpy(self.done).to(self.device)
        merge_dw = torch.from_numpy(self.dw).to(self.device)

        self.ptr_step = 0
        return merge_s,merge_a,merge_logprob_a,merge_r,merge_s_next,merge_done,merge_dw
# class PPO_ReplayBuffer():
#     def __init__(self, state_size,his_length, action_size, num_envs, max_steps, device):
#         self.device = device
#         self.num_envs = num_envs
#         self.state_size = state_size
#         self.action_size = action_size
#         self.max_steps = max_steps
#         self.ptr_step = np.zeros((num_envs,),dtype=np.int_);
#
#         self.s_cache = np.zeros((num_envs,max_steps,his_length, state_size), dtype=np.float32);
#         self.a_cache = np.zeros((num_envs,max_steps, action_size), dtype= np.float32);
#         self.logprob_a = np.zeros((num_envs,max_steps,action_size),dtype = np.float32);
#         self.r_cache = np.zeros((num_envs,max_steps, 1), dtype=np.float32);
#         self.s_next_cache = np.zeros((num_envs,max_steps,his_length, state_size), dtype=np.float32,);
#         self.done_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);
#         self.dw_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);
#
#
#     def add(self, state, action,logprob, reward, next_state, done,dw,valid_index):
#         self.s_cache[valid_index,self.ptr_step[valid_index],:,:] = state
#         self.a_cache[valid_index,self.ptr_step[valid_index],:] = action
#         self.logprob_a[valid_index,self.ptr_step[valid_index],:] = logprob
#         self.r_cache[valid_index,self.ptr_step[valid_index],:] = reward[:, np.newaxis]
#         self.s_next_cache[valid_index,self.ptr_step[valid_index],:,:] = next_state
#         self.done_cache[valid_index,self.ptr_step[valid_index],:] = done[:, np.newaxis]
#         new_dw = np.logical_or(done,dw)
#         self.dw_cache[valid_index,self.ptr_step[valid_index],:] = new_dw[:, np.newaxis]
#         self.ptr_step[valid_index]  = (self.ptr_step[valid_index] + 1)
#
#     def get_data(self):
#         cache_s_list = [self.s_cache[i,np.arange(self.ptr_step[i]),:,:] for i in range(self.num_envs)]
#         cache_a_list = [self.a_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
#         cache_logprob_a_list = [self.logprob_a[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
#         cache_r_list = [self.r_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
#         cache_s_next_list = [self.s_next_cache[i,np.arange(self.ptr_step[i]),:,:] for i in range(self.num_envs)]
#         cache_done_list = [self.done_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
#         cache_dw_list = [self.dw_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
#
#         merge_s_cache = torch.from_numpy(np.concatenate(cache_s_list, axis=0)).to(self.device)
#         merge_a_cache = torch.from_numpy(np.concatenate(cache_a_list, axis=0)).to(self.device)
#         merge_logprob_a_cache = torch.from_numpy(np.concatenate(cache_logprob_a_list,axis = 0)).to(self.device)
#         merge_r_cache = torch.from_numpy(np.concatenate(cache_r_list, axis=0)).to(self.device)
#         merge_s_next_cache = torch.from_numpy(np.concatenate(cache_s_next_list, axis=0)).to(self.device)
#         merge_done_cache = torch.from_numpy(np.concatenate(cache_done_list, axis=0)).to(self.device)
#         merge_dw_cache = torch.from_numpy(np.concatenate(cache_dw_list, axis=0)).to(self.device)
#
#         self.ptr_step = np.zeros((self.num_envs,),dtype=np.int_);
#         return merge_s_cache,merge_a_cache,merge_logprob_a_cache,merge_r_cache,merge_s_next_cache,merge_done_cache,merge_dw_cache

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