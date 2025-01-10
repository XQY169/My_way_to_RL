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

        self.l1 = nn.Linear(state_size, neu_size)
        self.lstm_layer = nn.LSTM(neu_size, neu_size, batch_first=True)

        self.alpha_head = nn.Linear(neu_size, action_size)
        self.beta_head = nn.Linear(neu_size, action_size)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        _,(out_hx,out_cx) = self.lstm_layer(a)
        out_hx = out_hx.squeeze(0)
        a = torch.tanh(out_hx)
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
    def __init__(self, state_dim,net_width):
        super(V_Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.lstm_layer = nn.LSTM(net_width, net_width,batch_first=True)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        _,(out_hx,out_cx) = self.lstm_layer(v)
        out_hx = out_hx.squeeze(0)
        v = torch.tanh(out_hx)
        v = self.C3(v)
        return v

# def process_his_state(s,s_,done,his_length,device):
#     his_s = torch.zeros((s.shape[0],his_length,s.shape[1])).to(device)
#     his_s_ = torch.zeros((s.shape[0],his_length,s.shape[1])).to(device)
#     s = torch.cat((torch.zeros((his_length-1,s.shape[1])),s),dim = 0)
#     s_ = torch.cat((torch.zeros((his_length-1,s_.shape[1])),s_),dim = 0)
#     done = torch.cat((torch.zeros((his_length-1,done.shape[1])),done),dim = 0)
#
#     for i in range(his_s.shape[0]):
#         mid_s = s[i:i+his_length,:]
#         mid_s_ = torch.roll(mid_s,1,0)
#         mid_s_[his_length,:] = s_[i+his_length,:]
#         mid_d = done[i:i+his_length,:]
#         if torch.any(mid_d):
#             max_done_index = torch.max(torch.nonzero(mid_d[:,0]))
#             mid_s[:max_done_index,:] = 0
#             if max_done_index >= 1:
#                 mid_s_[:max_done_index-1,:] = 0
#             his_s[i] = mid_s
#             his_s_[i] = mid_s_
#         else:
#             his_s[i] = mid_s
#             his_s_[i] = mid_s_
#
#     return his_s,his_s_

# class PPO_Critic(nn.Module):
#     def __init__(self, state_dim,
#                  mem_pre_lstm_neu_size = (128,),
#                  mem_lstm_neu_size = (128,),
#                  mem_after_lstm_neu_size = (128,)):
#         super(PPO_Critic, self).__init__()
#         self.state_dim = state_dim
#         self.mem_pre_lstm_layers = nn.ModuleList()
#         self.mem_lstm_layers = nn.ModuleList()
#         self.mem_after_lstm_layers = nn.ModuleList()
#
#         mem_pre_lstm_layer_size = [state_dim] + list(mem_pre_lstm_neu_size)
#         for h in range(len(mem_pre_lstm_layer_size) - 1):
#             self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h], mem_pre_lstm_layer_size[h + 1])
#                                          ,nn.ReLU()]
#
#         mem_lstm_layer_size = [mem_pre_lstm_layer_size[-1]] +  list(mem_lstm_neu_size)
#         for h in range(len(mem_lstm_layer_size) - 1):
#             self.mem_lstm_layers += [nn.LSTM(mem_lstm_layer_size[h], mem_lstm_layer_size[h + 1],batch_first=True)]
#
#         mem_after_lstm_layers_size = [mem_lstm_layer_size[-1]] + list(mem_after_lstm_neu_size)
#         for h in range(len(mem_after_lstm_layers_size) - 1):
#             self.mem_after_lstm_layers += [nn.Linear(mem_after_lstm_layers_size[h],mem_after_lstm_layers_size[h + 1]),
#                                            nn.ReLU()]
#         self.mem_after_lstm_layers += [nn.Linear(mem_after_lstm_layers_size[-1], 1),]
#
#     def forward(self, his_state):
#         for layer in self.mem_pre_lstm_layers:
#             his_state = layer(his_state)
#
#         for layer in self.mem_lstm_layers:
#             his_state,(out_hx,out_cx) = layer(his_state)
#         for layer in self.mem_after_lstm_layers:
#             out_hx = layer(out_hx)
#         return out_hx.squeeze(1)


class PPO_ReplayBuffer():
    def __init__(self, state_size,his_length, action_size, num_envs, max_steps, device):
        self.device = device
        self.num_envs = num_envs
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.ptr_step = np.zeros((num_envs,),dtype=np.int_);

        self.s_cache = np.zeros((num_envs,max_steps,his_length, state_size), dtype=np.float32);
        self.a_cache = np.zeros((num_envs,max_steps, action_size), dtype= np.float32);
        self.logprob_a = np.zeros((num_envs,max_steps,action_size),dtype = np.float32);
        self.r_cache = np.zeros((num_envs,max_steps, 1), dtype=np.float32);
        self.s_next_cache = np.zeros((num_envs,max_steps,his_length, state_size), dtype=np.float32,);
        self.done_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);
        self.dw_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);


    def add(self, state, action,logprob, reward, next_state, done,dw,valid_index):
        self.s_cache[valid_index,self.ptr_step[valid_index],:,:] = state
        self.a_cache[valid_index,self.ptr_step[valid_index],:] = action
        self.logprob_a[valid_index,self.ptr_step[valid_index],:] = logprob
        self.r_cache[valid_index,self.ptr_step[valid_index],:] = reward[:, np.newaxis]
        self.s_next_cache[valid_index,self.ptr_step[valid_index],:,:] = next_state
        self.done_cache[valid_index,self.ptr_step[valid_index],:] = done[:, np.newaxis]
        new_dw = np.logical_or(done,dw)
        self.dw_cache[valid_index,self.ptr_step[valid_index],:] = new_dw[:, np.newaxis]
        self.ptr_step[valid_index]  = (self.ptr_step[valid_index] + 1)

    def get_data(self):
        cache_s_list = [self.s_cache[i,np.arange(self.ptr_step[i]),:,:] for i in range(self.num_envs)]
        cache_a_list = [self.a_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_logprob_a_list = [self.logprob_a[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_r_list = [self.r_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_s_next_list = [self.s_next_cache[i,np.arange(self.ptr_step[i]),:,:] for i in range(self.num_envs)]
        cache_done_list = [self.done_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_dw_list = [self.dw_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]

        merge_s_cache = torch.from_numpy(np.concatenate(cache_s_list, axis=0)).to(self.device)
        merge_a_cache = torch.from_numpy(np.concatenate(cache_a_list, axis=0)).to(self.device)
        merge_logprob_a_cache = torch.from_numpy(np.concatenate(cache_logprob_a_list,axis = 0)).to(self.device)
        merge_r_cache = torch.from_numpy(np.concatenate(cache_r_list, axis=0)).to(self.device)
        merge_s_next_cache = torch.from_numpy(np.concatenate(cache_s_next_list, axis=0)).to(self.device)
        merge_done_cache = torch.from_numpy(np.concatenate(cache_done_list, axis=0)).to(self.device)
        merge_dw_cache = torch.from_numpy(np.concatenate(cache_dw_list, axis=0)).to(self.device)

        self.ptr_step = np.zeros((self.num_envs,),dtype=np.int_);
        return merge_s_cache,merge_a_cache,merge_logprob_a_cache,merge_r_cache,merge_s_next_cache,merge_done_cache,merge_dw_cache

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