import numpy as np
import torch.nn as nn
from torch.distributions import Normal,Beta
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

        u = torch.clamp(u, min=-7.5, max=7.5)
        a = self.expand*torch.tanh(u);

        if with_logprob:
            log_prob_y = (dist.log_prob(u) - torch.log(1-a.pow(2)/(self.expand**2+1e-6))-torch.log(torch.tensor([self.expand],dtype=torch.float))).sum(dim = 1,keepdim = True);
        else:
            log_prob_y = None;
        if torch.isnan(log_prob_y).any():
            print(u,a,log_prob_y,dist.log_prob(u),torch.log(1-a.pow(2)/(self.expand**2+1e-6)))
        return a,log_prob_y

    def get_logprob(self,state,action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x);
        std = torch.exp(log_std);
        dist = Normal(mu, std);
        clamp_action = torch.clamp(action,-self.expand+1e-6,self.expand - 1e-6)
        u = torch.atanh(clamp_action/self.expand)
        print('u',u)
        logprob_a = (dist.log_prob(u) - torch.log(1-clamp_action.pow(2)/(self.expand**2))-torch.log(torch.tensor([self.expand],dtype=torch.float))).sum(dim = 1,keepdim = True);
        print('logprob_a',logprob_a)
        return logprob_a
class PPO_Actor(nn.Module):
    def __init__(self, state_size, action_size, neu_size):
        super(PPO_Actor, self).__init__()

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

# class PPO_Actor(nn.Module):
#     def __init__(self, state_size, action_size, neu_size,action_expand):
#         super(PPO_Actor, self).__init__()
#         self.fc1 = nn.Linear(state_size, neu_size)
#         self.fc2 = nn.Linear(neu_size, neu_size)
#         self.mu = nn.Linear(neu_size, action_size)
#         self.std = nn.Linear(neu_size, action_size)
#         self.expand = action_expand
#
#     def forward(self, state,deterministic,with_logprob):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         mu = self.expand * torch.tanh(self.mu(x))
#         std = F.softplus(self.std(x));
#         if deterministic:
#             return mu
#         else:
#             dist = Normal(mu, std)
#             a = dist.sample()
#             a = torch.clamp(a, min=-self.expand, max=self.expand)
#             if with_logprob:
#                 return a,dist.log_prob(a)
#
#     def get_logprob(self,state,action):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         mu = self.expand * torch.tanh(self.mu(x))
#         std = F.softplus(self.std(x));
#         dist = Normal(mu, std);
#         return dist.log_prob(action)

class Q_Critic(nn.Module):
    def __init__(self, state_size, action_size, neu_size,):
        super(Q_Critic, self).__init__()
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

class V_Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(V_Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v
class SAC_ReplayBuffer():
    def __init__(self, state_size, action_size, max_size,device):
        self.device = device
        self.max_size = max_size
        self.state_size = state_size
        self.action_size = action_size
        self.ptr = 0;
        self.size = 0;
        self.s = np.zeros((max_size,state_size),dtype=np.float32);
        self.a = np.zeros((max_size,action_size),dtype=np.float32);
        self.r = np.zeros((max_size,1),dtype=np.float32);
        self.s_next = np.zeros((max_size,state_size),dtype=np.float32);
        self.dw = np.zeros((max_size,1),dtype = np.bool_);

    def add(self, state, action, reward, next_state, done):
        valid_index = np.arange(self.ptr,self.ptr+state.shape[0])%self.max_size
        # state,action,reward,next_state,done = state.astype(np.float32),action.astype(np.float32),reward.astype(np.float32),next_state.astype(np.float32),done.astype(np.bool_)
        self.s[valid_index] = state
        self.a[valid_index] = action
        self.r[valid_index] = reward
        self.s_next[valid_index] = next_state
        self.dw[valid_index] = done
        self.ptr = (self.ptr + state.shape[0]) % self.max_size;
        self.size = min((self.size+state.shape[0]),self.max_size);

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size,  size=(min(self.size, batch_size),));
        return (torch.from_numpy(self.s[ind]).to(self.device), torch.from_numpy(self.a[ind]).to(self.device),
                torch.from_numpy(self.r[ind]).to(self.device), torch.from_numpy(self.s_next[ind]).to(self.device), torch.from_numpy(self.dw[ind]).to(self.device))

class SAC_ReplayBuffer2():
    def __init__(self, state_size, action_size, num_envs, max_size,max_steps, device):
        self.device = device
        self.max_size = max_size
        self.num_envs = num_envs
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.ptr_step = np.zeros((num_envs,),dtype=np.float32);
        self.ptr = 0
        self.size = 0;
        self.s = torch.zeros((max_size, state_size), dtype=torch.float, device=self.device);
        self.a = torch.zeros((max_size, action_size), dtype=torch.float, device=self.device);
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.device);
        self.s_next = torch.zeros((max_size, state_size), dtype=torch.float, device=self.device);
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.device);

        self.s_cache = torch.zeros((num_envs,max_steps, state_size), dtype=torch.float, device=self.device);
        self.a_cache = torch.zeros((num_envs,max_steps, action_size), dtype=torch.float, device=self.device);
        self.r_cache = torch.zeros((num_envs,max_steps, 1), dtype=torch.float, device=self.device);
        self.s_next_cache = torch.zeros((num_envs,max_steps, state_size), dtype=torch.float, device=self.device);
        self.dw_cache = torch.zeros((num_envs,max_steps, 1), dtype=torch.bool, device=self.device);

    def add(self, state, action, reward, next_state, done,valid_index):
        state, action, reward, next_state, done = state.astype(np.float32), action.astype(
            np.float32), reward.astype(np.float32), next_state.astype(np.float32), done.astype(np.bool_)
        self.s_cache[valid_index,self.ptr_step[valid_index],:] = torch.from_numpy(state).to(self.device);
        self.a_cache[valid_index,self.ptr_step[valid_index],:] = torch.from_numpy(action).to(self.device);
        self.r_cache[valid_index,self.ptr_step[valid_index],:] = torch.from_numpy(reward[:, np.newaxis]).to(self.device);
        self.s_next_cache[valid_index,self.ptr_step[valid_index],:] = torch.from_numpy(next_state).to(self.device);
        self.dw_cache[valid_index,self.ptr_step[valid_index],:] = torch.from_numpy(done[:, np.newaxis]).to(self.device);
        self.ptr_step[valid_index]  = (self.ptr_step[valid_index] + 1)

    def merge_cache(self):
        cache_s_list = [self.s_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_a_list = [self.a_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_r_list = [self.r_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_s_next_list = [self.s_next_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_dw_list = [self.dw_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        merge_s_cache = torch.cat(cache_s_list, dim=0)
        merge_a_cache = torch.cat(cache_a_list, dim=0)
        merge_r_cache = torch.cat(cache_r_list, dim=0)
        merge_s_next_cache = torch.cat(cache_s_next_list, dim=0)
        merge_dw_cache = torch.cat(cache_dw_list, dim=0)

        self.ptr_step = np.zeros((self.num_envs,),dtype=np.float32);
        valid_index = np.arange(self.ptr,self.ptr+merge_s_cache.shape[0])%self.max_size
        self.s[valid_index,:] = merge_s_cache
        self.a[valid_index,:] = merge_a_cache
        self.r[valid_index,:] = merge_r_cache
        self.s_next[valid_index,:] = merge_s_next_cache
        self.dw[valid_index,:] = merge_dw_cache
        self.ptr = (self.ptr + merge_s_cache.shape[0]) % self.max_size;
        self.size = min((self.size+merge_s_cache.shape[0]),self.max_size);

    def sample(self, batch_size):
        ind = torch.randint(0,self.size,device = self.device,size = (min(self.size,batch_size),));
        return (self.s[ind],self.a[ind],
                self.r[ind],self.s_next[ind],self.dw[ind])

class PPO_ReplayBuffer():
    def __init__(self, state_size, action_size, num_envs, max_steps, device):
        self.device = device
        self.num_envs = num_envs
        self.state_size = state_size
        self.action_size = action_size
        self.max_steps = max_steps
        self.ptr_step = np.zeros((num_envs,),dtype=np.int_);

        self.s_cache = np.zeros((num_envs,max_steps, state_size), dtype=np.float32);
        self.a_cache = np.zeros((num_envs,max_steps, action_size), dtype= np.float32);
        self.logprob_a = np.zeros((num_envs,max_steps,action_size),dtype = np.float32);
        self.r_cache = np.zeros((num_envs,max_steps, 1), dtype=np.float32);
        self.s_next_cache = np.zeros((num_envs,max_steps, state_size), dtype=np.float32,);
        self.done_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);
        self.dw_cache = np.zeros((num_envs,max_steps, 1), dtype=np.bool_);


    def add(self, state, action,logprob, reward, next_state, done,dw,valid_index):
        self.s_cache[valid_index,self.ptr_step[valid_index],:] = state
        self.a_cache[valid_index,self.ptr_step[valid_index],:] = action
        self.logprob_a[valid_index,self.ptr_step[valid_index],:] = logprob
        self.r_cache[valid_index,self.ptr_step[valid_index],:] = reward[:, np.newaxis]
        self.s_next_cache[valid_index,self.ptr_step[valid_index],:] = next_state
        self.done_cache[valid_index,self.ptr_step[valid_index],:] = done[:, np.newaxis]
        new_dw = np.logical_or(done,dw)
        self.dw_cache[valid_index,self.ptr_step[valid_index],:] = new_dw[:, np.newaxis]
        self.ptr_step[valid_index]  = (self.ptr_step[valid_index] + 1)

    def get_data(self):
        cache_s_list = [self.s_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_a_list = [self.a_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_logprob_a_list = [self.logprob_a[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_r_list = [self.r_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
        cache_s_next_list = [self.s_next_cache[i,np.arange(self.ptr_step[i]),] for i in range(self.num_envs)]
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