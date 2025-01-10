import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


class phi_net(nn.Module):
    def __init__(self, state_dim, output_dim, neu_size):
        super(phi_net, self).__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.neu_size = neu_size
        self.fc1 = nn.Linear(state_dim, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.mu = nn.Linear(neu_size, output_dim)
        # self.log_std = nn.Linear(neu_size, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mu(x)
        return x


class transition_net(nn.Module):
    def __init__(self, abstraction_dim, action_dim, neu_size):
        super(transition_net, self).__init__()
        self.abstraction_dim = abstraction_dim
        self.action_dim = action_dim
        self.neu_size = neu_size
        self.fc1 = nn.Linear(abstraction_dim + action_dim, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.mu = nn.Linear(neu_size, abstraction_dim)
        self.log_std = nn.Linear(neu_size, abstraction_dim)

    def forward(self, state, action, next_state):
        combine_state = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        log_prob_state = torch.log(torch.exp(dist.log_prob(next_state)) + 1e-6)
        return log_prob_state

    def predict(self, state, action):
        state = torch.from_numpy(state).unsqueeze(0).type(torch.float32)
        action = torch.from_numpy(action).unsqueeze(0).type(torch.float32)
        combine_state = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist.sample()


# class transition_net(nn.Module):
#     def __init__(self, abstraction_dim,action_dim,neu_size):
#         super(transition_net, self).__init__()
#         self.abstraction_dim = abstraction_dim
#         self.action_dim = action_dim
#         self.neu_size = neu_size
#         self.fc1 = nn.Linear(abstraction_dim+action_dim, neu_size)
#         self.fc2 = nn.Linear(neu_size, neu_size)
#         self.mu = nn.Linear(neu_size, abstraction_dim)
#
#     def forward(self, state,action):
#         combine_state = torch.cat((state,action),dim=1)
#         x = F.relu(self.fc1(combine_state))
#         x = F.relu(self.fc2(x))
#         mu = self.mu(x)
#         return mu
#     def predict(self,state,action):
#         state = torch.from_numpy(state).unsqueeze(0).type(torch.float32)
#         action = torch.from_numpy(action).unsqueeze(0).type(torch.float32)
#         combine_state = torch.cat((state,action),dim=1)
#         x = F.relu(self.fc1(combine_state))
#         x = F.relu(self.fc2(x))
#         mu = self.mu(x)
#         return mu
class reward_net(nn.Module):
    def __init__(self, abstraction_dim, action_dim, reward_size, neu_size):
        super(reward_net, self).__init__()
        self.fc1 = nn.Linear(abstraction_dim * 2 + action_dim, neu_size)
        self.fc2 = nn.Linear(neu_size, neu_size)
        self.mu = nn.Linear(neu_size, reward_size)
        self.log_std = nn.Linear(neu_size, reward_size)

    def forward(self, state, action, next_state, r):
        combine_state = torch.cat((state, action, next_state), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist.log_prob(r)


class abstraction_module():
    def __init__(self, input_size, output_size, neu_size, action_size, lr, device):
        self.phi_net = phi_net(input_size, output_size, neu_size).to(device)
        self.phi_net_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=lr)
        self.tau = 0.005
        self.phi_net_target = copy.deepcopy(self.phi_net)
        for p in self.phi_net.parameters():
            p.requires_grad = False
        self.transition_net = transition_net(output_size, action_size, neu_size).to(device)
        self.transition_net_optimizer = torch.optim.Adam(self.transition_net.parameters(), lr=lr)
        self.reward_net = reward_net(output_size, action_size, 1, neu_size).to(device)
        self.reward_net_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)


    def add_noise(self, state, mean = 0,std = 1):
        noise = torch.normal(mean=torch.full_like(state, mean), std=std)
        return state + noise
    def train(self,replay_buffer,batch_size):
        s, a, r, s_next, dw = replay_buffer.sample(batch_size)
        rands = np.arange(len(s))
        random.shuffle(rands)
        with torch.no_grad():
            s_next_abs = self.phi_net_target(s_next)
        log_prob_r = self.reward_net(self.phi_net(s), a, s_next_abs, r)
        log_prob_s = self.transition_net(self.phi_net(s), a, s_next_abs)
        # log_prob_s_rand = self.transition_net(self.phi_net(s), a, s_next_abs[rands])
        # print(torch.mean(log_prob_s),torch.mean(log_prob_s_rand))
        loss3 = -torch.mean(log_prob_r) - torch.mean(log_prob_s)

        # loss3 =-torch.mean(log_prob_r)/10.0
        self.transition_net_optimizer.zero_grad();
        # self.reward_net_optimizer.zero_grad();
        self.phi_net_optimizer.zero_grad();
        loss3.backward();
        self.phi_net_optimizer.step();
        self.transition_net_optimizer.step()
        # self.reward_net_optimizer.step()

        for param,target_param in zip(self.phi_net.parameters(),self.phi_net_target.parameters()):
            target_param.data.copy_(self.tau*param+(1-self.tau)*target_param);

    def save(self, path, num):
        torch.save(self.phi_net.state_dict(), path + f'phi_net{num}.pth')
        torch.save(self.transition_net.state_dict(), path + f'transition_net{num}.pth')
        torch.save(self.reward_net.state_dict(), path + f'reward_net{num}.pth')

    def load(self, path, num):
        self.phi_net.load_state_dict(torch.load(path + f'phi_net{num}.pth'))
        self.transition_net.load_state_dict(torch.load(path + f'transition_net{num}.pth'))
        self.reward_net.load_state_dict(torch.load(path + f'reward_net{num}.pth'))