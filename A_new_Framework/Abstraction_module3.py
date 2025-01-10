import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
    def __init__(self, abstraction_dim, action_dim, neu_size,max_sigma = 1e1,min_sigma = 1e-2):
        super(transition_net, self).__init__()
        self.abstraction_dim = abstraction_dim
        self.action_dim = action_dim
        self.neu_size = neu_size
        self.state_layer = nn.Linear(abstraction_dim, neu_size)
        self.action_layer = nn.Linear(action_dim, neu_size)
        self.fc2 = nn.Linear(neu_size*2, neu_size)
        self.mu = nn.Linear(neu_size, abstraction_dim)
        self.fc_std = nn.Linear(neu_size, abstraction_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    # def forward(self, state, action, next_state):
    #     # combine_state = torch.cat((state, action), dim=1)
    #     state = self.state_layer(state)
    #     action = self.action_layer(action)
    #     combine_state = torch.cat([state, action], dim=1)
    #     x = F.relu(self.fc2(combine_state))
    #     mu = self.mu(x)
    #     fc_std = torch.sigmoid(self.fc_std(x))
    #     std = self.min_sigma + (self.max_sigma - self.min_sigma) * fc_std
    #     dist = Normal(mu, std)
    #     log_prob_state = torch.log(torch.exp(dist.log_prob(next_state)) + 1e-6)
    #     return log_prob_state
    def forward(self, state, action):
        # combine_state = torch.cat((state, action), dim=1)
        state = self.state_layer(state)
        action = self.action_layer(action)
        combine_state = torch.cat([state, action], dim=1)
        x = F.relu(self.fc2(combine_state))
        mu = self.mu(x)
        fc_std = torch.sigmoid(self.fc_std(x))
        std = self.min_sigma + (self.max_sigma - self.min_sigma) * fc_std
        return mu,std

    def predict(self, state, action):
        state = self.state_layer(state)
        action = self.action_layer(action)
        combine_state = torch.cat([state, action], dim=1)
        x = F.relu(self.fc2(combine_state))
        mu = self.mu(x)
        fc_std = torch.sigmoid(self.fc_std(x))
        std = self.min_sigma + (self.max_sigma - self.min_sigma) * fc_std
        dist = Normal(mu,std)
        return dist.sample()

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

    def predict(self, state, action,next_state):
        combine_state = torch.cat((state, action, next_state), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist.sample()


class abstraction_module():
    def __init__(self, input_size, output_size, neu_size, action_size, lr, device):
        self.phi_net = phi_net(input_size, output_size, neu_size).to(device)
        self.phi_net_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=lr)
        self.tau = 0.005
        self.phi_net_target = copy.deepcopy(self.phi_net)
        self.discount = 0.99
        self.bisim_coef = 0.5
        for p in self.phi_net.parameters():
            p.requires_grad = False
        self.transition_net = transition_net(output_size, action_size, neu_size).to(device)
        self.transition_net_optimizer = torch.optim.Adam(self.transition_net.parameters(), lr=lr)
        self.reward_net = reward_net(output_size, action_size, 1, neu_size).to(device)
        self.reward_net_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)

    def calculate_transition_reward_loss(self,s, a, r, s_next):
        s_abs = self.phi_net(s)
        s_next_abs_mu,s_next_abs_std = self.transition_net(s, a)
        s_next_abs = self.phi_net(s_next)
        diff = (s_next_abs_mu - s_next_abs.detach())/s_next_abs_std
        transition_loss = torch.mean(0.5*diff.pow(2) + torch.log(s_next_abs_std))

        predict_s_next_abs = self.transition_net.predict(s_abs,a)
        predict_r = self.reward_net.predict(s_abs.detach(),a,predict_s_next_abs)
        reward_loss = F.mse_loss(predict_r, r)
        total_loss = reward_loss + transition_loss
        return total_loss

    def calculate_phi_loss(self,s, a, r):
        s_abs = self.phi_net(s)
        batch_size = s.shape[0]
        perm = np.random.permutation(batch_size)
        s_abs2 = s_abs[perm]
        with torch.no_grad():
            s_next_mu1,s_next_std1 = self.transition_net(s_abs, a)
            r2 = r[perm]

        s_next_mu2 = s_next_mu1[perm]
        s_next_std2 = s_next_std1[perm]
        z_dist = F.smooth_l1_loss(s_abs,s_abs2,reduction='none')
        r_dist = F.smooth_l1_loss(r,r2,reduction='none')
        transition_dist = torch.sqrt((s_next_mu1 - s_next_mu2).pow(2) + (s_next_std1 - s_next_std2).pow(2))
        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()
        return loss
    def train(self,s,a,r,s_next,dw):
        transition_loss = self.calculate_transition_reward_loss(s,a,r,s_next)
        phi_loss = self.calculate_phi_loss(s,a,r)
        total_loss = self.bisim_coef*phi_loss + transition_loss
        self.phi_net_optimizer.zero_grad()
        self.reward_net_optimizer.zero_grad()
        self.transition_net_optimizer.zero_grad()
        total_loss.backward()
        self.phi_net_optimizer.step()
        self.reward_net_optimizer.step()
        self.transition_net_optimizer.step()


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