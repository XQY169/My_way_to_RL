import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class phi_net(nn.Module):
    def __init__(self, state_dim,action_dim, output_dim, neu_size):
        super(phi_net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.neu_size = neu_size
        self.fc1 = nn.Linear(state_dim+action_dim, neu_size)
        self.fc2 = nn.Linear(state_dim, neu_size)
        self.lstm_layer = nn.LSTM(neu_size, neu_size, batch_first=True)
        self.mu = nn.Linear(neu_size*2, output_dim)
        # self.log_std = nn.Linear(neu_size, output_dim)

    def forward(self, state):
        combine_state = state[:,:-1,:]
        last_state = state[:,-1,:].squeeze(1)
        last_state = last_state[:,:self.state_dim]
        x1 = torch.tanh(self.fc1(combine_state))
        _, (out_hx, out_cx) = self.lstm_layer(x1)
        out_hx = out_hx.squeeze(0)
        a = torch.relu(out_hx)
        b = torch.relu(self.fc2(last_state))
        c = torch.cat((a,b),1)
        x = self.mu(c)
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

    def forward(self, state, action):
        combine_state = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        return mu

    # def predict(self, state, action):
    #     state = torch.from_numpy(state).unsqueeze(0).type(torch.float32)
    #     action = torch.from_numpy(action).unsqueeze(0).type(torch.float32)
    #     combine_state = torch.cat((state, action), dim=1)
    #     x = F.relu(self.fc1(combine_state))
    #     x = F.relu(self.fc2(x))
    #     mu = self.mu(x)
    #     return mu


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

    def forward(self, state, action, next_state):
        combine_state = torch.cat((state, action, next_state), dim=1)
        x = F.relu(self.fc1(combine_state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        return mu


class abstraction_module():
    def __init__(self, input_size, output_size, neu_size, action_size, lr, device,scheluler = None):
        self.phi_net = phi_net(input_size,action_size, output_size, neu_size).to(device)
        self.phi_net_optimizer = torch.optim.Adam(self.phi_net.parameters(), lr=lr)
        self.scheluler = scheluler
        if scheluler:
            self.scheluler_net = CosineAnnealingLR(self.phi_net_optimizer,T_max = scheluler ,eta_min=5e-6)
        # scheluler = CosineAnnealingLR(self.phi_net_optimizer,T_max = 2000 ,eta_min=5e-6)

        # self.replay_buffer = SAC_ReplayBuffer(input_size,action_size,int(1e6),device)
        # self.tau = 0.1
        # self.phi_net_target = copy.deepcopy(self.phi_net)
        # for p in self.phi_net.parameters():
        #     p.requires_grad = False
        # self.phi_net_decoder = phi_net_decoder(output_size,input_size,neu_size).to(device)
        # self.phi_net_decoder_optimizer = torch.optim.Adam(self.phi_net_decoder.parameters(), lr=lr)

        self.transition_net = transition_net(output_size, action_size, neu_size).to(device)
        self.transition_net_optimizer = torch.optim.Adam(self.transition_net.parameters(), lr=lr)
        self.reward_net = reward_net(output_size, action_size, 1, neu_size).to(device)
        self.reward_net_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr)



    def train_with_data(self,s,a,r,s_next,batch_size = 64):
        perm = np.arange(s.shape[0])
        np.random.shuffle(perm)
        s,a,r,s_next = s[perm],a[perm],r[perm],s_next[perm]
        itera_num = int(math.ceil(s.shape[0] / batch_size))
        for i in range(itera_num):
            index = slice(i * batch_size, min((i + 1) * batch_size, s.shape[0]))
            with torch.no_grad():
                s_next_abs = self.phi_net(s_next[index])
                # s_next_abs = self.add_noise(s_next_abs)
            s_abs = self.phi_net(s[index])
            predict_r = self.reward_net(s_abs, a[index], s_next_abs)
            predict_s_next = self.transition_net(s_abs, a[index])
            # log_prob_resconstruct = self.phi_net_decoder(s_abs,s[index])
            loss =  F.mse_loss(predict_r,r[index]) + F.mse_loss(predict_s_next,s_next_abs)
            #,torch.mean(log_prob_resconstruct)-torch.mean(log_prob_r),torch.mean(log_prob_r)
            # print(torch.mean(log_prob_s),torch.mean(log_prob_r)) - torch.mean(log_prob_resconstruct)
            # print(torch.mean(log_prob_s),torch.mean(log_prob_r),torch.mean(log_prob_resconstruct))- 2.0*torch.mean(log_prob_resconstruct)
            self.transition_net_optimizer.zero_grad();
            self.reward_net_optimizer.zero_grad();
            self.phi_net_optimizer.zero_grad();
            # self.phi_net_decoder_optimizer.zero_grad()
            loss.backward();
            self.phi_net_optimizer.step();
            self.transition_net_optimizer.step()
            # self.phi_net_decoder_optimizer.step()
            self.reward_net_optimizer.step()
        if self.scheluler:
            self.scheluler_net.step()
    def save(self, path, num):
        torch.save(self.phi_net.state_dict(), path + f'phi_net{num}.pth')
        torch.save(self.transition_net.state_dict(), path + f'transition_net{num}.pth')
        torch.save(self.reward_net.state_dict(), path + f'reward_net{num}.pth')
        # torch.save(self.phi_net_decoder.state_dict(),path+f'phi_net_decoder{num}.pth')

    def load(self, path, num):
        self.phi_net.load_state_dict(torch.load(path + f'phi_net{num}.pth'))
        self.transition_net.load_state_dict(torch.load(path + f'transition_net{num}.pth'))
        self.reward_net.load_state_dict(torch.load(path + f'reward_net{num}.pth'))
        # self.phi_net_decoder.load_state_dict(torch.load(path + f'phi_net_decoder{num}.pth'))