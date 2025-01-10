import torch
import torch.nn.functional as F
import copy
import numpy as np
from utils import Normal_Actor,Q_Critic,SAC_ReplayBuffer
from Abstraction_module4 import abstraction_module

class SAC():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.tau = 0.005;

        self.actor = Normal_Actor(self.state_size,self.action_size,self.neu_size).to(self.device);
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.lr);

        self.critic = Q_Critic(self.state_size,self.action_size,self.neu_size).to(self.device);
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.lr);
        self.critic_target = copy.deepcopy(self.critic);
        if self.abs:
            self.abs_module = abstraction_module(self.state_size1,self.state_size,self.neu_size,self.action_size,self.lr,self.device)

        for p in self.critic_target.parameters():
            p.requires_grad = False;


        self.replay_buffer = SAC_ReplayBuffer(self.state_size1,self.action_size,self.num_envs,max_size = int(1e6),device = self.device)
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_size, dtype=float, requires_grad=True, device=self.device)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)


    def act(self, state,deterministic,with_logprob):
        with torch.no_grad():

            state = torch.FloatTensor(state).to(self.device);
            if self.abs:
                # print('before abs:',state)
                state = self.abs_module.phi_net(state);
                # print('after abs:',state)

            a,_ = self.actor.get_act(state,deterministic,with_logprob);
        return a.cpu().numpy();

    def process_action(self, a):
        return self.action_expand * np.tanh(a)
    def train(self):
        s,a,r,s_next,dw = self.replay_buffer.sample(self.batch_size);
        with torch.no_grad():#这个语句包括的所有新增变量都不会带梯度
            if self.abs:
                s_next_abs = self.abs_module.phi_net(s_next)
                s_abs = self.abs_module.phi_net(s)
            else:
                s_next_abs = s_next
                s_abs = s
            a_next,log_prob_a_next = self.actor.get_act(s_next_abs,deterministic = False,with_logprob = True);
            q1,q2 = self.critic_target(s_next_abs,a_next);
            target_q = torch.min(q1,q2);
            target_q = r +(~dw)*self.gamma*(target_q - self.alpha*log_prob_a_next);
            print(log_prob_a_next)
        current_q1,current_q2 = self.critic(s_abs,a);
        q_loss = F.mse_loss(current_q1,target_q) + F.mse_loss(current_q2,target_q);
        self.critic_optimizer.zero_grad();
        # self.abs_module.phi_net_optimizer.zero_grad();
        q_loss.backward();
        # self.abs_module.phi_net_optimizer.step();
        self.critic_optimizer.step();

        for p in self.critic.parameters():
            p.requires_grad = False;
        a2,log_prob_a_y = self.actor.get_act(s_abs,deterministic = False,with_logprob = True);
        current_q1,current_q2 = self.critic(s_abs,a2);
        Q = torch.min(current_q1,current_q2);
        a_loss = (self.alpha*log_prob_a_y - Q).mean();

        self.actor_optimizer.zero_grad();
        a_loss.backward();
        self.actor_optimizer.step();
        for p in self.critic.parameters():
            p.requires_grad = True;

        # ###################################################双模拟抽象
        if self.abs:
            for i in range(self.abs_train):
                self.abs_module.train(self.replay_buffer,self.abs_batch_size)
        ####################################################inverse model 抽象
        # loss4 = -torch.sum(self.inverse_net(s_abs,self.phi_net(s_next),a2));
        # a3,s = self.inverse_net(s_abs,self.phi_net(s_next))
        # loss4 = F.mse_loss(a2,a3)
        # # print(loss4.item());
        # self.inverse_net_optimizer.zero_grad();
        # self.phi_net_optimizer.zero_grad();
        # loss4.backward();
        # self.inverse_net_optimizer.step();
        # self.phi_net_optimizer.step()

        for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
            target_param.data.copy_(self.tau*param+(1-self.tau)*target_param);

        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_prob_a_y + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()

    def save(self,path,num):
        torch.save(self.actor.state_dict(),path+f'actor{num}.pth')
        torch.save(self.critic.state_dict(),path+f'critic{num}.pth')
        if self.abs:
            self.abs_module.save(path,num)

    def load(self,path,num):
        self.actor.load_state_dict(torch.load(path+f'actor{num}.pth'))
        self.critic.load_state_dict(torch.load(path+f'critic{num}.pth'))
        self.critic_target.load_state_dict(torch.load(path+f'critic{num}.pth'))
        for p in self.critic_target.parameters():
            p.requires_grad = False
        if self.abs:
            self.abs_module.load(path,num)

