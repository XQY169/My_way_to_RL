import torch.nn as nn
import torch
import copy
import math
import numpy as np
from torch.distributions import Categorical


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
class Actor(nn.Module):
    def __init__(self,input_size,neu_size,out_size):
        super(Actor,self).__init__();
        self.model = nn.Sequential(
            nn.Linear(input_size,neu_size),
            nn.Tanh(),
            nn.Linear(neu_size,neu_size),
            nn.Tanh(),
            nn.Linear(neu_size,out_size),
        );

    def forward(self,x):
        # prob = self.model(x);
        # return prob;
        pass;

    def pi(self,x,action_dim = 0):
        prob = self.model(x);
        prob = torch.softmax(prob,dim=action_dim);
        return prob;


class Critic(nn.Module):
    def __init__(self,input_size,neu_size):
        super(Critic,self).__init__();
        self.model = nn.Sequential(
            nn.Linear(input_size,neu_size),
            nn.Tanh(),
            nn.Linear(neu_size,neu_size),
            nn.Tanh(),
            nn.Linear(neu_size,1),
        )
    def forward(self,x):
        return self.model(x);


class PPO():
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        #构造actor，critic网络,变量有self.state_dim,self.neu_size,self.action_dim,self.lr
        self.actor = Actor(self.state_dim,self.neu_size,self.action_dim).to(device);
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.lr);
        self.critic = Critic(self.state_dim,self.neu_size).to(device);
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.lr);

        self.s_holder = np.zeros((self.data_size,self.state_dim),dtype=np.float32);
        self.a_holder = np.zeros((self.data_size,1),dtype=np.int64);
        self.r_holder = np.zeros((self.data_size,1),dtype = np.float32);
        self.next_s_holder = np.zeros((self.data_size,self.state_dim),dtype=np.float32);
        self.old_proba_holder = np.zeros((self.data_size,1),dtype=np.float32);
        self.dones_holder = np.zeros((self.data_size,1),dtype = np.bool_);
        self.dw_holder = np.zeros((self.data_size,1),dtype=np.bool_);

        self.idx = 0;
        self.idx_flag = 0;

    def act(self,s):
        s = np.array([s[2],s[8],s[12],s[16]]);
        s = torch.from_numpy(s).float().to(device);
        with torch.no_grad():
            pi = self.actor.pi(s,action_dim=0);
            m = Categorical(pi);
            a = m.sample().item();
            pi_a = pi[a].item();#item()用来转换成标量
            return a,pi_a;

    def learn(self):
        self.entropy_coef *= self.entropy_coef_decay;#熵的正则化
        data_size = self.idx if self.idx_flag == 0 else self.data_size;
        idxes = np.arange(data_size);
        s = torch.from_numpy(self.s_holder[idxes]).to(device);
        a = torch.from_numpy(self.a_holder[idxes]).to(device);
        r = torch.from_numpy(self.r_holder[idxes]).to(device);
        s_next = torch.from_numpy(self.next_s_holder[idxes]).to(device);
        old_proba = torch.from_numpy(self.old_proba_holder[idxes]).to(device);
        # print(old_proba)
        done = torch.from_numpy(self.dones_holder[idxes]).to(device);
        dw = torch.from_numpy(self.dw_holder[idxes]).to(device);
        with torch.no_grad():
            vs = self.critic(s);
            vs_ = self.critic(s_next);

            deltas = r + self.gamma*vs_*(~dw) - vs;
            deltas = deltas.cpu().flatten().numpy();
            adv = [0];

            for dlt,done in zip(deltas[::-1],done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma*self.lambd*adv[-1]*(~done);
                adv.append(advantage);
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1]);
            adv = torch.LongTensor(adv).unsqueeze(1).float().to(device);

            td_target = adv + vs;
        optim_iter_num = int(math.ceil(data_size/self.batch_size));
        for _ in range(self.epochs):
            perm = np.arange(data_size);
            np.random.shuffle(perm);
            perm = torch.from_numpy(perm).to(device);
            s,a,td_target,adv,old_proba = s[perm].clone(),a[perm].clone(),td_target[perm].clone(),adv[perm].clone(),old_proba[perm].clone();

            for i in range(optim_iter_num):
                index = slice(i*self.batch_size,min((i+1)*self.batch_size,data_size));
                prob = self.actor.pi(s[index],action_dim=1);
                # print(prob)
                entropy = Categorical(prob).entropy().sum(dim = 0,keepdim = True);
                prob_a = prob.gather(1,a[index]);
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_proba[index]));
                # test_old_prob = old_proba[index];
                # print(ratio)
                # print(prob_a[ratio == float('inf')])
                # print(test_old_prob[ratio == float('inf')])
                surr1 = ratio * adv[index];
                surr2 = torch.clamp(ratio,1-self.clip_rate,1+self.clip_rate) * adv[index];

                a_loss = -torch.min(surr1,surr2) - self.entropy_coef*entropy;

                self.actor_optimizer.zero_grad();
                a_loss.mean().backward();
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),40);
                self.actor_optimizer.step();
                # for name,para in self.actor.named_parameters():
                #     print('{}:{}'.format(name,para));
                c_loss = torch.pow((self.critic(s[index])-td_target[index]),2).mean();
                for name,para in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += torch.pow(para,2).sum() * self.l2_reg;

                self.critic_optimizer.zero_grad();
                c_loss.backward();
                self.critic_optimizer.step();

        # for name,para in self.actor.named_parameters():
        #     if 'model.0.weight' in name:
        #         print(para[0]);

    def store_transition(self,s,a,r,s_next,oldprob_a,done,dw):
        s = np.array([s[2],s[8],s[12],s[16]]);
        s_next = np.array([s_next[2],s_next[8],s_next[12],s_next[16]]);
        self.s_holder[self.idx ] = s;
        self.a_holder[self.idx ] = a;
        self.r_holder[self.idx ] = r;
        self.next_s_holder[self.idx] = s_next;
        self.old_proba_holder[self.idx ] = oldprob_a;
        self.dones_holder[self.idx ] = done;
        self.dw_holder[self.idx ] = dw;
        if self.idx + 1 == self.data_size:
            self.idx_flag = 1;
        self.idx = (self.idx + 1)% self.data_size;

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./models/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./models/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./models/ppo_critic{}.pth".format(episode)))
        self.actor.load_state_dict(torch.load("./models/ppo_actor{}.pth".format(episode)))

















