import numpy as np
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):

    def __init__(self,n_features,n_actions,n_neus):
        super(Net,self).__init__();
        layers = [
            nn.Linear(n_features,n_neus),
            nn.ReLU(),
            nn.Linear(n_neus,n_actions),
            nn.Softmax(dim=-1)
        ]

        for layer in layers:
            if(isinstance(layer,nn.Linear)):
                nn.init.normal(layer.weight.data,mean = 0,std = 0.1)

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x);

class PG():

    def __init__(self,n_features,n_actions,n_neus,Learning_rate,gamma):
        self.action_num = n_actions;
        self.gamma = gamma;
        self.policy = Net(n_features,n_actions,n_neus).to(device);
        self.rewards,self.obs,self.acts = [],[],[];
        self.renderflag = False;
        self.losses = [];
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr = Learning_rate);

    def act(self,state):
        state = torch.FloatTensor(state)
        probs= self.policy(state)
        # print(self.action_num)
        # print(probs.detach().numpy())
        # print('states:',state.detach().numpy())
        # print(self.policy.state_dict())
        # print('probs:',probs)
        action = np.random.choice(np.arange(self.action_num),p = probs.detach().numpy());


        return action;

    def store_transition(self,s,a,r):
        self.rewards.append(r);
        self.acts.append(a);
        self.obs.append(s);

    def learn(self):
        discounted_ep_r = np.zeros_like(self.rewards);
        running_add = 0;
        for t in reversed(range(0,len(self.rewards))):
            running_add = running_add*self.gamma + self.rewards[t];
            discounted_ep_r[t] = running_add;

        discounted_ep_r_norm = discounted_ep_r -  np.mean(discounted_ep_r);
        # print('discounted_ep_r_norm1:',discounted_ep_r_norm);
        discounted_ep_r_norm /= np.std(discounted_ep_r_norm);
        # print('discounted_ep_r_norm2:',discounted_ep_r_norm);

        self.optimizer.zero_grad();
        self.obs = np.array(self.obs);
        self.acts = np.array(self.acts);
        state_tensor = torch.FloatTensor(self.obs).to(device)
        reward_tensor = torch.FloatTensor(discounted_ep_r_norm).to(device)
        action_tensor = torch.LongTensor(self.acts)
        probs = self.policy(state_tensor).detach().numpy();
        log_prob = torch.log(self.policy(state_tensor));
        selected_log_probs = reward_tensor * log_prob[np.arange(len(self.acts)),self.acts];
        np.set_printoptions(threshold=np.inf)
        loss = -selected_log_probs.mean().to(device);
        # print('model:',self.policy.state_dict());
        self.losses.append(loss.detach().numpy())
        loss.backward();

        self.obs,self.acts,self.rewards = [],[],[];





