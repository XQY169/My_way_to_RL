import gymnasium
import ray
import torch
import numpy as np

@ray.remote
class MyEnv:
    def __init__(self,kwargs_env):
        self.env = gymnasium.make(kwargs_env['env_name'])
        self.seed = kwargs_env['seed']
        self.action_expand = kwargs_env['action_expand']

    def reset(self):
        s = self.env.reset(seed = self.seed)
        s = s[0]
        # processed_s = self.process_state(s)
        return s
    def step(self, a):
        a = self.process_action(a)
        s_, r, done, dw, info = self.env.step(a)
        s_ = self.process_state(s_)
        return s_, r, done, dw, info
    def get_act(self, kwargs,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0);
            if kwargs['abs'] == True:
                state = kwargs['phi_net'](state).squeeze(0)
            a, logprob_a = kwargs['actor'].get_act(state, deterministic = False, with_logprob = True)
            return a.cpu().numpy(), logprob_a.detach().cpu().numpy()
    def get_data(self,kwargs,max_steps):
        s = self.reset()
        datas = {'states':[],'actions':[],'logprob_as':[],'next_states':[],'rewards':[],'dones':[],'dws':[]}
        s = s[8:11]
        padding_s = np.zeros((kwargs['his_length'],kwargs['state_size']+kwargs['action_size']))
        padding_s[-1,:] = np.concatenate((s,np.zeros(kwargs['action_size'])))
        for i in range(max_steps):
            a,logprob_a = self.get_act(kwargs,padding_s)

            s_,r,done,dw,info = self.step(a)
            s_ = s_[8:11]
            padding_s[-1,:] = np.concatenate((s,a))
            padding_s_ = np.roll(padding_s,-1,axis=0)
            padding_s_[-1,:] = np.concatenate((s_,np.zeros(kwargs['action_size'])))
            datas['states'].append(padding_s)
            datas['next_states'].append(padding_s_)
            datas['actions'].append(a)
            datas['logprob_as'].append(logprob_a)
            datas['rewards'].append(r)
            datas['dones'].append(done)
            datas['dws'].append(dw)
            padding_s = padding_s_
            s = s_
            if done:
                break
        return datas

    def process_state(self,s):
        return s

    def process_action(self,a):
        act = self.action_expand*2*(a - 0.5)
        return act