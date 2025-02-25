import gymnasium
from gymnasium.vector import AsyncVectorEnv
import ray
import torch
import envpool
import numpy as np
class MyEnv:
    def __init__(self,kwargs_env):
        self.env = envpool.make(kwargs_env['env_name'],env_type="gymnasium",num_envs = kwargs_env['num_envs'])
        self.action_expand = kwargs_env['action_expand']

    def reset(self):
        s = self.env.reset()
        s = s[0]
        return s
    def step(self, a,valid_env):
        a = self.process_action(a)
        s_, r, done, dw, info = self.env.step(a,valid_env)
        s_ = self.process_state(s_)
        return s_, r, done, dw, info

    def get_data(self,agent,max_steps):
        s = self.reset()
        valid_env = np.arange(len(self.env))
        datas = {'states':[[] for i in range(len(self.env))],'actions':[[] for i in range(len(self.env))],
                 'logprob_as':[[] for i in range(len(self.env))],'next_states':[[] for i in range(len(self.env))],
                 'rewards':[[] for i in range(len(self.env))],'dones':[[] for i in range(len(self.env))],'dws':[[] for i in range(len(self.env))]}
        s = s[:,5:11]
        padding_s = np.zeros((len(self.env),agent.his_length,agent.state_size1+agent.action_size))
        padding_s[:,-1,:] = np.concatenate((s,np.zeros((len(self.env),agent.action_size))),axis=1)
        for i in range(max_steps):
            a,logprob_a = agent.act(padding_s,deterministic=False,with_logprob=True)

            s_,r,done,dw,info = self.step(a,valid_env)
            s_ = s_[:,5:11]
            padding_s[:,-1,:] = np.concatenate((s,a),axis = 1)
            padding_s_ = np.roll(padding_s,-1,axis=1)
            padding_s_[:,-1,:] = np.concatenate((s_,np.zeros((len(valid_env),agent.action_size))),axis=1)

            for i in range(len(valid_env)):
                datas['states'][valid_env[i]].append(padding_s[i])
                datas['next_states'][valid_env[i]].append(padding_s_[i])
                datas['actions'][valid_env[i]].append(a[i])
                datas['logprob_as'][valid_env[i]].append(logprob_a[i])
                datas['rewards'][valid_env[i]].append(r[i])
                datas['dones'][valid_env[i]].append(done[i])
                datas['dws'][valid_env[i]].append(dw[i])

            padding_s = padding_s_
            s = s_
            if done.any():
                valid_env = np.delete(valid_env, np.where(done == True)[0])
                s = s_[np.where(done == False)[0]]
                padding_s = padding_s[np.where(done == False)[0]]
                if len(valid_env) == 0:
                    break
        return datas

    def process_state(self,s):
        return s

    def process_action(self,a):
        act = self.action_expand*2*(a - 0.5)
        return act