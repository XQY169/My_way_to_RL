from typing import Optional
from ray import tune
import numpy as np
import ray
import time
import copy
from PPO import PPO_agent
from Envs import MyEnv


class MyTrainable(tune.Trainable):

    def setup(self, config):
        self.kwargs_agent = config['kwargs_agent']
        self.kwargs_env = config['kwargs_env']
        self.train_param = config['kwargs_train']
        self.model = PPO_agent(config['kwargs_tune'], self.kwargs_agent)
        num_envs = self.kwargs_env.get('num_envs', 1)
        self.envs = [MyEnv.remote(self.kwargs_env) for _ in range(num_envs)]

    def step(self):
        train_steps = 0
        rewards = []
        kwargs_remote = {
            'actor':copy.deepcopy(self.model.actor).cpu(),
            'phi_net':copy.deepcopy(self.model.abs_module.phi_net).cpu() if self.model.abs else None,
            'his_length':self.model.his_length,
            'state_size':self.model.state_size1,
            'action_size':self.model.action_size,
            'abs':self.model.abs
        }
        while train_steps < self.train_param['train_freq']:
            episode_datas = ray.get([
                env.get_data.remote(kwargs_remote,self.train_param['steps'])
                for env in self.envs
            ])
            states, actions, logprob_as, next_states, rewards_batch, dones, dws = self.process_episode_datas(
                episode_datas)
            reward = np.sum(rewards_batch) / len(episode_datas)
            rewards.append(reward)
            self.model.replay_buffer.add(states, actions, logprob_as, rewards_batch, next_states, dones, dws)
            train_steps += len(rewards_batch)

        for i in range(self.train_param['train_epoch']):
            self.model.train()
        return {'reward': np.mean(rewards)}

    def save(self, checkpoint_dir: Optional[str] = None):
        if checkpoint_dir:
            self.model.save(checkpoint_dir,0)
            return checkpoint_dir
        else:
            pass

    def load(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path:
            self.model.load(checkpoint_path,0)
        else:
            pass

    def process_episode_datas(self, episode_datas):
        states = np.concatenate([episode_data['states'] for episode_data in episode_datas], axis=0)
        next_states = np.concatenate([episode_data['next_states'] for episode_data in episode_datas], axis=0)
        actions = np.concatenate([episode_data['actions'] for episode_data in episode_datas], axis=0)
        rewards = np.concatenate([episode_data['rewards'] for episode_data in episode_datas], axis=0)
        logprob_as = np.concatenate([episode_data['logprob_as'] for episode_data in episode_datas], axis=0)
        dones = np.concatenate([episode_data['dones'] for episode_data in episode_datas], axis=0)
        dws = np.concatenate([episode_data['dws'] for episode_data in episode_datas], axis=0)
        return states, actions, logprob_as, next_states, rewards, dones, dws
