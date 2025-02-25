from typing import Optional

from ray import tune
import numpy as np
import envpool
from PPO import PPO_agent
from Envs import MyEnv


class MyTrainable(tune.Trainable):

    def setup(self, config):
        self.kwargs_agent = config['kwargs_agent']
        self.kwargs_env = config['kwargs_env']
        self.train_param = config['kwargs_train']
        self.model = PPO_agent(config['kwargs_tune'], self.kwargs_agent)
        self.env = MyEnv(self.kwargs_env)

    def step(self):
        train_steps = 0
        rewards = []
        while train_steps < self.train_param['train_freq']:

            episode_datas = self.env.get_data(self.model, self.train_param['steps'])
            states, actions, logprob_as, next_states, rewards_batch, dones, dws = self.process_episode_datas(
                episode_datas)
            reward = np.sum(rewards_batch) / self.kwargs_env['num_envs']
            rewards.append(reward)
            self.model.replay_buffer.add(states, actions, logprob_as, rewards_batch, next_states, dones, dws)
            train_steps += len(rewards_batch)

        for i in range(self.train_param['train_epoch']):
            self.model.train()
        return {'reward': np.mean(rewards)}

    def save(self, checkpoint_dir: Optional[str] = None):
        # # 保存模型时，根据环境自动映射到当前可用设备

        self.model.save(checkpoint_dir,0)
        return checkpoint_dir

    def load(self, checkpoint_path: Optional[str] = None):
        self.model.load(checkpoint_path,0)

    def process_episode_datas(self, episode_datas):
        states = np.concatenate(episode_datas['states'], axis=0)
        next_states = np.concatenate(episode_datas['next_states'], axis=0)
        actions = np.concatenate(episode_datas['actions'], axis=0)
        rewards = np.concatenate(episode_datas['rewards'] , axis=0)
        logprob_as = np.concatenate(episode_datas['logprob_as'] , axis=0)
        dones = np.concatenate(episode_datas['dones'] , axis=0)
        dws = np.concatenate(episode_datas['dws'], axis=0)
        return states, actions, logprob_as, next_states, rewards, dones, dws
