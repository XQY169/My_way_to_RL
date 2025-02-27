if __name__ == '__main__':
    import sys;
    from IPython import get_ipython;

    sys.modules['pydev_umd'] = None;
    if get_ipython() is not None:
        get_ipython().run_line_magic('reset', '-f');

import gym
from policy import PG
import matplotlib.pyplot as plt
import numpy as np
from xvfbwrapper import Xvfb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from time import time
lr = 0.01;
gamma = 0.9;
switch = 0;
env = gym.make('CartPole-v1');
env = env.unwrapped;
state_number = env.observation_space.shape[0];
action_num = env.action_space.n;
model = PG(state_number,action_num,10,lr,gamma);
rewards = [];
if switch == 0:
    for i in range(1000):
        print(i)
        r = 0;
        total_reward = 0;
        observation = env.reset();
        observation = observation[0];
        while(True):
            action = model.act(observation);
            next_observation,reward,done,_,info = env.step(action);
            model.store_transition(observation,action,reward);
            total_reward = total_reward + reward;
            observation = next_observation;
            if(done):
                model.learn();
                break;
        rewards.append(total_reward);
with Xvfb():
    after_training = 'after_training.mp4';
    env = gym.make('CartPole-v1',render_mode = "rgb_array");
    video = VideoRecorder(env,after_training);
    for i in range(10):
        observation = env.reset();
        observation = observation[0];
        while(True):
            env.render()
            video.capture_frame()
            action = model.act(observation);
            next_observation,reward,done,_,info = env.step(action);
            if(done):
                break;
    video.close();
    env.close();
plt.scatter(np.arange(len(rewards)),rewards);
plt.show()
