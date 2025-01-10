from Training2 import Trainer_and_Recorder
from PPO import PPO_agent
import torch
import gym
import numpy as np

def evaluate_agent( test_env, agent,test_epoch, steps):
    for epoch in range(test_epoch):
        s = test_env.reset();
        s = s[0]
        reward = 0;
        for i in range(steps):

            a = agent.act(s, deterministic=True, with_logprob=False);
            act = agent.process_action(a)
            s_, r, done, dw, info = test_env.step(act);

            reward += np.mean(r);
            s = s_;
            if done:
                break
                # s = self.env.reset();
                # s = s[0]
        print('i,reward:', i,reward)
if __name__ == '__main__':
    dev = torch.device("cpu");
    env = gym.make("Hopper-v4", render_mode = 'human')
    kwargs_agent = {'state_size1':11,'state_size':11,'action_size':3,'neu_size':150,'device':dev,'a_lr':0.0002,'c_lr':0.0002,'gamma':0.99,'lambd':0.95,
                'max_steps':3000,'K_epochs':10,'a_optim_batch_size':128,'c_optim_batch_size':128,'abs_train_size':128,'action_expand':1.0,'clip_rate':0.2,
                    'l2_reg':0.001,'num_envs':1,'entropy_coef':0.02,'abs':False,};#,'track_torch_grad':False
    model = PPO_agent(**kwargs_agent);
    model.load('./gym-hopper-v4-no_abs_no_time_64batch2/',600)
    evaluate_agent(env,model,2,1000)

