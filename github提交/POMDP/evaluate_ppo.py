from Training2 import Trainer_and_Recorder
from PPO2 import PPO_agent
import torch
import gym
import numpy as np

def evaluate_agent( test_env, agent,test_epoch, steps):
    for epoch in range(test_epoch):
        s = test_env.reset();
        s = s[0]
        # s = s[5:11]
        padding_s = np.zeros((agent.his_length, s.shape[0]))
        padding_s[-1,:] = s
        reward = 0;
        for i in range(steps):

            # a,_ = agent.act(padding_s, deterministic=False, with_logprob=True);
            a = agent.act(padding_s, deterministic=True, with_logprob=False)
            act = agent.process_action(a)
            s_, r, done, dw, info = test_env.step(act);
            # s_ = s_[5:11]
            padding_s_ = np.roll(padding_s, -1, axis=0)
            padding_s_[-1, :] = s_
            reward += np.mean(r);
            s = s_;
            padding_s = padding_s_
            if done:
                break
                # s = self.env.reset();
                # s = s[0]
        print('i,reward:', i,reward)
if __name__ == '__main__':
    dev = torch.device("cpu");
    env = gym.make("Hopper-v4", render_mode = 'human')
    kwargs_agent = {'state_size1':11,'state_size':11,'his_length':10,'action_size':3,'neu_size':150,'device':dev,'a_lr':0.0001,'c_lr':0.0001,'gamma':0.997,'lambd':0.95,
                'max_steps':2068,'K_epochs':10,'train_abs':5,'a_optim_batch_size':64,'c_optim_batch_size':64,'abs_train_size':128,'action_expand':1.0,'clip_rate':0.2,
                    'l2_reg':0.001,'num_envs':1,'entropy_coef':0.002,'abs':False,};#,'track_torch_grad':False
    model = PPO_agent(**kwargs_agent);
    model.load('./models/test_model4/',99999999)
    evaluate_agent(env,model,2,1000)

