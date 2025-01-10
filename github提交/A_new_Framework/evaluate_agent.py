from Training import Trainer_and_Recorder
from SAC import SAC
import torch
import gym
if __name__ == '__main__':
    dev = torch.device("cpu");
    env = gym.make("Hopper-v4", render_mode = 'human')
    kwargs_agent = {'state_size1':11,'state_size':11,'action_size':3,'neu_size':128,'device':dev,'lr':0.0001,'gamma':0.995,
              'alpha':0.2,'batch_size':256,'abs':False,'action_expand':1.0,'num_envs':1};
    model = SAC(**kwargs_agent);
    kwargs_TR = {'port':6006,'log_dir':'./runs/experiment1','save_model_path':'./model/',
                 'window_size':50,'tb_allow':True}
    TaR = Trainer_and_Recorder(model,env,kwargs_TR)
    TaR.evaluate_agent(env,test_epoch=2,steps = 1000)

