from Training2 import Trainer_and_Recorder
from PPO import PPO_agent
import torch
import envpool
# import gym
if __name__ == '__main__':
    dev = torch.device("cpu");
    seed = 64
    # env = envpool.make('Pendulum-v1', env_type="gym", num_envs=6)
    env = envpool.make('Hopper-v4', env_type="gym", num_envs=4,seed=seed)
    kwargs_agent = {'state_size1':6,'state_size':6,'action_size':3,'neu_size':150,'device':dev,'a_lr':0.0001,'c_lr':0.0001,'gamma':0.997,'lambd':0.95,
                'max_steps':2068,'K_epochs':10,'train_abs':5,'a_optim_batch_size':64,'c_optim_batch_size':64,'abs_train_size':128,'action_expand':1.0,'clip_rate':0.2,
                    'l2_reg':0.001,'num_envs':len(env),'entropy_coef':0.002,'abs':False,};#,'track_torch_grad':False
    model = PPO_agent(**kwargs_agent);
    # model.load('./hopper-v4_abs/',3900)
    kwargs_train = {'max_epochs':10000,'init_epochs':0,'steps':1000,'save_epoch':300,'train_epoch':1,'train_freq':2048}#
    kwargs_TR = {'port':6006,'log_dir':'./runs/experiment9','save_model_path':'./models/test_models2/',
                 'save_training_data_path':'./Training_data/test_model2.csv','window_size':50,'tb_allow':True}
#当abs_batch为64和256时，很容易陷入局部收敛。
    TaR = Trainer_and_Recorder(model,env,kwargs_TR)
    TaR.train_agent(kwargs_train)
    TaR.draw_rewards()
