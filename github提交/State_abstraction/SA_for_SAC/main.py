from Training import Trainer_and_Recorder
from SAC import SAC
import torch
import envpool
if __name__ == '__main__':
    dev = torch.device("cpu");
    # env_config = {
    #     "healthy_z_min": 0.3,
    #     "healthy_z_max": 1.0
    # }
    env = envpool.make('Hopper-v4', env_type="gym", num_envs=4)
    kwargs_agent = {'state_size1':11,'state_size':11,'action_size':3,'neu_size':128,'device':dev,'lr':0.0001,'gamma':0.997,
              'alpha':0.2,'batch_size':256,'abs_batch_size':128,'abs':False,'action_expand':1.0,'num_envs':len(env),
                    'adaptive_alpha':False,'abs_train':5};
    model = SAC(**kwargs_agent);
    # model.load('./model4/',3900)
    kwargs_train = {'max_epochs':10000,'init_epochs':0,'steps':1000,'save_epoch':300,'train_epoch':100,'train_steps':2048,}
    kwargs_TR = {'port':6006,'log_dir':'./runs/experiment1','save_model_path':'./models/model1/',
                 'save_training_data_path':'./Training_data/test_model.csv','window_size':50,'tb_allow':True}

    TaR = Trainer_and_Recorder(model,env,kwargs_TR)
    TaR.train_agent(kwargs_train)
    TaR.draw_rewards()
