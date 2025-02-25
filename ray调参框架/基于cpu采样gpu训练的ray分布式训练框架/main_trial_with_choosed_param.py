from MyTrainable import MyTrainable
import torch
import ray
from ray import tune
import logging

if __name__ == '__main__':
    dev = torch.device("cuda");

    seed = 64
    ray.init(logging_level=logging.ERROR)
    kwargs_env = {'env_name':'Hopper-v4','action_expand':1.0,'seed':seed,'num_envs':4}
    kwargs_agent = {'state_size1':3,'action_size':3,'neu_size':150,'device':dev,'a_lr':0.0001,'c_lr':0.0001,'gamma':0.997,'lambd':0.95,
                'max_steps':3048,'K_epochs':10,'a_optim_batch_size':64,'c_optim_batch_size':64,'clip_rate':0.2,
                    'l2_reg':0.001,'entropy_coef':0.002};#,'track_torch_grad':False
    kwargs_train = {'steps':1000,'train_epoch':1,'train_freq':2048}#
    kwargs_tunes = [#自定义trial参数
        {'state_size': 6, 'his_length': 20, 'abs': False},
        {'state_size':6,'his_length':15,'abs': False},
        {'state_size': 3, 'his_length': 50, 'abs': False},
        {'state_size': 6, 'his_length': 5, 'abs': False},
        {'train_abs':1 / 2,'abs_train_size':64,'state_size':120,'abs_lr':5,'abs_neu_size':50,'abs':True,'his_length':15},
        {'train_abs': 1 / 2, 'abs_train_size': 64, 'state_size': 120, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length': 15, 'scheduler':3000},
        {'train_abs': 1 / 2, 'abs_train_size': 64, 'state_size': 60, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length': 5},
        {'train_abs': 1 / 3, 'abs_train_size': 64, 'state_size': 120, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length':50},
        {'train_abs': 1 / 3, 'abs_train_size': 64, 'state_size': 120, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True, 'his_length': 15,'scheduler':2000},
        {'train_abs': 1 / 3, 'abs_train_size': 64, 'state_size': 60, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length': 5},
        {'train_abs': 1 / 4, 'abs_train_size': 64, 'state_size': 120, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length':15},
        {'train_abs': 1 / 4, 'abs_train_size': 64, 'state_size': 120, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length': 15,'scheduler':1500},
        {'train_abs': 1 / 4, 'abs_train_size': 64, 'state_size': 60, 'abs_lr': 5, 'abs_neu_size': 50, 'abs': True,'his_length': 5},
                   ]
    configs = [{
        'kwargs_tune':kwargs_tune,
        'kwargs_env':kwargs_env,
        'kwargs_agent':kwargs_agent,
        'kwargs_train':kwargs_train,
    } for kwargs_tune in kwargs_tunes]
    trials = [
        tune.Experiment(
            name = str(configs[i]['kwargs_tune']),
            run = MyTrainable,
            config = configs[i],
            resources_per_trial={'cpu':4,'gpu':2}
        )
        for i in range(len(configs))
    ]
    tune.run_experiments(
        trials,
        verbose=0,
    )
    ray.shutdown()
