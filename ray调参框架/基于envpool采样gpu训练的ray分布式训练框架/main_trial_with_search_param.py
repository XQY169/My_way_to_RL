from MyTrainable import MyTrainable
import torch
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import logging

if __name__ == '__main__':
    dev = torch.device("cuda");
    seed = 64
    ray.init(logging_level=logging.ERROR)
    kwargs_env = {'env_name':'Hopper-v4','action_expand':1.0,'seed':seed,'num_envs':4}
    kwargs_agent = {'state_size1':6,'action_size':3,'neu_size':150,'device':dev,'a_lr':0.0001,'c_lr':0.0001,'gamma':0.997,'lambd':0.95,
                'max_steps':3048,'K_epochs':10,'a_optim_batch_size':64,'c_optim_batch_size':64,'clip_rate':0.2,
                    'l2_reg':0.001,'entropy_coef':0.002,'abs':True,'his_length':15};#,'track_torch_grad':False
    kwargs_train = {'steps':1000,'train_epoch':1,'train_freq':2048}#
    kwargs_tune = {
        'train_abs':tune.choice([1/3,1/2,1,2]),
        'abs_train_size':tune.choice([64,128,256]),
        'state_size':tune.choice([80,100,120,140]),
         'abs_lr':tune.uniform(1,10),
        'abs_neu_size':tune.choice([50,100,150,200]),
                   }
    config = {
        'kwargs_tune':kwargs_tune,
        'kwargs_env':kwargs_env,
        'kwargs_agent':kwargs_agent,
        'kwargs_train':kwargs_train,
    }
    # scheduler = HyperBandScheduler(
    #     metric='reward',
    #     mode='max',
    #     max_t = 150,
    #     time_attr = 'training_iteration'
    # )
    analysis = tune.run(
        MyTrainable,
        num_samples=2,
        config = config,
        metric='reward',
        mode='max',
        # scheduler=scheduler,
        verbose=2,
        resources_per_trial={"cpu":2,'gpu':4},

    )
    print(analysis.best_config,analysis.best_result)
    # TaR = Trainer_and_Recorder(model,envs,kwargs_TR)
    # TaR.train_agent()
    ray.shutdown()
