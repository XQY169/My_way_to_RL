import envpool
from Training import Trainer_and_Recorder
from SAC import SAC
import torch

if __name__ == '__main__':
    dev = torch.device("cpu");
    kwargs = {'state_size1':27,'state_size':35,'action_size':8,'neu_size':128,'device':dev,'lr':0.0001,'gamma':0.995,
              'alpha':0.2,'batch_size':256,'abs':True,'action_expand':1.0};
    model = SAC(**kwargs);
    log_dir = './runs/experiment1'
    port = 6006
    max_epochs = 1000;
    steps = 1000;
    train_epoch = 50
    save_epoch = 300
    window_size = 50;
    env = envpool.make("Ant-v4", env_type="gym", num_envs=4)
    TaD = Trainer_and_Recorder(model,env, tensorboard_log_path=log_dir,port = port)
    TaD.train_agent(max_epochs,steps,train_epoch,save_epoch=save_epoch)
