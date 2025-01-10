import os
import sys
import shutil
import signal
import torch
import csv
import numpy as np
from utils2 import moving_average
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from tensorboard import program

class Trainer_and_Recorder():
    def __init__(self, agent,env,kwargs_TR):
        self.agent = agent
        self.env = env
        self.tensorboard_log_path = kwargs_TR['log_dir']
        self.save_model_path = kwargs_TR['save_model_path']
        os.makedirs(self.save_model_path, exist_ok=True)
        self.tensorboard_port = kwargs_TR['port']
        self.save_training_data_path = kwargs_TR['save_training_data_path']
        self.tb_allow = kwargs_TR['tb_allow']
        self.window_size = kwargs_TR['window_size']
        signal.signal(signal.SIGTERM,self.handler_terminate)
        # sys.excepthook = self.handler_various_exception
        # torch.autograd.set_detect_anomaly(kwargs_TR['track_torch_grad'])
        if self.tb_allow:
            if os.path.exists(self.tensorboard_log_path):
                shutil.rmtree(self.tensorboard_log_path)
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_log_path)
            self.tb = self.start_tensorboard()
        self.rewards = []

    def train_agent(self,train_param):
        train_steps = 0
        for epoch in range(train_param['max_epochs']):
            s = self.env.reset();
            s = s[0]
            valid_env = np.arange(len(self.env))
            # noise = np.random.randn(len(self.env),10)*10
            # # time_steps = np.zeros((len(valid_env),1))
            # s = np.concatenate((s,noise),axis = 1)
            # s = s[:,5:11]
            padding_s = np.zeros((len(self.env),self.agent.his_length,s.shape[1]))
            padding_s[:,-1,:] = s
            reward = 0;
            for i in range(train_param['steps']):
                if epoch < train_param['init_epochs']:
                    a = []
                    for j in range(len(valid_env)):
                        mid_a = self.env.action_space.sample()
                        a.append(mid_a)
                    a = np.array(a)
                else:
                    a, logprob_a = self.agent.act(padding_s, deterministic=False, with_logprob=True);
                    act = self.agent.process_action(a)

                s_, r, done, dw, info = self.env.step(act,valid_env);
                # s_ = s_[:,5:11]
                padding_s_ = np.roll(padding_s,-1,axis=1)
                padding_s_[:,-1,:] = s_
                # noise = np.random.randn(len(valid_env), 10) * 10
                #
                # # time_steps = np.zeros((len(valid_env),1))+(i+1)/train_param['steps']
                # s_ = np.concatenate((s_,noise),axis = 1)
                reward += np.sum(r)/len(self.env);
                # r = (r + 8.1)/8.1

                # self.agent.replay_buffer.add(s, a, r, s_, done);
                self.agent.replay_buffer.add(padding_s,a,logprob_a,r,padding_s_,done,dw,valid_env)
                s = s_;
                padding_s = padding_s_
                train_steps += len(valid_env)


                if done.any():
                    valid_env = np.delete(valid_env,np.where(done == True)[0])
                    s = s_[np.where(done == False)[0]]
                    padding_s = padding_s[np.where(done == False)[0]]
                    if len(valid_env) == 0:
                        break

            # self.agent.replay_buffer.merge_cache()
            if train_steps >= train_param['train_freq']:
                train_steps = 0
                for __ in range(train_param['train_epoch']):
                    self.agent.train()
            self.rewards.append(reward)
            if (epoch+1)% train_param['save_epoch'] == 0:
                self.agent.save(self.save_model_path, (epoch+1));

            if self.tb_allow:
                self.writer.add_scalar("reward", reward, epoch)
                self.writer.add_scalar("steps",i,epoch)
        self.close_tensorboard()

    def evaluate_agent(self,test_env,test_epoch,steps):
        for epoch in range(test_epoch):
            s = test_env.reset();
            s = s[0]
            reward = 0;
            for i in range(steps):

                a = self.agent.act(s, deterministic=True, with_logprob=False);
                act = self.agent.process_action(a)
                s_, r, done, dw, info = test_env.step(act);
                print(i)
                reward += np.mean(r);
                s = s_;
                if done:
                    break
                    # s = self.env.reset();
                    # s = s[0]
            print('reward:',reward)
    def start_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tensorboard_log_path, '--port', str(self.tensorboard_port),"--load_fast=false"])
        url = tb.launch()
        print(f"TensorBoard is running at: {url}")
        return tb

    def close_tensorboard(self):
        if self.tb_allow:
            self.save_training_data()
            self.writer.close()
            print("TensorBoard closed.")

    def save_training_data(self):
        ea = event_accumulator.EventAccumulator(self.tensorboard_log_path)
        ea.Reload()
        scalar_tags = ea.Tags()["scalars"]
        data = {tag: ea.Scalars(tag) for tag in scalar_tags}
        with open(self.save_training_data_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Tag", "Step", "Value","Wall time"])
            for tag, events in data.items():
                for event in events:
                    writer.writerow([tag, event.step, event.value,event.wall_time])

    def handler_terminate(self,signum, frame):
        self.close_tensorboard()
        self.draw_rewards()
        self.agent.save(self.save_model_path,99999999);
        sys.exit(0)

    # def handler_various_exception(self,exc_type,exc_value,exc_traceback):
    #     print('出错啦')
    def draw_rewards(self):
        rewards = moving_average(self.rewards,self.window_size)
        plt.figure(1)
        plt.plot(np.arange(len(rewards)), rewards,label = 'smooth', marker='.', color='blue');
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(len(self.rewards)),self.rewards,label = 'non_smooth',marker = '.',color = 'red')
        plt.legend()
        plt.show();