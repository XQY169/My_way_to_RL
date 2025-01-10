import os
import sys
import shutil
import signal
import numpy as np
from utils import moving_average
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

class Trainer_and_Recorder():
    def __init__(self, agent,env,save_model_path = './model/',tensorboard_log_path = './runs/experiment1',port = 6006,tb_allow = True,window_size = 50):
        self.agent = agent
        self.env = env
        self.env_num = len(env)
        self.tensorboard_log_path = tensorboard_log_path
        self.save_model_path = save_model_path
        self.tensorboard_port = port
        self.tb_allow = tb_allow
        self.window_size = window_size
        signal.signal(signal.SIGTERM,self.handler_terminate)
        signal.signal(signal.SIGINT, self.handler_terminate)
        if tb_allow:
            if os.path.exists(tensorboard_log_path):
                shutil.rmtree(tensorboard_log_path)
            os.makedirs(tensorboard_log_path, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_log_path)
            self.tb = self.start_tensorboard()
        self.rewards = []

    def train_agent(self,max_epochs,steps,train_epoch,save_epoch):
        for epoch in range(max_epochs):
            s = self.env.reset();
            s = s[0]
            reward = 0;
            for i in range(steps):
                a = self.agent.act(s, deterministic=False, with_logprob=False);
                s_, r, done, dw, info = self.env.step(a);
                reward += np.mean(r);
                self.agent.replay_buffer.add(s, a, r, s_, done);
                s = s_;
                if done.any():
                    s = self.env.reset();
                    s = s[0]
            self.rewards.append(reward)
            for __ in range(train_epoch):
                self.agent.train()
            # self.agent.abs_module.train(self.agent.replay_buffer,1024)
            if (epoch+1)% save_epoch == 0:
                self.agent.save(self.save_model_path, epoch);

            if self.tb_allow:
                self.writer.add_scalar("reward", reward, epoch)
        if self.tb_allow:
            self.close_tensorboard()
    def start_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tensorboard_log_path, '--port', str(self.tensorboard_port),"--load_fast=false"])
        url = tb.launch()
        print(f"TensorBoard is running at: {url}")
        return tb

    def close_tensorboard(self):
        self.writer.close()
        print("TensorBoard closed.")

    def handler_terminate(self,signum, frame):
        self.draw_rewards()
        self.close_tensorboard()
        self.agent.save(self.save_model_path,99999999);
        sys.exit(0)

    def draw_rewards(self):
        rewards = moving_average(self.rewards,self.window_size)
        plt.figure(1)
        plt.plot(np.arange(len(rewards)), rewards,label = 'smooth', marker='.', color='blue');
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(len(self.rewards)),self.rewards,label = 'non_smooth',marker = '.',color = 'red')
        plt.legend()
        plt.show();