import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from SAC import SAC,moving_average,save_results

if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    kwargs = {'state_size':3,'action_size':1,'neu_size':512,'device':dev,'lr':0.0001,'gamma':0.99,'alpha':0.0,'batch_size':200};
    model = SAC(**kwargs);
    env = gym.make('Pendulum-v1', render_mode='rgb_array');
    max_epochs = 200;
    steps = 200;
    window_size = 50;

    rewardes = [];
    for _ in range(max_epochs):
        gamma = 0.95;
        s = env.reset();
        s = s[0];
        start = True;
        rewards = 0;
        while (start):
            for i in range(steps):
                a = model.act(s,deterministic=False,with_logprob=False);
                s_, r, done, dw, info = env.step(a);
                print(dw)
                rewards += r;
                r = (r + 8.1) / 8.1;
                model.replay_buffer.add(s, a, r, s_, dw);
                if done or dw:
                    start = False;
                    break;
                s = s_;
            for __ in range(50):
                model.train();
        if _ % 3 == 0:
            print(_, ' ', rewards);
        rewardes.append(rewards);

    rewards2 = moving_average(rewardes,window_size);
    plt.plot(np.arange(len(rewards2)), rewards2,marker='.', color='blue');
    save_results(rewards2,rewardes,model,'test_model_no_soft_50',plt,**kwargs);
    plt.show();
    plt.close();