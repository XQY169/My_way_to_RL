import torch
import gym
import matplotlib.pyplot as plt
import numpy as np
import uuid

from attention_SAC import SAC,moving_average,save_results
from prometheus_client import Gauge,start_http_server
import time
kwargs = {'state_size': 3, 'action_size': 1, 'neu_size': 512, 'device': 'cpu', 'lr': 0.0001, 'gamma': 0.99,
          'alpha': 0.2,
          'batch_size': 256};
student_model = SAC(**kwargs)
env = gym.make('Pendulum-v1', render_mode='rgb_array');
rewardes = []
run_id = uuid.uuid4().hex
print(run_id)
first_goal = np.array([1.0,0.0,0.0],dtype = np.float32)
i = 0;
her_k = 4
reward_gauge = Gauge('reward_gauge','reward gauge',labelnames=['run_id'])
start_http_server(8000);
while(i < 200):
    s = env.reset()
    s = s[0]
    # s = np.concatenate((s,first_goal))
    start = 0
    rewards = 0
    episode_cache = []
    while start < 200:
        start += 1;
        a = student_model.act(s, deterministic=False, with_logprob=False)

        s_, r ,done,dw,info = env.step(a)

        # r = calcu_reward(first_goal,s)

        # print(s_)
        rewards += r
        r = (r + 8.1)/8.1
        # if start == 50:
        #     done = True
        # if done or dw:
        #     break;
        # s_ = np.concatenate((s_,first_goal))
        episode_cache.append((s, a, r, s_, done))
        student_model.replay_buffer.add(s, a, r, s_, done)
        s = s_;
    reward_gauge.labels(run_id = run_id).set(rewards)
    for k in range(50):
        student_model.train()
    #
    # for m,experience in enumerate(episode_cache):
    #     if (m + 1) == len(episode_cache):
    #         break;
    #     new_goals = get_new_goals(m,episode_cache,her_k)
    #     s,a,r,s_,done = experience[0], experience[1], experience[2], experience[3], experience[4]
    #     for new_goal in new_goals:
    #         s[3:6] = new_goal
    #         s_[3:6] = new_goal
    #         r = calcu_reward(new_goal,s)
    #         student_model.replay_buffer.add(s, a, r, s_, done)
    # new_goals = get_new_goals(episode_cache, her_k)
    # for goal in new_goals:
    #     rewards2 = 0
    #     for (s,a,r,s_,done) in episode_cache:
    #         s[3:6] = goal
    #         s_[3:6] = goal
    #         r = calcu_reward(goal,s)
    #         rewards2 += r
    #         student_model.replay_buffer.add(s, a, r, s_, done)
    #     print('her_goal',goal,'rewards',rewards2)
    # student_model.train()



    i = i + 1;
    rewardes.append(rewards)
    if i %3 == 0:
        print(i,'reward:',rewards)


print(np.mean(rewardes))
# with open('saved_model_buffer.pkl','wb') as f:
#     pickle.dump(student_model.replay_buffer, f)
rewards2 = moving_average(rewardes,50)
plt.plot(np.arange(len(rewards2)), rewards2,marker='.', color='blue');
save_results(rewards2,rewardes,student_model,'test_student_model5',plt,**kwargs);
plt.show();
