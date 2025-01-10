import torch
import gym

import numpy as np
from SAC import SAC,Actor, Critic,ReplayBuffer

env = gym.make('Pendulum-v1', render_mode='rgb_array');
seed1 = 42
seed2 = 500
# kwargs = {'state_size': 3, 'action_size': 1, 'neu_size': 512, 'device': 'cpu', 'lr': 0.0001, 'gamma': 0.99, 'alpha': 0.0,
#           'batch_size': 200};
# teacher_model = SAC(**kwargs)
teacher_model = Actor(3, 1, 512, mode=1)
teacher_model.load_state_dict(torch.load('test_model/actor.pth'))
student_model = Actor(3, 1, 256, mode=1)
student_optimizer = torch.optim.Adam(student_model.parameters(),lr = 0.0001)
buffer = ReplayBuffer(3, 1, 200000, 'cpu')

def act(model,state,deterministic,with_logprob):
    with torch.no_grad():
        state = torch.FloatTensor(state[np.newaxis, :]).to('cpu');
        a, _ = model(state, deterministic, with_logprob);
    return a.cpu().numpy()[0];

def create_data():
    for i in range(1000):
        s = env.reset(seed = seed1 +i )
        s = s[0]
        start = True
        while start:
            a = act(teacher_model,s,deterministic = False,with_logprob = False)
            s_,r,done,dw,info = env.step(a)
            buffer.add(s,a,r,s_,dw)
            if done or dw:
                start = False;
                break;
            s = s_;

def student_learn(batch_size):
    s,a,r,s_next,dw = buffer.sample(batch_size)
    u_log_probs = student_model.back_for_log_prob(s,a)
    loss = -u_log_probs.mean()
    student_optimizer.zero_grad()
    loss.backward()
    student_optimizer.step()

def test_student():
    rewards = []
    for i in range(100):
        s = env.reset(seed = seed2 +i )
        s = s[0]
        start = True
        reward = 0
        while start:
            a = act(student_model,s,deterministic = False,with_logprob = False)
            s_,r,done,dw,info = env.step(a)
            reward += r
            if done or dw:
                start = False;
                rewards.append(reward)
                break
            s = s_

    return rewards




if __name__ == '__main__':
    teacher_model.load_state_dict(torch.load("test_model/actor.pth"))
    create_data()
    for i in range(2000):
        student_learn(256)

    rewards = np.array(test_student())
    print(np.mean(rewards))



