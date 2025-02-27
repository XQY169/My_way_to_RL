from interaction_drone import Inter_env
from model import PPO
import numpy as np
import torch
if __name__ == '__main__':
    gui = True
    user_debug = True
    env = Inter_env(gui=gui, user_debug=user_debug,is_training=0,epoch = 100)
    kwargs = {'state_dim':4,'neu_size':56,'action_dim':3,'data_size':3600,'entropy_coef_decay':0.9,'gamma':0.99,'lambd':0.85,'batch_size':512,'epochs':300,'clip_rate':0.2,'entropy_coef':0.08,'l2_reg':0.0001,'lr':0.0001}
    model = PPO(**kwargs);
    # model.policy.load_state_dict(torch.load('model.pth'))
    env.run_pybullet_gui(model)
################################################
