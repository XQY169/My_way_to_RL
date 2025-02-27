import numpy as np
import time
import sys
import torch
import dill
from multiprocessing import Process,Queue
from My_RL_env import My_RL_env
from qt_version_0 import Mywindow
from PyQt5.QtWidgets import QApplication

class Inter_env:

    def __init__(self,gui,user_debug,is_training=0,epoch = 0):
        self.gui = gui;
        self.user_debug = user_debug;
        self.qt_to_bullet = Queue();
        self.bullet_to_qt = Queue();
        self.is_training = is_training;
        self.epoch = epoch;
        self.control_mode = 0;
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.train_env = dill.dumps(My_RL_env(gui=False,user_debug_gui=False));

    def run_pybullet(self,model):

        def deal_qt_to_bullet(obs,done,mid_last_action,last_action):
            if not self.qt_to_bullet.empty():
                receive_data = self.qt_to_bullet.get();
                if (receive_data['type'] == 'button_info'):
                    self.epoch = receive_data['data'][0];
                    self.batch = receive_data['data'][1];
                elif (receive_data['type'] == 'open_model'):
                    file_path = receive_data['data'];
                    if 'actor' in file_path:
                        self.model.actor.load_state_dict(torch.load(file_path,map_location=self.device));
                    elif 'critic' in file_path:
                        self.model.critic.load_state_dict(torch.load(file_path,map_location=self.device));
                elif (receive_data['type'] == 'save_model'):
                    file_path = receive_data['data'];
                    file_path = file_path.replace('.pth','');
                    torch.save(self.model.actor.state_dict(),file_path+'_actor.pth')
                    torch.save(self.model.critic.state_dict(),file_path+'_critic.pth')
                elif (receive_data['type'] == 'key_info'):
                    mid_last_action = np.clip(last_action[env.follow_num] + receive_data['data'], -1, 1);
                elif (receive_data['type'] == 'comboBox_vision'):
                    env.vision_index = receive_data['data'];
                elif (receive_data['type'] == 'comboBox_id'):
                    env.follow_num = receive_data['data'];
                    mid_last_action = last_action[env.follow_num];
                elif (receive_data['type'] == 'comboBox_mode'):
                    self.control_mode = receive_data['data'];
                elif (receive_data['type'] == 'comboBox_train'):
                    next_is_training = receive_data['data'][0];
                    self.gui = 1 ^ receive_data['data'][1];
                    self.user_debug = 1 ^ receive_data['data'][2];
                    self.is_training = next_is_training;
                    env.close();
                    done = 1;
                elif (receive_data['type'] == 'reset'):
                    obs, _ = env.reset();
                    mid_last_action = np.array([0.0, 0.0, 0.0, 0.0]);
                    last_action = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (env.NUM_DRONES, 1));

            return obs,done,mid_last_action,last_action;

        def deal_control_mode(obs,mid_last_action,last_action):
            prob = 1;
            if self.control_mode == 0:
                if last_action[env.follow_num, 0] < mid_last_action[0]:
                    store_action = 2;
                elif last_action[env.follow_num, 0] == mid_last_action[0]:
                    store_action = 1;
                else:
                    store_action = 0;
                last_action[env.follow_num] = mid_last_action;
                action = last_action;
            elif self.control_mode == 1:
                prediction,prob = self.model.act(obs);
                store_action = prediction;
                # print('prob:',prob)
                prediction = (prediction - 1) / 100;
                last_action[env.follow_num] = np.clip(last_action[env.follow_num] + prediction, -1, 1);
                action = last_action;
            elif self.control_mode == 2:
                prediction,prob = self.model.act(obs);
                store_action = prediction;
                prediction = (prediction - 1) / 100;
                np.add(mid_last_action,prediction,out=mid_last_action);
                np.clip(mid_last_action,-1,1,out=mid_last_action);
                action = mid_last_action;
                # action = np.array([0.0, 0.0, 0.0, 0.0]);
            else:
                store_action = 0;
                action = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (env.NUM_DRONES, 1));
            return store_action,action,prob;

        self.model = dill.loads(model)
        while(True):
            if self.is_training == 0: #开启验证模式
                env = My_RL_env(gui=self.gui, user_debug_gui=self.user_debug);
                obs, _ = env.reset();
                done = False
                if self.gui:#在的过程添加人机交互
                    mid_last_action = np.array([0.0, 0.0, 0.0, 0.0]);
                    last_action = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (env.NUM_DRONES, 1));
                    self.bullet_to_qt.put(
                        {'type': 'drone_info', 'nums': env.NUM_DRONES, 'IDs': env.DRONE_IDS});
                    while not done:
                        obs,done,mid_last_action,last_action = deal_qt_to_bullet(obs,done,mid_last_action,last_action);
                        if done == 1:
                            break;
                        _,action,_ = deal_control_mode(obs,mid_last_action,last_action);
                        obs, reward, done, truncated, info = env.step(action)
                        self.bullet_to_qt.put(
                            {'type': 'state', 'obs': obs, 'reward': reward, 'done': done, 'truncated': truncated, 'info': info});
                        done = 0;
                        time.sleep(1 / 1000000)
                else:#如果模拟gui也没有的话，那应该输出一些文字信息
                    while not done:
                        obs,done,_,_ = deal_qt_to_bullet(obs,done,[],[]);
                        if done == 1:
                            break;
                        done = 0;
                        time.sleep(0.1)
            elif self.is_training == 1:
                env = My_RL_env(gui=self.gui, user_debug_gui=self.user_debug);
                if self.gui:#在的过程添加人机交互
                    for i in range(self.epoch):
                        obs,_ = env.reset();
                        done = False;
                        mid_last_action = np.array([0.0, 0.0, 0.0, 0.0]);
                        last_action = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (env.NUM_DRONES, 1));
                        self.bullet_to_qt.put(
                            {'type': 'drone_info', 'nums': env.NUM_DRONES, 'IDs': env.DRONE_IDS});
                        total_rewards = 0.0;
                        while not done:
                            obs,done,mid_last_action,last_action = deal_qt_to_bullet(obs,done,mid_last_action,last_action);
                            if done == 1:
                                break;
                            store_action,action,old_prob= deal_control_mode(obs,mid_last_action,last_action);
                            # print('store_action:',store_action)
                            next_obs, reward, done, truncated, info = env.step(action);
                            total_rewards += reward;
                            self.model.store_transition(obs,store_action,reward,next_obs,old_prob,done,truncated);
                            obs = next_obs;
                            self.bullet_to_qt.put(
                                {'type': 'state', 'obs': obs, 'reward': reward, 'done': done, 'truncated': truncated, 'info': info});
                            time.sleep(1 / 1000000)
                        self.bullet_to_qt.put(
                            {'type': 'rewards', 'epoch': i, 'total_rewards': total_rewards});
                        print('epoch',i,'rewards:',total_rewards)
                        if self.is_training == 0:
                            break;
                        else:
                            self.model.learn()

                else:#如果模拟gui也没有的话，那应该输出一些文字信息，并且是多无人机训练，不带界面训练
                    if not self.gui:
                        self.control_mode = 2;
                    self.rewards = [];
                    for i in range(self.epoch):
                        obs,_ = env.reset();
                        mid_last_action = np.array([0.0, 0.0, 0.0, 0.0]);
                        # last_action = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), (env.NUM_DRONES, 1));
                        done = False;
                        _,done,[],[] = deal_qt_to_bullet([],done,[],[]);

                        total_rewards = 0.0;
                        if done == 1:
                            break;
                        while not done:
                            store_action,action,old_prob = deal_control_mode(obs,mid_last_action,[]);
                            next_obs, reward, done, truncated, info = env.step(action)
                            total_rewards = total_rewards + reward;
                            self.model.store_transition(obs,store_action,reward,next_obs,old_prob,done,truncated)
                            obs = next_obs;
                            # print('The obs is :{} ,the reward is :{},the done is :{},the action is :{}'.format(obs, reward, done, action));
                        self.rewards.append(total_rewards);
                        self.bullet_to_qt.put(
                                {'type': 'rewards', 'epoch': i, 'total_rewards': total_rewards});
                        self.model.learn()
                if self.is_training == 1:
                    env.close()
                    # torch.save(model.policy.state_dict(),'model.pth')
                    self.is_training = 0;
                if self.user_debug == 0:
                    torch.save(self.model.actor.state_dict(),'model_test_actor.pth');
                    torch.save(self.model.critic.state_dict(),'model_test_critic.pth');
                    break;





    def run_pyqt(self):
        app = QApplication(sys.argv)
        widget = Mywindow(self.qt_to_bullet, self.bullet_to_qt)
        widget.show()
        sys.exit(app.exec_())

    def run_pybullet_gui(self,model):
        model = dill.dumps(model);
        bullet_process = Process(target=self.run_pybullet,args=(model,));
        bullet_process.start();
        if self.gui:
            pyqt_process = Process(target=self.run_pyqt);
            pyqt_process.start();
            pyqt_process.join();
        bullet_process.join();
