if __name__ == '__main__':
    import sys;
    from IPython import get_ipython;

    sys.modules['pydev_umd'] = None;
    get_ipython().run_line_magic('reset', '-f');

import numpy as np

class Myenv():
    def __init__(self,cell_num,is_trea = 0):
        self.cell_num = cell_num;
        self.cur_state = np.zeros(cell_num,dtype=np.uint16);
        self.is_trea = is_trea;
        self.flag = False;
        # self.sums = np.cumsum(np.array(range(self.cell_num-1,-1,-1)));

    def init_state(self):
        self.cur_state[:] = 0;
        if self.is_trea == 0:
            hunter, hider = np.random.choice(self.cell_num, 2,replace = False);
        else:
            hunter,hider,trea = np.random.choice( self.cell_num, 3,replace = False);
        self.cur_state[hunter] = 1;
        self.cur_state[hider] = 2;
        if self.is_trea == 1:
            self.cur_state[trea] = 3;
        self.flag = hunter < hider;
        mode_num = self.return_mode_num();
        return self.cur_state,mode_num;

    # def return_mode_num(self):
    #     posi1 = np.where(self.cur_state == 1);
    #     posi1 = posi1[0][0];
    #     posi2 = np.where(self.cur_state == 2);
    #     posi2 = posi2[0][0];
    #     # print(posi1,posi2)
    #     if self.is_trea == 0:
    #         if posi1 == 0:
    #             mode_num = self.flag * int(self.cell_num*(self.cell_num - 1) / 2) + (posi2 - posi1) - 1;
    #         else:
    #             mode_num = self.flag * int(self.cell_num*(self.cell_num - 1) / 2) + self.sums[posi1 - 1] + (posi2 - posi1) - 1;
    #     else:
    #         posi3 = np.where(self.cur_state == 3);
    #         mode_num = 0;
    #     return mode_num;
    def return_mode_num(self):
        posi1 = np.where(self.cur_state == 1);
        posi1 = posi1[0][0];
        posi2 = np.where(self.cur_state == 2);
        posi2 = posi2[0][0];
        if(self.is_trea == 0):
            mode_num = posi1*(self.cell_num - 1)+(posi2- (posi2 > posi1));
        else:
            posi3 = np.where(self.cur_state == 3);
            if(len(posi3[0]) == 1):
                posi3 = posi3[0][0];
                mode_num = posi1*(self.cell_num - 1)*(self.cell_num-2) + (posi2- (posi2 > posi1))*(self.cell_num-2)+posi3-(posi3 > posi2)-(posi3 - posi1);
            else:
                mode_num = self.cell_num*(self.cell_num-1)*(self.cell_num-2) + posi1*(self.cell_num - 1) + (posi2- (posi2 > posi1));
        return mode_num;
    def step(self,action):
        if(self.is_trea == 0):#ret是是否结束的意思，reward是奖励
            ret,reward = self.__step_no_trea__(action);
        else:
            ret,reward = self.__step_is_trea__(action);
        mode_num = self.return_mode_num();
        return ret,reward,mode_num,self.cur_state;

    def __step_no_trea__(self,action):
        first_reward = 0;
        second_reward = 0;
        ret = 1;
        posi1 = np.where(self.cur_state == 1);
        posi1 = posi1[0][0];
        posi2 = np.where(self.cur_state == 2);
        posi2 = posi2[0][0];
        # print(posi1,posi2)
        if self.flag == True:
            if (action == 0):
                if (posi1 < posi2 - 1):
                    first_reward = 0.5;
                    self.cur_state[posi2] = 0;
                    self.cur_state[posi2 - 1] = 2;
                    posi2 = posi2 - 1;
                else:
                    first_reward = -1;
                    ret = -1;
                if ret == 1:
                    if posi1 + 1 < posi2:
                        second_reward = 0.5;
                        self.cur_state[posi1] = 0;
                        self.cur_state[posi1 + 1] = 1;

                    else:
                        second_reward = -1;
                        ret = -1;
            else:
                if (posi2 == self.cell_num - 1):
                    first_reward = 1;
                    ret = -1;
                else:
                    first_reward = 0.7;
                    self.cur_state[posi2] = 0;
                    self.cur_state[posi2 + 1] = 2;
                    self.cur_state[posi1] = 0;
                    self.cur_state[posi1 + 1] = 1;
        else:
            if (action == 1):
                if (posi1 > posi2 + 1):
                    first_reward = 0.5;
                    self.cur_state[posi2] = 0;
                    self.cur_state[posi2 + 1] = 2;
                    posi2 = posi2 + 1;

                else:
                    first_reward = -1;
                    ret = -1;
                if ret == 1:
                    if posi1 - 1 > posi2:
                        second_reward = 0.5;
                        self.cur_state[posi1] = 0;
                        self.cur_state[posi1 - 1] = 1;
                    else:
                        second_reward = -1;
                        ret = -1;
            else:
                if (posi2 == 0):
                    first_reward = 1;
                    ret = -1;
                else:
                    first_reward = 0.7;
                    self.cur_state[posi2] = 0;
                    self.cur_state[posi2 - 1] = 2;
                    self.cur_state[posi1] = 0;
                    self.cur_state[posi1 - 1] = 1;
        total_reward = first_reward + second_reward;
        return ret, total_reward;

    def __step_is_trea__(self,action):
        first_reward = 0;
        second_reward = 0;
        third_reward = 0;
        ret = 1;
        posi1 = np.where(self.cur_state == 1);
        posi1 = posi1[0][0];
        posi2 = np.where(self.cur_state == 2);
        posi2 = posi2[0][0];
        posi3 = np.where(self.cur_state == 3);
        if action == 0:
            if posi2 == 0:
                first_reward = 1;
                ret = -1;
            else:
                if(self.cur_state[posi2-1] == 0):
                    first_reward = 0.7;
                    self.cur_state[posi2-1] = 2;
                    self.cur_state[posi2] = 0;
                    posi2 = posi2 - 1;
                elif(self.cur_state[posi2-1] == 3):
                    third_reward = 2;
                    self.cur_state[posi2-1] = 2;
                    self.cur_state[posi2] = 0;
                    posi2 = posi2 - 1;
                elif(self.cur_state[posi2-1] == 1):
                    first_reward = -1;
                    ret = -1;
        else:
            if posi2 == self.cell_num-1:
                first_reward = 1;
                ret = -1;
            else:
                if(self.cur_state[posi2+1] == 0):
                    first_reward = 0.7;
                    self.cur_state[posi2+1] = 2;
                    self.cur_state[posi2] = 0;
                    posi2 = posi2 + 1;
                elif(self.cur_state[posi2+1] == 3):
                    third_reward = 2;
                    self.cur_state[posi2+1] = 2;
                    self.cur_state[posi2] = 0;
                    posi2 = posi2 + 1;
                elif(self.cur_state[posi2+1] == 1):
                    first_reward = -1;
                    ret = -1;
        flag = posi2 > posi1;
        if(flag == 1 and ret == 1):
            if (self.cur_state[posi1 + 1] == 0):
                second_reward = 0.5;
                self.cur_state[posi1 + 1] = 1;
                self.cur_state[posi1] = 0;
                posi1 = posi1 + 1;
            elif (self.cur_state[posi1 + 1] == 3):
                second_reward = 0;
                third_reward = 0;
                self.cur_state[posi1 + 1] = 1;
                self.cur_state[posi1] = 0;
                posi1 = posi1 + 1;
            elif (self.cur_state[posi1 + 1] == 2):
                second_reward = -1;
                ret = -1;
        elif(flag == 0 and ret == 1):
            if (self.cur_state[posi1 - 1] == 0):
                second_reward = 0.5;
                self.cur_state[posi1 - 1] = 1;
                self.cur_state[posi1] = 0;
                posi1 = posi1 - 1;
            elif (self.cur_state[posi1 - 1] == 3):
                second_reward = 0;
                third_reward = 0;
                self.cur_state[posi1 - 1] = 1;
                self.cur_state[posi1] = 0;
                posi1 = posi1 - 1;
            elif (self.cur_state[posi1 - 1] == 2):
                second_reward = -1;
                ret = -1;
        reward = first_reward + second_reward + third_reward;
        return ret,reward;


if __name__ == '__main__':
    env = Myenv(10);

    ret = 1;
    total_reward = 0;
    epoch = 10000;
    modes_num = [];
    for i in range(epoch):
        env.init_state();
        ret = 1;
        while(ret == 1):
            action = np.random.randint(0,2);
            print(env.cur_state,action)
            ret,reward,mode_num,cur_state = env.step(action)
            modes_num.append(mode_num)
