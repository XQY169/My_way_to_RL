from multiprocessing import Process,Queue
import dill
import gym
from stable_baselines3 import PPO

def deal_process1(model):
    print(model);
def deal_process2():
    print('nihao')

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    model = PPO('MlpPolicy',env,verbose=1)
    with dill.disable_patch_mp():
        process1 = Process(target=deal_process1,args=(model,))
    process2 = Process(target=deal_process2)

    process1.start()
    process2.start()
    process1.join()
    process2.join()