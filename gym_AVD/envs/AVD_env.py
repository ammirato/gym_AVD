import gym
from gym import error, spaces, utils
from gym.utils import seeding

class AVDEnv(gym.Env):
    metadata = {'no':True}

    def __init__(self):
        print 'Hello, world, AVD'  
    def _step(self, action):
        print 'Hello, world, AVD'  
    def _reset(self):
        print 'Hello, world, AVD'  
    def _render(self, mode='human', close=False):
        print 'Hello, world, AVD'  
