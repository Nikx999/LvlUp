import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np
import json

class LevelUpdateEnv(gym.Env):
    """A level update environment for OpenAI gym"""
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(LevelUpdateEnv, self).__init__()

        
        self.preward = 0
        self.usrlvl = 1.5
        self.act =0
        self.initial_level = 1.5

        # Actions of the format -.25 to .25
        self.action_space = spaces.Box(
            low = np.array([-0.005]), high = np.array([0.005]), shape = (1,), dtype=np.float32)

        #obs_space = <usr_level,question_level, time_taken, avg_time, correctness>
        self.observation_space = spaces.Box(
            low=np.array([1,1,0,0,-1]), high=np.array([5,5,600,600,1]), dtype=np.float32)
        self.reset()


    def step(self, action):
        #take action and update level
        def question_generator(usrlvl):
            queslvl =  np.random.normal(loc = usrlvl, scale = 0.5)
            if queslvl<1:
                queslvl = 1+np.random.random()
            elif queslvl>5:
                queslvl = 5-np.random.random()
            return queslvl

        def update_level(current_level,ques_level, correctness, max_time, time_taken, alpha= 0.6, beta = 0.2 ):
    
            level= current_level
            if level<1 or level >5:
                return -1000



    
            correct_marks =0
            time_marks = 0
            update=0
    
    
            if correctness==0:
                pass

            else:
        
            #Correctness marks
                if correctness==1:
                    val=ques_level
                    modval=val
                else:
                    val=ques_level-6
                    modval=-val
        
                x1=(1+correctness)/2
                x2=(correctness-1)/2
        
                correct_marks= (x1*((5-current_level)/5) *alpha*(val*(np.sqrt(np.exp((val**2)/37)-1)/5)))-(x2*((current_level)/5)*alpha*(val*(np.sqrt(np.exp((val**2)/37)-1)/5)))

                time_marks = (x1*((5-current_level)/5)*beta *(np.exp(-time_taken/(max_time*modval))))+(x2*((current_level)/5)*beta*(np.exp(-max_time/time_taken*modval)))

                update=correct_marks+ time_marks
                return update


        usr_level, ques_lvl, time, avg_time,correctness  = self.state
        self.usrlvl = usr_level
        
        
        reward = update_level(usr_level, ques_lvl, correctness, avg_time, time,0.6,0.2)
        rewardforuserlevel = np.log(np.sqrt(self.usrlvl)) if self.usrlvl >1 else 0
        self.preward =  rewardforuserlevel + 0.9*reward + 0.1*self.preward

        ques_level = question_generator(self.usrlvl)
        correctness = np.random.choice([-1,0,1])
        time = np.random.random()*600
        avg_time = np.random.random()*600
        self.next_ques(ques_level, correctness, time, avg_time)
        
        
        self.act = action

        obs = np.array([usr_level+action, ques_lvl, time, avg_time, correctness])
        done = False
        self.state = obs

        return obs, self.preward, done, {}




    def reset(self):
        
        self.state = np.array([self.initial_level, 1,0,0,0])
        return self.state

    def render(self, mode='console', close=False):

        # Render the environment to the screen
        if mode != 'console':
            raise NotImplementedError()
        print(f'Usr_lvl {self.usrlvl}')
        print(f'reward: {self.preward}')

    def next_ques(self, next_ques_level,next_correctness, next_time_taken, next_avg_time):
      obs = np.array([self.state[0], next_ques_level, next_time_taken, next_avg_time, next_correctness])
      self.state = obs

