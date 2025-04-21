import os
import gym
import time
import numpy as np
from sawyerEnv import sawyerEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pybullet as p
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Monitor import Monitor

################################################# Define Variables ########################################################################
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "pPdAb25"
log_dir = "log"
vName = "Current96.0"
modelName = log_dir + "/" + vName
envName  = log_dir + "/"  + vName + ".pkl"
################################################# Testing and Evaluation #################################################################

env = sawyerEnv(renders=False, isDiscrete=False, maxSteps=1024, graspType = graspType, orientation = orientation)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize.load(envName, env)
env.training = False	# not continue training the model while testing
env.norm_reward = False # reward normalization is not needed at test time
# load model 
model = PPO.load(modelName,env=env) 

test = 100
for i in range(test):
	obs = env.reset()
	done = False
	rewards = float('-inf')
	while (not done):
		action, _states = model.predict(obs)
		obs, rewards, done, info= env.step(action)

sus = model.get_env().get_attr("successGrasp")
print("SUCCESS RATE IS: ", str((sus[0]/test)*100) + "%" )
env.close()

############### write to txt #######################################
fileName = "log/" + str((sus[0]/test)*100) + ".txt"

with open(fileName, 'w') as f:
    f.write(str((sus[0]/test)*100))













