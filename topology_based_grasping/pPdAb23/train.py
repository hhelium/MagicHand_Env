import os
import gym
import time
import numpy as np
from datetime import datetime
from sawyerEnv import sawyerEnv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from typing import Callable
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Monitor import Monitor
import pandas as pd
from successRateCallBack import successRateCallBack
################################################# Define Variables ########################################################################
def linear_schedule(initial_value: float) -> Callable[[float], float]:
	"""
	Linear learning rate schedule.
	:param initial_value: Initial learning rate.
	:return: schedule that computes
	current learning rate depending on remaining progress
	"""
	def func(progress_remaining: float) -> float:
		"""
		Progress will decrease from 1 (beginning) to 0.
		:param progress_remaining:
		:return: current learning rate
		"""
		result = progress_remaining * initial_value
		if(progress_remaining >= 0.6):
			return initial_value * 1
		if(progress_remaining<0.6 and progress_remaining>=0.3):
			return initial_value * 0.9
		else:
			return initial_value * 0.8
	return func

n_envs = 1
#timeStep = 20480000*1000 #2048000 * 4 = 14:54:22
timeStep = 2048000 * 6 #30 = 4days
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "pPdAb23"
log_dir = "log"
fileName = log_dir + "/episodeData"
modelName = log_dir + "/" + graspType
envName = log_dir + "/" + graspType + ".pkl"

################################################# Training and Evaluation #################################################################
# create environment and custom callback
env = sawyerEnv(renders=False, isDiscrete=False, maxSteps=1024, graspType = graspType, orientation = orientation)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_reward=True) # when training norm_reward = True 
success = successRateCallBack(successRates = 0.99, verbose=1, check_freq = 250*100, path = log_dir, n_eval_episodes = 100)	
# load and train the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log = log_dir, learning_rate=linear_schedule(1.5e-5), gamma=0.96, gae_lambda=0.95, clip_range=0.2, batch_size=32)
model.learn(timeStep, callback=success)
# save training info
episodeData = model.get_env().get_attr("evaluation")
episodeData = np.array(episodeData, dtype=object)
np.save(fileName, episodeData)
# save model
model.save(modelName)
env.save(envName)
env.close()



