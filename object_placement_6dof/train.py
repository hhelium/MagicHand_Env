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
from successRateCallBack import successRateCallBack
import torch as th
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
		return progress_remaining * initial_value
	return func


#timestep = 20480000*1000 #2048000 * 4 = 14:54:22
timeStep = 2048000 * 6 #30 = 4days
timeStep = 2048 * 6 #30 = 4days

orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "poPdAb2"
log_dir = "log"
fileName = log_dir + "/episodeData"
modelName = log_dir + "/" + graspType
envName = log_dir + "/" + graspType + ".pkl"
################################################# Training and Evaluation #################################################################

# create environment and custom callback
env = sawyerEnv(renders=True, isDiscrete=False, maxSteps=6144, graspType = graspType, orientation = orientation)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_reward=True) # when training norm_reward = True 
success = successRateCallBack(successRates = 0.99, verbose=1, check_freq = 2048*25, path = log_dir, n_eval_episodes = 50)	
# load and train the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log = log_dir, learning_rate=1.6e-6, 
			gamma=0.985, use_sde=True, sde_sample_freq=1, clip_range=0.2, batch_size=32)
model.learn(timeStep, callback=success)
# save training info
episodeData = model.get_env().get_attr("evaluation")
episodeData = np.array(episodeData, dtype=object)
np.save(fileName, episodeData)
# save model
model.save(modelName)
env.save(envName)
env.close()







