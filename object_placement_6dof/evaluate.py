from sawyerEnv import sawyerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Monitor import Monitor

################################################# Define Variables ########################################################################
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "poPdAb2"
log_dir = "trained"
vName = "poPdAb2"
modelName = log_dir + "/" + vName
envName  = log_dir + "/"  + vName + ".pkl"
################################################# Testing and Evaluation #################################################################

env = sawyerEnv(renders=False, isDiscrete=False, maxSteps=6144, graspType = graspType, orientation = orientation)
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
	#done = False
	#rewards = -10
	while (True):
		action, _states = model.predict(obs)
		obs, rewards, done, info= env.step(action)

#sus = model.get_env().get_attr("successGrasp")
#print("SUCCESS RATE IS: ", str((sus[0]/test)*100) + "%" )
env.close()

############### write to txt #######################################
# fileName = "log/" + str((sus[0]/test)*100) + ".txt"

# with open(fileName, 'w') as f:
#     f.write(str((sus[0]/test)*100))












