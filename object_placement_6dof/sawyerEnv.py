import os
currentdir = os.getcwd()
parentdir = os.path.abspath(os.path.join(currentdir, os.pardir))
startdir = os.path.abspath(os.path.join(parentdir, os.pardir))
traydir = os.path.join(parentdir, "robot/table/table.urdf")
objectdir = os.path.join(parentdir, "3dmodels/object_align_meshes/")
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from sawyer import sawyer
import time
import pybullet as p
import random
import pybullet_data
import pandas as pd
from random import seed
from random import randint
import statistics
import time
largeValObservation = 1
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
from math import *
from stable_baselines3.common.running_mean_std import RunningMeanStd
#from pyb_utils.ghost import GhostObject, GhostSphere
class sawyerEnv(gym.Env):
	
	#TODO: 1. separate stages for rotation and approaching
	#      2. change grasp topology and grasp orientation
	metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
	def __init__(self, urdfRoot=pybullet_data.getDataPath(), 
		actionRepeat=1, 
		isEnableSelfCollision=True, 
		renders=False, 
		isDiscrete=False, 
		maxSteps=6000,
		graspType = "poPmAd35",
		orientation = 0,
		normOb= True, normReward=True, training = True, gamma=0.99):
		self.r = []
		self._isDiscrete = isDiscrete
		self._timeStep = 1. / 240.
		self._urdfRoot = urdfRoot
		self._actionRepeat = actionRepeat
		self._observation = []
		self._renders = renders
		self._maxSteps = maxSteps
		self._sawyerId = -1
		self.graspType = graspType
		self.orientation = orientation
		self.arm2hand = 0
		self._p = p
		self.num_envs = 1
		if self._renders:
			cid = p.connect(p.SHARED_MEMORY)
			if (cid < 0):
				cid = p.connect(p.GUI)
			#p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
			p.resetDebugVisualizerCamera(0.9, 135, -30, [0.99,0,0.15])
		else:
			p.connect(p.DIRECT)
		self.handPoint = 52   # "palm point" for each grasp type 
		lowerObservation = [-5]*6 + [-5]* 34
		upperObservation = [5]*6 + [5]* 34
		self.observation_space = spaces.Box(low=np.array(lowerObservation), high=np.array(upperObservation), dtype=np.float32)
		action_dim = 11
		self._action_bound = 1
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = spaces.Box(-action_high, action_high)
		self.viewer = None
		self.successTask = 0 # number of successfull grasp 
		self.evaluation = []
		self.height = -0.04
		self.seed()
		self.reset()

	def reset(self):
		self.lamda = 0
		self.disScaler = 6
		self.s1_reward = 0
		self.s1_reward_index = 0
		self.s2_reward = 0
		self.s2_reward_index = 0
		self.s3_reward = 0
		self.s3_reward_index = 0
		self.s4_reward = 0
		self.s4_reward_index = 0
		self.terminated = 0 # 1, grasp object 2, terminate
		self.terminated_task = 0 # 1 whole task terminate
		p.resetSimulation()
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.setTimeStep(self._timeStep)
		#p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
		p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
		p.setGravity(0, 0, -10)
		self._sawyer = sawyer(timeStep=self._timeStep, graspType = self.graspType, orientation = self.orientation, handPoint = self.handPoint)
		self.trayUid = p.loadURDF(traydir, [1.32, 0.000000, -0.3], p.getQuaternionFromEuler([(math.pi/2), 0, (math.pi/2)]), useFixedBase = 1, flags = 8)
		# load objects
		self.index = self.r2r()
		#self.index = 0
		self.objectFeature = self.loadObject(self.graspType, self.index)
		self.object_position = [0.99, 0, -0.11]
		orn = p.getQuaternionFromEuler([self.objectFeature[5] * math.pi, self.objectFeature[6]* math.pi, self.objectFeature[7] * math.pi])
		#self.objectId = p.loadURDF(self.objectFeature[1], [self.objectFeature[9], self.objectFeature[10], self.objectFeature[11]], orn)
		self.objectId = p.loadURDF(self.objectFeature[1], self.object_position, orn)

		# load goal position and orientation
		x_target  = random.choice(range(97, 100, 1))/100
		y_target  = random.choice(range(11, 16, 1))/100
		z_target  = random.choice(range(5, 8, 1))/100

		self.x_orn  = random.choice(range(-15, 16, 1))/100 # aciton rotate 0 pi or 0.5 pi around x
		self.y_orn  = random.choice(range(-15, 16, 1))/100 # aciton rotate 0 pi or 0.3 pi around y
		self.z_orn  = random.choice(range(-15, 16, 1))/100 # aciton rotate 0 pi or 0.3 pi around z

		self.position_target = [x_target, y_target, z_target]
		self.orn_target = p.getQuaternionFromEuler([self.x_orn * math.pi, self.y_orn* math.pi, self.z_orn * math.pi])
		self._envStepCounter = 0
		p.stepSimulation()
		self._observation = self.getExtendedObservation()
		self.episodeR = []
		self._graspSuccess = 0
		self._taskSuccess = 0
		self.show = 0
		self.contactPoint = 0
		self.stage = 0
		self.disError = 0.005 
		self.ornError = 3.14/(31.4)
		self.cid = -1
		if self.index == 0:
			scale = [0.0055, 0.005, 0.0055]

		elif self.index == 1:
			scale = [0.011, 0.011, 0.011]

		elif self.index == 2:
			scale = [0.009, 0.011, 0.011]
		else:
				
			scale = [0.01, 0.01, 0.01]

		visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, 
						fileName=self.objectFeature[-1], 
						rgbaColor=[0.56, 0.93, 0.56, 0.6],
						meshScale= scale)
		p.createMultiBody(0, baseVisualShapeIndex=visualShapeId, basePosition=self.position_target, baseOrientation = self.orn_target)
		return np.array(self._observation)

	def r2r(self):
		# random 0 - 3
		seed(round(time.time()))
		return randint(0, 3)

	def loadObject(self, graspName, index):
		i = index
		csvName = graspName + "_list.csv"		
		data = pd.read_csv(csvName)
		ob = data.iloc[i]['Object']
		l = data.iloc[i]['A'] * 0.01
		h = data.iloc[i]['B'] * 0.01
		w = data.iloc[i]['C'] * 0.01
		r = data.iloc[i]['Roll']
		p = data.iloc[i]['Pitch'] 
		y = data.iloc[i]['Yaw']
		shape = data.iloc[i]['Shape']
		objectPath = objectdir + ob + "/" + ob + ".urdf"
		objectPath_obj = objectdir + ob + "/" + "tinker.obj"
		return [ob, objectPath, l, h, w, r, p, y, shape, objectPath_obj]#, c_x, c_y, c_z]	

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def __del__(self):
		p.disconnect()

	def getExtendedObservation(self):
		# all norm force
		palmForce = [] 
		thumbForce = []
		indexForce = []
		middleForce = []
		ringForce = []
		pinkyForce = []
		# all distance while norm force not zero	
		palmDist = []  
		thumbDist = []
		indexDist = []
		middleDist = []
		ringDist = []
		pinkyDist = []
		# define each part of the hand
		palmLinks = [19, 20, 25, 29, 34, 38, 43, 47, 52, 56, 57]
		thumbLinks = [58, 59, 60, 61, 62, 63, 64]
		indexLinks = [48, 49, 50, 51, 53, 54, 55]
		middleLinks = [39, 40, 41, 42, 44, 45, 46]
		ringLinks = [30, 31, 32, 33, 35, 36, 37]
		pinkyLinks = [21, 22, 23, 24, 26, 27, 28]
		# find contact point
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)
		# fill force and dist
		for i in range(nums):
			if(contact[i][3] in palmLinks):
				palmForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					palmDist.append(contact[i][8]) # contact distance
			
			if(contact[i][3] in thumbLinks):
				thumbForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					thumbDist.append(contact[i][8]) # contact distance
			
			if(contact[i][3] in indexLinks):	
				indexForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					indexDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in middleLinks):	
				middleForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					middleDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in ringLinks):
				ringForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					ringDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in pinkyLinks):
				pinkyForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					pinkyDist.append(contact[i][8]) # contact distance
		upperLimit = 1
		if(len(palmDist) != 0):
			palmd = min(palmDist)
		else:
			palmd = upperLimit
		if(len(thumbDist) != 0):
			td = min(thumbDist)
		else:
			td = upperLimit
		if(len(indexDist) != 0):
			ind = min(indexDist)
		else:
			ind = upperLimit
		if(len(middleDist) != 0):
			md = min(middleDist)
		else:
			md = upperLimit
		if(len(ringDist) != 0):
			rd = min(ringDist)
		else:
			rd = upperLimit
		if(len(pinkyDist) != 0):
			pind = min(pinkyDist)
		else:
			pind = upperLimit		
		dist = [palmd, td, ind, md, rd, pind]

		norm = [abs(sum(palmForce)), abs(sum(thumbForce)), abs(sum(indexForce)), abs(sum(middleForce)),  abs(sum(ringForce)), abs(sum(pinkyForce))] 
		handState = p.getLinkState(self._sawyer.sawyerId, self.handPoint)
		handPos = handState[0]
		handOrn = handState[1]
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		#thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)
		#indexTip = p.getLinkState(self._sawyer.sawyerId, 51)	
		#midTip = p.getLinkState(self._sawyer.sawyerId, 42)	
		#ringTip = p.getLinkState(self._sawyer.sawyerId, 33)	
		#pinkyTip = p.getLinkState(self._sawyer.sawyerId, 24)
	
		obHand = self.relativePos(handPos, handOrn, obPos, obOrn)
		#obThumb = self.relativePos(thumbTip[0], thumbTip[1], obPos, obOrn)
		#obIndex = self.relativePos(indexTip[0], indexTip[1], obPos, obOrn)
		#obMid = self.relativePos(midTip[0], midTip[1], obPos, obOrn)
		#obRing = self.relativePos(ringTip[0], ringTip[1], obPos, obOrn)
		#obPinky = self.relativePos(pinkyTip[0], pinkyTip[1], obPos, obOrn)	
		# x_orn * math.pi, y_orn* math.pi, z_orn * math.pi
		#obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target 
		obOrn_target = self.orn_target
		# x_orn * math.pi, y_orn* math.pi, z_orn * math.pi
		obOrn_eluer = list(p.getEulerFromQuaternion(obOrn))
		obOrn_eluer_target = list(p.getEulerFromQuaternion(obOrn_target))
		ornDiff_x = obOrn_eluer[0] - obOrn_eluer_target[0]
		ornDiff_y = obOrn_eluer[1] - obOrn_eluer_target[1]
		ornDiff_z = obOrn_eluer[2] - obOrn_eluer_target[2] 
		obe2targetDist = self.distant(obPos, obPos_target)
		ornDiff = [ornDiff_x, ornDiff_y, ornDiff_z]		
		# dis between hand and target 
		hand2targetDist = self.distant(handPos, obPos_target)
		# relative position and orientation between hand and target 
		HandRtarget = self.relativePos(handPos, handOrn, obPos_target, obOrn_target)
		#print(HandRtarget)
		# relative position and orientation between object and target 
		obRtarget = self.relativePos(obPos, obOrn, obPos_target, obOrn_target)
		#print(ornDiff)
		#print(obe2targetDis)
		#self._observation = norm + dist + obHand + obThumb + obIndex + [self.objectFeature[8]] + [p.getClosestPoints(self._sawyer.sawyerId, self.objectId, 500, self.handPoint, -1)[0][8]] + [self.objectFeature[2], self.objectFeature[3], self.objectFeature[4]] + [obPos[2]] + ornDiff + [obe2targetDist]

		self._observation = norm + dist + [self.objectFeature[8]] + [p.getClosestPoints(self._sawyer.sawyerId, self.objectId, 500, self.handPoint, -1)[0][8]] + [self.objectFeature[2], self.objectFeature[3], self.objectFeature[4]] + [obPos[2]] + [obe2targetDist] + obRtarget + ornDiff + obHand + HandRtarget 
		#print(len(self._observation))
		#print("hand: ", handPos)
		#print("obps: ", obPos)
		return np.array(self._observation)

	def relativePos(self, handPos, handOrn, obPos, obOrn):
		invhandPos, invhandOrn = p.invertTransform(handPos, handOrn)
		handEul = p.getEulerFromQuaternion(handOrn)    
		obPosInHand, obOrnInHand = p.multiplyTransforms(invhandPos, invhandOrn, obPos, obOrn)
		projectedObPos2D = [obPosInHand[0], obPosInHand[1]]
		obEulerInHand = p.getEulerFromQuaternion(obOrnInHand)
		obInHandPosXYEulZ = [obPosInHand[0], obPosInHand[1], obPosInHand[2], obEulerInHand[0], obEulerInHand[1], obEulerInHand[2]]
		return obInHandPosXYEulZ

	# find which parts have contact with the object 
	# if a part provides norm force, we count it as 
        # a contact part. 1 for contact, 0 for no contact 
	def getContactPart(self):		
		contactParts = [0, 0, 0, 0, 0, 0] # palm, thumb, index, middle, ring, pink		
		# define each part of the hand
		palmLinks = [19, 20, 25, 29, 34, 38, 43, 47, 52, 56, 57]
		thumbLinks = [58, 59, 60, 61, 62, 63, 64]
		indexLinks = [48, 49, 50, 51, 53, 54, 55]
		middleLinks = [39, 40, 41, 42, 44, 45, 46]
		ringLinks = [30, 31, 32, 33, 35, 36, 37]
		pinkyLinks = [21, 22, 23, 24, 26, 27, 28]
		arm2handLinks = [19, 20, 21, 25, 26, 29, 30, 34, 35, 38, 39, 43, 44, 47, 48, 52, 53, 56, 57]

		# find contact point
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)
		# fill force and dist
		limitForce = 5
		for i in range(nums):
			if(contact[i][3] in palmLinks):
				if(contact[i][9] >= limitForce):
					contactParts[0] = 1
			
			if(contact[i][3] in thumbLinks):
				if(contact[i][9] >= limitForce):
					contactParts[1] = 1
					#print("thumb", contact[i][9] )
			
			if(contact[i][3] in indexLinks):
				if(contact[i][9] >= limitForce):
					contactParts[2] = 1
					#print("index", contact[i][9] )
			if(contact[i][3] in middleLinks):
				if(contact[i][9] >= limitForce):
					contactParts[3] = 1

			if(contact[i][3] in ringLinks):
				if(contact[i][9] >= limitForce):
					contactParts[4] = 1

			if(contact[i][3] in pinkyLinks):
				if(contact[i][9] >= limitForce):
					contactParts[5] = 1
			if(contact[i][3] in arm2handLinks):
				self.arm2hand = 1
		return contactParts

	def step(self, action):
	
		d1 = 0.01 # stage 1 scaler
		d2 = 0.0 # stage 2 scaler 
		d3 = 1.5 # finger move scaler
		d4 = 0.01 # orientation scaler
		d5 = 0 # stage 3 finger scaler
		orn = p.getQuaternionFromEuler([math.pi*(-0.5) , math.pi*(0), math.pi*(-0.5)])

		if(self.orn_fit() and self.pos_fit_s3()):
			self.stage = 5
			#print("Stage5")
			#print(self.xInRange_s5(), self.yInRange_s5(), self.zInRange_s5())						
			if(self.cid == -1):
				self.cid = p.createConstraint(self._sawyer.sawyerId, self.handPoint, self.objectId, -1, p.JOINT_FIXED, [0, 0, 0], [0., 0.07, -0.02], [0, 0, 0.0], childFrameOrientation = orn)				
			else:
				p.changeConstraint(self.cid, jointChildFrameOrientation=orn, maxForce=1)
			dpx_s5 = 0.002
			dpy_s5 = 0.002
			dpz_s5 = 0.002

			if self.xInRange_15mm():
				dpx_s5 = 0.001
			if self.yInRange_15mm():
				dpy_s5 = 0.001
			if self.zInRange_15mm():
				dpz_s5 = 0.001

			if self.xInRange_7mm():
				dpx_s5 = 0.0005
			if self.yInRange_7mm():
				dpy_s5 = 0.0005
			if self.zInRange_7mm():
				dpz_s5 = 0.0005

			if self.xInRange_s5():
				dpx_s5 = 0.00
			if self.yInRange_s5():
				dpy_s5 = 0.00
			if self.zInRange_s5():
				dpz_s5 = 0.00

			dx = action[0] * dpx_s5
			dy = action[1] * dpy_s5
			dz = action[2] * dpz_s5
			realAction = [dx, dy, dz, 0, 0, 0, 0, 0, 0, 0, 0]

		elif(self.pos_fit_s3()):
			self.stage = 4
			#("Stage4")
			#print(self.s4_ornX_InRange(), self.s4_ornY_InRange(), self.s4_ornZ_InRange())						
			if(self.cid == -1):
				self.cid = p.createConstraint(self._sawyer.sawyerId, self.handPoint, self.objectId, -1, p.JOINT_FIXED, [0, 0, 0], [0., 0.07, -0.02], [0, 0, 0.0], childFrameOrientation = orn)				
			else:
				p.changeConstraint(self.cid, jointChildFrameOrientation=orn, maxForce=1)

			dx = action[0] * 0.00
			dy = action[1] * 0.00
			dz = action[2] * 0.00
			dox, doy, doz = 0.004, 0.004, 0.004		
			if self.s4_ornX_InRange():
				dox = 0.00
			if self.s4_ornY_InRange():
				doy = 0.00
			if self.s4_ornZ_InRange():
				doz = 0.00

			d_ox = action[8] * dox
			d_oy = action[9] * doy			
			d_oz = action[10] * doz
			realAction = [dx, dy, dz, 0, 0, 0, 0, 0, d_ox, d_oy, d_oz]
			

		elif(self._graspSuccess and self.inGrasp()): # stage 3
			self.stage = 3
			#("Stage3")
			#print(self.xInRange_s3(), self.yInRange_s3(), self.zInRange_s3())
			#print(self.pos_fit_s3())			
			if(self.cid == -1):
				self.cid = p.createConstraint(self._sawyer.sawyerId, self.handPoint, self.objectId, -1, p.JOINT_FIXED, [0, 0, 0], [0., 0.07, -0.02], [0, 0, 0.0], childFrameOrientation = orn)				
			else:
				p.changeConstraint(self.cid, jointChildFrameOrientation=orn, maxForce=1)
	
			dpx_s3 = 0.01
			dpy_s3 = 0.01
			dpz_s3 = 0.01
			if self.xInRange_s3():
				dpx_s3 = 0.00
			if self.yInRange_s3():
				dpy_s3 = 0.00
			if self.zInRange_s3():
				dpz_s3 = 0.00
			dx = action[0] * dpx_s3
			dy = action[1] * dpy_s3
			dz = action[2] * dpz_s3
			realAction = [dx, dy, dz, 0, 0, 0, 0, 0, 0, 0, 0]
		elif(self.inPosition()): # stage 2
			self.stage = 2
			#print("Stage2")
			dx = action[0] * d2 * 0
			dy = action[1] * d2 * 0
			dz = action[2] * d2 * 0
			da1 = action[3] * 0.8
			da2 = action[4] * d3
			da3 = action[5] * d3 
			da4 = action[6] * d3 
			da5 = action[7] * d3 
			#d_ox = action[8] * d4
			#d_oy = action[9] * d4			
			#d_oz = action[10] * d4
			realAction = [dx, dy, dz, da1, da2, da3, da4, da5, 0, 0, 0]
		else: # stage 1
			self.stage = 1
			#print("Stage1")
			dpx_s1 = 0.01
			dpy_s1 = 0.01
			dpz_s1 = 0.008
			if self.xInRange():
				dpx_s1 = 0.00
			if self.yInRange():
				dpy_s1 = 0.00
			if self.zInRange():
				dpz_s1 = 0.00
			dx = action[0] * dpx_s1
			dy = action[1] * dpy_s1
			dz = action[2] * dpz_s1
			realAction = [dx, dy, dz, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0]
		return self.step1(realAction)

	def step1(self, action):
		for i in range(self._actionRepeat):
			self._sawyer.applyAction(action, self.terminated, self.terminated_task, self.stage, self.position_target) 
			p.stepSimulation()
			if self._termination(action):
				break
			self._envStepCounter += 1
			if self._renders:
				time.sleep(self._timeStep)
		reward = self._reward()
		self._observation = self.getExtendedObservation()
		scaler = 1.3
		if (self._graspSuccess):
			reward = reward * scaler
		if(self.pos_fit_s3()):
			reward = reward * scaler
		if(self.stage == 4 or self.stage == 5):
			if self.s4_ornX_InRange():
				reward = reward * scaler
			if self.s4_ornY_InRange():
				reward = reward * scaler
			if self.s4_ornZ_InRange():
				reward = reward * scaler

		if(self.stage == 5):

			if self.xInRange_15mm():
				reward = reward * 1.1
			if self.yInRange_15mm():
				reward = reward * 1.1
			if self.zInRange_15mm():
				reward = reward * 1.1

		if(self.stage == 5):

			if self.xInRange_7mm():
				reward = reward * 1.2
			if self.yInRange_7mm():
				reward = reward * 1.2
			if self.zInRange_7mm():
				reward = reward * 1.2


		if(self.stage == 5):

			if self.xInRange_1mm():
				reward = reward * 1.5
			if self.yInRange_1mm():
				reward = reward * 1.5
			if self.zInRange_1mm():
				reward = reward * 1.5

		if(self.stage == 5):

			if self.xInRange_s5():
				reward = reward * scaler
			if self.yInRange_s5():
				reward = reward * scaler
			if self.zInRange_s5():
				reward = reward * scaler
		if (self._taskSuccess):
			reward = reward * (2)

		debug = {'task_succeed': self._taskSuccess}
		done = self._termination(action)

		reward = reward  - self.lamda * len(self.episodeR)
		if done:
			self.episodeR.append(self._taskSuccess)
			self.evaluation.append(self.episodeR)
			#reward = reward  - self.η * error - self.lamda * len(self.episodeR)
		self.episodeR.append(reward)
		#print("reward = ", reward)
		return self._observation, reward, done, debug


	def render(self):
		return 0

	def _termination(self, action):	
		#if (self.orn_fit() or self._envStepCounter > self._maxSteps):
		if (self._envStepCounter > self._maxSteps):
			self._observation = self.getExtendedObservation()
			print("stop due to time out")
			return True
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		contactParts = self.getContactPart()
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		palmTip = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		#print(contactParts[1], contactParts[2]) 
		#print(contactParts)
		if(self.stage == 1):
			if((obPos[0]> self.object_position[0] + 0.015) or (obPos[0] < self.object_position[0] - 0.015)):
			#if( not ((obPos[0] + self.objectFeature[4]*0.5 + 0.006 < indexTip[0]) and (thumbTip[0] < obPos[0] - self.objectFeature[4]*0.5 - 0.006))):
				if( not ((obPos[0] + self.objectFeature[4]*0.5 < indexTip[0]) and (thumbTip[0] < obPos[0] - self.objectFeature[4]*0.5))):
					self._observation = self.getExtendedObservation()
					#print("ObjectNum: ", self.objectFeature[0])
					print("Terminated: x out of range")
					time.sleep(1)
					return True
			if((obPos[1]> self.object_position[1] + 0.015) or (obPos[1] < self.object_position[1]-0.015)):
				if((palmTip[1] >= obPos[1] +  self.objectFeature[2]*0.5) or  (palmTip[1] <= obPos[1] - self.objectFeature[2]*0.5)):
					self._observation = self.getExtendedObservation()
					#print("ObjectNum: ", self.objectFeature[0])
					print("Terminated: y out of range")
					time.sleep(1)
					return True			

		#if(contactParts[1] and contactParts[2] and contactParts[3] and self._graspSuccess !=1 and self.inPosition()): # todo: decide z coordinates
		if(self.stage == 2 and contactParts[1] and contactParts[2]): # todo: decide z coordinates
			#print(contactParts[1],contactParts[2])
			self.terminated = 1
			for i in range(300):
				self._sawyer.applyAction(action, self.terminated, self.terminated_task, self.stage, self.position_target)
				p.stepSimulation()
				objectPosCurrent = p.getBasePositionAndOrientation(self.objectId)[0]
				if (objectPosCurrent[2] > self.height):
					#print(objectPosCurrent[2] > self.height)
					#print("ObjectNum: ", self.objectFeature[0])
					#print("st:", self.st)
					self._observation = self.getExtendedObservation()
					self._graspSuccess = 1
					#print("Terminated: successfully grasp")
					#time.sleep(2)
					break


			if(not self._graspSuccess):
				self._observation = self.getExtendedObservation()
				print("Terminated: Object slipped")
				return True

		if(self._graspSuccess == 1 and self.orn_fit() and self.pos_fit_s5()):
			self._taskSuccess = 1
			#self.terminated_task = 1
			#for i in range(300):
			#	self._sawyer.applyAction(action, self.terminated, self.terminated_task, self.stage, self.position_target)
			#	p.stepSimulation()

			print("Terminated: Succeed")
			self._observation = self.getExtendedObservation()
			time.sleep(1)
			return True
			
		#self.terminated = 0
		return False
	
	def _reward(self): #TODO: move out of range penalty to step1
		reward = 0
		#reward_s1 = self.reward_s1()
		#reward_s2 = self.reward_s2() + self.s1_reward
		#reward_s3 = self.reward_s3() + self.s2_reward
		#reward_s4 = self.reward_s4() + self.s3_reward
		#reward_s5 = self.reward_s5() + self.s4_reward
		#contactParts = self.getContactPart()
		#print(self.orn_fit())
		#print(self.inPosition())

		if(self.pos_fit_s3() and self.orn_fit()):
			if self.s4_reward_index == 0:
				self.s4_reward = self.reward_s4()+ self.s3_reward + 150 + self.s1_reward 
				self.s4_reward_index == 1

			reward = self.reward_s5() + self.s4_reward 

		elif(self.pos_fit_s3()):
			if self.s3_reward_index == 0:
				self.s3_reward = self.reward_s3()+ 150 + self.s1_reward 
				self.s3_reward_index == 1

			reward = self.reward_s4() + self.s3_reward 

		elif(self._graspSuccess and self.inGrasp()): # stage 3

			reward = self.reward_s3() + 150 + self.s1_reward
			
		elif(self.inPosition()): # stage 2
			if self.s1_reward_index == 0:
				self.s1_reward = self.reward_s1() 
				self.s1_reward_index == 1

			reward = self.reward_s2() +  self.s1_reward
			#reward = transition + reward_s2
			#print("stage 2: ", reward)		
		else:# stage 1
			reward = self.reward_s1() 
			#print("stage 1: ", reward)
		#print(reward)
		return reward

	def reward_s1(self):
		#reward = 0
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		handPos = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		#print("handPos-obPos X: ", handPos[0] - obPos[0]  )
		#print("handPos-obPos Y: ", handPos[1] - obPos[1]  )
		#print("handPos-obPos Z: ", handPos[2] - obPos[2]  )
		d = self.distant(handPos, obPos)
		reward = (math.exp(-d*self.disScaler)) * 100
		#cp = p.getContactPoints(self.objectId, self._sawyer.sawyerId)
		#if(len(cp) > 0):
		#	reward = reward - 100
		# inhand reward self.objectFeature[2], self.objectFeature[3], self.objectFeature[4]
		if(self.xInRange()):
			reward = reward + 10	
		if(self.yInRange()):
			reward = reward + 10

		if(self.zInRange()):
			reward = reward + 20
		if(self.show == 1):		
			print("X: ", self.xInRange())
			print("Y: ", self.yInRange())
			print("Z: ", self.zInRange())
		return reward

	def reward_s2(self):
		contactParts = self.getContactPart()
		
		if(sum(contactParts) > 0):
			self.contactPoint = sum(contactParts)
		reward = self.contactPoint*75 # minus 2 since at least 2 contact parts are required				
		return reward

	def reward_s3(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		d = self.distant(obPos, obPos_target)		
		reward = (math.exp(-d*self.disScaler)) * 1000
		return reward

	def reward_s4(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		# x_orn * math.pi, y_orn* math.pi, z_orn * math.pi
		obOrn_eluer = list(p.getEulerFromQuaternion(obOrn))
		obOrn_eluer_target = list(p.getEulerFromQuaternion(obOrn_target))

		ornDiff_x = obOrn_eluer[0] - obOrn_eluer_target[0]
		ornDiff_y = obOrn_eluer[1] - obOrn_eluer_target[1]
		ornDiff_z = obOrn_eluer[2] - obOrn_eluer_target[2] 

		reward_x = (math.exp(-abs(ornDiff_x)*self.disScaler)) * 600
		reward_y = (math.exp(-abs(ornDiff_y)*self.disScaler)) * 600
		#print("reward_y: ", reward_y)
		reward_z = (math.exp(-abs(ornDiff_z)*self.disScaler)) * 600
		reward = reward_x + reward_y + reward_z
		return reward


	def reward_s5(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		d = self.distant(obPos, obPos_target)		
		reward = (math.exp(-d*self.disScaler)) * 3000
		return reward
	

	def xInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		return (obPos[0] + self.objectFeature[4]*0.5 + 0.003 < indexTip[0]) and (obPos[0] - self.objectFeature[4]*0.5 - 0.003> thumbTip[0])
	
	def yInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		palmTip = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		#return (palmTip[1] < obPos[1] + self.objectFeature[2]*0.25) and (palmTip[1] > obPos[1] - self.objectFeature[2]*0.25) 
		return (palmTip[1] < obPos[1] + 0.015) and (palmTip[1] > obPos[1] - 0.015) 

	def zInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		upper = (obPos[2]+ 0.48*self.objectFeature[3] > thumbTip[2]) and (obPos[2]+0.48*self.objectFeature[3] > indexTip[2])

		return upper 

	def inPosition(self):
		result = self.xInRange() and self.yInRange() and self.zInRange()
		return result


	def orn_fit(self):
		return self.s4_ornX_InRange() and self.s4_ornY_InRange() and self.s4_ornZ_InRange() 

	def pos_fit_s3(self):		
		return self.xInRange_s3() and self.yInRange_s3() and self.zInRange_s3()

	def xInRange_s3(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[0]- obPos_target[0])		
		return x <= 0.06

	def yInRange_s3(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[1]- obPos_target[1])		
		return x <= 0.06

	def zInRange_s3(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[2]- obPos_target[2])		
		return x <= 0.06


	def pos_fit_s5(self):		
		return self.xInRange_s5() and self.yInRange_s5() and self.zInRange_s5()

	def xInRange_15mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[0]- obPos_target[0])		
		return x <= 0.015

	def yInRange_15mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[1]- obPos_target[1])		
		return x <= 0.015

	def zInRange_15mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[2]- obPos_target[2])		
		return x <= 0.015

	def xInRange_7mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[0]- obPos_target[0])		
		return x <= 0.007


	def yInRange_7mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[1]- obPos_target[1])		
		return x <= 0.007

	def  zInRange_7mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[2]- obPos_target[2])		
		return x <= 0.007

	def xInRange_1mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[0]- obPos_target[0])		
		return x <= 0.0015

	def yInRange_1mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[1]- obPos_target[1])		
		return x <= 0.0015

	def  zInRange_1mm(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		x = abs(obPos[2]- obPos_target[2])		
		return x <= 0.0015

	def xInRange_s5(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		x = abs(obPos[0]- obPos_target[0])		
		return x <= 0.003


	def yInRange_s5(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		x = abs(obPos[1]- obPos_target[1])	
		return x <= 0.003


	def zInRange_s5(self):
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		x = abs(obPos[2]- obPos_target[2])		
		return x <= 0.003

	def s4_ornX_InRange(self):
		ornError = self.ornError
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		obOrn_eluer = list(p.getEulerFromQuaternion(obOrn))
		obOrn_eluer_target = list(p.getEulerFromQuaternion(obOrn_target))
		ornDiff_x = obOrn_eluer[0] - obOrn_eluer_target[0]
		return abs(ornDiff_x) <=ornError

	def s4_ornY_InRange(self):
		ornError = self.ornError
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		obOrn_eluer = list(p.getEulerFromQuaternion(obOrn))
		obOrn_eluer_target = list(p.getEulerFromQuaternion(obOrn_target))
		ornDiff_y = obOrn_eluer[1] - obOrn_eluer_target[1]
		return abs(ornDiff_y) <=ornError

	def s4_ornZ_InRange(self):
		ornError = self.ornError
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		obPos_target = self.position_target
		obOrn_target = self.orn_target
		obOrn_eluer = list(p.getEulerFromQuaternion(obOrn))
		obOrn_eluer_target = list(p.getEulerFromQuaternion(obOrn_target))
		ornDiff_z = obOrn_eluer[2] - obOrn_eluer_target[2]
		return abs(ornDiff_z) <=ornError

	def distant(self, a, b):
		return sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

	def eva(self):
		return self.evaluation

	def eMean(self):
		m = []
		for i in range(len(self.evaluation)):
			m.append(statistics.mean(self.evaluation[i]))
		return 	statistics.mean(m)

	def handReading(self):
		return self._sawyer.handReading	

	def sus(self):
		return self._taskSuccess

	def inGrasp1(self):		
		normT = [0, 0] # palm, thumb, index, middle, ring, pink		
		frictionT1 = [0, 0]
		frictionT2 = [0, 0]
		# define each part of the hand
		thumbLinks = [58, 59, 60, 61, 62, 63, 64]
		indexLinks = [48, 49, 50, 51, 53, 54, 55]
		# find contact point
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)
		# fill force and dist
		for i in range(nums):			
			if(contact[i][3] in thumbLinks):
				if(contact[i][9] > 0):
					normT[0] = 1
					#print("normT: ", contact[i][9])
				if(contact[i][10] > 0):
					frictionT1[0] = 1
					#print("frictionT1: ", contact[i][10])
				if(contact[i][12] > 0):
					frictionT2[0] = 1
					#print("frictionT2: ", contact[i][12])		
			if(contact[i][3] in indexLinks):
				if(contact[i][9] > 0):
					normT[1] = 1
					#print("normT: ", contact[i][9])
				if(contact[i][10] > 0):
					frictionT1[1] = 1
					#print("frictionT1: ", contact[i][10])
				if(contact[i][12] > 0):
					frictionT2[1] = 1
					#print("frictionT2: ", contact[i][12])		
			
		return ((normT[0] or frictionT1[0] or frictionT2[0]) or (normT[1] or frictionT1[1] or frictionT2[1]))

	def inGrasp(self):

		handPos = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		obPos = p.getBasePositionAndOrientation(self.objectId)[0] 
		distant = self.distant(handPos, obPos)
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)
		if distant >= 0.1 and nums <= 0:
			return False 

		return True



