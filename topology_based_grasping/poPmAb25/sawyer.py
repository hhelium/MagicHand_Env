import os, inspect
currentdir = os.getcwd()
parentdir = os.path.abspath(os.path.join(currentdir, os.pardir))
startdir = os.path.abspath(os.path.join(parentdir, os.pardir))
sawyerdir = os.path.join(startdir, "robot/sawyer_robot/sawyer_description/urdf/sawyer.urdf")
import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import random
from graspTypes import graspTypes
import time

class sawyer:

	def __init__(self, timeStep=0.01, graspType = "poPmAd35", orientation = 0, handPoint = 47):
		self.timeStep = timeStep
		self.maxVelocity = 10
		self.maxForce = 500.
		self.fingerAForce = 2
		self.fingerBForce = 2.5
		self.fingerTipForce = 2
		self.useInverseKinematics = 1
		self.useSimulation = 1
		self.useNullSpace = 21
		self.useOrientation = 1
		#self.palmIndex = 62
		self.palmIndex = handPoint
		self.prevPoseT =  (0.8673731684684753, 0.038900118321180344, -0.1448102593421936)
		self.prevPoseI =  (0.9581236243247986, 0.04836687073111534, -0.12839142978191376)
		self.prevPoseM =  (0.9593610167503357, 0.027400199323892593, -0.13042651116847992)
		self.prevPoseR =  (0.9665053486824036, 0.005422461312264204, -0.12007617205381393)
		self.prevPoseP =  (0.966949462890625, -0.015390047803521156, -0.11987597495317459)
		self.ja = []
		self.inPosition = 0
		#self.kukaGripperIndex = 7
		#lower limits for null space
		self.ll = [-3.0503, -5.1477, -3.8183, -3.0514, -3.0514, -2.9842, -2.9842, -4.7104, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 
			0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.85, 0.34, 0.17]
		#upper limits for null space
		self.ul = [3.0503, 0.9559, 2.2824, 3.0514, 3.0514, 2.9842, 2.9842, 4.7104, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 
			1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 2.15, 1.5, 1.5]
		#joint ranges for null space
		self.jr = [0, 0, 0, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.3, 1.16, 1.33]
		#restposes for null space
		self.rp = [0]*35
		#joint damping coefficents
		self.jd = [0.0001] * 35
		self.arm = [3, 4, 8, 9, 10, 11, 13, 16]
		self.hand = [21, 22, 23, 26, 27, 28, 30, 31, 32, 35, 36 ,37, 39, 40, 41, 44, 45, 46, 48, 49, 50, 53, 54, 55, 58, 61, 64]
		self.js = [3, 4, 8, 9, 10, 11, 13, 16, 21, 22, 23, 26, 27, 28, 30, 31, 32, 35, 36 ,37, 39, 40, 41, 44, 45, 46, 48, 49, 50, 53, 54, 55, 58, 61, 64]
		self.objectId = -100 
		self.sawyerId = -100
		self.readings = []
		self.graspType = graspType
		self.orientation = orientation
		self.reset()	

	def reset(self):
		p.setTimeStep(self.timeStep)
		
		self.sawyerId = p.loadURDF(sawyerdir, [0,0,-1], [0,0,0,3], useFixedBase = 1)
		#self.prevPose1 = p.getLinkState(self.sawyerId, 51)

		p.resetBasePositionAndOrientation(self.sawyerId, [-0.10000, 0.000000, 0.0000], [0.000000, 0.000000, 0.000000, 1.000000])

		self.gt = graspTypes(self.sawyerId)
		t2a = { "poPmAd35" : self.gt.poPmAd35, "poPmAb25" : self.gt.poPmAb25, "poPmAd25" : self.gt.poPmAd25, "poPdAb2" : self.gt.poPdAb2, "poPdAb23" : self.gt.poPdAb23, 
			"poPdAb24" : self.gt.poPdAb24, "iAb2" : self.gt.iAb2, "iAd2" : self.gt.iAd2, "iAd3": self.gt.iAd3, "pPdAb2": self.gt.pPdAb2, "pPdAb23": self.gt.pPdAb23, 
			"pPdAb24": self.gt.pPdAb24, "pPdAb25": self.gt.pPdAb25, "pPdAd25":self.gt.pPdAd25, "pSAb3":self.gt.pSAb3}

		self.handInitial, self.maxRange = t2a[self.graspType]()
		if (self.orientation == 0):
			self.armInitial = [-0.725201592068791072, 0.039273803429399234, 1.1960756155189447, 0.432204978167223, -1.6019898175193155, 0.7926017146715626, 0.6437789421536294, 						0.8425926422906315]
		else:
			self.armInitial = [-0.725201592068791072, 0.039273803429399234, 1.1960756155189447, 0.432204978167223, -1.6019898175193155, 0.7926017146715626, 0.6437789421536294, 						0.8425926422906315]
			#self.armInitial = [0, 0, 0, 0, 0, 0, 0, 0]
			
		self.handInitial, self.maxRange = t2a[self.graspType]()
		self.jointPositions = self.armInitial + self.handInitial
		self.numJoints = len(self.js)
		for i in range(self.numJoints):
			p.resetJointState(self.sawyerId, self.js[i], self.jointPositions[i])
			p.setJointMotorControl2(self.sawyerId, self.js[i], 
						p.POSITION_CONTROL, 
						targetPosition=self.jointPositions[i], 
						force= self.maxForce)

		self.palmPos = [0.99,0,0.1]
		self.endEffectorAngle = 0

		self.motorNames = []
		self.motorIndices = []

		for i in range(self.numJoints):
			jointInfo = p.getJointInfo(self.sawyerId, i)
			qIndex = jointInfo[3]
			if qIndex > -1:
				self.motorNames.append(str(jointInfo[1]))
				self.motorIndices.append(i)


	def getActionDimension(self):
		if (self.useInverseKinematics):
			return len(self.motorIndices)
		return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector


	def getObservationDimension(self):
		return len(self.getObservation())


	def applyAction(self, motorCommands, terminated):		
		#print(self.palmPos)
		if (self.useInverseKinematics):
		#if (terminatedArm == 0):
			# read action
			positonX = motorCommands[0]
			#print('positonX', positonX)
			positonY = motorCommands[1]
			positonZ = motorCommands[2]
			thumbS = motorCommands[3]
			indexS = motorCommands[4]
			middleS = motorCommands[5]
			ringS = motorCommands[6]
			pinkyS = motorCommands[7]

			################################################### control arm ###################################################
			# surface -0.15
			xbound = 0.05
			ybound = 0.05
			zbound = 0.15
			zboundLow = 0.16

			self.palmPos[0] = self.palmPos[0] + positonX
			if (self.palmPos[0] > 1 + xbound):
				self.palmPos[0] = 1 + xbound
			if (self.palmPos[0] < 1 - xbound):
				self.palmPos[0] = 1 - xbound

			self.palmPos[1] = self.palmPos[1] + positonY
			if (self.palmPos[1] <= 0 - ybound):
				self.palmPos[1] = 0 - ybound
			if (self.palmPos[1] >= 0 + ybound):
				self.palmPos[1] = 0 + ybound

			self.palmPos[2] = self.palmPos[2] + positonZ
			if (self.palmPos[2] >= zbound):
				self.palmPos[2] = zbound		

			if (self.palmPos[2] <= -zboundLow):
				self.palmPos[2] = -zboundLow

			#if (self.palmPos[2] <= -0.055):
				#self.palmPos[2] = -0.055

			# set palm postion and orientations
			pos = self.palmPos
			#print(p.getLinkState(self.sawyerId, 38)[0])
			#orn = p.getQuaternionFromEuler([-math.pi*(0.5 - (1/36)) , 0, -math.pi*0.5])
			orn = p.getQuaternionFromEuler([-math.pi*(0.5) , 0, -math.pi*0.5])
			# calculate joint angles for pos orn (35 revolute joints)
			jointPoses = p.calculateInverseKinematics(self.sawyerId,
                                               			self.palmIndex,
                                               			pos,
                                               			orn,
                                               			jointDamping=self.jd)
			# map 35 joints to 65 joints 
			j=0
			jointP = [0]*65
			for i in self.js:
				jointP[i] = jointPoses[j]
				j=j+1
			jointPoses = jointP
			# move each joint to its target angle in jointPoses
			for i in range(self.numJoints):
				#print(i)
				p.setJointMotorControl2(bodyUniqueId=self.sawyerId,
                                  				jointIndex=i,
                                  				controlMode=p.POSITION_CONTROL,
                                 				targetPosition=jointPoses[i],
                                  				targetVelocity=0,
                                  				force=self.maxForce,
                                  				maxVelocity=self.maxVelocity,
                                  				positionGain=0.03,
                                  				velocityGain=1)			

			################################################### Grasping ###################################################
			#if(b!=0):
			# assign action to hand 
			#[thumbl, thumbm, indexl, indexm, midl, midm, ringl, ringm, pinkl, pinkm ]
			#self.hand = [21, 22, 23, 26, 27, 28, 30, 31, 32, 35, 36 ,37, 39, 40, 41, 44, 45, 46, 48, 49, 50, 53, 54, 55, 58, 61, 64]
			scaler = [thumbS, thumbS, indexS, indexS, middleS, middleS, ringS, ringS, pinkyS, pinkyS]
			startPos = [self.handInitial[24], self.handInitial[25], self.handInitial[18], self.handInitial[19], self.handInitial[12], self.handInitial[13], self.handInitial[6], 
				    self.handInitial[7], self.handInitial[0], self.handInitial[1]] 	
	
			difference = np.zeros(10)
			for i in range(10):
				difference[i] = self.maxRange[i] - startPos[i]
			final = np.zeros(10)
			for i in range(10):
				final[i] = startPos[i] + (difference[i] * scaler[i])
			if(self.inPosition > 250):
				#self.prevPoseT = p.getLinkState(self.sawyerId, 62)[4] #
				#self.prevPoseI = p.getLinkState(self.sawyerId, 51)[4]
				#self.prevPoseM = p.getLinkState(self.sawyerId, 42)[4]
				#self.prevPoseR = p.getLinkState(self.sawyerId, 33)[4]
				#self.prevPoseP = p.getLinkState(self.sawyerId, 24)[4]
				#print("self.prevPoseT = ", self.prevPoseT)
				#print("self.prevPoseI = ", self.prevPoseI)
				#print("self.prevPoseM = ", self.prevPoseM)
				#print("self.prevPoseR = ", self.prevPoseR)
				#print("self.prevPoseP = ", self.prevPoseP)

				self.gt.indexF(final[2], final[3])	# 48, 53, 49, 54	# tip 50, 55 self.handInitial[20], self.handInitial[23]
				self.gt.midF(final[4], final[5])	# 39, 44, 40, 45	# tip 41, 46 self.handInitial[14], self.handInitial[17]
				self.gt.ringF(final[6], final[7])	# 30, 35, 31, 36	# tip 32, 37 self.handInitial[8], self.handInitial[11]
				self.gt.pinkyF(final[8], final[9])	# 21, 26, 22, 27	# tip 23, 28 self.handInitial[2], self.handInitial[5]
				self.gt.thumb(startPos[0], final[1])	# 58, 61, 64		# tip 	

				poseT = p.getLinkState(self.sawyerId, 62)[4] #
				poseI = p.getLinkState(self.sawyerId, 51)[4]
				poseM = p.getLinkState(self.sawyerId, 42)[4]
				poseR = p.getLinkState(self.sawyerId, 33)[4]
				poseP = p.getLinkState(self.sawyerId, 24)[4]

				#p.addUserDebugLine(self.prevPoseT, poseT, [1, 0, 0], 6, 5) #index
				self.prevPoseT= poseT	
				#p.addUserDebugLine(self.prevPoseI, poseI, [0, 1, 0], 6, 5) #index
				self.prevPoseI= poseI
				#p.addUserDebugLine(self.prevPoseM, poseM, [0, 0, 1], 6, 5) #index
				self.prevPoseM= poseM	
				#p.addUserDebugLine(self.prevPoseR, poseR, [1, 1, 0], 6, 5) #index
				self.prevPoseR= poseR	
				#p.addUserDebugLine(self.prevPoseP, poseP, [1, 0, 1], 6, 5) #index
				self.prevPoseP= poseP	
				#self.prevPoseT = [1, 0, -0.1418]
				#self.prevPoseI = [1, 0, -0.1418]
				#self.prevPoseM = [1, 0, -0.1418]
				#self.prevPoseR = [1, 0, -0.1418]
				#self.prevPoseP = [1, 0, -0.1418]
			# pick up object if terminated
			if (terminated == 1):
				state = p.getLinkState(self.sawyerId, self.palmIndex)[0]
				posT = [self.palmPos[0], self.palmPos[1], (self.palmPos[2] + 0.25)]
				#posT = [state[0], state[1], (state[2] + 0.25)]
				#p.setTimeStep(self._timeStep)
				self.gt.palmP(posT, orn)
		self.inPosition = self.inPosition + 1






