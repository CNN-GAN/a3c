#!/usr/bin/env python
import sys, os, math
# from rllab.spaces import Box, Discrete

import numpy as np
import time
## v-rep
from environment.vrep_plugin import vrep
import pickle as pickle

print ('import env vrep')

action_hlc = []
for a in range(-1, 2):
    for b in range(-1, 2):
        # for c in range(-1, 2):
        action = []
        action.append(a)
        action.append(b)
        action.append(0)
        action.append(0)
        action.append(0)
        # print action
        action_hlc.append(action)
        # print action_list

action_list = []
for a in range(-1, 2):
    for b in range(-1, 2):
        for c in range(-1, 2):
            for d in range(-1, 2):
                for e in range(-1, 2):
                    action = []
                    action.append(a)
                    action.append(b)
                    action.append(c)
                    action.append(d)
                    action.append(e)
                    # print action
                    action_list.append(action)
                    # print action_list

# print action_list
observation_space = 183
action_space = len(action_list)

class Simu_env():
    def __init__(self, port_num):
        # self.action_space = ['l', 'f', 'r', 'h', 'e']

        self.port_num = port_num
        self.dist_pre = 100

        self.path_used = 1
        self.step_inep = 0
        self.object_num = 0
        self.game_level = 4
        self.succed_time = 0
        self.pass_ep = 1
        self.ep_reap_time = 0
        self.hlc_index = 0

        self.connect_vrep()
        self.reset()

    #@property
    #def observation_space(self):
    #    return Box(low=-np.inf, high=np.inf, shape=(1, 182))

    #@property
    #def action_space(self):
    #    return Discrete(len(action_list))

    def convert_state(self, laser_points, current_pose, hlc_index):
        current_pose = np.asarray(current_pose)
        laser_points = np.asarray(laser_points)
        for i in range(len(laser_points)):
            if laser_points[i] > 1:
                laser_points[i] = 1
        
        state = np.append(laser_points, current_pose)
        state = np.append(state, hlc_index)
        
        state = state.flatten()
        return state

    def get_observation(self):
        # self.hlc_index = 0
        while True:
            # hlc_index = np.random.randint(len(action_hlc))
            # action = action_hlc[self.hlc_index]
            action = [0,1,0,0,0]
            found_pose, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('rwRobot', 'step_hlc', action)     
            if retBuffer == bytearray(b"t"):
                break

        res, retInts, current_pose, retStrings, found_pose = self.call_sim_function('rwRobot', 'step', [0,0,0,0,0])
        laser_points = self.get_laser_points()
        state = self.convert_state(laser_points, current_pose, self.hlc_index)
        return state

    def reset(self):
        # print ('reset')
        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('rwRobot', 'reset', [self.game_level])        
        state = self.get_observation()
        return state

    def step(self, action):
        self.step_inep += 1
        if self.step_inep % 10 == 0:
            self.hlc_index = (self.hlc_index+1)%len(action_hlc)

        if isinstance(action, np.int32) or isinstance(action, int) or isinstance(action, np.int64):
            action = action_list[action]

        res, retInts, current_pose, retStrings, found_pose = self.call_sim_function('rwRobot', 'step', action)
        laser_points = self.get_laser_points()

        #compute reward and is_finish
        res, retInts, dist, retStrings, found_pose = self.call_sim_function('rwRobot', 'get_dist', action)
        reward = 0
        if abs(dist[0]) > 0.05:
            reward -= 1
        if abs(current_pose[0]) > 0.05:
            reward -= 1
        if abs(current_pose[1] - 0.07234) > 0.05:
            reward -= 1

        if found_pose == bytearray(b"f"):       # when collision or no pose can be found
            reward -= 10           

        print (current_pose, dist[0], reward)
        print ([0,1,0,0,0])
        print (action)
        state_ = self.get_observation()
        # return Step(observation=state_, reward=reward, done=is_finish)
        return state_, reward, False, ''

    def compute_reward(self, action, path_x, path_y, found_pose):
        is_finish = False
        reward = -0.5

        # if abs(action[0]) == 1:
        #     reward -= 0.5

        # sum_action = np.sum(np.abs(action))
        # if (sum_action) == 0:
        #     reward -= 5

        dist = math.sqrt(path_x[-1]*path_x[-1] + path_y[-1]*path_y[-1])
        # dist = path_x[-1]
        # print (dist, self.dist_pre)
        diff = self.dist_pre - dist
        reward += diff * 20
        if reward > 0 and action[1] == 1:
            reward += diff * 10
        if action[1] == -1:
            reward -= abs(diff) * 10
        # if dist - self.dist_pre < -0.02:  # when closer to target
        #     reward += 1            # 1
        # else:
        #     reward -= 1            # -3

        self.dist_pre = dist

        if dist < 0.1:              # when reach to the target
            is_finish = True
            # self.succed_time += 1
            reward += 10            # 9
            self.ep_reap_time = 0
            self.pass_ep = 1
            # if self.succed_time > 10:
            #     self.game_level += 1
            #     self.succed_time = 0

        # if dist > 5:                # when too far away to the target
        #     is_finish = True
        #     self.succed_time = 0
        #     reward -= 2             # -3

        if found_pose == bytearray(b"f"):       # when collision or no pose can be found
            # is_finish = True 
            self.succed_time = 0 
            reward -= 10            # -11
            self.pass_ep = -1

        # if self.step_inep > 115:
        #     is_finish = True 
        #     self.succed_time = 0 
        #     reward -= 2             # -3
        #     self.pass_ep = -1

        return reward, is_finish



    ####################################  interface funcytion  ###################################

    def connect_vrep(self):

        clientID = vrep.simxStart('127.0.0.1', self.port_num, True, True, 5000, 5)
        if clientID != -1:
            print ('Connected to remote API server with port: ', self.port_num)
        else:
            print ('Failed connecting to remote API server with port: ', self.port_num)


        self.clientID = clientID
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)

    def disconnect_vrep(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)
        vrep.simxFinish(self.clientID)
        print ('Program ended')


    ########################################################################################################################################
    ###################################   interface function to communicate to the simulator ###############################################
    def call_sim_function(self, object_name, function_name, input_floats=[]):
        inputInts = []
        inputFloats = input_floats
        inputStrings = []
        inputBuffer = bytearray()
        res,retInts,retFloats,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID, object_name,vrep.sim_scripttype_childscript,
                    function_name, inputInts, inputFloats, inputStrings,inputBuffer, vrep.simx_opmode_blocking)

        # print 'function call: ', self.clientID
        return res, retInts, retFloats, retStrings, retBuffer

    def get_laser_points(self):
        res,retInts,retFloats,retStrings,retBuffer = self.call_sim_function('LaserScanner_2D', 'get_laser_points')
        return retFloats

    def get_global_path(self):
        res,retInts, path_raw, retStrings, retBuffer = self.call_sim_function('rwRobot', 'get_global_path')

        if len(path_raw) < 2 :
            print (path_raw)

        path_dist = []
        path_angle = []

        for i in range(0, len(path_raw), 2):       
            path_dist.append(path_raw[i])
            path_angle.append(path_raw[i+1])

        return path_dist, path_angle




# env = Simu_env(20000)
# print (env.action_space())
# print (env.observation_space())
