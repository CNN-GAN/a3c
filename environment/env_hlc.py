import sys, os, math
# from rllab.spaces import Box, Discrete
import numpy as np
import time
## v-rep
from environment.vrep_plugin import vrep
import pickle as pickle

print ('import env vrep')

action_list = []
for a in range(-1, 2):
    for b in range(-1, 2):
        # for c in range(-1, 2):
        action = []
        action.append(a)
        action.append(b)
        action.append(0)
        action.append(0)
        action.append(0)
        action_list.append(action)
        # print action_list

# print action_list
observation_space = 182
action_space = len(action_list)

class Simu_env():
    def __init__(self, port_num):
        # self.action_space = ['l', 'f', 'r', 'h', 'e']

        self.port_num = port_num
        self.dist_pre = 100

        self.path_used = 1
        self.step_inep = 0
        self.object_num = 0
        self.game_level = 3
        self.succed_time = 0
        self.pass_ep = 1
        self.ep_reap_time = 0

        self.connect_vrep()
        self.reset()

    #@property
    #def observation_space(self):
    #    return Box(low=-np.inf, high=np.inf, shape=(1, 182))

    #@property
    #def action_space(self):
    #    return Discrete(len(action_list))

    def get_robot_location(self):
        _, _, g_r_position, _, _ = self.call_sim_function('rwRobot', 'get_robot_position')
        return g_r_position

    def convert_state(self, laser_points, path):
        path = np.asarray(path)
        laser_points = np.asarray(laser_points)
        state = np.append(laser_points, path)
        state = state.flatten()

        # state = np.asarray(path)
        # state = state.flatten()
        return state

    def reset(self):
        # print ('reset')
        self.step_inep = 0

        res, retInts, retFloats, retStrings, retBuffer = self.call_sim_function('rwRobot', 'reset', [self.game_level])        
        state, reward, is_finish, info = self.step([0, 0, 0, 0, 0])
        return state

    def step(self, action):
        self.step_inep += 1

        # res,objs=vrep.simxGetObjects(self.clientID,vrep.sim_handle_all,vrep.simx_opmode_oneshot_wait)
        # if self.object_num != len(objs):
        #     print('connection failed! ', self.object_num, len(objs))
        #     # return Step(observation=state, reward=0, done=False)
        if isinstance(action, np.int32) or isinstance(action, int) or isinstance(action, np.int64):
            action = action_list[action]

        _, _, current_pose, _, found_pose = self.call_sim_function('rwRobot', 'step', action)

        # print (action, current_pose)
        laser_points = self.get_laser_points()
        path_x, path_y = self.get_global_path()  # the target position is located at the end of the list

        if len(path_x) < 1 or len(path_y) < 1:
            print ('bad path length')
            return [0, 0], 0, False, 'f'

        #compute reward and is_finish
        reward, is_finish = self.compute_reward(action, path_x, path_y, found_pose)

        path_f = []
        sub_path = [path_x[-1], path_y[-1]] # target x, target y (or angle)
        path_f.append(sub_path)

        state_ = self.convert_state(laser_points, path_f, )

        return state_, reward, is_finish, ''

    def compute_reward(self, action, path_x, path_y, found_pose):
        is_finish = False
        dist = math.sqrt(path_x[-1]*path_x[-1] + path_y[-1]*path_y[-1])
        reward = 0

        # sum_action = np.sum(np.abs(action))
        # if (sum_action) == 0:
        #     reward -= 5

        dist = math.sqrt(path_x[-1]*path_x[-1] + path_y[-1]*path_y[-1])
        # dist = path_x[-1]
        if dist >= self.dist_pre:
            reward -= 0.1

        self.dist_pre = dist
        # diff = self.dist_pre - dist
        # reward += diff * 20

        if dist < 0.2:              # when reach to the target
            is_finish = True
            # self.succed_time += 1
            reward = 10            # 9
            self.ep_reap_time = 0
            self.pass_ep = 1
            # if self.succed_time > 10:
            #     self.game_level += 1
            #     self.succed_time = 0

        if found_pose == bytearray(b"f"):       # when collision or no pose can be found
            is_finish = True 
            self.succed_time = 0 
            reward = -10            # -11
            self.pass_ep = -1

        # print (reward, diff, self.dist_pre)
        # print(reward)
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
