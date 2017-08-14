import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from environment import env_hlc
import os
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import pickle

load_model = True
LOG_DIR = './data/log'
N_WORKERS = 4 #multiprocessing.cpu_count()
print ('cpu: ', multiprocessing.cpu_count())
MAX_GLOBAL_EP = 10000
MAX_STEP_EP = 100
BATCH_SIZE = 50
GLOBAL_NET_SCOPE = 'Global_Net'
GAMMA = 0.95
ENTROPY_BETA = 0.001
LR_A = 0.002    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GLOBAL_EP = 0


batch_dirs = ['./batch/batch_collision/', './batch/batch_goal/', './batch/batch_unfinish/']
list = os.listdir(batch_dirs[0]) 
ep_count_c = len(list)
list = os.listdir(batch_dirs[1]) 
ep_count_g = len(list)
list = os.listdir(batch_dirs[2]) 
ep_count_u = len(list)


N_S = env_hlc.observation_space
N_A = env_hlc.action_space

v_img = np.zeros((100, 100), np.float32)

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope )
                # self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                # self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td)) * 0.5

                with tf.name_scope('a_loss'):
                    log_a_prob = tf.log(tf.clip_by_value(self.a_prob,1e-10,1.0))
                    log_prob = tf.reduce_sum(log_a_prob * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * log_a_prob, axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('total_loss'):
                    self.loss = self.c_loss * 0.5 + self.a_loss

                with tf.name_scope('local_grad'):
                    self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope )
                    self.grads = tf.gradients(self.loss, self.params)

                    # self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    # self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    # self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, globalAC.params)]
                    # self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    # self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_op = OPT_A.apply_gradients(zip(self.grads, globalAC.params))
                    # self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    # self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('feature'):
            # l_a = tf.layers.dense(self.s, 16, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.laser = tf.slice(self.s, [0, 0], [-1, 180])
            self.target = tf.slice(self.s, [0, 180], [-1, 2])

            # # process laser
            laser_reshape = tf.reshape(self.laser,shape=[-1, 180, 1]) 
            conv1 = tf.layers.conv1d(   inputs=laser_reshape,
                                        filters=16,
                                        kernel_size=8,
                                        padding="valid",
                                        activation=tf.nn.relu6,
                                        name = 'laser_conv1')
            conv2 = tf.layers.conv1d(   inputs=conv1,
                                        filters=32,
                                        kernel_size=4,
                                        padding="valid",
                                        activation=tf.nn.relu6,
                                        name = 'laser_conv2')     
            conv_flat = tf.contrib.layers.flatten(conv2)
            conv_fc = tf.layers.dense(inputs=conv_flat, units=200, activation=tf.nn.relu6, name = 'laser_conv_fc')

            # # process laser
            # laser_reshape = tf.reshape(self.laser,shape=[-1, 180]) 
            # laser_fc1 = tf.layers.dense(inputs=laser_reshape, units=90, activation=tf.nn.relu6, name = 'laser_fc1')
            # laser_fc2 = tf.layers.dense(inputs=laser_fc1, units=45, activation=tf.nn.relu6, name = 'laser_fc2')

            target_reshape = tf.reshape(self.target,shape=[-1, 2]) 
            # target_fc = tf.layers.dense(inputs=target_reshape, units=32, activation=tf.nn.relu6, name = 'target_fc1')
            # path_fc2 = tf.layers.dense(inputs=path_fc, units=32, activation=tf.nn.relu, name = 'target_fc2')

            # concat laser and target
            feature = tf.concat([conv_fc, target_reshape], 1, name = 'concat')
            feature = tf.layers.dense(inputs=feature, units=100, activation=tf.nn.relu, name = 'concat_fc1')

            #------------------------------------------------------------------------------------------------------------------------
            # feature = tf.layers.dense(inputs=self.s, units=64, activation=tf.nn.relu, name = 'concat_fc1')
            # feature = tf.layers.dense(inputs=feature, units=32, activation=tf.nn.relu, name = 'concat_fc2')
            # feature = tf.layers.dense(inputs=feature, units=45, activation=tf.nn.relu, name = 'concat_fc3')

        with tf.variable_scope('actor'):
            # l_a = tf.layers.dense(target_fc, 32, tf.nn.relu6, kernel_initializer=w_init, name='actor_fc')
            a_prob = tf.layers.dense(feature, N_A, tf.nn.softmax, kernel_initializer=w_init, name='actor_prob')
        with tf.variable_scope('critic'):
            # l_c = tf.layers.dense(target_fc, 32, tf.nn.relu6, kernel_initializer=w_init, name='critic_fc')
            v = tf.layers.dense(feature, 1, kernel_initializer=w_init, name='critic_value')  # state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        # v, c_loss, a_loss, _, _ = SESS.run([self.v, self.c_loss, self.a_loss, self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        v, c_loss, a_loss, _= SESS.run([self.v, self.c_loss, self.a_loss, self.update_op], feed_dict)  # local grads applies to global net
        
        return v, c_loss, a_loss 

    def update_global_cnet(self, feed_dict):
        v, c_loss, _ = SESS.run([self.v, self.c_loss, self.update_op], feed_dict)  # local grads applies to global net
        return v, c_loss

    def pull_global(self):  # run by a local
        SESS.run(self.pull_params_op)

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, prob_weights


class Worker(object):
    def __init__(self, name, env, saver, summary_writer, globalAC):
        self.env = env #gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.saver = saver
        self.summary_writer = summary_writer

        # if self.name == 'W_0':
        #     plt.ion() # enable interactivity

    def process_ep(self, buffer_r):
        buffer_v_target = []
        v_s_ = 0
        for r in buffer_r[::-1]:    # reverse buffer r
            v_s_ = r + GAMMA * v_s_
            buffer_v_target.append(v_s_)

        buffer_v_target.reverse()
        return buffer_v_target


    def get_ep_filename(self):
        batch_dirs = ['./batch/batch_collision/', './batch/batch_goal/', './batch/batch_unfinish/']
        ep_type = 0

        rand = np.random.rand()
        if rand < 0.3:
            ep_type = 0
        elif rand < 0.7:
            ep_type = 1
        else:            ep_type = 2

        dir = batch_dirs[ep_type]
        list = os.listdir(dir) # dir is your directory path
        number_files = len(list) -1

        # low_b = number_files * 4/5
        low_b = number_files - 200

        ep_i = np.random.randint(low_b, number_files)
        filename = dir + str(ep_i) + '.pkl'

        return filename, ep_type

    def get_batch(self):
        global BATCH_SIZE
        batch_size = 0

        while batch_size < BATCH_SIZE:
            filename, ep_type = self.get_ep_filename()
            try:
                f = open(filename, 'rb')
            except:
                continue      
            x = pickle.load(f)
            if batch_size == 0:
                batch = x
            else:
                batch = np.append(batch, x, axis = 1)
            batch_size = len(batch[0])

        # print(len(batch[0]))
        return batch

    def update_use_replay(self):
        x = self.get_batch()

        batch_s = x[0]
        batch_a = x[1]
        batch_r = x[2]
        batch_v_real = x[3]

        batch_s, batch_a, batch_v_real = np.vstack(batch_s), np.array(batch_a), np.vstack(batch_v_real)
        feed_dict = {
            self.AC.s: batch_s,
            self.AC.a_his: batch_a,
            self.AC.v_target: batch_v_real,
        }
        v, c_loss = self.AC.update_global_cnet(feed_dict)
        # print ('replay: ', c_loss)
        summary = tf.Summary()
        # summary.value.add(tag='Perf/Avg reward', simple_value=float(mean_reward))
        # summary.value.add(tag='Perf/Avg return', simple_value=float(mean_return))
        summary.value.add(tag='Loss/V loss', simple_value=float(c_loss))
        # summary.value.add(tag='Loss/A loss', simple_value=float(a_loss))
        summary_writer.add_summary(summary, GLOBAL_EP)
        summary_writer.flush()  


    def work(self):
        global GLOBAL_EP, MAX_STEP_EP
        # self.AC.pull_global()
        buffer_s, buffer_a, buffer_r, buffer_r_real = [], [], [], []
        batch_s, batch_a, batch_r, batch_v_real = [], [], [], []
        type_index = 'u'
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            self.AC.pull_global()
            start_prob = []
            last_prob = []
            for step_in_ep in range(MAX_STEP_EP):
            # while step_in_ep < MAX_STEP_EP:
                # if self.name == 'W_0':
                a, last_prob = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                if r == 10:
                    type_index = 'g'
                elif r == -10:
                    type_index = 'c'
                else:
                    type_index = 'u'

                if step_in_ep == 0:
                    start_prob = last_prob[0]

                if step_in_ep == MAX_STEP_EP-1 or done:
                    r = r + GAMMA * SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if done:
                    break
                s = s_

            buffer_v_target = self.process_ep(buffer_r)

            self.save_buffer(buffer_s, buffer_a, buffer_r, buffer_v_target, type_index)     

            # mean_reward = np.mean(buffer_r)
            # mean_return = np.mean(buffer_v_target)          
            # for i in range(200): 
            #     self.update_use_replay()

            batch_v_real.extend(buffer_v_target)
            batch_s.extend(buffer_s)
            batch_a.extend(buffer_a)
            batch_r.extend(buffer_r)
            buffer_s, buffer_a, buffer_r, buffer_r_real = [], [], [], []

            mean_reward = np.mean(batch_r)
            mean_return = np.mean(batch_v_real)
            
            GLOBAL_EP += 1

            if (len(batch_s) > BATCH_SIZE):
                batch_s, batch_a, batch_v_real = np.vstack(batch_s), np.array(batch_a), np.vstack(batch_v_real)
                feed_dict = {
                    self.AC.s: batch_s,
                    self.AC.a_his: batch_a,
                    self.AC.v_target: batch_v_real,
                }
                v, c_loss, a_loss = self.AC.update_global(feed_dict)
                # self.AC.pull_global()

                # print(batch_v_real[-1][0], v[-1][0], c_loss, a_loss)

                a_, prob = self.AC.choose_action(batch_s[-1])

                # if self.name == 'W_0':
                print(batch_a[-1], a_, 'predict_v:', v[-1], 'real_v:', batch_v_real[-1], batch_r[-1])
                print (last_prob)
                print (prob - start_prob)

                batch_s, batch_a, batch_r, batch_v_real = [], [], [], []

                self.write_summary(mean_reward, mean_return, c_loss, a_loss)


    def save_buffer(self, buffer_s, buffer_a, buffer_r, buffer_v_target, type_index):
        global ep_count_g, ep_count_c, ep_count_u
        save_batch = [buffer_s, buffer_a, buffer_r, buffer_v_target]

        if type_index == 'g':
            file_name = './batch/batch_goal/' + str(ep_count_g) + '.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(save_batch, f)
            ep_count_g += 1
        elif type_index == 'c':
            file_name = './batch/batch_collision/' + str(ep_count_c) + '.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(save_batch, f)
            ep_count_c += 1
        else:
            file_name = './batch/batch_unfinish/' + str(ep_count_u) + '.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(save_batch, f)
            ep_count_u += 1           


    def debug_function(prob):
        # if self.name == 'W_0':
        #     robot_loc = env.get_robot_location()
        #     row = 100 - int((robot_loc[1] + 2.5)*20)
        #     col = int((robot_loc[0] + 2.5)*20)
        #     v_img[row, col] = value

        #     min_v = v_img.min()
        #     max_v = v_img.max()
        #     scale = abs(max_v - min_v)
        #     img = (v_img - min_v)/scale * 255
        #     plt.clf()
        #     plt.imshow(img,cmap='gray')
        #     print (min_v, max_v, row, col)

        # print (a, r, prob)
        # if self.name == 'W_0':
            # cv2.circle(v_img, (row,col), )
        #     plt.clf()
            plt.plot(prob[0])
        #     plt.pause(0.001)


    def write_summary(self, mean_reward, mean_return, c_loss, a_loss):
        self.saver.save(SESS, './data/model.cptk') 
        # if self.name == 'W_0':
        summary = tf.Summary()

        summary.value.add(tag='Perf/Avg reward', simple_value=float(mean_reward))
        summary.value.add(tag='Perf/Avg return', simple_value=float(mean_return))
        summary.value.add(tag='Loss/V loss', simple_value=float(c_loss))
        summary.value.add(tag='Loss/A loss', simple_value=float(a_loss))
                
        # summary.value.add(tag='Losses/loss', simple_value=float(loss))
        # summary.histogram.add(tag='Losses/start_prob', simple_value=float(start_prob))
        summary_writer.add_summary(summary, GLOBAL_EP)
        summary_writer.flush()  

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        saver = tf.train.Saver(max_to_keep=5)
        summary_writer = tf.summary.FileWriter('data/log', SESS.graph)

        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            env = env_hlc.Simu_env(20000 + i)
            workers.append(Worker(i_name, env, saver, summary_writer, GLOBAL_AC))
    
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state('./data/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(SESS, ckpt.model_checkpoint_path)
            print ('loaded')
        else:
            print ('no model file')
    
    summary_writer = tf.summary.FileWriter('data/log', SESS.graph)
    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
COORD.join(worker_threads)