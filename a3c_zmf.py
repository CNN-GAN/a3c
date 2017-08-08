import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from environment import env_vrep
import os
import shutil
import matplotlib.pyplot as plt

load_model = True
LOG_DIR = './data/log'
N_WORKERS = 8 #multiprocessing.cpu_count()
print ('cpu: ', multiprocessing.cpu_count())
MAX_GLOBAL_EP = 1000
MAX_STEP_EP = 40
BATCH_SIZE = 500
GLOBAL_NET_SCOPE = 'Global_Net'
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_EP = 0

N_S = env_vrep.observation_space
N_A = env_vrep.action_space


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

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
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    self.entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * self.entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                
                with tf.name_scope('total_loss'):
                    self.loss = 0.5 * self.c_loss + self.a_loss

                with tf.name_scope('local_grad'):
                    self.local_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                    self.local_grads = tf.gradients(self.loss, self.local_params)
                    # self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    # self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    # self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    # self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_local_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.local_params, globalAC.local_params)]                    
                    # self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    # self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_global_op = OPT_A.apply_gradients(zip(self.local_grads, globalAC.local_params))                    
                    # self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    # self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('feature'):
            f_fc1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_init, name='f_fc1')
            
        with tf.variable_scope('actor_critic'):
            a_prob = tf.layers.dense(f_fc1, N_A, tf.nn.softmax, kernel_initializer=w_init, name='actor_fc')
            v = tf.layers.dense(f_fc1, 1, kernel_initializer=w_init, name='value_fc')  # state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        # SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        loss, local_grads, _ = SESS.run([self.loss, self.local_grads, self.update_global_op], feed_dict)  # local grads applies to global net
        return loss, local_grads

    def pull_global(self):  # run by a local
        # SESS.run([self.pull_a_params_op, self.pull_c_params_op])
        SESS.run([self.pull_local_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, env, saver, summary_writer, globalAC):
        self.env = env #gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)
        # self.summary_writer = summary_writer

    def process_ep(self, buffer_r):
        buffer_v_target = []
        v_s_ = 0
        for r in buffer_r[::-1]:    # reverse buffer r
            v_s_ = r + GAMMA * v_s_
            buffer_v_target.append(v_s_)

        buffer_v_target.reverse()
        return buffer_v_target

    def work(self):
        global GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_r_real = [], [], [], []
        batch_s, batch_a, batch_r_real = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            step_in_ep = 0
            while step_in_ep < MAX_STEP_EP:
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                if step_in_ep > MAX_STEP_EP:
                    done = True

                ep_r += r

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if done:
                    break
                s = s_
                total_step += 1
                step_in_ep += 1

            buffer_v_target = self.process_ep(buffer_r)

            batch_r_real.extend(buffer_v_target)
            batch_s.extend(buffer_s)
            batch_a.extend(buffer_a)
            buffer_s, buffer_a, buffer_r, buffer_r_real = [], [], [], []

            # print (len(batch_s), len(batch_a), len(batch_r_real))
            mean_return = np.mean(batch_r_real)
            if (len(batch_s) > BATCH_SIZE):
                batch_s, batch_a, batch_r_real = np.vstack(batch_s), np.array(batch_a), np.vstack(batch_r_real)
                feed_dict = {
                    self.AC.s: batch_s,
                    self.AC.a_his: batch_a,
                    self.AC.v_target: batch_r_real,
                }
                loss, grad = self.AC.update_global(feed_dict)

                self.AC.pull_global()

                batch_s, batch_a, batch_r_real = [], [], []
                saver.save(SESS, './data/model.cptk') 

                print (self.name, "updated", mean_return)
                GLOBAL_EP += 1
                # if self.name == 'W_0':
                summary = tf.Summary()
                # mean_return = np.mean(batch_r_real)
                summary.value.add(tag='Perf/Avg Reward', simple_value=float(mean_return))
                summary.value.add(tag='Losses/loss', simple_value=float(loss))
                # summary.histogram.add(tag='Losses/grad', simple_value=float(grad))
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
            env = env_vrep.Simu_env(20000 + i)
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