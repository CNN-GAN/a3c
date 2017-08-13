import tensorflow as tf 
import numpy as np
import pickle
import os

LR_A = 0.01
LR_C = 0.01

N_S = 182
N_A = 9
ENTROPY_BETA = 0.001


class ACNet(object):
    def __init__(self, scope):
        with tf.variable_scope(scope):

            self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
            self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
            self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

            self.a_prob, self.v = self._build_net()

            td = tf.subtract(self.v_target, self.v, name='TD_error')
            with tf.name_scope('c_loss'):
                self.c_loss = tf.reduce_mean(tf.square(td))

            # with tf.name_scope('a_loss'):
            #     log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
            #     exp_v = log_prob * td
            #     entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
            #     self.exp_v = ENTROPY_BETA * entropy + exp_v
            #     self.a_loss = tf.reduce_mean(-self.exp_v)

            with tf.name_scope('train'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=LR_C)
                self.train = optimizer.minimize(self.c_loss)

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('feature'):
            # l_a = tf.layers.dense(self.s, 16, tf.nn.relu6, kernel_initializer=w_init, name='la')
            self.laser = tf.slice(self.s, [0, 0], [-1, 180])
            self.target = tf.slice(self.s, [0, 180], [-1, 2])

            # process laser
            laser_reshape = tf.reshape(self.laser,shape=[-1, 180, 1]) 
            conv1 = tf.layers.conv1d(   inputs=laser_reshape,
                                        filters=16,
                                        kernel_size=3,
                                        padding="valid",
                                        activation=tf.nn.relu6,
                                        name = 'laser_conv1')
            conv_flat = tf.contrib.layers.flatten(conv1)
            # conv_fc = tf.layers.dense(inputs=conv_flat, units=26456, activation=tf.nn.relu6, name = 'laser_conv_fc')

            # # process laser
            # laser_reshape = tf.reshape(self.laser,shape=[-1, 180]) 
            # laser_fc1 = tf.layers.dense(inputs=laser_reshape, units=90, activation=tf.nn.relu6, name = 'laser_fc1')
            # laser_fc2 = tf.layers.dense(inputs=laser_fc1, units=45, activation=tf.nn.relu6, name = 'laser_fc2')

            target_reshape = tf.reshape(self.target,shape=[-1, 2]) 
            # target_fc = tf.layers.dense(inputs=target_reshape, units=32, activation=tf.nn.relu6, name = 'target_fc1')
            # path_fc2 = tf.layers.dense(inputs=path_fc, units=32, activation=tf.nn.relu, name = 'target_fc2')

            # concat laser and target
            feature = tf.concat([conv_flat, target_reshape], 1, name = 'concat')
            # concat_fc = tf.layers.dense(inputs=conv_fc, units=128, activation=tf.nn.relu, name = 'concat_fc1')

            #------------------------------------------------------------------------------------------------------------------------
            # feature = tf.layers.dense(inputs=self.s, units=100, activation=tf.nn.relu, name = 'concat_fc1')
            # feature = tf.layers.dense(inputs=feature, units=100, activation=tf.nn.relu, name = 'concat_fc2')

        with tf.variable_scope('actor'):
            # l_a = tf.layers.dense(target_fc, 32, tf.nn.relu6, kernel_initializer=w_init, name='actor_fc')
            a_prob = tf.layers.dense(feature, N_A, tf.nn.softmax, kernel_initializer=w_init, name='actor_prob')
        with tf.variable_scope('critic'):
            # l_c = tf.layers.dense(target_fc, 32, tf.nn.relu6, kernel_initializer=w_init, name='critic_fc')
            v = tf.layers.dense(feature, 1, kernel_initializer=w_init, name='critic_value')  # state value
        return a_prob, v

def write_summary(summary_writer, c_loss, index):
    # self.saver.save(SESS, './data/model.cptk') 
    # if self.name == 'W_0':
    summary = tf.Summary()

    summary.value.add(tag='Loss/V loss', simple_value=float(c_loss))
            
    # summary.value.add(tag='Losses/loss', simple_value=float(loss))
    # summary.histogram.add(tag='Losses/start_prob', simple_value=float(start_prob))
    summary_writer.add_summary(summary, index)
    summary_writer.flush()  


def get_ep_filename():
    global g_count, c_count, u_count
    batch_dirs = ['./batch/batch_collision/', './batch/batch_goal/', './batch/batch_unfinish/']
    ep_type = 0

    rand = np.random.rand()
    if rand < 0.33:
        ep_type = 0
        g_count += 1
    elif rand < 0.66:
        ep_type = 1
        c_count += 1
    else:
        ep_type = 2
        u_count += 1

    dir = batch_dirs[ep_type]
    list = os.listdir(dir) # dir is your directory path
    number_files = len(list)

    ep_i = np.random.randint(number_files)
    filename = dir + str(ep_i) + '.pkl'

    return filename, ep_type

# with tf.device("/gpu:0"):
AC = ACNet('AC_net')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter('./q_net_log', sess.graph)

i = 0
g_count = 0
c_count = 0
u_count = 0

while True:
    i += 1

    filename, ep_type = get_ep_filename()
    try:
        f = open(filename, 'rb')
    except:
        continue
    if f:
    # with open(filename, 'rb') as f:
        x = pickle.load(f)      

        batch_s = x[0]
        batch_a = x[1]
        batch_r = x[2]
        batch_v_real = x[3]

        batch_s, batch_a, batch_v_real = np.vstack(batch_s), np.array(batch_a), np.vstack(batch_v_real)
        feed_dict = {
            AC.s: batch_s,
            # self.AC.a_his: batch_a,
            AC.v_target: batch_v_real,
        }
        v, c_loss, _ = sess.run([AC.v, AC.c_loss, AC.train], feed_dict) 

        write_summary(summary_writer, c_loss, i)

        if i%1000 == 0:
            print (i, c_loss, batch_v_real[-1], v[-1])
            # print (g_count, c_count, u_count)
            g_count = 0
            c_count = 0
            u_count = 0

