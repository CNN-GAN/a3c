"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [http://adsabs.harvard.edu/abs/2017arXiv170702286H]
2. Proximal Policy Optimization Algorithms (OpenAI): [http://adsabs.harvard.edu/abs/2017arXiv170706347S]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal, kl_divergence
import numpy as np
import matplotlib.pyplot as plt
import gym

from environment import env_hlc

EP_MAX = 1000
EP_LEN = 150
GAMMA = 0.9
A_LR = 0.001
C_LR = 0.001
BATCH = 50
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self, s_dim, a_dim,):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')

        # actor
        pi, pi_params, feature = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, old_feature = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('surrogate'):
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
            surr = ratio * self.tfadv
        if METHOD['name'] == 'kl_pen':
            self.tflam = tf.placeholder(tf.float32, None, 'lambda')
            with tf.variable_scope('loss'):
                kl = tf.stop_gradient(kl_divergence(oldpi, pi))
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
        else:   # clipping method, find this is better
            with tf.variable_scope('loss'):
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter('data/log', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=5)

        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state('./data/')
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print ('loaded')
        else:
            print ('no model file')

    def update(self, s, a, r, m=20, b=10):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(m):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # some time explode, this is my method
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(m)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(b)]
        self.saver.save(self.sess, './data/model.cptk') 

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # self.laser = tf.slice(self.tfs, [0, 0], [-1, 180])
            # self.target = tf.slice(self.tfs, [0, 180], [-1, 2])

            # laser_reshape = tf.reshape(self.laser,shape=[-1, 180, 1]) 
            # conv1 = tf.layers.conv1d(   inputs=laser_reshape,
            #                             filters=150,
            #                             kernel_size=3,
            #                             padding="valid",
            #                             activation=tf.nn.relu,
            #                             name = 'laser_conv1')
            # conv_flat = tf.contrib.layers.flatten(conv1)

            # laser_reshape = tf.reshape(self.tfs,shape=[-1, 180]) 
            feature = tf.layers.dense(inputs=self.tfs, units=100, activation=tf.nn.relu, name = 'laser_conv_fc')
            # feature = tf.layers.dense(inputs=conv_fc, units=50, activation=tf.nn.relu, name = 'laser_conv_fc2')

            # target_reshape = tf.reshape(self.target,shape=[-1, 2]) 
            # target_fc = tf.layers.dense(inputs=target_reshape, units=15, activation=tf.nn.relu, name = 'target_fc1')

            # concat laser and target
            # concat_feature = tf.concat([conv_fc, target_fc], 1, name = 'concat')
            # concat_fc = tf.layers.dense(inputs=conv_fc, units=128, activation=tf.nn.relu, name = 'concat_fc1')

            mu = 2 * tf.layers.dense(feature, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(feature, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, feature

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = env_hlc.Simu_env(20000) #gym.make('Pendulum-v0').unwrapped
N_S = env_hlc.observation_space
N_A = env_hlc.action_space

ppo = PPO(s_dim=N_S, a_dim=N_A)
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(1, EP_LEN):    # one episode4
        # env.render()
        a = ppo.choose_action(s)
        action = [0,0,0,0,0]
        action[0] = a[0]
        action[1] = a[1]
        s_, r, done, _ = env.step(action)
        # print (action)
        # print(r, ep_r)

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if t % (BATCH-1) == 0 or t == EP_LEN-1:
        # if t == EP_LEN-1 or done:
            # print('update')
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br, m=A_UPDATE_STEPS, b=C_UPDATE_STEPS)

        # if done:
        #     break

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()