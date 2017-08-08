import threading
import multiprocessing
import os
from time import sleep
from time import time
import tensorflow as tf

from environment import env_vrep
from a3c_agent import AC_Network, Worker


max_episode_length = 40
gamma = .99 # discount rate for advantage estimation and reward discounting
load_model = False
model_path = './data'
num_workers = 4

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network(env_vrep.observation_space, env_vrep.action_space, 'global', None) # Generate global network
    # num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        env = env_vrep.Simu_env(20000 + i)
        workers.append(Worker(env, i, env_vrep.observation_space, env_vrep.action_space, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('data/log', sess.graph)

    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, summary_writer, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)