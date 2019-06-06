from rl_env import Agent
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import softmax
import time
from functools import reduce

import tensorflow as tf
from tensorflow import keras


class PolicyGradientAgent(Agent):
    """Agent that takes random legal actions."""

    def __init__(self, num_actions, state_shape, 
                gamma=0.99, num_units_actor=256, num_units_critic=256, eps_actor=0.001, eps_critic=0.001):
        """Initialize the agent."""
        self.action_probs, self.x, self.train, self.loss, self.advantage, self.actions = self.create_actor_net(
            num_actions, state_shape, num_units_actor, eps_actor)
        self.critic = self.create_critic_net(state_shape, num_units_critic, eps_critic)

        # init tf session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)



    def act(self, observation, train=False):
        if observation['current_player_offset'] != 0:
            return None
        
        state = observation['vectorized']
        legal_moves = observation['legal_moves_as_int']

        policy = self.sess.run(self.action_probs, {self.x: [state]}).squeeze()
        # only consider the legal actions
        policy_legal = np.full(policy.shape, -np.inf)
        policy_legal[legal_moves] = policy[legal_moves]
        policy_legal = softmax(policy_legal)
        
        action = np.random.choice(policy_legal.shape[0], p=policy_legal)
        logprob = np.log(policy_legal[action])
        
        if (train):
            return action, logprob
        else: 
            return action 

    def get_value_estimate(self, state):
        return self.critic.predict(np.reshape(state, (1,-1))).squeeze()

    def create_critic_net(self, state_shape, num_units, eps):
        model = keras.Sequential([
            keras.layers.Dense(num_units, input_dim=state_shape, activation=tf.nn.relu),
            keras.layers.Dense(num_units, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        opt = keras.optimizers.Adam(eps)
        model.compile(loss='mean_squared_error',
                    optimizer=opt,
                    metrics=['mean_squared_error'])
        return model

    def create_actor_net(self, num_actions, state_shape, num_units, eps):
        x = tf.placeholder(tf.float32, shape=[None,state_shape], name='x')
        actions = tf.placeholder(tf.float32, shape=[None,num_actions], name='actions')
        
        h1 = tf.layers.dense(x, units=num_units, activation=tf.nn.relu, name='h1')
        h2 = tf.layers.dense(h1, units=num_units, activation=tf.nn.relu, name='h2')
        logits = tf.layers.dense(h2, units=num_actions, activation=None, name='logits')
        y = tf.nn.softmax(logits, name='y')
        
        negative_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions, logits=logits, name='neg_log_prob')
        #negative_log_prob = tf.losses.softmax_cross_entropy(onehot_labels=actions, logits=logits)
        #negative_log_prob = -tf.log(y)
        
        advantage = tf.placeholder(tf.float32, shape=[None,], name='advantage')
        
        loss = tf.reduce_mean(negative_log_prob*advantage, name='loss')#tf.reduce_mean(tf.multiply(negative_log_prob, advantage))
        optimizer = tf.train.AdamOptimizer(eps)
        train = optimizer.minimize(loss, name='train')
        
        return y, x, train, loss, advantage, actions

    def train_actor(self, x_train, advantages, actions):
        _, loss = self.sess.run((self.train, self.loss), {self.x: x_train, self.advantage: advantages, self.actions: actions})
        return loss

    def train_critic(self, x_train, rewards):
        loss, _ = self.critic.train_on_batch(x_train, rewards)
        return loss

    def save_actor_model(self, name='actor_model'):
        self.saver.save(self.sess, name)

    def load_actor_model(self, name='actor_model'):
        self.saver = tf.train.import_meta_graph(name+'.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        
        self.x = graph.get_tensor_by_name('x:0')
        self.actions = graph.get_tensor_by_name('actions:0')
        self.action_probs = graph.get_tensor_by_name('y:0')
        self.advantage = graph.get_tensor_by_name('advantage:0')
        self.loss = graph.get_tensor_by_name('loss:0')
        self.train = graph.get_tensor_by_name('train:0')
