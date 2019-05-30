"""Playable class used to play games with the server"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

import dqn_agent as dqn
import run_experiment as xp
import vectorizer

class RLPlayer(object):

    def __init__(self,agent,env,observation_size,history_size,tf_device='/cpu:*'):

        """Initializes the agent and constructs its graph.
        Vars:
          observation_size: int, size of observation vector on one time step.
          history_size: int, number of time steps to stack.
          graph_template: function for building the neural network graph.
          tf_device: str, Tensorflow device on which to run computations.
        """

        if env==None:
            print("Specify environment")
            return
        # We use the environment as a library to transform observations and actions(e.g. vectorize)
        self.env = env
        self.num_actions = self.env.num_moves()
        self.observation_size = observation_size
        self.history_size = history_size


        if agent=="DQN":
            graph_template = dqn.dqn_template
            # Build DQN Network
            with tf.device(tf_device):
                # The state of the agent. The last axis is the number of past observations
                # that make up the state.
                self.states_shape = (1, self.observation_size, self.history_size)
                self.state = np.zeros(self.states_shape)
                self.state_ph = tf.placeholder(tf.uint8, self.states_shape, name='state_ph')

                # Keeps track of legal agents performable by the player
                self.legal_actions_ph = tf.placeholder(tf.float32,[self.num_actions],name='legal_actions_ph')

                # Build the Graph that maps State-Action Pairs to Q-Values
                net = tf.make_template('Online', graph_template)
                self._q = net(state=self.state_ph, num_actions=self.num_actions)

                # This will be used to extract the next action of the agent
                self._q_argmax = tf.argmax(self._q + self.legal_actions_ph, axis=1)[0]

            # Set up a session and initialize variables.
            self._sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
            self._init_op = tf.global_variables_initializer()
            self._sess.run(self._init_op)

        else:
            print("Specify Agent")
            return

    def load_model_weights(self,path,iteration_number):

        self.saver = tf.train.Saver()
        self.saver.restore(self._sess,
                            os.path.join(path,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

    def act(self, observation, legal_actions_as_int):

        # Convert observation into a batch-based format.
        self.state[0, :, 0] = observation

        # Choose the action maximizing the q function for the current state.
        action = self._sess.run(self._q_argmax,
                                {self.state_ph: self.state,
                                 self.legal_actions_ph: legal_actions})

        assert legal_actions[action] == 0.0, 'Expected legal action.'
        return action

# if __name__=="__main__":
#
#     ### Set up the environment
#     game_type = "Hanabi-Full"
#     num_players = 2
#
#     env = xp.create_environment(game_type=game_type, num_players=num_players)
#
#     # Setup Obs Stacker that keeps track of Observation for all agents ! Already includes logic for distinguishing the view between different agents
#     history_size = 1
#     obs_stacker = xp.create_obs_stacker(env,history_size=history_size)
#     observation_size = obs_stacker.observation_size()
#     obs_vectorizer = vectorizer.ObservationVectorizer(env)
#
#
#     ### Set up the RL-Player, reload weights from trained model
#     agent = "DQN"
#
#     ### Specify model weights to be loaded
#     # path = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/dqn_sp_4pl_1000_it/playable_models"
#     # iteration_no = 1950
