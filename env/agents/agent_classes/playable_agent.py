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

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where trained models are stored')

class RLPlayer(object):

    def __init__(self,agent,env,observation_size,tf_device='/cpu:*'):
        self.observation_size = observation_size
        self.stack_size = 1
        self.num_actions = env.num_moves()
        if env==None:
            print("Specify environment")


        if agent=="DQN":
            graph_template = dqn.dqn_template
            # Build DQN Network
            with tf.device(tf_device):
                # The state of the agent. The last axis is the number of past observations
                # that make up the state.
                self.states_shape = (1, self.observation_size, self.stack_size)
                self.state = np.zeros(self.states_shape)
                self.state_ph = tf.placeholder(tf.uint8, self.states_shape, name='state_ph')

                # Keeps track of legal agents performable by the player
                self.legal_actions_ph = tf.placeholder(tf.float32,
                                                       [self.num_actions],
                                                       name='legal_actions_ph')
                # Build the Network that maps State-Action Pairs to Q-Values
                net = tf.make_template('Online', graph_template)
                self._q = net(state=self.state_ph, num_actions=self.num_actions)

                # This will be used to extract the next action of the agent
                self._q_argmax = tf.argmax(self._q + self.legal_actions_ph, axis=1)[0]

                # Set up a session and initialize variables.
            #self.saver = tf.train.Saver()
            self._sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
            self._init_op = tf.global_variables_initializer()
            self._sess.run(self._init_op)


        else:
            print("Specify Agent")
            return

    def load_model_weights(self,path,iteration_number):
        self.saver.restore(self._sess,
                            os.path.join(path,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

    def select_action(self, observation, legal_actions):
        # Convert observation into a batch-based format.
        self.state[0, :, 0] = observation

        # Choose the action maximizing the q function for the current state.
        action = self._sess.run(self._q_argmax,
                                {self.state_ph: self.state,
                                 self.legal_actions_ph: legal_actions})
        assert legal_actions[action] == 0.0, 'Expected legal action.'
        return action

if __name__=="__main__":

    ### Set up the environment
    game_type = "Hanabi-Small"
    num_players = 4
    env = xp.create_environment(game_type,num_players)

    # TBD: Decide where to place this one
    obs_stacker = xp.create_obs_stacker(env)
    observation_size = obs_stacker.observation_size()

    ### Set up the RL-Player
    agent = "DQN"

    player = RLPlayer(agent=agent,env=env,observation_size=observation_size)

    path = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/test/checkpoints"
    #bool = player.load_model_weights(path,iteration_number=0)
    #print(bool)

    # Simulate 1 Move

    # Parse the current players observation to a vector
    obs_stacker.reset_stack()
    observations = env.reset()
    current_player, legal_moves, observation_vector = (xp.parse_observations(observations, env.num_moves(), obs_stacker))
    #
    action = player.select_action(observation_vector, legal_moves)
    #
    print(action)
