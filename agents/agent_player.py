"""Playable class used to play games with the server"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/rainbow/')
sys.path.append(rel_path)
<<<<<<< HEAD
=======
print(sys.path)
>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0

import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

import dqn_agent as dqn
<<<<<<< HEAD
import run_experiment_ui as xp
=======
import run_experiment as xp
>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0
import vectorizer
import rainbow_agent as rainbow
import functools
from third_party.dopamine import logger

class RLPlayer(object):

    def __init__(self,agent_config):

<<<<<<< HEAD
        """
        Main Interface that allows a trained agent to interact with other Hanabi-Environments
        """
=======
        """Initializes the agent and constructs its graph.
        Vars:
          observation_size: int, size of observation vector on one time step.
          history_size: int, number of time steps to stack.
          graph_template: function for building the neural network graph.
          tf_device: str, Tensorflow device on which to run computations.
        """

        # if env==None:
        #     print("Specify environment")
        #     return
        # We use the environment as a library to transform observations and actions(e.g. vectorize)
>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0

        self.observation_size = agent_config["observation_size"]
        self.players = agent_config["players"]
        self.history_size = agent_config["history_size"]
        self.vectorized_observation_shape = agent_config["observation_size"]

        self.obs_stacker = xp.create_obs_stacker(self.history_size, self.vectorized_observation_shape, self.players)
        self.num_actions = agent_config["max_moves"]
<<<<<<< HEAD
        #self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_test"
        #self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"


        self.base_dir = "/home/cawa/Documents/SoSe19/NIP/hanabi/agents/rainbow_10kit/"
        # self.base_dir = "/home/cawa/Documents/SoSe19/NIP/hanabi/"



        #self.base_dir = "/home/grinwald/Projects/TUB/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"
        self.experiment_logger = logger.Logger('{}/logs'.format(self.base_dir))

        path_rainbow = os.path.join(self.base_dir,'checkpoints')
        #print(path_rainbow)

        self.agent = xp.create_agent(self.observation_size, self.num_actions, self.players, "Rainbow")
        # print("====================")
        # print("Created Agent successfully")
        # print("====================")
        self.agent.eval_mode = True
        self.agent.partial_reload = True
        # print(self.agent.partial_reload)

        start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_rainbow,"ckpt")
        print("\n---------------------------------------------------")
        print("Initialized Model weights at start iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

=======
        self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_test"

        self.experiment_logger = logger.Logger('{}/logs'.format(self.base_dir))

        path_rainbow = os.path.join(self.base_dir,'checkpoints')
        # path_rainbow = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit/checkpoints"

        num_players = agent_config["players"]
        self.agent = xp.create_agent(self.observation_size, self.num_actions, self.players,"Rainbow")
        self.agent.eval_mode = True

        start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_rainbow,"ckpt")
        print("\n---------------------------------------------------")
        print("Creating agent from trained model at iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

    def load_model_weights(self,path,iteration_number):

        self.saver = tf.train.Saver()
        self.saver.restore(self._sess,
                            os.path.join(path,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True

>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0
    '''
    args:
        observation: expects an already vectorized observation from vectorizer.ObservationVectorizer
    returns:
<<<<<<< HEAD
        action dict object
=======
        an integer, representing the appropriate action to take
>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0
    '''

    def act(self, observation):

        # Returns Integer Action
        action = self.agent._select_action(observation["vectorized"], observation["legal_moves_as_int_formated"])
<<<<<<< HEAD
        # print(action)
=======
        print(action)
>>>>>>> 129841ad840271fc580ae63a9531f96031df82c0

        # Decode it back to dictionary object
        move_dict = observation["legal_moves"][np.where(np.equal(action,observation["legal_moves_as_int"]))[0][0]]

        return move_dict
