"""Playable class used to play games with the server"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf

import agents.rainbow.run_experiment_ui as xp
import agents.rainbow.rainbow_agent as rainbow
from agents.rainbow.rainbow_agent import rainbow_template

from agents.rainbow.third_party.dopamine import logger

class RainbowAdHocRLPlayer(object):

    def __init__(self, observation_size, num_players, max_moves, agent_version, history_size=1, layers=1, type="Rainbow"):

        tf.reset_default_graph()

        agent_config = {
            "observation_size": observation_size,
            "num_players": num_players,
            "history_size": history_size,
            "max_moves": max_moves,
            "type": type
        }

        self.observation_size = agent_config["observation_size"]
        self.num_players = agent_config["num_players"]
        self.history_size = agent_config["history_size"]
        self.vectorized_observation_shape = agent_config["observation_size"]
        self.num_actions = agent_config["max_moves"]


        if agent_config["type"] == "rainbow_10kit":
            self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"

        elif agent_config["type"] == "Rainbow":
            if agent_version == "10k":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_10kit"

            if agent_version == "20k":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_20kit"

            if agent_version == "custom_r1":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/rainbow_custom_r_discard_playable"

            if agent_version == "team1_adhoc":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/team1_adhoc"

            if agent_version == "team2_adhoc":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/team2_adhoc"

            if agent_version == "team3_adhoc":
                self.base_dir = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/agents/trained_models/team3_adhoc"

            self.experiment_logger = logger.Logger(self.base_dir+'/logs')

            self.agent = rainbow.RainbowAgent(
                observation_size=self.observation_size,
                num_actions=self.num_actions,
                num_players=self.num_players,
                num_layers=layers
                )
            path_weights = os.path.join(self.base_dir,'checkpoints')
            start_iteration, experiment_checkpointer = xp.initialize_checkpointing(self.agent,self.experiment_logger,path_weights,"ckpt")

        print("\n---------------------------------------------------")
        print("Initialized Model weights at start iteration: {}".format(start_iteration))
        print("---------------------------------------------------\n")

    def act(self, observation):

        vectorized_observation = observation.observation['state']
        legal_moves_mask = observation.observation['mask']

        # Returns Integer Action
        action = self.agent._select_action(vectorized_observation, legal_moves_mask)

        return action