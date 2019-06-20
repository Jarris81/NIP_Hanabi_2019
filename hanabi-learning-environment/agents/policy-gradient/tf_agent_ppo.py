from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import tf_agents.networks as nets
import tf_agents.agents as agents
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver

import matplotlib.pyplot as plt

tf.compat.v1.enable_v2_behavior()

import rl_env


class HanabiEnv(py_environment.PyEnvironment):

    def __init__(self, game_type='Hanabi-Full', num_players=2):
        self.env = rl_env.make(
            environment_name=game_type, num_players=num_players, pyhanabi_path=None)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=int, minimum=0, maximum=self.env.num_moves()-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.env.vectorized_observation_shape(), dtype=np.int32, minimum=0, maximum=1, name='observation')
        print(self)
        self.observations = self.env.reset()
        player = self.observations['current_player']
        self._state = self.observations['player_observations'][player]['vectorized']
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    #def time_step_spec(self):

    def _reset(self):
        observations = self.env.reset()
        player = observations['current_player']
        self._state = observations['player_observations'][player]['vectorized']
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        
        player = self.observations['current_player']
        print(player)
        legal = self.observations['player_observations'][player]['legal_moves_as_int']
        print(action, legal, not int(action) in legal)
        if (not int(action) in legal):
            print("abort")
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0, discount=1.0)

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # pass action to hanabi env
        print("action taken:", int(action))
        self.observations, reward, is_done, _ = self.env.step(int(action))
        player = self.observations['current_player']
        self._state = self.observations['player_observations'][player]['vectorized']
        self._episode_ended = is_done

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(np.array(self._state, dtype=np.int32), reward=reward, discount=1.0)



def main():
    environment = HanabiEnv()
    utils.validate_py_environment(environment, episodes=1)

main()