"""A DQN episode runner using the RL environment."""

from __future__ import print_function

import sys
import getopt
import rl_env
from absl import flags
from agents.rainbow.dqn_agent import DQNAgent
from agents.rainbow import run_experiment
from dopamine.discrete_domains import logger
import tensorflow as tf

class DQNPlayer(object):
  """Runner class."""

  def __init__(self, flags):

      # with tf.Session() as sess:
      #     saver = tf.train.Saver()
      #     # Restore variables from disk.
      #     saver.restore(sess, "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/train_res_dqn_100_4pl/checkpoints/ckpt.450")
      #     print("Model restored.")

      self.environment = run_experiment.create_environment(num_players=flags["players"])
      self.obs_stacker = run_experiment.create_obs_stacker(self.environment)
      self.agent = run_experiment.create_agent(self.environment, self.obs_stacker)
      ### Load from Checkpoint ###
      self.agent._playSaver.restore(self.agent._sess, "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/train_res_dqn_100_4pl/checkpoints/playmodel/model.ckpt")
      #print(self.agent.optimizer)

  def play(self):

      """Run episodes."""
      rewards = []
      for episode in range(flags['num_episodes']):
          step_number, total_reward = run_experiment.run_one_episode(agent,environment,obs_stacker)
          rewards.append(total_reward)
      print('Total steps played: %.3f' % step_number)
      print('Total Reward: %.3f' % total_reward)

      return rewards

if __name__ == "__main__":
    flags = {'players': 4}
    player = DQNPlayer(flags)
    #agent.play()
