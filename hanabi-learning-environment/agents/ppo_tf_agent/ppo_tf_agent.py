from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
import actor_distribution_network_custom

from tf_agents.policies import greedy_policy
from tf_agents.networks import utils as net_utils
from tf_agents.utils import nest_utils

import matplotlib.pyplot as plt

tf.compat.v1.enable_v2_behavior()

import rl_env
import pyhanabi_env_wrapper

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    total_ep_len = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        ep_len = 0.0
        while not time_step.is_last():
            ep_len += 1.0
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return = time_step.reward.numpy()[0]
        total_return += -episode_return
        total_ep_len += ep_len

    avg_return = total_return / num_episodes
    avg_ep_len = total_ep_len / num_episodes
    tf_env.reset()
    return avg_return, avg_ep_len

pyhanabi_env = rl_env.make(environment_name='Hanabi-Full', num_players=2)
py_env = pyhanabi_env_wrapper.PyhanabiEnvWrapper(pyhanabi_env)
tf_env = tf_py_environment.TFPyEnvironment(py_env)

actor_net = actor_distribution_network_custom.ActorDistributionNetwork(tf_env.observation_spec(), 
    tf_env.action_spec(), fc_layer_params=[64,64], environment=py_env)

value_net = nets.value_network.ValueNetwork(
    tf_env.observation_spec(), fc_layer_params=[64,64])

tf_agent = agents.ppo.ppo_agent.PPOAgent(
    tf_env.time_step_spec(), tf_env.action_spec(), actor_net=actor_net, value_net=value_net,
    optimizer=tf.train.AdamOptimizer())


replay_buffer = TFUniformReplayBuffer(tf_agent.collect_data_spec, tf_env.batch_size)

driver = DynamicEpisodeDriver(
    tf_env, tf_agent.collect_policy, observers=[replay_buffer.add_batch])

# Evaluate the agent's policy once before training.
eval_env = tf_env
eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
avg_return, avg_ep_len = compute_avg_return(eval_env, eval_policy, 100)
returns = [avg_return]
ep_lens = [avg_ep_len]
print(returns, ep_lens)

NUM_ITERATIONS = 1000
losses_info = []
print("before loop")
for i in range(NUM_ITERATIONS):
    if((i+1) % 50 == 0):
        print(i+1, end="->", flush=True)
    driver.run()
    
    experience = replay_buffer.gather_all()
    loss_info = tf_agent.train(experience)
    replay_buffer.clear()
    losses_info.append(loss_info)

avg_return, avg_ep_len = compute_avg_return(eval_env, eval_policy, 100)
returns.append(avg_return)
ep_lens.append(avg_ep_len)

print(returns, ep_lens)

losses = list(map(lambda l: float(l[0]), losses_info))
losses_pol_grad = list(map(lambda l: float(l[1].policy_gradient_loss), losses_info))
plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
plt.plot(losses)
plt.subplot(1,2,2)
plt.plot(losses_pol_grad)
#plt.ylabel('Average Return') 
#plt.xlabel('Step')
#plt.ylim()
plt.savefig('losses_ppo.svg')