from policy_gradient_agent import PolicyGradientAgent
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import reduce
from datetime import datetime

import rl_env


NUM_ITERATIONS = 100
NUM_EPISODES = 20

NUM_PLAYERS = 2


def main():
    # init agent and hanabi environment
    t0 = time.time()
    env = create_environment(num_players=NUM_PLAYERS)
    state_shape = env.vectorized_observation_shape()[0]

    agent = PolicyGradientAgent(env.num_moves(), state_shape)

    losses = []
    for it in range(NUM_ITERATIONS):
        # print progress
        if(it % 10 == 0):
            print('iteration #', it, ', time passed: ', time.time()-t0)

        # lists for storing the trajectories
        actions = [] 
        states = []
        rewards = []
        baselines = []
        logprobs = []
        ep_lens = []

        for ep in range(NUM_EPISODES):
            # run one episode
            ep_actions, ep_states, ep_rewards, ep_baselines, ep_logprobs, ep_len = run_one_episode(env, agent)
            
            # store episode data
            actions.append(ep_actions)
            states.append(ep_states)
            rewards.append(ep_rewards)
            baselines.append(ep_baselines)
            logprobs.append(ep_logprobs)
            ep_lens.append(ep_len)

        # collected a bunch of episodes, now calculate discounted rewards (Rts) and advantages (Ats) 
        Rts = calculate_discounted_rewards(rewards)
        Ats = np.array(Rts) - np.array(baselines)
        # 'flatten' the lists
        x_train = np.array(reduce(lambda x,y: np.vstack((x,y)), states))
        adv = np.array(reduce(lambda x,y: np.hstack((x,y)), Ats))
        rew = np.array(reduce(lambda x,y: np.hstack((x,y)), Rts))
        acts = np.array(reduce(lambda x,y: np.vstack((x,y)), actions))

        # normalize rewards and advantages
        adv -= adv.mean()
        adv /= adv.std() + 10**-10
        rew -= rew.mean()
        rew /= rew.std() + 10**-10
        
        # train critic and actor
        actor_loss = agent.train_actor(x_train, adv, acts)
        critic_loss = agent.train_critic(x_train, rew)
        losses.append([actor_loss, critic_loss])
    

    t1 = time.time()
    print('ran', NUM_ITERATIONS, 'iterations with', NUM_EPISODES, 'episodes each in', t1-t0, 'seconds')
    agent.save_actor_model('tmp/actor_model_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    evaluate(agent, env, 1000)
    plot_losses(losses)









def create_environment(game_type='Hanabi-Full', num_players=2):
  """Creates the Hanabi environment.

  Args:
    game_type: Type of game to play. Currently the following are supported:
      Hanabi-Full: Regular game.
      Hanabi-Small: The small version of Hanabi, with 2 cards and 2 colours.
    num_players: Int, number of players to play this game.

  Returns:
    A Hanabi environment.
  """
  return rl_env.make(
      environment_name=game_type, num_players=num_players, pyhanabi_path=None)

def run_one_episode(env, agent):
    # reset environment
    observations = env.reset()
    reward = 0
    ep_len = 0
    is_done = False
    actions = []
    states = []
    rewards = []
    baselines = []
    logprobs = []

    while(not is_done):
        # extract player's observation from observations
        cur_player = observations['current_player']
        observation = observations['player_observations'][cur_player]
        state = observation['vectorized']

        # get next action
        action, logprob = agent.act(observation, train=True)

        # store variables
        _action = np.zeros(env.num_moves())
        _action[action] = 1
        actions.append(_action)
        states.append(state)
        logprobs.append(logprob)

        # get baseline(value estimate)
        baselines.append(agent.get_value_estimate(state))

        # do next step
        observations, reward, is_done, _ = env.step(action)

        # store reward after executing action
        rewards.append(reward)

        ep_len += 1
    
    return actions, states, rewards, baselines, logprobs, ep_len

def calculate_discounted_rewards(rewards, gamma=0.95):
    Rts = []
    
    for ep in range(len(rewards)):
        Rt = 0
        T = len(rewards[ep])
        Rts.append(np.zeros(T))
        for t in reversed(range(T)):
            Rt += rewards[ep][t]
            Rts[ep][t] = Rt
            Rt *= gamma
        
    return Rts

def evaluate(agent, env, num_episodes):
    t0 = time.time()
    rewards = np.zeros(num_episodes)

    # collect episodes
    for ep in range(num_episodes):
        # reset environment
        observations = env.reset()
        reward = 0
        is_done = False

        # collect one trajectory
        while(not is_done):
            # extract player's observation from observations
            cur_player = observations['current_player']
            observation = observations['player_observations'][cur_player]

            # get next action and do next step
            action = agent.act(observation, train=False)
            observations, reward, is_done, _ = env.step(action)

        # store reward if episode finished
        rewards[ep] = -reward
    t1 = time.time()
    print('evaluated', num_episodes, 'steps in', t1-t0, 'seconds, average reward:', rewards.mean())

def plot_losses(losses):
    # invert losses to make positive=better
    neg_losses = -1*np.array(losses)
    
    #plt.figure(0, (18,20))

    plt.subplot(2,1,1)
    plt.plot(neg_losses[:,0])
    plt.title('Actor loss')

    plt.subplot(2,1,2)
    plt.plot(neg_losses[:,1])
    plt.title('Critic loss')
    
    plt.savefig('losses.svg')


if __name__== "__main__":
    main()