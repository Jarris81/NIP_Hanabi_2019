import os
import sys

rel_path = os.path.join(os.environ['PYTHONPATH'],'agents/')
sys.path.append(rel_path)

import numpy as np
import agents.rainbow.run_experiment as xp
import agents.pyhanabi_vectorizer as vectorizer
from agents.rainbow_adhoc_player import RainbowAdHocRLPlayer

if __name__=="__main__":

    # Game-simulation Parameters
    max_reward = 0
    total_reward_over_all_ep = 0
    eval_episodes = 20
    LENIENT_SCORE = False

    game_type = "Hanabi-Full"
    num_players = 4
    history_size = 1
    observation_size = 1041

    # Simulation objects
    env = xp.create_environment(game_type=game_type, num_players=num_players)
    obs_stacker = xp.create_obs_stacker(env, history_size)
    moves_vectorizer = vectorizer.LegalMovesVectorizer(env)
    max_moves = env.num_moves()

    ### Reinforcement Learning Agent Player
    agents = [
                RainbowAdHocRLPlayer(observation_size, num_players, history_size, max_moves, type="Rainbow", version="custom_r1"),
                RainbowAdHocRLPlayer(observation_size, num_players, history_size, max_moves, type="Rainbow", version="custom_r1"),
                RainbowAdHocRLPlayer(observation_size, num_players, history_size, max_moves, type="Rainbow", version="custom_r1"),
                RainbowAdHocRLPlayer(observation_size, num_players, history_size, max_moves, type="Rainbow", version="custom_r1")
              ]

    for agent in agents:
        agent.eval_mode = True

    # Game Loop: Simulate # eval_episodes independent games
    for ep in range(eval_episodes):

        observations = env.reset()

        current_player, legal_moves, observation_vector  = xp.parse_observations(observations, env.num_moves(), obs_stacker)
        current_player_observation = observations["player_observations"][current_player]
        current_player_observation["legal_moves_as_int_formated"] = moves_vectorizer.get_legal_moves_as_int_formated(current_player_observation["legal_moves_as_int"])

        action = agents[current_player].act(current_player_observation)

        is_done = False
        total_reward = 0
        step_number = 0

        has_played = {current_player}

        # Keep track of per-player reward.
        reward_since_last_action = np.zeros(env.players)

        # simulate whole game
        while not is_done:

            observations, reward, is_done, _ = env.step(action.item())

            modified_reward = max(reward, 0) if LENIENT_SCORE else reward
            total_reward += modified_reward

            reward_since_last_action += modified_reward
            if modified_reward >= 0:
                total_reward_over_all_ep += modified_reward

            step_number += 1

            if is_done:
                print("Game is done")
                print(f"Steps taken {step_number}, Total reward: {total_reward}")
                if max_reward < total_reward:
                    max_reward = total_reward

            current_player, legal_moves, observation_vector_d  = xp.parse_observations(observations, env.num_moves(), obs_stacker)
            current_player_observation = observations["player_observations"][current_player]
            current_player_observation["legal_moves_as_int_formated"] = moves_vectorizer.get_legal_moves_as_int_formated(current_player_observation["legal_moves_as_int"])

            if current_player in has_played:
              action = agents[current_player].act(current_player_observation)

            else:
              action = agents[current_player].act(current_player_observation)
              has_played.add(current_player)

            # Reset this player's reward accumulator.
            reward_since_last_action[current_player] = 0

    print(f"Average reward over all actions: {total_reward_over_all_ep/eval_episodes}")
    print(f"Max episode reached over {eval_episodes} games: {max_reward}")

    #agents[current_player].end_episode(reward_since_last_action)
