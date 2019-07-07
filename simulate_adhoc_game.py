import rl_env

from agents.rainbow_adhoc_player import RainbowAdHocRLPlayer
from ppo_tf_agent.pyhanabi_env_wrapper import PyhanabiEnvWrapper

class AdHocExperiment():

    def __init__(self, game_type, agents):

        self.num_players = len(agents)
        self.game_type = game_type
        self.history_size = 1

        self.env = rl_env.make(environment_name=self.game_type, num_players=self.num_players)
        self.py_env = PyhanabiEnvWrapper(self.env)

        self.observation_size = 1041
        self.max_moves = self.env.num_moves()

        self.agents = agents

        for agent in agents:
            agent.eval_mode = True

    def run_experiment(self, num_games):
        max_reward = 0
        total_reward_over_all_ep = 0
        eval_episodes = num_games
        LENIENT_SCORE = True

        game_rewards_list = []

        # Game Loop: Simulate # eval_episodes independent games
        for ep in range(eval_episodes):

            time_step = self.py_env.reset()
            total_reward = 0
            step_number = 0

            # simulate whole game
            while not time_step.is_last():

                current_player = self.py_env._env.state.cur_player()
                action = self.agents[current_player].act(time_step)

                time_step = self.py_env.step(action)
                reward = time_step.reward

                modified_reward = max(reward, 0) if LENIENT_SCORE else reward
                total_reward += modified_reward

                if modified_reward >= 0:
                    total_reward_over_all_ep += modified_reward

                step_number += 1

                if time_step.is_last():
                    game_rewards_list.append(total_reward)
                    print("Game is done")
                    print(f"Steps taken {step_number}, Total reward: {total_reward}")
                    if max_reward < total_reward:
                        max_reward = total_reward

        print(f"Average reward over all actions: {total_reward_over_all_ep / eval_episodes}")
        print(f"Max episode reached over {eval_episodes} games: {max_reward}")

        return game_rewards_list





if __name__=="__main__":
    num_players = 4
    game_type = "Hanabi-Full"
    observation_size = 1041
    max_moves = 38
    agent_version = "custom_r1"

    agents = [RainbowAdHocRLPlayer(observation_size, num_players, max_moves, agent_version) for _ in range(num_players)]

    xp = AdHocExperiment(game_type, agents)

    num_episodes = 5

    game_rewards_list = xp.run_experiment(num_episodes)

    print(game_rewards_list)


