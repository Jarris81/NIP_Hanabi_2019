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

    def act(self, observation, legal_actions):


        #TODO
        '''
        1. Build function to convert observation to vectorized version
        2. Build function to convert playable action to dict
        '''

        # Convert observation into a batch-based format.
        self.state[0, :, 0] = observation

        # Choose the action maximizing the q function for the current state.
        action = self._sess.run(self._q_argmax,
                                {self.state_ph: self.state,
                                 self.legal_actions_ph: legal_actions})

        assert legal_actions[action] == 0.0, 'Expected legal action.'
        return action

    def transform_legal_moves_int(self,observation):

        vectorized_moves = []
        for move in observation["legal_moves"]:
            vectorized_moves = vectorized_moves.append(self.env.game.get_move_uid(move))

        '''
        args:
        Observation :
        observation = {
            'current_player': self.players.index(self.agent_name),
            'current_player_offset': 0,
            'life_tokens': self.life_tokens,
            'information_tokens': self.information_tokens,
            'num_players': self.num_players,
            'deck_size': self.deck_size,
            'fireworks': self.fireworks,
            'legal_moves': self.get_legal_moves(),
            'observed_hands': self.get_sorted_hand_list(),  # moves own hand to front
            'discard_pile': self.discard_pile,
            'card_knowledge': self.get_card_knowledge(),
            'vectorized': None,  # Currently not needed, we can implement it later on demand
            'last_moves': self.last_moves  # actually not contained in the returned dict of the
            # rl_env.HanabiEnvobservation._extract_from_dict method, but we need a history so we add this here.
            # Similarly, it can be added by appending obs_dict['last_moves'] = observation.last_moves() in said method.
        }
        '''

        vectorized = self.env.ObservationEncoder.encode(observation)
        return vectorized


def get_mock_observation_mid_state_2pl():

    vectorized = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 1
    information_tokens = 8
    deck_size = 31
    current_player = 0
    current_player_offset = 0

    legal_moves_as_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 20, 22, 23, 25, 28, 29, 31, 33, 34, 36, 37]

    legal_moves = [{'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'G', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'B', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 4, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 1, 'B': 1, 'R': 0, 'W': 2, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'G', 'rank': 2}, {'color': 'G', 'rank': 2}, {'color': 'B', 'rank': 3}, {'color': 'G', 'rank': 4}, {'color': 'R', 'rank': 2}]]

    discard_pile = [{'color': 'W', 'rank': 0}, {'color': 'B', 'rank': 2}, {'color': 'B', 'rank': 4}, {'color': 'R', 'rank': 0}, {'color': 'G', 'rank': 3}]

    card_knowledge = [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]

    observation = {
        'current_player': current_player,
        'current_player_offset': current_player_offset,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': []  # actually not contained in the returned dict of th
    }

    return observation

def get_mock_observation_init_state():

    vectorized = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_players = 2
    life_tokens = 3
    information_tokens = 8
    deck_size = 40

    legal_moves_as_int = [5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18]

    legal_moves = [{'card_index': 0, 'action_type': 'PLAY'}, {'card_index': 1, 'action_type': 'PLAY'}, {'card_index': 2, 'action_type': 'PLAY'}, {'card_index': 3, 'action_type': 'PLAY'}, {'card_index': 4, 'action_type': 'PLAY'}, {'color': 'R', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'Y', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'color': 'W', 'target_offset': 1, 'action_type': 'REVEAL_COLOR'}, {'target_offset': 1, 'rank': 0, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 2, 'action_type': 'REVEAL_RANK'}, {'target_offset': 1, 'rank': 3, 'action_type': 'REVEAL_RANK'}]

    fireworks = {'Y': 0, 'B': 0, 'R': 0, 'W': 0, 'G': 0}

    observed_hands = [[{'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}, {'color': None, 'rank': -1}], [{'color': 'W', 'rank': 3}, {'color': 'R', 'rank': 2}, {'color': 'Y', 'rank': 2}, {'color': 'Y', 'rank': 0}, {'color': 'R', 'rank': 0}]]

    discard_pile = []

    card_knowledge = [[{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}], [{'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}, {'color': None, 'rank': None}]]


    observation = {
        'current_player': 0,
        'current_player_offset': 0,
        'life_tokens': life_tokens,
        'information_tokens': information_tokens,
        'num_players': num_players,
        'deck_size': deck_size,
        'fireworks': fireworks,
        'legal_moves': legal_moves,
        'observed_hands': observed_hands,  # moves own hand to front
        'discard_pile': discard_pile,
        'card_knowledge': card_knowledge,
        'vectorized': vectorized,  # Currently not needed, we can implement it later on demand
        'last_moves': []  # actually not contained in the returned dict of th
    }

    return observation

if __name__=="__main__":

    ### Set up the environment
    game_type = "Hanabi-Full"
    num_players = 2

    # env1 = xp.create_environment(game_type=game_type, num_players=num_players)
    env2 = xp.create_environment(game_type=game_type, num_players=num_players)

    # Setup Obs Stacker that keeps track of Observation for all agents ! Already includes logic for distinguishing the view between different agents
    history_size = 1
    obs_stacker = xp.create_obs_stacker(env2,history_size=history_size)
    observation_size = obs_stacker.observation_size()

    ### Set up the RL-Player, reload weights from trained model
    agent = "DQN"

    ### Specify model weights to be loaded
    path = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/dqn_sp_4pl_1000_it/playable_models"
    iteration_no = 1950

    #player = RLPlayer(agent,env,observation_size,history_size)

    # Simulate 1 Move
    # Parse the current players observation to a vector
    # obs_stacker.reset_stack()
    # print(len(encoder.obs_vec))
    #observations = env.reset()

    # mock_observation_1 = get_mock_observation_init_state()
    mock_observation_2 = get_mock_observation_mid_state_2pl()

    # obs_vectorizer1 = vectorizer.ObservationVectorizer(env1)
    obs_vectorizer2 = vectorizer.ObservationVectorizer(env2)

    # vectorized_obs_vectorizer1 = obs_vectorizer1.vectorize_observation(mock_observation_1)
    # vectorized_obs_mock1 = mock_observation_1["vectorized"]

    vectorized_obs_vectorizer2 = obs_vectorizer2.vectorize_observation(mock_observation_2)
    vectorized_obs_mock2 = mock_observation_2["vectorized"]

    # wrong_indices1 = np.where(np.equal(vectorized_obs_vectorizer1, vectorized_obs_mock1)*1 != 1)
    wrong_indices2 = np.where(np.equal(vectorized_obs_vectorizer2, vectorized_obs_mock2)*1 != 1)

    # EVALUATION
    # 1. Vector was wrong by 0 elements
    # print("Wrong indices Init State mock obs: {}".format(wrong_indices1))
    # print("Length of wrong indices: {}\n".format(wrong_indices1[0].shape))
    
    print("Wrong indices Mid State mock obs: {}".format(wrong_indices2))
    print("Length of wrong indices: {}\n".format(wrong_indices2[0].shape))

    for idx in wrong_indices2[0]:
        print("Wrongly Encoded Value at index: {} : {}".format(idx,vectorized_obs_vectorizer2[idx]))
        print("Right Value at index: {} should have been: {}\n".format(idx,vectorized_obs_mock2[idx]))
        # print("Right value should be: {}".format(vectorized_obs_mock2[int(idx)]))

    # print(mock_observation)
    # encoded = env.observation_encoder.encode(mock_observation)

    # print(observations)

    # print(env.observation_encoder.shape())

    # print(observations)
    # current_player, legal_moves, observation_vector = (xp.parse_observations(observations, env.num_moves(), obs_stacker))
    # print(observation_vector)



    # action = player.act(observation_vector, legal_moves)
    # #action = env._build_move(action.item())
    # print("Player: {}, move: {}".format(current_player,env.game.get_move(action)))



    #observation =
