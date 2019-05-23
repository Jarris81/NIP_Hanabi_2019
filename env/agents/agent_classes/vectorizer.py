import numpy as np
np.set_printoptions(threshold=np.inf)
import pyhanabi as utils


COLOR_CHAR = ["R", "Y", "G", "W", "B"]

'''
Used to vectorize/encode player-dependent state-dicts and action dicts that are used
for agents that where trained with Hanabi Game Environment
For more details check:
GitHub Wiki - Hanabi Env Doc - Encodings
'''

class ObservationVectorizer(object):
    def __init__(self,env):
            '''
            Encoding Order =
             HandEncoding
            +BoardEncoding
            +DiscardEncoding
            +LastAcionEncoding
            +CardKnowledgeEncoding
            '''
            self.env = env

            self.num_players = self.env.game.num_players()
            # print("num players: {}".format(num_players))
            self.num_colors = self.env.game.num_colors()
            # print("num colors: {}".format(num_colors))
            self.num_ranks = self.env.game.num_ranks()
            # print("num ranks: {}".format(num_ranks))
            self.hand_size = self.env.game.hand_size()
            # print("hand_size: {}".format(hand_size))
            self.max_info_tokens = self.env.game.max_information_tokens()
            # print("max info: {}".format(max_info_tokens))
            self.max_life_tokens = self.env.game.max_life_tokens()
            # print("max life: {}".format(max_life_tokens))
            self.max_moves = self.env.game.max_moves()
            # print("max moves: {}".format(max_moves))
            self.bits_per_card = self.num_colors * self.num_ranks
            # print("bits per card {}".format(bits_per_card))
            self.max_deck_size = 0
            for color in range(self.num_colors):
                for rank in range(self.num_ranks):
                    self.max_deck_size += self.env.game.num_cards(color,rank)
            # print("max deck size: {}".format(max_deck_size))

            # Compute total state length
            self.hands_bit_length = (self.num_players - 1) * self.hand_size * \
                            self.bits_per_card + self.num_players

            self.board_bit_length = self.max_deck_size - self.num_players * \
                            self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens

            self.discard_pile_bit_length = self.max_deck_size

            self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                            self.num_colors + self.num_ranks \
                                + self.hand_size + self.hand_size + self.bits_per_card + 2

            self.card_knowledge_bit_length = self.num_players * self.hand_size *\
                            (self.bits_per_card + self.num_colors + self.num_ranks)

            self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                    + self.last_action_bit_length + self.card_knowledge_bit_length
            self.obs_vec = np.zeros(self.total_state_length)

    def get_vector_length(self):
        return self.total_state_length

    def vectorize_observation(self,obs):
        self.obs = obs

        self.encode_hands(obs)
        self.encode_board(obs)

    def encode_hands(self,obs):
        # start of the vectorized observation
        self.offset = 0
        # don't use own hand
        hands = obs["observed_hands"]
        for player_hand in hands:
            if (player_hand[0]["color"] != None):
                num_cards = 0
                for card in player_hand:
                    rank = card["rank"]
                    color = utils.color_char_to_idx(card["color"]) # Order: COLOR_CHAR = ["R", "Y", "G", "W", "B"]
                    # print("Converted color {} to {}".format(card['color'],color))
                    card_index = color * self.num_ranks + rank
                    self.obs_vec[self.offset + card_index] = 1
                    ### Test, first mock Blue card with rank 0 poins to index: 4*num_ranks = 20
                    assert self.obs_vec[20] == 1
                    num_cards += 1
                    self.offset += self.bits_per_card
                if num_cards < self.hand_size:
                    self.offset += (self.hand_size - self.num_cards) * self.bits_per_card

        #For each player, set a bit if their hand is missing a card
        for i,player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:
                self.obs_vec[self.offset+i] = 1
            self.offset += 1

        assert self.offset - self.hands_bit_length == 0
        return True

    def encode_board(self, obs):
        # encode the deck size:
        for i in range(obs["deck_size"]):
            self.obs_vec[self.offset + i] = 1
        self.offset += self.max_deck_size - self.hand_size * self.num_players

        # encode fireworks
        fireworks = obs["fireworks"]
        for c in range(len(fireworks)):
            color = utils.color_idx_to_char(c)
            print(fireworks[color])
            if fireworks[color] > 0:
                self.obs_vec[self.offset + fireworks[color] - 1] = 1
            self.offset += self.num_ranks

        # encode info tokens
        info_tokens = obs["information_tokens"]
        for t in range(info_tokens):
            self.obs_vec[self.offset + t] = 1
        self.offset += self.max_info_tokens

        # encode life tokens
        life_tokens = obs["life_tokens"]
        for l in range(life_tokens):
            self.obs_vec[self.offset + l] = 1
        self.offset += self.max_life_tokens

        assert self.offset - (self.hands_bit_length + self.board_bit_length) == 0
        return True
