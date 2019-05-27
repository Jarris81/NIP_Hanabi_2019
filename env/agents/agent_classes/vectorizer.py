import numpy as np
np.set_printoptions(threshold=np.inf)
import pyhanabi as utils

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

### Helper Functions ###
### Here we can do more, e.g.: Is this cad already in he Discard Stack ?

# Need o add:

# Check other players cards
# Check Board
# Check Discaad Pile

def color_plausible(color, hand_knowledge, hands, discard_pile_knowledge, fireworks_knowledge, num_colors, num_ranks, num_cards):
    color = utils.color_idx_to_char(color)
    # print("Hand Knowledge: {}".format(hand_knowledge))
    # print("Color: {}".format(color))
    plausible = True
    for card_knowledge in hand_knowledge:
        if (card_knowledge["color"] == color):
            plausible = False
    # print("Color is plausible ? -> {}".format(plausible))
    return plausible

def rank_plausible(rank, color, hand_knowledge, hands, discard_pile_knowledge, fireworks_knowledge, num_colors, num_ranks, num_cards):
    plausible = True
    # print("Hand Knowledge: {}".format(hand_knowledge))
    # print("Rank: {}".format(rank))

    #TODO
    # 1. Need to encode COMMON card knowledge: What do I know, what the other player knows ?
    # 2. Infere from opponents hand if card plausible - Isn't actually used ... but can be infered from board
    # 3. Infere from Firework if card plausible - Isn't actually used ... but can be infered from board
    #
    # OWN HAND
    for card_knowledge in hand_knowledge:
        if (card_knowledge["rank"] == rank):
            plausible = False

    # OPONENTS HANDS
    # counts contains full number of cards of each color. If after iterating through fireworks, discard_pile and ops hands stil cards left -> rank plausible
    # count = num_cards(color,rank)
    #
    # color_char = utils.color_idx_to_char(color)
    #
    # print("Color we are looking at: {}".format(color_char))
    # print("Rank we are looking at: {}".format(rank))

    # # iterate over each opponents hand
    # for hand in hands:
    #     for card in hand:
    #         tmp_card = card
    #         if tmp_card["color"] == color_char:
    #             tmp_color = utils.color_char_to_idx(tmp_card["color"])
    #             tmp_rank = tmp_card["rank"]
    #             if tmp_rank == rank:
    #                 print("card {}".format(tmp_card))
    #                 count -= 1
    #                 if count <= 0:
    #                     plausible = False
    #
    # print("Rank plausible ? -> {}\n".format(plausible))
    #
    # # Iterate over discard pile
    # for card in discard_pile_knowledge:
    #     tmp_card = card
    #     if tmp_card["color"] == color_char:
    #         tmp_color = utils.color_char_to_idx(tmp_card["color"])
    #         tmp_rank = tmp_card["rank"]
    #         if tmp_rank == rank:
    #             print("card {}".format(tmp_card))
    #             count -= 1
    #             if count <= 0:
    #                 plausible = False
    # print("Rank plausible ? -> {}\n".format(plausible))


    return plausible

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
            # print("num ranks: {}".format(self.num_ranks))
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
            print("hand encoding section in range: {}-{}".format(0,self.hands_bit_length))

            self.board_bit_length = self.max_deck_size - self.num_players * \
                            self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens
            print("board encoding section in range: {}-{}".format(self.hands_bit_length,self.hands_bit_length+self.board_bit_length))

            self.discard_pile_bit_length = self.max_deck_size
            print("discard pile encoding in range: {}-{}".format(self.hands_bit_length+self.board_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length))

            self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                            self.num_colors + self.num_ranks \
                                + self.hand_size + self.hand_size + self.bits_per_card + 2
            print("last action encoding in range {}-{}".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length))
            self.card_knowledge_bit_length = self.num_players * self.hand_size *\
                            (self.bits_per_card + self.num_colors + self.num_ranks)
            print("card knowledge encoding in range: {}-{}\n".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length+self.card_knowledge_bit_length))
            self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                    + self.last_action_bit_length + self.card_knowledge_bit_length
            self.obs_vec = np.zeros(self.total_state_length)

    def get_vector_length(self):
        return self.total_state_length

    def vectorize_observation(self,obs):
        self.obs = obs

        self.encode_hands(obs)
        self.encode_board(obs)
        self.encode_discards(obs)
        self.encode_last_action(obs)
        self.encode_card_knowledge(obs)

        return self.obs_vec

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
            # print(fireworks[color])
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

    def encode_discards(self,obs):
        discard_pile = obs["discard_pile"]
        counts = np.zeros(self.num_colors*self.num_ranks)
        for card in discard_pile:
            color = utils.color_char_to_idx(card["color"])
            rank = card["rank"]
            counts[color*self.num_ranks + rank] += 1

        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                num_discarded = counts[c*self.num_ranks+r]
                for i in range(int(num_discarded)):
                    self.obs_vec[self.offset+i] = 1
                self.offset += self.env.game.num_cards(c,r)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length) == 0
        return True

    # TODO finish last moves
    def encode_last_action(self,obs):
        last_moves = obs["last_moves"]
        if last_moves == []:
            self.offset += self.last_action_bit_length
        else:
            print("TODO LAST ACTION ENCODING")

        # print("Actual length before card_knowledge encoding: {}".format(self.offset))

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True

    def encode_card_knowledge(self,obs):
        hands = obs["observed_hands"]
        card_knowledge_list = obs["card_knowledge"]
        discard_pile_knowledge = obs["discard_pile"]
        fireworks_knowledge = obs["fireworks"]

        # print("Card Knowledge: {}\n".format(card_knowledge_list))
        num_cards = 0

        #### REMOVE AFTER DEBUGGING ####
        error_list = [254, 255, 278, 286, 306, 488, 489, 490, 491, 492, 498, 499, 500,
       501, 502, 503, 504, 505, 506, 507, 523, 524, 525, 526, 527]


        for ih,hand_knowledge in enumerate(card_knowledge_list):
            # print("Card Knowledge List: {}".format(card_knowledge_list))
            print("Player index: {}".format(ih))
            for card_knowledge in hand_knowledge:
                # print("Card Knowledge: {}".format(card_knowledge))
                for color in range(self.num_colors):
                    if color_plausible(color, hand_knowledge,hands,discard_pile_knowledge,fireworks_knowledge,self.num_colors,self.num_ranks,self.env.game.num_cards):
                        for rank in range(self.num_ranks):
                            if rank_plausible(rank, color, hand_knowledge,hands,discard_pile_knowledge,fireworks_knowledge,self.num_colors,self.num_ranks,self.env.game.num_cards):
                                # print("Color: {}".format(color))
                                # print("Rank: {}".format(rank))
                                card_index = color * self.num_ranks + rank
                                # print("Card Index: {}".format(card_index))
                                if ((self.offset+card_index) in error_list):
                                    print(self.offset+card_index)
                                    print("Failed encoded card: {}, with index: {}, at hand_index: {}".format(card_knowledge, card_index, ih))
                                    print("Wrongly assigned 'plausible' to color: {}, rank: {}".format(utils.color_idx_to_char(color),rank))
                                # if (ih==1):
                                #     print(card)
                                self.obs_vec[self.offset + card_index] = 1
                                # print("changed obs_vec")

                # print("Self Offset before increasing: {}".format(self.offset))
                self.offset += self.bits_per_card
                # print("Self Offset after increasing: {}".format(self.offset))

                # Encode explicitly revealed colors and ranks
                if card_knowledge["color"] != None:
                    color = utils.color_char_to_idx(card_knowledge["color"])
                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card_knowledge["rank"] != None:
                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1
                # print("Num cards: {}".format(num_cards))

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length + self.card_knowledge_bit_length) == 0
        return True
