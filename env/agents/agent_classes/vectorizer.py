import numpy as np
np.set_printoptions(threshold=np.inf)
import pyhanabi as utils

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

def color_plausible(color, player_card_knowledge, card_id, card_color_revealed, last_hand, last_player_action, last_player_card_knowledge):

    plausible = True
    if last_hand:
        if last_player_action == "PLAY" or last_player_action == "DISCARD":
            player_card_knowledge = player_card_knowledge + last_player_card_knowledge
            # print(player_card_knowledge)
            # print("Extended Player Card Knowledge: {}\n".format(player_card_knowledge))
            if card_id == (len(last_player_card_knowledge)-1):
                return plausible

    color = utils.color_idx_to_char(color)
    # print(color)
    if card_color_revealed == True :
        if color == player_card_knowledge[card_id]["color"]:
            return plausible
        else:
            plausible = False
            return plausible
    # print("Hand Knowledge: {}".format(player_card_knowledge))

    for i,card in enumerate(player_card_knowledge):

        # print("Check our Card: {}, ID:{} against looped(plausible) Card: {}, ID:{}".format(player_card_knowledge[card_id],card_id,card,i))
        # print("Color that we are checking if plausible: {}\n".format(color))
        # print(card)
        tmp_color = card["color"]
        if (tmp_color == color):
            plausible = False

    # print("Color is plausible ? -> {}\n".format(plausible))

    return plausible

def rank_plausible(rank, color, player_card_knowledge, card_id, card_rank_revealed, last_hand, last_player_action, last_player_card_knowledge):

    plausible = True
    if last_hand:
        if last_player_action == "PLAY" or last_player_action == "DISCARD":
            player_card_knowledge = player_card_knowledge + last_player_card_knowledge
            # print("Extended Player Card Knowledge: {}\n".format(player_card_knowledge))
            if card_id == (len(player_card_knowledge)-1):
                return plausible

    if card_rank_revealed == True:
        if rank == player_card_knowledge[card_id][rank]:
            return True
        else:
            return False

    for i,card in enumerate(player_card_knowledge):
        tmp_rank = card["rank"]
        if (card["rank"] == rank):
            plausible = False
    # print("RANK IS PLAUSIBLE: {}\n".format(plausible))

    #TODO: POSSIBLE IMPROVEMENTS
    # 1. Need to encode COMMON card knowledge: What do I know, what the other player knows ?
    # 2. Infere from opponents hand if card plausible - Isn't actually used ... but can be infered from board (code below)
    # 3. Infere from Firework if card plausible - Isn't actually used ... but can be infered from board (code below)



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

class LegalMovesVectorizer(object):
    '''
    // Uid mapping.  h=hand_size, p=num_players, c=colors, r=ranks
    // 0, h-1: discard
    // h, 2h-1: play
    // 2h, 2h+(p-1)c-1: color hint
    // 2h+(p-1)c, 2h+(p-1)c+(p-1)r-1: rank hint
    '''
    def __init__(self, env):
        self.env = env
        self.num_players = self.env.game.num_players()
        self.num_ranks = self.env.game.num_ranks()
        self.num_colors = self.env.game.num_colors()
        self.hand_size = self.env.game.hand_size()
        self.max_reveal_color_moves = (self.num_players - 1) * self.num_colors

    def legal_moves_to_int(self, legal_moves):
        return [self.get_move_uid(move) for move in legal_moves]


    def get_move_uid(self, move):
        if move["action_type"] == "DISCARD":
            card_index = move["card_index"]
            return card_index

        elif move["action_type"] == "PLAY":
            card_index = move["card_index"]
            return self.hand_size + card_index

        elif move["action_type"] == "REVEAL_COLOR":
            target_offset = move["target_offset"]
            color = utils.color_char_to_idx(move["color"])
            return self.hand_size + self.hand_size + (target_offset-1) * self.num_colors + color

        elif move["action_type"] == "REVEAL_RANK":
            rank = move["rank"]
            target_offset = move["target_offset"]
            return self.hand_size + self.hand_size + self.max_reveal_color_moves + (target_offset-1) * self.num_ranks + rank
        else:
            return -1


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

            self.last_player_card_knowledge = []

            self.last_player_action = None

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
                    color = utils.color_char_to_idx(card["color"])
                    # print("Converted color {} to {}".format(card['color'],color))
                    card_index = color * self.num_ranks + rank
                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card
                if num_cards < self.hand_size:
                    print(self.num_cards)
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

    def encode_last_action(self,obs):
        if self.last_player_action == None:
            self.offset += self.last_action_bit_length
        else:
            last_move_type = self.last_player_action["MOVE_TYPE"]
            #print("Last Move Type: {}".format(last_move_type))
            self.obs_vec[self.offset + self.last_player_action["PLAYER"]] = 1
            self.offset += self.num_players

            if last_move_type == "PLAY":
                self.obs_vec[self.offset] = 1
            elif last_move_type == "DISCARD":
                self.obs_vec[self.offset + 1] = 1
            elif last_move_type == "REVEAL_COLOR":
                self.obs_vec[self.offset + 2] = 1
            elif last_move_type == "REVEAL_RANK":
                self.obs_vec[self.offset + 3] = 1
            else:
                print("ACTION UNKNOWN")
                return
            self.offset += 4

            # NEED TO COMPUTE RELATIVE OFFSET FROM CURRENT PLAYER
            if last_move_type == "REVEAL_COLOR" or last_move_type == "REVEAL_RANK":
                observer_relative_target = (self.last_player_action["PLAYER"] + self.last_player_action["TARGET_OFFSET"]) % self.num_players
                self.obs_vec[self.offset + observer_relative_target] = 1

            self.offset += self.num_players

            if last_move_type == "REVEAL_COLOR":
                last_move_color = self.last_player_action["COLOR"]
                self.obs_vec[self.offset + utils.color_char_to_idx(last_move_color)] = 1

            self.offset += self.num_colors

            if last_move_type == "REVEAL_RANK":
                last_move_rank = self.last_player_action["RANK"]
                self.obs_vec[self.offset + last_move_rank] = 1

            self.offset += self.num_ranks

            # If multiple positions where selected
            if last_move_type == "REVEAL_COLOR" or last_move_type == "REVEAL_RANK":
                positions = self.last_player_action["POSITIONS"]
                for pos in positions:
                    self.obs_vec[self.offset + pos] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index = self.last_player_action["CARD_ID"]
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index_hgame = utils.color_char_to_idx(self.last_player_action["COLOR"]) * self.num_ranks + self.last_player_action["RANK"]
                print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == "PLAY":
                if self.last_player_action["SCORED"]:
                    self.obs_vec[self.offset] = 1

                ### IF INFO TOKEN WAS ADDED
                if self.last_player_action["INFO_ADD"]:
                    self.obs_vec[self.offset + 1] = 1

            self.offset += 2

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True

    def encode_card_knowledge(self,obs):

        hands = obs["observed_hands"]
        card_knowledge_list = obs["card_knowledge"]
        discard_pile_knowledge = obs["discard_pile"]
        fireworks_knowledge = obs["fireworks"]

        for ih, player_card_knowledge in enumerate(card_knowledge_list):
            num_cards = 0

            # print("##########################################")
            # print("############### PLAYER {} ################".format(ih+1))
            # print("##########################################\n")

            ### TREAT LAST HAND EDGE CASES
            if ih == self.num_players-1:
                last_hand = True
            else:
                last_hand = False

            for card_id, card in enumerate(player_card_knowledge):

                # print("##########################################")
                # print("tmp-card id: {}".format(card_id))
                # print("##########################################")

                if card["color"] != None:
                    card_color_revealed = True
                else:
                    card_color_revealed = False
                if card["rank"] != None:
                    card_rank_revealed = True
                else:
                    card_rank_revealed = False

                for color in range(self.num_colors):

                    if color_plausible(color, player_card_knowledge, card_id, card_color_revealed, last_hand, self.last_player_action["MOVE_TYPE"], self.last_player_card_knowledge):
                    #if color_plausible(color, player_hand, hands, discard_pile_knowledge, fireworks_knowledge,self.num_colors,self.num_ranks,self.env.game.num_cards):

                        for rank in range(self.num_ranks):

                            if rank_plausible(rank, color, player_card_knowledge, card_id, card_rank_revealed, last_hand, self.last_player_action["MOVE_TYPE"], self.last_player_card_knowledge):
                            #if rank_plausible(rank, color, hand_knowledge, hands,discard_pile_knowledge, fireworks_knowledge,self.num_colors,self.num_ranks,self.env.game.num_cards):

                                card_index = color * self.num_ranks + rank

                                # if ((self.offset+card_index) in error_list):
                                #     print(self.offset+card_index)
                                #     print("\nFailed encoded card: {}, with index: {}, at hand_index: {}".format(card, card_id, ih))
                                #     print("Wrongly assigned 'plausible' to color: {}, rank: {}\n".format(utils.color_idx_to_char(color),rank))
                                self.obs_vec[self.offset + card_index] = 1

                self.offset += self.bits_per_card
                # print("Self Offset after increasing: {}".format(self.offset))

                # Encode explicitly revealed colors and ranks
                if card["color"] != None:
                    color = utils.color_char_to_idx(card["color"])
                    # print("Color: {} at card: {}, from player {} was hinted\n".format(card["color"],card_id,ih))
                    # if ((self.offset + color) in error_list):
                        # print("\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                        # print("Color: {} at card: {}, with obs_vec_index:{} from player {} was wrongly set as hinted\n".format(card["color"],card,self.offset + color,ih))

                    self.obs_vec[self.offset + color] = 1

                # print("Offset BEFORE color add: {}".format(self.offset))
                self.offset += self.num_colors
                # print("Offset AFTER color add: {}".format(self.offset))

                if card["rank"] != None:
                    rank = card["rank"]
                    # if ((self.offset + rank) in error_list):
                    #     print("\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    #     print("Rank: {} at card: {}, with obs_vec_index:{} from player 1{} was wrongly set as hinted\n".format(rank,card,self.offset + color,ih))
                    # print("Rank: {} at card: {}, from player {} was hinted".format(rank,card_id,ih))
                    self.obs_vec[self.offset + rank] = 1

                # print("Offset BEFORE rank add: {}".format(self.offset))
                self.offset += self.num_ranks
                # print("Offset AFTER rank add: {}".format(self.offset))
                num_cards += 1
                # print("Num cards: {}".format(num_cards))

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        self.last_player_card_knowledge = card_knowledge_list[0]

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length + self.card_knowledge_bit_length) == 0
        return True
