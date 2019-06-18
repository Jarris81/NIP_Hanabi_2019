import numpy as np
np.set_printoptions(threshold=np.inf)
import pyhanabi as utils
import copy

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

def color_plausible(color, player_card_knowledge, card_id, card_color_revealed, last_hand, last_player_action, last_player_card_knowledge):

    plausible = True
    print("LAST HAND")
    print(last_hand)
    if last_hand:

        if last_player_action != None:

            # print("LAST HAND AND LAST PLAYER ACTION != NONE")
            # print(last_player_action)

            if last_player_action["action_type"] == "PLAY" or last_player_action["action_type"] == "DISCARD":

                player_card_knowledge = player_card_knowledge + last_player_card_knowledge

                # print("LAST HAND CARD ID")
                # print(card_id)
                #
                # print("LAST PLAYER CARD KNOWLEDGE LENGTH")
                # print(len(last_player_card_knowledge))

                if card_id == (len(last_player_card_knowledge)-1):

                    return plausible

    color = utils.color_idx_to_char(color)
    if card_color_revealed == True :
        if color == player_card_knowledge[card_id]["color"]:
            return plausible
        else:
            plausible = False
            return plausible

    for i,card in enumerate(player_card_knowledge):

        tmp_color = card["color"]
        if (tmp_color == color):
            plausible = False

    print(f"COMPUTED THAT COLOR: {color} for card {card_id} is plausible=={plausible}")

    return plausible

def rank_plausible(rank, color, player_card_knowledge, card_id, card_rank_revealed, last_hand, last_player_action, last_player_card_knowledge):

    plausible = True
    if last_hand:
        if last_player_action != None:
            if last_player_action["action_type"] == "PLAY" or last_player_action["action_type"] == "DISCARD":

                player_card_knowledge = player_card_knowledge + last_player_card_knowledge

                if card_id == (len(last_player_card_knowledge)-1):

                    return plausible
    # print(f"rank: {rank}")
    # print(f"color: {color}")
    # print(f"player_card_knowledge: {player_card_knowledge}")

    if card_rank_revealed == True:
        print(rank)
        print(player_card_knowledge[card_id]["rank"])
        if rank == player_card_knowledge[card_id]["rank"]:
            return True
        else:
            return False

    for i,card in enumerate(player_card_knowledge):
        tmp_rank = card["rank"]
        if (card["rank"] == rank):
            plausible = False

    print(f"COMPUTED THAT RANK: {rank} for card {card_id} is plausible=={plausible}")

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
        self.num_moves = self.env.num_moves()

    def get_legal_moves_as_int(self, legal_moves):
        legal_moves_as_int = [-np.Inf for _ in range(self.num_moves)]
        tmp_legal_moves_as_int = [self.get_move_uid(move) for move in legal_moves]

        for move in tmp_legal_moves_as_int:
            legal_moves_as_int[move] = 0.0

        return [self.get_move_uid(move) for move in legal_moves]

    def get_legal_moves_as_int_formated(self,legal_moves_as_int):

        new_legal_moves = np.full(self.num_moves, -float('inf'))

        if legal_moves_as_int:
            new_legal_moves[legal_moves_as_int] = 0
        return new_legal_moves

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
            self.num_colors = self.env.game.num_colors()
            self.num_ranks = self.env.game.num_ranks()
            self.hand_size = self.env.game.hand_size()
            self.max_info_tokens = self.env.game.max_information_tokens()
            self.max_life_tokens = self.env.game.max_life_tokens()
            self.max_moves = self.env.game.max_moves()
            self.bits_per_card = self.num_colors * self.num_ranks
            self.max_deck_size = 0
            for color in range(self.num_colors):
                for rank in range(self.num_ranks):
                    self.max_deck_size += self.env.game.num_cards(color,rank)

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

    # NOTE FOR NOW JUST PARSING VERY LAST 2 MOVES
    def parse_last_moves(self,last_moves):
        print("PASE LAST MOVE FUNCTION")
        print("LAST MOVES")
        last_moves_dict = last_moves.move().to_dict()
        print(last_moves_dict)

        if last_moves_dict["action_type"] == "REVEAL_COLOR" or last_moves_dict["action_type"] == "REVEAL_RANK":
            color = utils.color_idx_to_char(last_moves.move().color())
            rank = last_moves.move().rank()

            print("COLOR AND RANK")
            print(color)
            print(rank)

            last_moves_dict.update([
                ("color",color),
                ("rank",rank)
            ])

        elif last_moves_dict["action_type"] == "PLAY" or last_moves_dict["action_type"] == "DISCARD":
            color = utils.color_idx_to_char(last_moves.color())
            rank = last_moves.rank()

            print("COLOR AND RANK")
            print(color)
            print(rank)

            last_moves_dict.update([
                ("color",color),
                ("rank",rank)
            ])

        last_moves_dict.update([
            ("player", last_moves.player()),
            ("scored", last_moves.scored()),
            ("information_token", last_moves.information_token()),
            ("card_info_revealed", last_moves.card_info_revealed())
        ])

        print("LAST MOVES DICT")
        print(last_moves_dict)
        # print(last_moves)
        # last_moves["player"] = str(last_moves.player())
        # last_moves["scored"] = last_moves.scored()
        # last_moves["information_token"] = last_moves.information_token()
        # last_moves["color"] = last_moves.color()
        # last_moves["rank"] = last_moves.rank()
        # last_moves["card_info_revealed"] = last_moves.card_info_revealed()
        last_moves_dict = [last_moves_dict]
        return last_moves_dict

    # NOTE: change back to what it was before
    def vectorize_observation(self,obs):
        print("================================")
        print("LAST MOVES IN OBSERVATION VECTORIZER AFTER CALLLING VECTORIZED FUNCTION")
        if len(obs["last_moves"]) > 2:
            print(obs["last_moves"][1])
        print("================================")
        self.obs_vec = np.zeros(self.total_state_length)
        self.obs = obs
        if obs["last_moves"] != []:

            o = copy.copy(obs)

            obs["last_moves"] = self.parse_last_moves(obs["last_moves"][0])
            print(obs["last_moves"])

            print("LAST MOVES LENGTH")
            print(len(o["last_moves"]))

            if obs["last_moves"][0]["action_type"] != "DEAL":
                print("ENTERED != DEAL")
                self.last_player_action = obs["last_moves"][0]

            elif len(o["last_moves"]) >= 2:
                print("ENTERED LEN > 2")
                obs["last_moves"] = self.parse_last_moves(o["last_moves"][1])

                print("=====================")
                print("LINE 220")
                print(f"obs['last_moves']")
                print("=====================")

                print("\nSETTING LAST PLAYER ACTION\n")
                self.last_player_action = obs["last_moves"][0]
                print("=====================")
                print("LINE 220")
                print(self.last_player_action)
                print("=====================")
            else:
                print("ENTERED ELSE")
                self.last_player_action = []

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
        #print("==================================")
        #print(hands)
        #print("==================================")
        for i,player_hand in enumerate(hands):
            if (player_hand[0]["color"] != None):
                #print(f"Player no. {i} hand: {player_hand}")
                num_cards = 0
                for card in player_hand:
                    #print(card)
                    rank = card["rank"]
                    color = utils.color_char_to_idx(card["color"])
                    card_index = color * self.num_ranks + rank
                    #print(f"CARD INDEX: {card_index}")
                    #print(f"OFFSET TO SET: {card_index+self.offset}")
                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card
                    #print(f"ENCODE HANDS LINE 237, OFFSET : {self.offset}")
                if num_cards < self.hand_size:
                    #print("ENTERED NUM_CARDS < HAND_SIZE")
                    self.offset += (self.hand_size - self.num_cards) * self.bits_per_card

        #For each player, set a bit if their hand is missing a card
        for i,player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:
                #print(f"OFFSET TO SET IN LINE 250 {self.offset+i}")
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


    # NOTE: CHANGE ACTION TYPE BACK
    def encode_last_action(self,obs):
        if self.last_player_action == []:
            print(f"Last Player Action: {self.last_player_action}")
            print(f"Setting Offset to: {self.offset+self.last_action_bit_length}")
            self.offset += self.last_action_bit_length
        else:
            print("=========================")
            print("LINE 336")
            print(f"Last Player Action: {self.last_player_action}")
            print("=========================")
            print("LAST ACTION IN OBSERVATION OBJECT")
            print(obs["last_moves"])
            print("=========================")
            last_move_type = self.last_player_action["action_type"]
            self.obs_vec[self.offset + self.last_player_action["player"]] = 1
            self.offset += self.num_players

            print(f"SELF OFFSET IN LINE 349: {self.offset}")

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
                observer_relative_target = (self.last_player_action["player"] + self.last_player_action["target_offset"]) % self.num_players
                self.obs_vec[self.offset + observer_relative_target] = 1

            self.offset += self.num_players

            if last_move_type == "REVEAL_COLOR":
                last_move_color = self.last_player_action["color"]
                self.obs_vec[self.offset + utils.color_char_to_idx(last_move_color)] = 1

            self.offset += self.num_colors

            if last_move_type == "REVEAL_RANK":
                last_move_rank = self.last_player_action["rank"]
                self.obs_vec[self.offset + last_move_rank] = 1

            self.offset += self.num_ranks

            # If multiple positions where selected
            if last_move_type == "REVEAL_COLOR" or last_move_type == "REVEAL_RANK":
                positions = self.last_player_action["card_info_revealed"]
                for pos in positions:
                    self.obs_vec[self.offset + pos] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index = self.last_player_action["card_index"]
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size

            if last_move_type == "PLAY" or last_move_type == "DISCARD":
                card_index_hgame = utils.color_char_to_idx(self.last_player_action["color"]) * self.num_ranks + self.last_player_action["rank"]
                print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == "PLAY":
                if self.last_player_action["scored"]:
                    self.obs_vec[self.offset] = 1

                ### IF INFO TOKEN WAS ADDED
                if self.last_player_action["information_token"]:
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

            print("##########################################")
            print("############### PLAYER {} ################".format(ih+1))
            print("##########################################\n")

            ### TREAT LAST HAND EDGE CASES
            if ih == self.num_players-1:
                last_hand = True
            else:
                last_hand = False

            if ih == 3:
                print("CURRENT CARD KNOWLEDGE")
                print(player_card_knowledge)
                print("LAST PLAYER CARD KNOWLEDGE")
                print(self.last_player_card_knowledge)

            for card_id, card in enumerate(player_card_knowledge):

                print("##########################################")
                print("tmp-card id: {}".format(card_id))
                print("##########################################")

                print(card)

                if card["color"] != None:
                    card_color_revealed = True
                else:
                    card_color_revealed = False
                if card["rank"] != None:
                    card_rank_revealed = True
                else:
                    card_rank_revealed = False

                for color in range(self.num_colors):

                    if color_plausible(color, player_card_knowledge, card_id, card_color_revealed, last_hand, self.last_player_action, self.last_player_card_knowledge):

                        for rank in range(self.num_ranks):

                            if rank_plausible(rank, color, player_card_knowledge, card_id, card_rank_revealed, last_hand, self.last_player_action, self.last_player_card_knowledge):

                                card_index = color * self.num_ranks + rank

                                # if ((self.offset+card_index) in error_list):
                                #     print(self.offset+card_index)
                                #     print("\nFailed encoded card: {}, with index: {}, at hand_index: {}".format(card, card_id, ih))
                                #     print("Wrongly assigned 'plausible' to color: {}, rank: {}\n".format(utils.color_idx_to_char(color),rank))
                                print("OFFSET TO BE SET IN 533")
                                print(self.offset + card_index)
                                self.obs_vec[self.offset + card_index] = 1

                self.offset += self.bits_per_card

                # Encode explicitly revealed colors and ranks
                if card["color"] != None:
                    color = utils.color_char_to_idx(card["color"])
                    print("OFFSET TO BE SET IN 542")
                    print(self.offset + color)
                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card["rank"] != None:
                    rank = card["rank"]

                    print("OFFSET TO BE SET IN 551")
                    print(self.offset + rank)

                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        self.last_player_card_knowledge = card_knowledge_list[0]

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length + self.card_knowledge_bit_length) == 0
        return True
