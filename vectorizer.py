import numpy as np

np.set_printoptions(threshold=np.inf)
import pyhanabi as utils

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

PLAY = 1
DISCARD = 2
REVEAL_COLOR = 3
REVEAL_RANK = 4


'''
Used to vectorize/encode player-dependent state-dicts and action dicts that are used
for agents that where trained with Hanabi Game Environment
For more details check:
GitHub Wiki - Hanabi Env Doc - Encodings
'''


'''
UTIL Classes
'''
class HandKnowledge(object):
    def __init__(self, hand_size, num_ranks, num_colors):
        self.num_ranks = num_ranks
        self.num_colors = num_colors
        self.hand = [CardKnowledge(num_ranks, num_colors) for _ in range(hand_size)]

    def sync_colors(self, card_ids, color):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for c in range(len(self.hand[cid].colors)):
                    if c != color:
                        self.hand[cid].colors[c] = None
            else:
                self.hand[cid].colors[color] = None

    def sync_ranks(self, card_ids, rank):

        for cid in range(len(self.hand)):
            if cid in card_ids:
                for r in range(len(self.hand[cid].ranks)):
                    if r != rank:
                        self.hand[cid].ranks[r] = None
            else:
                self.hand[cid].ranks[rank] = None

    def remove_card(self, card_id, deck_empty = False):
        new_hand = []
        for c_id,card in enumerate(self.hand):
            if c_id != card_id:
                new_hand.append(card)

        # NOTE: STOP REFILLING IF DECK IS EMPTY
        if not deck_empty:
            while len(new_hand) < len(self.hand):
                new_hand.append(CardKnowledge(self.num_ranks, self.num_colors))

        self.hand = new_hand

class CardKnowledge(object):
    def __init__(self, num_ranks, num_colors):
        self.colors = [c for c in range(num_colors)]
        self.ranks = [r for r in range(num_ranks)]

    def color_plausible(self, color):
        # print("\n INSIDE COLOR PLAUSIBLE FUNCTION")
        # print(self.colors[color])
        # print("\n")
        return self.colors[color] != None

    def rank_plausible(self, rank):
        # print("\n INSIDE RANK PLAUSIBLE FUNCTION")
        # print(self.ranks[rank])
        # print("\n")
        return self.ranks[rank] != None

    def remove_rank(self, rank):
        if rank in self.ranks:
            self.ranks[rank] = None

    def remove_color(self, color):
        if color in self.colors:
            self.colors[color] = None



class ObservationVectorizer(object):

    def __init__(self, env):
        '''
        Encoding Order =
         HandEncoding
        +BoardEncoding
        +DiscardEncoding
        +LastAcionEncoding
        +CardKnowledgeEncoding
        '''
        self.env = env
        self.obs = None
        self.num_players = self.env.num_players
        self.num_colors = self.env.num_colors
        self.num_ranks = self.env.num_ranks
        self.hand_size = self.env.hand_size
        self.max_info_tokens = self.env.max_information_tokens
        self.max_life_tokens = self.env.max_life_tokens
        self.max_moves = self.env.max_moves
        self.bits_per_card = self.num_colors * self.num_ranks
        self.max_deck_size = 0
        self.variant = self.env.variant
        # start of the vectorized observation
        self.offset = None

        for color in range(self.num_colors):
            for rank in range(self.num_ranks):
                self.max_deck_size += self.env.num_cards(color, rank, self.variant)
        """ Bit lengths """
        # Compute total state length
        self.hands_bit_length = (self.num_players - 1) * self.hand_size * self.bits_per_card + self.num_players
        # print("hand encoding section in range: {}-{}".format(0,self.hands_bit_length))

        self.board_bit_length = self.max_deck_size - self.num_players * \
                                self.hand_size + self.num_colors * self.num_ranks \
                                + self.max_info_tokens + self.max_life_tokens
        # print("board encoding section in range: {}-{}".format(self.hands_bit_length,self.hands_bit_length+self.board_bit_length))

        self.discard_pile_bit_length = self.max_deck_size
        # print("discard pile encoding in range: {}-{}".format(self.hands_bit_length+self.board_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length))

        self.last_action_bit_length = self.num_players + 4 + self.num_players + \
                                      self.num_colors + self.num_ranks \
                                      + self.hand_size + self.hand_size + self.bits_per_card + 2
        # print("last action encoding in range {}-{}".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length))
        self.card_knowledge_bit_length = self.num_players * self.hand_size * \
                                         (self.bits_per_card + self.num_colors + self.num_ranks)
        # print("card knowledge encoding in range: {}-{}\n".format(self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length,self.hands_bit_length+self.board_bit_length+self.discard_pile_bit_length+self.last_action_bit_length+self.card_knowledge_bit_length))
        self.total_state_length = self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length \
                                  + self.last_action_bit_length + self.card_knowledge_bit_length
        self.obs_vec = np.zeros(self.total_state_length)

        self.player_knowledge = [HandKnowledge(self.hand_size, self.num_ranks, self.num_colors) for _ in range(self.num_players)]

        self.last_player_action = None

    def get_vector_length(self):
        return self.total_state_length

    def vectorize_observation(self, obs):
        # REST OBSERVATION VECTOR
        self.obs_vec = np.zeros(self.total_state_length)
        self.obs = obs

        print("LASTMOVES", obs["last_moves"])

        if obs["last_moves"] != []:

            if obs["last_moves"][0].move().type() != "DEAL":

                self.last_player_action = obs["last_moves"][0]

                print(f"Assigned last player action {self.last_player_action}")

            else:
                self.last_player_action = obs["last_moves"][1]

                print(f"Assigned last player action {self.last_player_action}")

        self.encode_hands(obs)
        self.encode_board(obs)
        self.encode_discards(obs)
        self.encode_last_action()
        self.encode_card_knowledge(obs)

        return self.obs_vec

    def encode_hands(self, obs):
        self.offset = 0
        # don't use own hand
        hands = obs["observed_hands"]
        for player_hand in hands:
            if player_hand[0]["color"] is not None:
                num_cards = 0
                for card in player_hand:
                    rank = card["rank"]
                    color = utils.color_char_to_idx(card["color"])
                    card_index = color * self.num_ranks + rank

                    self.obs_vec[self.offset + card_index] = 1
                    num_cards += 1
                    self.offset += self.bits_per_card
                if num_cards < self.hand_size:
                    self.offset += (self.hand_size - num_cards) * self.bits_per_card

        # For each player, set a bit if their hand is missing a card
        for i, player_hand in enumerate(hands):
            if len(player_hand) < self.hand_size:
                self.obs_vec[self.offset + i] = 1
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

    def encode_discards(self, obs):
        discard_pile = obs["discard_pile"]
        counts = np.zeros(self.num_colors * self.num_ranks)
        for card in discard_pile:
            color = utils.color_char_to_idx(card["color"])
            rank = card["rank"]
            counts[color * self.num_ranks + rank] += 1

        for c in range(self.num_colors):
            for r in range(self.num_ranks):
                num_discarded = counts[c * self.num_ranks + r]
                for i in range(int(num_discarded)):
                    self.obs_vec[self.offset + i] = 1
                self.offset += self.env.num_cards(c, r, self.variant)

        assert self.offset - (self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length) == 0
        return True

    def encode_last_action(self):
        if self.last_player_action is None:
            self.offset += self.last_action_bit_length
        else:
            last_move_type = self.last_player_action.move().type()
            print("Last Move Type: {}".format(last_move_type))
            self.obs_vec[self.offset + self.last_player_action.player()] = 1
            self.offset += self.num_players

            if last_move_type == PLAY:
                self.obs_vec[self.offset] = 1
            elif last_move_type == DISCARD:
                self.obs_vec[self.offset + 1] = 1
            elif last_move_type == REVEAL_COLOR:
                self.obs_vec[self.offset + 2] = 1
            elif last_move_type == REVEAL_RANK:
                self.obs_vec[self.offset + 3] = 1
            else:
                print("ACTION UNKNOWN")
                return
            self.offset += 4

            # NEED TO COMPUTE RELATIVE OFFSET FROM CURRENT PLAYER
            if last_move_type == REVEAL_COLOR or last_move_type == REVEAL_RANK:
                observer_relative_target = (self.last_player_action.player() + self.last_player_action.move().target_offset()) % self.num_players
                self.obs_vec[self.offset + observer_relative_target] = 1

            self.offset += self.num_players

            if last_move_type == REVEAL_COLOR:
                last_move_color = self.last_player_action.move().color()
                self.obs_vec[self.offset + utils.color_char_to_idx(last_move_color)] = 1

            self.offset += self.num_colors

            if last_move_type == REVEAL_RANK:
                last_move_rank = self.last_player_action.move().rank()
                self.obs_vec[self.offset + last_move_rank] = 1

            self.offset += self.num_ranks

            # If multiple positions where selected
            if last_move_type == REVEAL_COLOR or last_move_type == REVEAL_RANK:
                print(f"LAST PLAYDER ACTIONS !!!!!{self.last_player_action}")
                positions = self.last_player_action.card_info_revealed()
                for pos in positions:
                    self.obs_vec[self.offset + pos] = 1

            self.offset += self.hand_size

            if last_move_type == PLAY or last_move_type == DISCARD:
                card_index = self.last_player_action.move().card_index()
                self.obs_vec[self.offset + card_index] = 1

            self.offset += self.hand_size

            if last_move_type == PLAY or last_move_type == DISCARD:
                card_index_hgame = utils.color_char_to_idx(self.last_player_action.move().color()) * self.num_ranks + \
                                   self.last_player_action.move().rank()
                print(self.offset + card_index_hgame)
                self.obs_vec[self.offset + card_index_hgame] = 1

            self.offset += self.bits_per_card

            if last_move_type == PLAY:
                if self.last_player_action.scored():
                    self.obs_vec[self.offset] = 1

                # IF INFO TOKEN WAS ADDED
                if self.last_player_action.information_token():
                    self.obs_vec[self.offset + 1] = 1

            self.offset += 2

        assert self.offset - (
                self.hands_bit_length + self.board_bit_length + self.discard_pile_bit_length + self.last_action_bit_length) == 0
        return True

    def encode_card_knowledge(self, obs):

        card_knowledge_list = obs["card_knowledge"]
        current_player_id = obs["current_player"]

        ### SYNC CARD KNOWLEDGE AFTER HINT GIVEN ###
        if self.last_player_action != None:
            last_move_type = self.last_player_action.move().type()
            if last_move_type == REVEAL_COLOR:
                player_hand_to_sync = (self.last_player_action.player() + self.last_player_action.move().target_offset() + current_player_id) % self.num_players
                card_pos_to_sync = self.last_player_action.card_info_revealed()
                color_to_sync = utils.color_char_to_idx(self.last_player_action.move().color())
                print("\n==============================")
                print(f"SYNCING CARD KNOWLEDGE OF PLAYER: {player_hand_to_sync}")
                print("================================\n")
                self.player_knowledge[player_hand_to_sync].sync_colors(card_pos_to_sync, color_to_sync)


        if self.last_player_action != None:
            last_move_type = self.last_player_action.move().type()
            if last_move_type == REVEAL_RANK:
                player_hand_to_sync = (self.last_player_action.player() + self.last_player_action.move().target_offset() + current_player_id) % self.num_players
                card_pos_to_sync = self.last_player_action.card_info_revealed()
                rank_to_sync = self.last_player_action.move().rank()
                print("\n==============================")
                print(f"SYNCING CARD KNOWLEDGE OF PLAYER: {player_hand_to_sync}")
                print("================================\n")
                self.player_knowledge[player_hand_to_sync].sync_ranks(card_pos_to_sync, rank_to_sync)

        if self.last_player_action != None:
            last_move_type = self.last_player_action.move().type()
            if last_move_type == PLAY or last_move_type == DISCARD:
                player_id = (self.last_player_action.player() + current_player_id) % self.num_players
                card_id = self.last_player_action.move().card_index()

                self.player_knowledge[player_id].remove_card(card_id)

        for ih, player_card_knowledge in enumerate(card_knowledge_list):
            num_cards = 0

            print("##########################################")
            print("############### TMP PLAYER CARD KNOWLEDGE ID {} ################".format(ih))
            print(player_card_knowledge)
            print("##########################################\n")

            rel_player_pos = (current_player_id + ih) % self.num_players

            for card_id, card in enumerate(player_card_knowledge):

                print("\n###########################")
                print("CARD")
                print(card)
                print("###########################\n")

                for color in range(self.num_colors):

                    if self.player_knowledge[rel_player_pos].hand[card_id].color_plausible(color):

                        for rank in range(self.num_ranks):

                            if self.player_knowledge[rel_player_pos].hand[card_id].rank_plausible(rank):

                                card_index = card_index = color * self.num_ranks + rank
                                print("\n===============================")
                                print(f"CURRENT PLAYER REAL: {current_player_id}, CURRENT PLAYER HAND: {ih}, CURRENT PLAYER RELATIVE {rel_player_pos}")
                                print(f"OFFSET TO BE SET IN 555: {self.offset + card_index}")
                                print(f"ASSIGNING PLAUSIBLE=TRUE TO CARD_ID: {card_id}, FOR COLOR: {color} AND RANK: {rank}")
                                print("SET CARD KNOWLEDGE FOR THIS HAND")
                                print("COLORS")
                                print(self.player_knowledge[rel_player_pos].hand[0].colors,self.player_knowledge[rel_player_pos].hand[1].colors,self.player_knowledge[rel_player_pos].hand[2].colors,self.player_knowledge[rel_player_pos].hand[3].colors)
                                print("RANKS")
                                print(self.player_knowledge[rel_player_pos].hand[0].ranks,self.player_knowledge[rel_player_pos].hand[1].ranks,self.player_knowledge[rel_player_pos].hand[2].ranks,self.player_knowledge[rel_player_pos].hand[3].ranks)
                                print("===============================\n")

                                self.obs_vec[self.offset + card_index] = 1

                self.offset += self.bits_per_card

                # Encode explicitly revealed colors and ranks
                if card["color"] is not None:
                    color = utils.color_char_to_idx(card["color"])

                    self.obs_vec[self.offset + color] = 1

                self.offset += self.num_colors

                if card["rank"] is not None:
                    rank = card["rank"]
                    self.obs_vec[self.offset + rank] = 1

                self.offset += self.num_ranks
                num_cards += 1

            if num_cards < self.hand_size:
                self.offset += (self.hand_size - num_cards) * (self.bits_per_card + self.num_colors + self.num_ranks)

        print(self.offset)
        assert self.offset - (
                    self.hands_bit_length +
                    self.board_bit_length +
                    self.discard_pile_bit_length +
                    self.last_action_bit_length +
                    self.card_knowledge_bit_length) == 0

        return True


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
        self.num_players = self.env.num_players
        self.num_ranks = self.env.num_ranks
        self.num_colors = self.env.num_colors
        self.hand_size = self.env.hand_size
        self.max_reveal_color_moves = (self.num_players - 1) * self.num_colors
        self.num_moves = self.env.max_moves

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
            print(move)
            return -2
