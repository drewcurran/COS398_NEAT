import numpy as np

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, RESOURCES
from catanatron_gym.features import iter_players
from catanatron.state_functions import player_key

class NEATPlayer(Player):
    def __init__(self, color, player, features, is_bot=True):
        self.color = color
        self.player = player
        self.features = features
        self.actions = []
        self.robber_actions = {}
        for action in ACTIONS_ARRAY:
            if action[0] != ActionType.MOVE_ROBBER:
                self.actions.append(Action(self.color, action[0], action[1]))
            else:
                self.robber_actions[action[1]] = []
        self.is_bot = is_bot

        assert player.nn.num_inputs == len(features)
        assert player.nn.num_outputs == len(self.actions) + len(self.robber_actions)
    
    ### Decide action based on the game state
    def decide(self, game, playable_actions):
        # For efficiency, return only playable action if applicable
        if len(playable_actions) == 1:
            return playable_actions[0]
        
        # If robber is in play, only possible actions are robber moves
        if playable_actions[0][1] == ActionType.MOVE_ROBBER:
            robber_moves_only = True
        else:
            robber_moves_only = False
        
        # Find all the possible robber moves if applicable
        if robber_moves_only:
            for action in playable_actions:
                tile = action[2][0]
                if action not in self.robber_actions[tile]:
                    if len(self.robber_actions[tile]) == 1 and self.robber_actions[tile][0][2][1] is None:
                        self.robber_actions[tile] = []
                    self.robber_actions[tile].append(action)

        # Get the features of the current game state
        inputs = self.get_feature_values(game)

        # Get output from the neural network
        outputs = self.player.output(inputs)
        
        # Apply mask to output
        robber_actions = list(self.robber_actions.values())
        for i in range(len(self.actions) + len(self.robber_actions)):
            if i < len(self.actions):
                if not self.actions[i] in playable_actions:
                    outputs[i] = 0
            else:
                possible_robber_actions = robber_actions[i - len(self.actions)]
                if len(possible_robber_actions) == 0 or not possible_robber_actions[0] in playable_actions:
                    outputs[i] = 0
        
        # Choose output with the highest value
        decision = np.argmax(outputs)
        if decision < len(self.actions):
            action = self.actions[decision]
        else:
            possible_actions = list(self.robber_actions.values())[decision - len(self.actions)]
            action = possible_actions[np.random.randint(len(possible_actions))]
        
        print("Decision: %.3d, Confidence: %.4f, Action: %s" % (decision, np.max(outputs), action))
        return action
    
    ### Get the features of the current game state
    def get_feature_values(self, game):
        features = {}
        pkey = player_key(game.state, self.color)
        
        # Feature for each resource amount in hand
        for resource in RESOURCES:
            features[f"PLAYER_{resource}_IN_HAND"] = game.state.player_state[pkey + f"_{resource}_IN_HAND"]

        # Feature for longest road and max longest road
        max_longest = 0
        for i, color in iter_players(game.state.colors, self.color):
          key = player_key(game.state, color)
          longest = game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]
          if color == self.color:
              features["PLAYER_LONGEST_ROAD"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD"] = max_longest

        # Feature for army and max army
        max_army = 0
        for i, color in iter_players(game.state.colors, self.color):
          key = player_key(game.state, color)
          army = game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]
          if color == self.color:
              features["PLAYER_LONGEST_ROAD_DIFFERENCE"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD_DIFFERENCE"] = max_army

        # Feature for highest producing tile for each player

        return [1] + [0] * 20