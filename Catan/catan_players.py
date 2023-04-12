import numpy as np

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, RESOURCES
from catanatron_gym.features import iter_players
from catanatron.state_functions import player_key

class NEATPlayer(Player):
    def __init__(self, color, player, features, is_bot=True):
        assert player.nn.num_inputs == len(features)
        assert player.nn.num_outputs == len(ACTIONS_ARRAY)

        self.color = color
        self.player = player
        self.features = features
        self.all_actions = []
        for action in ACTIONS_ARRAY:
            self.all_actions.append(Action(self.color, action[0], action[1]))
        self.is_bot = is_bot
    
    ### Decide action based on the game state
    def decide(self, game, playable_actions):
        # For efficiency, return only playable action if applicable
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Get the features of the current game state
        inputs = self.get_feature_values(game)

        # Get output from the neural network
        outputs = self.player.output(inputs)

        # If robber is in play, only possible actions are robber moves
        if playable_actions[0][1] == ActionType.MOVE_ROBBER:
            # Apply mask to output
            for i in range(len(self.all_actions)):
                if not self.all_actions[i] in playable_actions:
                    if self.all_actions[i][1] != ActionType.MOVE_ROBBER:
                        outputs[i] = 0
                    else:
                        outputs[i] = 0
        
        # Choose output with the highest value
        decision = np.argmax(outputs)
        action = self.all_actions[decision]

        # If robber is in play, find color to rob
        if action[1] == ActionType.MOVE_ROBBER:
            value = action[2]
            while action not in playable_actions:
                color = game.state.colors[np.random.randint(len(game.state.colors))]
                action[2] = (value, color, None)
        print(action)
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
              features["PLAYER_LONGEST_ROAD_DIFFERENCE"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD_DIFFERENCE"] = max_longest

        # Feature for knights and max knights
        max_longest = 0
        for i, color in iter_players(game.state.colors, self.color):
          key = player_key(game.state, color)
          longest = game.state.player_state[key + "_LONGEST_ROAD_LENGTH"]
          if color == self.color:
              features["PLAYER_LONGEST_ROAD_DIFFERENCE"] = longest
          elif longest > max_longest:
              max_longest = longest
        features["MAX_LONGEST_ROAD_DIFFERENCE"] = max_longest

        # Feature for highest producing tile for each player

        return [1] + [0] * 20