import random

from catanatron.state_functions import (
    player_key,
)
from catanatron.models.player import Player
from catanatron.game import Game


class NEATPlayer(Player):
    """
    Should return one of the playable_actions.

    Args:
        game (Game): complete game state. read-only.
        playable_actions (Iterable[Action]): options to choose from
    Return:
        action (Action): Chosen element of playable_actions
    """

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            key = player_key(game_copy.state, self.color)
            value = game_copy.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_value = value
                best_actions = [action]

        return random.choice(best_actions)
