from catanatron import Game, RandomPlayer, Color
from catan_players import NEATPlayer

players = [RandomPlayer(Color.RED), NEATPlayer(Color.BLUE), RandomPlayer(Color.WHITE), RandomPlayer(Color.ORANGE)]
game = Game(players)

print(game.play())  # returns winning color