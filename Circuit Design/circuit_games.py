'''
circuit_games.py
Description: Circuit design game definitions for the NEAT algorithm.
Author: Drew Curran
'''

import sys
sys.path.append('C:\\Users\\drewc\\Documents\\GitHub\\COS398_CatanAI\\')

import numpy as np
from matplotlib import pyplot as plt
import argparse
import pickle

from NEAT.population import Population

### Parse the command line arguments