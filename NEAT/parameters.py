'''
parameters.py
Description: Parameters specified for algorithm
Author: Drew Curran
'''

import numpy as np

# Restrictions
MAX_WEIGHT = 1.0

# Reproduction
PR_CLONE = 0.25
PR_INHERIT_FITTER = 0.5
PR_ENABLE = 0.25

# Mutation
PR_MUTATE_NEURON = 0.01
PR_MUTATE_GENE = 0.1
PR_MUTATE_WEIGHTS = 0.8
PR_WEIGHT_RANDOM = 0.1
WEIGHT_PERTURB = 0.02

# Speciation
EXCESS_DISJOINT_COEFF = 1.0
WEIGHT_DIFF_COEFF = 2.0
GENETIC_DISTANCE = 3.0
GENE_OFFSET = 20
DISTANCE_MODIFIER = 0.1
GENERATION_GRACE = 5
SPECIES_WANTED = 20
SPECIES_BUFFER = 2

# Evaluation
LEAKY_RELU_SCALE = 0.1
SIGMOID_SCALE = 1.0

# Selection
SPECIES_CULL_RATE = 0.5
MAX_STALENESS = 15

def leaky_relu(value):
    return value if value > 0 else LEAKY_RELU_SCALE * value

def sigmoid(value):
    return 1 / (1 + np.exp(-value))