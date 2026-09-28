"""Reproducible random CPPNs for the tests."""
import os
import random
import tempfile

import neat

TEMPLATE = os.path.join(os.path.dirname(__file__), "config_cppn_template")
BASIC_ACTIVATIONS = "gauss sin tanh"
ALL_ACTIVATIONS = "sigmoid tanh sin gauss relu softplus identity clamped abs hat square cube"


def cppn_config(num_inputs, activations=BASIC_ACTIVATIONS):
    with open(TEMPLATE) as f:
        text = f.read().replace("{num_inputs}", str(num_inputs)).replace("{activations}", activations)
    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as f:
        f.write(text)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, f.name)
    os.unlink(f.name)
    return config


def random_cppn(config, seed, mutations=20):
    """A CPPN from a fresh genome mutated `mutations` times, so it has hidden nodes."""
    state = random.getstate()
    random.seed(seed)
    genome = neat.DefaultGenome(seed)
    genome.configure_new(config.genome_config)
    for _ in range(mutations):
        genome.mutate(config.genome_config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    random.setstate(state)
    return net
