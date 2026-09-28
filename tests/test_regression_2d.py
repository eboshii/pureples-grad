"""In 2D with the published settings, ESNetworkND builds exactly the network pureples builds."""
import pytest

from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.es_hyperneat.es_hyperneat_nd import ESNetworkND
from pureples.shared.substrate import Substrate
from tests.cppns import cppn_config, random_cppn

SUBSTRATE = Substrate([(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)], [(0.0, 1.0)])

PARAMS = {"initial_depth": 1, "max_depth": 3, "variance_threshold": 0.03, "band_threshold": 0.3,
          "iteration_level": 2, "division_threshold": 0.5, "max_weight": 5.0, "activation": "sigmoid"}

PARAM_SETS = [
    PARAMS,
    {**PARAMS, "initial_depth": 2, "max_depth": 4, "division_threshold": 0.03},
    {**PARAMS, "band_threshold": 0.05, "variance_threshold": 0.1},
]

PUBLISHED = {"split_test": "variance", "cache": False, "fix_iteration": False}


class RawESNetwork(ESNetwork):
    """pureples without clean_net, so the comparison covers every connection the search finds."""

    def clean_net(self, connections):
        return set(), connections


class RawESNetworkND(ESNetworkND):
    def clean_net(self, connections):
        return set(), connections


def as_dict(conns):
    return {((c.x1, c.y1), (c.x2, c.y2)): c.weight for c in conns}


@pytest.mark.parametrize("params", PARAM_SETS)
def test_search_matches_pureples(params):
    """Every connection found by the three search phases, before cleaning."""
    config = cppn_config(5)
    nonempty = 0
    for seed in range(100):
        cppn = random_cppn(config, seed)
        expected = as_dict(RawESNetwork(SUBSTRATE, cppn, params).es_hyperneat()[1])
        got = RawESNetworkND(SUBSTRATE, cppn, {**params, **PUBLISHED}).es_hyperneat()[1]
        assert got == expected, seed
        nonempty += bool(expected)
    assert nonempty >= 30  # the comparison must cover real searches, not just empty ones


@pytest.mark.parametrize("params", [PARAM_SETS[0], PARAM_SETS[2]])
def test_cleaned_network_matches_pureples(params):
    """The final hidden nodes and connections after clean_net."""
    config = cppn_config(5)
    for seed in range(100):
        cppn = random_cppn(config, seed)
        nodes, conns = ESNetwork(SUBSTRATE, cppn, params).es_hyperneat()
        got = ESNetworkND(SUBSTRATE, cppn, {**params, **PUBLISHED}).es_hyperneat()
        assert got == (set(nodes), as_dict(conns)), seed
