"""n-D trees, split tests, the feed-forward axis and the iteration fix."""
import itertools

import numpy as np
import pytest

from pureples.es_hyperneat.es_hyperneat_nd import SPLIT_TESTS, ESNetworkND
from pureples.shared.substrate import Substrate
from tests.cppns import cppn_config, random_cppn

PARAMS = {"initial_depth": 1, "max_depth": 3, "variance_threshold": 0.03, "band_threshold": 0.3,
          "iteration_level": 1, "division_threshold": 0.5, "max_weight": 5.0, "activation": "sigmoid"}


def substrate(n):
    """Inputs on the y = -1 face (a grid of +-0.5 on the other axes), one output at y = 1."""
    ins = [c[:1] + (-1.0,) + c[1:] for c in itertools.product((-0.5, 0.5), repeat=n - 1)]
    outs = [tuple(1.0 if i == 1 else 0.0 for i in range(n))]
    return Substrate(ins, outs)


def count_cells(root):
    return 1 + sum(count_cells(c) for c in root.cs) if root.cs else 1


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("split_test", SPLIT_TESTS)
def test_runs_and_is_feedforward(n, split_test):
    config = cppn_config(2 * n + 1)
    # Keep 4D small: every hidden node found starts its own 16-way tree.
    params = {**PARAMS, "split_test": split_test}
    if n == 4:
        params.update(max_depth=2, iteration_level=0)
    for seed in range(3):
        net = ESNetworkND(substrate(n), random_cppn(config, seed), params)
        nodes, conns = net.es_hyperneat()
        for src, tgt in conns:
            assert len(src) == len(tgt) == n
            assert src[1] < tgt[1]
        net.create_phenotype_network()


@pytest.mark.parametrize("n", [2, 3, 4])
def test_children_tile_the_parent(n):
    config = cppn_config(2 * n + 1)
    net = ESNetworkND(substrate(n), random_cppn(config, 0), {**PARAMS, "initial_depth": 2, "max_depth": 2})
    root = net.division_initialization((0.0,) * n, True)
    assert len(root.cs) == 2 ** n
    assert count_cells(root) == 1 + 2 ** n + 4 ** n
    centres = {c.centre for c in root.cs}
    assert centres == set(itertools.product((-0.5, 0.5), repeat=n))


def test_feedforward_axis():
    config = cppn_config(7)
    net = ESNetworkND(substrate(3), random_cppn(config, 1), {**PARAMS, "feedforward_axis": 2,
                                                             "band_threshold": 0.05})
    for src, tgt in net.es_hyperneat()[1]:
        assert src[2] < tgt[2]


def test_gradient_score_approximates_variance_on_small_cells():
    """To first order, Var over the 2^n child centres = (r/2)^2 * |grad w|^2, so the ratio tends to 1 as r -> 0."""
    config = cppn_config(7)
    rng = np.random.default_rng(0)
    ratios = []
    for seed in range(20):
        net = ESNetworkND(substrate(3), random_cppn(config, seed), PARAMS)
        coord = (0.0, -1.0, 0.0)
        r = 1e-4
        for _ in range(5):
            centre = tuple(rng.uniform(-0.9, 0.9, 3))
            _, g = net.query.weight_and_grad(coord, centre, True)
            if g @ g < 1e-6:  # flat, e.g. inside pureples' |w| <= 0.2 dead band
                continue
            ws = [net.query.weight(coord, tuple(x + s * r / 2 for x, s in zip(centre, sig)), True)
                  for sig in net.signs]
            if 0.0 in ws:  # a child fell into the dead band: the field has a kink here
                continue
            ratios.append(np.var(ws) / ((r / 2) ** 2 * float(g @ g)))
    assert len(ratios) >= 10
    assert np.median(ratios) == pytest.approx(1.0, rel=1e-3)
    assert np.allclose(ratios, 1.0, rtol=5e-2)  # second-order terms on high-curvature CPPNs


def test_iteration_fix_explores_each_node_once():
    config = cppn_config(5)
    params = {**PARAMS, "iteration_level": 4, "band_threshold": 0.05, "variance_threshold": 0.1}
    for seed in range(20):
        cppn = random_cppn(config, seed)
        fixed = ESNetworkND(substrate(2), cppn, {**params, "fix_iteration": True})
        published = ESNetworkND(substrate(2), cppn, {**params, "fix_iteration": False})
        explored = {"fixed": [], "published": []}
        for name, net in (("fixed", fixed), ("published", published)):
            original = net._explore
            net._explore = lambda coord, outgoing, o=original, k=name: (
                explored[k].append((coord, outgoing)), o(coord, outgoing))[1]
            net.es_hyperneat()
        assert len(explored["fixed"]) == len(set(explored["fixed"]))
        assert len(explored["fixed"]) <= len(explored["published"])
