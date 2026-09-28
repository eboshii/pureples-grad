"""The reverse pass matches central finite differences, and the forward pass matches neat-python."""
import numpy as np
import pytest

from pureples.shared.diff_cppn import forward_backward
from tests.cppns import ALL_ACTIVATIONS, cppn_config, random_cppn


@pytest.mark.parametrize("num_inputs", [5, 7, 13])
def test_gradient_matches_finite_differences(num_inputs):
    config = cppn_config(num_inputs, ALL_ACTIVATIONS)
    rng = np.random.default_rng(0)
    checked = 0
    for seed in range(40):
        net = random_cppn(config, seed)
        x = list(rng.uniform(-1, 1, num_inputs))
        out, grad = forward_backward(net, x)
        assert out == net.activate(x)
        for i in range(num_inputs):
            fd = [central_difference(net, x, i, h) for h in (1e-5, 1e-6)]
            # Near a kink (relu, abs, hat, a clamp) the two step sizes disagree; skip those.
            if abs(fd[0] - fd[1]) > 1e-4 * max(1.0, abs(fd[1])):
                continue
            assert grad[i] == pytest.approx(fd[1], rel=1e-4, abs=1e-6), (seed, i)
            checked += 1
    assert checked > 0.9 * 40 * num_inputs


def central_difference(net, x, i, h):
    xp, xm = list(x), list(x)
    xp[i] += h
    xm[i] -= h
    return (net.activate(xp)[0] - net.activate(xm)[0]) / (2 * h)
