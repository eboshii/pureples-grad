"""
Reverse-mode derivatives of a neat-python FeedForwardNetwork with respect to its inputs.

The forward pass repeats FeedForwardNetwork.activate operation for operation, so its
output is bit-identical to neat-python's. The backward pass then applies the chain
rule from the first output back to every input, in one sweep over the nodes.
"""
import math

import neat.activations as act
import neat.aggregations as agg


def _clamp_grad(z, scale, bound):
    """Derivative factor of max(-bound, min(bound, scale * z)): zero once clamped."""
    return scale if -bound < scale * z < bound else 0.0


def _sigmoid_d(z):
    s = act.sigmoid_activation(z)
    return _clamp_grad(z, 5.0, 60.0) * s * (1.0 - s)


def _tanh_d(z):
    t = act.tanh_activation(z)
    return _clamp_grad(z, 2.5, 60.0) * (1.0 - t * t)


def _sin_d(z):
    return _clamp_grad(z, 5.0, 60.0) * math.cos(5.0 * z)


def _gauss_d(z):
    if not -3.4 < z < 3.4:
        return 0.0
    return -10.0 * z * math.exp(-5.0 * z * z)


def _softplus_d(z):
    zz = 5.0 * z
    if not -60.0 < zz < 60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-zz))


def _selu_d(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam if z > 0.0 else lam * alpha * math.exp(z)


_DERIVATIVES_BY_NAME = {
    "sigmoid": _sigmoid_d,
    "tanh": _tanh_d,
    "sin": _sin_d,
    "gauss": _gauss_d,
    "relu": lambda z: 1.0 if z > 0.0 else 0.0,
    "elu": lambda z: 1.0 if z > 0.0 else math.exp(z),
    "lelu": lambda z: 1.0 if z > 0.0 else 0.005,
    "selu": _selu_d,
    "softplus": _softplus_d,
    "identity": lambda z: 1.0,
    "clamped": lambda z: 1.0 if -1.0 < z < 1.0 else 0.0,
    "inv": lambda z: -1.0 / (z * z) if z != 0.0 else 0.0,
    "log": lambda z: 1.0 / z if z > 1e-7 else 0.0,
    "exp": lambda z: math.exp(z) if -60.0 < z < 60.0 else 0.0,
    "abs": lambda z: 1.0 if z > 0.0 else (-1.0 if z < 0.0 else 0.0),
    "hat": lambda z: (-1.0 if z > 0.0 else 1.0) if -1.0 < z < 1.0 else 0.0,
    "square": lambda z: 2.0 * z,
    "cube": lambda z: 3.0 * z * z,
}

# Keyed by function object; neat-python 0.92 lacks elu, lelu and selu.
ACTIVATION_DERIVATIVES = {getattr(act, name + "_activation"): d
                          for name, d in _DERIVATIVES_BY_NAME.items()
                          if hasattr(act, name + "_activation")}


def _aggregation_grads(agg_func, xs):
    """Partial derivatives of an aggregation with respect to each of its arguments."""
    k = len(xs)
    if agg_func is agg.sum_aggregation:
        return [1.0] * k
    if agg_func is agg.mean_aggregation:
        return [1.0 / k] * k if k else []
    if agg_func is agg.product_aggregation:
        grads = []
        for i in range(k):
            p = 1.0
            for j, x in enumerate(xs):
                if j != i:
                    p *= x
            grads.append(p)
        return grads
    if agg_func in (agg.max_aggregation, agg.min_aggregation, agg.maxabs_aggregation):
        grads = [0.0] * k
        if k:
            grads[xs.index(agg_func(xs))] = 1.0
        return grads
    raise ValueError(f"No derivative for aggregation {agg_func.__name__}")


def forward_backward(net, inputs):
    """
    Evaluate a FeedForwardNetwork and differentiate its first output.
    Returns (outputs, gradient of outputs[0] with respect to each input).
    """
    values = dict(zip(net.input_nodes, inputs))
    tape = []
    for node, act_func, agg_func, bias, response, links in net.node_evals:
        xs = [values.get(i, 0.0) * w for i, w in links]
        s = agg_func(xs)
        z = bias + response * s
        values[node] = act_func(z)
        tape.append((node, act_func, agg_func, response, links, xs, z))
    outputs = [values.get(i, 0.0) for i in net.output_nodes]

    adj = {net.output_nodes[0]: 1.0}
    for node, act_func, agg_func, response, links, xs, z in reversed(tape):
        a = adj.get(node, 0.0)
        if a == 0.0:
            continue
        dz = a * ACTIVATION_DERIVATIVES[act_func](z) * response
        if dz == 0.0:
            continue
        for (i, w), g in zip(links, _aggregation_grads(agg_func, xs)):
            adj[i] = adj.get(i, 0.0) + dz * g * w
    return outputs, [adj.get(i, 0.0) for i in net.input_nodes]
