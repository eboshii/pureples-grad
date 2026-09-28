"""
ES-HyperNEAT on substrates of any dimension, with a choice of split test.

With n = 2, split_test="variance", cache=False and fix_iteration=False this reproduces
es_hyperneat.ESNetwork exactly (see tests/test_regression_2d.py). The differences are:

* cells have 2^n children and band pruning checks 2n neighbours;
* the feed-forward rule (pureples: source y < target y) uses a configurable axis;
* fix_iteration=True explores each hidden node once in the iterated search (pureples
  re-explores nodes found two or more rounds earlier);
* split_test chooses how a cell decides to split:
    "variance"      variance of the CPPN at the 2^n child centres (published method);
    "corner_cache"  variance at the 2^n cell corners, with every CPPN query cached;
    "gradient"      (r/2)^2 * |grad w(centre)|^2, one forward and one backward pass;
    "gradient_band" as "gradient", and band pruning uses r * max_i |dw/dx_i|
                    instead of querying 2n neighbours.
"""
import itertools

import neat
import numpy as np

from pureples.shared.diff_cppn import forward_backward

SPLIT_TESTS = ("variance", "corner_cache", "gradient", "gradient_band")


class CPPNQuery:
    """
    Queries the CPPN for weights and their gradients, counting every pass.
    Applies pureples' weight transform: |w| <= 0.2 becomes 0, the rest is rescaled.
    """

    def __init__(self, cppn, max_weight, cache):
        self.cppn = cppn
        self.max_weight = max_weight
        self.cache = {} if cache else None
        self.forward_passes = 0
        self.backward_passes = 0

    @staticmethod
    def _inputs(coord, point, outgoing):
        return tuple(coord) + tuple(point) + (1.0,) if outgoing else tuple(point) + tuple(coord) + (1.0,)

    def _transform(self, raw):
        if abs(raw) > 0.2:
            return ((raw - 0.2) / 0.8 if raw > 0 else (raw + 0.2) / 0.8) * self.max_weight
        return 0.0

    def weight(self, coord, point, outgoing):
        key = self._inputs(coord, point, outgoing)
        if self.cache is not None and key in self.cache:
            return self.cache[key][0]
        self.forward_passes += 1
        w = self._transform(self.cppn.activate(list(key))[0])
        if self.cache is not None:
            self.cache[key] = (w, None)
        return w

    def weight_and_grad(self, coord, point, outgoing):
        """Weight at point and its gradient with respect to point's coordinates."""
        key = self._inputs(coord, point, outgoing)
        if self.cache is not None and key in self.cache and self.cache[key][1] is not None:
            return self.cache[key]
        self.forward_passes += 1
        self.backward_passes += 1
        outputs, d_inputs = forward_backward(self.cppn, list(key))
        raw = outputs[0]
        n = len(point)
        d_point = np.array(d_inputs[n:2 * n] if outgoing else d_inputs[:n])
        scale = self.max_weight / 0.8 if abs(raw) > 0.2 else 0.0
        result = (self._transform(raw), d_point * scale)
        if self.cache is not None:
            self.cache[key] = result
        return result


class Cell:
    """A cell of the 2^n-tree: centre, half-width, depth and (lazily) its weight."""

    __slots__ = ("centre", "width", "lvl", "w", "grad", "cs")

    def __init__(self, centre, width, lvl):
        self.centre = centre
        self.width = width
        self.lvl = lvl
        self.w = None
        self.grad = None
        self.cs = None


class ESNetworkND:
    """
    The evolvable substrate network, in n dimensions.
    Takes the same params as es_hyperneat.ESNetwork, plus the optional keys
    "split_test", "cache", "feedforward_axis" and "fix_iteration".
    """

    def __init__(self, substrate, cppn, params):
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.split_test = params.get("split_test", "variance")
        if self.split_test not in SPLIT_TESTS:
            raise ValueError(f"split_test must be one of {SPLIT_TESTS}")
        self.feedforward_axis = params.get("feedforward_axis", 1)
        self.fix_iteration = params.get("fix_iteration", True)
        cache = params.get("cache", self.split_test != "variance")
        self.query = CPPNQuery(cppn, self.max_weight, cache)
        self.dim = len(substrate.input_coordinates[0])
        self.signs = list(itertools.product((-1.0, 1.0), repeat=self.dim))
        self.connections = {}
        self.activations = 2 ** params["max_depth"] + 1
        activation_functions = neat.activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])

    # ---- Phenotype -------------------------------------------------------------

    def create_phenotype_network(self):
        """Create a RecurrentNetwork using the ES-HyperNEAT approach."""
        inputs = [tuple(c) for c in self.substrate.input_coordinates]
        outputs = [tuple(c) for c in self.substrate.output_coordinates]
        input_nodes = list(range(len(inputs)))
        output_nodes = list(range(len(inputs), len(inputs) + len(outputs)))
        coords_to_id = dict(zip(inputs + outputs, input_nodes + output_nodes))

        hidden_nodes, connections = self.es_hyperneat()
        for coord in sorted(hidden_nodes):
            coords_to_id[coord] = len(coords_to_id)

        links = {}
        for (src, tgt), w in connections.items():
            links.setdefault(coords_to_id[tgt], []).append((coords_to_id[src], w))
        node_evals = [(idx, self.activation, sum, 0.0, 1.0, l) for idx, l in links.items()]
        return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)

    # ---- Weights ---------------------------------------------------------------

    def _w(self, coord, cell, outgoing):
        if cell.w is None:
            cell.w = self.query.weight(coord, cell.centre, outgoing)
        return cell.w

    def _grad(self, coord, cell, outgoing):
        if cell.grad is None:
            cell.w, cell.grad = self.query.weight_and_grad(coord, cell.centre, outgoing)
        return cell.grad

    @staticmethod
    def get_weights(p):
        """All leaf weights under a cell, as in pureples."""
        temp = []

        def loop(pp):
            if pp.cs is not None:
                for c in pp.cs:
                    loop(c)
            else:
                temp.append(pp.w)
        loop(p)
        return temp

    def _children(self, p):
        half = p.width / 2.0
        return [Cell(tuple(x + s * half for x, s in zip(p.centre, sig)), half, p.lvl + 1)
                for sig in self.signs]

    def _corner_variance(self, coord, p, outgoing):
        return np.var([self.query.weight(coord, tuple(x + s * p.width for x, s in zip(p.centre, sig)), outgoing)
                       for sig in self.signs])

    def _gradient_score(self, coord, p, outgoing):
        g = self._grad(coord, p, outgoing)
        return (p.width / 2.0) ** 2 * float(g @ g)

    # ---- Tree ------------------------------------------------------------------

    def division_initialization(self, coord, outgoing):
        """Build the 2^n-tree, splitting cells whose split test exceeds division_threshold."""
        root = Cell((0.0,) * self.dim, 1.0, 1)
        q = [root]
        while q:
            p = q.pop(0)
            p.cs = self._children(p)
            if self.split_test == "variance":
                for c in p.cs:
                    self._w(coord, c, outgoing)
                forced = p.lvl < self.initial_depth
                split = forced or (p.lvl < self.max_depth and
                                   np.var(self.get_weights(p)) > self.division_threshold)
            else:
                split = p.lvl < self.initial_depth or (
                    p.lvl < self.max_depth and self._division_score(coord, p, outgoing) > self.division_threshold)
            if split:
                q.extend(p.cs)
        return root

    def _division_score(self, coord, p, outgoing):
        if self.split_test == "corner_cache":
            return self._corner_variance(coord, p, outgoing)
        return self._gradient_score(coord, p, outgoing)

    def _pruning_score(self, coord, c, outgoing):
        """pureples: variance of all leaf weights under c (0 for a leaf)."""
        if c.cs is None:
            return 0.0
        if self.split_test == "variance":
            return np.var(self.get_weights(c))
        if self.split_test == "corner_cache":
            return self._corner_variance(coord, c, outgoing)
        return self._gradient_score(coord, c, outgoing)

    def _band(self, coord, c, step, outgoing):
        if self.split_test == "gradient_band":
            return step * float(np.max(np.abs(self._grad(coord, c, outgoing))))
        w = self._w(coord, c, outgoing)
        best = 0.0
        for i in range(self.dim):
            d = []
            for s in (-1.0, 1.0):
                nb = list(c.centre)
                nb[i] += s * step
                d.append(abs(w - self.query.weight(coord, tuple(nb), outgoing)))
            best = max(best, min(d))
        return best

    def pruning_extraction(self, coord, p, outgoing):
        """Express connections where the weight differs from its neighbours (band pruning)."""
        # pureples checks the four neighbours in the order left, right, top, bottom and
        # combines them as max(min(top, bottom), min(left, right)): the same per-axis rule.
        for c in p.cs:
            if self._pruning_score(coord, c, outgoing) > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing)
                continue
            if self._band(coord, c, p.width, outgoing) > self.band_threshold:
                w = self._w(coord, c, outgoing)
                src, tgt = (tuple(coord), c.centre) if outgoing else (c.centre, tuple(coord))
                a = self.feedforward_axis
                if w != 0.0 and src[a] < tgt[a] and src != tgt:
                    self.connections[src, tgt] = w

    def _explore(self, coord, outgoing):
        self.connections = {}
        root = self.division_initialization(coord, outgoing)
        self.pruning_extraction(coord, root, outgoing)
        return self.connections

    def es_hyperneat(self):
        """Find the hidden nodes and their connections."""
        inputs = [tuple(c) for c in self.substrate.input_coordinates]
        outputs = [tuple(c) for c in self.substrate.output_coordinates]
        hidden_nodes = set()
        connections1, connections2, connections3 = {}, {}, {}

        for coord in inputs:  # Explore from inputs.
            connections1.update(self._explore(coord, True))
            hidden_nodes.update(tgt for _, tgt in connections1)

        unexplored = set(hidden_nodes)
        explored = set()
        for _ in range(self.iteration_level):  # Explore from hidden.
            for coord in unexplored:
                connections2.update(self._explore(coord, True))
                hidden_nodes.update(tgt for _, tgt in connections2)
            if self.fix_iteration:
                explored |= unexplored
                unexplored = hidden_nodes - explored
            else:  # pureples as published
                unexplored = hidden_nodes - unexplored

        for coord in outputs:  # Explore to outputs.
            connections3.update(self._explore(coord, False))

        connections = {**connections1, **connections2, **connections3}
        return self.clean_net(connections)

    def clean_net(self, connections):
        """Keep only nodes on a path from an input to an output."""
        inputs = {tuple(c) for c in self.substrate.input_coordinates}
        outputs = {tuple(c) for c in self.substrate.output_coordinates}

        def reachable(start, edges):
            seen, frontier = set(start), list(start)
            while frontier:
                node = frontier.pop()
                for nxt in edges.get(node, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            return seen

        fwd, bwd = {}, {}
        for src, tgt in connections:
            fwd.setdefault(src, []).append(tgt)
            bwd.setdefault(tgt, []).append(src)
        true_nodes = reachable(inputs, fwd) & reachable(outputs, bwd)
        true_connections = {k: w for k, w in connections.items() if k[0] in true_nodes and k[1] in true_nodes}
        return true_nodes - inputs - outputs, true_connections

    def stats(self):
        """Counters for the last es_hyperneat() call and everything before it."""
        return {"forward_passes": self.query.forward_passes, "backward_passes": self.query.backward_passes}
