# pureples-grad: n-dimensional ES-HyperNEAT with gradient split tests

pureples-grad is a fork of [PUREPLES](https://github.com/ukuleleplayer/pureples) that adds
`pureples/es_hyperneat/es_hyperneat_nd.py`, an ES-HyperNEAT that works on substrates of any
dimension and can decide where to split the tree with a gradient instead of samples.
The original `es_hyperneat.py` is unchanged and serves as the baseline.

**Changes relative to PUREPLES**

* **n-D tree.** Cells have 2ⁿ children and band pruning checks 2n neighbours (±1 step along each axis).
* **Split tests**, chosen with `params["split_test"]`:
  * `variance`: variance of the CPPN at the 2ⁿ child centres, as published (Risi & Stanley, 2012).
  * `corner_cache`: variance at the 2ⁿ cell corners, with every CPPN query cached so neighbouring cells share corners.
  * `gradient`: split if (r/2)²‖∇w‖² exceeds the threshold, where r is the cell's half-width. To first order this equals the variance over the child centres, and it costs one forward and one backward pass through the CPPN.
  * `gradient_band`: as `gradient`, and band pruning uses r·maxᵢ|∂w/∂xᵢ| instead of querying 2n neighbours.
* **Derivatives of the CPPN.** `pureples/shared/diff_cppn.py` runs a neat-python network forward (bit-identical to `activate`) and back, giving ∂w/∂input for every input in one pass.
* **Feed-forward axis.** PUREPLES keeps a connection only if its source has a smaller y than its target. `params["feedforward_axis"]` chooses the axis (default 1, which is y).
* **Iteration fix.** In PUREPLES the iterated hidden-node search sets `unexplored = hidden_nodes - unexplored`, which re-explores nodes found two or more rounds earlier. With `params["fix_iteration"] = True` (the default in the new module) each hidden node is explored once. The published behaviour is kept for the regression test.
* **Counters.** `ESNetworkND.stats()` reports the number of CPPN forward and backward passes.

**Checks** (`python -m pytest tests`)

* `test_regression_2d.py`: with n = 2 and the published settings (`split_test="variance"`, `cache=False`, `fix_iteration=False`), the new module builds exactly the same hidden nodes, connections and weights as the original on 100 random CPPNs, for each of three parameter sets (30 CPPNs for the deep set).
* `test_diff_cppn.py`: the backward pass matches central finite differences on random CPPNs using 12 activation functions, and the forward pass matches neat-python exactly.
* `test_nd.py`: trees in 2–4 dimensions tile each cell correctly, every split test builds a feed-forward network, the feed-forward axis is respected, the iteration fix explores each node once, and on small cells the gradient score matches the sampled variance to within 1%.

**Requirements.** `pip install -r requirements.txt`. PUREPLES targets neat-python 0.92; neat-python 2.0 rejects its config files, so the version is pinned.

---

<img src="https://github.com/ukuleleplayer/pureples/blob/master/PUREPLES.png" alt="REPLES LOGO" width="400" height="400">

PUREPLES - Pure Python Library for ES-HyperNEAT
===============================================

About
-----
This is a library of evolutionary algorithms with a focus on neuroevolution, implemented in pure python, depending on the [neat-python](https://github.com/CodeReclaimers/neat-python) implementation. It contains a faithful implementation of both HyperNEAT and ES-HyperNEAT which are briefly described below.

**NEAT** (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks.  
**HyperNEAT** (Hypercube-based NEAT) is a method developed by Kenneth O. Stanley utilizing NEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain.  
**ES-HyperNEAT** (Evolvable-substrate HyperNEAT) is a method developed by Sebastian Risi and Kenneth O. Stanley utilizing HyperNEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain. In contrast to HyperNEAT, the substrate used during evolution is able to evolve. This rids the user of some initial work and often creates a more suitable substrate.

The library is extensible in regards to easy transition between experimental domains.

Getting started
---------------
This section briefly describes how to install and run experiments.  

### Installation Guide
First, make sure you have the dependencies installed: `numpy`, `neat-python`, `graphviz`, `matplotlib` and `gym`.  
All the above can be installed using [pip](https://pip.pypa.io/en/stable/installing/).  
Next, download the source code and run `setup.py` (`pip install .`) from the root folder. Now you're able to use **PUREPLES**!

### Experimenting
How to experiment using NEAT will not be described, since this is the responsibility of the `neat-python` library.

Setting up an experiment for **HyperNEAT**:
* Define a substrate with input nodes and output nodes as a list of tuples. The hidden nodes is a list of lists of tuples where the inner lists represent layers. The first list is the topmost layer, the last the bottommost.
* Create a configuration file defining various NEAT specific parameters which are used for the CPPN.
* Define a fitness function setting the fitness of each genome. This is where the CPPN and the ANN is constructed for each generation - use the `create_phenotype_network` method from the `hyperneat` module.
* Create a population with the configuration file made in (2).
* Run the population with the fitness function made in (3) and the configuration file made in (2). The output is the genome solving the task or the one closest to solving it.

Setting up an experiment for **ES-HyperNEAT**:
Use the same setup as HyperNEAT except for:
* Not declaring hidden nodes when defining the substrate.
* Declaring ES-HyperNEAT specific parameters.
* Using the `create_phenotype_network` method residing in the `es_hyperneat` module when creating the ANN.

If one is trying to solve an experiment defined by the [OpenAI Gym](https://gym.openai.com/) it is even easier to experiment. In the `shared` module a file called `gym_runner` is able to do most of the work. Given the number of generations, the environment to run, a configuration file, and a substrate, the relevant runner will take care of everything regarding population, fitness function etc.

Please refer to the sample experiments included for further details on experimenting. 

