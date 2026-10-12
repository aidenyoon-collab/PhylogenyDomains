#!/usr/bin/env python3
"""
Balanced Minimum Evolution (BME) tree search, and a quantitative measure of how far a
distance matrix is from additive.

Minimum evolution is used here as an alternative to UPGMA and neighbor joining. Neither
FastME nor scikit-bio is assumed to be installed, and Biopython's DistanceTreeConstructor
offers only `nj` and `upgma`, so BME is implemented directly.

METHOD. Under the balanced (Pauplin) edge-weighting, the total length of a topology T
given a distance matrix d has the closed form

    L(T) = sum_{i<j} 2^(1 - p_ij) * d_ij                                  [Pauplin 2000]

where p_ij is the TOPOLOGICAL distance between leaves i and j, i.e. the number of edges
on the path connecting them. Minimizing L(T) over topologies is balanced minimum
evolution. We search as FastME does: start from the Neighbor-Joining tree (NJ is itself a
greedy heuristic for BME) and apply nearest-neighbour interchange (NNI) hill-climbing,
taking the best improving swap each round until no swap reduces L.

  Pauplin Y. 2000. Direct calculation of a tree length using a distance matrix.
    J. Mol. Evol. 51: 41-47.  https://doi.org/10.1007/s002390010065
  Desper R. & Gascuel O. 2002. Fast and accurate phylogeny reconstruction algorithms
    based on the minimum-evolution principle. J. Comput. Biol. 9: 687-705.
    https://doi.org/10.1089/106652702761034136

ADDITIVITY. NJ and ME are both consistent only when the distances are close to additive
(they satisfy the four-point condition). For a quartet {i,j,k,l} additivity requires the
two largest of d_ij+d_kl, d_ik+d_jl, d_il+d_jk to be EQUAL. `additivity_deviation`
samples quartets and reports the relative gap between the two largest sums, which is 0
for a perfectly additive matrix. This turns "NJ is disadvantaged by a tied matrix" from
an assertion into a measurement.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


def _node_index(tree) -> Tuple[Dict, List]:
    nodes = list(tree.traverse())
    return {id(nd): i for i, nd in enumerate(nodes)}, nodes


def topological_distances(tree, leaf_order: Sequence[str]) -> np.ndarray:
    """
    All-pairs edge-count distances between leaves, in the order given.

    Degree-2 internal nodes are CONTRACTED first. Pauplin's formula is defined on the
    unrooted topology, and a rooted binary tree has a degree-2 root that inflates every
    path crossing it by one edge -- which silently corrupts L(T). (This was caught by the
    4-leaf unit test: the rooted form of ((A,B),(C,D)) returned 10.0 instead of 15.0.)
    """
    idx, nodes = _node_index(tree)
    leaf_ids = {id(nd) for nd in nodes if nd.is_leaf()}
    adj: Dict[int, set] = {id(nd): set() for nd in nodes}
    for nd in nodes:
        for ch in nd.children:
            adj[id(nd)].add(id(ch)); adj[id(ch)].add(id(nd))
    for nid in [k for k in adj if k not in leaf_ids]:
        if nid in adj and len(adj[nid]) == 2:
            x, y = tuple(adj[nid])
            adj[x].discard(nid); adj[y].discard(nid)
            adj[x].add(y); adj[y].add(x)
            del adj[nid]
    keys = list(adj)
    remap = {k: i for i, k in enumerate(keys)}
    rows, cols = [], []
    for k in keys:
        for nb in adj[k]:
            rows.append(remap[k]); cols.append(remap[nb])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), len(keys)))
    dist = shortest_path(g, method="D", directed=False, unweighted=True)
    pos = {nd.name: remap[id(nd)] for nd in nodes if nd.is_leaf()}
    sel = [pos[name] for name in leaf_order]
    return dist[np.ix_(sel, sel)]


def pauplin_length(tree, D: np.ndarray, leaf_order: Sequence[str]) -> float:
    """Balanced (Pauplin) tree length L(T) = sum_{i<j} 2^(1-p_ij) d_ij."""
    P = topological_distances(tree, leaf_order)
    iu = np.triu_indices(len(leaf_order), k=1)
    return float(np.sum(np.power(2.0, 1.0 - P[iu]) * D[iu]))


def _candidate_nnis(tree) -> List[Tuple]:
    """(child_of_v, sibling_of_v) pairs whose exchange is a valid NNI on an internal edge."""
    out = []
    for v in tree.traverse():
        if v.is_leaf() or v.is_root():
            continue
        u = v.up
        sibs = [c for c in u.children if c is not v]
        if not sibs:
            continue
        for c in v.children:
            for s in sibs:
                out.append((c, s))
    return out


def _swap(a, b) -> None:
    """Exchange the parents of two nodes (an NNI when they straddle one internal edge)."""
    pa, pb = a.up, b.up
    a.detach(); b.detach()
    pa.add_child(b); pb.add_child(a)


def bme_nni_search(tree, D: np.ndarray, leaf_order: Sequence[str],
                   max_rounds: int = 200) -> Tuple[object, float, int, int]:
    """
    NNI hill-climbing on the balanced tree length, starting from `tree` (modified in place).
    Returns (tree, final_length, n_swaps_accepted, n_rounds).
    """
    best = pauplin_length(tree, D, leaf_order)
    swaps = 0
    for rnd in range(1, max_rounds + 1):
        cands = _candidate_nnis(tree)
        best_gain, best_pair = 0.0, None
        for a, b in cands:
            if a.up is b.up:            # same parent: exchange is a no-op
                continue
            _swap(a, b)
            L = pauplin_length(tree, D, leaf_order)
            _swap(a, b)                 # undo
            gain = best - L
            if gain > best_gain + 1e-12:
                best_gain, best_pair = gain, (a, b)
        if best_pair is None:
            return tree, best, swaps, rnd
        _swap(*best_pair)
        best -= best_gain
        swaps += 1
    return tree, best, swaps, max_rounds


def additivity_deviation(D: np.ndarray, n_quartets: int = 20000,
                         rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Four-point condition check. For each sampled quartet the two LARGEST of the three
    pairwise sums must be equal if the matrix is additive; the relative gap between them
    measures the violation. Returns mean/median relative gap and the exactly-satisfied
    fraction.
    """
    rng = rng or np.random.default_rng(0)
    n = D.shape[0]
    gaps = np.empty(n_quartets)
    exact = 0
    for t in range(n_quartets):
        i, j, k, l = rng.choice(n, size=4, replace=False)
        s = np.sort([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])[::-1]
        gaps[t] = (s[0] - s[1]) / s[0] if s[0] > 0 else 0.0
        if s[0] - s[1] <= 1e-12 * max(s[0], 1.0):
            exact += 1
    return {"mean_relative_gap": float(gaps.mean()),
            "median_relative_gap": float(np.median(gaps)),
            "fraction_exactly_additive": exact / n_quartets,
            "n_quartets": n_quartets}
