"""Seam finding with a Graph Cut: minimum s-t cut / maximum flow (assignment item 4).

Owner:  <name / SIT ID>
Status: TODO. The function below is a stub. Write your implementation in its body.

The interface contract is in ``seam_carving/algorithms/__init__.py``. Check your work with::

    pytest tests/test_algorithms.py -k graphcut -v

The tests compare the cost of your seam against Dynamic Programming and against brute force,
so they also prove that the cut is optimal.

Points for the report (items 4 and 5c):
* Why does every finite cut cross each row exactly once, and why is that crossing a connected
  seam? (It follows from the infinite edges described below.)
* Max-flow min-cut theorem: the capacity of the minimum cut is the cost of the best seam.
* Complexity. The graph has V = H*W nodes and E ~ 4*H*W edges. Edmonds-Karp is O(V * E^2).
  Dinic is O(V^2 * E) in general and faster in practice. DP is O(H*W). Measure the gap with
  ``python -m seam_carving.benchmark scaling``.
* Graph cuts pay off in the *video* setting of Rubinstein et al. (2008), where seams become
  surfaces. Mention it when you compare the approaches.
"""

from __future__ import annotations

import numpy as np


def find_vertical_seam_graph_cut(energy: np.ndarray) -> np.ndarray:
    """Return the vertical seam with the lowest total energy, found as a minimum s-t cut.

    Parameters
    ----------
    energy:
        ``(H, W)`` float array, read-only.

    Returns
    -------
    np.ndarray
        ``(H,)`` integer array. ``seam[i]`` is the column removed from row ``i``.

    Graph for a vertical seam (one node per pixel, node id ``i * W + j``)::

        source -> (i, 0)          capacity INF   first column is always on the source side
        (i, j) -> (i, j+1)        capacity e[i, j]   cutting this edge removes pixel (i, j)
        (i, j+1) -> (i, j)        capacity INF   S is a prefix of each row, so 1 cut per row
        (i, j) -> (i-1, j-1)      capacity INF   these two diagonal edges make
        (i, j) -> (i+1, j-1)      capacity INF   neighbouring rows' cuts differ by <= 1 column
        (i, W-1) -> sink          see GOTCHA below

    Run max-flow from source to sink. Then search (BFS/DFS) from the source along edges that
    still have residual capacity to get the source side ``S`` of the minimum cut. In each row,
    the rightmost pixel in ``S`` is the seam pixel.
    """
    # TODO(team): build the graph, run max-flow, read the seam off the cut, then delete the
    # `raise` at the bottom.
    #
    # Suggested structure (separate helpers can be unit tested and split between people):
    #
    #   def build_seam_graph(energy) -> (capacity structure, source, sink)
    #       Use an adjacency list of edges [to, capacity, index_of_reverse_edge], or a
    #       dict-of-dicts residual graph. INF: use a large *finite* number such as
    #       energy.sum() + 1.0. With float("inf") you get inf - inf = nan in the residual
    #       updates.
    #
    #   def max_flow(graph, source, sink) -> float
    #       Edmonds-Karp (BFS for the shortest augmenting path), as the brief suggests.
    #       Dinic (BFS level graph + blocking flow) is usually much faster.
    #
    #   def source_side(graph, source) -> np.ndarray[bool]  # length H*W
    #       BFS from the source using only edges with residual capacity > 0.
    #
    #   seam[i] = the largest j with in_S[i * W + j] True
    #
    # GOTCHA (the tests check this): the brief connects the last column to the sink with an
    # INF edge. Then pixel (i, W-1) can never be on the source side, so a seam can never use
    # the last column. If W == 1 the INF path source -> (i, 0) -> sink even makes the flow
    # infinite. The fix: give the edge (i, W-1) -> sink capacity e[i, W-1] (or add an extra
    # column of nodes on the sink side). Explain this choice in the report.
    #
    # Performance: pure-Python max-flow on a 300x200 image is slow (about 60k nodes per seam).
    # Develop on samples/small_shapes.png and use the benchmark to quantify the cost.
    raise NotImplementedError(
        "Graph Cut seam finder is not implemented yet (seam_carving/algorithms/graph_cut.py)"
    )
