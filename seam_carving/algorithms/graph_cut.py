"""Optimal seam finding with an s-t minimum cut (assignment item 4).

Edmonds-Karp repeatedly augments a shortest residual path found by BFS. The final BFS
identifies the source side of a minimum cut. With V = H*W + 2 and E = O(H*W), this takes
O(V*E**2) time in the worst case and O(H*W) space. Use small images for comparisons:
pure-Python max-flow is considerably slower than the O(H*W) dynamic program.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def find_vertical_seam_graph_cut(energy: np.ndarray) -> np.ndarray:
    """Return an optimal connected seam on a read-only, nonnegative (H, W) energy map.

    Each row's source-side pixels form a prefix: backward and backward-diagonal edges
    have capacity greater than any seam cost. This forces exactly one crossing per row
    and keeps adjacent crossings within one column of each other.

    The edge from the last pixel to the sink has that pixel's energy, rather than the
    brief's infinite capacity, so a seam may include the last column (including W == 1).
    """
    height, width = energy.shape

    # Every pixel is a node, numbered row * width + col. Two extra nodes stand for the "left
    # side" (source) and the "right side" (sink) of the image. A seam is a cut that separates them.
    source, sink = height * width, height * width + 1

    # residual[u][v] = how much more flow we can still push along u -> v. Dicts mean we only
    # store the edges that actually exist.
    residual: list[dict[int, float]] = [{} for _ in range(sink + 1)]

    # "Infinite" only has to be bigger than any seam could possibly cost. The total energy of the
    # whole image plus 1 does that, and it avoids real float infinities, which misbehave when
    # you subtract from them.
    infinite = float(np.sum(energy, dtype=np.float64)) + 1.0

    def add_edge(start: int, end: int, capacity: float) -> None:
        # Add to any capacity already there. The reverse edge starts at 0 so that flow can be
        # "undone" later, which max-flow needs.
        residual[start][end] = residual[start].get(end, 0.0) + capacity
        residual[end].setdefault(start, 0.0)

    for row in range(height):
        # The first pixel of every row is hooked up to the source with unlimited capacity, so it
        # can never be cut off: every row starts on the source side.
        add_edge(source, row * width, infinite)
        for col in range(width):
            node = row * width + col
            cost = float(energy[row, col])
            if col + 1 < width:
                # The forward edge costs this pixel's energy. A cut has to sever one of these in
                # every row, so the price of the cut is exactly the energy of the pixels removed.
                add_edge(node, node + 1, cost)

                # The backward edge is unlimited: if a pixel is on the source side, so is
                # everything to its left. Each row's source side is then a prefix of columns, so
                # every row is cut exactly once.
                add_edge(node + 1, node, infinite)
            else:
                # The last pixel joins the sink at its own energy (not infinity) so that it
                # can be the seam pixel too. See the note in the docstring.
                add_edge(node, sink, cost)

            # Diagonal edges tie neighbouring rows together: if (row, col) is on the source side,
            # then so are (row - 1, col - 1) and (row + 1, col - 1). That stops the cut moving
            # more than one column between rows, which keeps the seam connected.
            if col > 0:
                if row > 0:
                    add_edge(node, node - width - 1, infinite)
                if row + 1 < height:
                    add_edge(node, node + width - 1, infinite)

    # Edmonds-Karp: keep finding a shortest path (by number of edges) that still has spare
    # capacity, and push as much flow along it as it can take. Stop when there is none left.
    while True:
        # Breadth-first search from the source. parent[n] says how we reached node n, and -1
        # means "not reached yet".
        parent = [-1] * len(residual)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for neighbour, capacity in residual[node].items():
                if capacity > 0.0 and parent[neighbour] == -1:
                    parent[neighbour] = node
                    queue.append(neighbour)

        # The sink can't be reached any more, so no more flow fits and we have the maximum flow.
        if parent[sink] == -1:
            break

        # Walk back from the sink to find the bottleneck: the smallest spare capacity on the path.
        amount = infinite
        node = sink
        while node != source:
            previous = parent[node]
            amount = min(amount, residual[previous][node])
            node = previous

        # Push that much flow. Each edge loses that capacity going forward and gains the same
        # amount going backward, which is what lets a later path undo this choice.
        node = sink
        while node != source:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous

    # The last search failed, but it still marked every node it could reach. By the max-flow
    # min-cut theorem those nodes are the source side of the minimum cut.
    reachable = np.array(parent[:source]) != -1

    # In each row the reachable pixels are a prefix, and the seam pixel is the rightmost of them.
    # Flipping each row left-to-right lets argmax find the first True from the right; we then
    # convert that back into a normal column number.
    return width - 1 - np.argmax(reachable.reshape(height, width)[:, ::-1], axis=1)
