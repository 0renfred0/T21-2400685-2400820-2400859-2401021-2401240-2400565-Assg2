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
    source, sink = height * width, height * width + 1
    residual: list[dict[int, float]] = [{} for _ in range(sink + 1)]
    infinite = float(np.sum(energy, dtype=np.float64)) + 1.0

    def add_edge(start: int, end: int, capacity: float) -> None:
        residual[start][end] = residual[start].get(end, 0.0) + capacity
        residual[end].setdefault(start, 0.0)

    for row in range(height):
        add_edge(source, row * width, infinite)
        for col in range(width):
            node = row * width + col
            cost = float(energy[row, col])
            if col + 1 < width:
                add_edge(node, node + 1, cost)
                add_edge(node + 1, node, infinite)
            else:
                add_edge(node, sink, cost)
            if col > 0:
                if row > 0:
                    add_edge(node, node - width - 1, infinite)
                if row + 1 < height:
                    add_edge(node, node + width - 1, infinite)

    while True:
        parent = [-1] * len(residual)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] == -1:
            node = queue.popleft()
            for neighbour, capacity in residual[node].items():
                if capacity > 0.0 and parent[neighbour] == -1:
                    parent[neighbour] = node
                    queue.append(neighbour)
        if parent[sink] == -1:
            break

        amount = infinite
        node = sink
        while node != source:
            previous = parent[node]
            amount = min(amount, residual[previous][node])
            node = previous
        node = sink
        while node != source:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous

    reachable = np.array(parent[:source]) != -1
    # The final failed BFS visits precisely the source side of the minimum cut.
    return width - 1 - np.argmax(reachable.reshape(height, width)[:, ::-1], axis=1)
