from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

SOURCE_COLORS = (RED, BLUE, YELLOW)
PAIR_OUTPUTS = (MAGENTA, ORANGE, AZURE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIX_COLORS = {
    frozenset(): GREEN,
    frozenset({RED}): RED,
    frozenset({BLUE}): BLUE,
    frozenset({YELLOW}): YELLOW,
    frozenset({RED, BLUE}): MAGENTA,
    frozenset({RED, YELLOW}): ORANGE,
    frozenset({BLUE, YELLOW}): AZURE,
    frozenset({RED, BLUE, YELLOW}): MAROON,
}


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _line(canvas: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        canvas[r0, lo : hi + 1] = color
    else:
        lo, hi = sorted((r0, r1))
        canvas[lo : hi + 1, c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    passable = (g == GREY) | (g == GREEN)
    arrivals: dict[tuple[int, int], set[int]] = {}

    for color in SOURCE_COLORS:
        for r, c in np.argwhere(g == color):
            starts = []
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if _inside(h, w, nr, nc) and bool(passable[nr, nc]):
                    starts.append((nr, nc))
            if len(starts) != 1:
                continue

            start = starts[0]
            seen = {start}
            q = deque([start])
            while q:
                cr, cc = q.popleft()
                if int(g[cr, cc]) == GREEN:
                    arrivals.setdefault((cr, cc), set()).add(int(color))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not _inside(h, w, nr, nc):
                        continue
                    if not bool(passable[nr, nc]) or (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))

    out = g.copy()
    for r, c in np.argwhere(g == GREEN):
        out[r, c] = MIX_COLORS[frozenset(arrivals.get((int(r), int(c)), set()))]
    return out


def _connect(adj: dict[tuple[int, int], set[tuple[int, int]]], a: tuple[int, int], b: tuple[int, int]) -> None:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _free_neighbors(
    node: tuple[int, int],
    lattice_h: int,
    lattice_w: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = []
    r, c = node
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        nxt = (nr, nc)
        if 0 <= nr < lattice_h and 0 <= nc < lattice_w and nxt not in occupied:
            out.append(nxt)
    return out


def _build_graph(source_count: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    for _ in range(100):
        lattice_h = random.randint(3, 5)
        lattice_w = random.randint(3, 5)
        want_deg = 4 if random.random() < 0.4 else 3

        root_candidates = []
        for r in range(lattice_h):
            for c in range(lattice_w):
                degree = sum(
                    0 <= r + dr < lattice_h and 0 <= c + dc < lattice_w
                    for dr, dc in DIR4
                )
                if degree >= want_deg:
                    root_candidates.append((r, c))
        if not root_candidates:
            continue

        root = random.choice(root_candidates)
        adj: dict[tuple[int, int], set[tuple[int, int]]] = {root: set()}
        occupied = {root}
        neighbors = _free_neighbors(root, lattice_h, lattice_w, occupied)
        for nxt in random.sample(neighbors, want_deg):
            occupied.add(nxt)
            _connect(adj, root, nxt)

        for _ in range(random.randint(1, 5)):
            leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1]
            extendable_leaves = [
                node
                for node in leaves
                if _free_neighbors(node, lattice_h, lattice_w, occupied)
            ]
            branchable_paths = [
                node
                for node, nbrs in adj.items()
                if len(nbrs) == 2 and _free_neighbors(node, lattice_h, lattice_w, occupied)
            ]

            roll = random.random()
            if branchable_paths and roll < 0.25:
                node = random.choice(branchable_paths)
                nxt = random.choice(_free_neighbors(node, lattice_h, lattice_w, occupied))
                occupied.add(nxt)
                _connect(adj, node, nxt)
                continue

            branchable_leaves = [
                node
                for node in extendable_leaves
                if len(_free_neighbors(node, lattice_h, lattice_w, occupied)) >= 2
            ]
            if branchable_leaves and roll < 0.6:
                node = random.choice(branchable_leaves)
                free = _free_neighbors(node, lattice_h, lattice_w, occupied)
                add_count = 2 if random.random() < 0.7 else 1
                for nxt in random.sample(free, min(add_count, len(free))):
                    occupied.add(nxt)
                    _connect(adj, node, nxt)
                continue

            if not extendable_leaves:
                break
            node = random.choice(extendable_leaves)
            free = _free_neighbors(node, lattice_h, lattice_w, occupied)
            prev = next(iter(adj[node]))
            straight = (int(node[0] + node[0] - prev[0]), int(node[1] + node[1] - prev[1]))
            if straight in free and random.random() < 0.65:
                nxt = straight
            else:
                nxt = random.choice(free)
            occupied.add(nxt)
            _connect(adj, node, nxt)

        leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1]
        branch_nodes = [node for node, nbrs in adj.items() if len(nbrs) >= 3]
        if len(leaves) >= max(3, source_count) and branch_nodes:
            return adj
    return None


def _random_transform(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, k=random.randint(0, 3))
    return np.array(out, copy=True)


def _make_patch(source_colors: tuple[int, ...]) -> np.ndarray | None:
    graph = _build_graph(len(source_colors))
    if graph is None:
        return None

    span = random.randint(2, 3)
    max_r = max(node[0] for node in graph)
    max_c = max(node[1] for node in graph)
    patch = np.full((3 + max_r * span, 3 + max_c * span), BLACK, dtype=int)

    def fine(node: tuple[int, int]) -> tuple[int, int]:
        return (1 + node[0] * span, 1 + node[1] * span)

    drawn: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node, nbrs in graph.items():
        for nbr in nbrs:
            edge = tuple(sorted((node, nbr)))
            if edge in drawn:
                continue
            drawn.add(edge)
            _line(patch, fine(node), fine(nbr), GREY)

    branch_nodes = [node for node, nbrs in graph.items() if len(nbrs) >= 3]
    output_count = random.randint(1, len(branch_nodes))
    for node in random.sample(branch_nodes, output_count):
        r, c = fine(node)
        patch[r, c] = GREEN

    leaves = [node for node, nbrs in graph.items() if len(nbrs) == 1]
    chosen_leaves = random.sample(leaves, len(source_colors))
    shuffled_colors = list(source_colors)
    random.shuffle(shuffled_colors)
    for color, leaf in zip(shuffled_colors, chosen_leaves):
        nbr = next(iter(graph[leaf]))
        dr = int(leaf[0] - nbr[0])
        dc = int(leaf[1] - nbr[1])
        r, c = fine(leaf)
        patch[r + dr, c + dc] = int(color)

    return _random_transform(patch)


def _sample_source_tuple(size: int) -> tuple[int, ...]:
    if size == 0:
        return ()
    if size == 1:
        return (random.choice(SOURCE_COLORS),)
    if size == 2:
        return random.choice(((RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)))
    return SOURCE_COLORS


def _rectangles_disjoint(
    placed: list[tuple[int, int, int, int]],
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    for r0, c0, h0, w0 in placed:
        if not (top + height <= r0 or r0 + h0 <= top or left + width <= c0 or c0 + w0 <= left):
            return False
    return True


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].size, reverse=True)
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in patches)

    for _ in range(120):
        lower = max(24, int(np.ceil(np.sqrt(total_area * 1.8))))
        h = random.randint(max(lower, max_h), max(lower, max_h) + 8)
        w = random.randint(max(lower, max_w), max(lower, max_w) + 10)
        placed: list[tuple[int, int, int, int]] = []
        positions: dict[int, tuple[int, int]] = {}

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            spot = None
            for _ in range(250):
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                if _rectangles_disjoint(placed, top, left, ph, pw):
                    spot = (top, left)
                    break
            if spot is None:
                ok = False
                break
            top, left = spot
            placed.append((top, left, ph, pw))
            positions[idx] = (top, left)

        if not ok:
            continue

        grid = np.full((h, w), BLACK, dtype=int)
        for idx, patch in enumerate(patches):
            top, left = positions[idx]
            ph, pw = patch.shape
            mask = patch != BLACK
            grid[top : top + ph, left : left + pw][mask] = patch[mask]
        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    specs = [
        _sample_source_tuple(0),
        _sample_source_tuple(1),
        _sample_source_tuple(2),
        _sample_source_tuple(3),
    ]
    if random.random() < 0.6:
        specs.append(_sample_source_tuple(random.choice((1, 2, 3))))
    random.shuffle(specs)

    patches = []
    for spec in specs:
        patch = _make_patch(spec)
        if patch is None:
            return None
        patches.append(patch)

    grid = _pack_patches(patches)
    if grid is None:
        return None

    out = main(grid)
    output_colors = {int(out[r, c]) for r, c in np.argwhere(grid == GREEN)}
    if GREEN not in output_colors:
        return None
    if MAROON not in output_colors:
        return None
    if not any(color in SOURCE_COLORS for color in output_colors):
        return None
    if not any(color in PAIR_OUTPUTS for color in output_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid input after 300 attempts")
