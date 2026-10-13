from __future__ import annotations

import random

import numpy as np


BG = 0
TREE_COLOR = 5
PROBE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SPACING = 3
COARSE_H_RANGE = (4, 7)
COARSE_W_RANGE = (4, 7)
TARGET_NODE_RANGE = (6, 12)
EXTRA_MARGIN_CAP = 8
GEN_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _tree_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert colors, "expected non-background colors"

    scored: list[tuple[int, int, int]] = []
    for color in colors:
        comps = _components4(grid == color)
        largest = max(len(comp) for comp in comps)
        total = sum(len(comp) for comp in comps)
        scored.append((largest, total, color))

    scored.sort(reverse=True)
    assert scored[0][0] > 1, "tree must contain multiple cells"
    if len(scored) > 1:
        assert scored[0][:2] > scored[1][:2], "tree color must be distinguishable from probes"
    return int(scored[0][2])


def _degree_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    degrees = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            degrees[r, c] = sum(1 for nr, nc in _neighbors4(r, c, h, w) if bool(mask[nr, nc]))
    return degrees


def _trace_to_branchpoint(
    tree_mask: np.ndarray,
    degrees: np.ndarray,
    leaf: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = tree_mask.shape
    path: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = leaf

    while True:
        r, c = cur
        if int(degrees[r, c]) >= 3:
            break
        path.append(cur)
        next_steps = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c, h, w)
            if bool(tree_mask[nr, nc]) and (prev is None or (nr, nc) != prev)
        ]
        if not next_steps:
            break
        assert len(next_steps) == 1, "branch path must have a unique inward continuation"
        prev, cur = cur, next_steps[0]

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each probed leaf branch inward until the first tree fork, then erase the probes.

    Input structure:
    - bg is the dominant color.
    - One large monochrome foreground component forms a 4-connected, one-pixel-wide tree with no loops.
    - Additional foreground pixels are singleton colored probes placed on background cells orthogonally adjacent to tree leaves.

    Rule:
    - Detect the tree color and compute each tree pixel's 4-neighbor degree within that tree.
    - For every probe, find its unique adjacent leaf pixel and follow the tree inward, always taking the only next tree cell that is not the previous one.
    - Recolor every traversed tree pixel with the probe color, but stop before recoloring the first degree-3-or-more branchpoint.
    - Erase all probe pixels to background; unprobed branches and branchpoints stay in the original tree color.

    Key insight:
    - The foreground skeleton is a tree, so each probe-to-interior route is a unique path whose endpoint is determined purely by the first high-degree junction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _dominant_color(g)
    tree_color = _tree_color(g, bg)
    tree_mask = g == tree_color

    tree_components = _components4(tree_mask)
    assert len(tree_components) == 1, "expected one tree component"
    degrees = _degree_map(tree_mask)
    branchpoints = np.argwhere(tree_mask & (degrees >= 3))
    assert branchpoints.size > 0, "expected at least one branchpoint"

    edge_count = int(degrees[tree_mask].sum() // 2)
    assert edge_count == int(tree_mask.sum()) - 1, "foreground tree must be acyclic"

    out = g.copy()
    probe_positions = sorted(
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != tree_color))
    )
    assert probe_positions, "expected at least one probe"

    painted_by: dict[tuple[int, int], int] = {}
    used_leaves: set[tuple[int, int]] = set()

    for pr, pc in probe_positions:
        probe_color = int(g[pr, pc])
        adjacent_leaves = [
            (nr, nc)
            for nr, nc in _neighbors4(pr, pc, h, w)
            if bool(tree_mask[nr, nc]) and int(degrees[nr, nc]) == 1
        ]
        assert len(adjacent_leaves) == 1, "each probe must touch exactly one tree leaf"
        leaf = adjacent_leaves[0]
        assert leaf not in used_leaves, "a leaf may have at most one probe"
        used_leaves.add(leaf)

        out[pr, pc] = bg
        for cell in _trace_to_branchpoint(tree_mask, degrees, leaf):
            prev_color = painted_by.get(cell)
            assert prev_color in (None, probe_color), "different probes may not overlap before a branchpoint"
            painted_by[cell] = probe_color
            out[cell] = probe_color

    return out


def _coarse_neighbors(node: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = node
    return _neighbors4(r, c, h, w)


def _add_edge(adj: dict[tuple[int, int], set[tuple[int, int]]], a: tuple[int, int], b: tuple[int, int]) -> None:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def _build_coarse_tree(rng: random.Random) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    coarse_h = rng.randint(*COARSE_H_RANGE)
    coarse_w = rng.randint(*COARSE_W_RANGE)
    root_candidates = [(r, c) for r in range(1, coarse_h - 1) for c in range(1, coarse_w - 1)]
    if not root_candidates:
        return None

    root = rng.choice(root_candidates)
    root_neighbors = _coarse_neighbors(root, coarse_h, coarse_w)
    root_degree = 4 if rng.random() < 0.5 else 3
    if len(root_neighbors) < root_degree:
        root_degree = min(3, len(root_neighbors))
    if root_degree < 3:
        return None

    adj: dict[tuple[int, int], set[tuple[int, int]]] = {root: set()}
    for child in rng.sample(root_neighbors, root_degree):
        _add_edge(adj, root, child)

    target_nodes = rng.randint(
        max(root_degree + 3, TARGET_NODE_RANGE[0]),
        min(TARGET_NODE_RANGE[1], coarse_h * coarse_w),
    )
    while len(adj) < target_nodes:
        candidates = [
            node
            for node in adj
            if any(nbr not in adj for nbr in _coarse_neighbors(node, coarse_h, coarse_w))
        ]
        if not candidates:
            break

        weights = []
        for node in candidates:
            degree = len(adj[node])
            if degree == 1:
                weights.append(6)
            elif degree == 2:
                weights.append(3)
            else:
                weights.append(2)
        parent = rng.choices(candidates, weights=weights, k=1)[0]
        options = [nbr for nbr in _coarse_neighbors(parent, coarse_h, coarse_w) if nbr not in adj]
        if not options:
            continue
        _add_edge(adj, parent, rng.choice(options))

        if len(adj) >= root_degree + 4 and rng.random() < 0.15:
            break

    leaves = [node for node, nbrs in adj.items() if len(nbrs) == 1]
    branchpoint_count = sum(1 for nbrs in adj.values() if len(nbrs) >= 3)
    if len(leaves) < 3 or branchpoint_count == 0:
        return None
    return adj


def _render_tree(
    adj: dict[tuple[int, int], set[tuple[int, int]]],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    nodes = sorted(adj)
    min_r = min(r for r, _ in nodes)
    min_c = min(c for _, c in nodes)
    node_positions = {
        node: (1 + (node[0] - min_r) * SPACING, 1 + (node[1] - min_c) * SPACING)
        for node in nodes
    }

    tree_cells: set[tuple[int, int]] = set()
    for a in nodes:
        ar, ac = node_positions[a]
        tree_cells.add((ar, ac))
        for b in adj[a]:
            if a > b:
                continue
            br, bc = node_positions[b]
            dr = 0 if ar == br else (1 if br > ar else -1)
            dc = 0 if ac == bc else (1 if bc > ac else -1)
            rr, cc = ar, ac
            tree_cells.add((rr, cc))
            while (rr, cc) != (br, bc):
                rr += dr
                cc += dc
                tree_cells.add((rr, cc))

    probes: dict[tuple[int, int], tuple[int, int]] = {}
    for leaf, neighbors in adj.items():
        if len(neighbors) != 1:
            continue
        parent = next(iter(neighbors))
        lr, lc = node_positions[leaf]
        dr = leaf[0] - parent[0]
        dc = leaf[1] - parent[1]
        probes[leaf] = (lr + dr, lc + dc)

    return tree_cells, probes


def _embed_coords(
    coords: set[tuple[int, int]],
    offset: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in coords}


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    adj = _build_coarse_tree(rng)
    if adj is None:
        return None

    tree_cells, probe_slots = _render_tree(adj)
    leaves = list(probe_slots)
    if not leaves:
        return None

    max_probes = min(len(leaves), len(PROBE_COLORS))
    if max_probes == 0:
        return None
    if max_probes == 1:
        probe_count = 1
    elif rng.random() < 0.7:
        probe_count = rng.randint(1, max_probes - 1)
    else:
        probe_count = rng.randint(1, max_probes)

    chosen_leaves = rng.sample(leaves, probe_count)
    chosen_colors = rng.sample(PROBE_COLORS, probe_count)
    probe_cells = {probe_slots[leaf] for leaf in chosen_leaves}

    all_cells = tree_cells | probe_cells
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    if min_r < 0 or min_c < 0:
        return None
    local_h = max(r for r, _ in all_cells) + 1
    local_w = max(c for _, c in all_cells) + 1
    if local_h > 30 or local_w > 30:
        return None

    extra_h = rng.randint(0, min(EXTRA_MARGIN_CAP, 30 - local_h))
    extra_w = rng.randint(0, min(EXTRA_MARGIN_CAP, 30 - local_w))
    top = rng.randint(0, extra_h)
    left = rng.randint(0, extra_w)
    bottom = extra_h - top
    right = extra_w - left
    height = local_h + top + bottom
    width = local_w + left + right
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r, c in _embed_coords(tree_cells, (top, left)):
        grid[r, c] = TREE_COLOR
    for leaf, color in zip(chosen_leaves, chosen_colors):
        pr, pc = probe_slots[leaf]
        rr, cc = pr + top, pc + left
        if grid[rr, cc] != BG:
            return None
        grid[rr, cc] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
