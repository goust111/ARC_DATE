from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PITCH = 5
GENERATION_ATTEMPTS = 500
NODE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CHILD_DIRS = ("down", "left", "right")


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            idx = len(comps)
            stack = [(r, c)]
            labels[r, c] = idx
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = idx
                        stack.append((nr, nc))
            comps.append(coords)
    return labels, comps


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0] - 1) if np.all(grid[r] == GREY)]
    assert rows, "expected a grey separator row above the palette"
    return int(rows[-1])


def _read_palette(grid: np.ndarray, palette_row: int) -> list[int]:
    row = grid[palette_row]
    cols = np.flatnonzero(row != BLACK)
    assert 3 <= len(cols) <= 5, "palette must contain 3-5 colors"
    assert np.all(np.diff(cols) == 2), "palette cells must be separated by one black pixel"
    colors = [int(row[c]) for c in cols]
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    assert all(color not in {BLACK, GREY, MAROON} for color in colors)
    return colors


def _build_graph(main_region: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[int, set[int]], dict[int, set[str]]]:
    node_mask = (main_region != BLACK) & (main_region != MAROON)
    labels, comps = _components(node_mask)
    assert comps, "expected at least one node component"
    graph = {idx: set() for idx in range(len(comps))}
    incident_dirs = {idx: set() for idx in range(len(comps))}
    h, w = main_region.shape

    for idx, coords in enumerate(comps):
        for r, c in coords:
            for name, (dr, dc) in DIRS.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or main_region[nr, nc] != MAROON:
                    continue
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w and main_region[rr, cc] == MAROON:
                    rr += dr
                    cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                other = int(labels[rr, cc])
                if other == -1 or other == idx:
                    continue
                graph[idx].add(other)
                incident_dirs[idx].add(name)

    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    assert edge_count == len(comps) - 1, "expected a tree of connected nodes"
    return comps, graph, incident_dirs


def _find_root(comps: list[list[tuple[int, int]]], incident_dirs: dict[int, set[str]]) -> int:
    exact = [idx for idx, dirs in incident_dirs.items() if dirs == {"down"}]
    if len(exact) == 1:
        return int(exact[0])

    relaxed = [
        idx
        for idx, dirs in incident_dirs.items()
        if "down" in dirs and "up" not in dirs and "left" not in dirs and "right" not in dirs
    ]
    assert relaxed, "could not identify the root node"
    top = min(min(r for r, _ in comps[idx]) for idx in relaxed)
    best = [idx for idx in relaxed if min(r for r, _ in comps[idx]) == top]
    assert len(best) == 1, "root must be unique"
    return int(best[0])


def _depths_from_root(graph: dict[int, set[int]], root: int) -> dict[int, int]:
    depths = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt in depths:
                continue
            depths[nxt] = depths[node] + 1
            queue.append(nxt)
    assert len(depths) == len(graph), "tree must be connected"
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each node by its depth in the maroon-connected tree using the bottom palette cyclically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BLACK), "expected black background"

    sep_row = _find_separator_row(g)
    palette = _read_palette(g, sep_row + 1)
    main_region = g[:sep_row, :]

    comps, graph, incident_dirs = _build_graph(main_region)
    root = _find_root(comps, incident_dirs)
    depths = _depths_from_root(graph, root)

    out = g.copy()
    for idx, coords in enumerate(comps):
        color = int(palette[depths[idx] % len(palette)])
        for r, c in coords:
            out[r, c] = color
    return out


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return "up"
    if dr == 1 and dc == 0:
        return "down"
    if dr == 0 and dc == -1:
        return "left"
    if dr == 0 and dc == 1:
        return "right"
    raise ValueError("positions must be grid-adjacent")


def _grow_tree(
    rng: np.random.Generator,
    lattice_h: int,
    lattice_w: int,
    target_nodes: int,
    *,
    require_horizontal: bool,
    require_branch: bool,
    require_wrap_depth: int | None,
) -> tuple[list[tuple[int, int]], list[int | None], list[list[int]]] | None:
    if lattice_h < 2 or lattice_w < 3 or target_nodes < 2:
        return None

    root_c = int(rng.integers(1, lattice_w - 1))
    positions = [(0, root_c), (1, root_c)]
    parents: list[int | None] = [None, 0]
    children: list[list[int]] = [[1], []]
    used = {positions[0]: 0, positions[1]: 1}

    while len(positions) < target_nodes:
        candidates: list[tuple[int, str]] = []
        for idx, (r, c) in enumerate(positions):
            degree = len(children[idx]) + (0 if parents[idx] is None else 1)
            if idx == 0 or degree >= 3:
                continue

            used_dirs = set()
            if parents[idx] is not None:
                used_dirs.add(_direction((r, c), positions[parents[idx]]))
            for child in children[idx]:
                used_dirs.add(_direction((r, c), positions[child]))

            for name in CHILD_DIRS:
                if name in used_dirs:
                    continue
                dr, dc = DIRS[name]
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < lattice_h and 0 <= nxt[1] < lattice_w):
                    continue
                if nxt in used:
                    continue
                weight = 4 if name == "down" else 3
                candidates.extend([(idx, name)] * weight)

        if not candidates:
            return None

        parent_idx, move = candidates[int(rng.integers(len(candidates)))]
        pr, pc = positions[parent_idx]
        dr, dc = DIRS[move]
        nxt = (pr + dr, pc + dc)
        child_idx = len(positions)
        positions.append(nxt)
        parents.append(parent_idx)
        children.append([])
        children[parent_idx].append(child_idx)
        used[nxt] = child_idx

    horizontal_edges = sum(
        1
        for idx, parent_idx in enumerate(parents)
        if parent_idx is not None and positions[idx][0] == positions[parent_idx][0]
    )
    branching_nodes = sum(1 for kids in children if len(kids) >= 2)
    depths = [0] * len(positions)
    for idx in range(1, len(positions)):
        depths[idx] = depths[parents[idx]] + 1  # type: ignore[index]

    if require_horizontal and horizontal_edges == 0:
        return None
    if require_branch and branching_nodes == 0:
        return None
    if require_wrap_depth is not None and max(depths) < require_wrap_depth:
        return None
    return positions, parents, children


def _normalize_positions(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    return [(r - min_r, c - min_c) for r, c in positions]


def _center_of(lattice_pos: tuple[int, int], top_pad: int, left_pad: int) -> tuple[int, int]:
    lr, lc = lattice_pos
    return top_pad + 1 + lr * PITCH, left_pad + 1 + lc * PITCH


def _incident_dirs(
    positions: list[tuple[int, int]],
    parents: list[int | None],
    children: list[list[int]],
    idx: int,
) -> set[str]:
    r, c = positions[idx]
    dirs = set()
    if parents[idx] is not None:
        dirs.add(_direction((r, c), positions[parents[idx]]))
    for child in children[idx]:
        dirs.add(_direction((r, c), positions[child]))
    return dirs


def _render_tree(
    rng: np.random.Generator,
    positions: list[tuple[int, int]],
    parents: list[int | None],
    children: list[list[int]],
    palette: np.ndarray,
) -> np.ndarray:
    positions = _normalize_positions(positions)
    max_r = max(r for r, _ in positions)
    max_c = max(c for _, c in positions)
    top_pad = int(rng.integers(0, 3))
    left_pad = int(rng.integers(0, 4))
    bottom_pad = int(rng.integers(0, 3))
    right_pad = int(rng.integers(0, 4))

    last_center_r, last_center_c = _center_of((max_r, max_c), top_pad, left_pad)
    main_h = last_center_r + 2 + bottom_pad
    main_w = last_center_c + 2 + right_pad
    palette_span = 2 * len(palette) - 1
    main_w = max(main_w, palette_span + int(rng.integers(0, 4)))
    grid = np.full((main_h + 2, main_w), BLACK, dtype=int)
    sep_row = main_h
    grid[sep_row, :] = GREY

    start = int(rng.integers(0, main_w - palette_span + 1))
    for i, color in enumerate(palette):
        grid[sep_row + 1, start + 2 * i] = int(color)

    node_colors = rng.choice(np.array(NODE_COLORS, dtype=int), size=len(positions), replace=False)
    centers = [_center_of(pos, top_pad, left_pad) for pos in positions]

    for idx, (cr, cc) in enumerate(centers):
        grid[cr, cc] = int(node_colors[idx])
        for name in _incident_dirs(positions, parents, children, idx):
            dr, dc = DIRS[name]
            grid[cr + dr, cc + dc] = int(node_colors[idx])

    for idx, parent_idx in enumerate(parents):
        if parent_idx is None:
            continue
        r0, c0 = centers[parent_idx]
        r1, c1 = centers[idx]
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            grid[r0, lo + 2 : hi - 1] = MAROON
        else:
            lo, hi = sorted((r0, r1))
            grid[lo + 2 : hi - 1, c0] = MAROON

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mode = str(rng.choice(np.array(["basic", "horizontal", "branch", "wrap"], dtype=object)))
    palette_len = int(rng.integers(3, 6))

    if mode == "branch":
        target_nodes = int(rng.integers(5, 8))
    elif mode == "wrap":
        target_nodes = int(rng.integers(max(5, palette_len + 1), 8))
    else:
        target_nodes = int(rng.integers(3, 8))

    require_horizontal = mode in {"horizontal", "branch"} or (mode == "wrap" and rng.random() < 0.5)
    require_branch = mode == "branch"
    require_wrap_depth = palette_len if mode == "wrap" else None

    min_h = max(3, require_wrap_depth + 1 if require_wrap_depth is not None else 3)
    lattice_h = int(rng.integers(min_h, 8))
    lattice_w = int(rng.integers(3, 7))
    if target_nodes > lattice_h * lattice_w:
        return None

    grown = _grow_tree(
        rng,
        lattice_h,
        lattice_w,
        target_nodes,
        require_horizontal=require_horizontal,
        require_branch=require_branch,
        require_wrap_depth=require_wrap_depth,
    )
    if grown is None:
        return None

    positions, parents, children = grown
    palette = rng.choice(np.array(NODE_COLORS, dtype=int), size=palette_len, replace=False)
    grid = _render_tree(rng, positions, parents, children, palette)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
