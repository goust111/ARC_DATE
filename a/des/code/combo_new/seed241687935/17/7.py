import random
from collections import Counter, defaultdict

import numpy as np


BG_COLOR = 0
NODE_COLOR = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
PAIR_COUNTS = (1, 2)  # 1 -> 2H+2V, 2 -> 4H+4V
MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_node_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        ok = True
        for comp in comps:
            if len(comp) != 4:
                ok = False
                break
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0, c1 - c0) != (1, 1):
                ok = False
                break
        if ok:
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one 2x2 node color"
    return candidates[0]


def _node_ports(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    node_color: int,
) -> list[tuple[int, str, int, int]]:
    r0, c0, r1, c1 = bbox
    positions = [
        ((r0, c0 - 1), "L"),
        ((r1, c0 - 1), "L"),
        ((r0, c1 + 1), "R"),
        ((r1, c1 + 1), "R"),
        ((r0 - 1, c0), "U"),
        ((r0 - 1, c1), "U"),
        ((r1 + 1, c0), "D"),
        ((r1 + 1, c1), "D"),
    ]
    ports: list[tuple[int, str, int, int]] = []
    seen_colors: set[int] = set()
    h, w = grid.shape
    for (rr, cc), side in positions:
        if 0 <= rr < h and 0 <= cc < w:
            color = int(grid[rr, cc])
            if color not in (bg, node_color):
                assert color not in seen_colors, "a node should expose each path color at most once"
                seen_colors.add(color)
                ports.append((color, side, rr, cc))
    assert len(ports) == 2, "each node must connect exactly two path colors"
    return ports


def _recover_paths(
    grid: np.ndarray,
    bg: int,
    node_color: int,
) -> tuple[dict[int, dict[str, object]], set[tuple[int, int]]]:
    node_cells: set[tuple[int, int]] = set()
    endpoints: dict[int, list[tuple[str, int, int]]] = defaultdict(list)

    for comp in _components(grid == node_color):
        box = _bbox(comp)
        node_cells.update(comp)
        for color, side, rr, cc in _node_ports(grid, box, bg, node_color):
            endpoints[color].append((side, rr, cc))

    paths: dict[int, dict[str, object]] = {}
    for color, recs in endpoints.items():
        assert len(recs) == 2, "each path color must touch exactly two nodes"
        sides = [side for side, _, _ in recs]
        if all(side in ("L", "R") for side in sides):
            rows = {rr for _, rr, _ in recs}
            assert len(rows) == 1, "horizontal endpoints must share one row"
            row = next(iter(rows))
            c0, c1 = sorted(cc for _, _, cc in recs)
            cells = {(row, c) for c in range(c0, c1 + 1)}
            orient = "H"
        else:
            assert all(side in ("U", "D") for side in sides), "path endpoints must agree on orientation"
            cols = {cc for _, _, cc in recs}
            assert len(cols) == 1, "vertical endpoints must share one column"
            col = next(iter(cols))
            r0, r1 = sorted(rr for _, rr, _ in recs)
            cells = {(r, col) for r in range(r0, r1 + 1)}
            orient = "V"

        visible = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        assert visible <= cells, "visible pixels must belong to the recovered straight path"
        assert not (cells & node_cells), "paths must stay outside node cells"
        paths[color] = {"orient": orient, "cells": cells}

    return paths, node_cells


def _unique_bottom_to_top_order(
    grid: np.ndarray,
    paths: dict[int, dict[str, object]],
    node_cells: set[tuple[int, int]],
) -> list[int]:
    edges: set[tuple[int, int]] = set()
    colors = sorted(paths)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            path_a = paths[color_a]
            path_b = paths[color_b]
            if path_a["orient"] == path_b["orient"]:
                continue

            inter = [p for p in path_a["cells"] & path_b["cells"] if p not in node_cells]
            if not inter:
                continue
            assert len(inter) == 1, "two straight orthogonal paths can intersect at only one cell"
            rr, cc = inter[0]
            visible = int(grid[rr, cc])
            assert visible in (color_a, color_b), "a crossing must show one of its two path colors"
            edges.add((color_b, color_a) if visible == color_a else (color_a, color_b))

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        indeg = {color: 0 for color in remaining}
        for below, above in edges:
            if below in remaining and above in remaining:
                indeg[above] += 1
        roots = [color for color in remaining if indeg[color] == 0]
        assert len(roots) == 1, "crossing relations must determine a unique layer order"
        color = roots[0]
        order.append(color)
        remaining.remove(color)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stacked path order from hidden crossings and keep only the bottom path.

    Input structure:
    - The background is black.
    - Grey 2x2 square nodes anchor the ends of several one-cell-wide horizontal
      and vertical colored paths.
    - Each non-grey, non-background color belongs to one straight path, and
      crossings show only the color of the path on top.

    Rule:
    - Detect every node and use the node-adjacent colored port pixels to recover
      each path's two endpoints, orientation, and full straight segment.
    - For every recovered horizontal/vertical crossing, read the visible color
      to infer which path is above the other, then combine all such relations
      into one unique bottom-to-top ordering.
    - Output only the lowest path together with all grey nodes; erase all other
      path pixels to background, including formerly visible upper crossings.

    Key insight:
    - The missing pixels of a hidden lower path are recoverable from its two
      node endpoints, and the visible crossing colors provide the partial order
      needed to identify the unique bottom layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    node_color = _infer_node_color(g, bg)
    paths, node_cells = _recover_paths(g, bg, node_color)
    order = _unique_bottom_to_top_order(g, paths, node_cells)
    bottom = order[0]

    out = np.full_like(g, bg)
    out[g == node_color] = node_color
    for r, c in paths[bottom]["cells"]:
        out[r, c] = bottom
    return out


def _choose_pair_bases(lo: int, hi: int, pair_count: int) -> list[int]:
    if pair_count == 1:
        return [random.randint(lo, hi)]
    first = random.randint(lo, hi - 5)
    second = random.randint(first + 5, hi)
    return [first, second]


def _paint_path(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    h_min = 14 if pair_count == 1 else 20
    h_max = 24 if pair_count == 1 else 28
    w_min = 14 if pair_count == 1 else 20
    w_max = 24 if pair_count == 1 else 28
    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)

    row_bases = _choose_pair_bases(5, h - 7, pair_count)
    col_bases = _choose_pair_bases(5, w - 7, pair_count)
    h_rows = [r for base in row_bases for r in (base, base + 1)]
    v_cols = [c for base in col_bases for c in (base, base + 1)]

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for base in row_bases:
        grid[base : base + 2, 1:3] = NODE_COLOR
        grid[base : base + 2, w - 3 : w - 1] = NODE_COLOR
    for base in col_bases:
        grid[1:3, base : base + 2] = NODE_COLOR
        grid[h - 3 : h - 1, base : base + 2] = NODE_COLOR

    h_ids = list(range(len(h_rows)))
    v_ids = list(range(len(v_cols)))
    random.shuffle(h_ids)
    random.shuffle(v_ids)
    start = random.choice(("H", "V"))
    layer_specs: list[tuple[str, int]] = []
    while h_ids:
        if start == "H":
            layer_specs.append(("H", h_ids.pop()))
            layer_specs.append(("V", v_ids.pop()))
        else:
            layer_specs.append(("V", v_ids.pop()))
            layer_specs.append(("H", h_ids.pop()))

    total_paths = len(layer_specs)
    palette = random.sample(PATH_COLORS, total_paths)
    h_colors = {i: palette[i] for i in range(len(h_rows))}
    v_colors = {i: palette[len(h_rows) + i] for i in range(len(v_cols))}

    path_cells: dict[int, list[tuple[int, int]]] = {}
    for i, row in enumerate(h_rows):
        path_cells[h_colors[i]] = [(row, c) for c in range(3, w - 3)]
    for i, col in enumerate(v_cols):
        path_cells[v_colors[i]] = [(r, col) for r in range(3, h - 3)]

    layer_colors = [
        h_colors[idx] if orient == "H" else v_colors[idx]
        for orient, idx in layer_specs
    ]
    for color in layer_colors:
        _paint_path(grid, path_cells[color], color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-layer path network")
