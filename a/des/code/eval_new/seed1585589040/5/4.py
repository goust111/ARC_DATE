from __future__ import annotations

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
CYAN = 8
BROWN = 9

OUTPUT_PALETTE = [BLUE, GREEN, YELLOW, ORANGE, RED]
INPUT_PALETTE = [BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, BROWN]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [
        c
        for c in range(w)
        if np.all(grid[:, c] == GREY) and c > 0 and c < w - 1 and c == w - c - 1
    ]
    assert len(candidates) == 1, "expected one centered grey separator column"
    return candidates[0]


def _degrees(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros(mask.shape, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    deg *= mask
    return deg


def _neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    h, w = mask.shape
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            pts.append((nr, nc))
    return pts


def _trace_one_path(
    mask: np.ndarray, degrees: np.ndarray, start_leaf: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    path = [start_leaf]
    prev: tuple[int, int] | None = None
    curr = start_leaf

    while True:
        if prev is not None and int(degrees[curr]) == 1:
            return path, curr

        next_candidates = [pt for pt in _neighbors(mask, curr[0], curr[1]) if pt != prev]
        assert next_candidates, "path must continue until it reaches its other leaf"

        if prev is None or len(next_candidates) == 1:
            nxt = next_candidates[0]
        else:
            dr = curr[0] - prev[0]
            dc = curr[1] - prev[1]
            straight = (curr[0] + dr, curr[1] + dc)
            assert straight in next_candidates, "crossing traversal must continue straight"
            nxt = straight

        prev, curr = curr, nxt
        path.append(curr)


def _reconstruct_paths(panel: np.ndarray) -> tuple[list[dict], np.ndarray]:
    mask = panel == GREY
    degrees = _degrees(mask)
    leaves = sorted(map(tuple, np.argwhere(mask & (degrees == 1))))
    assert leaves and len(leaves) % 2 == 0, "expected marker leaves in pairs"
    leaf_set = set(leaves)
    used_leaves: set[tuple[int, int]] = set()
    paths: list[dict] = []

    for leaf in leaves:
        if leaf in used_leaves:
            continue
        cells, other_leaf = _trace_one_path(mask, degrees, leaf)
        assert other_leaf in leaf_set and other_leaf != leaf, "trace must end at the paired leaf"
        used_leaves.add(leaf)
        used_leaves.add(other_leaf)
        paths.append(
            {
                "cells": tuple(cells),
                "leafs": tuple(sorted((leaf, other_leaf))),
            }
        )

    assert len(used_leaves) == len(leaves), "every leaf must belong to exactly one path"
    return paths, degrees


def _assign_input_colors(paths: list[dict], left_panel: np.ndarray) -> None:
    seen: set[int] = set()
    for path in paths:
        leaf_colors = {
            int(left_panel[r, c])
            for r, c in path["leafs"]
            if int(left_panel[r, c]) not in (BLACK, GREY)
        }
        assert len(leaf_colors) == 1, "each path marker pair must identify one source color"
        color = next(iter(leaf_colors))
        assert color not in seen, "source path colors must be distinct"
        seen.add(color)
        path["input_color"] = color


def _topological_order(n: int, edges: list[set[int]]) -> list[int]:
    indeg = [0] * n
    for src in range(n):
        for dst in edges[src]:
            indeg[dst] += 1

    order: list[int] = []
    remaining = set(range(n))
    while remaining:
        zeros = sorted(i for i in remaining if indeg[i] == 0)
        assert len(zeros) == 1, "cover relations must determine a unique total order"
        node = zeros[0]
        remaining.remove(node)
        order.append(node)
        for dst in edges[node]:
            indeg[dst] -= 1

    return order


def _infer_layer_order(paths: list[dict], left_panel: np.ndarray, degrees: np.ndarray) -> list[int]:
    cell_to_paths: dict[tuple[int, int], list[int]] = {}
    for idx, path in enumerate(paths):
        for cell in path["cells"]:
            cell_to_paths.setdefault(cell, []).append(idx)

    edges = [set() for _ in paths]
    for r, c in map(tuple, np.argwhere(degrees == 4)):
        owners = cell_to_paths.get((int(r), int(c)), [])
        assert len(owners) == 2, "each crossing must belong to exactly two paths"
        a, b = owners
        visible = int(left_panel[r, c])
        color_a = int(paths[a]["input_color"])
        color_b = int(paths[b]["input_color"])
        assert visible in (color_a, color_b), "left crossing color must show the top path"
        if visible == color_a:
            edges[b].add(a)
        else:
            edges[a].add(b)

    return _topological_order(len(paths), edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer path depth from the colored left panel and repaint the grey right panel by depth order.

    Input structure:
    - The background is black and the grid is split into equal left and right panels by one full grey column.
    - The left panel contains several one-cell-wide orthogonal colored paths whose endpoints carry same-color 2x1 markers.
    - The right panel contains the same path skeleton in grey, including grey endpoint markers and grey crossings.

    Rule:
    - Reconstruct each grey path on the right by tracing from marker leaf to marker leaf, continuing straight through every crossing.
    - Use the matching marker positions on the left to identify each path color, then read every crossing: the visible color is the higher path and the hidden one is lower.
    - Topologically sort all cover relations to get the unique global order from lowest to highest, then repaint the right-panel paths in the fixed sequence blue, green, yellow, orange, red.
    - At right-panel crossings, the higher path's new color remains visible because higher layers are painted last; the left panel and separator stay unchanged.

    Key insight:
    - The local over/under information at colored crossings defines a global depth order, and the grey panel can be solved once each skeleton path is traced independently of its final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "panels must have equal size"

    paths, degrees = _reconstruct_paths(right)
    assert 1 <= len(paths) <= len(OUTPUT_PALETTE), "path count must fit the fixed output palette"
    _assign_input_colors(paths, left)
    order = _infer_layer_order(paths, left, degrees)

    recolored = right.copy()
    for rank, path_idx in enumerate(order):
        new_color = OUTPUT_PALETTE[rank]
        for r, c in paths[path_idx]["cells"]:
            recolored[r, c] = new_color

    out = g.copy()
    out[:, sep + 1 :] = recolored
    return out


def _sample_spaced_positions(count: int, size: int) -> list[int] | None:
    choices = list(range(2, size - 2))
    if len(choices) < count:
        return None
    for _ in range(200):
        pts = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(pts, pts[1:])):
            return pts
    return None


def _path_specs(
    panel_h: int, panel_w: int
) -> tuple[list[dict], list[int]] | None:
    n_paths = random.randint(3, 5)
    start_orient = random.choice(("h", "v"))
    layer_orients = [
        start_orient if i % 2 == 0 else ("v" if start_orient == "h" else "h")
        for i in range(n_paths)
    ]
    n_h = sum(orient == "h" for orient in layer_orients)
    n_v = n_paths - n_h

    rows = _sample_spaced_positions(n_h, panel_h)
    cols = _sample_spaced_positions(n_v, panel_w)
    if rows is None or cols is None:
        return None

    random.shuffle(rows)
    random.shuffle(cols)
    colors = random.sample(INPUT_PALETTE, n_paths)

    specs: list[dict] = []
    h_idx = 0
    v_idx = 0
    for orient, color in zip(layer_orients, colors):
        if orient == "h":
            row = rows[h_idx]
            h_idx += 1
            specs.append(
                {
                    "orient": "h",
                    "color": color,
                    "cells": [(row, c) for c in range(1, panel_w - 1)],
                    "leafs": [
                        (row + random.choice((-1, 1)), 1),
                        (row + random.choice((-1, 1)), panel_w - 2),
                    ],
                }
            )
        else:
            col = cols[v_idx]
            v_idx += 1
            specs.append(
                {
                    "orient": "v",
                    "color": color,
                    "cells": [(r, col) for r in range(1, panel_h - 1)],
                    "leafs": [
                        (1, col + random.choice((-1, 1))),
                        (panel_h - 2, col + random.choice((-1, 1))),
                    ],
                }
            )

    return specs, layer_orients


def _render_panels(panel_h: int, panel_w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    left = np.full((panel_h, panel_w), BLACK, dtype=int)
    right = np.full((panel_h, panel_w), BLACK, dtype=int)

    for spec in specs:
        for r, c in spec["cells"]:
            right[r, c] = GREY
        for r, c in spec["leafs"]:
            right[r, c] = GREY

    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["cells"]:
            left[r, c] = color
        for r, c in spec["leafs"]:
            left[r, c] = color

    return left, right


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    sep = inp.shape[1] // 2
    panel_h, panel_w = inp.shape[0], sep
    right = np.full((panel_h, panel_w), BLACK, dtype=int)

    for rank, spec in enumerate(specs):
        color = OUTPUT_PALETTE[rank]
        for r, c in spec["cells"]:
            right[r, c] = color
        for r, c in spec["leafs"]:
            right[r, c] = color

    out = inp.copy()
    out[:, sep + 1 :] = right
    return out


def _is_valid_input(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except AssertionError:
        return False
    return isinstance(out, np.ndarray) and out.shape == inp.shape


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(10, 14)
    panel_w = random.randint(10, 14)
    sampled = _path_specs(panel_h, panel_w)
    if sampled is None:
        return None
    specs, _layer_orients = sampled

    left, right = _render_panels(panel_h, panel_w, specs)
    grid = np.full((panel_h, 2 * panel_w + 1), BLACK, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w] = GREY
    grid[:, panel_w + 1 :] = right

    if not _is_valid_input(grid):
        return None
    if not np.array_equal(main(grid), _expected_output(grid, specs)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-path ARC input")
