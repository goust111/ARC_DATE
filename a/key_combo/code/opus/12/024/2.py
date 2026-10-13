from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

STAMP_ENDPOINTS = {
    BLUE: (-1, 0),
    GREEN: (0, 1),
    YELLOW: (1, 0),
    ORANGE: (0, -1),
}
LINE_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
MIN_CENTER_SEPARATION = 5
GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (12, 16)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _anchors_from_grid(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid) == RED)]


def _pairwise_separated(anchors: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(anchors):
        for r1, c1 in anchors[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < MIN_CENTER_SEPARATION:
                return False
    return True


def _stamp_grid(shape: tuple[int, int], anchors: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for r, c in anchors:
        if _in_bounds(shape, r, c):
            out[r, c] = RED
        for color, (dr, dc) in STAMP_ENDPOINTS.items():
            rr, cc = r + dr, c + dc
            if _in_bounds(shape, rr, cc):
                out[rr, cc] = color
    return out


def _collect_connector_paths(base: np.ndarray) -> dict[str, object]:
    paths: list[tuple[int, list[tuple[int, int]]]] = []
    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    blocked_pairs = 0
    chain_groups = 0
    horizontal_segments = 0
    vertical_segments = 0

    for color in LINE_COLORS:
        coords = np.argwhere(base == color)
        rows: dict[int, list[int]] = defaultdict(list)
        cols: dict[int, list[int]] = defaultdict(list)
        for r, c in coords:
            rows[int(r)].append(int(c))
            cols[int(c)].append(int(r))

        for r, row_cols in rows.items():
            ordered = sorted(row_cols)
            if len(ordered) >= 3:
                chain_groups += 1
            for c0, c1 in zip(ordered, ordered[1:]):
                cells = [(r, c) for c in range(c0 + 1, c1)]
                if not cells:
                    continue
                values = [int(base[cell]) for cell in cells]
                if all(value in (BLACK, color) for value in values):
                    if any(value == BLACK for value in values):
                        horizontal_segments += 1
                        paths.append((color, cells))
                        for cell, value in zip(cells, values):
                            if value == BLACK:
                                claims[cell].add(color)
                else:
                    blocked_pairs += 1

        for c, col_rows in cols.items():
            ordered = sorted(col_rows)
            if len(ordered) >= 3:
                chain_groups += 1
            for r0, r1 in zip(ordered, ordered[1:]):
                cells = [(r, c) for r in range(r0 + 1, r1)]
                if not cells:
                    continue
                values = [int(base[cell]) for cell in cells]
                if all(value in (BLACK, color) for value in values):
                    if any(value == BLACK for value in values):
                        vertical_segments += 1
                        paths.append((color, cells))
                        for cell, value in zip(cells, values):
                            if value == BLACK:
                                claims[cell].add(color)
                else:
                    blocked_pairs += 1

    conflicting_cells = {cell for cell, colors in claims.items() if len(colors) > 1}
    return {
        "paths": paths,
        "blocked_pairs": blocked_pairs,
        "chain_groups": chain_groups,
        "horizontal_segments": horizontal_segments,
        "vertical_segments": vertical_segments,
        "conflicting_cells": conflicting_cells,
        "line_cells": set(claims),
    }


def _apply_connector_paths(base: np.ndarray, paths: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    out = base.copy()
    for color, cells in paths:
        for r, c in cells:
            if out[r, c] == BLACK:
                out[r, c] = color
    return out


def _analyze_anchors(shape: tuple[int, int], anchors: list[tuple[int, int]]) -> dict[str, object]:
    stamped = _stamp_grid(shape, anchors)
    connector_info = _collect_connector_paths(stamped)
    clipped_anchors = 0
    for r, c in anchors:
        if any(not _in_bounds(shape, r + dr, c + dc) for dr, dc in STAMP_ENDPOINTS.values()):
            clipped_anchors += 1
    return {
        "stamped": stamped,
        "output": _apply_connector_paths(stamped, connector_info["paths"]),
        "clipped_anchors": clipped_anchors,
        **connector_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a fixed four-endpoint pattern at each red anchor, then connect aligned same-colored endpoint pixels through black gaps.

    Input structure:
    - The grid is black except for several red anchor pixels.
    - Each red pixel is the center of an implicit 3x3 stamp whose four edge-midpoint cells are colored.
    - Anchors may lie near the boundary, so the hidden stamp can be clipped by the grid edge.

    Rule:
    - Replace every red anchor by a fixed stamp: center red, top blue, right green, bottom yellow, left orange, with black corners.
    - After all stamps are placed, for each non-black, non-red endpoint color, connect adjacent same-color pixels that share a row or a column.
    - A connector fills only black cells between the two endpoints; if any intervening cell is occupied by another color, that connector is skipped.

    Key insight:
    - The visible anchors only indirectly specify the result: they first generate colored endpoints, and those generated endpoints then determine the horizontal and vertical connection graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    anchors = _anchors_from_grid(g)
    stamped = _stamp_grid(g.shape, anchors)
    connector_info = _collect_connector_paths(stamped)
    return _apply_connector_paths(stamped, connector_info["paths"])


def _symmetry_transforms(points: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    return [
        [(r, c) for r, c in points],
        [(r, -c) for r, c in points],
        [(-r, c) for r, c in points],
        [(-r, -c) for r, c in points],
        [(c, r) for r, c in points],
        [(c, -r) for r, c in points],
        [(-c, r) for r, c in points],
        [(-c, -r) for r, c in points],
    ]


def _fit_points(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    force_edge: bool,
) -> list[tuple[int, int]] | None:
    height, width = shape
    valid = []
    for transformed in _symmetry_transforms(points):
        rows = [r for r, _ in transformed]
        cols = [c for _, c in transformed]
        if max(rows) - min(rows) > height - 1:
            continue
        if max(cols) - min(cols) > width - 1:
            continue
        valid.append(transformed)
    if not valid:
        return None

    transformed = random.choice(valid)
    rows = [r for r, _ in transformed]
    cols = [c for _, c in transformed]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    row_low, row_high = -r_min, height - 1 - r_max
    col_low, col_high = -c_min, width - 1 - c_max

    row_shift = None
    col_shift = None
    if force_edge:
        edges: list[str] = []
        if row_low <= -r_min <= row_high:
            edges.append("top")
        if row_low <= height - 1 - r_max <= row_high:
            edges.append("bottom")
        if col_low <= -c_min <= col_high:
            edges.append("left")
        if col_low <= width - 1 - c_max <= col_high:
            edges.append("right")
        if edges:
            edge = random.choice(edges)
            if edge == "top":
                row_shift = -r_min
            elif edge == "bottom":
                row_shift = height - 1 - r_max
            elif edge == "left":
                col_shift = -c_min
            else:
                col_shift = width - 1 - c_max

    if row_shift is None:
        row_shift = random.randint(row_low, row_high)
    if col_shift is None:
        col_shift = random.randint(col_low, col_high)

    return [(r + row_shift, c + col_shift) for r, c in transformed]


def _motif_chain3() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    step1 = random.randint(5, 6)
    step2 = random.randint(5, 6)
    return [(0, 0), (0, step1), (0, step1 + step2)], {"need_chain": True}


def _motif_blocked3() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    span = random.randint(10, 12)
    blocker_col = random.randint(5, span - 5)
    return [(0, 0), (0, span), (-1, blocker_col)], {"need_blocked": True}


def _motif_rectangle4() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    dr = random.randint(5, 6)
    dc = random.randint(5, 6)
    return [(0, 0), (0, dc), (dr, 0), (dr, dc)], {}


def _motif_two_pairs4() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    h_gap = random.randint(7, 9)
    v_top = random.randint(5, 6)
    v_bottom = v_top + random.randint(5, 6)
    v_col = h_gap + random.randint(5, 6)
    return [(0, 0), (0, h_gap), (v_top, v_col), (v_bottom, v_col)], {}


def _motif_mixed5() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    step1 = random.randint(5, 6)
    step2 = random.randint(5, 6)
    pair_col = step1 + step2 + random.randint(3, 4)
    pair_top = random.randint(5, 6)
    pair_bottom = pair_top + random.randint(5, 6)
    return [
        (0, 0),
        (0, step1),
        (0, step1 + step2),
        (pair_top, pair_col),
        (pair_bottom, pair_col),
    ], {"need_chain": True}


def _random_motif() -> tuple[list[tuple[int, int]], dict[str, bool]]:
    motif_builders = [
        _motif_chain3,
        _motif_chain3,
        _motif_blocked3,
        _motif_blocked3,
        _motif_rectangle4,
        _motif_two_pairs4,
        _motif_mixed5,
    ]
    return random.choice(motif_builders)()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*GEN_SIZE_RANGE)
        width = random.randint(*GEN_SIZE_RANGE)
        shape = (height, width)

        raw_points, requirements = _random_motif()
        anchors = _fit_points(raw_points, shape, force_edge=random.random() < 0.35)
        if anchors is None:
            continue
        if not _pairwise_separated(anchors):
            continue

        analysis = _analyze_anchors(shape, anchors)
        if analysis["conflicting_cells"]:
            continue
        if len(analysis["line_cells"]) == 0:
            continue
        if requirements.get("need_blocked") and analysis["blocked_pairs"] == 0:
            continue
        if requirements.get("need_chain") and analysis["chain_groups"] == 0:
            continue

        grid = np.full(shape, BLACK, dtype=int)
        for r, c in anchors:
            grid[r, c] = RED

        out = main(grid)
        if np.array_equal(out, analysis["stamped"]):
            continue
        if not np.array_equal(out, analysis["output"]):
            continue

        return grid

    raise RuntimeError("failed to generate a valid non-degenerate input")
