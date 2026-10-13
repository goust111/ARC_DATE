from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    Color = _ARC_UTILS.Color


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
BG = Color.BLACK
MIN_SIDE = 10
MAX_SIDE = 16
MAX_ATTEMPTS = 1200


def _ordered_pair(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = sorted((int(r), int(c)) for r, c in points)
    return coords[0], coords[1]


def _path_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _horizontal_first_path(
    a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[int, int]]:
    ar, ac = map(int, a)
    br, bc = map(int, b)

    dc = 1 if bc >= ac else -1
    cells = [(ar, c) for c in range(ac, bc + dc, dc)]

    if br != ar:
        dr = 1 if br > ar else -1
        cells.extend((r, bc) for r in range(ar + dr, br + dr, dr))
    return cells


def _extract_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        pts = np.argwhere(grid == color)
        assert pts.shape == (2, 2), f"color {color} must appear exactly twice"
        pairs[color] = _ordered_pair(pts)
    return pairs


def _render_paths(
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[int, int], dict[tuple[int, int], list[int]]]:
    lengths = {color: _path_length(*pair) for color, pair in pairs.items()}
    assert len(set(lengths.values())) == len(lengths), "path lengths must be distinct"

    visits: dict[tuple[int, int], list[int]] = {}
    for color, (a, b) in pairs.items():
        for cell in _horizontal_first_path(a, b):
            visits.setdefault(cell, []).append(int(color))

    out = np.full(shape, BG, dtype=int)
    for cell, colors in visits.items():
        winner = min(colors, key=lambda color: lengths[color])
        out[cell] = int(winner)
    return out, lengths, visits


def _overlay_by_order(
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    shape: tuple[int, int],
    order: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        for r, c in _horizontal_first_path(*pairs[color]):
            out[r, c] = int(color)
    return out


def _intersection_roles(
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    lengths: dict[int, int],
    visits: dict[tuple[int, int], list[int]],
) -> list[tuple[tuple[int, int], int, int, int]]:
    roles: list[tuple[tuple[int, int], int, int, int]] = []
    for cell, colors in visits.items():
        if len(colors) <= 1:
            continue
        assert len(colors) == 2, "generator must avoid triple intersections"
        c1, c2 = colors
        a1 = pairs[c1][0]
        a2 = pairs[c2][0]
        upper, lower = (c1, c2) if a1[0] < a2[0] else (c2, c1)
        winner = upper if lengths[upper] < lengths[lower] else lower
        roles.append((cell, upper, lower, winner))
    return roles


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    pairs = _extract_pairs(g)
    assert 3 <= len(pairs) <= 4
    assert set(pairs).issubset(set(PALETTE))

    rows: list[int] = []
    cols: list[int] = []
    directions: list[int] = []
    endpoints: set[tuple[int, int]] = set()
    for color, (a, b) in pairs.items():
        assert a[0] != b[0] and a[1] != b[1], f"color {color} endpoints must differ in row and col"
        rows.extend([a[0], b[0]])
        cols.extend([a[1], b[1]])
        directions.append(1 if b[1] > a[1] else -1)
        endpoints.add(a)
        endpoints.add(b)

    assert len(set(rows)) == len(rows), "all marker rows must be distinct"
    assert len(set(cols)) == len(cols), "all marker cols must be distinct"
    assert 1 in directions and -1 in directions, "generator must cover both horizontal directions"

    out, lengths, visits = _render_paths(pairs, g.shape)
    roles = _intersection_roles(pairs, lengths, visits)
    assert len(roles) >= 2, "need multiple intersections to disambiguate the priority rule"
    assert all(cell not in endpoints for cell, _u, _l, _w in roles), "crossings must stay off endpoints"
    assert np.count_nonzero(out) > 2 * len(pairs)
    assert np.count_nonzero(out) < (g.shape[0] * g.shape[1]) // 2

    winner_is_upper = [winner == upper for _cell, upper, _lower, winner in roles]
    assert any(winner_is_upper) and any(not flag for flag in winner_is_upper), (
        "one sample must contain both upper-shorter and lower-shorter crossings"
    )

    top_order = sorted(pairs, key=lambda color: pairs[color][0][0])
    asc_overlay = _overlay_by_order(pairs, g.shape, top_order)
    desc_overlay = _overlay_by_order(pairs, g.shape, list(reversed(top_order)))
    assert not np.array_equal(out, asc_overlay), "sample must disagree with top-to-bottom drawing order"
    assert not np.array_equal(out, desc_overlay), "sample must disagree with bottom-to-top drawing order"


def main(grid: np.ndarray) -> np.ndarray:
    """Route each same-color marker pair with a horizontal-first orthogonal path and color crossings by the shorter path.

    Input structure:
    - A black background contains 3 or 4 colors chosen from blue, red, green, and yellow.
    - Each present color appears exactly twice as singleton marker pixels.
    - All marker rows are distinct from one another, and all marker columns are distinct from one another.

    Rule:
    - For each color independently, sort its two markers lexicographically and call them A then B.
    - Draw that color's horizontal-first orthogonal path: move from A across A's row to B's column, then down or up that column to B's row, including both endpoints.
    - If a cell belongs to exactly one path, keep that path's color; if it belongs to two paths, use the color of the path whose two endpoints have the smaller Manhattan distance.
    - Leave every cell that no path visits black, and keep the output size equal to the input size.

    Key insight:
    - The crucial signal is not drawing order but endpoint distance: every crossing must be resolved by comparing the full Manhattan lengths of the two competing routed paths.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _extract_pairs(g)
    out, _lengths, _visits = _render_paths(pairs, g.shape)
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(3, 4)
    h = random.randint(max(MIN_SIDE, 2 * color_count + 2), MAX_SIDE)
    w = random.randint(max(MIN_SIDE, 2 * color_count + 2), MAX_SIDE)

    top_pool = list(range(1, h // 2))
    bottom_pool = list(range(h // 2 + 1, h - 1))
    col_pool = list(range(1, w - 1))
    if len(top_pool) < color_count or len(bottom_pool) < color_count or len(col_pool) < 2 * color_count:
        return None

    colors = random.sample(PALETTE, color_count)
    top_rows = sorted(random.sample(top_pool, color_count))
    bottom_rows = random.sample(bottom_pool, color_count)
    chosen_cols = random.sample(col_pool, 2 * color_count)
    random.shuffle(chosen_cols)
    start_cols = chosen_cols[:color_count]
    end_cols = chosen_cols[color_count:]

    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for idx, color in enumerate(colors):
        a = (int(top_rows[idx]), int(start_cols[idx]))
        b = (int(bottom_rows[idx]), int(end_cols[idx]))
        pairs[int(color)] = (a, b)

    grid = np.full((h, w), BG, dtype=int)
    for color, (a, b) in pairs.items():
        grid[a] = int(color)
        grid[b] = int(color)

    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routed-crossing ARC input")
