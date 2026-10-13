from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7)
ARM_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

MIN_MARKERS = 5
MAX_MARKERS = 6
MIN_STEP = 5
MAX_STEP = 8
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return [(int(r - r0), int(c - c0)) for r, c in coords]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    if random.random() < 0.5:
        arr = arr[:, ::-1]
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    return _normalize_coords([tuple(map(int, rc)) for rc in arr])


def _choose_steps(count: int) -> tuple[int, ...]:
    for _ in range(200):
        steps = tuple(random.randint(MIN_STEP, MAX_STEP) for _ in range(count))
        if len(set(steps)) < 2:
            continue
        if {step % 2 for step in steps} != {0, 1}:
            continue
        return steps
    raise AssertionError("failed to choose varied mixed-parity steps")


def _raw_cross5() -> list[tuple[int, int]]:
    a, b, c, d = _choose_steps(4)
    return [(0, 0), (0, -a), (0, b), (-c, 0), (d, 0)]


def _raw_cross6() -> list[tuple[int, int]]:
    a, b, c, d, e = _choose_steps(5)
    coords = [(0, 0), (0, -a), (0, b), (-c, 0), (d, 0)]
    if random.random() < 0.5:
        coords.append((0, b + e))
    else:
        coords.append((d + e, 0))
    return coords


def _raw_elbow5() -> list[tuple[int, int]]:
    a, b, c, d = _choose_steps(4)
    return [(0, 0), (0, a), (0, a + b), (c, 0), (c + d, 0)]


def _raw_rectangle6() -> list[tuple[int, int]]:
    a, b, c, d = _choose_steps(4)
    return [(0, 0), (0, a), (0, a + b), (c, 0), (c + d, 0), (c + d, a + b)]


def _make_raw_centers() -> list[tuple[int, int]]:
    motif = random.choice((_raw_cross5, _raw_cross6, _raw_elbow5, _raw_rectangle6))
    return _transform_coords(motif())


def _adjacent_pairs(markers: dict[tuple[int, int], int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in markers:
        rows[r].append(c)
        cols[c].append(r)

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, group in rows.items():
        ordered = sorted(group)
        for c0, c1 in zip(ordered, ordered[1:]):
            pairs.append(((r, c0), (r, c1)))
    for c, group in cols.items():
        ordered = sorted(group)
        for r0, r1 in zip(ordered, ordered[1:]):
            pairs.append(((r0, c), (r1, c)))
    return pairs


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo + 2, hi - 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo + 2, hi - 1)]
    raise AssertionError("anchors must be aligned")


def _stripe_color(
    cell: tuple[int, int],
    a: tuple[int, int],
    b: tuple[int, int],
    color_a: int,
    color_b: int,
) -> int:
    r, c = cell
    d1 = abs(r - a[0]) + abs(c - a[1])
    d2 = abs(r - b[0]) + abs(c - b[1])
    if d1 == d2:
        return int(color_a if a < b else color_b)
    if d1 < d2:
        near_color, far_color, d = color_a, color_b, d1
    else:
        near_color, far_color, d = color_b, color_a, d2
    return int(near_color if d % 2 == 0 else far_color)


def _find_markers(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    markers: dict[tuple[int, int], int] = {}
    footprint: set[tuple[int, int]] = set()
    arm_cells: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(g[r, c])
            if center in {bg, GREY}:
                continue
            if any(int(g[r + dr, c + dc]) != GREY for dr, dc in ARM_OFFSETS):
                continue
            if any(int(g[r + dr, c + dc]) != bg for dr, dc in CORNER_OFFSETS):
                continue
            markers[(r, c)] = center
            footprint.add((r, c))
            for dr, dc in ARM_OFFSETS:
                rr, cc = r + dr, c + dc
                assert (rr, cc) not in arm_cells, "markers may not overlap"
                arm_cells.add((rr, cc))
                footprint.add((rr, cc))

    assert markers, "expected at least one hollow-plus marker"
    assert len(set(markers.values())) == len(markers), "anchor colors must be unique"

    non_bg = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    assert non_bg == footprint, "all foreground pixels must belong to valid markers"
    for rr, cc in arm_cells:
        assert int(g[rr, cc]) == GREY, "marker arms must be grey"
    return markers, bg


def _render_from_markers(grid: np.ndarray, markers: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    painted: set[tuple[int, int]] = set()
    pairs = _adjacent_pairs(markers)
    assert pairs, "expected at least one adjacent aligned pair"

    for a, b in pairs:
        cells = _cells_between(a, b)
        assert cells, "adjacent anchors must leave a drawable gap"
        for cell in cells:
            assert int(grid[cell]) == bg, "connectors may only pass through background"
            assert cell not in painted, "connector paths may not overlap"
            painted.add(cell)
            out[cell] = _stripe_color(cell, a, b, markers[a], markers[b])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each adjacent aligned hollow-plus anchor pair with a distance-striped two-color line.

    Input structure:
    - The background is black, and every foreground object is a 3x3 hollow plus whose four arm pixels are grey.
    - Each hollow plus has a single non-grey colored center pixel chosen uniquely from a small visible palette.
    - Some marker centers share rows or columns, and only background cells lie between consecutive aligned centers.

    Rule:
    - Detect every hollow-plus marker and read its center as the anchor position and anchor color.
    - For each row group and each column group, sort anchors and connect only consecutive anchors, starting just outside one grey arm and ending just before the opposing grey arm.
    - For every connector cell, compare its Manhattan distances to the two anchor centers: use the nearer anchor color when that nearer distance is even, otherwise use the farther anchor color; if both distances are equal, use the lexicographically earlier anchor's color.
    - Leave all grey arm pixels and colored center pixels unchanged.

    Key insight:
    - The topology comes from the hollow-plus centers, but the line colors come from both endpoint anchors and from parity of distance along the shared straight segment, so adjacency and striping are both necessary.
    """
    markers, bg = _find_markers(np.asarray(grid, dtype=int))
    return _render_from_markers(np.asarray(grid, dtype=int), markers, bg)


def _build_grid(centers: list[tuple[int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(centers, colors):
        grid[r, c] = int(color)
        for dr, dc in ARM_OFFSETS:
            grid[r + dr, c + dc] = GREY
    return grid


def _place_centers(raw_centers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in raw_centers]
    cols = [c for _, c in raw_centers]
    span_r = max(rows) - min(rows)
    span_c = max(cols) - min(cols)

    top = random.randint(1, 4)
    bottom = random.randint(1, 4)
    left = random.randint(1, 4)
    right = random.randint(1, 4)

    h = span_r + top + bottom + 1
    w = span_c + left + right + 1
    assert h <= 30 and w <= 30
    return [(r + top, c + left) for r, c in raw_centers]


def _validate_generated(grid: np.ndarray) -> None:
    markers, bg = _find_markers(grid)
    assert bg == BG, "generator must keep black as the dominant background"
    assert MIN_MARKERS <= len(markers) <= MAX_MARKERS

    row_sizes = Counter(r for r, _ in markers)
    col_sizes = Counter(c for _, c in markers)
    assert max(row_sizes.values()) >= 3, "need a blocked row chain"
    assert max(col_sizes.values()) >= 3, "need a blocked column chain"

    pairs = _adjacent_pairs(markers)
    horiz = sum(1 for a, b in pairs if a[0] == b[0])
    vert = len(pairs) - horiz
    assert horiz > 0 and vert > 0, "need both horizontal and vertical connections"

    path_cells: set[tuple[int, int]] = set()
    parities: set[int] = set()
    for a, b in pairs:
        span = abs(a[0] - b[0]) + abs(a[1] - b[1])
        parities.add(span % 2)
        cells = _cells_between(a, b)
        assert cells, "every adjacent pair must expose at least one connector pixel"
        for cell in cells:
            assert int(grid[cell]) == BG
            assert cell not in path_cells, "generated connector paths must not cross"
            path_cells.add(cell)
    assert parities == {0, 1}, "need both even and odd anchor separations"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "output must add at least one connector"

    for (r, c), color in markers.items():
        assert int(out[r, c]) == color
        for dr, dc in ARM_OFFSETS:
            assert int(out[r + dr, c + dc]) == GREY

    for a, b in pairs:
        for cell in _cells_between(a, b):
            assert int(out[cell]) == _stripe_color(cell, a, b, markers[a], markers[b])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        raw_centers = _make_raw_centers()
        centers = _place_centers(raw_centers)
        h = max(r for r, _ in centers) + 2
        w = max(c for _, c in centers) + 2
        if h > 30 or w > 30:
            continue

        colors = random.sample(list(ANCHOR_COLORS), len(centers))
        grid = _build_grid(centers, colors, (h, w))
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
