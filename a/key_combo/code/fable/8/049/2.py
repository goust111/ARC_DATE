from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_RANGE = (13, 22)
QUADRANT_TEMPLATES = (
    ("UL", "DR", "DR", "DR"),
    ("UR", "DL", "DL", "DL"),
    ("UL", "UL", "UL", "DR"),
    ("UR", "UR", "UR", "DL"),
    ("UL", "UL", "UR", "DR", "DR"),
    ("UL", "UL", "DL", "DR", "DR"),
    ("UR", "UR", "UL", "DL", "DL"),
    ("UR", "UR", "DL", "DL", "DR"),
)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _vertical_dir(r: int, height: int) -> int:
    up = int(r)
    down = int(height - 1 - r)
    assert up != down, "row midpoint ties are not allowed"
    return -1 if up < down else 1


def _horizontal_dir(c: int, width: int) -> int:
    left = int(c)
    right = int(width - 1 - c)
    assert left != right, "column midpoint ties are not allowed"
    return -1 if left < right else 1


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != bg)
    assert len(coords) >= 2, "expected multiple isolated markers"

    markers: list[tuple[int, int, int]] = []
    for r, c in coords:
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
                if (rr, cc) != (r, c):
                    assert int(grid[rr, cc]) == int(bg), "foreground pixels must be isolated"
        markers.append((r, c, color))
    return sorted(markers)


def _pair_hits(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    shape: tuple[int, int],
) -> tuple[bool, bool]:
    ra, ca, _ = a
    rb, cb, _ = b
    height, width = map(int, shape)
    h_ok = cb < ca if _horizontal_dir(ca, width) < 0 else cb > ca
    v_ok = ra < rb if _vertical_dir(rb, height) < 0 else ra > rb
    return bool(h_ok), bool(v_ok)


def _compute_intersections(
    markers: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int, int, int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    intersections: list[tuple[int, int, int, int, int]] = []
    usage: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ai, a in enumerate(markers):
        ra, _ca, _acol = a
        for bi, b in enumerate(markers):
            if ai == bi:
                continue
            _rb, cb, bcol = b
            h_ok, v_ok = _pair_hits(a, b, shape)
            if h_ok and v_ok:
                cell = (int(ra), int(cb))
                usage.setdefault(cell, []).append((ai, bi))
                intersections.append((int(ra), int(cb), int(bcol), int(ai), int(bi)))
    return intersections, usage


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    markers = _extract_markers(g, bg)
    occupied = {(r, c) for r, c, _ in markers}
    intersections, usage = _compute_intersections(markers, g.shape)
    for cell, providers in usage.items():
        assert len(providers) == 1, "each crossing cell must belong to exactly one ordered pair"
        assert cell not in occupied, "intersection cells may not already contain a marker"

    out = g.copy()
    for r, c, color, _ai, _bi in intersections:
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the isolated seeds and color only the crossings of their invisible edge-seeking rays.

    Input structure:
    - A black background contains several colored single-pixel markers.
    - Every marker is isolated from every other marker, and no marker sits at a row or column midpoint tie.
    - Each marker therefore has one unique nearer horizontal edge and one unique nearer vertical edge.

    Rule:
    - For each marker, imagine one horizontal ray along its row toward the nearer left/right edge and one vertical ray along its column toward the nearer top/bottom edge; do not draw those rays.
    - For every ordered pair of distinct markers A and B, test whether A's horizontal ray and B's vertical ray meet at one cell.
    - If they do, color that crossing cell with B's color, keep all original markers, and leave every other non-crossing cell black.

    Key insight:
    - The visible answer comes only from pairwise intersections of unseen rays; the crucial asymmetry is that the crossing inherits the color of the marker supplying the vertical ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    return _expected_output(g)


def _side_rows(height: int, vertical_side: str) -> list[int]:
    rows = [r for r in range(1, height - 1) if 2 * r != height - 1]
    if vertical_side == "U":
        return [r for r in rows if 2 * r < height - 1]
    return [r for r in rows if 2 * r > height - 1]


def _side_cols(width: int, horizontal_side: str) -> list[int]:
    cols = [c for c in range(1, width - 1) if 2 * c != width - 1]
    if horizontal_side == "L":
        return [c for c in cols if 2 * c < width - 1]
    return [c for c in cols if 2 * c > width - 1]


def _well_spaced(value: int, used: list[int]) -> bool:
    return all(abs(int(value) - int(prev)) >= 2 for prev in used)


def _pick_quadrants() -> list[str]:
    return list(random.choice(QUADRANT_TEMPLATES))


def _place_markers(height: int, width: int, quadrants: list[str]) -> list[tuple[int, int]] | None:
    used_rows: list[int] = []
    used_cols: list[int] = []
    placements: list[tuple[int, int] | None] = [None] * len(quadrants)
    order = list(range(len(quadrants)))
    random.shuffle(order)

    for idx in order:
        quad = quadrants[idx]
        row_options = [r for r in _side_rows(height, quad[0]) if _well_spaced(r, used_rows)]
        col_options = [c for c in _side_cols(width, quad[1]) if _well_spaced(c, used_cols)]
        if not row_options or not col_options:
            return None
        random.shuffle(row_options)
        random.shuffle(col_options)

        placed = False
        for r in row_options:
            for c in col_options:
                placed_coords = [pos for pos in placements if pos is not None]
                if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in placed_coords):
                    continue
                placements[idx] = (int(r), int(c))
                used_rows.append(int(r))
                used_cols.append(int(c))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None]


def _relation_counts(markers: list[tuple[int, int, int]], shape: tuple[int, int]) -> dict[tuple[bool, bool], int]:
    counts: dict[tuple[bool, bool], int] = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for ai, a in enumerate(markers):
        for bi, b in enumerate(markers):
            if ai == bi:
                continue
            counts[_pair_hits(a, b, shape)] += 1
    return counts


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    bg = _get_bg(inp)
    if int(bg) != BG:
        return False

    try:
        markers = _extract_markers(inp, bg)
    except AssertionError:
        return False

    if not (3 <= len(markers) <= 5):
        return False
    if len({color for _r, _c, color in markers}) != len(markers):
        return False

    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return False
    if any(abs(a - b) < 2 for i, a in enumerate(rows) for b in rows[i + 1 :]):
        return False
    if any(abs(a - b) < 2 for i, a in enumerate(cols) for b in cols[i + 1 :]):
        return False

    h_dirs = {_horizontal_dir(c, inp.shape[1]) for c in cols}
    v_dirs = {_vertical_dir(r, inp.shape[0]) for r in rows}
    if h_dirs != {-1, 1} or v_dirs != {-1, 1}:
        return False

    relations = _relation_counts(markers, inp.shape)
    if relations[(True, True)] < 2:
        return False
    if relations[(True, False)] < 1 or relations[(False, True)] < 1:
        return False
    if relations[(True, True)] >= len(markers) * (len(markers) - 1):
        return False

    intersections, usage = _compute_intersections(markers, inp.shape)
    if not intersections:
        return False
    if len(intersections) > 10:
        return False
    if len(usage) != len(intersections):
        return False

    occupied = {(r, c) for r, c, _color in markers}
    if any(cell in occupied for cell in usage):
        return False
    if len({bi for _r, _c, _color, _ai, bi in intersections}) < 2:
        return False

    extra_cells = set(map(tuple, np.argwhere(out != inp)))
    expected_cells = {(r, c) for r, c, _color, _ai, _bi in intersections}
    if extra_cells != expected_cells:
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    quadrants = _pick_quadrants()
    n_markers = len(quadrants)
    placements = _place_markers(height, width, quadrants)
    if placements is None or len(placements) != n_markers:
        return None

    colors = random.sample(COLOR_POOL, n_markers)
    grid = np.full((height, width), BG, dtype=int)
    for (r, c), color in zip(placements, colors):
        grid[r, c] = int(color)

    expected = _expected_output(grid)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-ray-intersection input")
