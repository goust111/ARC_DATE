from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MIN_PAIRS = 2
MAX_PAIRS = 5
GRID_MIN = 10
GRID_MAX = 20
PAIR_ATTEMPTS = 300
MAX_ATTEMPTS = 400


def _paint(out: np.ndarray, r: int, c: int, color: int) -> None:
    height, width = out.shape
    assert 0 <= r < height and 0 <= c < width, "paint cell out of bounds"
    current = int(out[r, c])
    assert current in (BG, color), "generated strips and arrowheads must stay disjoint"
    out[r, c] = color


def _extract_pairs(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds ARC size limit: {g.shape}"

    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != BG):
        color = int(g[r, c])
        assert color != YELLOW, "yellow is reserved for output arrowheads"
        coords_by_color.setdefault(color, []).append((int(r), int(c)))

    assert coords_by_color, "expected at least one colored pair"
    pairs: list[dict[str, int | str]] = []
    for color, pts in sorted(coords_by_color.items()):
        assert len(pts) == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = pts
        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            assert c_lo < c_hi, "horizontal pair endpoints must be distinct"
            assert 0 < r0 < g.shape[0] - 1, "horizontal pair needs room for yellow caps"
            pairs.append({"color": color, "axis": "h", "r": r0, "c0": c_lo, "c1": c_hi})
        else:
            assert c0 == c1, "same-colored endpoints must align horizontally or vertically"
            r_lo, r_hi = sorted((r0, r1))
            assert r_lo < r_hi, "vertical pair endpoints must be distinct"
            assert 0 < c0 < g.shape[1] - 1, "vertical pair needs room for yellow caps"
            pairs.append({"color": color, "axis": "v", "c": c0, "r0": r_lo, "r1": r_hi})
    return pairs


def _draw_pair(out: np.ndarray, pair: dict[str, int | str]) -> None:
    color = int(pair["color"])
    if pair["axis"] == "h":
        r = int(pair["r"])
        c0 = int(pair["c0"])
        c1 = int(pair["c1"])
        for c in range(c0, c1 + 1):
            _paint(out, r, c, color)
        for rr in (r - 1, r + 1):
            _paint(out, rr, c0, YELLOW)
            _paint(out, rr, c1, YELLOW)
        return

    c = int(pair["c"])
    r0 = int(pair["r0"])
    r1 = int(pair["r1"])
    for r in range(r0, r1 + 1):
        _paint(out, r, c, color)
    for cc in (c - 1, c + 1):
        _paint(out, r0, cc, YELLOW)
        _paint(out, r1, cc, YELLOW)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each aligned color pair into a strip and cap both ends with yellow T-marks.

    Input structure:
    - The background is black.
    - Several non-background colors appear, and each such color occupies exactly two single cells.
    - Each same-colored pair is aligned either on one row or on one column.

    Rule:
    - Copy the input grid.
    - For every same-colored pair, fill the full straight segment between its two endpoints with that same color.
    - At each segment endpoint, add a yellow one-cell cap on both sides perpendicular to the segment, creating a T-shaped arrow end.

    Key insight:
    - The pair itself determines both the segment endpoints and its orientation; once that axis is known, the yellow cap direction is simply the perpendicular axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for pair in _extract_pairs(g):
        _draw_pair(out, pair)
    return out


def _footprint(spec: dict[str, int | str]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if spec["axis"] == "h":
        r = int(spec["r"])
        c0 = int(spec["c0"])
        c1 = int(spec["c1"])
        for c in range(c0, c1 + 1):
            cells.add((r, c))
        cells.add((r - 1, c0))
        cells.add((r + 1, c0))
        cells.add((r - 1, c1))
        cells.add((r + 1, c1))
        return cells

    c = int(spec["c"])
    r0 = int(spec["r0"])
    r1 = int(spec["r1"])
    for r in range(r0, r1 + 1):
        cells.add((r, c))
    cells.add((r0, c - 1))
    cells.add((r0, c + 1))
    cells.add((r1, c - 1))
    cells.add((r1, c + 1))
    return cells


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    halo.add((nr, nc))
    return halo


def _random_spec(axis: str, color: int, shape: tuple[int, int]) -> dict[str, int | str] | None:
    height, width = shape
    if axis == "h":
        max_len = min(7, width)
        if height < 3 or max_len < 2:
            return None
        length = random.randint(2, max_len)
        r = random.randint(1, height - 2)
        c0 = random.randint(0, width - length)
        return {"color": color, "axis": "h", "r": r, "c0": c0, "c1": c0 + length - 1}

    max_len = min(7, height)
    if width < 3 or max_len < 2:
        return None
    length = random.randint(2, max_len)
    c = random.randint(1, width - 2)
    r0 = random.randint(0, height - length)
    return {"color": color, "axis": "v", "c": c, "r0": r0, "r1": r0 + length - 1}


def _render_input(shape: tuple[int, int], specs: list[dict[str, int | str]]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        if spec["axis"] == "h":
            inp[int(spec["r"]), int(spec["c0"])] = color
            inp[int(spec["r"]), int(spec["c1"])] = color
        else:
            inp[int(spec["r0"]), int(spec["c"])] = color
            inp[int(spec["r1"]), int(spec["c"])] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
    axes = ["h", "v"] + [random.choice(("h", "v")) for _ in range(num_pairs - 2)]
    random.shuffle(axes)
    colors = random.sample(INPUT_COLORS, k=num_pairs)

    min_side = max(GRID_MIN, 8 + num_pairs)
    max_side = max(min_side, GRID_MAX)
    shape = (random.randint(min_side, max_side), random.randint(min_side, max_side))

    specs: list[dict[str, int | str]] = []
    blocked: set[tuple[int, int]] = set()

    # Place full output footprints first so every generated input stays readable after transformation.
    for axis, color in zip(axes, colors):
        placed = False
        for _ in range(PAIR_ATTEMPTS):
            spec = _random_spec(axis, color, shape)
            if spec is None:
                return None
            cells = _footprint(spec)
            if any(cell in blocked for cell in cells):
                continue
            specs.append(spec)
            blocked.update(_halo(cells, shape))
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(shape, specs)
    expected = inp.copy()
    for spec in specs:
        _draw_pair(expected, spec)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid strip-and-arrow input")
