from __future__ import annotations

import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

TRIANGLE_COLORS = (BLUE, RED, GREEN)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
BASE_LENGTHS = (3, 4)
HEIGHTS = (3, 4, 5)
GRID_MIN = 18
GRID_MAX = 24
YELLOW_COUNT_RANGE = (4, 8)
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _span_offsets(width: int) -> range:
    left = (width - 1) // 2
    right = width - 1 - left
    return range(-left, right + 1)


def _width_profile(base_len: int, height: int) -> list[int]:
    if height == 1:
        return [1]
    widths = []
    for level in range(height):
        frac = (height - 1 - level) / (height - 1)
        width = 1 + math.ceil((base_len - 1) * frac)
        widths.append(int(width))
    widths[0] = int(base_len)
    widths[-1] = 1
    for i in range(1, len(widths)):
        widths[i] = min(widths[i], widths[i - 1])
        widths[i] = max(widths[i], 1)
    return widths


def _tip_of(spec: dict) -> tuple[int, int]:
    dr, dc = DIRECTIONS[spec["direction"]]
    gr, gc = spec["gray"]
    step = spec["height"] - 1
    return gr + dr * step, gc + dc * step


def _triangle_cells(spec: dict) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    dr, dc = DIRECTIONS[spec["direction"]]
    gr, gc = spec["gray"]
    full = set()
    color_cells = set()
    for level, width in enumerate(_width_profile(spec["base_len"], spec["height"])):
        ar = gr + dr * level
        ac = gc + dc * level
        for off in _span_offsets(width):
            if dr != 0:
                cell = (ar, ac + off)
            else:
                cell = (ar + off, ac)
            full.add(cell)
            if not (level == 0 and off == 0):
                color_cells.add(cell)
    return full, color_cells


def _band_cells(spec: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRECTIONS[spec["direction"]]
    width = spec["base_len"]
    r, c = _tip_of(spec)
    cells = set()
    while 0 <= r < h and 0 <= c < w:
        for off in _span_offsets(width):
            rr, cc = (r, c + off) if dr != 0 else (r + off, c)
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
        r += dr
        c += dc
    return cells


def _component_coords(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    seen = set()
    comps = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        comp = set()
        seen.add(start)
        while stack:
            rr, cc = stack.pop()
            comp.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                    nxt = (nr, nc)
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        comps.append(comp)
    return comps


def _spec_signature(spec: dict) -> tuple:
    return (
        spec["color"],
        spec["direction"],
        spec["base_len"],
        spec["height"],
        tuple(spec["gray"]),
    )


def _parse_triangle(grid: np.ndarray, comp: set[tuple[int, int]]) -> dict:
    gray_cells = [cell for cell in comp if int(grid[cell]) == GREY]
    assert len(gray_cells) == 1, "each triangle must contain exactly one gray base marker"
    gray = gray_cells[0]

    colors = {int(grid[cell]) for cell in comp if int(grid[cell]) != GREY}
    assert len(colors) == 1 and next(iter(colors)) in TRIANGLE_COLORS
    color = next(iter(colors))

    exposed = []
    gr, gc = gray
    for name, (dr, dc) in DIRECTIONS.items():
        if (gr + dr, gc + dc) not in comp:
            exposed.append(name)
    assert len(exposed) == 1, "gray marker must have exactly one exposed side"
    direction = OPPOSITE[exposed[0]]

    if direction in {"up", "down"}:
        base_len = sum(1 for r, _ in comp if r == gr)
        tip_row = min(r for r, _ in comp) if direction == "up" else max(r for r, _ in comp)
        tip_candidates = [(r, c) for r, c in comp if r == tip_row]
    else:
        base_len = sum(1 for _, c in comp if c == gc)
        tip_col = min(c for _, c in comp) if direction == "left" else max(c for _, c in comp)
        tip_candidates = [(r, c) for r, c in comp if c == tip_col]

    assert len(tip_candidates) == 1, "triangle tip must be a single pixel"
    tip = tip_candidates[0]
    height = abs(tip[0] - gr) + abs(tip[1] - gc) + 1

    spec = {
        "color": color,
        "direction": direction,
        "base_len": int(base_len),
        "height": int(height),
        "gray": gray,
    }
    full, _color_cells = _triangle_cells(spec)
    assert full == comp, "triangle geometry does not match the supported filled wedge family"
    return spec


def _extract_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    triangle_mask = (g != BG) & (g != YELLOW)
    comps = _component_coords(triangle_mask)
    specs = [_parse_triangle(g, comp) for comp in comps]
    assert 2 <= len(specs) <= 3
    colors = [spec["color"] for spec in specs]
    assert len(set(colors)) == len(colors), "triangle colors must be distinct"
    return specs


def _sorted_specs(specs: list[dict]) -> list[dict]:
    return sorted(specs, key=lambda spec: (_tip_of(spec)[0], _tip_of(spec)[1], spec["color"]))


def _yellow_output_color(count: int) -> int:
    if count == 0:
        return YELLOW
    if count == 1:
        return ORANGE
    return MAROON


def _render_output_from_specs(input_grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(input_grid, dtype=int).copy()
    footprints = [_band_cells(spec, out.shape) for spec in specs]

    for spec in _sorted_specs(specs):
        for cell in sorted(_band_cells(spec, out.shape)):
            if int(out[cell]) == BG:
                out[cell] = spec["color"]

    for r, c in np.argwhere(np.asarray(input_grid) == YELLOW):
        cell = (int(r), int(c))
        count = sum(cell in footprint for footprint in footprints)
        out[cell] = _yellow_output_color(count)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Project each marked triangle's same-color strip through its tip and recolor yellow singletons by strip-crossing count.

    Input structure:
    - A black background contains 2-3 filled monochrome triangles, each with a single gray pixel replacing the center of its base.
    - Triangle colors are distinct and come from blue, red, and green.
    - Several yellow singleton pixels are isolated from one another and from the triangles.
    - Every triangle points up, down, left, or right, and the gray base marker has one exposed side outside the triangle.

    Rule:
    - For each triangle, use the gray marker's exposed side to identify the base side, then project away from that base, through the triangle, and out past the one-pixel tip.
    - The projected strip keeps the triangle's color and has constant width equal to the triangle's base length.
    - When two strips compete for the same background cell, the strip whose tip appears earlier in row-major order keeps that cell.
    - Count how many projected strips pass through each yellow singleton: 0 keeps yellow, 1 becomes orange, and 2 or more become maroon.
    - Triangles, gray markers, and visible strip pixels remain; recolored singleton pixels are shown on top of the strips.

    Key insight:
    - The gray pixel determines the projection direction while the triangle's base length determines the strip thickness, so solving requires recovering both orientation and width before counting strip crossings at the yellow pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    return _render_output_from_specs(g, _extract_specs(g))


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                cell = (r + dr, c + dc)
                if _in_bounds(shape, cell):
                    out.add(cell)
    return out


def _interval(spec: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    band = _band_cells(spec, shape)
    rows = [r for r, _ in band]
    cols = [c for _, c in band]
    return min(rows), min(cols), max(rows), max(cols)


def _make_spec(color: int, direction: str, base_len: int, height: int, tip: tuple[int, int]) -> dict:
    dr, dc = DIRECTIONS[direction]
    tr, tc = tip
    gray = (tr - dr * (height - 1), tc - dc * (height - 1))
    spec = {
        "color": int(color),
        "direction": direction,
        "base_len": int(base_len),
        "height": int(height),
        "gray": gray,
    }
    return spec


def _spec_ok(spec: dict, shape: tuple[int, int]) -> bool:
    full, _ = _triangle_cells(spec)
    return full and all(_in_bounds(shape, cell) for cell in full)


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    return bool(_halo(a, shape) & b)


def _sample_tip_row(rng: random.Random, direction: str, height: int, cross_row: int, other_width: int, h: int) -> int | None:
    clearance = 1 + other_width // 2
    if direction == "up":
        lo, hi = cross_row + clearance, h - height
    else:
        lo, hi = height - 1, cross_row - clearance
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _sample_tip_col(rng: random.Random, direction: str, height: int, cross_col: int, other_width: int, w: int) -> int | None:
    clearance = 1 + other_width // 2
    if direction == "right":
        lo, hi = height - 1, cross_col - clearance
    else:
        lo, hi = cross_col + clearance, w - height
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _build_cross_specs(shape: tuple[int, int], rng: random.Random) -> list[dict] | None:
    h, w = shape
    n = rng.choice((2, 3))
    colors = rng.sample(TRIANGLE_COLORS, n)

    if n == 2:
        cross_row = rng.randint(5, h - 6)
        cross_col = rng.randint(5, w - 6)

        v_width = rng.choice(BASE_LENGTHS)
        h_width = rng.choice(BASE_LENGTHS)
        v_height = rng.choice(HEIGHTS)
        h_height = rng.choice(HEIGHTS)
        v_dir = rng.choice(("up", "down"))
        h_dir = rng.choice(("left", "right"))

        v_tip_row = _sample_tip_row(rng, v_dir, v_height, cross_row, h_width, h)
        h_tip_col = _sample_tip_col(rng, h_dir, h_height, cross_col, v_width, w)
        if v_tip_row is None or h_tip_col is None:
            return None

        specs = [
            _make_spec(colors[0], v_dir, v_width, v_height, (v_tip_row, cross_col)),
            _make_spec(colors[1], h_dir, h_width, h_height, (cross_row, h_tip_col)),
        ]
        return specs

    layout = rng.choice(("vvh", "hhv"))
    cross_row = rng.randint(5, h - 6)
    cross_col = rng.randint(5, w - 6)

    if layout == "vvh":
        col_gap = rng.randint(5, 7)
        cols = [cross_col - col_gap // 2, cross_col + (col_gap - col_gap // 2)]
        if not (3 <= cols[0] < cols[1] <= w - 4):
            return None
        h_width = rng.choice(BASE_LENGTHS)
        h_height = rng.choice(HEIGHTS)
        h_dir = rng.choice(("left", "right"))

        v_specs = []
        max_clear = 0
        for i, col in enumerate(cols):
            v_width = rng.choice(BASE_LENGTHS)
            v_height = rng.choice(HEIGHTS)
            v_dir = rng.choice(("up", "down"))
            tip_row = _sample_tip_row(rng, v_dir, v_height, cross_row, h_width, h)
            if tip_row is None:
                return None
            v_specs.append(_make_spec(colors[i], v_dir, v_width, v_height, (tip_row, col)))
            max_clear = max(max_clear, 1 + v_width // 2)
        if h_dir == "right":
            lo, hi = h_height - 1, min(cols) - max_clear
        else:
            lo, hi = max(cols) + max_clear, w - h_height
        if lo > hi:
            return None
        tip_col = rng.randint(lo, hi)
        return v_specs + [_make_spec(colors[2], h_dir, h_width, h_height, (cross_row, tip_col))]

    row_gap = rng.randint(5, 7)
    rows = [cross_row - row_gap // 2, cross_row + (row_gap - row_gap // 2)]
    if not (3 <= rows[0] < rows[1] <= h - 4):
        return None
    v_width = rng.choice(BASE_LENGTHS)
    v_height = rng.choice(HEIGHTS)
    v_dir = rng.choice(("up", "down"))

    h_specs = []
    max_clear = 0
    for i, row in enumerate(rows):
        h_width = rng.choice(BASE_LENGTHS)
        h_height = rng.choice(HEIGHTS)
        h_dir = rng.choice(("left", "right"))
        tip_col = _sample_tip_col(rng, h_dir, h_height, cross_col, v_width, w)
        if tip_col is None:
            return None
        h_specs.append(_make_spec(colors[i], h_dir, h_width, h_height, (row, tip_col)))
        max_clear = max(max_clear, 1 + h_width // 2)
    tip_row = _sample_tip_row(rng, v_dir, v_height, cross_row, max(spec["base_len"] for spec in h_specs), h)
    if tip_row is None:
        return None
    return h_specs + [_make_spec(colors[2], v_dir, v_width, v_height, (tip_row, cross_col))]


def _valid_specs(specs: list[dict], shape: tuple[int, int]) -> bool:
    triangle_cells = []
    for spec in specs:
        if not _spec_ok(spec, shape):
            return False
        triangle_cells.append(_triangle_cells(spec)[0])

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if triangle_cells[i] & triangle_cells[j]:
                return False
            if _touching(triangle_cells[i], triangle_cells[j], shape):
                return False

    bands = [_band_cells(spec, shape) for spec in specs]
    for i, cells in enumerate(triangle_cells):
        for j, band in enumerate(bands):
            if i != j and cells & band:
                return False

    if len(specs) == 3:
        intervals = [_interval(spec, shape) for spec in specs]
        verticals = [idx for idx, spec in enumerate(specs) if spec["direction"] in {"up", "down"}]
        horizontals = [idx for idx, spec in enumerate(specs) if spec["direction"] in {"left", "right"}]
        if len(verticals) == 2:
            a, b = verticals
            if not (intervals[a][3] < intervals[b][1] or intervals[b][3] < intervals[a][1]):
                return False
        if len(horizontals) == 2:
            a, b = horizontals
            if not (intervals[a][2] < intervals[b][0] or intervals[b][2] < intervals[a][0]):
                return False

    overlap_exists = False
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            if bands[i] & bands[j]:
                overlap_exists = True
    return overlap_exists


def _coverage_map(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    cov = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _band_cells(spec, shape):
            cov[r, c] += 1
    return cov


def _pick_from_pool(
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    rng.shuffle(candidates)
    for cell in candidates:
        if cell in forbidden:
            continue
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen):
            continue
        return cell
    return None


def _choose_yellows(specs: list[dict], shape: tuple[int, int], rng: random.Random) -> list[tuple[int, int]] | None:
    cov = _coverage_map(specs, shape)
    triangle_cells = set()
    for spec in specs:
        triangle_cells |= _triangle_cells(spec)[0]
    forbidden = _halo(triangle_cells, shape)

    category = {0: [], 1: [], 2: []}
    for r in range(shape[0]):
        for c in range(shape[1]):
            cell = (r, c)
            if cell in triangle_cells:
                continue
            count = int(cov[cell])
            key = 2 if count >= 2 else count
            category[key].append(cell)

    def edge_pref(cells: list[tuple[int, int]], want_covered_neighbor: bool) -> list[tuple[int, int]]:
        preferred = []
        fallback = []
        for cell in cells:
            covered_neighbor = any(
                0 <= nr < shape[0]
                and 0 <= nc < shape[1]
                and cov[nr, nc] > 0
                for nr, nc in _neighbors4(*cell)
            )
            if covered_neighbor == want_covered_neighbor:
                preferred.append(cell)
            else:
                fallback.append(cell)
        return preferred + fallback

    chosen: list[tuple[int, int]] = []
    for pool in (category[2][:], edge_pref(category[1], False), edge_pref(category[0], True)):
        cell = _pick_from_pool(pool, chosen, forbidden, rng)
        if cell is None:
            return None
        chosen.append(cell)

    target = rng.randint(*YELLOW_COUNT_RANGE)
    pools = {
        2: category[2][:],
        1: edge_pref(category[1], False),
        0: edge_pref(category[0], True),
    }
    category_order = [1, 0, 2, 1, 0, 2]
    while len(chosen) < target:
        added = False
        for key in category_order:
            cell = _pick_from_pool(pools[key], chosen, forbidden, rng)
            if cell is not None:
                chosen.append(cell)
                added = True
                break
        if not added:
            break
    return chosen if len(chosen) >= 4 else None


def _render_input(shape: tuple[int, int], specs: list[dict], yellows: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        full, color_cells = _triangle_cells(spec)
        for cell in color_cells:
            grid[cell] = spec["color"]
        grid[spec["gray"]] = GREY
        assert full == color_cells | {spec["gray"]}
    for cell in yellows:
        assert int(grid[cell]) == BG
        grid[cell] = YELLOW
    return grid


def _validate_instance(grid: np.ndarray, specs: list[dict], yellows: list[tuple[int, int]]) -> bool:
    try:
        parsed = _extract_specs(grid)
    except AssertionError:
        return False
    if { _spec_signature(spec) for spec in parsed } != { _spec_signature(spec) for spec in specs }:
        return False

    if not (4 <= len(yellows) <= 8):
        return False
    if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1 for i, a in enumerate(yellows) for b in yellows[i + 1 :]):
        return False

    cov = _coverage_map(specs, grid.shape)
    yellow_counts = [int(cov[cell]) for cell in yellows]
    if not any(count == 0 for count in yellow_counts):
        return False
    if not any(count == 1 for count in yellow_counts):
        return False
    if not any(count >= 2 for count in yellow_counts):
        return False

    expected = _render_output_from_specs(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    outputs = {_yellow_output_color(count) for count in yellow_counts}
    if outputs != {YELLOW, ORANGE, MAROON}:
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))
    specs = _build_cross_specs(shape, rng)
    if specs is None or not _valid_specs(specs, shape):
        return None

    yellows = _choose_yellows(specs, shape, rng)
    if yellows is None:
        return None

    grid = _render_input(shape, specs, yellows)
    return grid if _validate_instance(grid, specs, yellows) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 4")
