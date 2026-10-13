from __future__ import annotations

import itertools

import numpy as np


BG = 0
GLYPH_COLORS = (1, 2, 3, 4, 6)
DIVIDER = 5
START = 7

MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
DELTAS = ((2, 2), (2, -2), (-2, 2), (-2, -2))
IDENTITY_ORDER = tuple(range(4))

GEN_HEIGHT_RANGE = (13, 19)
GEN_WIDTH_RANGE = (11, 19)
GEN_GLYPH_COUNT_CHOICES = (3, 4, 5)
GEN_RETRY_LIMIT = 400

_SEQUENCE_CACHE: dict[int, list[tuple[int, ...]]] = {}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_glyphs(grid: np.ndarray) -> list[tuple[int, int]]:
    top = grid[:2, 1:]
    used_cols = np.where(np.any(top != BG, axis=0))[0]
    assert used_cols.size > 0, "expected a glyph strip in the top two rows"
    strip_w = int(used_cols[-1]) + 1
    assert np.array_equal(used_cols, np.arange(strip_w)), "glyph strip must be contiguous from column 1"
    assert strip_w % 2 == 0, "glyph strip width must be even"

    glyphs: list[tuple[int, int]] = []
    for offset in range(0, strip_w, 2):
        block = grid[:2, 1 + offset : 1 + offset + 2]
        nonzero = block[block != BG]
        assert nonzero.size == 3, "each glyph must have exactly three colored cells"
        colors = set(map(int, nonzero))
        assert len(colors) == 1, "each glyph must be monochrome"
        color = next(iter(colors))
        assert color in GLYPH_COLORS, "glyph colors must come from the visible palette"
        missing = np.argwhere(block == BG)
        assert missing.shape == (1, 2), "each glyph must miss exactly one corner"
        missing_corner = tuple(map(int, missing[0]))
        glyphs.append((color, MISSING_CORNERS.index(missing_corner)))

    colors = [color for color, _missing_idx in glyphs]
    assert 3 <= len(glyphs) <= 5, "expected 3 to 5 glyphs"
    assert len(set(colors)) == len(colors), "glyph colors must be distinct"
    assert np.all(grid[:2, 1 + strip_w :] == BG), "top rows beyond the strip must be black"
    return glyphs


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == START)
    assert coords.shape == (1, 2), "expected exactly one orange start pixel"
    return tuple(map(int, coords[0]))


def _simulate_output(
    shape: tuple[int, int],
    start: tuple[int, int],
    glyphs: list[tuple[int, int]],
    *,
    mapping: tuple[tuple[int, int], ...] = DELTAS,
    order: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    out[:, 0] = DIVIDER
    out[start] = START

    r, c = start
    order = tuple(range(len(glyphs))) if order is None else order
    for idx in order:
        color, missing_idx = glyphs[idx]
        dr, dc = mapping[missing_idx]
        mid = (r + dr // 2, c + dc // 2)
        end = (r + dr, c + dc)
        if not (_in_bounds(shape, *mid) and _in_bounds(shape, *end)):
            return None
        if mid[0] < 2 or end[0] < 2 or mid[1] < 1 or end[1] < 1:
            return None
        out[mid] = color
        out[end] = color
        r, c = end

    out[start] = START
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top glyph strip into cumulative diagonal moves that paint colored path segments from the orange start.

    Input structure:
    - The grid has a black background with a full-height gray divider in the leftmost column.
    - The top two rows contain 3 to 5 adjacent 2x2 monochrome L-glyphs, each with one missing corner and a distinct color.
    - A single orange pixel below the strip marks the sprite's starting position.

    Rule:
    - Read the glyphs from left to right and decode each missing corner into the opposite diagonal move: missing upper-left means move down-right, upper-right means down-left, lower-left means up-right, and lower-right means up-left.
    - Starting from the orange pixel, execute the moves cumulatively with step size 2 in both row and column.
    - For each step, color the diagonal midpoint and endpoint with that glyph's color, overwriting any earlier path color, then erase the glyph strip back to black while preserving the gray divider and the orange start marker.

    Key insight:
    - The L-glyphs are movement instructions rather than pictures to copy: the absent corner encodes the opposite diagonal direction, and the left-to-right order controls a cumulative path whose later steps depend on every earlier endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 7, "grid is too small for the described puzzle"
    assert np.all(g[:, 0] == DIVIDER), "leftmost column must be the gray divider"

    glyphs = _parse_glyphs(g)
    start = _find_start(g)

    out = g.copy()
    out[:2, 1:] = BG

    r, c = start
    for color, missing_idx in glyphs:
        dr, dc = DELTAS[missing_idx]
        mid = (r + dr // 2, c + dc // 2)
        end = (r + dr, c + dc)
        assert _in_bounds(g.shape, *mid) and _in_bounds(g.shape, *end), "decoded path must stay in bounds"
        assert mid[0] >= 2 and end[0] >= 2, "decoded path must stay below the glyph strip"
        assert mid[1] >= 1 and end[1] >= 1, "decoded path must stay to the right of the divider"
        out[mid] = color
        out[end] = color
        r, c = end

    out[start] = START
    return out


def _sequence_geometry(seq: tuple[int, ...]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    pos = (0, 0)
    painted: list[tuple[int, int]] = []
    used = set()
    endpoints = [(0, 0)]
    endpoint_set = {(0, 0)}

    for idx, step in enumerate(seq):
        dr, dc = DELTAS[step]
        if idx and (dr, dc) == (-DELTAS[seq[idx - 1]][0], -DELTAS[seq[idx - 1]][1]):
            return None
        mid = (pos[0] + dr // 2, pos[1] + dc // 2)
        end = (pos[0] + dr, pos[1] + dc)
        if mid in used or end in used or end in endpoint_set:
            return None
        used.add(mid)
        used.add(end)
        painted.extend([mid, end])
        endpoints.append(end)
        endpoint_set.add(end)
        pos = end
    return painted, endpoints


def _sequence_ok(seq: tuple[int, ...], target_distinct: int) -> bool:
    geom = _sequence_geometry(seq)
    if geom is None:
        return False
    dr_signs = {1 if DELTAS[step][0] > 0 else -1 for step in seq}
    dc_signs = {1 if DELTAS[step][1] > 0 else -1 for step in seq}
    turns = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
    return (
        len(set(seq)) >= target_distinct
        and dr_signs == {-1, 1}
        and dc_signs == {-1, 1}
        and turns >= 2
    )


def _all_valid_sequences(n: int) -> list[tuple[int, ...]]:
    if n not in _SEQUENCE_CACHE:
        valid = [seq for seq in itertools.product(range(4), repeat=n) if _sequence_geometry(seq) is not None]
        _SEQUENCE_CACHE[n] = valid
    return _SEQUENCE_CACHE[n]


def _sample_sequence(n: int, rng: np.random.Generator) -> tuple[int, ...] | None:
    target_distinct = 4 if n >= 4 and float(rng.random()) < 0.65 else min(3, n)
    require_repeat = n == 5 and float(rng.random()) < 0.55
    candidates = []
    for seq in _all_valid_sequences(n):
        if not _sequence_ok(seq, target_distinct):
            continue
        if require_repeat and max(seq.count(step) for step in range(4)) < 2:
            continue
        candidates.append(seq)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _choose_canvas_and_start(
    seq: tuple[int, ...], n_glyphs: int, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    geom = _sequence_geometry(seq)
    assert geom is not None
    painted, endpoints = geom
    points = [(0, 0), *painted, *endpoints]
    rel_rows = [r for r, _c in points]
    rel_cols = [c for _r, c in points]
    min_r, max_r = min(rel_rows), max(rel_rows)
    min_c, max_c = min(rel_cols), max(rel_cols)

    min_h = max(GEN_HEIGHT_RANGE[0], max_r - min_r + 3)
    min_w = max(GEN_WIDTH_RANGE[0], max_c - min_c + 2, 2 * n_glyphs + 2)
    if min_h >= GEN_HEIGHT_RANGE[1] or min_w >= GEN_WIDTH_RANGE[1]:
        return None

    height_choices = list(range(min_h, GEN_HEIGHT_RANGE[1]))
    width_choices = list(range(min_w, GEN_WIDTH_RANGE[1]))
    rng.shuffle(height_choices)
    rng.shuffle(width_choices)

    for h in height_choices[:5]:
        for w in width_choices[:6]:
            row_low = 2 - min_r
            row_high = h - 1 - max_r
            col_low = 1 - min_c
            col_high = w - 1 - max_c
            pref_row_low = max(row_low, h // 2)
            pref_row_high = min(row_high, h - 3)
            pref_col_low = max(col_low, max(2, w // 3))
            pref_col_high = min(col_high, w - 2)
            if pref_row_low > pref_row_high or pref_col_low > pref_col_high:
                continue
            start_r = int(rng.integers(pref_row_low, pref_row_high + 1))
            start_c = int(rng.integers(pref_col_low, pref_col_high + 1))
            return (h, w), (start_r, start_c)
    return None


def _render_input(shape: tuple[int, int], start: tuple[int, int], glyphs: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[:, 0] = DIVIDER
    for idx, (color, missing_idx) in enumerate(glyphs):
        c0 = 1 + 2 * idx
        block = np.full((2, 2), color, dtype=int)
        mr, mc = MISSING_CORNERS[missing_idx]
        block[mr, mc] = BG
        grid[:2, c0 : c0 + 2] = block
    grid[start] = START
    return grid


def _visible_color_check(out: np.ndarray, glyphs: list[tuple[int, int]]) -> bool:
    for color, _missing_idx in glyphs:
        if int(np.sum(out == color)) == 0:
            return False
    return True


def _validate_case(grid: np.ndarray) -> bool:
    try:
        glyphs = _parse_glyphs(grid)
        start = _find_start(grid)
        solved = main(grid)
    except AssertionError:
        return False

    expected = _simulate_output(grid.shape, start, glyphs)
    if expected is None or not np.array_equal(solved, expected):
        return False
    if not np.all(solved[:, 0] == DIVIDER):
        return False
    if not np.all(solved[:2, 1:] == BG):
        return False
    if solved[start] != START:
        return False
    if not _visible_color_check(solved, glyphs):
        return False

    for idx in range(len(glyphs)):
        shorter = glyphs[:idx] + glyphs[idx + 1 :]
        alt = _simulate_output(grid.shape, start, shorter)
        if alt is not None and np.array_equal(alt, solved):
            return False

    intended_order = tuple(range(len(glyphs)))
    for perm in itertools.permutations(range(len(glyphs))):
        if perm == intended_order:
            continue
        alt = _simulate_output(grid.shape, start, glyphs, order=perm)
        if alt is not None and np.array_equal(alt, solved):
            return False

    for perm in itertools.permutations(range(4)):
        if perm == IDENTITY_ORDER:
            continue
        mapping = tuple(DELTAS[idx] for idx in perm)
        alt = _simulate_output(grid.shape, start, glyphs, mapping=mapping)
        if alt is not None and np.array_equal(alt, solved):
            return False

    half_step_mapping = tuple((dr // 2, dc // 2) for dr, dc in DELTAS)
    alt = _simulate_output(grid.shape, start, glyphs, mapping=half_step_mapping)
    if alt is not None and np.array_equal(alt, solved):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_glyphs = int(rng.choice(GEN_GLYPH_COUNT_CHOICES))
    seq = _sample_sequence(n_glyphs, rng)
    if seq is None:
        return None

    canvas = _choose_canvas_and_start(seq, n_glyphs, rng)
    if canvas is None:
        return None
    shape, start = canvas

    color_pool = list(GLYPH_COLORS)
    rng.shuffle(color_pool)
    glyphs = [(int(color_pool[idx]), int(seq[idx])) for idx in range(n_glyphs)]
    grid = _render_input(shape, start, glyphs)
    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-glyph path input")
