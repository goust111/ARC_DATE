from __future__ import annotations

import random

import numpy as np


BLACK = 0
TEAL = 8
PALETTE = tuple(color for color in range(1, 10) if color != TEAL)

GEN_ATTEMPTS = 400
GRID_H_MIN = 6
GRID_H_MAX = 12
GRID_W_MAX = 30
PATTERN_W_RANGE = (2, 6)
EXACT_TOUCH_HEIGHTS = (3, 5)
OVERLAP_HEAVY_HEIGHTS = (4, 6)
TOP_MARGIN_MAX = 4
LEFT_MARGIN_RANGE = (1, 5)
RIGHT_MARGIN_EXTRA_MAX = 5
COLOR_COUNT_CHOICES = (2, 3)
FILL_MODES = ("noise", "row_runs", "column_bands", "diagonal_bands")


def _foreground_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected one foreground pattern"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != BLACK), "foreground must fill its bbox as a rectangle"
    assert int(coords.shape[0]) == patch.size, "expected exactly one rectangular foreground object"
    return r0, c0, r1, c1


def _extract_pattern(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    bbox = _foreground_bbox(grid)
    r0, c0, r1, c1 = bbox
    pattern = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert pattern.shape[0] >= 3 and pattern.shape[1] >= 2
    assert not np.array_equal(pattern, np.fliplr(pattern)), "pattern must be horizontally asymmetric"
    return bbox, pattern


def _render_transformation(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    (r0, c0, _r1, _c1), pattern = _extract_pattern(g)
    ph, pw = pattern.shape
    H, W = g.shape

    right_top = c0 + pw + ph - 1
    assert right_top + pw <= W, "canvas too narrow for mirrored shear"

    out = np.zeros_like(g)
    left_mask = np.zeros_like(g, dtype=bool)
    right_mask = np.zeros_like(g, dtype=bool)
    mirror = np.fliplr(pattern)

    for r in range(ph):
        row = r0 + r
        left_start = c0 + r
        right_start = right_top - r
        assert 0 <= row < H
        assert 0 <= left_start and left_start + pw <= W
        assert 0 <= right_start and right_start + pw <= W

        out[row, left_start : left_start + pw] = pattern[r]
        left_mask[row, left_start : left_start + pw] = True

        # The mirrored copy is drawn after the left copy, which resolves overlap rows deterministically.
        out[row, right_start : right_start + pw] = mirror[r]
        right_mask[row, right_start : right_start + pw] = True

    for r in range(ph):
        row = r0 + r
        left_cols = np.flatnonzero(left_mask[row])
        right_cols = np.flatnonzero(right_mask[row])
        left_edge = int(left_cols[-1])
        right_edge = int(right_cols[0])
        if right_edge - left_edge <= 1:
            continue
        gap = out[row, left_edge + 1 : right_edge]
        gap[gap == BLACK] = TEAL

    return out, left_mask, right_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the lone rectangular motif, shear the two copies inward, and fill their row-wise wedge gap with teal.

    Input structure:
    - A black background contains exactly one non-black connected rectangle whose tight bbox is fully filled.
    - The rectangle uses 2-3 non-black colors and is horizontally asymmetric so its mirror has a distinct orientation.
    - The canvas has enough empty space to the right for two opposing sheared copies and their gap.

    Rule:
    - Crop the rectangular motif P and create its left-right mirror P'.
    - On a same-size black output canvas, place row r of P at the input bbox top row and column c_left + r, where c_left is the input bbox left edge.
    - Place row r of P' at the same output row and column c_left + W + H - 1 - r, so the mirrored copy shears inward from the right.
    - For each row, find the rightmost pixel of the left copy and the leftmost pixel of the mirrored copy; recolor every black cell strictly between them teal.
    - If a row has no black cells between the two copies because they touch or overlap, leave that row unfilled.

    Key insight:
    - The output is determined by combining two indispensable steps: mirror to reverse row orientation, then opposite shears to make the inter-copy distance shrink by row, producing a wedge whose filled width varies from row to row.
    """
    out, _left_mask, _right_mask = _render_transformation(grid)
    return out


def _ensure_palette_coverage(pattern: np.ndarray, palette: list[int]) -> None:
    positions = random.sample(range(pattern.size), len(palette))
    rows, cols = np.divmod(positions, pattern.shape[1])
    pattern[rows, cols] = np.asarray(palette, dtype=int)


def _fill_pattern(ph: int, pw: int, palette: list[int]) -> np.ndarray:
    pattern = np.zeros((ph, pw), dtype=int)
    mode = random.choice(FILL_MODES)

    if mode == "noise":
        pattern[:] = np.random.choice(palette, size=(ph, pw))
    elif mode == "row_runs":
        for r in range(ph):
            if pw == 2:
                cuts = [1]
            else:
                cut_count = random.randint(1, min(3, pw - 1))
                cuts = sorted(random.sample(range(1, pw), cut_count))
            last = None
            starts = [0] + cuts
            ends = cuts + [pw]
            for c0, c1 in zip(starts, ends):
                choices = [color for color in palette if color != last] or palette
                last = random.choice(choices)
                pattern[r, c0:c1] = last
    elif mode == "column_bands":
        for c in range(pw):
            pattern[:, c] = random.choice(palette)
        for _ in range(random.randint(1, ph)):
            rr = random.randrange(ph)
            cc = random.randrange(pw)
            pattern[rr, cc] = random.choice(palette)
    else:
        rows, cols = np.indices((ph, pw))
        band = random.randint(1, max(1, min(3, pw)))
        phase = random.randrange(len(palette))
        pattern[:] = np.asarray(palette)[((rows + cols) // band + phase) % len(palette)]
        if random.random() < 0.5:
            pattern = np.fliplr(pattern)

    _ensure_palette_coverage(pattern, palette)
    return pattern


def _pick_pattern_height() -> int:
    if random.random() < 0.5:
        return random.choice(EXACT_TOUCH_HEIGHTS)
    return random.choice(OVERLAP_HEAVY_HEIGHTS)


def _valid_pattern(pattern: np.ndarray, palette: list[int]) -> bool:
    if BLACK in pattern or TEAL in pattern:
        return False
    if not np.array_equal(np.unique(pattern), np.array(sorted(palette), dtype=int)):
        return False
    if np.array_equal(pattern, np.fliplr(pattern)):
        return False
    if all(np.array_equal(row, row[::-1]) for row in pattern):
        return False
    if len({tuple(map(int, row)) for row in pattern}) == 1:
        return False
    if not any(len(set(map(int, row))) >= 2 for row in pattern):
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    ph = _pick_pattern_height()
    pw = random.randint(*PATTERN_W_RANGE)
    color_count = random.choice(COLOR_COUNT_CHOICES)
    palette = random.sample(list(PALETTE), color_count)
    pattern = _fill_pattern(ph, pw, palette)
    if not _valid_pattern(pattern, palette):
        return None

    min_h = max(GRID_H_MIN, ph)
    max_h = min(GRID_H_MAX, ph + TOP_MARGIN_MAX)
    if min_h > max_h:
        return None
    H = random.randint(min_h, max_h)
    top = random.randint(0, H - ph)

    left_min, left_max = LEFT_MARGIN_RANGE
    c0 = random.randint(left_min, left_max)
    required_w = max(2 * pw + 2 * ph - 1, c0 + 2 * pw + ph - 1)
    max_w = min(GRID_W_MAX, required_w + RIGHT_MARGIN_EXTRA_MAX)
    if required_w > max_w:
        return None
    W = random.randint(required_w, max_w)

    grid = np.zeros((H, W), dtype=int)
    grid[top : top + ph, c0 : c0 + pw] = pattern
    return grid


def _raw_gap_widths(ph: int) -> list[int]:
    return [ph - 1 - 2 * r for r in range(ph)]


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    bbox, pattern = _extract_pattern(grid)
    r0, c0, _r1, _c1 = bbox
    ph, pw = pattern.shape
    raw_gaps = _raw_gap_widths(ph)
    expected = [max(0, width) for width in raw_gaps]
    assert any(width > 0 for width in raw_gaps), "must have at least one filled row"
    assert any(width == 0 for width in raw_gaps), "must cover the touching no-gap branch"
    assert any(width < 0 for width in raw_gaps), "must cover the overlap no-gap branch"

    out, left_mask, right_mask = _render_transformation(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)

    support = left_mask | right_mask | (out == TEAL)
    assert np.all(out[~support] == BLACK), "only the sheared copies and teal gap may remain"

    mirror = np.fliplr(pattern)
    right_top = c0 + pw + ph - 1
    for r in range(ph):
        row = r0 + r
        left_start = c0 + r
        right_start = right_top - r
        left_slice = slice(left_start, left_start + pw)
        right_slice = slice(right_start, right_start + pw)

        left_only = left_mask[row, left_slice] & ~right_mask[row, left_slice]
        right_only = right_mask[row, right_slice] & ~left_mask[row, right_slice]

        actual_left = out[row, left_slice]
        actual_right = out[row, right_slice]
        assert np.array_equal(actual_left[left_only], pattern[r][left_only]), "left shear must preserve row content"
        assert np.array_equal(actual_right[right_only], mirror[r][right_only]), "right shear must be mirrored"

        fill_cols = np.flatnonzero(out[row] == TEAL)
        if expected[r] == 0:
            assert fill_cols.size == 0 or fill_cols[0] > fill_cols[-1]
        else:
            assert fill_cols.size == expected[r], "teal width must match the row gap"
            left_edge = int(np.flatnonzero(left_mask[row])[-1])
            right_edge = int(np.flatnonzero(right_mask[row])[0])
            assert fill_cols[0] == left_edge + 1
            assert fill_cols[-1] == right_edge - 1


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirror-shear wedge instance")
