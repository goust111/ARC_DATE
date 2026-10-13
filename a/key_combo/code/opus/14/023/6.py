from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

TILE_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
MASK_SIZE = 2
TILE_SIZE = 4


def _trim_zero_border(grid: np.ndarray) -> np.ndarray:
    rows = np.any(grid != BLACK, axis=1)
    cols = np.any(grid != BLACK, axis=0)
    assert rows.any() and cols.any(), "region cannot be empty"
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _reflection_orbit(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
            }
        )
    )


def _all_reflection_orbits(shape: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    h, w = map(int, shape)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = _reflection_orbit(r, c, h, w)
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert not (full_rows and full_cols), "expected one separator orientation"

    if full_rows:
        assert len(full_rows) == 1, "expected one full separator row"
        sep = full_rows[0]
        top = _trim_zero_border(g[:sep, :])
        bottom = _trim_zero_border(g[sep + 1 :, :])
        return top, bottom

    assert len(full_cols) == 1, "expected one full separator column"
    sep = full_cols[0]
    left = _trim_zero_border(g[:, :sep])
    right = _trim_zero_border(g[:, sep + 1 :])
    return left, right


def _extract_tile(region: np.ndarray) -> np.ndarray:
    h, w = map(int, region.shape)
    grey_rows = [r for r in range(h) if np.all(region[r, :] == GREY)]
    grey_cols = [c for c in range(w) if np.all(region[:, c] == GREY)]
    assert len(grey_rows) == 5 and len(grey_cols) == 5, "expected a 4x4 partitioned grid"

    row_spans = [grey_rows[i + 1] - grey_rows[i] - 1 for i in range(4)]
    col_spans = [grey_cols[i + 1] - grey_cols[i] - 1 for i in range(4)]
    assert len(set(row_spans)) == 1 and len(set(col_spans)) == 1, "all partition cells must be equal"
    cell_h = row_spans[0]
    cell_w = col_spans[0]
    assert cell_h >= 2 and cell_w >= 2, "cells must have room for a unique payload pixel"

    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for br in range(TILE_SIZE):
        for bc in range(TILE_SIZE):
            block = region[
                grey_rows[br] + 1 : grey_rows[br + 1],
                grey_cols[bc] + 1 : grey_cols[bc + 1],
            ]
            payload = block[(block != BLACK) & (block != GREY)]
            assert payload.size == 1, "each partition cell must contain exactly one payload pixel"
            tile[br, bc] = int(payload[0])

    assert np.all(np.isin(tile, TILE_COLORS)), "tile payload colors must be visible non-grey colors"
    return tile


def _corner_windows(shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = map(int, shape)
    return [
        np.array([[0, 0], [1, 1]], dtype=int),
        np.array([[0, w - 2], [1, w - 1]], dtype=int),
        np.array([[h - 2, 0], [h - 1, 1]], dtype=int),
        np.array([[h - 2, w - 2], [h - 1, w - 1]], dtype=int),
    ]


def _window_bounds(shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    return [
        (0, 0, 2, 2),
        (0, w - 2, 2, w),
        (h - 2, 0, h, 2),
        (h - 2, w - 2, h, w),
    ]


def _repair_small_region(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, region.shape)
    assert (h, w) in {(3, 3), (4, 4)}, "expected a 3x3 or 4x4 symmetry panel"
    placeholder_coords = np.argwhere(region == MAROON)
    assert len(placeholder_coords) == 1, "expected exactly one placeholder"
    ph_r, ph_c = map(int, placeholder_coords[0])

    repaired = region.copy()
    for orbit in _all_reflection_orbits(region.shape):
        values = [int(region[r, c]) for r, c in orbit]
        visible = {v for v in values if v != MAROON}
        assert visible, "placeholder orbit must expose a visible color"
        assert visible.issubset({BLACK, MAGENTA}), "mask clues must be black or magenta"
        assert len(visible) == 1, "each damaged orbit must have one recoverable color"
        fill = visible.pop()
        for r, c in orbit:
            if repaired[r, c] == MAROON:
                repaired[r, c] = fill

    candidate_windows = []
    for bounds in _window_bounds(region.shape):
        r0, c0, r1, c1 = bounds
        if r0 <= ph_r < r1 and c0 <= ph_c < c1:
            candidate_windows.append(bounds)
    assert len(candidate_windows) == 1, "placeholder must identify a unique 2x2 corner mask"

    r0, c0, r1, c1 = candidate_windows[0]
    mask_window = repaired[r0:r1, c0:c1]
    vals, counts = np.unique(mask_window, return_counts=True)
    counts_map = {int(v): int(c) for v, c in zip(vals, counts)}
    assert counts_map.get(MAGENTA, 0) == 2 and counts_map.get(BLACK, 0) == 2, "mask must contain two foreground and two background cells"

    mask = (mask_window == MAGENTA).astype(int)
    return repaired, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the large partitioned panel into a 4x4 tile, repair the small reflection-damaged mask, and Kronecker-expand the tile through that mask.

    Input structure:
    - A black background grid is split by one full grey row or one full grey column into two regions.
    - The larger region is a grey-lined 4x4 partition whose 16 equal cells each contain exactly one non-black, non-grey payload pixel.
    - The smaller region is a 3x3 or 4x4 panel made only of black, magenta, and one maroon placeholder; it is left-right and top-bottom symmetric once the placeholder is repaired.
    - The maroon placeholder lies inside one unique 2x2 corner subgrid of the small panel, and that repaired corner contains exactly two magenta and two black cells.

    Rule:
    - Compress the large region by reading the unique payload color from each partition cell to form a 4x4 tile matrix.
    - Repair the maroon placeholder by taking the unique visible color from its horizontal/vertical reflection orbit in the small panel.
    - Extract the repaired 2x2 corner subgrid that contains the placeholder, treating magenta as 1 and black as 0.
    - Output the Kronecker product of that 2x2 binary mask with the 4x4 tile, so foreground mask cells stamp a full tile copy and background mask cells become all black.

    Key insight:
    - The big panel is not the final picture but the tile content, while the small repaired symmetric panel is not the final picture either but the binary layout that decides where tile copies appear.
    """
    first, second = _split_regions(np.asarray(grid, dtype=int))

    errors = []
    for large, small in ((first, second), (second, first)):
        try:
            tile = _extract_tile(large)
            _repaired, mask = _repair_small_region(small)
            out = np.kron(mask, tile)
            assert out.shape == (MASK_SIZE * TILE_SIZE, MASK_SIZE * TILE_SIZE)
            return out.astype(int)
        except AssertionError as exc:
            errors.append(str(exc))

    raise AssertionError(f"failed to identify the two semantic regions: {errors}")


def _sample_tile(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        tile = rng.choice(TILE_COLORS, size=(TILE_SIZE, TILE_SIZE), replace=True)
        if len(set(map(int, tile.flat))) < 3:
            continue
        if np.all(tile == tile[0, 0]):
            continue
        if np.array_equal(tile, np.flipud(tile)) and np.array_equal(tile, np.fliplr(tile)):
            continue
        if len({tuple(map(int, row)) for row in tile}) == 1:
            continue
        if len({tuple(map(int, col)) for col in tile.T}) == 1:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a non-degenerate 4x4 tile")


def _render_large_region(tile: np.ndarray, cell_size: int, rng: np.random.Generator) -> np.ndarray:
    side = TILE_SIZE * cell_size + (TILE_SIZE + 1)
    region = np.full((side, side), GREY, dtype=int)
    for br in range(TILE_SIZE):
        for bc in range(TILE_SIZE):
            r0 = 1 + br * (cell_size + 1)
            c0 = 1 + bc * (cell_size + 1)
            region[r0 : r0 + cell_size, c0 : c0 + cell_size] = BLACK
            pr = r0 + int(rng.integers(cell_size))
            pc = c0 + int(rng.integers(cell_size))
            region[pr, pc] = int(tile[br, bc])
    return region


def _reflect_full_from_corner(mask2x2: np.ndarray, size: int) -> np.ndarray:
    region = np.zeros((size, size), dtype=int)
    for r in range(2):
        for c in range(2):
            value = int(mask2x2[r, c])
            color = MAGENTA if value else BLACK
            for rr, cc in _reflection_orbit(r, c, size, size):
                region[rr, cc] = color
    return region


def _window_reflection(region: np.ndarray, corner_idx: int) -> np.ndarray:
    bounds = _window_bounds(region.shape)[corner_idx]
    r0, c0, r1, c1 = bounds
    return region[r0:r1, c0:c1]


def _eligible_placeholder_cells(region: np.ndarray, corner_idx: int) -> list[tuple[int, int]]:
    bounds = _window_bounds(region.shape)[corner_idx]
    r0, c0, r1, c1 = bounds
    out = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if int(region[r, c]) != BLACK:
                continue
            if len(_reflection_orbit(r, c, *region.shape)) == 1:
                continue
            out.append((r, c))
    return out


def _sample_small_region(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mask_patterns = [
        np.array([[1, 1], [0, 0]], dtype=int),
        np.array([[1, 0], [1, 0]], dtype=int),
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [0, 1]], dtype=int),
        np.array([[0, 0], [1, 1]], dtype=int),
    ]

    for _ in range(300):
        size = int(rng.choice([3, 4]))
        prototype = mask_patterns[int(rng.integers(len(mask_patterns)))]
        full = _reflect_full_from_corner(prototype, size)

        if np.all(full[0, :] == BLACK) or np.all(full[-1, :] == BLACK):
            continue
        if np.all(full[:, 0] == BLACK) or np.all(full[:, -1] == BLACK):
            continue

        corner_idx = int(rng.integers(4))
        window = _window_reflection(full, corner_idx)
        if int(np.count_nonzero(window == MAGENTA)) != 2 or int(np.count_nonzero(window == BLACK)) != 2:
            continue

        candidates = _eligible_placeholder_cells(full, corner_idx)
        if not candidates:
            continue

        ph_r, ph_c = candidates[int(rng.integers(len(candidates)))]
        inp = full.copy()
        inp[ph_r, ph_c] = MAROON
        expected_mask = (window == MAGENTA).astype(int)
        return inp, expected_mask

    raise RuntimeError("failed to sample a non-degenerate small region")


def _place_vertical(
    left: np.ndarray,
    right: np.ndarray,
    left_on_big_side: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    a = left if left_on_big_side else right
    b = right if left_on_big_side else left
    pad_top_a = int(rng.integers(0, 3))
    pad_bot_a = int(rng.integers(0, 3))
    pad_top_b = int(rng.integers(0, 3))
    pad_bot_b = int(rng.integers(0, 3))
    pad_left_a = int(rng.integers(0, 3))
    pad_right_a = int(rng.integers(0, 3))
    pad_left_b = int(rng.integers(0, 3))
    pad_right_b = int(rng.integers(0, 3))

    h = max(pad_top_a + a.shape[0] + pad_bot_a, pad_top_b + b.shape[0] + pad_bot_b)
    w_left = pad_left_a + a.shape[1] + pad_right_a
    w_right = pad_left_b + b.shape[1] + pad_right_b
    grid = np.zeros((h, w_left + 1 + w_right), dtype=int)
    grid[:, w_left] = GREY

    ra = pad_top_a
    ca = pad_left_a
    rb = pad_top_b
    cb = w_left + 1 + pad_left_b
    grid[ra : ra + a.shape[0], ca : ca + a.shape[1]] = a
    grid[rb : rb + b.shape[0], cb : cb + b.shape[1]] = b
    return grid


def _place_horizontal(
    top: np.ndarray,
    bottom: np.ndarray,
    top_is_big: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    a = top if top_is_big else bottom
    b = bottom if top_is_big else top
    pad_top_a = int(rng.integers(0, 3))
    pad_bot_a = int(rng.integers(0, 3))
    pad_top_b = int(rng.integers(0, 3))
    pad_bot_b = int(rng.integers(0, 3))
    pad_left_a = int(rng.integers(0, 3))
    pad_right_a = int(rng.integers(0, 3))
    pad_left_b = int(rng.integers(0, 3))
    pad_right_b = int(rng.integers(0, 3))

    w = max(pad_left_a + a.shape[1] + pad_right_a, pad_left_b + b.shape[1] + pad_right_b)
    h_top = pad_top_a + a.shape[0] + pad_bot_a
    h_bottom = pad_top_b + b.shape[0] + pad_bot_b
    grid = np.zeros((h_top + 1 + h_bottom, w), dtype=int)
    grid[h_top, :] = GREY

    ra = pad_top_a
    ca = pad_left_a
    rb = h_top + 1 + pad_top_b
    cb = pad_left_b
    grid[ra : ra + a.shape[0], ca : ca + a.shape[1]] = a
    grid[rb : rb + b.shape[0], cb : cb + b.shape[1]] = b
    return grid


def _semantic_checks(inp: np.ndarray, expected_out: np.ndarray) -> None:
    assert inp.ndim == 2 and max(inp.shape) <= 30, f"input exceeds ARC size limit: {inp.shape}"
    assert expected_out.shape == (8, 8), "output must be 8x8"
    assert np.array_equal(main(inp), expected_out), "solver must reproduce the generated target"

    first, second = _split_regions(inp)
    parsed = []
    for large, small in ((first, second), (second, first)):
        try:
            tile = _extract_tile(large)
            _repaired, mask = _repair_small_region(small)
            parsed.append((tile, mask))
        except AssertionError:
            continue
    assert len(parsed) == 1, "input must expose exactly one valid large/small role assignment"

    tile, mask = parsed[0]
    assert len(set(map(int, tile.flat))) >= 3, "tile must not collapse to a trivial palette"
    assert mask.sum() == 2, "mask must use both foreground and background"

    nonzero_blocks = 0
    for br, bc in itertools.product(range(2), range(2)):
        block = expected_out[br * 4 : (br + 1) * 4, bc * 4 : (bc + 1) * 4]
        if np.any(block != BLACK):
            nonzero_blocks += 1
            assert np.array_equal(block, tile), "every foreground block must be the extracted tile"
        else:
            assert np.all(block == BLACK), "background blocks must stay black"
    assert nonzero_blocks == 2, "the repaired mask must activate exactly two tile copies"


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    tile = _sample_tile(rng)
    cell_size = int(rng.integers(2, 5))
    large = _render_large_region(tile, cell_size, rng)
    small, mask = _sample_small_region(rng)
    expected_out = np.kron(mask, tile)

    orientation = int(rng.integers(2))
    big_first = bool(rng.integers(2))
    if orientation == 0:
        inp = _place_vertical(large, small, big_first, rng)
    else:
        inp = _place_horizontal(large, small, big_first, rng)

    if max(inp.shape) > 30:
        return None

    try:
        _semantic_checks(inp, expected_out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
