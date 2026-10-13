from __future__ import annotations

import random

import numpy as np


BG = 0
PATTERN_COLORS = (1, 2)
BLOCKER_COLOR = 3
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_H_RANGE = (12, 16)
GRID_W_RANGE = (15, 20)
PERIOD_CHOICES = (2, 3)
VISIBLE_WIDTH_CHOICES = (3, 4)
BLOCKER_COUNT_RANGE = (3, 5)
GENERATION_RETRIES = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _visible_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == PATTERN_COLORS[0]) | (grid == PATTERN_COLORS[1])


def _visible_width(grid: np.ndarray) -> int:
    cols = np.flatnonzero(_visible_mask(grid).any(axis=0))
    assert cols.size > 0, "expected a left-side visible motif"
    assert int(cols[0]) == 0, "visible motif must start at the left edge"
    return int(cols[-1]) + 1


def _minimal_period(prefix: np.ndarray) -> int:
    h, w = prefix.shape
    assert h > 0 and w > 0
    for period in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(prefix[:, c], prefix[:, c % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("failed to infer a column period")


def _tile_full_width(tile: np.ndarray, width: int) -> np.ndarray:
    h, period = tile.shape
    out = np.empty((h, width), dtype=int)
    for c in range(width):
        out[:, c] = tile[:, c % period]
    return out


def _blockers(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(grid == BLOCKER_COLOR)]


def _erase_mask(shape: tuple[int, int], blockers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    blocker_set = set(blockers)
    for r0, c0 in blockers:
        for dr, dc in DIAGONALS:
            r, c = r0 + dr, c0 + dc
            while _in_bounds(shape, r, c):
                if (r, c) not in blocker_set:
                    mask[r, c] = True
                r += dr
                c += dc
    return mask


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    visible_w = _visible_width(g)
    assert 3 <= visible_w <= 4, "expected a 3-4 column visible motif"
    prefix = g[:, :visible_w].copy()
    assert not np.any(prefix == BLOCKER_COLOR), "blockers must lie to the right of the visible motif"

    period = _minimal_period(prefix)
    assert period in PERIOD_CHOICES, "expected a minimal period of 2 or 3"

    tile = prefix[:, :period]
    out = _tile_full_width(tile, w)

    blockers = _blockers(g)
    for r, c in blockers:
        out[r, c] = BLOCKER_COLOR

    erase = _erase_mask(g.shape, blockers)
    out[erase & (out != BLOCKER_COLOR)] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the left-side minimal-period motif across the grid, then erase it along diagonal rays cast from the green blockers.

    Input structure:
    - The background is black.
    - The leftmost 3-4 columns show a partial motif made of blue, red, and black cells.
    - The motif has a minimal column period of 2 or 3, while 3-5 green blocker pixels are scattered in the right-side region.

    Rule:
    - Infer the smallest positive column period consistent with the visible left motif columns.
    - Repeat that period across the full grid width to reconstruct the full periodic pattern.
    - Keep each green blocker in place, then cast rays from every blocker along all four diagonal directions until the grid boundary.
    - Erase every non-blocker cell touched by any ray to black.

    Key insight:
    - The visible left prefix is not the final object but the seed tile for a full-width periodic pattern, and the green pixels are emitters whose diagonal erase mask must be applied after the tiling is reconstructed.
    """
    return _solve(grid)


def _random_column(height: int) -> np.ndarray:
    max_seg_len = max(2, height // 4)
    for _ in range(80):
        col = np.zeros(height, dtype=int)
        intervals: list[tuple[int, int]] = []
        segment_count = random.randint(2, 3)
        for _ in range(segment_count):
            for _ in range(40):
                seg_len = random.randint(1, max_seg_len)
                start = random.randint(0, height - seg_len)
                end = start + seg_len - 1
                if all(end < a - 1 or start > b + 1 for a, b in intervals):
                    intervals.append((start, end))
                    col[start : end + 1] = random.choice(PATTERN_COLORS)
                    break
        if np.count_nonzero(col) >= 3:
            return col
    raise AssertionError("failed to build a visible motif column")


def _build_tile(height: int, period: int) -> np.ndarray | None:
    for _ in range(200):
        cols: list[np.ndarray] = []
        while len(cols) < period:
            col = _random_column(height)
            if all(not np.array_equal(col, prev) for prev in cols):
                cols.append(col)
        tile = np.column_stack(cols)
        if not np.any(tile == PATTERN_COLORS[0]) or not np.any(tile == PATTERN_COLORS[1]):
            continue
        if any(not np.any(tile[:, c] != BG) for c in range(period)):
            continue
        if _minimal_period(tile) != period:
            continue
        return tile
    return None


def _pick_blockers(base: np.ndarray, visible_w: int) -> list[tuple[int, int]] | None:
    h, w = base.shape
    allowed_cols = list(range(visible_w + 1, w))
    if len(allowed_cols) < BLOCKER_COUNT_RANGE[1]:
        return None

    total_fg = int(np.count_nonzero(base))
    for _ in range(300):
        count = random.randint(*BLOCKER_COUNT_RANGE)
        rows = random.sample(range(h), count)
        cols = random.sample(allowed_cols, count)
        random.shuffle(cols)
        blockers = list(zip(rows, cols))

        erase = _erase_mask(base.shape, blockers)
        erased_fg = erase & (base != BG)
        erased_count = int(erased_fg.sum())
        if erased_count < max(8, h // 2):
            continue
        if erased_count > total_fg - 6:
            continue

        if int(erased_fg[:, :visible_w].sum()) < 1:
            continue
        if int(erased_fg[:, visible_w:].sum()) < 5:
            continue

        erased_colors = set(map(int, np.unique(base[erased_fg])))
        erased_colors.discard(BG)
        if erased_colors != set(PATTERN_COLORS):
            continue

        covered = {int(base[r, c]) for r, c in blockers}
        if BG not in covered or not ({1, 2} & covered):
            continue

        min_blocker_col = min(c for _, c in blockers)
        max_blocker_col = max(c for _, c in blockers)
        if min_blocker_col > visible_w + 3:
            continue
        if max_blocker_col < w - 3:
            continue

        return blockers
    return None


def _render_input(tile: np.ndarray, width: int, visible_w: int, blockers: list[tuple[int, int]]) -> np.ndarray:
    h = tile.shape[0]
    inp = np.zeros((h, width), dtype=int)
    prefix = _tile_full_width(tile, visible_w)
    inp[:, :visible_w] = prefix
    for r, c in blockers:
        inp[r, c] = BLOCKER_COLOR
    return inp


def _is_semantic_instance(inp: np.ndarray, out: np.ndarray, period: int, visible_w: int) -> bool:
    prefix = inp[:, :visible_w]
    if _minimal_period(prefix) != period:
        return False

    blockers = _blockers(inp)
    if not (3 <= len(blockers) <= 5):
        return False
    rows = [r for r, _ in blockers]
    cols = [c for _, c in blockers]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return False
    if any(c <= visible_w for _, c in blockers):
        return False

    if np.any(_visible_mask(inp[:, visible_w:])):
        return False

    if set(map(int, np.unique(prefix))) - {0, 1, 2}:
        return False

    if not np.any(out[:, visible_w:] == 1) and not np.any(out[:, visible_w:] == 2):
        return False

    if not np.any(out == 1) or not np.any(out == 2) or not np.any(out == BLOCKER_COLOR):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    period = random.choice(PERIOD_CHOICES)
    visible_options = [w for w in VISIBLE_WIDTH_CHOICES if w >= period]
    visible_w = random.choice(visible_options)

    tile = _build_tile(height, period)
    if tile is None:
        return None

    base = _tile_full_width(tile, width)
    blockers = _pick_blockers(base, visible_w)
    if blockers is None:
        return None

    inp = _render_input(tile, width, visible_w, blockers)
    out = _solve(inp)

    if not _is_semantic_instance(inp, out, period, visible_w):
        return None

    pred = main(inp)
    if not np.array_equal(pred, out):
        return None

    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid periodic-tiling-with-diagonal-erasure input")
