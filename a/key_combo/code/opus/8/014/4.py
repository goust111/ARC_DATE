from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import paste
except Exception:
    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        mask: np.ndarray | None = None,
        overwrite: bool = True,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = patch.shape
        if not clip and (r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]):
            raise ValueError("paste out of bounds")
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(dst.shape[0], r0 + h)
        cc1 = min(dst.shape[1], c0 + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0
        pr1 = pr0 + (rr1 - rr0)
        pc1 = pc0 + (cc1 - cc0)
        patch_view = patch[pr0:pr1, pc0:pc1]
        select = np.ones_like(patch_view, dtype=bool)
        if mask is not None:
            select &= mask[pr0:pr1, pc0:pc1].astype(bool)
        if transparent is not None:
            select &= patch_view != int(transparent)
        if overwrite:
            dst[rr0:rr1, cc0:cc1][select] = patch_view[select]
        else:
            block = dst[rr0:rr1, cc0:cc1]
            select &= block == 0
            block[select] = patch_view[select]
        return dst


BG = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (13, 19)
GRID_WIDTH_CHOICES = (13, 15, 17, 19)
SEED_COUNT_CHOICES = (2, 3, 4)
SEED_ROW_MARGIN = 2
SEED_MIN_GAP = 3
GENERATION_ATTEMPTS = 500
_SEEN_OUTPUT_KEYS: set[tuple[tuple[int, int], tuple[int, ...], tuple[int, ...]]] = set()

PATCH_LIBRARY = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
)


def _seed_arrays_from_center(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mid = int(grid.shape[1] // 2)
    seed_rows = np.flatnonzero(grid[:, mid] != BG).astype(int)
    seed_colors = grid[seed_rows, mid].astype(int)
    return seed_rows, seed_colors, mid


def _paint_from_seeds(
    shape: tuple[int, int],
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    seed_colors: np.ndarray,
    *,
    metric: str = "chebyshev",
    tie_black: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros(shape, dtype=int)
    if seed_rows.size == 0:
        empty = np.zeros(shape, dtype=int)
        return out, empty, np.zeros(shape + (0,), dtype=bool)

    rr, cc = np.indices(shape)
    dr = np.abs(rr[:, :, None] - seed_rows[None, None, :])
    dc = np.abs(cc[:, :, None] - seed_cols[None, None, :])
    if metric == "chebyshev":
        dists = np.maximum(dr, dc)
    elif metric == "manhattan":
        dists = dr + dc
    else:
        raise ValueError(f"unsupported metric: {metric}")

    min_dist = dists.min(axis=2)
    closest = dists == min_dist[:, :, None]
    winner = dists.argmin(axis=2)
    if tie_black:
        paint_mask = (closest.sum(axis=2) == 1) & (min_dist % 2 == 0)
    else:
        paint_mask = min_dist % 2 == 0
    out[paint_mask] = seed_colors[winner[paint_mask]]
    return out, min_dist, closest


def _slow_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mid = w // 2
    seeds = [(int(r), int(g[r, mid])) for r in range(h) if g[r, mid] != BG]

    out = np.zeros_like(g)
    for r in range(h):
        for c in range(w):
            best_dist = None
            best_color = BG
            tie = False
            for sr, color in seeds:
                dist = max(abs(r - sr), abs(c - mid))
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_color = color
                    tie = False
                elif dist == best_dist:
                    tie = True
            if best_dist is not None and not tie and best_dist % 2 == 0:
                out[r, c] = best_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract center-column seeds and expand even Chebyshev rings with nearest-owner tie cuts.

    Input structure
    - The grid has a black background and scattered colored foreground pixels.
    - Only the odd-width grid's center column matters: its 2-4 non-black pixels are the seeds.
    - Foreground pixels outside the center column are distractors and must be ignored.

    Rule
    - Create an all-black output of the same shape.
    - For every output cell, compute its Chebyshev distance to each center-column seed.
    - If the nearest seed is unique and that minimum distance is even, color the cell with that seed's color.
    - If the minimum distance is odd, or if two or more seeds tie for nearest, leave the cell black.

    Key insight
    - The input is first filtered down to a one-dimensional seed list on the center axis, then that reduced layout is unfolded back into two-dimensional square rings split by nearest-seed ownership boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] % 2 == 1, "expected an odd number of columns"

    seed_rows, seed_colors, mid = _seed_arrays_from_center(g)
    seed_cols = np.full(seed_rows.shape, mid, dtype=int)
    out, _, _ = _paint_from_seeds(g.shape, seed_rows, seed_cols, seed_colors, metric="chebyshev", tie_black=True)
    return out


def _patch_from_coords(coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    return patch


def _choose_seed_rows(rng: np.random.Generator, height: int, count: int) -> tuple[int, ...]:
    rows = range(SEED_ROW_MARGIN, height - SEED_ROW_MARGIN)
    options = [
        combo
        for combo in combinations(rows, count)
        if all(combo[i + 1] - combo[i] >= SEED_MIN_GAP for i in range(len(combo) - 1))
    ]
    if not options:
        raise ValueError("no valid seed rows")
    return tuple(options[int(rng.integers(len(options)))])


def _can_place_patch(
    occupied: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    rows, cols = np.where(patch != BG)
    abs_rows = rows + int(r0)
    abs_cols = cols + int(c0)
    h, w = occupied.shape
    if abs_rows.size == 0:
        return False
    if abs_rows.min() < 0 or abs_cols.min() < 0 or abs_rows.max() >= h or abs_cols.max() >= w:
        return False
    if np.any(blocked[abs_rows, abs_cols]):
        return False
    for r, c in zip(abs_rows, abs_cols):
        r0n = max(0, int(r) - 1)
        r1n = min(h, int(r) + 2)
        c0n = max(0, int(c) - 1)
        c1n = min(w, int(c) + 2)
        if np.any(occupied[r0n:r1n, c0n:c1n]):
            return False
    return True


def _candidate_positions(
    shape: tuple[int, int],
    occupied: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    mid: int,
    *,
    side: str | None = None,
    near_axis: bool | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch.shape
    rows, cols = np.where(patch != BG)
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            abs_cols = cols + c0
            if side == "left" and np.any(abs_cols >= mid):
                continue
            if side == "right" and np.any(abs_cols <= mid):
                continue
            if near_axis is True and np.min(np.abs(abs_cols - mid)) > 2:
                continue
            if near_axis is False and np.min(np.abs(abs_cols - mid)) < 4:
                continue
            if _can_place_patch(occupied, blocked, patch, r0, c0):
                candidates.append((r0, c0))
    return candidates


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
    mid: int,
    *,
    side: str | None = None,
    near_axis: bool | None = None,
) -> bool:
    candidates = _candidate_positions(grid.shape, occupied, blocked, patch, mid, side=side, near_axis=near_axis)
    if not candidates:
        return False
    r0, c0 = candidates[int(rng.integers(len(candidates)))]
    paste(grid, patch, int(r0), int(c0), transparent=BG)
    occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch != BG
    return True


def _all_foreground_seeds(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(grid != BG)
    colors = grid[rows, cols].astype(int)
    return rows.astype(int), cols.astype(int), colors


def _case_key(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...], tuple[int, ...]]:
    rows, colors, _mid = _seed_arrays_from_center(grid)
    return tuple(grid.shape), tuple(map(int, rows)), tuple(map(int, colors))


def _validate_case(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30 or grid.shape[1] % 2 != 1:
        return False

    seed_rows, seed_colors, mid = _seed_arrays_from_center(grid)
    if not (2 <= seed_rows.size <= 4):
        return False
    if len(set(map(int, seed_colors))) != seed_colors.size:
        return False
    if seed_rows.size > 1 and np.min(np.diff(seed_rows)) < SEED_MIN_GAP:
        return False
    expected_center = np.zeros(grid.shape[0], dtype=int)
    expected_center[seed_rows] = seed_colors
    if not np.array_equal(grid[:, mid], expected_center):
        return False
    if not np.any(grid[:, :mid] != BG) or not np.any(grid[:, mid + 1 :] != BG):
        return False

    fast = main(grid)
    slow = _slow_expected(grid)
    if not np.array_equal(fast, slow):
        return False

    seed_cols = np.full(seed_rows.shape, mid, dtype=int)
    _, min_dist, closest = _paint_from_seeds(grid.shape, seed_rows, seed_cols, seed_colors, metric="chebyshev", tie_black=True)
    tie_even = (closest.sum(axis=2) > 1) & (min_dist % 2 == 0)
    if not np.any(tie_even):
        return False

    no_tie_cut, _, _ = _paint_from_seeds(grid.shape, seed_rows, seed_cols, seed_colors, metric="chebyshev", tie_black=False)
    if np.array_equal(fast, no_tie_cut):
        return False

    manhattan, _, _ = _paint_from_seeds(grid.shape, seed_rows, seed_cols, seed_colors, metric="manhattan", tie_black=True)
    if np.array_equal(fast, manhattan):
        return False

    all_rows, all_cols, all_colors = _all_foreground_seeds(grid)
    all_seed_out, _, _ = _paint_from_seeds(grid.shape, all_rows, all_cols, all_colors, metric="chebyshev", tie_black=True)
    if np.array_equal(fast, all_seed_out):
        return False

    fill_ratio = float(np.count_nonzero(fast)) / float(fast.size)
    if not (0.12 <= fill_ratio <= 0.72):
        return False

    for row, color in zip(seed_rows, seed_colors):
        d = np.maximum(np.abs(np.arange(grid.shape[0])[:, None] - int(row)), np.abs(np.arange(grid.shape[1])[None, :] - mid))
        if not np.any((fast == int(color)) & (d == 2)):
            return False

    return True


def _build_case(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    width = int(rng.choice(np.array(GRID_WIDTH_CHOICES)))
    seed_count = int(rng.choice(np.array(SEED_COUNT_CHOICES)))
    if height - 2 * SEED_ROW_MARGIN < 1:
        return None

    try:
        seed_rows = _choose_seed_rows(rng, height, seed_count)
    except ValueError:
        return None
    seed_colors = tuple(int(c) for c in rng.choice(np.array(COLOR_POOL), size=seed_count, replace=False))

    grid = np.zeros((height, width), dtype=int)
    mid = width // 2
    occupied = np.zeros_like(grid, dtype=bool)
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[:, mid] = True

    for row, color in zip(seed_rows, seed_colors):
        grid[int(row), mid] = int(color)
        occupied[int(row), mid] = True

    side = ("left", "right")[int(rng.integers(2))]
    near_patch = _patch_from_coords(PATCH_LIBRARY[int(rng.integers(len(PATCH_LIBRARY)))], int(rng.choice(np.array(seed_colors))))
    if not _place_patch(grid, occupied, blocked, near_patch, rng, mid, side=side, near_axis=True):
        return None

    far_side = "right" if side == "left" else "left"
    far_patch = _patch_from_coords(PATCH_LIBRARY[int(rng.integers(len(PATCH_LIBRARY)))], int(rng.choice(np.array(seed_colors))))
    if not _place_patch(grid, occupied, blocked, far_patch, rng, mid, side=far_side, near_axis=False):
        return None

    extra_count = int(rng.integers(1, 5))
    for _ in range(extra_count):
        patch = _patch_from_coords(PATCH_LIBRARY[int(rng.integers(len(PATCH_LIBRARY)))], int(rng.choice(np.array(seed_colors))))
        side_choice = (None, "left", "right")[int(rng.integers(3))]
        near_axis = (None, True, False)[int(rng.integers(3))]
        _place_patch(grid, occupied, blocked, patch, rng, mid, side=side_choice, near_axis=near_axis)

    if not _validate_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_case(rng)
        if grid is None:
            continue
        key = _case_key(grid)
        if key in _SEEN_OUTPUT_KEYS:
            continue
        _SEEN_OUTPUT_KEYS.add(key)
        return grid
    raise RuntimeError("failed to generate a valid unique input for puzzle 4")
