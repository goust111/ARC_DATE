from __future__ import annotations

import itertools
import random

import numpy as np


COLOR_POOL = tuple(range(1, 10))
CYCLE_LENGTH_CHOICES = (3, 4, 5, 6)
GRID_SIDE_MIN = 10
GRID_SIDE_MAX = 28
PATCH_COUNT_RANGE = (1, 4)
PATCH_HEIGHT_MAX = 7
PATCH_WIDTH_MAX = 8
PATCH_PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 500
EXHAUSTIVE_SOLVER_MAX_COLORS = 7


def _render_weave(
    shape: tuple[int, int], cycle: tuple[int, ...] | list[int], sign: int, phase: int
) -> np.ndarray:
    height, width = map(int, shape)
    order = np.asarray(cycle, dtype=int)
    period = len(order)
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    indices = (phase + cols + sign * rows) % period
    return order[indices]


def _best_assignment(counts: np.ndarray) -> tuple[list[int], int]:
    period = int(counts.shape[0])
    state_count = 1 << period
    dp = np.full(state_count, -10**9, dtype=int)
    prev_mask = np.full(state_count, -1, dtype=int)
    prev_color = np.full(state_count, -1, dtype=int)
    dp[0] = 0

    for mask in range(state_count):
        residue = mask.bit_count()
        if residue >= period or dp[mask] < 0:
            continue
        base_score = int(dp[mask])
        for color_idx in range(period):
            if mask & (1 << color_idx):
                continue
            next_mask = mask | (1 << color_idx)
            score = base_score + int(counts[residue, color_idx])
            if score > dp[next_mask]:
                dp[next_mask] = score
                prev_mask[next_mask] = mask
                prev_color[next_mask] = color_idx

    assignment = [0] * period
    mask = state_count - 1
    for residue in range(period - 1, -1, -1):
        assignment[residue] = int(prev_color[mask])
        mask = int(prev_mask[mask])
    return assignment, int(dp[-1])


def _solve_exhaustive(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], int, int], int]:
    colors = sorted(set(map(int, grid.flat)))
    period = len(colors)
    anchor = colors[0]
    others = [color for color in colors if color != anchor]
    rows = np.arange(grid.shape[0], dtype=int)[:, None]
    cols = np.arange(grid.shape[1], dtype=int)[None, :]

    best_match = -1
    best_grid: np.ndarray | None = None
    best_meta: tuple[tuple[int, ...], int, int] | None = None

    for perm in itertools.permutations(others):
        order = np.array((anchor, *perm), dtype=int)
        for sign in (1, -1):
            base = cols + sign * rows
            for phase in range(period):
                prediction = order[(phase + base) % period]
                match_count = int(np.count_nonzero(prediction == grid))
                if match_count > best_match:
                    best_match = match_count
                    best_grid = prediction.copy()
                    best_meta = (tuple(map(int, order)), int(sign), int(phase))

    assert best_grid is not None and best_meta is not None
    return best_grid, best_meta, best_match


def _solve_residue_majority(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, ...], int, int], int]:
    colors = sorted(set(map(int, grid.flat)))
    period = len(colors)
    color_to_index = {color: idx for idx, color in enumerate(colors)}
    rows = np.arange(grid.shape[0], dtype=int)[:, None]
    cols = np.arange(grid.shape[1], dtype=int)[None, :]

    best_match = -1
    best_grid: np.ndarray | None = None
    best_meta: tuple[tuple[int, ...], int, int] | None = None

    for sign in (1, -1):
        counts = np.zeros((period, period), dtype=int)
        for r in range(grid.shape[0]):
            residues = (np.arange(grid.shape[1], dtype=int) + sign * r) % period
            for c, residue in enumerate(residues):
                color = int(grid[r, c])
                counts[int(residue), color_to_index[color]] += 1

        assignment, _score = _best_assignment(counts)
        order = np.array([colors[idx] for idx in assignment], dtype=int)
        prediction = order[(cols + sign * rows) % period]
        match_count = int(np.count_nonzero(prediction == grid))
        if match_count > best_match:
            best_match = match_count
            best_grid = prediction.copy()
            best_meta = (tuple(map(int, order)), int(sign), 0)

    assert best_grid is not None and best_meta is not None
    return best_grid, best_meta, best_match


def _infer_weave(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], int, int], int]:
    period = len(set(map(int, grid.flat)))
    if period <= EXHAUSTIVE_SOLVER_MAX_COLORS:
        return _solve_exhaustive(grid)
    return _solve_residue_majority(grid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the painted patches by reconstructing the diagonal phase-shifted color weave.

    Input structure:
    - The full grid is a periodic multicolor weave with no background color.
    - Colors follow a fixed left-to-right cycle, while each next row is shifted by one cycle step.
    - One or more axis-aligned rectangles have been repainted solidly with a color that already belongs to the cycle.

    Rule:
    - Infer the cycle order, the row-shift direction, and the global phase from the cells that still obey the weave.
    - Rebuild the full diagonal weave implied by that global law.
    - Overwrite only the law-violating cells; cells that already match the inferred weave stay unchanged.

    Key insight:
    - The occluder is not an extra marker color: it is one of the pattern colors, so membership is useless.
    - The only reliable test for occlusion is whether a cell fits the single global diagonal cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    repaired, _meta, _match_count = _infer_weave(g)
    return repaired


def _visible_signal_is_strong(base: np.ndarray, painted: np.ndarray, cycle: list[int]) -> bool:
    mismatch = painted != base
    mismatch_count = int(np.count_nonzero(mismatch))
    if mismatch_count == 0:
        return False

    ratio = mismatch_count / float(base.size)
    if ratio < 0.08 or ratio > 0.32:
        return False

    clean = ~mismatch
    for color in cycle:
        if int(np.count_nonzero(base[clean] == color)) < 5:
            return False

    min_clean_per_row = max(3, base.shape[1] // 4)
    min_clean_per_col = max(3, base.shape[0] // 4)
    if np.any(np.count_nonzero(clean, axis=1) < min_clean_per_row):
        return False
    if np.any(np.count_nonzero(clean, axis=0) < min_clean_per_col):
        return False
    return True


def _try_place_patches(base: np.ndarray, cycle: list[int]) -> np.ndarray | None:
    height, width = map(int, base.shape)
    patch_target = random.randint(*PATCH_COUNT_RANGE)
    painted = base.copy()
    reserved = np.zeros_like(base, dtype=bool)
    placed = 0

    for _ in range(PATCH_PLACEMENT_ATTEMPTS):
        if placed >= patch_target:
            break

        max_patch_h = min(PATCH_HEIGHT_MAX, height - 1)
        max_patch_w = min(PATCH_WIDTH_MAX, width - 1)
        if max_patch_h < 2 or max_patch_w < 2:
            return None

        patch_h = random.randint(2, max_patch_h)
        patch_w = random.randint(2, max_patch_w)
        r0 = random.randint(0, height - patch_h)
        c0 = random.randint(0, width - patch_w)

        rr0 = max(0, r0 - 1)
        rr1 = min(height, r0 + patch_h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(width, c0 + patch_w + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue

        paint_color = random.choice(cycle)
        original_patch = base[r0 : r0 + patch_h, c0 : c0 + patch_w]
        mismatch_count = int(np.count_nonzero(original_patch != paint_color))
        if mismatch_count < max(4, original_patch.size // 3):
            continue

        painted[r0 : r0 + patch_h, c0 : c0 + patch_w] = paint_color
        reserved[r0 : r0 + patch_h, c0 : c0 + patch_w] = True
        placed += 1

    if placed == 0:
        return None
    return painted


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        period = random.choice(CYCLE_LENGTH_CHOICES)
        cycle = random.sample(COLOR_POOL, period)
        sign = random.choice((1, -1))
        phase = random.randrange(period)

        min_side = max(GRID_SIDE_MIN, period + 4)
        height = random.randint(min_side, GRID_SIDE_MAX)
        width = random.randint(min_side, GRID_SIDE_MAX)
        base = _render_weave((height, width), cycle, sign, phase)

        painted = _try_place_patches(base, cycle)
        if painted is None:
            continue
        if not _visible_signal_is_strong(base, painted, cycle):
            continue

        repaired, _meta, _match_count = _infer_weave(painted)
        if not np.array_equal(repaired, base):
            continue
        return painted

    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
