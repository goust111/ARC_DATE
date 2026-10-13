from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(10))
MAX_GRID_SIDE = 30

PERIOD_MIN = 2
PERIOD_MAX = 7
REPEATS_MIN = 2
REPEATS_MAX = 5
PALETTE_SIZE_MIN = 2
PALETTE_SIZE_MAX = 4
DEFECT_MIN = 2
DEFECT_MAX = 5

TILE_RETRIES = 120
GENERATE_RETRIES = 400


def _divisors(n: int, *, proper_only: bool = False) -> list[int]:
    out = [d for d in range(1, int(n) + 1) if int(n) % d == 0]
    if proper_only:
        out = [d for d in out if d < int(n)]
    return out


def _axis_period(tile: np.ndarray, axis: int) -> int:
    size = int(tile.shape[axis])
    for step in _divisors(size):
        if step == size:
            return size
        if axis == 0:
            if np.array_equal(tile[:-step, :], tile[step:, :]):
                return step
        else:
            if np.array_equal(tile[:, :-step], tile[:, step:]):
                return step
    return size


def _fold_score(grid: np.ndarray, pr: int, pc: int) -> tuple[int, int]:
    mismatches = 0
    ambiguous = 0
    for r in range(pr):
        for c in range(pc):
            values = grid[r::pr, c::pc].reshape(-1)
            colors, counts = np.unique(values, return_counts=True)
            best = int(counts.max())
            mismatches += int(values.size - best)
            ambiguous += int(np.count_nonzero(counts == best) > 1)
    return mismatches, ambiguous


def _candidate_periods(length: int) -> list[int]:
    candidates = _divisors(length, proper_only=True)
    return candidates if candidates else [int(length)]


def _best_period_pair(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int, int, int, int], int]:
    h, w = map(int, grid.shape)
    best_pair: tuple[int, int] | None = None
    best_key: tuple[int, int, int, int, int] | None = None
    n_best = 0

    for pr in _candidate_periods(h):
        for pc in _candidate_periods(w):
            mismatches, ambiguous = _fold_score(grid, pr, pc)
            key = (mismatches, ambiguous, pr * pc, pr, pc)
            if best_key is None or key < best_key:
                best_pair = (pr, pc)
                best_key = key
                n_best = 1
            elif key == best_key:
                n_best += 1

    assert best_pair is not None and best_key is not None
    return best_pair, best_key, n_best


def _reconstruct_output(grid: np.ndarray, pr: int, pc: int) -> np.ndarray:
    out = np.empty((pr, pc), dtype=int)

    for r in range(pr):
        for c in range(pc):
            values = grid[r::pr, c::pc].reshape(-1)
            colors, counts = np.unique(values, return_counts=True)
            winner = int(np.argmax(counts))
            max_count = int(counts[winner])
            assert np.count_nonzero(counts == max_count) == 1, "ambiguous modulo class"

            base_color = int(colors[winner])
            out[r, c] = base_color

            other_mask = colors != base_color
            other_counts = counts[other_mask]
            other_colors = colors[other_mask]
            if other_colors.size:
                assert int(other_counts.sum()) == 1, "expected at most one defect per modulo class"
                out[r, c] = int(other_colors[0])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a defected periodic tiling back into its fundamental tile and keep the folded defect colors.

    Input structure:
    - The full grid is made by repeating one unknown seed tile in both the row
      and column directions.
    - A small number of cells are changed to colors that do not belong to the
      seed tile; these are isolated defects scattered across different copies.
    - No two defects land on the same tile coordinate after taking row and
      column positions modulo the true tile periods.

    Rule:
    - Infer the smallest row period and smallest column period that explain the
      repeated pattern.
    - For each modulo-period cell class, recover the original tile color by the
      majority color among all repeated copies.
    - Output the recovered seed tile, then overwrite each folded tile position
      with its defect color when that modulo class contains one defect.

    Key insight:
    - The periodic tiling supplies a shared coordinate system: after folding the
      whole grid by the true periods, each defect becomes a single overlay on the
      canonical tile instead of noise to discard.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0

    (pr, pc), _key, n_best = _best_period_pair(g)
    assert n_best == 1, "period inference must be unique"
    return _reconstruct_output(g, pr, pc)


def _render_tiling(tile: np.ndarray, reps_r: int, reps_c: int) -> np.ndarray:
    return np.tile(tile, (int(reps_r), int(reps_c)))


def _random_tile(pr: int, pc: int) -> tuple[np.ndarray, list[int]] | None:
    max_palette = min(PALETTE_SIZE_MAX, pr * pc, len(PALETTE) - 2)
    if max_palette < PALETTE_SIZE_MIN:
        return None

    for _ in range(TILE_RETRIES):
        palette_size = random.randint(PALETTE_SIZE_MIN, max_palette)
        palette = random.sample(PALETTE, palette_size)
        tile = np.array(
            [[random.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        if len(np.unique(tile)) < 2:
            continue
        if _axis_period(tile, axis=0) != pr:
            continue
        if _axis_period(tile, axis=1) != pc:
            continue
        return tile, palette
    return None


def _contains_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    for r0 in range(gh - ph + 1):
        for c0 in range(gw - pw + 1):
            if np.array_equal(grid[r0 : r0 + ph, c0 : c0 + pw], patch):
                return True
    return False


def _apply_defects(
    tile: np.ndarray,
    reps_r: int,
    reps_c: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, int]]]:
    grid = _render_tiling(tile, reps_r, reps_c)
    output = tile.copy()

    tile_colors = set(map(int, np.unique(tile)))
    defect_palette = [color for color in PALETTE if color not in tile_colors]
    assert defect_palette

    copy_coords = [(tr, tc) for tr in range(reps_r) for tc in range(reps_c)]
    mod_coords = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]

    max_defects = min(DEFECT_MAX, len(copy_coords) - 1, len(mod_coords))
    assert max_defects >= DEFECT_MIN
    defect_count = random.randint(DEFECT_MIN, max_defects)

    chosen_copies = random.sample(copy_coords, defect_count)
    chosen_mods = random.sample(mod_coords, defect_count)

    defects: list[tuple[int, int, int, int, int]] = []
    for (tile_r, tile_c), (mod_r, mod_c) in zip(chosen_copies, chosen_mods):
        color = random.choice(defect_palette)
        rr = tile_r * tile.shape[0] + mod_r
        cc = tile_c * tile.shape[1] + mod_c
        grid[rr, cc] = color
        output[mod_r, mod_c] = color
        defects.append((tile_r, tile_c, mod_r, mod_c, int(color)))

    return grid, output, defects


def _validate_attempt(
    grid: np.ndarray,
    tile: np.ndarray,
    output: np.ndarray,
    defects: list[tuple[int, int, int, int, int]],
    reps_r: int,
    reps_c: int,
) -> bool:
    pr, pc = map(int, tile.shape)
    base_grid = _render_tiling(tile, reps_r, reps_c)

    if grid.ndim != 2 or output.ndim != 2:
        return False
    if max(grid.shape) > MAX_GRID_SIDE or max(output.shape) > MAX_GRID_SIDE:
        return False
    if _axis_period(tile, axis=0) != pr or _axis_period(tile, axis=1) != pc:
        return False
    if np.count_nonzero(grid != base_grid) != len(defects):
        return False

    defect_mods = {(mod_r, mod_c) for _tr, _tc, mod_r, mod_c, _color in defects}
    defect_copies = {(tr, tc) for tr, tc, _mr, _mc, _color in defects}
    if len(defect_mods) != len(defects):
        return False
    if len(defect_copies) != len(defects):
        return False
    if len(defects) < DEFECT_MIN or len(defect_copies) < 2:
        return False
    if all((tr, tc) == (0, 0) for tr, tc in defect_copies):
        return False

    tile_colors = set(map(int, np.unique(tile)))
    for _tr, _tc, _mr, _mc, color in defects:
        if color in tile_colors:
            return False

    if np.array_equal(output, tile):
        return False
    if _contains_patch(grid, output):
        return False

    best_pair, _best_key, n_best = _best_period_pair(grid)
    if best_pair != (pr, pc) or n_best != 1:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, output):
        return False
    return True


def _attempt() -> np.ndarray | None:
    pr = random.randint(PERIOD_MIN, PERIOD_MAX)
    pc = random.randint(PERIOD_MIN, PERIOD_MAX)

    max_reps_r = min(REPEATS_MAX, MAX_GRID_SIDE // pr)
    max_reps_c = min(REPEATS_MAX, MAX_GRID_SIDE // pc)
    if max_reps_r < REPEATS_MIN or max_reps_c < REPEATS_MIN:
        return None

    reps_r = random.randint(REPEATS_MIN, max_reps_r)
    reps_c = random.randint(REPEATS_MIN, max_reps_c)

    tile_pack = _random_tile(pr, pc)
    if tile_pack is None:
        return None
    tile, _ = tile_pack

    if reps_r * reps_c - 1 < DEFECT_MIN:
        return None

    try:
        grid, output, defects = _apply_defects(tile, reps_r, reps_c)
    except AssertionError:
        return None

    if not _validate_attempt(grid, tile, output, defects, reps_r, reps_c):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-folding input")
