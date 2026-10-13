from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared_for_opus_2_022",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color


CONTROL_COLOR = Color.PINK
MARKER_COLOR = Color.RED
MISSING_COLOR = Color.BLACK
WALL_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON]


def _read_shift(grid: np.ndarray) -> int:
    control = np.asarray(grid[:, 0], dtype=int)
    marker_rows = np.flatnonzero(control == MARKER_COLOR)
    assert marker_rows.size == 1, "control column must contain exactly one red marker"
    assert np.all((control == CONTROL_COLOR) | (control == MARKER_COLOR)), "control column must be pink except for the red marker"
    return int(marker_rows[0])


def _phase_fit(grid: np.ndarray, pr: int, pc: int) -> tuple[bool, np.ndarray, np.ndarray]:
    tile = np.full((pr, pc), -1, dtype=int)
    seen = np.zeros((pr, pc), dtype=bool)
    for r, c in np.argwhere(grid != MISSING_COLOR):
        rr = int(r) % pr
        cc = int(c) % pc
        color = int(grid[r, c])
        if seen[rr, cc] and int(tile[rr, cc]) != color:
            return False, seen, tile
        tile[rr, cc] = color
        seen[rr, cc] = True
    return True, seen, tile


def _infer_min_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = map(int, grid.shape)
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            consistent, seen, tile = _phase_fit(grid, pr, pc)
            if consistent and bool(seen.all()):
                return pr, pc, tile
    raise AssertionError("failed to infer a complete periodic tile")


def _render_tiled(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    row_ids = np.arange(h) % tile.shape[0]
    col_ids = np.arange(w) % tile.shape[1]
    return tile[row_ids[:, None], col_ids[None, :]].astype(int, copy=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo the control-column cyclic shift, recover the smallest repeating wallpaper tile, and fill the black damage.

    Input structure:
    - The leftmost column is a pink control column containing exactly one red marker pixel.
    - The remaining columns are a color wallpaper that originally repeated by a small row-period/col-period tile.
    - Before corruption, that wallpaper was cyclically shifted downward by the marker row index, then some cells were replaced by black.

    Rule:
    - Read the red marker row index r from the control column and delete that control column.
    - Cyclically shift the remaining grid upward by r rows to restore the original periodic alignment.
    - Find the smallest row period pr and column period pc whose modulo-phase cells agree on every visible non-black pixel.
    - Reconstruct the full pr x pc tile from those consistent phase colors and tile it back across the whole grid to replace the black damage.

    Key insight:
    - The missing cells are easy only after undoing the hidden cyclic roll: without that alignment step, the same modulo phase contains conflicting colors and the small wallpaper period disappears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a control column plus wallpaper"

    shift = _read_shift(g)
    wall = g[:, 1:]
    aligned = np.roll(wall, -shift, axis=0)
    _pr, _pc, tile = _infer_min_period(aligned)
    return _render_tiled(tile, aligned.shape)


# --------------- generator ---------------

GEN_ROW_PERIODS = [3, 4]
GEN_COL_PERIODS = [3, 4]
GEN_HEIGHT_RANGE = range(9, 16)
GEN_WIDTH_RANGE = range(10, 16)
GEN_COLOR_COUNT_CHOICES = [3, 4]
GEN_MISSING_FRACTION_RANGE = (0.15, 0.25)
GEN_TILE_ATTEMPTS = 200
GEN_RETRIES = 400


def _axis_period(tile: np.ndarray, axis: int) -> int:
    length = tile.shape[axis]
    for period in range(1, length + 1):
        if np.array_equal(tile, np.roll(tile, -period, axis=axis)):
            return period
    raise AssertionError("every finite tile must have a period")


def _make_tile(pr: int, pc: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(GEN_TILE_ATTEMPTS):
        values = [random.choice(colors) for _ in range(pr * pc)]
        tile = np.array(values, dtype=int).reshape(pr, pc)

        if len(set(map(int, tile.flat))) != len(colors):
            continue
        if _axis_period(tile, 0) != pr or _axis_period(tile, 1) != pc:
            continue
        if len({tuple(map(int, tile[r, :])) for r in range(pr)}) != pr:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(pc)}) != pc:
            continue
        return tile
    return None


def _valid_heights(pr: int) -> list[int]:
    return [h for h in GEN_HEIGHT_RANGE if h % pr != 0]


def _pick_keeper_count(group_size: int) -> int:
    if group_size <= 2:
        return 1
    return 1 if random.random() < 0.6 else 2


def _corrupt_aligned(aligned: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    h, w = map(int, aligned.shape)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            groups.setdefault((r % pr, c % pc), []).append((r, c))

    protected: set[tuple[int, int]] = set()
    for cells in groups.values():
        keep = _pick_keeper_count(len(cells))
        protected.update(random.sample(cells, keep))

    all_cells = [(r, c) for r in range(h) for c in range(w)]
    candidates = [cell for cell in all_cells if cell not in protected]
    min_missing = max(1, int(np.ceil(GEN_MISSING_FRACTION_RANGE[0] * h * w)))
    max_missing = int(np.floor(GEN_MISSING_FRACTION_RANGE[1] * h * w))
    if max_missing < min_missing or len(candidates) < min_missing:
        return None

    missing_count = random.randint(min_missing, min(max_missing, len(candidates)))
    missing_cells = set(random.sample(candidates, missing_count))

    corrupted = aligned.copy()
    for r, c in missing_cells:
        corrupted[r, c] = MISSING_COLOR

    if np.any(np.all(corrupted == MISSING_COLOR, axis=0)):
        return None
    if np.any(np.all(corrupted == MISSING_COLOR, axis=1)):
        return None
    return corrupted


def _add_control_column(wall: np.ndarray, shift: int) -> np.ndarray:
    control = np.full((wall.shape[0], 1), CONTROL_COLOR, dtype=int)
    control[shift, 0] = MARKER_COLOR
    return np.hstack([control, wall])


def _direct_completion_matches_target(
    rolled_corrupted: np.ndarray,
    shift: int,
    expected: np.ndarray,
) -> bool:
    try:
        _pr, _pc, direct_tile = _infer_min_period(rolled_corrupted)
    except AssertionError:
        return False
    direct_completed = _render_tiled(direct_tile, rolled_corrupted.shape)
    return np.array_equal(np.roll(direct_completed, -shift, axis=0), expected)


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    aligned_corrupted: np.ndarray,
    rolled_corrupted: np.ndarray,
    shift: int,
    true_pr: int,
    true_pc: int,
) -> bool:
    consistent, seen, _tile = _phase_fit(aligned_corrupted, true_pr, true_pc)
    if not consistent or not bool(seen.all()):
        return False

    inferred_pr, inferred_pc, inferred_tile = _infer_min_period(aligned_corrupted)
    if (inferred_pr, inferred_pc) != (true_pr, true_pc):
        return False
    if not np.array_equal(_render_tiled(inferred_tile, aligned_corrupted.shape), expected):
        return False

    rolled_consistent, _rolled_seen, _rolled_tile = _phase_fit(rolled_corrupted, true_pr, true_pc)
    if rolled_consistent:
        return False

    if _direct_completion_matches_target(rolled_corrupted, shift, expected):
        return False

    if not np.any(aligned_corrupted == MISSING_COLOR):
        return False

    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    pr = random.choice(GEN_ROW_PERIODS)
    pc = random.choice(GEN_COL_PERIODS)
    height = random.choice(_valid_heights(pr))
    width = random.choice(list(GEN_WIDTH_RANGE))
    shift_choices = [r for r in range(1, height) if r % pr != 0]
    if not shift_choices:
        return None
    shift = random.choice(shift_choices)

    color_count = random.choice(GEN_COLOR_COUNT_CHOICES)
    colors = random.sample(WALL_COLORS, color_count)
    tile = _make_tile(pr, pc, colors)
    if tile is None:
        return None

    expected = _render_tiled(tile, (height, width))
    aligned_corrupted = _corrupt_aligned(expected, pr, pc)
    if aligned_corrupted is None:
        return None

    rolled_corrupted = np.roll(aligned_corrupted, shift, axis=0)
    inp = _add_control_column(rolled_corrupted, shift)

    if not _validate_instance(inp, expected, aligned_corrupted, rolled_corrupted, shift, pr, pc):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic-roll wallpaper instance")
