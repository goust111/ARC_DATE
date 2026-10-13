import random

import numpy as np


BACKGROUND = 0
MARKER = 5
OCCLUDER = 8
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 500


def _marker_index(row: np.ndarray) -> int:
    cols = np.flatnonzero(row == MARKER)
    assert len(cols) == 1, "each row must contain exactly one grey marker"
    return int(cols[0])


def _restore_rows(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    restored = np.empty_like(grid)
    shifts: list[int] = []
    for r in range(grid.shape[0]):
        shift = _marker_index(grid[r])
        restored[r] = np.roll(grid[r], -shift)
        shifts.append(shift)
    assert np.all(restored[:, 0] == MARKER), "row restoration must return all markers to column 0"
    return restored, shifts


def _repair_mirror_damage(picture: np.ndarray) -> np.ndarray:
    out = np.asarray(picture, dtype=int).copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            left = int(out[r, c])
            right = int(out[r, mc])
            assert not (left == OCCLUDER and right == OCCLUDER), "both mirror cells cannot be occluded"
            if left == OCCLUDER:
                out[r, c] = right
            elif right == OCCLUDER:
                out[r, mc] = left
            else:
                assert left == right, "visible mirror cells must already match after realignment"
        if w % 2 == 1:
            assert int(out[r, w // 2]) != OCCLUDER, "the center column cannot be occluded"
    return out


def _remove_marker_without_realign(grid: np.ndarray) -> np.ndarray:
    rows = []
    for r in range(grid.shape[0]):
        c = _marker_index(grid[r])
        rows.append(np.concatenate([grid[r, :c], grid[r, c + 1 :]]))
    return np.stack(rows, axis=0)


def _try_common_roll_solution(grid: np.ndarray, shift: int) -> np.ndarray | None:
    rolled = np.roll(grid, -int(shift), axis=1)
    picture = rolled[:, 1:]
    try:
        return _repair_mirror_damage(picture)
    except AssertionError:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo each row's cyclic shift using the grey marker, then repair the mirrored picture by copying across each occluder.

    Input structure:
    - The grid is rectangular and every row contains exactly one grey marker pixel.
    - Before scrambling, all grey markers were in the leftmost column and the picture to their right was left-right symmetric.
    - Some non-grey picture cells were replaced by color 8, and no occluded cell has an occluded mirror partner.

    Rule:
    - For each row, cyclically shift it left until its grey marker returns to the first column.
    - Remove that recovered leftmost grey column.
    - In the remaining picture, replace every color-8 cell with the color at its horizontal mirror position.

    Key insight:
    - The grey pixel encodes that row's private cyclic offset, so the symmetry needed for repair appears only after rowwise realignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert int(np.sum(g == MARKER)) == g.shape[0], "grey should appear exactly once per row"

    restored, _ = _restore_rows(g)
    picture = restored[:, 1:]
    return _repair_mirror_damage(picture)


def _sample_partition(total: int, min_parts: int, max_parts: int, lo: int, hi: int, rng: np.random.Generator) -> list[int] | None:
    valid_counts = [n for n in range(min_parts, max_parts + 1) if lo * n <= total <= hi * n]
    if not valid_counts:
        return None
    parts = int(rng.choice(valid_counts))
    values: list[int] = []
    remaining = int(total)
    for i in range(parts):
        slots = parts - i - 1
        low = max(lo, remaining - hi * slots)
        high = min(hi, remaining - lo * slots)
        if low > high:
            return None
        value = int(rng.integers(low, high + 1))
        values.append(value)
        remaining -= value
    rng.shuffle(values)
    return values


def _expand_blocks(cells: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    return np.repeat(np.repeat(cells, row_heights, axis=0), col_widths, axis=1)


def _make_symmetric_picture(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(6, 15))
    width = int(rng.integers(7, 19))
    left_width = width // 2

    row_heights = _sample_partition(height, 2, 5, 1, 4, rng)
    col_widths = _sample_partition(left_width, 2, 4, 1, 3, rng)
    if row_heights is None or col_widths is None:
        return None

    palette_size = int(rng.integers(3, 6))
    palette = list(map(int, rng.choice(DRAW_COLORS, size=palette_size, replace=False)))
    cell_choices = np.array([BACKGROUND] + palette, dtype=int)

    probs = np.array([0.40] + [0.60 / len(palette)] * len(palette), dtype=float)
    coarse_left = rng.choice(cell_choices, size=(len(row_heights), len(col_widths)), p=probs).astype(int)
    if len(set(map(int, coarse_left.flatten()))) < 3:
        return None
    if np.count_nonzero(coarse_left != BACKGROUND) < max(3, coarse_left.size // 3):
        return None

    left = _expand_blocks(coarse_left, row_heights, col_widths)
    picture = np.full((height, width), BACKGROUND, dtype=int)
    picture[:, :left_width] = left
    picture[:, width - left_width :] = left[:, ::-1]

    if width % 2 == 1:
        center_choices = np.array([BACKGROUND] + palette, dtype=int)
        center_probs = np.array([0.55] + [0.45 / len(palette)] * len(palette), dtype=float)
        center_cells = rng.choice(center_choices, size=len(row_heights), p=center_probs).astype(int)
        if np.all(center_cells == BACKGROUND):
            center_cells[int(rng.integers(0, len(center_cells)))] = int(rng.choice(palette))
        picture[:, left_width] = np.repeat(center_cells, row_heights)

    if np.count_nonzero(picture != BACKGROUND) < max(8, picture.size // 6):
        return None
    return picture


def _damage_picture(clean: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int]]]:
    damaged = clean.copy()
    h, w = clean.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            if clean[r, c] != BACKGROUND and clean[r, mc] != BACKGROUND:
                candidates.append((r, c))

    assert candidates, "need at least one non-background mirror pair"
    max_damage = min(6, len(candidates))
    count = int(rng.integers(1, max_damage + 1))
    chosen_ids = rng.choice(len(candidates), size=count, replace=False)

    damaged_cells: list[tuple[int, int]] = []
    for idx in np.atleast_1d(chosen_ids):
        r, c = candidates[int(idx)]
        mc = w - 1 - c
        if bool(rng.integers(0, 2)):
            damaged[r, c] = OCCLUDER
            damaged_cells.append((r, c))
        else:
            damaged[r, mc] = OCCLUDER
            damaged_cells.append((r, mc))
    return damaged, damaged_cells


def _scramble_rows(damaged: np.ndarray, shifts: list[int]) -> np.ndarray:
    full = np.concatenate([np.full((damaged.shape[0], 1), MARKER, dtype=int), damaged], axis=1)
    scrambled = np.empty_like(full)
    for r, shift in enumerate(shifts):
        scrambled[r] = np.roll(full[r], int(shift))
    return scrambled


def _instance_is_valid(inp: np.ndarray, clean: np.ndarray, shifts: list[int], damaged_cells: list[tuple[int, int]]) -> bool:
    if len(set(shifts)) < 2 or all(shift == 0 for shift in shifts):
        return False
    if not damaged_cells:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, clean):
        return False

    restored, restored_shifts = _restore_rows(inp)
    if restored_shifts != list(shifts):
        return False
    damaged = restored[:, 1:]
    if not np.any(damaged == OCCLUDER):
        return False
    if np.array_equal(damaged, clean):
        return False

    wrong_cut = _remove_marker_without_realign(inp)
    try:
        wrong_cut_repaired = _repair_mirror_damage(wrong_cut)
    except AssertionError:
        wrong_cut_repaired = None
    if wrong_cut_repaired is not None and np.array_equal(wrong_cut_repaired, clean):
        return False

    for common_shift in range(inp.shape[1]):
        shortcut = _try_common_roll_solution(inp, common_shift)
        if shortcut is not None and np.array_equal(shortcut, clean):
            return False

    raw_without_markers = inp.copy()
    raw_without_markers[raw_without_markers == MARKER] = BACKGROUND
    if np.array_equal(raw_without_markers[:, 1:], clean):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        clean = _make_symmetric_picture(rng)
        if clean is None:
            continue

        damaged, damaged_cells = _damage_picture(clean, rng)
        full_width = damaged.shape[1] + 1
        shifts = list(map(int, rng.integers(0, full_width, size=damaged.shape[0])))
        scrambled = _scramble_rows(damaged, shifts)

        if _instance_is_valid(scrambled, clean, shifts, damaged_cells):
            return scrambled

    raise RuntimeError("failed to generate a valid scroll-reset mirror-repair input")

