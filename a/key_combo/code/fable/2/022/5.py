from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = list(range(1, 10))
GEN_ATTEMPTS = 500
MAX_ROW_PERIOD = 5
MAX_COL_PERIOD = 7


def _row_key(row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in row.tolist())


def _canonical_roll(row: np.ndarray) -> tuple[int, ...]:
    best = _row_key(row)
    for shift in range(1, row.size):
        candidate = _row_key(np.roll(row, shift))
        if candidate < best:
            best = candidate
    return best


def _same_up_to_roll(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    for shift in range(a.size):
        if np.array_equal(a, np.roll(b, shift)):
            return True
    return False


def _string_period(row: np.ndarray) -> int:
    width = int(row.size)
    for period in range(1, width + 1):
        if np.array_equal(row[:-period], row[period:]):
            return period
    return width


def _common_string_period(rows: np.ndarray) -> int:
    width = int(rows.shape[1])
    for period in range(1, width + 1):
        if all(np.array_equal(row[:-period], row[period:]) for row in rows):
            return period
    return width


def _vertical_period(row_keys: list[tuple[int, ...]]) -> int:
    height = len(row_keys)
    for period in range(1, height + 1):
        if all(row_keys[r] == row_keys[r % period] for r in range(height)):
            return period
    return height


def _roll_matches(row: np.ndarray, target: np.ndarray) -> list[int]:
    return [shift for shift in range(row.size) if np.array_equal(np.roll(row, shift), target)]


def _repeat_row(base: np.ndarray, width: int) -> np.ndarray:
    return np.take(base, np.arange(width) % base.size).astype(int)


def _build_wallpaper(phase_rows: np.ndarray, height: int) -> np.ndarray:
    return np.vstack([phase_rows[r % phase_rows.shape[0]] for r in range(height)]).astype(int)


def _phase_counts(height: int, row_period: int) -> list[int]:
    return [len(range(phase, height, row_period)) for phase in range(row_period)]


def _restore_metadata(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height >= 2 and width >= 2

    keys = [_canonical_roll(g[r]) for r in range(height)]
    row_period = _vertical_period(keys)
    assert row_period < height, "expected a repeated vertical row cycle"

    template_rows: list[np.ndarray] = []
    exact_mode_counts: list[int] = []
    for phase in range(row_period):
        phase_rows = [_row_key(g[r]) for r in range(phase, height, row_period)]
        ranked = Counter(phase_rows).most_common(2)
        assert ranked, "each phase must appear at least once"
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "phase template must be uniquely determined"
        exact_mode_counts.append(int(ranked[0][1]))
        template_rows.append(np.array(ranked[0][0], dtype=int))

    templates = np.stack(template_rows)
    col_period = _common_string_period(templates)
    assert col_period < width, "expected a non-trivial horizontal period"

    restored = np.empty_like(g)
    restore_shifts: list[int] = []
    for r in range(height):
        target = templates[r % row_period]
        matches = _roll_matches(g[r], target)
        assert len(matches) == 1, "each row must have a unique rollback amount"
        restore_shifts.append(int(matches[0]))
        restored[r] = np.roll(g[r], matches[0])

    for r in range(height - row_period):
        assert np.array_equal(restored[r], restored[r + row_period]), "restored rows must repeat vertically"
    assert _common_string_period(restored[:row_period]) == col_period

    return {
        "restored": restored,
        "row_period": row_period,
        "col_period": col_period,
        "templates": templates,
        "restore_shifts": np.array(restore_shifts, dtype=int),
        "row_keys": keys,
        "exact_mode_counts": exact_mode_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Undo cyclically rolled horizontal bands to recover the original periodic wallpaper.

    Input structure:
    - The grid is a full-color wallpaper crop with no blank cells.
    - Rows belong to a repeating vertical phase cycle, and each phase row also follows a shorter horizontal period.
    - Some contiguous horizontal bands have been cyclically rolled left or right by an unmarked amount.

    Rule:
    - Group rows by their roll-invariant phase identity to infer the wallpaper's vertical period.
    - For each phase, use the uniquely most common exact row as the unshifted template row.
    - For every row that disagrees with its phase template, find the unique cyclic rollback that makes it match and apply it.

    Key insight:
    - A band shift preserves each row's cyclic content class but destroys its absolute phase, so the solution is to infer the row phase cycle first and then recover each row's hidden horizontal offset relative to the majority template of that phase.
    """
    return _restore_metadata(np.asarray(grid, dtype=int))["restored"]


def _row_has_no_cyclic_symmetry(row: np.ndarray) -> bool:
    return all(not np.array_equal(row, np.roll(row, shift)) for shift in range(1, row.size))


def _generate_phase_rows(row_period: int, col_period: int, width: int, rng: random.Random) -> np.ndarray | None:
    palette_size = rng.randint(3, min(6, len(PALETTE)))
    palette = rng.sample(PALETTE, palette_size)
    rows: list[np.ndarray] = []

    for _ in range(200):
        if len(rows) == row_period:
            break
        base = np.array([rng.choice(palette) for _ in range(col_period)], dtype=int)
        full_row = _repeat_row(base, width)
        if len(set(base.tolist())) < 2:
            continue
        if _string_period(full_row) != col_period:
            continue
        if not _row_has_no_cyclic_symmetry(full_row):
            continue
        if any(_same_up_to_roll(full_row, prev) for prev in rows):
            continue
        rows.append(full_row)

    if len(rows) != row_period:
        return None

    phase_rows = np.stack(rows)
    if _common_string_period(phase_rows) != col_period:
        return None
    if len({_canonical_roll(row) for row in phase_rows}) != row_period:
        return None
    return phase_rows


def _sample_bands(height: int, width: int, row_period: int, rng: random.Random) -> list[tuple[int, int, int]] | None:
    phase_limits = [(count - 1) // 2 for count in _phase_counts(height, row_period)]
    total_limit = (height - 1) // 2
    max_band_len = min(4, height - 1)
    if total_limit <= 0 or max_band_len <= 0:
        return None

    for _ in range(200):
        target_bands = rng.randint(1, min(4, max(1, height // 3)))
        bands: list[tuple[int, int, int]] = []
        shifted_by_phase = [0] * row_period
        total_shifted = 0
        blocked = np.zeros(height, dtype=bool)

        for band_index in range(target_bands):
            options: list[tuple[int, int]] = []
            for start in range(height):
                for band_len in range(1, max_band_len + 1):
                    end = start + band_len
                    if end > height:
                        break
                    if blocked[start:end].any():
                        continue
                    if total_shifted + band_len > total_limit:
                        continue

                    trial = shifted_by_phase[:]
                    ok = True
                    for r in range(start, end):
                        phase = r % row_period
                        trial[phase] += 1
                        if trial[phase] > phase_limits[phase]:
                            ok = False
                            break
                    if ok:
                        options.append((start, end))

            if not options:
                break

            weights = [3 if (end - start) > 1 else 1 for start, end in options]
            start, end = rng.choices(options, weights=weights, k=1)[0]

            shift_choices = list(range(1, width))
            shift = rng.choice(shift_choices)
            if rng.random() < 0.5:
                shift = -shift
            bands.append((start, end, shift))

            total_shifted += end - start
            for r in range(start, end):
                shifted_by_phase[r % row_period] += 1

            left = max(0, start - 1)
            right = min(height, end + 1)
            blocked[left:right] = True

            if band_index + 1 == target_bands:
                break

        if not bands:
            continue

        bands.sort()
        if total_shifted == 0 or total_shifted > total_limit:
            continue
        if any(shifted_by_phase[phase] > phase_limits[phase] for phase in range(row_period)):
            continue
        return bands

    return None


def _apply_bands(wallpaper: np.ndarray, bands: list[tuple[int, int, int]]) -> np.ndarray:
    grid = wallpaper.copy()
    for start, end, shift in bands:
        for r in range(start, end):
            grid[r] = np.roll(grid[r], shift)
    return grid


def _validate_instance(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    row_period: int,
    col_period: int,
    bands: list[tuple[int, int, int]],
) -> bool:
    try:
        meta = _restore_metadata(input_grid)
    except AssertionError:
        return False

    if not np.array_equal(meta["restored"], output_grid):
        return False
    if meta["row_period"] != row_period or meta["col_period"] != col_period:
        return False
    if not np.array_equal(main(input_grid), output_grid):
        return False

    height, width = input_grid.shape
    restore_shifts = meta["restore_shifts"]
    if int(np.count_nonzero(restore_shifts)) == 0:
        return False
    if int(np.count_nonzero(restore_shifts)) > (height - 1) // 2:
        return False

    expected_restore = np.zeros(height, dtype=int)
    for start, end, applied_shift in bands:
        rollback = (-applied_shift) % width
        for r in range(start, end):
            expected_restore[r] = rollback
    if not np.array_equal(restore_shifts, expected_restore):
        return False

    for phase in range(row_period):
        rows = list(range(phase, height, row_period))
        zero_count = sum(int(restore_shifts[r]) == 0 for r in rows)
        if zero_count <= len(rows) // 2:
            return False

        ranked = Counter(_row_key(input_grid[r]) for r in rows).most_common(2)
        if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
            return False

    if _common_string_period(output_grid[:row_period]) != col_period:
        return False
    if any(not np.array_equal(output_grid[r], output_grid[r + row_period]) for r in range(height - row_period)):
        return False

    return True


def _generate_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    row_period = rng.randint(2, MAX_ROW_PERIOD)
    col_period = rng.randint(2, MAX_COL_PERIOD)

    possible_widths = [w for w in range(max(10, 2 * col_period + 1), 31) if w % col_period != 0]
    if not possible_widths:
        return None
    width = rng.choice(possible_widths)

    min_height = max(8, 3 * row_period)
    max_height = min(30, 6 * row_period + (row_period - 1))
    if min_height > max_height:
        return None
    height = rng.randint(min_height, max_height)

    phase_rows = _generate_phase_rows(row_period, col_period, width, rng)
    if phase_rows is None:
        return None

    wallpaper = _build_wallpaper(phase_rows, height)
    bands = _sample_bands(height, width, row_period, rng)
    if bands is None:
        return None

    input_grid = _apply_bands(wallpaper, bands)
    if np.array_equal(input_grid, wallpaper):
        return None

    if not _validate_instance(input_grid, wallpaper, row_period, col_period, bands):
        return None

    return input_grid, wallpaper


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _generate_instance(random)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid rolled-wallpaper instance")
