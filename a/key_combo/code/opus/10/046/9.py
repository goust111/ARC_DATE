from __future__ import annotations

import random

import numpy as np


def _pair_consistent(grid: np.ndarray, period_r: int, period_c: int) -> bool:
    phase_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != 0):
        color = int(grid[r, c])
        key = (int(r % period_r), int(c % period_c))
        prev = phase_colors.get(key)
        if prev is None:
            phase_colors[key] = color
        elif prev != color:
            return False
    return True


def _infer_periods(grid: np.ndarray) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    for period_r in range(1, height + 1):
        for period_c in range(1, width + 1):
            if _pair_consistent(grid, period_r, period_c):
                return period_r, period_c
    raise AssertionError("expected at least one valid period pair")


def _reconstruct_tile(grid: np.ndarray, period_r: int, period_c: int) -> np.ndarray:
    tile = np.zeros((period_r, period_c), dtype=int)
    seen = np.zeros((period_r, period_c), dtype=bool)
    for r, c in np.argwhere(grid != 0):
        rr = int(r % period_r)
        cc = int(c % period_c)
        color = int(grid[r, c])
        if seen[rr, cc]:
            assert int(tile[rr, cc]) == color, "same phase must agree"
        else:
            tile[rr, cc] = color
            seen[rr, cc] = True
    assert seen.all(), "every tile phase must be visible at least once"
    assert np.all(tile != 0), "black is reserved for holes"
    return tile


def _phase_row_hole_counts(grid: np.ndarray, period_r: int) -> list[int]:
    counts = []
    for row_phase in range(period_r):
        rows = np.arange(grid.shape[0]) % period_r == row_phase
        counts.append(int(np.sum(grid[rows, :] == 0)))
    return counts


def _exact_row_period(tile: np.ndarray) -> int:
    height = int(tile.shape[0])
    for period_r in range(1, height + 1):
        ok = True
        for r in range(height):
            if not np.array_equal(tile[r], tile[r % period_r]):
                ok = False
                break
        if ok:
            return period_r
    raise AssertionError("tile must have a row period")


def _exact_col_period(tile: np.ndarray) -> int:
    width = int(tile.shape[1])
    for period_c in range(1, width + 1):
        ok = True
        for c in range(width):
            if not np.array_equal(tile[:, c], tile[:, c % period_c]):
                ok = False
                break
        if ok:
            return period_c
    raise AssertionError("tile must have a col period")


def _build_tile(height: int, width: int) -> np.ndarray | None:
    palette_size = random.randint(3, min(6, height * width, 9))
    palette = random.sample(range(1, 10), palette_size)
    tile = np.array(random.choices(palette, k=height * width), dtype=int).reshape(height, width)

    positions = list(np.ndindex(height, width))
    random.shuffle(positions)
    for color, pos in zip(palette, positions):
        tile[pos] = color

    if len(np.unique(tile)) < 3:
        return None
    if _exact_row_period(tile) != height:
        return None
    if _exact_col_period(tile) != width:
        return None
    return tile


def _phase_positions(height: int, width: int, period_r: int, period_c: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    positions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            positions.setdefault((r % period_r, c % period_c), []).append((r, c))
    return positions


def _sample_holes_from_row(
    positions_by_phase: dict[tuple[int, int], list[tuple[int, int]]],
    row_phase: int,
    count: int,
    used: set[tuple[int, int]],
    period_c: int,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []

    per_phase_used = {phase: 0 for phase in positions_by_phase if phase[0] == row_phase}
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for col_phase in range(period_c):
        phase = (row_phase, col_phase)
        for r, c in positions_by_phase[phase]:
            if (r, c) not in used:
                candidates.append((r, c, phase))
    random.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    for r, c, phase in candidates:
        remaining_visible = len(positions_by_phase[phase]) - per_phase_used[phase] - 1
        if remaining_visible < 1:
            continue
        picked.append((r, c))
        per_phase_used[phase] += 1
        if len(picked) == count:
            return picked
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the wallpaper tile, count the most-corrupted phase row, and repeat the tile horizontally that many times.

    Input structure:
    - The grid is a wallpaper made by repeating one non-black rectangular tile.
    - Some wallpaper cells are replaced by black (0) holes; black is not part of the tile itself.
    - Enough non-black cells remain to recover the unique minimal row/column periods and every tile phase.

    Rule:
    - Infer the smallest row period and column period consistent with all non-black cells.
    - Reconstruct the complete tile from phase-aligned non-black observations.
    - Count black holes for each tile phase row `r mod period_r`; let `n` be the unique maximum count.
    - Output the reconstructed tile repeated horizontally `n` times with no gaps.

    Key insight:
    - The holes are not only corruption to repair: after periodic reconstruction, their distribution across phase rows encodes the output repeat count.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2 and grid.size > 0, "expected a non-empty 2D grid"
    assert np.any(grid == 0), "expected at least one black hole"

    period_r, period_c = _infer_periods(grid)
    tile = _reconstruct_tile(grid, period_r, period_c)

    counts = _phase_row_hole_counts(grid, period_r)
    n = max(counts)
    assert n > 0, "expected a positive repeat count"
    assert counts.count(n) == 1, "expected a unique most-corrupted phase row"
    return np.tile(tile, (1, n))


_TILE_SIZE_OPTIONS = (3, 4)
_REPEAT_RANGE = (2, 4)
_DENSE_COUNT_RANGE = (2, 5)
_OTHER_COUNT_MAX = 2
_GEN_ATTEMPTS = 400


def _validate_input(inp: np.ndarray, tile: np.ndarray, dense_row: int, dense_count: int) -> bool:
    period_r, period_c = _infer_periods(inp)
    if (period_r, period_c) != tile.shape:
        return False

    rebuilt = _reconstruct_tile(inp, period_r, period_c)
    if not np.array_equal(rebuilt, tile):
        return False

    counts = _phase_row_hole_counts(inp, period_r)
    if counts[dense_row] != dense_count:
        return False
    if counts.count(max(counts)) != 1 or int(np.argmax(counts)) != dense_row:
        return False
    if max(counts) < 2 or max(counts) > 5:
        return False

    expected = np.tile(tile, (1, dense_count))
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if max(inp.shape) > 30 or max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period_r = random.choice(_TILE_SIZE_OPTIONS)
    period_c = random.choice(_TILE_SIZE_OPTIONS)
    tile = _build_tile(period_r, period_c)
    if tile is None:
        return None

    repeat_r = random.randint(*_REPEAT_RANGE)
    repeat_c = random.randint(*_REPEAT_RANGE)
    wallpaper = np.tile(tile, (repeat_r, repeat_c))
    height, width = map(int, wallpaper.shape)

    positions_by_phase = _phase_positions(height, width, period_r, period_c)
    dense_row = random.randrange(period_r)
    max_dense = min(_DENSE_COUNT_RANGE[1], len([1 for r in range(height) for c in range(width) if r % period_r == dense_row]))
    dense_count = random.randint(_DENSE_COUNT_RANGE[0], max_dense)

    row_targets = [0] * period_r
    row_targets[dense_row] = dense_count
    other_rows = [row for row in range(period_r) if row != dense_row]
    random.shuffle(other_rows)
    for row_phase in other_rows:
        if dense_count == 2:
            row_targets[row_phase] = random.randint(0, 1)
        else:
            row_targets[row_phase] = random.randint(0, min(_OTHER_COUNT_MAX, dense_count - 1))
    if period_r > 1 and all(row_targets[row] == 0 for row in other_rows):
        row_targets[random.choice(other_rows)] = 1

    holes: set[tuple[int, int]] = set()
    for row_phase, target in enumerate(row_targets):
        picked = _sample_holes_from_row(positions_by_phase, row_phase, target, holes, period_c)
        if picked is None:
            return None
        holes.update(picked)

    inp = wallpaper.copy()
    for r, c in holes:
        inp[r, c] = 0

    if not _validate_input(inp, tile, dense_row, dense_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
