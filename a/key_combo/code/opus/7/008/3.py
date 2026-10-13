import random

import numpy as np


BG = 0
MARKER = 4
PALETTE = (1, 2, 3, 5)
HEIGHT_RANGE = range(8, 13)
PERIOD_RANGE = range(4, 7)
REPEAT_RANGE = range(3, 6)
MAX_ATTEMPTS = 400


def _periods_with_wildcard(grid: np.ndarray, wildcard: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    periods = []
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        ok = True
        for r in range(height):
            for c in range(period):
                vals = {int(v) for v in g[r, c:width:period] if int(v) != wildcard}
                if len(vals) > 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            periods.append(period)
    return periods


def _infer_period(grid: np.ndarray, wildcard: int = MARKER) -> int:
    periods = _periods_with_wildcard(grid, wildcard)
    if not periods:
        raise ValueError("no valid horizontal period")
    return periods[0]


def _recover_unit(grid: np.ndarray, period: int, wildcard: int = MARKER) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    if width % period != 0:
        raise ValueError("grid width must be a multiple of the inferred period")

    unit = np.full((height, period), -1, dtype=int)
    unresolved = []
    for r in range(height):
        for c in range(period):
            vals = {int(v) for v in g[r, c:width:period] if int(v) != wildcard}
            if len(vals) > 1:
                raise ValueError("inconsistent orbit values")
            if vals:
                unit[r, c] = vals.pop()
            else:
                unresolved.append((r, c))

    pending = unresolved
    while pending:
        next_pending = []
        changed = False
        for r, c in pending:
            mirror = period - 1 - c
            if unit[r, mirror] != -1:
                unit[r, c] = int(unit[r, mirror])
                changed = True
            else:
                next_pending.append((r, c))
        if not changed:
            raise ValueError("symmetry cannot resolve all marker holes")
        pending = next_pending

    if not np.array_equal(unit, unit[:, ::-1]):
        raise ValueError("recovered unit is not vertically symmetric")
    return unit


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore all yellow marker holes by combining horizontal periodicity with vertical mirror symmetry inside each repeat unit.

    Input structure:
    - A black background contains a horizontally repeated pattern with period 4-6 columns and height around 10 rows.
    - The repeated unit is vertically mirror-symmetric inside each period block.
    - Yellow cells are marker holes that replace some original pixels, including some positions whose entire periodic orbit is hidden.

    Rule:
    - Treat yellow as the only wildcard color and infer the shortest horizontal period consistent with the remaining pixels.
    - For each row and period-column orbit, use any visible non-yellow copy to recover that orbit's true value.
    - If an orbit is fully hidden across all repeats, copy the value from its mirror partner inside the period block.
    - Replace every yellow marker with the recovered value and leave all other pixels unchanged.

    Key insight:
    - Periodicity recovers holes when another repeat still shows the same orbit, but some orbits are fully occluded and can only be restored from the unit's internal mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    period = _infer_period(g, MARKER)
    unit = _recover_unit(g, period, MARKER)

    out = g.copy()
    holes = np.argwhere(g == MARKER)
    for r, c in holes:
        out[int(r), int(c)] = int(unit[int(r), int(c) % period])
    return out


def _compose_symmetric_unit(columns: list[np.ndarray], period: int) -> np.ndarray:
    if period % 2 == 0:
        full_columns = columns + columns[::-1]
    else:
        full_columns = columns + columns[-2::-1]
    return np.stack(full_columns, axis=1).astype(int)


def _random_column(height: int, colors: tuple[int, ...]) -> np.ndarray:
    column = np.zeros(height, dtype=int)
    for r in range(height):
        if random.random() < 0.45:
            column[r] = BG
        else:
            column[r] = random.choice(colors)
    return column


def _build_unit(height: int, period: int, colors: tuple[int, ...]) -> np.ndarray | None:
    left_count = (period + 1) // 2
    for _ in range(MAX_ATTEMPTS):
        columns = [_random_column(height, colors) for _ in range(left_count)]
        for color in colors:
            col = random.randrange(left_count)
            row = random.randrange(height)
            columns[col][row] = color

        unit = _compose_symmetric_unit(columns, period)
        if set(int(v) for v in np.unique(unit) if int(v) != BG) != set(colors):
            continue
        density = float(np.count_nonzero(unit != BG)) / float(unit.size)
        if not 0.2 <= density <= 0.8:
            continue
        if len({tuple(unit[r]) for r in range(height)}) < 3:
            continue
        if _periods_with_wildcard(unit, -1)[0] != period:
            continue
        if np.array_equal(unit, unit[:, :1]):
            continue
        return unit
    return None


def _mark_orbit(
    grid: np.ndarray,
    row: int,
    col_in_period: int,
    repeats: int,
    period: int,
    chosen_repeats: list[int],
) -> None:
    for repeat_idx in chosen_repeats:
        grid[row, repeat_idx * period + col_in_period] = MARKER


def _visible_count(grid: np.ndarray, row: int, col_in_period: int, repeats: int, period: int) -> int:
    return sum(int(grid[row, repeat_idx * period + col_in_period] != MARKER) for repeat_idx in range(repeats))


def _can_mark(
    grid: np.ndarray,
    row: int,
    col_in_period: int,
    repeats: int,
    period: int,
    chosen_repeats: list[int],
) -> bool:
    if not chosen_repeats:
        return False
    remaining = _visible_count(grid, row, col_in_period, repeats, period)
    newly_marked = sum(int(grid[row, repeat_idx * period + col_in_period] != MARKER) for repeat_idx in chosen_repeats)
    remaining -= newly_marked
    if remaining > 0:
        return True

    mirror = period - 1 - col_in_period
    if mirror == col_in_period:
        return False
    return _visible_count(grid, row, mirror, repeats, period) > 0


def _choose_key_pair(unit: np.ndarray) -> tuple[int, list[int]] | None:
    height, period = unit.shape
    candidates = []
    for c in range(period // 2):
        rows = [r for r in range(height) if int(unit[r, c]) != BG]
        if rows:
            candidates.append((c, rows))
    if not candidates:
        return None

    random.shuffle(candidates)
    for c, rows in candidates:
        random.shuffle(rows)
        count = random.randint(1, min(3, len(rows)))
        return c, rows[:count]
    return None


def _apply_markers(unit: np.ndarray, repeats: int) -> tuple[np.ndarray, dict] | None:
    height, period = unit.shape
    grid = np.tile(unit, (1, repeats))
    key = _choose_key_pair(unit)
    if key is None:
        return None
    c, rows = key
    mirror = period - 1 - c
    info_rows = []

    for row in rows:
        _mark_orbit(grid, row, c, repeats, period, list(range(repeats)))
        extra = random.sample(range(repeats), random.randint(1, repeats - 1))
        _mark_orbit(grid, row, mirror, repeats, period, extra)
        info_rows.append((row, tuple(sorted(extra))))

    extra_orbits = random.randint(height // 2, height + period)
    for _ in range(extra_orbits):
        row = random.randrange(height)
        col = random.randrange(period)
        visible = [repeat_idx for repeat_idx in range(repeats) if grid[row, repeat_idx * period + col] != MARKER]
        if len(visible) <= 1:
            continue
        mark_count = random.randint(1, min(2, len(visible) - 1))
        chosen = random.sample(visible, mark_count)
        if not _can_mark(grid, row, col, repeats, period, chosen):
            continue
        _mark_orbit(grid, row, col, repeats, period, chosen)

    if np.count_nonzero(grid == MARKER) == 0:
        return None
    return grid, {"key_col": c, "mirror_col": mirror, "rows": info_rows}


def _marker_candidates(grid: np.ndarray) -> list[int]:
    width = grid.shape[1]
    candidates = []
    for color in sorted(int(v) for v in np.unique(grid)):
        periods = _periods_with_wildcard(grid, color)
        if periods and periods[0] < width:
            candidates.append(color)
    return candidates


def _validate_instance(inp: np.ndarray, out: np.ndarray, period: int, meta: dict) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.any(inp == MARKER):
        return False
    if np.any(out == MARKER):
        return False

    periods = _periods_with_wildcard(inp, MARKER)
    if not periods or periods[0] != period:
        return False
    if _marker_candidates(inp) != [MARKER]:
        return False

    repeats = inp.shape[1] // period
    recovered = _recover_unit(inp, period, MARKER)
    expected = np.tile(recovered, (1, repeats))
    if not np.array_equal(expected, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    key_col = int(meta["key_col"])
    mirror_col = int(meta["mirror_col"])
    if key_col == mirror_col:
        return False

    needs_symmetry = False
    needs_periodicity = False
    for row, masked_mirror_repeats in meta["rows"]:
        if _visible_count(inp, row, key_col, repeats, period) != 0:
            return False
        if _visible_count(inp, row, mirror_col, repeats, period) == 0:
            return False
        needs_symmetry = True
        if masked_mirror_repeats:
            for repeat_idx in masked_mirror_repeats:
                if inp[row, repeat_idx * period + key_col] == MARKER and inp[row, repeat_idx * period + mirror_col] == MARKER:
                    needs_periodicity = True
                    break

    return needs_symmetry and needs_periodicity


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(tuple(HEIGHT_RANGE))
    period = random.choice(tuple(PERIOD_RANGE))
    max_repeats = min(max(REPEAT_RANGE), 30 // period)
    repeats = random.randint(min(REPEAT_RANGE), max_repeats)
    color_count = random.randint(3, 4)
    colors = tuple(random.sample(PALETTE, color_count))

    unit = _build_unit(height, period, colors)
    if unit is None:
        return None

    marked = _apply_markers(unit, repeats)
    if marked is None:
        return None
    inp, meta = marked
    out = np.tile(unit, (1, repeats))

    if not _validate_instance(inp, out, period, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-plus-symmetry ARC input")
