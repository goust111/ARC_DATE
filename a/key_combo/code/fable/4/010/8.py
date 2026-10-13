from __future__ import annotations

import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _candidate_completions(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Enumerate fully observed oblique-period models in the grid's own coordinates."""
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    known = g != BLACK
    rr, cc = np.nonzero(known)
    values = g[rr, cc]

    candidates: list[tuple[int, int, np.ndarray]] = []
    # A repeating row must display at least two copies of its base sequence.
    for period in range(2, width // 2 + 1):
        for shift in range(1, period):
            phase = (cc - rr * shift) % period
            base = np.full(period, -1, dtype=int)
            valid = True
            for index, value in zip(phase, values):
                index = int(index)
                value = int(value)
                if base[index] == -1:
                    base[index] = value
                elif base[index] != value:
                    valid = False
                    break
            if not valid or np.any(base == -1):
                continue

            rows = np.arange(height, dtype=int)[:, None]
            cols = np.arange(width, dtype=int)[None, :]
            out = base[(cols - rows * shift) % period]
            candidates.append((period, shift, out))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Fill black holes by continuing the uniquely determined row-shifted wallpaper.

    Input structure:
    - The grid is a colored wallpaper with black missing cells; black is not a
      wallpaper color.
    - Every row repeats one base sequence, and row r is the base sequence cyclically
      shifted right by r times one fixed nonzero shift.
    - The visible cells expose every base phase and uniquely determine the smallest
      row period, its row-to-row shift, and all missing colors.

    Rule:
    - Ignore black cells and test row periods that repeat at least twice across the
      width together with every possible nonzero right shift for that period.
    - A visible cell at (r, c) constrains base phase (c - r * shift) modulo the
      period. Reconstruct the wallpaper from the compatible phase colors and fill
      every black cell, requiring all compatible models to give the same result.

    Key insight:
    - The repetition lives on an oblique lattice: compensate for the accumulated
      cyclic row shift before matching horizontal phases.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid must be at most 30x30"
    assert np.any(g == BLACK), "expected black missing cells"
    assert np.any(g != BLACK), "expected visible wallpaper cells"

    candidates = _candidate_completions(g)
    assert candidates, "visible cells do not determine an oblique repeating wallpaper"

    distinct_outputs: dict[bytes, np.ndarray] = {}
    for _period, _shift, completion in candidates:
        distinct_outputs.setdefault(completion.tobytes(), completion)
    assert len(distinct_outputs) == 1, "visible clues allow different completed wallpapers"

    smallest_period = min(period for period, _shift, _out in candidates)
    canonical = [item for item in candidates if item[0] == smallest_period]
    assert len(canonical) == 1, "smallest period does not have a unique row shift"

    out = next(iter(distinct_outputs.values())).copy()
    assert np.array_equal(out[g != BLACK], g[g != BLACK]), "completion changed a visible cell"
    assert np.all(out != BLACK), "wallpaper colors must be non-black"
    return out


def _full_models(full: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Return models for a complete wallpaper (used only as a generator oracle)."""
    return _candidate_completions(full)


def _flat_row_model_exists(grid: np.ndarray) -> bool:
    """Whether visible clues can be explained without any row-to-row shift."""
    g = np.asarray(grid, dtype=int)
    width = int(g.shape[1])
    for period in range(1, width // 2 + 1):
        base = np.full(period, -1, dtype=int)
        valid = True
        for r, c in np.argwhere(g != BLACK):
            phase = int(c) % period
            value = int(g[r, c])
            if base[phase] == -1:
                base[phase] = value
            elif base[phase] != value:
                valid = False
                break
        if valid and np.all(base != -1):
            return True
    return False


def _make_base(period: int) -> np.ndarray | None:
    color_count = random.randint(3, min(5, period))
    colors = random.sample(ARC_COLORS, color_count)
    for _ in range(100):
        values = colors + [random.choice(colors) for _ in range(period - color_count)]
        random.shuffle(values)
        base = np.asarray(values, dtype=int)
        # Reject any shorter cyclic period; equality of colors is the only signal.
        if all(
            not np.array_equal(base, np.resize(base[:small], period))
            for small in range(1, period)
        ):
            return base
    return None


def _render_wallpaper(height: int, width: int, base: np.ndarray, shift: int) -> np.ndarray:
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    return base[(cols - rows * shift) % len(base)]


def _add_blob(mask: np.ndarray) -> None:
    """Add a small connected irregular or rectangular black patch."""
    height, width = mask.shape
    target_size = random.randint(2, 7)
    r = random.randrange(height)
    c = random.randrange(width)
    cells = {(r, c)}
    frontier = [(r, c)]
    while frontier and len(cells) < target_size:
        cr, cc = random.choice(frontier)
        options = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier.remove((cr, cc))
            continue
        cell = random.choice(options)
        cells.add(cell)
        frontier.append(cell)
    for cell in cells:
        mask[cell] = True


def _has_repeat_witness(mask: np.ndarray, period: int) -> bool:
    """A hole has a visible same-row copy exactly one period away."""
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c]:
                continue
            if c >= period and not mask[r, c - period]:
                return True
            if c + period < width and not mask[r, c + period]:
                return True
    return False


def _has_shift_witness(mask: np.ndarray, full: np.ndarray, period: int, shift: int) -> bool:
    """A row phase hidden in one row is recoverable only after oblique alignment."""
    height, width = mask.shape
    for r in range(height):
        row_phases = (np.arange(width) - r * shift) % period
        for phase in range(period):
            cells = np.flatnonzero(row_phases == phase)
            if cells.size < 2 or not bool(np.all(mask[r, cells])):
                continue
            visible_elsewhere = False
            for other_r in range(height):
                if other_r == r:
                    continue
                other_phases = (np.arange(width) - other_r * shift) % period
                if np.any((other_phases == phase) & ~mask[other_r]):
                    visible_elsewhere = True
                    break
            if not visible_elsewhere:
                continue
            # The same unshifted columns must not accidentally give the right color.
            if any(int(full[r, c]) != int(full[0, c]) for c in cells):
                return True
    return False


def _sample_mask(full: np.ndarray, period: int, shift: int) -> np.ndarray:
    height, width = full.shape
    mask = np.zeros(full.shape, dtype=bool)

    # Force a decisive anti-shortcut case: hide every occurrence of one phase in
    # one interior row, so that row cannot repair itself independently.
    witness_row = random.randrange(1, height)
    phases = (np.arange(width) - witness_row * shift) % period
    phase_choices = list(range(period))
    random.shuffle(phase_choices)
    for phase in phase_choices:
        cells = np.flatnonzero(phases == phase)
        if cells.size >= 2 and any(int(full[witness_row, c]) != int(full[0, c]) for c in cells):
            mask[witness_row, cells] = True
            break

    # Force a separate horizontal-period witness while leaving its repeated copy visible.
    repeat_row = random.randrange(height)
    repeat_col = random.randrange(width - period)
    mask[repeat_row, repeat_col] = True
    mask[repeat_row, repeat_col + period] = False

    target = random.randint(max(10, full.size // 8), max(11, full.size // 4))
    for _ in range(random.randint(2, 5)):
        if int(mask.sum()) >= target:
            break
        _add_blob(mask)
    while int(mask.sum()) < target:
        mask[random.randrange(height), random.randrange(width)] = True
    return mask


def _valid_generated(
    inp: np.ndarray, full: np.ndarray, period: int, shift: int, mask: np.ndarray
) -> bool:
    height, width = inp.shape
    missing = int(mask.sum())
    if not (10 <= missing <= int(0.32 * height * width)):
        return False
    if np.any(np.sum(~mask, axis=1) < period):
        return False

    # Every oblique phase remains visibly represented somewhere.
    rows, cols = np.nonzero(~mask)
    if len(set(map(int, (cols - rows * shift) % period))) != period:
        return False
    if not _has_repeat_witness(mask, period):
        return False
    if not _has_shift_witness(mask, full, period, shift):
        return False
    if _flat_row_model_exists(inp):
        return False

    candidates = _candidate_completions(inp)
    if not candidates:
        return False
    smallest = min(item[0] for item in candidates)
    canonical = [(p, s) for p, s, _out in candidates if p == smallest]
    if canonical != [(period, shift)]:
        return False
    if len({out.tobytes() for _p, _s, out in candidates}) != 1:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, full))


def _generation_attempt() -> np.ndarray | None:
    period = random.randint(3, 8)
    width = random.randint(2 * period + 2, min(30, 4 * period + 4))
    height = random.randint(6, 16)
    shift = random.randint(1, period - 1)
    base = _make_base(period)
    if base is None:
        return None

    full = _render_wallpaper(height, width, base, shift)
    full_models = _full_models(full)
    smallest = min((p for p, _s, _out in full_models), default=-1)
    if [(p, s) for p, s, _out in full_models if p == smallest] != [(period, shift)]:
        return None

    mask = _sample_mask(full, period, shift)
    inp = full.copy()
    inp[mask] = BLACK
    if _valid_generated(inp, full, period, shift, mask):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely solvable row-shifted wallpaper with black holes."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-shifted wallpaper")
