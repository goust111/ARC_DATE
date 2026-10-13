import numpy as np


BG = 0
BLUE = 1
RED = 2
WIDTH = 10
FULL_HEIGHT = 10
_RNG = np.random.default_rng()


def _shift_row_right(row: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(row)
    if shift < row.size:
        out[shift:] = row[:-shift]
    return out


def _blue_shift_matches(grid: np.ndarray, period: int, shift: int) -> bool:
    blue = grid == BLUE
    height, width = grid.shape
    for r in range(height - period):
        expected = np.zeros(width, dtype=bool)
        if shift < width:
            expected[shift:] = blue[r, :-shift]
        if not np.array_equal(expected, blue[r + period]):
            return False
    return True


def _valid_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    assert width == WIDTH
    candidates: list[tuple[int, int]] = []
    for period in (2, 3):
        if height <= period:
            continue
        for shift in (1, 2):
            if _blue_shift_matches(grid, period, shift):
                candidates.append((period, shift))
    return candidates


def _infer_period_shift(grid: np.ndarray) -> tuple[int, int]:
    candidates = _valid_candidates(grid)
    assert candidates, "expected at least one valid (period, shift) pair"
    best_period = min(period for period, _ in candidates)
    best = [candidate for candidate in candidates if candidate[0] == best_period]
    assert len(best) == 1, "smallest period must determine a unique horizontal shift"
    return best[0]


def _extend_canvas(grid: np.ndarray, period: int, shift: int) -> np.ndarray:
    height, width = grid.shape
    assert width == WIDTH
    out = np.zeros((FULL_HEIGHT, WIDTH), dtype=int)
    out[:height] = grid
    for r in range(height, FULL_HEIGHT):
        out[r] = _shift_row_right(out[r - period], shift)
    return out


def _invert_marked_rows(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for r in range(out.shape[0]):
        if not np.any(out[r] == RED):
            continue
        blue_mask = out[r] == BLUE
        black_mask = out[r] == BG
        out[r, blue_mask] = BG
        out[r, black_mask] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the periodic row pattern to 10x10, then invert only rows carrying propagated red markers.

    Input structure:
    - The grid is 4 to 6 rows high and exactly 10 columns wide.
    - Background cells are black (0), foreground pattern cells are blue (1), and some rows contain one red marker (2).
    - Blue rows follow an unknown vertical period of 2 or 3 rows; each repeat is shifted right by 1 or 2 cells with truncation at the right edge.

    Rule:
    - Infer the unique period and horizontal shift from the blue pixels alone, using the smallest valid period.
    - Copy the input onto the top of a 10x10 black canvas.
    - For each missing row, shift the entire row from `period` rows above to the right by the inferred amount; blue and red both propagate, and overflow is discarded.
    - After the canvas is filled, invert each row that contains a red marker: blue becomes black, black becomes blue, and the red marker stays red.

    Key insight:
    - The red dot is not static noise; it rides the same periodic row extension as the blue pattern, and its propagated presence is the trigger for row-wise inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] == WIDTH
    assert 4 <= g.shape[0] <= 6
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, RED})

    period, shift = _infer_period_shift(g)
    extended = _extend_canvas(g, period, shift)
    return _invert_marked_rows(extended)


def _sample_height(period: int) -> int:
    if period == 2:
        return int(_RNG.integers(4, 7))
    return int(_RNG.integers(5, 7))


def _sample_base_rows(period: int, shift: int, height: int) -> list[np.ndarray]:
    durable_row = int(_RNG.integers(0, period))
    rows: list[np.ndarray] = []
    for base_r in range(period):
        steps_in_input = (height - 1 - base_r) // period
        safe_input_col = 9 - shift * steps_in_input
        cols: set[int] = set()

        if base_r == durable_row:
            steps_to_bottom = (FULL_HEIGHT - 1 - base_r) // period
            safe_bottom_col = 9 - shift * steps_to_bottom
            cols.add(int(_RNG.integers(0, safe_bottom_col + 1)))

        cols.add(int(_RNG.integers(0, safe_input_col + 1)))
        target_count = int(_RNG.integers(2, 5))
        remaining = [c for c in range(WIDTH) if c not in cols]
        if remaining and len(cols) < target_count:
            extra = _RNG.choice(remaining, size=min(target_count - len(cols), len(remaining)), replace=False)
            cols.update(int(c) for c in np.atleast_1d(extra))

        row = np.zeros(WIDTH, dtype=int)
        row[list(cols)] = BLUE
        rows.append(row)
    return rows


def _marker_future_steps(base_r: int, period: int, height: int) -> list[int]:
    steps = []
    k = 1
    while base_r + k * period < FULL_HEIGHT:
        if base_r + k * period >= height:
            steps.append(k)
        k += 1
    return steps


def _eligible_marker_cols(base_row: np.ndarray, base_r: int, period: int, shift: int, height: int) -> list[int]:
    future_steps = _marker_future_steps(base_r, period, height)
    if not future_steps:
        return []
    usable: list[int] = []
    for c in np.where(base_row == BG)[0]:
        if any(int(c) + shift * k < WIDTH for k in future_steps):
            usable.append(int(c))
    return usable


def _choose_base_markers(base_rows: list[np.ndarray], period: int, shift: int, height: int) -> dict[int, int] | None:
    markable = [r for r in range(period) if np.any(base_rows[r] == BG)]
    if len(markable) < 1:
        return None

    propagating = [r for r in markable if _eligible_marker_cols(base_rows[r], r, period, shift, height)]
    if not propagating:
        return None

    marked_count = int(_RNG.integers(1, min(len(markable), period - 1) + 1))
    chosen = {int(_RNG.choice(propagating))}
    others = [r for r in markable if r not in chosen]
    if marked_count > 1:
        extra = _RNG.choice(others, size=marked_count - 1, replace=False)
        chosen.update(int(r) for r in np.atleast_1d(extra))

    markers: dict[int, int] = {}
    for r in chosen:
        if r in propagating:
            choices = _eligible_marker_cols(base_rows[r], r, period, shift, height)
        else:
            choices = [int(c) for c in np.where(base_rows[r] == BG)[0]]
        markers[r] = int(_RNG.choice(choices))
    return markers


def _build_raw_canvas(base_rows: list[np.ndarray], markers: dict[int, int], period: int, shift: int) -> np.ndarray:
    raw = np.zeros((FULL_HEIGHT, WIDTH), dtype=int)
    for r, row in enumerate(base_rows):
        raw[r] = row.copy()
        if r in markers:
            raw[r, markers[r]] = RED
    for r in range(period, FULL_HEIGHT):
        raw[r] = _shift_row_right(raw[r - period], shift)
    return raw


def _validate_case(inp: np.ndarray, raw: np.ndarray, out: np.ndarray, period: int, shift: int) -> bool:
    candidates = _valid_candidates(inp)
    if candidates != [(period, shift)]:
        return False
    if not np.any(inp == RED):
        return False
    if not all(np.any(inp[r] == BLUE) for r in range(inp.shape[0])):
        return False
    if not any(np.any(raw[r] == RED) for r in range(inp.shape[0], FULL_HEIGHT)):
        return False
    if not any(np.any(raw[r] == BLUE) and not np.any(raw[r] == RED) for r in range(inp.shape[0], FULL_HEIGHT)):
        return False
    if not any(np.any(raw[r] == RED) and not np.array_equal(raw[r], out[r]) for r in range(FULL_HEIGHT)):
        return False
    if not any((not np.any(raw[r] == RED)) and np.array_equal(raw[r], out[r]) for r in range(FULL_HEIGHT)):
        return False
    if np.array_equal(raw, out):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    period = int(_RNG.choice([2, 3]))
    shift = int(_RNG.choice([1, 2]))
    height = _sample_height(period)
    base_rows = _sample_base_rows(period, shift, height)
    markers = _choose_base_markers(base_rows, period, shift, height)
    if markers is None:
        return None

    raw = _build_raw_canvas(base_rows, markers, period, shift)
    inp = raw[:height].copy()
    out = _invert_marked_rows(raw)

    if not _validate_case(inp, raw, out, period, shift):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(1000):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid periodic-marker inversion case")
