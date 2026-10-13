from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

# --------------- generator (custom) ---------------

MIN_ROWS = 5
MAX_ROWS = 10
MIN_COLS = 5
MAX_COLS = 12
MIN_AREA = 30

MIN_PERIOD = 3
MAX_PERIOD = 8

MIN_MISSING = 5
MAX_MISSING = 18

MIN_VISIBLE_RUNS = 3
MIN_WITNESS_TURNS = 2
MAX_GENERATION_ATTEMPTS = 300


def _spiral_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = h - 1, w - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1

        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1

    return coords


def _ccw_spiral_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = h - 1, w - 1

    while top <= bottom and left <= right:
        for r in range(top, bottom + 1):
            coords.append((r, left))
        left += 1

        for c in range(left, right + 1):
            coords.append((bottom, c))
        bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, right))
            right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((top, c))
            top += 1

    return coords


def _row_major_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _col_major_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for c in range(w) for r in range(h)]


def _values_on_path(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[int]:
    return [int(grid[r, c]) for r, c in coords]


def _visible_runs(values: list[int]) -> list[tuple[int, list[int]]]:
    runs: list[tuple[int, list[int]]] = []
    start: int | None = None

    for idx, value in enumerate(values):
        if value != BLACK and start is None:
            start = idx
        elif value == BLACK and start is not None:
            runs.append((start, list(values[start:idx])))
            start = None

    if start is not None:
        runs.append((start, list(values[start:])))
    return runs


def _run_min_period(run: list[int]) -> int:
    assert run, "visible run must be nonempty"
    for period in range(1, len(run) + 1):
        if all(run[idx] == run[idx - period] for idx in range(period, len(run))):
            return period
    raise AssertionError("every finite run should have a period")


def _is_primitive(pattern: list[int]) -> bool:
    return all(
        any(pattern[idx] != pattern[idx - period] for idx in range(period, len(pattern)))
        for period in range(1, len(pattern))
    )


def _infer_period_and_colors(values: list[int]) -> tuple[int, list[int]]:
    runs = _visible_runs(values)
    assert runs, "expected at least one visible segment"

    candidates = sorted({_run_min_period(run) for _start, run in runs})
    for period in candidates:
        colors: list[int | None] = [None] * period
        ok = True

        for idx, value in enumerate(values):
            if value == BLACK:
                continue
            slot = idx % period
            if colors[slot] is None:
                colors[slot] = value
            elif colors[slot] != value:
                ok = False
                break

        if ok and all(color is not None for color in colors):
            return period, [int(color) for color in colors]

    raise AssertionError("could not infer a unique shortest period from visible spiral cells")


def _complete_by_coords(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    values = _values_on_path(grid, coords)
    period, colors = _infer_period_and_colors(values)

    out = np.asarray(grid, dtype=int).copy()
    for idx, (r, c) in enumerate(coords):
        expected = colors[idx % period]
        if int(out[r, c]) == BLACK:
            out[r, c] = expected
        else:
            assert int(out[r, c]) == expected, "visible cells must already match the inferred cycle"

    return out, period


def _ring_depth(shape: tuple[int, int], coord: tuple[int, int]) -> int:
    h, w = shape
    r, c = coord
    return min(r, c, h - 1 - r, w - 1 - c)


def _turn_count(coords: list[tuple[int, int]]) -> int:
    if len(coords) < 3:
        return 0
    dirs = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(coords, coords[1:])]
    return sum(prev != curr for prev, curr in zip(dirs, dirs[1:]))


def _sample_dimensions() -> tuple[int, int]:
    for _ in range(100):
        mode = random.choice(("square", "wide", "tall"))
        if mode == "square":
            side = random.randint(6, 10)
            h, w = side, side
        elif mode == "wide":
            h = random.randint(5, 9)
            w = random.randint(max(h + 1, 7), 12)
        else:
            w = random.randint(5, 9)
            h = random.randint(max(w + 1, 7), 10)
        if h * w >= MIN_AREA:
            return h, w
    raise AssertionError("failed to sample a valid canvas")


def _sample_period(period_len: int) -> list[int]:
    if random.random() < 0.45:
        return random.sample(PALETTE, period_len)

    for _ in range(500):
        palette = random.sample(PALETTE, random.randint(2, min(5, period_len)))
        pattern = [random.choice(palette) for _ in range(period_len)]
        if len(set(pattern)) >= 2 and _is_primitive(pattern):
            return pattern

    raise AssertionError("failed to sample a primitive color cycle")


def _build_output(shape: tuple[int, int], pattern: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    coords = _spiral_coords(*shape)
    sequence = [pattern[idx % len(pattern)] for idx in range(len(coords))]
    out = np.empty(shape, dtype=int)
    for (r, c), value in zip(coords, sequence):
        out[r, c] = value
    return out, coords, sequence


def _sample_visible_mask(
    shape: tuple[int, int],
    coords: list[tuple[int, int]],
    sequence: list[int],
    period_len: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    n = len(coords)
    max_witness = min(n - MIN_MISSING, max(2 * period_len, 3 * period_len + 6))
    if max_witness < 2 * period_len:
        return None

    for _ in range(120):
        witness_len = random.randint(2 * period_len, max_witness)
        witness_start = random.randint(0, n - witness_len)
        witness_slice = sequence[witness_start : witness_start + witness_len]
        if _run_min_period(witness_slice) != period_len:
            continue
        if _turn_count(coords[witness_start : witness_start + witness_len]) < MIN_WITNESS_TURNS:
            continue

        target_lo = max(MIN_MISSING, period_len // 2 + 1)
        target_hi = min(MAX_MISSING, n - witness_len)
        if target_hi < target_lo:
            continue

        visible = np.ones(n, dtype=bool)
        protected = np.zeros(n, dtype=bool)
        protected[witness_start : witness_start + witness_len] = True
        target_missing = random.randint(target_lo, target_hi)

        for _ in range(2500):
            if int((~visible).sum()) >= target_missing:
                break
            span = random.choice((1, 1, 2, 2, 3, 4))
            start = random.randint(0, n - span)
            if protected[start : start + span].any():
                continue
            if (~visible[start : start + span]).any():
                continue
            visible[start : start + span] = False

        missing = np.flatnonzero(~visible)
        if missing.size < target_missing:
            continue
        if len(_visible_runs([1 if keep else 0 for keep in visible.tolist()])) < MIN_VISIBLE_RUNS:
            continue
        if len({int(idx % period_len) for idx in missing}) < min(3, period_len):
            continue
        if any(not np.any(visible[np.arange(slot, n, period_len)]) for slot in range(period_len)):
            continue
        if len({_ring_depth(shape, coords[int(idx)]) for idx in missing}) < 2:
            continue

        return visible, (witness_start, witness_len)

    return None


def _validate_sample(
    inp: np.ndarray,
    out: np.ndarray,
    period_len: int,
    coords: list[tuple[int, int]],
    witness: tuple[int, int],
) -> bool:
    recovered, inferred = _complete_by_coords(inp, coords)
    if inferred != period_len or not np.array_equal(recovered, out):
        return False

    start, length = witness
    witness_values = _values_on_path(inp, coords)[start : start + length]
    if BLACK in witness_values:
        return False
    if _run_min_period(witness_values) != period_len:
        return False

    h, w = inp.shape
    for alt_coords in (_row_major_coords(h, w), _col_major_coords(h, w), _ccw_spiral_coords(h, w)):
        try:
            alt_out, _alt_period = _complete_by_coords(inp, alt_coords)
        except Exception:
            continue
        if np.array_equal(alt_out, out):
            return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black holes by reading the colored cells along the hidden clockwise spiral and continuing its shortest repeating cycle.

    Input structure:
    - A rectangular grid is almost fully colored, with black cells used only as missing entries.
    - Reading cells in the clockwise spiral that starts at the top-left corner and peels inward gives a color sequence with a strict shortest period.
    - At least one visible contiguous spiral segment is complete enough to reveal that shortest period.

    Rule:
    - Enumerate the grid cells in the top-left-starting clockwise spiral order.
    - From the non-black visible runs, infer the shortest period that is globally consistent with every visible spiral position.
    - Fill each black cell with the color required by that periodic spiral sequence, leaving non-black cells unchanged.

    Key insight:
    - The apparent local row or column rhythm is a distraction; the true one-dimensional carrier is the unseen inward spiral path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords = _spiral_coords(*g.shape)
    out, _period = _complete_by_coords(g, coords)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = _sample_dimensions()
        area = shape[0] * shape[1]
        max_period = min(MAX_PERIOD, (area - MIN_MISSING) // 2)
        if max_period < MIN_PERIOD:
            continue

        period_len = random.randint(MIN_PERIOD, max_period)
        pattern = _sample_period(period_len)
        out, coords, sequence = _build_output(shape, pattern)
        sampled = _sample_visible_mask(shape, coords, sequence, period_len)
        if sampled is None:
            continue

        visible, witness = sampled
        inp = out.copy()
        for idx, keep in enumerate(visible):
            if not keep:
                r, c = coords[idx]
                inp[r, c] = BLACK

        if _validate_sample(inp, out, period_len, coords, witness):
            return inp

    raise RuntimeError("failed to generate a valid spiral-period completion puzzle")
