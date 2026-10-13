"""ARC task: consume a corner instruction tape while detouring around grey bars."""

from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5


def _instruction_queue(g: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    """Read a straight edge tape outward from its uniquely occupied corner."""
    h, w = g.shape
    cells = [tuple(map(int, p)) for p in np.argwhere((g == BLUE) | (g == GREEN))]
    assert len(cells) >= 2, "instruction tape must contain at least two pixels"
    cell_set = set(cells)
    candidates: list[list[tuple[int, int]]] = []
    for corner, step, limit in (
        ((0, 0), (0, 1), w), ((0, 0), (1, 0), h),
        ((0, w - 1), (0, -1), w), ((0, w - 1), (1, 0), h),
        ((h - 1, 0), (0, 1), w), ((h - 1, 0), (-1, 0), h),
        ((h - 1, w - 1), (0, -1), w), ((h - 1, w - 1), (-1, 0), h),
    ):
        ordered = [(corner[0] + k * step[0], corner[1] + k * step[1])
                   for k in range(len(cells))]
        if len(cells) <= limit and set(ordered) == cell_set:
            candidates.append(ordered)
    assert len(candidates) == 1, "tape must be one unambiguous corner-anchored edge run"
    ordered = candidates[0]
    return [int(g[p]) for p in ordered], ordered


def _grey_bars(g: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], int]]:
    """Return horizontal grey runs and a cell-to-run lookup."""
    bars: list[tuple[int, int, int]] = []
    lookup: dict[tuple[int, int], int] = {}
    for r in np.flatnonzero(np.any(g == GREY, axis=1)):
        cols = np.flatnonzero(g[int(r)] == GREY)
        assert len(cols) >= 2 and np.all(np.diff(cols) == 1), "each grey bar must be one horizontal run"
        idx = len(bars)
        bars.append((int(r), int(cols[0]), int(cols[-1])))
        for c in cols:
            lookup[(int(r), int(c))] = idx
    assert bars, "expected at least one grey obstacle bar"
    return bars, lookup


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a red inward trace whose successive grey-bar detours are controlled by a corner instruction tape.

    Input structure:
    - The dominant background contains one red marker on the top or bottom border.
    - Separate one-cell-thick horizontal grey bars cross the marker's possible lanes.
    - One corner-anchored edge tape is a contiguous sequence of blue and green pixels,
      with one instruction for every grey bar.

    Rule:
    - Travel vertically inward from the red marker.  On the k-th encountered bar,
      blue sends the trace around its left side relative to travel and green around
      its right side; paint only traversed background cells red, then continue in
      the shifted vertical lane.  Preserve every grey bar and stop at the opposite
      border.  Erase every pixel of the consumed instruction tape.

    Key insight:
    - The tape is an ordered queue: its k-th pixel controls the k-th encounter,
      regardless of either object's horizontal position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5
    h, w = g.shape
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    assert bg not in {BLUE, RED, GREEN, GREY}
    assert set(map(int, values)) <= {bg, BLUE, RED, GREEN, GREY}

    starts = np.argwhere(g == RED)
    assert starts.shape == (1, 2), "expected one red start marker"
    r, c = map(int, starts[0])
    assert (r == 0) ^ (r == h - 1), "start must lie on exactly one horizontal border"
    dr = 1 if r == 0 else -1

    instructions, tape_cells = _instruction_queue(g)
    bars, bar_at = _grey_bars(g)
    assert len(instructions) == len(bars), "tape length must equal obstacle count"

    out = g.copy()
    for rr, cc in tape_cells:
        out[rr, cc] = bg

    def paint(rr: int, cc: int) -> None:
        assert 0 <= rr < h and 0 <= cc < w
        assert int(g[rr, cc]) in {bg, RED}, "trace may only paint background"
        out[rr, cc] = RED

    def paint_line(a: tuple[int, int], b: tuple[int, int]) -> None:
        ar, ac = a
        br, bc = b
        assert ar == br or ac == bc
        sr = 0 if ar == br else (1 if br > ar else -1)
        sc = 0 if ac == bc else (1 if bc > ac else -1)
        for k in range(max(abs(br - ar), abs(bc - ac)) + 1):
            paint(ar + k * sr, ac + k * sc)

    seen: set[int] = set()
    paint(r, c)
    for instruction in instructions:
        rr = r + dr
        hit: tuple[int, int] | None = None
        while 0 <= rr < h:
            if int(g[rr, c]) == GREY:
                hit = (rr, c)
                break
            assert int(g[rr, c]) in {bg, RED}, "unexpected object in the forward lane"
            rr += dr
        assert hit is not None, "every instruction must meet a new grey bar"
        hr, hc = hit
        idx = bar_at[hit]
        assert idx not in seen, "a grey bar may be encountered only once"
        seen.add(idx)
        _bar_r, c0, c1 = bars[idx]

        # Facing down, geometric left is east; facing up it is west.
        side_sign = dr if instruction == BLUE else -dr
        side_c = c1 + 1 if side_sign > 0 else c0 - 1
        assert 0 <= side_c < w
        before = (hr - dr, c)
        if (r, c) != before:
            paint_line((r + dr, c), before)
        beside_before = (hr - dr, side_c)
        paint_line(before, beside_before)
        paint(hr, side_c)
        r, c = hr, side_c

    assert len(seen) == len(bars), "all and only the grey bars must be encountered"
    last_r = h - 1 if dr > 0 else 0
    if r != last_r:
        paint_line((r + dr, c), (last_r, c))
    return out


def _make_instructions(n: int) -> list[int]:
    """Balanced enough to expose both branches, with random order and repeats."""
    seq = [BLUE, GREEN]
    seq.extend(random.choice((BLUE, GREEN)) for _ in range(n - 2))
    random.shuffle(seq)
    return seq


def _attempt_generate() -> np.ndarray | None:
    n = random.randint(4, 6)
    h = random.randint(max(15, 3 * n + 3), min(25, 4 * n + 7))
    w = random.randint(14, 24)
    dr = random.choice((-1, 1))
    instructions = _make_instructions(n)

    # Bar rows are ordered in the actual travel frame, never by absolute row.
    gaps = [random.randint(2, 4) for _ in range(n - 1)]
    first = random.randint(3, 4)
    levels = [first]
    for gap in gaps:
        levels.append(levels[-1] + gap)
    if levels[-1] > h - 3:
        return None
    rows = levels if dr > 0 else [h - 1 - x for x in levels]

    grid = np.full((h, w), BG, dtype=int)
    lane = random.randint(4, w - 5)
    grid[0 if dr > 0 else h - 1, lane] = RED
    bar_specs: list[tuple[int, int, int]] = []
    for row, instruction in zip(rows, instructions):
        sign = dr if instruction == BLUE else -dr
        options: list[tuple[int, int]] = []
        for length in range(2, 5):
            for left_part in range(length):
                c0 = lane - left_part
                c1 = c0 + length - 1
                target = c1 + 1 if sign > 0 else c0 - 1
                if 2 <= c0 and c1 <= w - 3 and 2 <= target <= w - 3:
                    options.append((c0, c1))
        if not options:
            return None
        c0, c1 = random.choice(options)
        grid[row, c0:c1 + 1] = GREY
        bar_specs.append((row, c0, c1))
        lane = c1 + 1 if sign > 0 else c0 - 1

    # Put the queue on a random corner/edge orientation.  The start lane is kept
    # away from corners, so the two roles never overlap.
    corner = random.choice((0, 1, 2, 3))  # TL, TR, BL, BR
    horizontal = random.choice((False, True))
    if corner == 0:
        cells = [(0, k) if horizontal else (k, 0) for k in range(n)]
    elif corner == 1:
        cells = [(0, w - 1 - k) if horizontal else (k, w - 1) for k in range(n)]
    elif corner == 2:
        cells = [(h - 1, k) if horizontal else (h - 1 - k, 0) for k in range(n)]
    else:
        cells = [(h - 1, w - 1 - k) if horizontal else (h - 1 - k, w - 1) for k in range(n)]
    if any(grid[p] != BG for p in cells):
        return None
    for p, color in zip(cells, instructions):
        grid[p] = color

    # Semantic checks: transformation succeeds, consumes the whole queue, keeps
    # every obstacle, and actually draws beyond merely erasing the tape.
    try:
        out = main(grid)
    except AssertionError:
        return None
    if any(out[p] != BG for p in cells):
        return None
    if any(not np.all(out[r, c0:c1 + 1] == GREY) for r, c0, c1 in bar_specs):
        return None
    if np.count_nonzero((out == RED) & (grid == BG)) < h // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, nondegenerate instances covering both detour directions."""
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instruction-tape trace puzzle")
