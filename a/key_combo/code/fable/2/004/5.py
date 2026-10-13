from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import math
import random

import numpy as np


_ARC = SourceFileLoader(
    "_shared_arc_utils_fable_2_004_5",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
get_bg = _ARC.get_bg


GRID_LIMIT = 30
BG = int(Color.BLACK)
MARKER = int(Color.GREY)
PALETTE = (
    int(Color.BLUE),
    int(Color.RED),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
)
HEIGHT_RANGE = (2, 5)
WIDTH_RANGE = (8, 15)
GEN_RETRIES = 400


def _roll(block: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(block, int(shift) % int(block.shape[1]), axis=1)


def _marker_height(grid: np.ndarray) -> int:
    h = 0
    while h < grid.shape[0] and int(grid[h, 0]) == MARKER:
        h += 1
    return int(h)


def _candidate_steps(motif: np.ndarray, copy1: np.ndarray, copy2: np.ndarray) -> list[int]:
    width = int(motif.shape[1])
    candidates = []
    for step in range(1, width):
        if not np.array_equal(copy1, _roll(motif, step)):
            continue
        if not np.array_equal(copy2, _roll(copy1, 2 * step)):
            continue
        candidates.append(int(step))
    return candidates


def _paint_full_output(motif: np.ndarray, total_rows: int, step: int, bg: int) -> np.ndarray:
    h, w = map(int, motif.shape)
    out = np.full((int(total_rows), w), int(bg), dtype=int)
    out[:h] = motif

    prev = motif
    row = h
    block_index = 1
    while row < total_rows:
        prev = _roll(prev, block_index * step)
        take = min(h, total_rows - row)
        out[row : row + take] = prev[:take]
        row += h
        block_index += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Continue the wrapped horizontal drifts of the top motif to fill the blank suffix.

    Input structure
    - The background is black, and the leftmost column starts with one contiguous grey run whose length is the motif height h.
    - The top h rows contain the motif, except that the grey run temporarily replaces the motif's first-column cells and should be treated as a marker.
    - Below the motif are exactly two visible h-row copies: the first is a row-wise right cyclic shift by s, and the second is the first copy shifted right by 2s.

    Rule
    - Erase the grey marker cells back to background to recover the true motif.
    - Infer the unique base step s that makes both visible copies consistent with wrapped right shifts.
    - Keep extending downward: the next copy is the previous copy shifted right by 3s, then by 4s, and so on, truncating the final copy at the bottom edge if needed.

    Key insight
    - The drift is not constant: the first transition alone suggests a fixed shift, but the second visible copy forces an arithmetic growth of the per-copy wrapped shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    motif_h = _marker_height(g)
    assert 0 < motif_h < h, "expected a non-empty top marker run with suffix rows below it"
    assert h >= 3 * motif_h, "expected the motif plus two visible copies"

    motif = g[:motif_h].copy()
    motif[:, 0] = bg
    copy1 = g[motif_h : 2 * motif_h]
    copy2 = g[2 * motif_h : 3 * motif_h]

    steps = _candidate_steps(motif, copy1, copy2)
    assert len(steps) == 1, "expected a unique wrapped drift step"
    return _paint_full_output(motif, h, steps[0], bg)


def _nonperiodic(motif: np.ndarray) -> bool:
    return all(not np.array_equal(motif, _roll(motif, shift)) for shift in range(1, motif.shape[1]))


def _random_row(width: int, palette: tuple[int, ...]) -> np.ndarray:
    for _ in range(80):
        row = np.full(int(width), BG, dtype=int)
        pos = 1 + random.randint(0, 1)
        run_count = 0
        while pos < width and run_count < 4:
            run_len = random.randint(1, min(3, width - pos))
            row[pos : pos + run_len] = random.choice(palette)
            run_count += 1
            pos += run_len + random.randint(1, 2)
            if run_count >= 2 and random.random() < 0.35:
                break
        filled = int(np.count_nonzero(row))
        if 3 <= filled <= max(4, width - 3):
            return row
    raise RuntimeError("failed to create a motif row")


def _random_motif(height: int, width: int, step: int) -> np.ndarray | None:
    for _ in range(120):
        palette = tuple(random.sample(PALETTE, random.randint(2, 4)))
        motif = np.vstack([_random_row(width, palette) for _ in range(height)]).astype(int)
        motif[:, 0] = BG

        for _ in range(random.randint(1, max(2, height))):
            col = random.randint(1, width - 1)
            r0 = random.randint(0, height - 1)
            r1 = random.randint(r0, min(height - 1, r0 + random.randint(1, 2)))
            color = int(motif[r0, col]) if int(motif[r0, col]) != BG and random.random() < 0.6 else random.choice(palette)
            motif[r0 : r1 + 1, col] = color

        if any(int(np.count_nonzero(motif[r])) < 2 for r in range(height)):
            continue

        if not np.any(motif[:, width - step :] != BG):
            r = random.randrange(height)
            span = random.randint(1, min(2, step))
            start = random.randint(max(1, width - step), width - span)
            motif[r, start : start + span] = random.choice(palette)

        if not _nonperiodic(motif):
            continue
        return motif
    return None


def _choose_extra_rows(motif_h: int) -> int:
    extra_max = min(GRID_LIMIT - 3 * motif_h, 3 * motif_h)
    assert extra_max >= 1
    full_choices = [k * motif_h for k in range(1, extra_max // motif_h + 1)]
    partial_choices = [n for n in range(1, extra_max + 1) if n % motif_h != 0]
    if partial_choices and random.random() < 0.6:
        return int(random.choice(partial_choices))
    return int(random.choice(full_choices))


def _gen_attempt() -> np.ndarray | None:
    motif_h = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    step_choices = [s for s in range(1, width) if math.gcd(s, width) == 1]
    step = random.choice(step_choices)

    motif = _random_motif(motif_h, width, step)
    if motif is None:
        return None

    total_rows = 3 * motif_h + _choose_extra_rows(motif_h)
    expected = _paint_full_output(motif, total_rows, step, BG)

    inp = expected.copy()
    inp[3 * motif_h :] = BG
    inp[:motif_h, 0] = MARKER

    recovered = inp[:motif_h].copy()
    recovered[:, 0] = BG
    copy1 = inp[motif_h : 2 * motif_h]
    copy2 = inp[2 * motif_h : 3 * motif_h]

    steps = _candidate_steps(recovered, copy1, copy2)
    if steps != [step]:
        return None
    if np.array_equal(copy2, _roll(copy1, step)):
        return None
    if not np.any(recovered[:, width - step :] != BG):
        return None
    if not np.any(expected[3 * motif_h :] != BG):
        return None
    if not np.all(inp[3 * motif_h :] == BG):
        return None
    counts = Counter(map(int, inp.flat))
    if counts[BG] <= max((count for color, count in counts.items() if color != BG), default=0):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid accelerating-drift input")
