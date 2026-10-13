import itertools
import random

import numpy as np


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(x) for x in np.unique(grid)]


def _candidate_outputs(grid: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for pair in itertools.combinations(_colors_present(g), 2):
        mask = np.isin(g, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        rh = int(rows.sum())
        cw = int(cols.sum())
        if rh == 0 or cw == 0:
            continue
        if int(mask.sum()) != rh * cw:
            continue
        out = g[np.ix_(rows, cols)]
        if set(map(int, np.unique(out))) == set(pair):
            candidates.append((tuple(sorted(pair)), out.copy()))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique two-color image whose mask fills its active row/column product."""
    candidates = _candidate_outputs(np.asarray(grid, dtype=int))
    assert len(candidates) == 1
    return candidates[0][1]


GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
TARGET_H_RANGE = (4, 9)
TARGET_W_RANGE = (4, 9)
DISTRACTOR_RECT_COUNT_RANGE = (0, 4)
GENERATION_ATTEMPTS = 400


def _sample_active_indices(length: int, count: int, gappy: bool) -> list[int]:
    if not gappy:
        start = random.randint(0, length - count)
        return list(range(start, start + count))
    for _ in range(50):
        idx = sorted(random.sample(range(length), count))
        if any(b - a > 1 for a, b in zip(idx, idx[1:])):
            return idx
    start = random.randint(0, length - count)
    return list(range(start, start + count))


def _make_pattern(height: int, width: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    branch = random.randrange(4)
    if branch == 0:
        mask = np.random.rand(height, width) < random.uniform(0.3, 0.7)
    elif branch == 1:
        mask = np.zeros((height, width), dtype=bool)
        r = 0
        on = random.choice([False, True])
        while r < height:
            run = random.randint(1, min(3, height - r))
            mask[r : r + run, :] = on
            on = not on
            r += run
    elif branch == 2:
        mask = np.zeros((height, width), dtype=bool)
        c = 0
        on = random.choice([False, True])
        while c < width:
            run = random.randint(1, min(3, width - c))
            mask[:, c : c + run] = on
            on = not on
            c += run
    else:
        mask = np.zeros((height, width), dtype=bool)
        r0 = random.randint(0, height - 2)
        r1 = random.randint(r0 + 1, height - 1)
        c0 = random.randint(0, width - 2)
        c1 = random.randint(c0 + 1, width - 1)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if random.random() < 0.5:
            mask = ~mask
    if mask.all() or (~mask).all():
        rr, cc = random.randrange(height), random.randrange(width)
        mask[rr, cc] = ~mask[rr, cc]
    out = np.full((height, width), a, dtype=int)
    out[mask] = b
    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _add_distractors(
    grid: np.ndarray,
    active_rows: set[int],
    active_cols: set[int],
    palette: list[int],
) -> None:
    h, w = grid.shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r not in active_rows or c not in active_cols
    ]
    random.shuffle(candidates)
    for _ in range(random.randint(*DISTRACTOR_RECT_COUNT_RANGE)):
        if not candidates:
            break
        color = random.choice(palette)
        placed = False
        for r, c in candidates[:60]:
            max_h = min(3, h - r)
            max_w = min(3, w - c)
            rh = random.randint(1, max_h)
            cw = random.randint(1, max_w)
            cells = [(rr, cc) for rr in range(r, r + rh) for cc in range(c, c + cw)]
            if any((rr in active_rows and cc in active_cols) for rr, cc in cells):
                continue
            _paint_rect(grid, r, c, rh, cw, color)
            placed = True
            break
        if not placed:
            continue


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    target_h = random.randint(*TARGET_H_RANGE)
    target_w = random.randint(*TARGET_W_RANGE)
    if target_h >= h:
        target_h = h - 1
    if target_w >= w:
        target_w = w - 1
    if target_h < 2 or target_w < 2:
        return None

    palette = random.sample(range(10), random.randint(4, 7))
    bg = palette[0]
    target_colors = tuple(random.sample(palette[1:], 2))
    distractor_colors = [c for c in palette if c not in target_colors and c != bg]

    row_mode = random.choice([False, True])
    col_mode = random.choice([False, True])
    active_rows = _sample_active_indices(h, target_h, row_mode)
    active_cols = _sample_active_indices(w, target_w, col_mode)

    grid = np.full((h, w), bg, dtype=int)
    pattern = _make_pattern(target_h, target_w, target_colors)
    grid[np.ix_(active_rows, active_cols)] = pattern

    if distractor_colors:
        _add_distractors(grid, set(active_rows), set(active_cols), distractor_colors)

    candidates = _candidate_outputs(grid)
    target_set = tuple(sorted(target_colors))
    target_out = pattern
    matched = [
        out for pair, out in candidates if pair == target_set and np.array_equal(out, target_out)
    ]
    if len(candidates) != 1 or len(matched) != 1:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, target_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
