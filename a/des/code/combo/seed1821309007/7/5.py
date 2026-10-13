import random

import numpy as np


BG = 0
AXIS = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _overlay(base: np.ndarray, folded: np.ndarray) -> np.ndarray:
    """Both arrays are indexed from the fold axis outward."""
    out = base.copy()
    depth = folded.shape[0]
    window = out[:depth].copy()
    mask = folded != BG
    window[mask] = folded[mask]
    out[:depth] = window
    return out


def _solve_row_axis(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == AXIS)]
    assert len(rows) == 1, "expected exactly one full red row"
    axis = rows[0]

    top = g[:axis, :]
    bottom = g[axis + 1 :, :]
    assert top.shape[0] > 0 and bottom.shape[0] > 0, "fold axis must split two non-empty sides"
    assert top.shape[0] != bottom.shape[0], "one side must be strictly smaller"

    top_from_axis = np.flipud(top)
    bottom_from_axis = bottom.copy()

    if top.shape[0] < bottom.shape[0]:
        return _overlay(bottom_from_axis, top_from_axis)

    merged = _overlay(top_from_axis, bottom_from_axis)
    return np.flipud(merged)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the smaller side across a full red row/column and merge with the larger side.

    Red marks the fold axis. The smaller half is flipped onto the larger half.
    At overlaps, the folded side wins whenever both cells are non-background.
    The red axis is removed, and the output keeps the larger half's size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_axes = [r for r in range(g.shape[0]) if np.all(g[r, :] == AXIS)]
    col_axes = [c for c in range(g.shape[1]) if np.all(g[:, c] == AXIS)]
    assert len(row_axes) + len(col_axes) == 1, "expected exactly one full red row or column"

    if row_axes:
        return _solve_row_axis(g)
    return _solve_row_axis(g.T).T


def _pick_color(exclude: set[int] | None = None) -> int:
    bad = exclude or set()
    options = [c for c in PALETTE if c not in bad]
    return random.choice(options)


def _paint_shape(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> bool:
    if r0 >= r1 or c0 >= c1:
        return False
    mode = random.choice(("rect", "hline", "vline"))
    h = r1 - r0
    w = c1 - c0

    if mode == "rect":
        hh = random.randint(1, min(3, h))
        ww = random.randint(1, min(4, w))
        rr = random.randint(r0, r1 - hh)
        cc = random.randint(c0, c1 - ww)
        arr[rr : rr + hh, cc : cc + ww] = color
        return True

    if mode == "hline":
        if w < 2:
            return False
        ww = random.randint(2, min(5, w))
        rr = random.randint(r0, r1 - 1)
        cc = random.randint(c0, c1 - ww)
        arr[rr, cc : cc + ww] = color
        return True

    if h < 2:
        return False
    hh = random.randint(2, min(4, h))
    rr = random.randint(r0, r1 - hh)
    cc = random.randint(c0, c1 - 1)
    arr[rr : rr + hh, cc] = color
    return True


def _ensure_cell(arr: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    r, c = random.choice(cells)
    arr[r, c] = color


def _make_patterns(small_len: int, large_len: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    fold = np.full((small_len, width), BG, dtype=int)
    large = np.full((large_len, width), BG, dtype=int)

    for _ in range(random.randint(1, 3)):
        _paint_shape(fold, 0, small_len, 0, width, _pick_color())

    if np.count_nonzero(fold) < 3:
        for _ in range(3 - np.count_nonzero(fold)):
            _ensure_cell(fold, [(r, c) for r in range(small_len) for c in range(width)], _pick_color())

    free_overlap = [(r, c) for r in range(small_len) for c in range(width) if fold[r, c] == BG]
    for _ in range(random.randint(1, 2)):
        color = _pick_color()
        if random.random() < 0.7 and free_overlap:
            mask = np.full_like(large, False, dtype=bool)
            for r, c in free_overlap:
                mask[r, c] = True
            choices = np.argwhere(mask)
            if choices.size > 0:
                rr, cc = choices[random.randrange(len(choices))]
                large[int(rr), int(cc)] = color
        else:
            _paint_shape(large, 0, large_len, 0, width, color)

    if large_len > small_len:
        for _ in range(random.randint(0, 2)):
            _paint_shape(large, small_len, large_len, 0, width, _pick_color())

    fold_cells = [(r, c) for r in range(small_len) for c in range(width) if fold[r, c] != BG]
    if not fold_cells:
        return None
    conflict_count = random.randint(1, min(4, len(fold_cells)))
    for r, c in random.sample(fold_cells, conflict_count):
        large[r, c] = _pick_color(exclude={int(fold[r, c])})

    overlap_large_only = [(r, c) for r in range(small_len) for c in range(width) if large[r, c] != BG and fold[r, c] == BG]
    overlap_fold_only = [(r, c) for r in range(small_len) for c in range(width) if fold[r, c] != BG and large[r, c] == BG]
    overlap_both = [(r, c) for r in range(small_len) for c in range(width) if fold[r, c] != BG and large[r, c] != BG]

    if not overlap_fold_only:
        candidates = [(r, c) for r, c in fold_cells if large[r, c] != BG]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        large[r, c] = BG
        overlap_fold_only.append((r, c))

    if not overlap_large_only:
        cells = free_overlap
        if cells:
            _ensure_cell(large, cells, _pick_color())
            overlap_large_only = [(r, c) for r in range(small_len) for c in range(width) if large[r, c] != BG and fold[r, c] == BG]
        elif large_len > small_len:
            _ensure_cell(
                large,
                [(r, c) for r in range(small_len, large_len) for c in range(width)],
                _pick_color(),
            )
            overlap_large_only = [(r, c) for r in range(small_len) for c in range(width) if large[r, c] != BG and fold[r, c] == BG]

    overlap_both = [(r, c) for r in range(small_len) for c in range(width) if fold[r, c] != BG and large[r, c] != BG]
    if not overlap_both or not overlap_large_only or not overlap_fold_only:
        return None

    if np.count_nonzero(fold) > (small_len * width) // 2:
        return None
    if np.count_nonzero(large) > (large_len * width * 3) // 5:
        return None

    return fold, large


def _build_row_case() -> tuple[np.ndarray, np.ndarray] | None:
    width = random.randint(8, 16)
    small_len = random.randint(2, 6)
    large_len = random.randint(max(4, small_len + 1), 9)
    fold, large = _make_patterns(small_len, large_len, width) or (None, None)
    if fold is None or large is None:
        return None

    smaller_is_top = random.random() < 0.5
    if smaller_is_top:
        top_len, bottom_len = small_len, large_len
        top = np.flipud(fold)
        bottom = large
        expected = _overlay(large, fold)
    else:
        top_len, bottom_len = large_len, small_len
        top = np.flipud(large)
        bottom = fold
        expected = np.flipud(_overlay(large, fold))

    grid = np.full((top_len + 1 + bottom_len, width), BG, dtype=int)
    grid[:top_len, :] = top
    grid[top_len, :] = AXIS
    grid[top_len + 1 :, :] = bottom
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    built = _build_row_case()
    if built is None:
        return None
    grid, expected = built
    if random.random() < 0.5:
        grid = grid.T
        expected = expected.T

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-overlay input")
