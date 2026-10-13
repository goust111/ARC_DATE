from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300
PATTERN_MODES = ("row_runs", "column_bands", "blocky")


def _active_rows(grid: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert rows.size > 0, "expected one sheared foreground region"
    assert np.all(np.diff(rows) == 1), "foreground rows must be contiguous"
    return rows


def _infer_sheared_rectangle(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    rows = _active_rows(grid)
    rel = np.arange(rows.size)
    lefts = []
    rights = []
    for r in rows:
        cols = np.flatnonzero(grid[int(r)] != BG)
        assert cols.size > 0
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))

    lefts = np.asarray(lefts, dtype=int)
    rights = np.asarray(rights, dtype=int)
    bases = lefts - rel
    ends = rights - rel
    assert np.all(bases == bases[0]), "row starts must advance by one column per row"
    assert np.all(ends == ends[0]), "row ends must advance by one column per row"
    width = int(ends[0] - bases[0] + 1)
    assert width >= 2
    return rows, int(bases[0]), width


def _unshear(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    rows, base_col, width = _infer_sheared_rectangle(grid)
    rect = np.zeros((rows.size, width), dtype=int)
    for idx, r in enumerate(rows):
        start = base_col + idx
        rect[idx] = grid[int(r), start : start + width]
    return rect, rows, base_col


def _fill_mirror_holes(rect: np.ndarray) -> np.ndarray:
    out = rect.copy()
    h, w = out.shape
    for c in range(w // 2):
        left = out[:, c].copy()
        right = out[:, w - 1 - c].copy()
        known = (left != BG) & (right != BG)
        assert np.all(left[known] == right[known]), "known mirror pairs must already agree"

        left_missing = (left == BG) & (right != BG)
        right_missing = (right == BG) & (left != BG)
        both_missing = (left == BG) & (right == BG)
        assert not np.any(both_missing), "each hole must have a visible mirror mate"

        out[left_missing, c] = right[left_missing]
        out[right_missing, w - 1 - c] = left[right_missing]

    if w % 2 == 1:
        assert np.all(out[:, w // 2] != BG), "self-mirrored center cells cannot be holes"

    assert np.all(out != BG), "the unsheared rectangle must be fully restored"
    return out


def _reshear(rect: np.ndarray, rows: np.ndarray, base_col: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    h, w = rect.shape
    assert h == rows.size
    for idx, r in enumerate(rows):
        start = base_col + idx
        out[int(r), start : start + w] = rect[idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo the row-wise shear, complete black holes by left-right mirror symmetry, and shear the pattern back.

    Input structure:
    - The grid has a black background with one sheared colored parallelogram.
    - Each foreground row is the corresponding row of a hidden full rectangle shifted right by one extra cell per row.
    - Some cells inside that hidden rectangle are black holes, while the rectangle's outer left and right edges stay visible.

    Rule:
    - Find the contiguous foreground rows and read each row's left endpoint; these endpoints form an arithmetic progression with step +1.
    - Shift row `i` left by its inferred offset to recover the hidden rectangular domain.
    - Treat black cells inside the recovered rectangle as holes and fill each one from its horizontal mirror partner, then shear the completed rows back to their original positions.

    Key insight:
    - The mirror axis is only visible before the shear: on the slanted canvas the symmetry is disguised as a row-dependent offset, so the puzzle must be solved in the unsheared rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect, rows, base_col = _unshear(g)
    filled = _fill_mirror_holes(rect)
    return _reshear(filled, rows, base_col, g.shape)


def _ensure_palette_coverage(arr: np.ndarray, palette: list[int]) -> None:
    positions = random.sample(range(arr.size), len(palette))
    rr, cc = np.divmod(positions, arr.shape[1])
    arr[rr, cc] = np.asarray(palette, dtype=int)


def _row_runs_half(h: int, half_w: int, palette: list[int]) -> np.ndarray:
    arr = np.zeros((h, half_w), dtype=int)
    for r in range(h):
        cut_count = random.randint(0, min(half_w - 1, 2))
        cuts = sorted(random.sample(range(1, half_w), cut_count)) if cut_count else []
        starts = [0] + cuts
        ends = cuts + [half_w]
        last = None
        for c0, c1 in zip(starts, ends):
            choices = [col for col in palette if col != last] or palette
            color = random.choice(choices)
            arr[r, c0:c1] = color
            last = color
    return arr


def _column_bands_half(h: int, half_w: int, palette: list[int]) -> np.ndarray:
    arr = np.zeros((h, half_w), dtype=int)
    for c in range(half_w):
        arr[:, c] = random.choice(palette)
    for _ in range(random.randint(1, max(1, h // 2))):
        r = random.randrange(h)
        c0 = random.randrange(half_w)
        c1 = random.randrange(c0 + 1, half_w + 1)
        arr[r, c0:c1] = random.choice(palette)
    return arr


def _blocky_half(h: int, half_w: int, palette: list[int]) -> np.ndarray:
    arr = np.zeros((h, half_w), dtype=int)
    rows = [0]
    while rows[-1] < h:
        rows.append(min(h, rows[-1] + random.randint(1, 3)))
    cols = [0]
    while cols[-1] < half_w:
        cols.append(min(half_w, cols[-1] + random.randint(1, 2)))
    for r0, r1 in zip(rows, rows[1:]):
        last = None
        for c0, c1 in zip(cols, cols[1:]):
            choices = [col for col in palette if col != last] or palette
            color = random.choice(choices)
            arr[r0:r1, c0:c1] = color
            last = color
    return arr


def _make_symmetric_rectangle(h: int, w: int, palette: list[int]) -> np.ndarray:
    half_w = (w + 1) // 2
    mode = random.choice(PATTERN_MODES)
    if mode == "row_runs":
        half = _row_runs_half(h, half_w, palette)
    elif mode == "column_bands":
        half = _column_bands_half(h, half_w, palette)
    else:
        half = _blocky_half(h, half_w, palette)

    _ensure_palette_coverage(half, palette)
    rect = np.zeros((h, w), dtype=int)
    for c in range(half_w):
        rect[:, c] = half[:, c]
        rect[:, w - 1 - c] = half[:, c]
    return rect


def _candidate_hole_pairs(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(1, w // 2)]


def _punch_holes(rect: np.ndarray) -> np.ndarray:
    h, w = rect.shape
    candidates = _candidate_hole_pairs(h, w)
    assert candidates, "width must be large enough to support asymmetric mirror holes"
    out = rect.copy()
    max_holes = min(len(candidates), max(2, h + w // 3))
    hole_count = random.randint(1, max_holes)
    for r, c in random.sample(candidates, hole_count):
        if random.random() < 0.5:
            out[r, c] = BG
        else:
            out[r, w - 1 - c] = BG
    return out


def _shear_onto_canvas(rect: np.ndarray, top: int, left: int, canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = rect.shape
    out = np.zeros(canvas_shape, dtype=int)
    for r in range(h):
        start = left + r
        out[top + r, start : start + w] = rect[r]
    return out


def _sample_dimensions() -> tuple[int, int]:
    h = random.randint(4, 9)
    parity = random.choice((0, 1))
    widths = [w for w in range(5, 13) if w % 2 == parity]
    w = random.choice(widths)
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_dimensions()
    palette_size = random.randint(2, 4)
    palette = random.sample(PALETTE, palette_size)
    rect = _make_symmetric_rectangle(h, w, palette)
    input_rect = _punch_holes(rect)

    shear_w = w + h - 1
    top = random.randint(0, 4)
    bottom = random.randint(0, 4)
    left = random.randint(0, 4)
    right = random.randint(0, 4)
    canvas_h = top + h + bottom
    canvas_w = left + shear_w + right
    if canvas_h > 30 or canvas_w > 30:
        return None

    grid = _shear_onto_canvas(input_rect, top, left, (canvas_h, canvas_w))
    expected = _shear_onto_canvas(rect, top, left, (canvas_h, canvas_w))

    try:
        result = main(grid)
    except Exception:
        return None

    if not np.array_equal(result, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
