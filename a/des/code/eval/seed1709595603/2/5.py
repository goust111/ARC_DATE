from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

LOCK_TO_BG = {
    RED: ORANGE,
    BLUE: AZURE,
    GREEN: YELLOW,
    MAGENTA: MAROON,
}
LOCK_COLORS = tuple(LOCK_TO_BG)

GEN_RETRIES = 300
QUADRANT_SPAN_RANGE = (6, 10)


def _find_dividers(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    divider_rows = [
        r for r in range(h) if np.all(np.isin(grid[r, :], (GREY, MAROON)))
    ]
    divider_cols = [
        c for c in range(w) if np.all(np.isin(grid[:, c], (GREY, MAROON)))
    ]
    assert len(divider_rows) == 1, "expected a unique divider row"
    assert len(divider_cols) == 1, "expected a unique divider column"
    return int(divider_rows[0]), int(divider_cols[0])


def _quadrant_specs(
    h: int, w: int, mid_r: int, mid_c: int
) -> list[dict[str, int | str]]:
    return [
        {
            "name": "tl",
            "r0": 0,
            "r1": mid_r - 1,
            "c0": 0,
            "c1": mid_c - 1,
            "row_step": -1,
            "col_step": -1,
        },
        {
            "name": "tr",
            "r0": 0,
            "r1": mid_r - 1,
            "c0": mid_c + 1,
            "c1": w - 1,
            "row_step": -1,
            "col_step": 1,
        },
        {
            "name": "bl",
            "r0": mid_r + 1,
            "r1": h - 1,
            "c0": 0,
            "c1": mid_c - 1,
            "row_step": 1,
            "col_step": -1,
        },
        {
            "name": "br",
            "r0": mid_r + 1,
            "r1": h - 1,
            "c0": mid_c + 1,
            "c1": w - 1,
            "row_step": 1,
            "col_step": 1,
        },
    ]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _completes_rectangle(
    grid: np.ndarray,
    lock_color: int,
    bbox: tuple[int, int, int, int],
    target_cells: list[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    for r, c in target_cells:
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        patch[r - r0, c - c0] = lock_color
    return bool(np.all(patch == lock_color))


def _repair_quadrant(
    inp: np.ndarray,
    out: np.ndarray,
    mid_r: int,
    mid_c: int,
    spec: dict[str, int | str],
) -> None:
    r0 = int(spec["r0"])
    r1 = int(spec["r1"])
    c0 = int(spec["c0"])
    c1 = int(spec["c1"])
    row_step = int(spec["row_step"])
    col_step = int(spec["col_step"])

    quad = inp[r0 : r1 + 1, c0 : c1 + 1]
    lock_colors = [int(color) for color in np.unique(quad) if int(color) != BLACK]
    assert len(lock_colors) == 1, "each quadrant must contain exactly one lock color"
    lock_color = lock_colors[0]
    assert lock_color in LOCK_TO_BG, "unexpected lock color"

    qr0, qc0, qr1, qc1 = _bbox(quad == lock_color)
    bbox = (r0 + qr0, c0 + qc0, r0 + qr1, c0 + qc1)

    row_key_cells = [
        (mid_r, c) for c in range(c0, c1 + 1) if int(inp[mid_r, c]) == MAROON
    ]
    col_key_cells = [
        (r, mid_c) for r in range(r0, r1 + 1) if int(inp[r, mid_c]) == MAROON
    ]
    assert bool(row_key_cells) ^ bool(col_key_cells), "expected exactly one key per quadrant"

    if row_key_cells:
        key_cells = row_key_cells
        target_cells = [(r + row_step, c) for r, c in key_cells]
    else:
        key_cells = col_key_cells
        target_cells = [(r, c + col_step) for r, c in key_cells]

    if not _completes_rectangle(inp, lock_color, bbox, target_cells):
        return

    for r, c in key_cells:
        out[r, c] = GREY
    for r, c in target_cells:
        out[r, c] = lock_color

    quad_out = out[r0 : r1 + 1, c0 : c1 + 1]
    quad_out[quad_out == BLACK] = LOCK_TO_BG[lock_color]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each maroon key into the facing lock notch when the translated shape
    exactly restores a solid rectangle, then recolor that quadrant's background.
    """
    inp = np.asarray(grid, dtype=int)
    assert inp.ndim == 2, "expected a 2D grid"

    mid_r, mid_c = _find_dividers(inp)
    out = inp.copy()

    for spec in _quadrant_specs(inp.shape[0], inp.shape[1], mid_r, mid_c):
        _repair_quadrant(inp, out, mid_r, mid_c, spec)

    return out


def _sample_edge_pattern(edge_len: int) -> tuple[int, ...]:
    interior = edge_len - 2
    assert interior >= 3, "edge must support both match and mismatch patterns"

    count = random.randint(2, min(4, interior))
    span = count
    if interior >= count + 1 and random.random() < 0.45:
        span = count + 1

    start = random.randint(0, interior - span)
    rel = list(range(start, start + span))
    if span > count:
        gap_index = random.randint(1, span - 2)
        rel.pop(gap_index)

    return tuple(pos + 1 for pos in rel)


def _different_pattern(edge_len: int, target: tuple[int, ...]) -> tuple[int, ...]:
    for _ in range(100):
        cand = _sample_edge_pattern(edge_len)
        if cand != target:
            return cand
    raise RuntimeError("failed to sample a mismatched key pattern")


def _draw_lock(
    grid: np.ndarray,
    spec: dict[str, int | str],
    mid_r: int,
    mid_c: int,
    lock_color: int,
    use_row_key: bool,
    matched: bool,
) -> None:
    r0 = int(spec["r0"])
    r1 = int(spec["r1"])
    c0 = int(spec["c0"])
    c1 = int(spec["c1"])
    row_step = int(spec["row_step"])
    col_step = int(spec["col_step"])

    qh = r1 - r0 + 1
    qw = c1 - c0 + 1

    if use_row_key:
        max_w = min(7, qw - 1)
        max_h = min(6, qh)
        if max_w < 5 or max_h < 3:
            raise ValueError("quadrant too small for row-facing lock")

        width = random.randint(5, max_w)
        height = random.randint(3, max_h)

        if row_step < 0:
            lock_r1 = r1
            lock_r0 = lock_r1 - height + 1
        else:
            lock_r0 = r0
            lock_r1 = lock_r0 + height - 1

        if col_step < 0:
            left_min = c0
            left_max = c1 - width
        else:
            left_min = c0 + 1
            left_max = c1 - width + 1
        if left_max < left_min:
            raise ValueError("no legal row-facing placement")

        lock_c0 = random.randint(left_min, left_max)
        lock_c1 = lock_c0 + width - 1

        grid[lock_r0 : lock_r1 + 1, lock_c0 : lock_c1 + 1] = lock_color
        notch = _sample_edge_pattern(width)
        edge_row = lock_r1 if row_step < 0 else lock_r0
        for offset in notch:
            grid[edge_row, lock_c0 + offset] = BLACK

        key_shape = notch if matched else _different_pattern(width, notch)
        for offset in key_shape:
            grid[mid_r, lock_c0 + offset] = MAROON
    else:
        max_h = min(7, qh - 1)
        max_w = min(6, qw)
        if max_h < 5 or max_w < 3:
            raise ValueError("quadrant too small for column-facing lock")

        height = random.randint(5, max_h)
        width = random.randint(3, max_w)

        if col_step < 0:
            lock_c1 = c1
            lock_c0 = lock_c1 - width + 1
        else:
            lock_c0 = c0
            lock_c1 = lock_c0 + width - 1

        if row_step < 0:
            top_min = r0
            top_max = r1 - height
        else:
            top_min = r0 + 1
            top_max = r1 - height + 1
        if top_max < top_min:
            raise ValueError("no legal column-facing placement")

        lock_r0 = random.randint(top_min, top_max)
        lock_r1 = lock_r0 + height - 1

        grid[lock_r0 : lock_r1 + 1, lock_c0 : lock_c1 + 1] = lock_color
        notch = _sample_edge_pattern(height)
        edge_col = lock_c1 if col_step < 0 else lock_c0
        for offset in notch:
            grid[lock_r0 + offset, edge_col] = BLACK

        key_shape = notch if matched else _different_pattern(height, notch)
        for offset in key_shape:
            grid[lock_r0 + offset, mid_c] = MAROON


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        top_h = random.randint(*QUADRANT_SPAN_RANGE)
        bottom_h = random.randint(*QUADRANT_SPAN_RANGE)
        left_w = random.randint(*QUADRANT_SPAN_RANGE)
        right_w = random.randint(*QUADRANT_SPAN_RANGE)

        mid_r = top_h
        mid_c = left_w
        h = top_h + 1 + bottom_h
        w = left_w + 1 + right_w

        grid = np.full((h, w), BLACK, dtype=int)
        grid[mid_r, :] = GREY
        grid[:, mid_c] = GREY

        matched_indices = set(random.sample(range(4), 2))
        lock_colors = random.sample(list(LOCK_COLORS), 4)
        use_row_key_flags = [bool(random.getrandbits(1)) for _ in range(4)]

        try:
            for idx, spec in enumerate(_quadrant_specs(h, w, mid_r, mid_c)):
                _draw_lock(
                    grid,
                    spec,
                    mid_r,
                    mid_c,
                    lock_colors[idx],
                    use_row_key_flags[idx],
                    idx in matched_indices,
                )
            _ = main(grid)
            return grid
        except (AssertionError, ValueError, RuntimeError):
            continue

    raise RuntimeError("failed to generate a valid puzzle instance")
