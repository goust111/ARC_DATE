import random

import numpy as np


BG = 7  # orange
PALETTE = [c for c in range(10) if c != BG]
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each foreground color into a horizontal or vertical bar.

    For every non-background color:
    - compare its total row span and column span across all of its pixels
    - if column span >= row span, draw a horizontal bar of the same color
      on the color's center row, starting at the left edge
    - otherwise draw a vertical bar on the color's center column, starting
      at the top edge
    - the bar length equals the number of pixels of that color
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _background_color(g)

    out = np.full((h, w), bg, dtype=int)
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert coords.size > 0
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        row_span = r1 - r0
        col_span = c1 - c0
        count = int(len(coords))

        if col_span >= row_span:
            row = (r0 + r1) // 2
            assert count <= w
            out[row, :count] = color
        else:
            col = (c0 + c1) // 2
            assert count <= h
            out[:count, col] = color
    return out


def _mark_padded(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    r1 = min(h, top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _horizontal_cells(box_h: int, box_w: int, count: int) -> set[tuple[int, int]]:
    row_by_col = [random.randrange(box_h) for _ in range(box_w)]
    row_by_col[0] = 0
    row_by_col[-1] = box_h - 1
    cells = {(row_by_col[c], c) for c in range(box_w)}

    remaining = [(r, c) for r in range(box_h) for c in range(box_w) if (r, c) not in cells]
    random.shuffle(remaining)
    for cell in remaining[: count - len(cells)]:
        cells.add(cell)
    return cells


def _vertical_cells(box_h: int, box_w: int, count: int) -> set[tuple[int, int]]:
    col_by_row = [random.randrange(box_w) for _ in range(box_h)]
    col_by_row[0] = 0
    col_by_row[-1] = box_w - 1
    cells = {(r, col_by_row[r]) for r in range(box_h)}

    remaining = [(r, c) for r in range(box_h) for c in range(box_w) if (r, c) not in cells]
    random.shuffle(remaining)
    for cell in remaining[: count - len(cells)]:
        cells.add(cell)
    return cells


def _place_horizontal(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    region_rows: tuple[int, int],
    region_cols: tuple[int, int],
    max_len: int,
    used_rows: set[int],
) -> bool:
    row_lo, row_hi = region_rows
    col_lo, col_hi = region_cols
    region_h = row_hi - row_lo
    region_w = col_hi - col_lo

    for _ in range(120):
        box_h = random.randint(2, min(4, region_h))
        min_w = max(5, box_h + 2)
        max_w = min(10, region_w)
        if min_w > max_w:
            return False
        box_w = random.randint(min_w, max_w)
        max_count = min(max_len, box_h * box_w)
        if max_count < box_w:
            return False
        count = random.randint(box_w, max_count)

        top = random.randint(row_lo, row_hi - box_h)
        left = random.randint(col_lo, col_hi - box_w)
        center_row = top + (box_h - 1) // 2
        if center_row in used_rows or not _free(occupied, top, left, box_h, box_w):
            continue

        for r, c in _horizontal_cells(box_h, box_w, count):
            grid[top + r, left + c] = color
        _mark_padded(occupied, top, left, box_h, box_w)
        used_rows.add(center_row)
        return True
    return False


def _place_vertical(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    region_rows: tuple[int, int],
    region_cols: tuple[int, int],
    max_len: int,
    used_cols: set[int],
) -> bool:
    row_lo, row_hi = region_rows
    col_lo, col_hi = region_cols
    region_h = row_hi - row_lo
    region_w = col_hi - col_lo

    for _ in range(120):
        box_w = random.randint(2, min(4, region_w))
        min_h = max(5, box_w + 2)
        max_h = min(10, region_h)
        if min_h > max_h:
            return False
        box_h = random.randint(min_h, max_h)
        max_count = min(max_len, box_h * box_w)
        if max_count < box_h:
            return False
        count = random.randint(box_h, max_count)

        top = random.randint(row_lo, row_hi - box_h)
        left = random.randint(col_lo, col_hi - box_w)
        center_col = left + (box_w - 1) // 2
        if center_col in used_cols or not _free(occupied, top, left, box_h, box_w):
            continue

        for r, c in _vertical_cells(box_h, box_w, count):
            grid[top + r, left + c] = color
        _mark_padded(occupied, top, left, box_h, box_w)
        used_cols.add(center_col)
        return True
    return False


def _bars_do_not_overlap(out: np.ndarray, bg: int) -> bool:
    return int(np.count_nonzero(out != bg))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    split_row = random.randint(8, h - 8)
    split_col = random.randint(8, w - 8)
    n_h, n_v = random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])
    colors = random.sample(PALETTE, n_h + n_v)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    horiz_rows = (split_row, h)
    horiz_cols = (0, split_col)
    vert_rows = (0, split_row)
    vert_cols = (split_col, w)

    for color in colors[:n_h]:
        if not _place_horizontal(grid, occupied, color, horiz_rows, horiz_cols, split_col, used_rows):
            return None
    for color in colors[n_h:]:
        if not _place_vertical(grid, occupied, color, vert_rows, vert_cols, split_row, used_cols):
            return None

    out = main(grid)
    if _bars_do_not_overlap(out, BG) != int(np.count_nonzero(grid != BG)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
