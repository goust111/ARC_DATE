from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_MAX_RETRIES = 400
GEN_LAYOUT_RETRIES = 80


def _mode_color(values) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    top = Counter(corners).most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return int(top[0][0])
    return _mode_color(grid.ravel())


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    mask = grid != int(bg)
    assert bool(mask.any()), "expected a non-background board"
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _full_line_color(line: np.ndarray) -> int | None:
    vals = np.unique(line)
    if len(vals) == 1:
        return int(vals[0])
    return None


def _find_separator_color(board: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    row_colors = [_full_line_color(board[r, :]) for r in range(board.shape[0])]
    col_colors = [_full_line_color(board[:, c]) for c in range(board.shape[1])]

    candidates = []
    for color in np.unique(board):
        color = int(color)
        if color == int(bg):
            continue
        row_hits = [idx for idx, val in enumerate(row_colors) if val == color]
        col_hits = [idx for idx, val in enumerate(col_colors) if val == color]
        if row_hits and col_hits:
            candidates.append((len(row_hits) + len(col_hits), color, row_hits, col_hits))
    assert candidates, "expected a separator color spanning full rows and columns"
    candidates.sort(reverse=True)
    _score, sep, rows, cols = candidates[0]
    return int(sep), rows, cols


def _spans_from_separators(length: int, separator_indices: list[int]) -> list[tuple[int, int]]:
    seps = set(int(i) for i in separator_indices)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in seps:
            if start is not None:
                spans.append((int(start), int(idx - 1)))
                start = None
        elif start is None:
            start = int(idx)
    if start is not None:
        spans.append((int(start), int(length - 1)))
    return spans


def _direction_from_pointer(local_pos: tuple[int, int], size: int) -> str:
    center = (size - 1) / 2.0
    dr = float(local_pos[0]) - center
    dc = float(local_pos[1]) - center
    assert dr != 0 or dc != 0, "pointer cannot sit at the region center"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _parse_board(grid: np.ndarray):
    bg = _background_color(grid)
    r0, r1, c0, c1 = _non_bg_bbox(grid, bg)
    board = grid[r0 : r1 + 1, c0 : c1 + 1]
    sep, sep_rows, sep_cols = _find_separator_color(board, bg)

    row_spans = _spans_from_separators(board.shape[0], sep_rows)
    col_spans = _spans_from_separators(board.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one region"

    row_sizes = {end - start + 1 for start, end in row_spans}
    col_sizes = {end - start + 1 for start, end in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "regions must have uniform size"
    region_size = int(next(iter(row_sizes)))
    assert region_size == int(next(iter(col_sizes))), "regions must be square"

    fills = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    directions: list[list[str]] = []
    for rr, (rs, re) in enumerate(row_spans):
        dir_row: list[str] = []
        for cc, (cs, ce) in enumerate(col_spans):
            patch = board[rs : re + 1, cs : ce + 1]
            vals, counts = np.unique(patch, return_counts=True)
            assert len(vals) >= 2, "each region must contain a fill color and a pointer"
            fill = int(vals[int(np.argmax(counts))])
            pointer_cells = np.argwhere(patch != fill)
            assert len(pointer_cells) == 1, "each region must contain exactly one pointer pixel"
            pr, pc = map(int, pointer_cells[0])
            fills[rr, cc] = fill
            dir_row.append(_direction_from_pointer((pr, pc), region_size))
        directions.append(dir_row)

    return {
        "bg": int(bg),
        "sep": int(sep),
        "bbox": (int(r0), int(r1), int(c0), int(c1)),
        "board": board,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "region_size": int(region_size),
        "fills": fills,
        "directions": directions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each square region by dereferencing the neighboring region indicated by its internal pointer pixel.

    Input structure:
    - A dominant background surrounds one rectangular board.
    - Inside the board, full grey rows and full grey columns partition the area into equal-sized square regions.
    - Each region is filled with one solid color and contains exactly one single-pixel pointer of another color.

    Rule:
    - For each region, locate the pointer pixel and compare its offset from the region center; the axis with larger absolute offset determines whether it points up, down, left, or right.
    - Replace the region's fill color with the fill color of the orthogonally adjacent region in that direction.
    - If the pointed direction leaves the region grid, keep that region's original fill color.
    - Grey separators and the outer background stay unchanged, and the pointer pixel is overwritten by the region's new fill color.

    Key insight:
    - The lone odd-color pixel is not noise: its position encodes a cardinal direction that tells each region which neighbor's color to copy.
    """
    g = np.asarray(grid, dtype=int)
    info = _parse_board(g)
    fills = info["fills"]
    directions = info["directions"]
    row_spans = info["row_spans"]
    col_spans = info["col_spans"]
    r0, r1, c0, c1 = info["bbox"]

    out = g.copy()
    board_out = out[r0 : r1 + 1, c0 : c1 + 1]
    n_rows, n_cols = fills.shape
    for rr, (rs, re) in enumerate(row_spans):
        for cc, (cs, ce) in enumerate(col_spans):
            dr, dc = DIR_STEP[directions[rr][cc]]
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                new_color = int(fills[nr, nc])
            else:
                new_color = int(fills[rr, cc])
            board_out[rs : re + 1, cs : ce + 1] = new_color

    return out


def _pointer_candidates(size: int) -> dict[str, list[tuple[int, int]]]:
    out = {name: [] for name in DIR_STEP}
    center = (size - 1) / 2.0
    for r in range(size):
        for c in range(size):
            dr = float(r) - center
            dc = float(c) - center
            if dr == 0 and dc == 0:
                continue
            if abs(dr) > abs(dc):
                out["up" if dr < 0 else "down"].append((int(r), int(c)))
            elif abs(dc) > abs(dr):
                out["left" if dc < 0 else "right"].append((int(r), int(c)))
    return out


def _assign_region_colors(n_rows: int, n_cols: int, palette: list[int]) -> np.ndarray:
    fills = np.zeros((n_rows, n_cols), dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            blocked = set()
            if r > 0:
                blocked.add(int(fills[r - 1, c]))
            if c > 0:
                blocked.add(int(fills[r, c - 1]))
            choices = [color for color in palette if color not in blocked]
            fills[r, c] = int(random.choice(choices))
    return fills


def _assign_directions(n_rows: int, n_cols: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    base_dirs = (list(DIR_STEP) * ((len(cells) + 3) // 4))[: len(cells)]
    for _ in range(GEN_LAYOUT_RETRIES):
        random.shuffle(base_dirs)
        random.shuffle(cells)
        dirs = np.empty((n_rows, n_cols), dtype=object)
        for pos, direction in zip(cells, base_dirs):
            dirs[pos] = direction

        seen = set(str(v) for v in dirs.ravel())
        if seen != set(DIR_STEP):
            continue

        has_valid = False
        has_invalid = False
        for r in range(n_rows):
            for c in range(n_cols):
                dr, dc = DIR_STEP[str(dirs[r, c])]
                rr = int(r + dr)
                cc = int(c + dc)
                valid = 0 <= rr < n_rows and 0 <= cc < n_cols
                has_valid |= valid
                has_invalid |= not valid
        if has_valid and has_invalid:
            return dirs
    return None


def _choose_canvas_margins(board_h: int, board_w: int, grey_count: int) -> tuple[int, int, int, int] | None:
    max_mh = 30 - board_h
    max_mw = 30 - board_w
    if max_mh < 2 or max_mw < 2:
        return None

    valid_pairs = []
    for mh in range(2, max_mh + 1):
        for mw in range(2, max_mw + 1):
            bg_cells = board_h * mw + board_w * mh + mh * mw
            if bg_cells > grey_count:
                valid_pairs.append((mh, mw))
    if not valid_pairs:
        return None

    mh, mw = random.choice(valid_pairs)
    top = random.randint(1, mh - 1)
    bottom = mh - top
    left = random.randint(1, mw - 1)
    right = mw - left
    return int(top), int(bottom), int(left), int(right)


def _expected_from_metadata(
    grid: np.ndarray,
    fills: np.ndarray,
    dirs: np.ndarray,
    top: int,
    left: int,
    cell_size: int,
) -> np.ndarray:
    out = grid.copy()
    n_rows, n_cols = fills.shape
    for r in range(n_rows):
        rs = top + 1 + r * (cell_size + 1)
        for c in range(n_cols):
            cs = left + 1 + c * (cell_size + 1)
            dr, dc = DIR_STEP[str(dirs[r, c])]
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                color = int(fills[rr, cc])
            else:
                color = int(fills[r, c])
            out[rs : rs + cell_size, cs : cs + cell_size] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    max_cell_h = (28 - (n_rows + 1)) // n_rows
    max_cell_w = (28 - (n_cols + 1)) // n_cols
    max_cell = min(5, max_cell_h, max_cell_w)
    if max_cell < 3:
        return None
    cell_size = random.randint(3, max_cell)

    board_h = n_rows * cell_size + (n_rows + 1)
    board_w = n_cols * cell_size + (n_cols + 1)
    grey_count = (n_rows + 1) * board_w + (n_cols + 1) * board_h - (n_rows + 1) * (n_cols + 1)
    margins = _choose_canvas_margins(board_h, board_w, grey_count)
    if margins is None:
        return None
    top, bottom, left, right = margins

    colors = list(range(10))
    random.shuffle(colors)
    bg, sep, pointer = map(int, colors[:3])
    fill_pool = [int(color) for color in colors[3:]]
    if len(fill_pool) < 4:
        return None
    palette_size = random.randint(4, min(6, len(fill_pool)))
    palette = random.sample(fill_pool, palette_size)

    fills = _assign_region_colors(n_rows, n_cols, palette)
    dirs = _assign_directions(n_rows, n_cols)
    if dirs is None:
        return None

    grid = np.full((board_h + top + bottom, board_w + left + right), bg, dtype=int)
    board = grid[top : top + board_h, left : left + board_w]
    board[:, :] = sep

    candidates = _pointer_candidates(cell_size)
    for direction in DIR_STEP:
        if not candidates[direction]:
            return None

    for r in range(n_rows):
        rs = 1 + r * (cell_size + 1)
        for c in range(n_cols):
            cs = 1 + c * (cell_size + 1)
            board[rs : rs + cell_size, cs : cs + cell_size] = int(fills[r, c])
            pr, pc = random.choice(candidates[str(dirs[r, c])])
            board[rs + pr, cs + pc] = pointer

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    expected = _expected_from_metadata(grid, fills, dirs, top, left, cell_size)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_RETRIES} attempts")
