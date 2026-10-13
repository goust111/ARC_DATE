import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
CANVAS_COLORS = (3, 4, 5, 6, 7, 8, 9)
BRANCHES = ("none", "vertical", "horizontal", "both")
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a visible canvas region"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _copy_stacks(stacks: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[cell[:] for cell in row] for row in stacks]


def _canvas_to_stacks(canvas: np.ndarray) -> list[list[list[int]]]:
    h, w = canvas.shape
    return [
        [[int(canvas[r, c])] if int(canvas[r, c]) != BG else [] for c in range(w)]
        for r in range(h)
    ]


def _render_stacks(stacks: list[list[list[int]]]) -> np.ndarray:
    h = len(stacks)
    w = len(stacks[0])
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if stacks[r][c]:
                out[r, c] = stacks[r][c][-1]
    return out


def _fold_vertical(
    stacks: list[list[list[int]]], fold_col: int | None
) -> list[list[list[int]]]:
    if fold_col is None:
        return _copy_stacks(stacks)

    h = len(stacks)
    w = len(stacks[0])
    assert 0 <= fold_col < w - 1

    min_col = min(0, 2 * fold_col + 2 - w)
    out_w = fold_col - min_col + 1
    out = [[[] for _ in range(out_w)] for _ in range(h)]

    for r in range(h):
        for c in range(fold_col + 1):
            out[r][c - min_col].extend(stacks[r][c])

    for r in range(h):
        for c in range(fold_col + 1, w):
            target_c = 2 * fold_col + 1 - c - min_col
            out[r][target_c].extend(stacks[r][c])

    return out


def _fold_horizontal(
    stacks: list[list[list[int]]], fold_row: int | None
) -> list[list[list[int]]]:
    if fold_row is None:
        return _copy_stacks(stacks)

    h = len(stacks)
    w = len(stacks[0])
    assert 0 <= fold_row < h - 1

    min_row = min(0, 2 * fold_row + 2 - h)
    out_h = fold_row - min_row + 1
    out = [[[] for _ in range(w)] for _ in range(out_h)]

    for r in range(fold_row + 1):
        for c in range(w):
            out[r - min_row][c].extend(stacks[r][c])

    for r in range(fold_row + 1, h):
        target_r = 2 * fold_row + 1 - r - min_row
        for c in range(w):
            out[target_r][c].extend(stacks[r][c])

    return out


def _fold_canvas(
    canvas: np.ndarray, fold_col: int | None, fold_row: int | None
) -> np.ndarray:
    stacks = _canvas_to_stacks(canvas)
    stacks = _fold_vertical(stacks, fold_col)
    stacks = _fold_horizontal(stacks, fold_row)
    return _render_stacks(stacks)


def _candidate_left_cols(width: int, fold_col: int) -> list[int]:
    cols = []
    for c_left in range(1, fold_col + 1):
        c_right = 2 * fold_col + 1 - c_left
        if 1 <= c_right <= width - 2:
            cols.append(c_left)
    return cols


def _candidate_top_rows(height: int, fold_row: int) -> list[int]:
    rows = []
    for r_top in range(1, fold_row + 1):
        r_bottom = 2 * fold_row + 1 - r_top
        if 1 <= r_bottom <= height - 2:
            rows.append(r_top)
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the marked rectangular canvas according to the top and left crease bars and return the visible stacked result.

    Input structure:
    - Black is the background.
    - A single rectangular canvas is the unique non-instruction region made of colors other than black, blue, and red; its colored perimeter makes the canvas bbox explicit.
    - The row immediately above the canvas is a horizontal instruction bar with at most one red pixel, and the column immediately left of the canvas is a vertical instruction bar with at most one blue pixel.

    Rule:
    - A red pixel in the top bar at canvas column c means: fold the part to the right of c leftward across the gap after c, placing the moved right-side layers on top of the stationary left side.
    - A blue pixel in the left bar at canvas row r means: after the vertical fold, fold the part below r upward across the gap after r, placing the moved lower layers on top of the stationary upper side.
    - Black stays transparent during overlap: each output cell shows the topmost non-black layer after both folds, or black if every layer there is black.

    Key insight:
    - The solver must first locate the canvas rectangle, then decode the two edge markers as crease positions, and finally simulate the ordered reflections while preserving layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvas_mask = ~np.isin(g, (BG, BLUE, RED))
    r0, c0, r1, c1 = _bbox(canvas_mask)
    assert r0 > 0 and c0 > 0, "canvas must have a visible top bar row and left bar column"

    canvas = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = canvas.shape
    assert h % 2 == 0 and w % 2 == 0
    assert not np.any(np.isin(canvas, (BLUE, RED)))
    assert np.all(canvas[0, :] != BG)
    assert np.all(canvas[-1, :] != BG)
    assert np.all(canvas[:, 0] != BG)
    assert np.all(canvas[:, -1] != BG)

    top_bar = g[r0 - 1, c0 : c1 + 1]
    left_bar = g[r0 : r1 + 1, c0 - 1]
    assert np.all(np.isin(top_bar, (BG, RED)))
    assert np.all(np.isin(left_bar, (BG, BLUE)))

    red_cols = np.flatnonzero(top_bar == RED)
    blue_rows = np.flatnonzero(left_bar == BLUE)
    assert len(red_cols) <= 1 and len(blue_rows) <= 1

    fold_col = int(red_cols[0]) if len(red_cols) else None
    fold_row = int(blue_rows[0]) if len(blue_rows) else None
    return _fold_canvas(canvas, fold_col, fold_row)


def _sample_canvas_side() -> int:
    return 2 * random.randint(3, 9)


def _sample_fold(length: int) -> int:
    return random.randint(2, length - 4)


def _shape_crosses_fold(
    cells: list[tuple[int, int]], fold_col: int | None, fold_row: int | None
) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if fold_col is not None and min(cols) <= fold_col < max(cols):
        return True
    if fold_row is not None and min(rows) <= fold_row < max(rows):
        return True
    return False


def _place_shape(
    canvas: np.ndarray,
    occupied: np.ndarray,
    fold_col: int | None,
    fold_row: int | None,
    colors: list[int],
) -> bool:
    h, w = canvas.shape
    for _ in range(40):
        shape = random.choice(SHAPES)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if h - sh - 1 < 1 or w - sw - 1 < 1:
            return False
        r0 = random.randint(1, h - sh - 1)
        c0 = random.randint(1, w - sw - 1)
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if _shape_crosses_fold(cells, fold_col, fold_row):
            continue
        if any(occupied[r, c] for r, c in cells):
            continue
        color = random.choice(colors)
        for r, c in cells:
            canvas[r, c] = color
            occupied[r, c] = True
        return True
    return False


def _build_canvas(
    height: int, width: int, fold_col: int | None, fold_row: int | None, branch: str
) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=int)
    frame_color = random.choice(CANVAS_COLORS)
    interior_colors = [color for color in CANVAS_COLORS if color != frame_color]

    canvas[0, :] = frame_color
    canvas[-1, :] = frame_color
    canvas[:, 0] = frame_color
    canvas[:, -1] = frame_color
    occupied = canvas != BG

    random.shuffle(interior_colors)
    if branch == "both":
        left_cols = _candidate_left_cols(width, fold_col)
        top_rows = _candidate_top_rows(height, fold_row)
        if not left_cols or not top_rows:
            return None
        c_left = random.choice(left_cols)
        r_top = random.choice(top_rows)
        c_right = 2 * fold_col + 1 - c_left
        r_bottom = 2 * fold_row + 1 - r_top
        quad = [(r_top, c_left), (r_top, c_right), (r_bottom, c_left), (r_bottom, c_right)]
        for (r, c), color in zip(quad, random.sample(interior_colors, 4)):
            canvas[r, c] = color
            occupied[r, c] = True
    elif branch == "vertical":
        left_cols = _candidate_left_cols(width, fold_col)
        if not left_cols:
            return None
        c_left = random.choice(left_cols)
        c_right = 2 * fold_col + 1 - c_left
        r = random.randint(1, height - 2)
        colors = random.sample(interior_colors, 2)
        canvas[r, c_left] = colors[0]
        canvas[r, c_right] = colors[1]
        occupied[r, c_left] = True
        occupied[r, c_right] = True
    elif branch == "horizontal":
        top_rows = _candidate_top_rows(height, fold_row)
        if not top_rows:
            return None
        r_top = random.choice(top_rows)
        r_bottom = 2 * fold_row + 1 - r_top
        c = random.randint(1, width - 2)
        colors = random.sample(interior_colors, 2)
        canvas[r_top, c] = colors[0]
        canvas[r_bottom, c] = colors[1]
        occupied[r_top, c] = True
        occupied[r_bottom, c] = True

    extra_shapes = random.randint(4, 8)
    placed = 0
    for _ in range(extra_shapes * 3):
        if _place_shape(canvas, occupied, fold_col, fold_row, interior_colors):
            placed += 1
            if placed >= extra_shapes:
                break

    if placed < 3:
        return None
    return canvas


def _assemble_input(
    canvas: np.ndarray, fold_col: int | None, fold_row: int | None
) -> np.ndarray:
    h, w = canvas.shape
    top_pad = random.randint(1, 5)
    left_pad = random.randint(1, 5)
    bottom_pad = random.randint(0, 4)
    right_pad = random.randint(0, 4)
    grid = np.zeros((top_pad + h + bottom_pad, left_pad + w + right_pad), dtype=int)
    r0, c0 = top_pad, left_pad
    grid[r0 : r0 + h, c0 : c0 + w] = canvas
    if fold_col is not None:
        grid[r0 - 1, c0 + fold_col] = RED
    if fold_row is not None:
        grid[r0 + fold_row, c0 - 1] = BLUE
    return grid


def _generate_attempt() -> np.ndarray | None:
    branch = random.choice(BRANCHES)
    height = _sample_canvas_side()
    width = _sample_canvas_side()
    fold_col = _sample_fold(width) if branch in ("vertical", "both") else None
    fold_row = _sample_fold(height) if branch in ("horizontal", "both") else None

    canvas = _build_canvas(height, width, fold_col, fold_row, branch)
    if canvas is None:
        return None

    grid = _assemble_input(canvas, fold_col, fold_row)
    if max(grid.shape) > 30:
        return None

    expected = _fold_canvas(canvas, fold_col, fold_row)
    if max(expected.shape) > 30:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if branch == "none" and not np.array_equal(solved, canvas):
        return None
    if branch != "none" and np.array_equal(solved, canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-canvas input")
