import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6


_SYMBOL_TO_COLOR = {
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)): GREEN,
    ((1, 0, 1), (0, 1, 0), (1, 0, 1)): BLUE,
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): YELLOW,
    ((1, 0, 1), (1, 0, 1), (1, 1, 1)): PINK,
}
_SYMBOL_ITEMS = list(_SYMBOL_TO_COLOR.items())
_RNG = np.random.default_rng()
_GEN_RETRY_LIMIT = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_symbol(block: np.ndarray) -> int | None:
    if block.shape != (3, 3):
        return None
    if not set(map(int, np.unique(block))).issubset({BLACK, GREY}):
        return None
    mask = tuple(tuple(int(v == GREY) for v in row) for row in block)
    return _SYMBOL_TO_COLOR.get(mask)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_mask = grid == RED
    assert red_mask.any(), "expected one red frame"
    r0, c0, r1, c1 = _bbox(red_mask)
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]

    border = np.zeros_like(frame, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    assert np.all(frame[border] == RED), "frame border must be fully red"
    assert not np.any(frame[~border] == RED), "frame interior must not contain red"
    assert int(red_mask.sum()) == int(border.sum()), "all red pixels must belong to the frame"
    assert frame.shape[0] >= 8 and frame.shape[1] >= 8
    return r0, c0, r1, c1


def _is_marker(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    if coords.shape[0] not in (2, 3):
        return False
    cols = np.unique(coords[:, 1])
    if len(cols) != 1:
        return False
    rows = np.sort(coords[:, 0])
    return bool(np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + len(rows))))


def _extract_palette_and_run_len(grid: np.ndarray, frame_left: int) -> tuple[list[int], int]:
    left = grid[:, :frame_left]
    assert left.size > 0
    assert set(map(int, np.unique(left))).issubset({BLACK, GREY}), "left region must use only black/grey"

    grey_mask = left == GREY
    grey_counts = grey_mask.sum(axis=0)
    best_score = -1
    symbol_col0 = None
    for c0 in range(left.shape[1] - 2):
        score = int(grey_counts[c0 : c0 + 3].sum())
        if score > best_score:
            best_score = score
            symbol_col0 = c0
    assert symbol_col0 is not None and best_score > 0

    candidates = []
    for start in range(left.shape[0] - 2):
        seq_rows = []
        palette = []
        row = start
        while row + 2 < left.shape[0] and len(seq_rows) < 4:
            color = _decode_symbol(left[row : row + 3, symbol_col0 : symbol_col0 + 3])
            if color is None:
                break
            seq_rows.append(row)
            palette.append(int(color))
            row += 4

        if not 2 <= len(seq_rows) <= 4:
            continue

        used_windows = np.zeros_like(grey_mask, dtype=bool)
        for row in seq_rows:
            used_windows[row : row + 3, symbol_col0 : symbol_col0 + 3] = True
        leftover = np.argwhere(grey_mask & ~used_windows)
        if not _is_marker(leftover):
            continue

        ok = True
        for row in seq_rows[:-1]:
            if np.any(left[row + 3, symbol_col0 : symbol_col0 + 3] != BLACK):
                ok = False
                break
        if ok:
            candidates.append((seq_rows, palette, int(leftover.shape[0])))

    assert candidates, "failed to parse symbol stack and marker"
    candidates.sort(key=lambda item: (-len(item[0]), item[0][0]))
    best_rows, best_palette, run_len = candidates[0]
    assert sum(len(rows) == len(best_rows) and rows[0] == best_rows[0] for rows, _, _ in candidates) == 1
    return best_palette, run_len


def main(grid: np.ndarray) -> np.ndarray:
    """Decode stacked grey symbols into a palette and use the grey marker height as the per-color run length to fill the red frame interior column by column.

    Input structure:
    - The grid has a black background, one hollow red rectangular frame on the right, and a left strip made only of black and grey.
    - The left strip contains 2-4 stacked 3x3 grey symbol masks separated by one black row.
    - A separate vertical grey marker segment of height 2 or 3 appears above the symbols or immediately to their left.

    Rule:
    - Decode each 3x3 symbol mask through the task codebook to obtain an ordered palette from top to bottom.
    - Read the grey marker height as the run length.
    - Keep the red frame border, erase the left grey structures to black, and fill the frame interior column by column from top to bottom.
    - Within each column, repeat each palette color for `run length` cells, cycle through the palette, and restart from the first palette color at the top of every new column.

    Key insight:
    - The symbols say which colors to use, the marker says how long each color run lasts, and the frame interior says how many cells must be filled, so the answer is forced only after all three parts are combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_frame(g)
    palette, run_len = _extract_palette_and_run_len(g, c0)

    out = np.full_like(g, BLACK)
    out[g == RED] = RED
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    stripe = np.array([palette[(r // run_len) % len(palette)] for r in range(inner_h)], dtype=int)
    out[r0 + 1 : r1, c0 + 1 : c1] = stripe[:, None]
    assert out.shape == g.shape
    assert inner_w >= 6 and inner_h >= 6
    return out


def _draw_symbol(grid: np.ndarray, top: int, left: int, symbol_mask: tuple[tuple[int, ...], ...]) -> None:
    arr = np.array(symbol_mask, dtype=int)
    grid[top : top + 3, left : left + 3] = np.where(arr == 1, GREY, BLACK)


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = RED
    grid[bottom, left : right + 1] = RED
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED


def _expected_output(grid: np.ndarray, frame_top: int, frame_left: int, inner_h: int, inner_w: int, palette: list[int], run_len: int) -> np.ndarray:
    out = np.full_like(grid, BLACK)
    _draw_frame(out, frame_top, frame_left, inner_h, inner_w)
    stripe = np.array([palette[(r // run_len) % len(palette)] for r in range(inner_h)], dtype=int)
    out[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w] = stripe[:, None]
    return out


def _sample_inner_height(run_len: int) -> int:
    divisible = [h for h in range(6, 11) if h % run_len == 0]
    non_divisible = [h for h in range(6, 11) if h % run_len != 0]
    if non_divisible and _RNG.random() < 0.5:
        return int(_RNG.choice(non_divisible))
    return int(_RNG.choice(divisible if divisible else range(6, 11)))


def _generate_attempt() -> np.ndarray | None:
    run_len = int(_RNG.choice([2, 3]))
    symbol_count = int(_RNG.integers(2, 5))
    order = _RNG.choice(len(_SYMBOL_ITEMS), size=symbol_count, replace=False)
    symbols = [_SYMBOL_ITEMS[int(i)][0] for i in order]
    palette = [int(_SYMBOL_ITEMS[int(i)][1]) for i in order]

    inner_h = _sample_inner_height(run_len)
    inner_w = int(_RNG.integers(6, 11))
    marker_mode = "above" if _RNG.random() < 0.5 else "left"

    stack_h = 4 * symbol_count - 1
    if marker_mode == "above":
        symbol_left = int(_RNG.integers(0, 2))
        marker_col = symbol_left + int(_RNG.integers(0, 3))
        marker_top = 0
        symbol_top = run_len + 1
        left_used_w = symbol_left + 3
        left_used_h = symbol_top + stack_h
    else:
        symbol_left = 1
        symbol_top = int(_RNG.integers(0, 3))
        marker_col = 0
        marker_top = symbol_top
        left_used_w = 4
        left_used_h = max(symbol_top + stack_h, marker_top + run_len)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    total_h = max(left_used_h, outer_h) + int(_RNG.integers(0, 4))
    frame_top = int(_RNG.integers(0, total_h - outer_h + 1))
    frame_left = left_used_w + int(_RNG.integers(2, 6))
    total_w = frame_left + outer_w + int(_RNG.integers(0, 3))

    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    grid[marker_top : marker_top + run_len, marker_col] = GREY
    for idx, symbol_mask in enumerate(symbols):
        _draw_symbol(grid, symbol_top + 4 * idx, symbol_left, symbol_mask)
    _draw_frame(grid, frame_top, frame_left, inner_h, inner_w)

    expected = _expected_output(grid, frame_top, frame_left, inner_h, inner_w, palette, run_len)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
