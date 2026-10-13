import random

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 9]

_GEN_CALL_COUNT = 0
_MAX_ATTEMPTS = 200


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _parse_structure(grid: np.ndarray) -> tuple[list[int], int, list[int], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    sep_candidates = [c for c in range(1, w - 1) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != BLACK]
    assert len(sep_candidates) == 1
    sep_col = sep_candidates[0]
    assert sep_col == 1
    bar_color = int(g[0, sep_col])

    palette: list[int] = []
    seen_black = False
    for r in range(h):
        value = int(g[r, 0])
        if value == BLACK:
            seen_black = True
            continue
        assert not seen_black
        assert value != bar_color
        palette.append(value)
    assert len(palette) >= 2

    top_main = g[0, sep_col + 1 :]
    token_colors = {int(v) for v in top_main if int(v) not in (BLACK, bar_color)}
    assert len(token_colors) == 1
    token_color = next(iter(token_colors))
    token_cols = [sep_col + 1 + c for c in np.flatnonzero(top_main == token_color)]
    assert len(token_cols) >= 2

    bars: list[tuple[int, int, int]] = []
    for r in range(1, h):
        row = g[r, sep_col + 1 :]
        stray = {int(v) for v in row if int(v) not in (BLACK, bar_color)}
        assert not stray
        if not np.any(row == bar_color):
            continue
        gaps = _runs(row != bar_color)
        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert np.all(row[gap_start : gap_end + 1] == BLACK)
        bars.append((r, gap_start, gap_end))

    assert bars
    assert bars[-1][0] < h - 1
    assert all(bars[i][0] > bars[i - 1][0] for i in range(1, len(bars)))
    return palette, sep_col, token_cols, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top token through dashed bars and recolor it by the number of wide gaps it traverses.

    Input structure:
    - The leftmost column is a top-down palette of colored cells followed by black cells, and the next column is a full-height separator.
    - To the right of the separator is the main field: the top row contains several same-colored tokens, and lower rows may contain horizontal bars in the separator's color.
    - Each bar spans the full main field except for one contiguous gap.

    Rule:
    - Each token falls straight down in its own column.
    - At a bar row, the token passes only if it sits inside that bar's gap and the gap width is at least 2; bar cells and width-1 gaps both stop the token one row above the bar.
    - A token that passes k bars is recolored with the (k+1)-th palette color, its start cell is cleared to black, and if nothing stops it then it lands on the bottom row.

    Key insight:
    - The visible hole is not always open: single-cell gaps behave like sealed plugs, so the token color is determined by how many multi-cell gaps it truly crosses.
    """
    g = np.asarray(grid, dtype=int)
    palette, sep_col, token_cols, bars = _parse_structure(g)
    h, _w = g.shape
    main_start = sep_col + 1

    out = g.copy()
    out[0, token_cols] = BLACK

    for col in token_cols:
        local_col = col - main_start
        passed = 0
        landing_row = h - 1
        for bar_row, gap_start, gap_end in bars:
            gap_width = gap_end - gap_start + 1
            if gap_width >= 2 and gap_start <= local_col <= gap_end:
                passed += 1
                continue
            landing_row = bar_row - 1
            break
        assert passed < len(palette)
        assert out[landing_row, col] == BLACK
        out[landing_row, col] = palette[passed]

    return out


def _make_nested_gaps(main_width: int, wide_count: int, mirror: bool) -> tuple[list[int], list[tuple[int, int]], int]:
    token_cols = sorted(random.sample(range(main_width - 1), wide_count + 1))
    right_edge = token_cols[-1] + 1
    wide_gaps = [(token_cols[i], right_edge) for i in range(1, wide_count + 1)]
    stopper_col = token_cols[-1]

    if mirror:
        token_cols = [main_width - 1 - c for c in token_cols]
        wide_gaps = [(main_width - 1 - b, main_width - 1 - a) for a, b in wide_gaps]
        stopper_col = main_width - 1 - stopper_col

    return token_cols, wide_gaps, stopper_col


def _build_case(has_stopper: bool) -> np.ndarray:
    wide_count = random.randint(2, 5)
    total_bars = wide_count + int(has_stopper)
    palette_len = wide_count + 1

    main_width = random.randint(max(8, wide_count + 4), min(18, wide_count + 9))
    mirror = bool(random.getrandbits(1))
    token_cols, wide_gaps, stopper_col = _make_nested_gaps(main_width, wide_count, mirror)

    first_bar = random.randint(2, 4)
    bar_rows = [first_bar]
    for _ in range(1, total_bars):
        bar_rows.append(bar_rows[-1] + random.randint(2, 4))

    height = bar_rows[-1] + random.randint(2, 5)
    width = main_width + 2
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    palette = random.sample(PALETTE_POOL, palette_len)
    grid[:palette_len, 0] = palette
    grid[:, 1] = GREY

    for idx, row in enumerate(bar_rows):
        grid[row, 2:] = GREY
        if idx < wide_count:
            gap_start, gap_end = wide_gaps[idx]
            grid[row, 2 + gap_start : 2 + gap_end + 1] = BLACK
        else:
            grid[row, 2 + stopper_col] = BLACK

    for local_col in token_cols:
        grid[0, 2 + local_col] = TEAL

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    has_stopper = (_GEN_CALL_COUNT % 2) == 0
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_case(has_stopper=has_stopper)
        try:
            out = main(grid)
        except AssertionError:
            continue

        main_area = out[:, 2:]
        token_mask = (main_area != BLACK) & (main_area != GREY)
        if int(np.count_nonzero(token_mask)) != int(np.count_nonzero(grid[0, 2:] == TEAL)):
            continue

        _GEN_CALL_COUNT += 1
        return grid

    raise RuntimeError("failed to generate valid input for task 3")
