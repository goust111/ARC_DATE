import itertools
import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
ROW_BLOCK_COUNT_RANGE = (2, 4)
COL_BLOCK_COUNT_RANGE = (2, 4)
BLOCK_SIZE_RANGE = (2, 4)
SEPARATOR_RANGE = (1, 3)
PADDING_RANGE = (1, 4)
INPUT_SIDE_MIN = 13
INPUT_SIDE_MAX = 28
FILLER_COUNT_CHOICES = (2, 3, 4)
COLOR_POOL = tuple(range(10))


def _colors_present(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid)]


def _ordered_colors(grid: np.ndarray) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in map(int, grid.ravel()):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _find_target_pair(
    grid: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[tuple[int, int], np.ndarray, np.ndarray, np.ndarray]] = []

    for a, b in itertools.combinations(_colors_present(g), 2):
        mask = (g == a) | (g == b)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        if not rows.any() or not cols.any():
            continue
        if int(mask.sum()) != int(rows.sum()) * int(cols.sum()):
            continue
        cropped = g[np.ix_(rows, cols)]
        if set(map(int, np.unique(cropped))) != {int(a), int(b)}:
            continue
        candidates.append(((int(a), int(b)), rows, cols, cropped.copy()))

    assert len(candidates) == 1, "expected exactly one valid row-column product pair"
    return candidates[0]


def _rank_context_colors(grid: np.ndarray, target_pair: tuple[int, int]) -> list[int]:
    counts = Counter(map(int, grid.ravel()))
    first_seen = {color: idx for idx, color in enumerate(_ordered_colors(grid))}
    fillers = [color for color in first_seen if color not in target_pair]
    assert len(fillers) >= 2, "expected at least two non-target colors"

    ranked = sorted(fillers, key=lambda color: (-counts[color], first_seen[color]))
    return [ranked[i % len(ranked)] for i in range(4)]


def _add_quadrant_border(core: np.ndarray, border_colors: list[int]) -> np.ndarray:
    h, w = core.shape
    out_h, out_w = h + 4, w + 4
    out = np.empty((out_h, out_w), dtype=int)
    out[2 : 2 + h, 2 : 2 + w] = core

    for r in range(out_h):
        for c in range(out_w):
            if 2 <= r < h + 2 and 2 <= c < w + 2:
                continue
            if r < 2:
                out[r, c] = border_colors[0 if c < out_w // 2 else 1]
            elif r >= out_h - 2:
                out[r, c] = border_colors[2 if c < out_w // 2 else 3]
            elif c < 2:
                out[r, c] = border_colors[0 if r < out_h // 2 else 2]
            else:
                out[r, c] = border_colors[1 if r < out_h // 2 else 3]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique two-color row-column product image and surround it with a frequency-coded quadrant border.

    Input structure:
    - A medium multicolor grid contains exactly one pair of colors whose cells fill the full cross product of their active rows and active columns.
    - Other colors occupy only the rows or columns excluded from that hidden product image, acting as filler or separators.
    - The filler colors have distinct frequencies, and black is treated like any other filler color when it is present but not part of the target pair.

    Rule:
    - Enumerate all color pairs and find the unique pair whose union occupies every cell in its touched rows-by-columns rectangle.
    - Crop the grid to those active rows and active columns, preserving the resulting compact two-color subgrid.
    - Count all non-target colors, rank them by descending frequency, cycle the ranked list to four entries if needed, and use those four colors as a two-cell border split into upper-left, upper-right, lower-left, and lower-right quadrants.

    Key insight:
    - The valid content comes from the only color pair with a complete row-column product mask, while the discarded context colors still matter because their frequencies determine the border decoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    target_pair, _rows, _cols, core = _find_target_pair(g)
    border_colors = _rank_context_colors(g, target_pair)
    return _add_quadrant_border(core, border_colors)


def _make_target_core(height: int, width: int, colors: tuple[int, int]) -> np.ndarray:
    bg, fg = colors
    for _ in range(100):
        core = np.full((height, width), bg, dtype=int)
        for row in range(height):
            core[row, random.randrange(width)] = fg

        missing_cols = [c for c in range(width) if not np.any(core[:, c] == fg)]
        random.shuffle(missing_cols)
        for missing_col in missing_cols:
            core[random.randrange(height), missing_col] = fg

        for row in range(height):
            if random.random() < 0.4:
                anchors = np.flatnonzero(core[row] == fg)
                anchor = int(random.choice(anchors.tolist()))
                direction = random.choice((-1, 1))
                span = random.randint(1, min(3, width - 1))
                end = min(width - 1, max(0, anchor + direction * span))
                lo, hi = sorted((anchor, end))
                core[row, lo : hi + 1] = fg

        fg_count = int(np.count_nonzero(core == fg))
        if fg_count < max(height, width):
            continue
        if fg_count > max((height * width) // 2, max(height, width) + 2):
            continue
        if all(np.any(core[row] == fg) for row in range(height)) and all(
            np.any(core[:, col] == fg) for col in range(width)
        ):
            return core

    raise RuntimeError("failed to generate target two-color core")


def _choose_layout() -> tuple[int, int, list[int], list[int]]:
    for _ in range(200):
        row_blocks = random.randint(*ROW_BLOCK_COUNT_RANGE)
        col_blocks = random.randint(*COL_BLOCK_COUNT_RANGE)
        row_sizes = [random.randint(*BLOCK_SIZE_RANGE) for _ in range(row_blocks)]
        col_sizes = [random.randint(*BLOCK_SIZE_RANGE) for _ in range(col_blocks)]
        row_sep = random.randint(*SEPARATOR_RANGE)
        col_sep = random.randint(*SEPARATOR_RANGE)
        pad_top = random.randint(*PADDING_RANGE)
        pad_bottom = random.randint(*PADDING_RANGE)
        pad_left = random.randint(*PADDING_RANGE)
        pad_right = random.randint(*PADDING_RANGE)

        height = sum(row_sizes) + row_sep * (row_blocks - 1) + pad_top + pad_bottom
        width = sum(col_sizes) + col_sep * (col_blocks - 1) + pad_left + pad_right
        if not (INPUT_SIDE_MIN <= height <= INPUT_SIDE_MAX):
            continue
        if not (INPUT_SIDE_MIN <= width <= INPUT_SIDE_MAX):
            continue

        rows: list[int] = []
        cursor = pad_top
        for size in row_sizes:
            rows.extend(range(cursor, cursor + size))
            cursor += size + row_sep

        cols: list[int] = []
        cursor = pad_left
        for size in col_sizes:
            cols.extend(range(cursor, cursor + size))
            cursor += size + col_sep

        return height, width, rows, cols

    raise RuntimeError("failed to choose a valid input layout")


def _choose_palette() -> tuple[tuple[int, int], list[int]]:
    filler_count = random.choice(FILLER_COUNT_CHOICES)
    non_black = list(range(1, 10))
    target_pair = tuple(random.sample(non_black, 2))

    fillers: list[int]
    if random.random() < 0.5:
        others = [color for color in non_black if color not in target_pair]
        fillers = [0] + random.sample(others, filler_count - 1)
    else:
        others = [color for color in non_black if color not in target_pair]
        fillers = random.sample(others, filler_count)

    random.shuffle(fillers)
    return target_pair, fillers


def _outside_positions(
    height: int, width: int, active_rows: np.ndarray, active_cols: np.ndarray
) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(height)
        for c in range(width)
        if not (active_rows[r] and active_cols[c])
    ]


def _fill_context(
    grid: np.ndarray, active_rows: np.ndarray, active_cols: np.ndarray, fillers: list[int]
) -> bool:
    height, width = grid.shape
    outside = _outside_positions(height, width, active_rows, active_cols)
    if len(outside) < len(fillers) * 3:
        return False

    offset = random.randrange(len(fillers))
    for r, c in outside:
        grid[r, c] = fillers[(r + 2 * c + offset) % len(fillers)]

    random.shuffle(outside)
    counts = Counter(map(int, grid[~(active_rows[:, None] & active_cols[None, :])].ravel()))
    for i in range(len(fillers) - 2, -1, -1):
        cur = fillers[i]
        nxt = fillers[i + 1]
        while counts[cur] <= counts[nxt]:
            donor_index = None
            for idx, (r, c) in enumerate(outside):
                donor = int(grid[r, c])
                if donor != cur and counts[donor] > 1:
                    donor_index = idx
                    break
            if donor_index is None:
                return False
            r, c = outside.pop(donor_index)
            donor = int(grid[r, c])
            grid[r, c] = cur
            counts[cur] += 1
            counts[donor] -= 1

    final_counts = [counts[color] for color in fillers]
    return len(set(final_counts)) == len(final_counts)


def _build_input() -> np.ndarray | None:
    height, width, row_ids, col_ids = _choose_layout()
    target_pair, fillers = _choose_palette()

    core = _make_target_core(len(row_ids), len(col_ids), target_pair)
    grid = np.empty((height, width), dtype=int)
    active_rows = np.zeros(height, dtype=bool)
    active_cols = np.zeros(width, dtype=bool)
    active_rows[row_ids] = True
    active_cols[col_ids] = True

    if not _fill_context(grid, active_rows, active_cols, fillers):
        return None

    grid[np.ix_(row_ids, col_ids)] = core

    expected = _add_quadrant_border(core, _rank_context_colors(grid, target_pair))
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
