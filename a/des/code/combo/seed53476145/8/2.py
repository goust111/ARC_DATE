import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_MIN = 8
HEIGHT_MAX = 20
WIDTH_MIN = 10
WIDTH_MAX = 20
FENCE_COUNT_MIN = 2
FENCE_COUNT_MAX = 5
TOP_MARGIN_MAX = 2
BOTTOM_MARGIN_MAX = 2
CHANNEL_GAP_MIN = 1
CHANNEL_GAP_MAX = 4
PALETTE_SIZE_MIN = 3
PALETTE_SIZE_MAX = 5
GENERATE_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 100


def _fence_rows(grid: np.ndarray) -> list[int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r] != BLACK) and np.any(grid[r] == GREY)
    ]
    assert len(rows) >= 2, "expected at least two horizontal fence rows"

    non_fence = np.ones(grid.shape[0], dtype=bool)
    non_fence[rows] = False
    assert np.all(grid[non_fence] == BLACK), "corridors must be pure black in the input"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color vertical connections between adjacent grey fence rows.

    Input structure
    - The background is black.
    - Full-width grey horizontal fence rows divide the grid into corridors.
    - Fence rows may contain non-black, non-grey marker pixels.

    Rule
    - For each adjacent pair of fence rows, compare cells in the same column.
    - If both fence rows contain the same non-grey marker color in a column,
      fill the corridor cells between them with that color.
    - Leave fence rows and unmatched markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fences = _fence_rows(g)
    out = g.copy()

    for upper, lower in zip(fences, fences[1:]):
        assert upper < lower
        for col in range(g.shape[1]):
            top = int(g[upper, col])
            bottom = int(g[lower, col])
            if top in (BLACK, GREY):
                continue
            if top == bottom:
                out[upper + 1 : lower, col] = top

    return out


def _sample_layout() -> tuple[int, list[int]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        fence_count = random.randint(FENCE_COUNT_MIN, FENCE_COUNT_MAX)
        top_margin = random.randint(0, TOP_MARGIN_MAX)
        bottom_margin = random.randint(0, BOTTOM_MARGIN_MAX)
        gaps = [
            random.randint(CHANNEL_GAP_MIN, CHANNEL_GAP_MAX)
            for _ in range(fence_count - 1)
        ]
        height = top_margin + bottom_margin + fence_count + sum(gaps)
        if not (HEIGHT_MIN <= height <= HEIGHT_MAX):
            continue

        rows = [top_margin]
        for gap in gaps:
            rows.append(rows[-1] + gap + 1)
        return height, rows
    return None


def _channel_candidates(
    row_marks: list[dict[int, int]], row_idx: int, width: int
) -> list[int]:
    candidates: list[int] = []
    for col in range(width):
        if col in row_marks[row_idx] or col in row_marks[row_idx + 1]:
            continue
        if row_idx > 0 and col in row_marks[row_idx - 1]:
            continue
        if row_idx + 2 < len(row_marks) and col in row_marks[row_idx + 2]:
            continue
        candidates.append(col)
    return candidates


def _unpaired_candidates(
    row_marks: list[dict[int, int]], row_idx: int, width: int
) -> list[int]:
    candidates: list[int] = []
    for col in range(width):
        if col in row_marks[row_idx]:
            continue
        if row_idx > 0 and col in row_marks[row_idx - 1]:
            continue
        if row_idx + 1 < len(row_marks) and col in row_marks[row_idx + 1]:
            continue
        candidates.append(col)
    return candidates


def _add_required_pairs(
    row_marks: list[dict[int, int]], width: int, palette: tuple[int, ...]
) -> tuple[int, int] | None:
    pair_total = 0
    mismatch_total = 0

    for row_idx in range(len(row_marks) - 1):
        candidates = _channel_candidates(row_marks, row_idx, width)
        if not candidates:
            return None

        pair_cap = min(3 if width >= 15 else 2, len(candidates))
        if pair_cap < 1:
            return None
        pair_count = random.randint(1, pair_cap)
        pair_cols = random.sample(candidates, pair_count)
        for col in pair_cols:
            color = random.choice(palette)
            row_marks[row_idx][col] = color
            row_marks[row_idx + 1][col] = color
        pair_total += pair_count

        if random.random() < 0.55:
            mismatch_candidates = _channel_candidates(row_marks, row_idx, width)
            if mismatch_candidates:
                col = random.choice(mismatch_candidates)
                top_color, bottom_color = random.sample(palette, 2)
                row_marks[row_idx][col] = top_color
                row_marks[row_idx + 1][col] = bottom_color
                mismatch_total += 1

    return pair_total, mismatch_total


def _add_unpaired_markers(
    row_marks: list[dict[int, int]], width: int, palette: tuple[int, ...]
) -> int:
    all_candidates = [
        (row_idx, col)
        for row_idx in range(len(row_marks))
        for col in _unpaired_candidates(row_marks, row_idx, width)
    ]
    if not all_candidates:
        return 0

    row_idx, col = random.choice(all_candidates)
    row_marks[row_idx][col] = random.choice(palette)
    added = 1

    extra_target = random.randint(0, min(4, width // 4))
    for _ in range(extra_target):
        next_candidates = [
            (r, c)
            for r in range(len(row_marks))
            for c in _unpaired_candidates(row_marks, r, width)
        ]
        if not next_candidates:
            break
        r, c = random.choice(next_candidates)
        row_marks[r][c] = random.choice(palette)
        added += 1

    return added


def _grid_from_marks(
    height: int, width: int, fence_rows: list[int], row_marks: list[dict[int, int]]
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row in fence_rows:
        grid[row, :] = GREY
    for row_idx, row in enumerate(fence_rows):
        for col, color in row_marks[row_idx].items():
            grid[row, col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    height, fence_rows = layout
    width = random.randint(WIDTH_MIN, WIDTH_MAX)
    palette_size = random.randint(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX)
    palette = tuple(random.sample(MARKER_COLORS, palette_size))

    row_marks: list[dict[int, int]] = [dict() for _ in fence_rows]
    pair_data = _add_required_pairs(row_marks, width, palette)
    if pair_data is None:
        return None
    pair_total, mismatch_total = pair_data
    unpaired_total = _add_unpaired_markers(row_marks, width, palette)
    if pair_total < 1 or unpaired_total < 1:
        return None

    if any(len(marks) >= width for marks in row_marks):
        return None

    grid = _grid_from_marks(height, width, fence_rows, row_marks)
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if mismatch_total == 0 and random.random() < 0.35:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
