from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BAR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
COUNT_PAIRS = ((3, 4), (4, 3), (3, 5), (5, 3))
MAX_GENERATION_ATTEMPTS = 400


def _find_bar_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == BAR, axis=1))
    assert len(rows) == 1, "expected exactly one full-width gray bar"
    return int(rows[0])


def _extract_pixels(grid: np.ndarray, bar_row: int) -> list[dict[str, int]]:
    pixels: list[dict[str, int]] = []
    coords = np.argwhere((grid != BG) & (grid != BAR))
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        side = -1 if rr < bar_row else 1
        assert rr != bar_row, "colored pixels cannot lie on the bar row"
        pixels.append(
            {
                "row": rr,
                "col": cc,
                "color": int(grid[rr, cc]),
                "side": side,
                "dist": abs(rr - bar_row) - 1,
            }
        )
    return pixels


def _ordered_pixels(pixels: list[dict[str, int]], *, reverse_distance: bool = False) -> list[dict[str, int]]:
    side_rank = {-1: 0, 1: 1}
    return sorted(
        pixels,
        key=lambda pixel: (
            -pixel["dist"] if reverse_distance else pixel["dist"],
            side_rank[pixel["side"]],
            pixel["col"],
            pixel["row"],
            pixel["color"],
        ),
    )


def _settle_pixels(
    grid: np.ndarray,
    *,
    reverse_distance: bool = False,
    recolor_majority: bool = True,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bar_row = _find_bar_row(g)
    pixels = _ordered_pixels(_extract_pixels(g, bar_row), reverse_distance=reverse_distance)

    out = np.zeros_like(g)
    out[bar_row, :] = BAR
    occupied = {(bar_row, c) for c in range(w)}
    settled: list[dict[str, int]] = []

    for pixel in pixels:
        row = pixel["row"]
        col = pixel["col"]
        step = 1 if pixel["side"] < 0 else -1
        while 0 <= row + step < h and (row + step, col) not in occupied:
            row += step
        out[row, col] = pixel["color"]
        occupied.add((row, col))
        settled.append(
            {
                "row": row,
                "col": col,
                "color": pixel["color"],
                "side": pixel["side"],
                "dist": pixel["dist"],
            }
        )

    if recolor_majority:
        top_count = sum(pixel["side"] < 0 for pixel in settled)
        bottom_count = len(settled) - top_count
        if top_count > bottom_count:
            for pixel in settled:
                if pixel["side"] < 0:
                    out[pixel["row"], pixel["col"]] = BAR
        elif bottom_count > top_count:
            for pixel in settled:
                if pixel["side"] > 0:
                    out[pixel["row"], pixel["col"]] = BAR

    return out, settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide upper and lower loose pixels toward a full-width gray bar, then recolor the more crowded side gray.

    Input structure:
    - A black background contains exactly one gray horizontal bar spanning the full width and one cell in height.
    - Above and below the bar are isolated non-gray single pixels with distinct colors.
    - The pixels sit in the same columns they will keep after sliding; only their row positions change.

    Rule:
    - Find the full-width gray row R. Pixels above R slide downward; pixels below R slide upward.
    - Process pixels by increasing distance to the bar, where distance is `abs(row - R) - 1`; ties prefer the upper side.
    - A pixel keeps sliding vertically until the next step would hit the gray bar or a pixel that has already stopped.
    - After all pixels settle, count how many stopped above the bar and how many stopped below it.
    - Recolor every stopped pixel on the more populous side to gray; the less populous side keeps its original colors. The bar stays gray.

    Key insight:
    - The gray bar is both the destination and the first obstacle, so early arrivals claim the inner cells and force later pixels in the same column to stack farther away; only after that stacking is complete does the majority side get recolored.
    """
    out, _settled = _settle_pixels(grid, reverse_distance=False, recolor_majority=True)
    return out


def _sample_column_multiset(
    count: int,
    candidate_cols: list[int],
    *,
    need_duplicate: bool,
    forced_shared_col: int | None,
) -> list[int]:
    if need_duplicate:
        unique_count = count - 1
        chosen = random.sample(candidate_cols, unique_count)
        if forced_shared_col is not None and forced_shared_col not in chosen:
            replace_idx = random.randrange(unique_count)
            chosen[replace_idx] = forced_shared_col
        dup_col = forced_shared_col if forced_shared_col is not None else random.choice(chosen)
        cols = chosen + [dup_col]
    else:
        cols = random.sample(candidate_cols, count)
        if forced_shared_col is not None and forced_shared_col not in cols:
            replace_idx = random.randrange(count)
            cols[replace_idx] = forced_shared_col
    random.shuffle(cols)
    return cols


def _sample_distances(room_size: int, multiplicity: int) -> list[int]:
    if multiplicity == 1:
        choices = list(range(room_size))
        weights = [3 if d <= 1 or d >= room_size - 2 else 1 for d in choices]
        return [random.choices(choices, weights=weights, k=1)[0]]

    all_pairs = [
        (a, b)
        for a in range(room_size)
        for b in range(a + 2, room_size)
    ]
    informative_pairs = [pair for pair in all_pairs if pair[0] <= 1 and pair[1] >= room_size // 2]
    pair_pool = informative_pairs if informative_pairs else all_pairs
    pair = random.choice(pair_pool)
    return list(pair)


def _place_side(
    side: int,
    count: int,
    bar_row: int,
    grid_height: int,
    candidate_cols: list[int],
    *,
    need_duplicate: bool,
    forced_shared_col: int | None,
    colors: list[int],
) -> list[tuple[int, int, int]]:
    room_size = bar_row if side < 0 else grid_height - bar_row - 1
    cols = _sample_column_multiset(
        count,
        candidate_cols,
        need_duplicate=need_duplicate,
        forced_shared_col=forced_shared_col,
    )
    counts = Counter(cols)
    cells: list[tuple[int, int, int]] = []
    color_iter = iter(colors)
    for col, multiplicity in counts.items():
        distances = _sample_distances(room_size, multiplicity)
        random.shuffle(distances)
        for dist in distances:
            row = bar_row - 1 - dist if side < 0 else bar_row + 1 + dist
            cells.append((row, col, next(color_iter)))
    random.shuffle(cells)
    return cells


def _side_column_counts(grid: np.ndarray, bar_row: int, side: int) -> Counter[int]:
    mask = (grid != BG) & (grid != BAR)
    coords = np.argwhere(mask)
    cols = [int(c) for r, c in coords if (int(r) < bar_row) == (side < 0)]
    return Counter(cols)


def _pixels_are_isolated(grid: np.ndarray) -> bool:
    coords = np.argwhere((grid != BG) & (grid != BAR))
    pts = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in pts:
                    return False
    return True


def _validate_generated_case(grid: np.ndarray) -> bool:
    bar_row = _find_bar_row(grid)
    pixels = _extract_pixels(grid, bar_row)
    colors = [pixel["color"] for pixel in pixels]

    if len(colors) != len(set(colors)):
        return False
    if any(color in (BG, BAR) for color in colors):
        return False
    if not _pixels_are_isolated(grid):
        return False

    top_count = sum(pixel["side"] < 0 for pixel in pixels)
    bottom_count = len(pixels) - top_count
    if top_count == bottom_count:
        return False

    top_cols = _side_column_counts(grid, bar_row, -1)
    bottom_cols = _side_column_counts(grid, bar_row, 1)
    if max(top_cols.values(), default=0) < 2 and max(bottom_cols.values(), default=0) < 2:
        return False

    correct, settled = _settle_pixels(grid, reverse_distance=False, recolor_majority=True)
    before_recolor, settled_plain = _settle_pixels(grid, reverse_distance=False, recolor_majority=False)
    wrong_order, _ = _settle_pixels(grid, reverse_distance=True, recolor_majority=True)

    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, wrong_order):
        return False
    if np.array_equal(correct, before_recolor):
        return False
    if len(settled) != len(settled_plain):
        return False

    majority_side = -1 if top_count > bottom_count else 1
    for pixel in settled:
        expected = BAR if pixel["side"] == majority_side else pixel["color"]
        if int(correct[pixel["row"], pixel["col"]]) != expected:
            return False

    minority_cells = sum(int(correct[pixel["row"], pixel["col"]]) != BAR for pixel in settled if pixel["side"] != majority_side)
    if minority_cells == 0:
        return False

    return True


def _generate_case() -> np.ndarray:
    height = random.randint(14, 18)
    width = random.randint(12, 18)
    bar_row = random.randint(5, height - 6)
    top_count, bottom_count = random.choice(COUNT_PAIRS)
    colors = random.sample(PALETTE, top_count + bottom_count)
    top_colors = colors[:top_count]
    bottom_colors = colors[top_count:]

    candidate_cols = list(range(1, width - 1, 2))
    assert len(candidate_cols) >= 5

    duplicate_mode = random.choice(("top", "bottom", "both"))
    share_column = random.random() < 0.65
    shared_col = random.choice(candidate_cols) if share_column else None

    top_cells = _place_side(
        -1,
        top_count,
        bar_row,
        height,
        candidate_cols,
        need_duplicate=duplicate_mode in ("top", "both"),
        forced_shared_col=shared_col if share_column and random.random() < 0.8 else None,
        colors=top_colors,
    )
    bottom_cells = _place_side(
        1,
        bottom_count,
        bar_row,
        height,
        candidate_cols,
        need_duplicate=duplicate_mode in ("bottom", "both"),
        forced_shared_col=shared_col if share_column and random.random() < 0.8 else None,
        colors=bottom_colors,
    )

    grid = np.zeros((height, width), dtype=int)
    grid[bar_row, :] = BAR
    for row, col, color in top_cells + bottom_cells:
        grid[row, col] = color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if _validate_generated_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid sliding-majority case")
