from __future__ import annotations

import random

import numpy as np


BG = 0
LEFT_MARK = 9
RIGHT_MARK = 3

STRIPE_COLORS = (1, 2, 4, 5, 6, 7, 8)
GRID_WIDTH_RANGE = (7, 28)
STRIPE_COUNT_RANGE = (4, 7)
TOTAL_HEIGHT_MAX = 26
GEN_RETRIES = 300


def _row_ranges(heights: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for height in heights:
        end = start + int(height) - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _extract_stripes(grid: np.ndarray) -> tuple[list[dict[str, int]], dict[int, int]]:
    h, w = grid.shape
    assert w >= 3, "grid must have border columns plus interior"
    assert set(map(int, np.unique(grid[:, 0]))).issubset({BG, LEFT_MARK}), "unexpected colors on left border"
    assert set(map(int, np.unique(grid[:, -1]))).issubset({BG, RIGHT_MARK}), "unexpected colors on right border"

    stripes: list[dict[str, int]] = []
    row_to_stripe: dict[int, int] = {}
    r = 0
    while r < h:
        row = grid[r, 1:-1]
        vals = np.unique(row)
        assert len(vals) == 1, "each stripe row must be monochrome across the interior"
        color = int(vals[0])
        assert color != BG, "stripe rows cannot be background-colored"

        r0 = r
        while r + 1 < h and int(grid[r + 1, 1]) == color:
            next_vals = np.unique(grid[r + 1, 1:-1])
            assert len(next_vals) == 1 and int(next_vals[0]) == color, "stripe rows must stay contiguous and monochrome"
            r += 1
        r1 = r

        stripe_idx = len(stripes)
        stripes.append({"color": color, "height": r1 - r0 + 1})
        for rr in range(r0, r1 + 1):
            row_to_stripe[rr] = stripe_idx
        r += 1

    assert stripes, "at least one stripe is required"
    return stripes, row_to_stripe


def _marker_rows_to_stripes(rows: list[int], row_to_stripe: dict[int, int]) -> list[int]:
    stripe_ids = [int(row_to_stripe[int(r)]) for r in rows]
    assert len(stripe_ids) == len(set(stripe_ids)), "multiple markers cannot point to the same stripe"
    return stripe_ids


def _paint_output(stripes: list[dict[str, int]], width: int, left_ids: list[int], right_ids: list[int]) -> np.ndarray:
    total_height = sum(int(stripe["height"]) for stripe in stripes)
    out = np.full((total_height, width), BG, dtype=int)

    cursor_top = 0
    for idx in left_ids:
        height = int(stripes[idx]["height"])
        out[cursor_top : cursor_top + height, 1:-1] = int(stripes[idx]["color"])
        cursor_top += height

    cursor_bottom = total_height
    for idx in reversed(right_ids):
        height = int(stripes[idx]["height"])
        cursor_bottom -= height
        out[cursor_bottom : cursor_bottom + height, 1:-1] = int(stripes[idx]["color"])

    selected = set(left_ids) | set(right_ids)
    cursor_mid = cursor_top
    for idx, stripe in enumerate(stripes):
        if idx in selected:
            continue
        height = int(stripe["height"])
        out[cursor_mid : cursor_mid + height, 1:-1] = int(stripe["color"])
        cursor_mid += height

    assert cursor_mid == cursor_bottom, "middle stripes must exactly fill the remaining rows"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder horizontal color bands according to border markers, moving left-marked bands to the top and right-marked bands to the bottom.

    Input structure:
    - Background is black on the left and right border columns.
    - Every row belongs to one horizontal stripe that fills the interior columns, and each stripe spans one or more consecutive rows in a single color.
    - Maroon pixels on the left border select stripes; green pixels on the right border select other stripes.

    Rule:
    - Identify stripe row ranges from the monochrome interior rows.
    - Map each left-border maroon marker and right-border green marker to the stripe containing that row.
    - Place left-marked stripes at the top in marker top-to-bottom order, keep unmarked stripes in their original relative order in the middle, and place right-marked stripes at the bottom by stacking markers from bottom to top.
    - Preserve every stripe's height and color, and clear all border markers back to black.

    Key insight:
    - The marker row does not move by itself; it names the whole stripe that contains it, while the marker side determines whether that stripe is extracted upward or downward.
    """
    g = np.asarray(grid, dtype=int)
    stripes, row_to_stripe = _extract_stripes(g)

    left_rows = [int(r) for r in np.flatnonzero(g[:, 0] == LEFT_MARK)]
    right_rows = [int(r) for r in np.flatnonzero(g[:, -1] == RIGHT_MARK)]

    left_ids = _marker_rows_to_stripes(left_rows, row_to_stripe)
    right_ids = _marker_rows_to_stripes(right_rows, row_to_stripe)
    assert set(left_ids).isdisjoint(right_ids), "a stripe cannot be selected from both sides"

    return _paint_output(stripes, int(g.shape[1]), left_ids, right_ids)


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _choose_marker_groups(n_stripes: int, rng: random.Random) -> tuple[list[int], list[int]] | None:
    triples = [
        (left_count, right_count, n_stripes - left_count - right_count)
        for left_count in range(1, n_stripes - 1)
        for right_count in range(1, n_stripes - left_count)
        if n_stripes - left_count - right_count >= 1
    ]
    for _ in range(200):
        left_count, right_count, middle_count = rng.choice(triples)
        labels = ["L"] * left_count + ["R"] * right_count + ["M"] * middle_count
        rng.shuffle(labels)
        left_ids = [i for i, label in enumerate(labels) if label == "L"]
        right_ids = [i for i, label in enumerate(labels) if label == "R"]
        out_order = left_ids + [i for i, label in enumerate(labels) if label == "M"] + right_ids
        if out_order != list(range(n_stripes)):
            return left_ids, right_ids
    return None


def _build_input(
    colors: list[int],
    heights: list[int],
    width: int,
    left_ids: list[int],
    right_ids: list[int],
    rng: random.Random,
) -> np.ndarray:
    total_height = sum(heights)
    grid = np.full((total_height, width), BG, dtype=int)

    ranges = _row_ranges(heights)
    for idx, (r0, r1) in enumerate(ranges):
        grid[r0 : r1 + 1, 1:-1] = int(colors[idx])

    for idx in left_ids:
        r0, r1 = ranges[idx]
        grid[rng.randint(r0, r1), 0] = LEFT_MARK
    for idx in right_ids:
        r0, r1 = ranges[idx]
        grid[rng.randint(r0, r1), -1] = RIGHT_MARK

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_stripes = rng.randint(*STRIPE_COUNT_RANGE)
    total_height = rng.randint(n_stripes + 2, TOTAL_HEIGHT_MAX)
    heights = _random_composition(total_height, n_stripes, rng)
    width = rng.randint(*GRID_WIDTH_RANGE)
    colors = list(rng.sample(STRIPE_COLORS, n_stripes))

    marker_groups = _choose_marker_groups(n_stripes, rng)
    if marker_groups is None:
        return None
    left_ids, right_ids = marker_groups

    inp = _build_input(colors, heights, width, left_ids, right_ids, rng)
    stripes = [{"color": int(color), "height": int(height)} for color, height in zip(colors, heights)]
    expected = _paint_output(stripes, width, left_ids, right_ids)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
