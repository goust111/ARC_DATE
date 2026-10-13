from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _consecutive_top_depth(grid: np.ndarray, col: int, color: int, stop_row: int) -> int:
    depth = 0
    while depth < stop_row and int(grid[depth, col]) == int(color):
        depth += 1
    return int(depth)


def _scan_run(template: np.ndarray, row: int, col: int, step: int, bg: int) -> int:
    count = 0
    cc = int(col) + int(step)
    while 0 <= cc < template.shape[1] and int(template[row, cc]) == int(bg):
        count += 1
        cc += int(step)
    return int(count)


def _extract_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    base_r = h - 1
    base_color = int(g[base_r, 0])
    assert int(base_color) != int(BG)
    assert np.all(g[base_r, :] == int(base_color)), "expected a full-width baseline on the last row"

    depths = np.array(
        [_consecutive_top_depth(g, c, base_color, base_r) for c in range(w)],
        dtype=int,
    )
    assert np.all(depths >= 1), "expected a top-connected ceiling in every column"

    slots: list[dict[str, int]] = []
    for c in range(w):
        rows = [r for r in range(base_r) if int(g[r, c]) not in {int(BG), int(base_color)}]
        if not rows:
            continue
        color_vals = {int(g[r, c]) for r in rows}
        assert len(color_vals) == 1, "each input bar must be monochrome"
        height = len(rows)
        expected_rows = list(range(base_r - height, base_r))
        assert rows == expected_rows, "each input bar must be a contiguous vertical segment on the baseline"
        assert int(base_r - height) > int(depths[c]), "input bars must leave a black gap below the ceiling"
        slots.append({"col": int(c), "color": int(next(iter(color_vals))), "height": int(height)})

    assert len(slots) >= 3, "expected at least three bar slots"
    return {
        "grid": g,
        "h": int(h),
        "w": int(w),
        "base_r": int(base_r),
        "base_color": int(base_color),
        "depths": depths,
        "slots": sorted(slots, key=lambda slot: int(slot["col"])),
    }


def _erase_bars(scene) -> np.ndarray:
    template = scene["grid"].copy()
    base_r = int(scene["base_r"])
    for slot in scene["slots"]:
        c = int(slot["col"])
        height = int(slot["height"])
        template[base_r - height : base_r, c] = BG
    return template


def _render_sorted_output(scene) -> np.ndarray:
    slots = list(scene["slots"])
    shapes = sorted(
        (int(slot["height"]) for slot in slots),
        key=lambda height: int(height),
    )
    template = _erase_bars(scene)
    out = template.copy()
    base_r = int(scene["base_r"])
    depths = np.asarray(scene["depths"], dtype=int)

    for slot, height in zip(slots, shapes):
        c = int(slot["col"])
        color = int(slot["color"])
        clearance = int(base_r - depths[c])
        vertical = min(int(height), int(clearance))
        out[base_r - vertical : base_r, c] = int(color)

        overflow = int(height - vertical)
        if overflow <= 0:
            continue

        fold_r = int(depths[c])
        left_space = _scan_run(template, fold_r, c, -1, BG)
        right_space = _scan_run(template, fold_r, c, 1, BG)
        assert left_space != right_space, "fold direction must be uniquely determined"
        step = -1 if left_space > right_space else 1
        assert max(left_space, right_space) >= overflow, "folded tail must fit inside the empty side"

        for offset in range(1, overflow + 1):
            cc = int(c + step * offset)
            assert int(template[fold_r, cc]) == int(BG), "folded tail must stay inside black space"
            assert int(out[fold_r, cc]) == int(BG), "folded tails must not collide"
            out[fold_r, cc] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redistribute the vertical bars by height, then fold any overflow under the ceiling toward the side with more room.

    Input structure:
    - A full-width gray baseline spans the last row.
    - Several single-column monochrome bars of distinct colors stand on that baseline.
    - Every column also has a gray ceiling segment hanging down from the top edge, leaving black space below it.

    Rule:
    - Sort the bar heights from shortest to tallest, and sort the original bar columns from left to right.
    - Put the i-th shortest bar into the i-th leftmost slot, recoloring it to that slot's original color.
    - A reassigned bar grows upward from the baseline; if it reaches the slot's ceiling clearance, any remaining length turns horizontal on the row directly below that ceiling.
    - The folded part extends toward the side with the longer consecutive black run under that ceiling. The baseline and ceiling stay unchanged.

    Key insight:
    - Heights belong to the movable bar shapes, but clearances and colors belong to fixed slots, so the sort-and-reassign step must happen before deciding whether each bar stays vertical or folds.
    """
    scene = _extract_scene(grid)
    return _render_sorted_output(scene)


def _sample_spaced_columns(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    if count <= 0:
        return []
    if high - low + 1 < count:
        return None
    for _ in range(200):
        cols = sorted(int(c) for c in np.random.choice(np.arange(low, high + 1), size=count, replace=False))
        if all(cols[i + 1] - cols[i] >= min_gap for i in range(len(cols) - 1)):
            return cols
    return None


def _build_input_grid(base_r: int, depths: np.ndarray, slot_cols: list[int], slot_colors: list[int], input_heights: list[int]) -> np.ndarray:
    h = int(base_r + 1)
    grid = np.full((h, len(depths)), BG, dtype=int)
    for c, depth in enumerate(depths):
        grid[: int(depth), c] = GREY
    grid[base_r, :] = GREY

    for c, color, height in zip(slot_cols, slot_colors, input_heights):
        grid[base_r - height : base_r, c] = int(color)
    return grid


def _layout_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    k = int(np.random.randint(3, 6))
    c_small = int(np.random.randint(max(6, k + 1), 9))
    overflow = int(np.random.randint(1, 4))
    c_big = int(c_small + overflow + np.random.randint(1, 4))

    small_pool = np.arange(2, c_small)
    if len(small_pool) < k - 1:
        return None
    small_heights = sorted(int(v) for v in np.random.choice(small_pool, size=k - 1, replace=False))
    tall_height = int(c_small + overflow)
    sorted_heights = small_heights + [tall_height]

    base_r = int(c_big + np.random.randint(3, 8))
    h = int(base_r + 1)
    if h > 30:
        return None

    d_small = int(base_r - c_small)
    d_big = int(base_r - c_big)
    if d_big < 1 or d_small <= d_big:
        return None

    w = int(np.random.randint(max(12, 2 * k + 6), 25))
    last_col = int(np.random.randint(max(w // 2 + 1, k + 3), w - 2))
    earlier = _sample_spaced_columns(k - 1, 1, last_col - 2, 2)
    if earlier is None:
        return None
    slot_cols = earlier + [last_col]

    depths = np.random.randint(1, d_small + 1, size=w, dtype=int)
    for c in slot_cols[:-1]:
        depths[c] = int(d_big)
    depths[slot_cols[-1]] = int(d_small)

    slot_colors = [int(c) for c in np.random.choice(BAR_COLORS, size=k, replace=False)]
    last_input_height = int(np.random.choice(small_heights))
    remaining = list(small_heights)
    remaining.remove(last_input_height)
    big_slot_heights = remaining + [tall_height]
    np.random.shuffle(big_slot_heights)
    input_heights = big_slot_heights + [last_input_height]

    inp = _build_input_grid(base_r, depths, slot_cols, slot_colors, input_heights)
    scene = _extract_scene(inp)
    expected = _render_sorted_output(scene)

    fold_row = int(d_small)
    template = _erase_bars(scene)
    left_space = _scan_run(template, fold_row, slot_cols[-1], -1, BG)
    right_space = _scan_run(template, fold_row, slot_cols[-1], 1, BG)
    if left_space == right_space:
        return None

    overflow_amounts = [
        int(height > (scene["base_r"] - scene["depths"][slot["col"]]))
        for slot, height in zip(scene["slots"], sorted(sorted_heights))
    ]
    if sum(overflow_amounts) != 1:
        return None
    if np.array_equal(inp, expected):
        return None

    if np.random.rand() < 0.5:
        inp = np.fliplr(inp).copy()
        expected = np.fliplr(expected).copy()

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        attempt = _layout_attempt()
        if attempt is None:
            continue
        inp, expected = attempt

        try:
            actual = main(inp)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if inp.shape != expected.shape or max(inp.shape) > 30:
            continue

        scene = _extract_scene(inp)
        heights = [int(slot["height"]) for slot in scene["slots"]]
        if len(set(heights)) != len(heights):
            continue
        if sorted(heights) == heights:
            continue

        sorted_heights = sorted(heights)
        clearances = [int(scene["base_r"] - scene["depths"][slot["col"]]) for slot in scene["slots"]]
        overflow_flags = [int(height > clearance) for height, clearance in zip(sorted_heights, clearances)]
        if sum(overflow_flags) != 1:
            continue
        if not any(height <= clearance for height, clearance in zip(sorted_heights, clearances)):
            continue

        return inp

    raise RuntimeError("generate_input failed after repeated attempts")
