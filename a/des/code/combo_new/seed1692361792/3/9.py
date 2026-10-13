import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GEN_RETRIES = 400
GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (12, 30)
SEPARATOR_MARGIN = 4
MIN_BOTTOM_ROWS = 5
SCENARIO_TYPES = ("all_match", "extra_keys", "extra_locks", "both_extra")
MAX_KEY_OR_LOCKS = 4
WIDTH_CHOICES = (1, 1, 2, 2, 3)
MIN_GAP = 1
EDGE_PAD = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match upper colored key stripes to lower lock gaps of the same height and transfer the key colors into those gaps.

    Input structure:
    - The background is black.
    - One full horizontal separator row splits the grid into an upper key region and a lower lock region.
    - Above the separator are several vertical colored stripes touching the separator from above; each stripe has a unique color and a measurable height.
    - Below the separator are several vertical stripes in the separator color; each stripe reaches the bottom edge but contains one continuous black gap.

    Rule:
    - Measure the height of every upper key stripe.
    - Measure the height of every lower lock gap.
    - Pair upper keys and lower gaps one-to-one whenever their heights are equal.
    - Fill each matched gap with the paired key color, erase all upper keys to black, and leave unmatched locks plus the separator unchanged.

    Key insight:
    - The same visible quantity, height, is encoded in two different roles: as solid colored bars above and as missing segments below, so the solver must first detect the role change and only then match by height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_row, lock_color = _find_separator_row(g)
    keys = _find_keys(g, sep_row, lock_color)
    locks = _find_locks(g, sep_row, lock_color)

    out = g.copy()
    for key in keys:
        for r, c in key["cells"]:
            out[r, c] = BG

    keys_by_height: dict[int, list[dict]] = {}
    locks_by_height: dict[int, list[dict]] = {}
    for key in sorted(keys, key=lambda item: (item["left"], item["color"])):
        keys_by_height.setdefault(int(key["height"]), []).append(key)
    for lock in sorted(locks, key=lambda item: item["left"]):
        locks_by_height.setdefault(int(lock["gap_height"]), []).append(lock)

    for height in sorted(set(keys_by_height) & set(locks_by_height)):
        for key, lock in zip(keys_by_height[height], locks_by_height[height]):
            r0, r1 = lock["gap_rows"]
            c0, c1 = lock["cols"]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(key["color"])

    return out


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) != BG and np.all(row == row[0]):
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full non-background separator row"
    sep_row = int(candidates[0])
    assert 0 < sep_row < grid.shape[0] - 1, "separator must split the grid into upper and lower regions"
    return sep_row, int(grid[sep_row, 0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_keys(grid: np.ndarray, sep_row: int, lock_color: int) -> list[dict]:
    top = grid[:sep_row]
    mask = (top != BG) & (top != lock_color)
    comps = _components(mask)
    assert comps, "expected at least one key stripe above the separator"

    keys = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        colors = {int(top[r, c]) for r, c in comp}
        assert len(colors) == 1, "each key stripe should be monochrome"
        assert max(rows) == sep_row - 1, "each key must touch the separator from above"
        keys.append(
            {
                "color": int(next(iter(colors))),
                "height": int(max(rows) - min(rows) + 1),
                "left": int(min(cols)),
                "cells": [(int(r), int(c)) for r, c in comp],
            }
        )
    return keys


def _true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(flags) - 1)))
    return runs


def _find_locks(grid: np.ndarray, sep_row: int, lock_color: int) -> list[dict]:
    bottom = grid[sep_row + 1 :]
    assert bottom.shape[0] >= 3, "expected room for a lock gap below the separator"

    lock_cols = np.any(bottom == lock_color, axis=0).tolist()
    intervals = _true_runs(lock_cols)
    assert intervals, "expected at least one lower lock stripe"

    locks = []
    for c0, c1 in intervals:
        row_is_gap = []
        for local_r in range(bottom.shape[0]):
            segment = bottom[local_r, c0 : c1 + 1]
            if np.all(segment == lock_color):
                row_is_gap.append(False)
            else:
                assert np.all(segment == BG), "lock rows must be fully present or fully absent"
                row_is_gap.append(True)
        assert not row_is_gap[0] and not row_is_gap[-1], "lock gaps must be internal"
        gap_runs = [(a, b) for a, b in _true_runs(row_is_gap) if a > 0 and b < bottom.shape[0] - 1]
        assert len(gap_runs) == 1, "each lock must contain exactly one internal gap"
        gap0, gap1 = gap_runs[0]
        locks.append(
            {
                "left": int(c0),
                "cols": (int(c0), int(c1)),
                "gap_rows": (int(sep_row + 1 + gap0), int(sep_row + 1 + gap1)),
                "gap_height": int(gap1 - gap0 + 1),
            }
        )
    return locks


def _choose_counts(top_h: int, bottom_h: int) -> tuple[str, int, int, int] | None:
    max_shared = min(MAX_KEY_OR_LOCKS, top_h, bottom_h - 2)
    if max_shared < 1:
        return None

    scenario = random.choice(SCENARIO_TYPES)
    if scenario == "all_match":
        matched = random.randint(1, max_shared)
        return scenario, matched, matched, matched

    if scenario == "extra_keys":
        matched = random.randint(1, min(3, max_shared))
        extra_keys = random.randint(1, min(2, MAX_KEY_OR_LOCKS - matched))
        return scenario, matched + extra_keys, matched, matched

    if scenario == "extra_locks":
        matched = random.randint(1, min(3, max_shared))
        extra_locks = random.randint(1, min(2, MAX_KEY_OR_LOCKS - matched))
        return scenario, matched, matched + extra_locks, matched

    matched = random.randint(1, min(2, max_shared))
    extra_keys = random.randint(1, min(2, MAX_KEY_OR_LOCKS - matched))
    extra_locks = random.randint(1, min(2, MAX_KEY_OR_LOCKS - matched))
    return scenario, matched + extra_keys, matched + extra_locks, matched


def _choose_heights(
    top_h: int, bottom_h: int, key_count: int, lock_count: int, matched_count: int
) -> tuple[list[int], list[int], set[int]] | None:
    key_pool = list(range(1, top_h + 1))
    lock_pool = list(range(1, bottom_h - 1))
    shared_pool = list(range(1, min(top_h, bottom_h - 2) + 1))
    if len(key_pool) < key_count or len(lock_pool) < lock_count or len(shared_pool) < matched_count:
        return None

    matched = set(random.sample(shared_pool, matched_count))

    key_extra_need = key_count - matched_count
    key_extra_pool = [height for height in key_pool if height not in matched]
    if len(key_extra_pool) < key_extra_need:
        return None
    key_extra = set(random.sample(key_extra_pool, key_extra_need))

    lock_extra_need = lock_count - matched_count
    lock_extra_pool = [height for height in lock_pool if height not in matched and height not in key_extra]
    if len(lock_extra_pool) < lock_extra_need:
        return None
    lock_extra = set(random.sample(lock_extra_pool, lock_extra_need))

    key_heights = list(matched | key_extra)
    lock_heights = list(matched | lock_extra)
    random.shuffle(key_heights)
    random.shuffle(lock_heights)
    return key_heights, lock_heights, matched


def _random_partition(total: int, buckets: int) -> list[int]:
    parts = [0] * buckets
    for _ in range(total):
        parts[random.randrange(buckets)] += 1
    return parts


def _place_intervals(width: int, stripe_widths: list[int]) -> list[tuple[int, int]] | None:
    count = len(stripe_widths)
    required = sum(stripe_widths) + (count - 1) * MIN_GAP + 2 * EDGE_PAD
    if required > width:
        return None
    slack = width - required
    extras = _random_partition(slack, count + 1)

    intervals = []
    col = EDGE_PAD + extras[0]
    for idx, stripe_w in enumerate(stripe_widths):
        c0 = int(col)
        c1 = int(col + stripe_w - 1)
        intervals.append((c0, c1))
        col = c1 + 1
        if idx < count - 1:
            col += MIN_GAP + extras[idx + 1]
    return intervals


def _choose_widths(width: int, count: int) -> tuple[list[int], list[tuple[int, int]]] | None:
    for _ in range(80):
        stripe_widths = [int(random.choice(WIDTH_CHOICES)) for _ in range(count)]
        intervals = _place_intervals(width, stripe_widths)
        if intervals is not None:
            return stripe_widths, intervals
    return None


def _render_scene(
    shape: tuple[int, int],
    sep_row: int,
    lock_color: int,
    key_specs: list[dict],
    lock_specs: list[dict],
    matched: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = int(lock_color)

    for spec in key_specs:
        c0, c1 = spec["cols"]
        key_h = int(spec["height"])
        grid[sep_row - key_h : sep_row, c0 : c1 + 1] = int(spec["color"])

    for spec in lock_specs:
        c0, c1 = spec["cols"]
        gap_start = int(spec["gap_start"])
        gap_height = int(spec["gap_height"])
        grid[sep_row + 1 :, c0 : c1 + 1] = int(lock_color)
        grid[gap_start : gap_start + gap_height, c0 : c1 + 1] = BG

    out = grid.copy()
    by_key_height = {int(spec["height"]): spec for spec in key_specs}
    by_lock_height = {int(spec["gap_height"]): spec for spec in lock_specs}

    for spec in key_specs:
        c0, c1 = spec["cols"]
        key_h = int(spec["height"])
        out[sep_row - key_h : sep_row, c0 : c1 + 1] = BG

    for height_value in matched:
        lock = by_lock_height[int(height_value)]
        key = by_key_height[int(height_value)]
        c0, c1 = lock["cols"]
        gap_start = int(lock["gap_start"])
        gap_height = int(lock["gap_height"])
        out[gap_start : gap_start + gap_height, c0 : c1 + 1] = int(key["color"])

    return grid, out


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    if height < SEPARATOR_MARGIN + MIN_BOTTOM_ROWS + 1:
        return None

    sep_row = random.randint(SEPARATOR_MARGIN, height - MIN_BOTTOM_ROWS - 1)
    top_h = int(sep_row)
    bottom_h = int(height - sep_row - 1)
    counts = _choose_counts(top_h, bottom_h)
    if counts is None:
        return None
    _scenario, key_count, lock_count, matched_count = counts

    heights = _choose_heights(top_h, bottom_h, key_count, lock_count, matched_count)
    if heights is None:
        return None
    key_heights, lock_heights, matched = heights

    key_layout = _choose_widths(width, key_count)
    lock_layout = _choose_widths(width, lock_count)
    if key_layout is None or lock_layout is None:
        return None
    _, key_intervals = key_layout
    _, lock_intervals = lock_layout

    lock_color = random.randint(1, 9)
    key_colors = random.sample([color for color in range(1, 10) if color != lock_color], key_count)

    key_specs = []
    for key_h, color, cols in zip(key_heights, key_colors, key_intervals, strict=False):
        key_specs.append(
            {
                "height": int(key_h),
                "color": int(color),
                "cols": (int(cols[0]), int(cols[1])),
            }
        )

    random.shuffle(lock_heights)
    lock_specs = []
    for gap_h, cols in zip(lock_heights, lock_intervals, strict=False):
        top_piece = random.randint(1, bottom_h - gap_h - 1)
        gap_start = int(sep_row + 1 + top_piece)
        lock_specs.append(
            {
                "gap_height": int(gap_h),
                "gap_start": int(gap_start),
                "cols": (int(cols[0]), int(cols[1])),
            }
        )

    inp, expected = _render_scene((height, width), sep_row, lock_color, key_specs, lock_specs, matched)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
