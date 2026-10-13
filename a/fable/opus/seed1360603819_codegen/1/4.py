from __future__ import annotations

import random
from collections import deque

import numpy as np


MASK_COLOR = 9
RESULT_COLOR_BY_COUNT = {1: 4, 2: 2, 3: 5}
BASE_COLORS = (1, 3, 6, 7, 8)

GENERATION_RETRIES = 500
MIN_GRID_SIDE = 8
MAX_GRID_SIDE = 18
BLOCK_SIDE_OPTIONS = (2, 3)
MAX_INFERRED_PERIOD = 6
MAX_RECT_SIDE = 6
MAX_RECT_AREA = 20
MAX_MASK_FRACTION = 0.35
EXTRA_RECT_OPTIONS = (0, 1, 2)
SHAPES_BY_COLOR_COUNT = {
    2: ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)),
    3: ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)),
}


def _candidate_tile(grid: np.ndarray, mask: np.ndarray, period_r: int, period_c: int) -> np.ndarray | None:
    seen: dict[tuple[int, int], int] = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if mask[r, c]:
                continue
            key = (r % period_r, c % period_c)
            color = int(grid[r, c])
            prev = seen.get(key)
            if prev is None:
                seen[key] = color
            elif prev != color:
                return None

    if len(seen) != period_r * period_c:
        return None

    tile = np.empty((period_r, period_c), dtype=int)
    for (r, c), color in seen.items():
        tile[r, c] = color

    colors = np.unique(tile)
    if MASK_COLOR in colors:
        return None
    if not (2 <= colors.size <= 3):
        return None
    return tile


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mask = g == MASK_COLOR
    height, width = g.shape

    periods = [(pr * pc, pr, pc) for pr in range(1, height + 1) for pc in range(1, width + 1)]
    periods.sort()

    for _area, period_r, period_c in periods:
        tile = _candidate_tile(g, mask, period_r, period_c)
        if tile is not None:
            return tile

    raise AssertionError("failed to infer a unique periodic tile")


def _paint_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    period_r, period_c = tile.shape
    rows = np.arange(shape[0]) % period_r
    cols = np.arange(shape[1]) % period_c
    return tile[rows[:, None], cols]


def _mask_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rectangles: list[tuple[int, int, int, int]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert r1 - r0 + 1 > 1 and c1 - c0 + 1 > 1, "mask components must be rectangles larger than 1x1"
            assert bool(mask[r0 : r1 + 1, c0 : c1 + 1].all()), "mask component must be solid"
            rectangles.append((r0, c0, r1, c1))

    rectangles.sort()
    return rectangles


def _covered_color_count(tile: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    period_r, period_c = tile.shape
    r0, c0, r1, c1 = rect
    rows = np.arange(r0, r1 + 1) % period_r
    cols = np.arange(c0, c1 + 1) % period_c
    return int(np.unique(tile[rows[:, None], cols]).size)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tile = _infer_tile(g)
    rectangles = _mask_rectangles(g == MASK_COLOR)
    assert rectangles, "expected at least one maroon rectangle"

    out = g.copy()
    for rect in rectangles:
        count = _covered_color_count(tile, rect)
        assert count in RESULT_COLOR_BY_COUNT, "covered color count must be in 1..3"
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = RESULT_COLOR_BY_COUNT[count]
    return out


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _separated(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    return (
        rect_a[2] + 1 < rect_b[0]
        or rect_b[2] + 1 < rect_a[0]
        or rect_a[3] + 1 < rect_b[1]
        or rect_b[3] + 1 < rect_a[1]
    )


def _fits(rect: tuple[int, int, int, int], chosen: list[tuple[int, int, int, int]]) -> bool:
    return all(_separated(rect, other) for other in chosen)


def _random_raw_tile(color_count: int) -> np.ndarray | None:
    meta_h, meta_w = random.choice(SHAPES_BY_COLOR_COUNT[color_count])
    block_h = random.choice(BLOCK_SIDE_OPTIONS)
    block_w = random.choice(BLOCK_SIDE_OPTIONS)
    palette = random.sample(BASE_COLORS, color_count)

    meta = np.random.choice(palette, size=(meta_h, meta_w))
    if set(map(int, np.unique(meta))) != set(palette):
        return None

    block = np.ones((block_h, block_w), dtype=int)
    return np.kron(meta.astype(int), block)


def _repeat_count_bounds(period: int) -> tuple[int, int]:
    low = max(3, (MIN_GRID_SIDE + period - 1) // period)
    high = max(low, MAX_GRID_SIDE // period)
    return low, high


def _enumerate_rectangles(pattern: np.ndarray) -> dict[int, list[tuple[int, int, int, int]]]:
    height, width = pattern.shape
    candidates = {1: [], 2: [], 3: []}

    for r0 in range(1, height - 1):
        h_max = min(MAX_RECT_SIDE, height - 1 - r0)
        if h_max < 2:
            continue
        for c0 in range(1, width - 1):
            w_max = min(MAX_RECT_SIDE, width - 1 - c0)
            if w_max < 2:
                continue
            for h in range(2, h_max + 1):
                r1 = r0 + h - 1
                for w in range(2, w_max + 1):
                    c1 = c0 + w - 1
                    rect = (r0, c0, r1, c1)
                    if _rect_area(rect) > MAX_RECT_AREA:
                        continue
                    count = int(np.unique(pattern[r0 : r1 + 1, c0 : c1 + 1]).size)
                    if count in candidates:
                        candidates[count].append(rect)

    for rects in candidates.values():
        rects.sort(key=_rect_area)
    return candidates


def _choose_core_rectangles(
    candidates: dict[int, list[tuple[int, int, int, int]]], targets: list[int]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(targets, key=lambda count: len(candidates[count]))
    chosen: list[tuple[int, int, int, int]] = []

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True
        count = order[index]
        pool = candidates[count][: min(len(candidates[count]), 200)]
        random.shuffle(pool)
        for rect in pool:
            if _fits(rect, chosen):
                chosen.append(rect)
                if backtrack(index + 1):
                    return True
                chosen.pop()
        return False

    return chosen if backtrack(0) else None


def _overlay_masks(pattern: np.ndarray, rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    masked = pattern.copy()
    for r0, c0, r1, c1 in rectangles:
        masked[r0 : r1 + 1, c0 : c1 + 1] = MASK_COLOR
    return masked


def _expected_output(pattern: np.ndarray, masked: np.ndarray, rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = masked.copy()
    for rect in rectangles:
        count = int(np.unique(pattern[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1]).size)
        out[rect[0] : rect[2] + 1, rect[1] : rect[3] + 1] = RESULT_COLOR_BY_COUNT[count]
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = random.choices((2, 3), weights=(2, 3))[0]
    raw_tile = _random_raw_tile(color_count)
    if raw_tile is None:
        return None

    tile = _infer_tile(raw_tile)
    period_r, period_c = tile.shape
    if period_r > MAX_INFERRED_PERIOD or period_c > MAX_INFERRED_PERIOD:
        return None

    rep_lo_r, rep_hi_r = _repeat_count_bounds(period_r)
    rep_lo_c, rep_hi_c = _repeat_count_bounds(period_c)
    reps_r = random.randint(rep_lo_r, rep_hi_r)
    reps_c = random.randint(rep_lo_c, rep_hi_c)
    pattern = _paint_from_tile(tile, (period_r * reps_r, period_c * reps_c))

    candidates = _enumerate_rectangles(pattern)
    targets = list(range(1, color_count + 1))
    if any(not candidates[count] for count in targets):
        return None

    rectangles = _choose_core_rectangles(candidates, targets)
    if rectangles is None:
        return None

    extra_rectangles = random.choice(EXTRA_RECT_OPTIONS)
    extra_pool = [rect for count in targets for rect in candidates[count]]
    random.shuffle(extra_pool)
    for rect in extra_pool:
        if extra_rectangles == 0:
            break
        if rect in rectangles or not _fits(rect, rectangles):
            continue
        rectangles.append(rect)
        extra_rectangles -= 1

    masked = _overlay_masks(pattern, rectangles)
    if np.mean(masked == MASK_COLOR) > MAX_MASK_FRACTION:
        return None

    visible_colors = set(map(int, np.unique(masked[masked != MASK_COLOR])))
    if visible_colors != set(map(int, np.unique(pattern))):
        return None

    inferred_tile = _infer_tile(masked)
    if inferred_tile.shape != tile.shape or not np.array_equal(inferred_tile, tile):
        return None

    expected = _expected_output(pattern, masked, rectangles)
    if not np.array_equal(main(masked), expected):
        return None

    return masked


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
