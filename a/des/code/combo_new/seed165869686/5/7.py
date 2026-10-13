import random
from collections import Counter

import numpy as np


BLACK = 0
GRAY = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PERIOD_CHOICES = (2, 3, 4)
MASK_COUNT_CHOICES = (1, 2, 3)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        assert 0 < r0 and 0 < c0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == GRAY), "gray mask must be a solid rectangle"
        assert (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3, "mask needs an interior"
        rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _observed_row_color(row: np.ndarray) -> int | None:
    visible = row[(row != BLACK) & (row != GRAY)]
    if visible.size == 0:
        return None
    colors = np.unique(visible)
    assert colors.size == 1, "each stripe row must have a single visible stripe color"
    return int(colors[0])


def _infer_pattern(observed: list[int | None]) -> list[int]:
    n = len(observed)
    for period in PERIOD_CHOICES:
        if period > n:
            continue
        pattern: list[int | None] = [None] * period
        ok = True
        for idx, color in enumerate(observed):
            if color is None:
                continue
            slot = idx % period
            if pattern[slot] is None:
                pattern[slot] = color
            elif pattern[slot] != color:
                ok = False
                break
        if not ok or any(color is None for color in pattern):
            continue
        concrete = [int(color) for color in pattern]
        if len(set(concrete)) != period:
            continue
        return concrete
    raise AssertionError("failed to infer a unique stripe period from visible rows")


def _stripe_row_map(grid: np.ndarray) -> tuple[dict[int, int], list[int]]:
    stripe_rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != BLACK)]
    assert stripe_rows, "expected at least one stripe row"
    observed = [_observed_row_color(grid[r]) for r in stripe_rows]
    pattern = _infer_pattern(observed)

    row_to_color: dict[int, int] = {}
    for idx, r in enumerate(stripe_rows):
        color = int(pattern[idx % len(pattern)])
        if observed[idx] is not None:
            assert observed[idx] == color
        row_to_color[r] = color
    return row_to_color, pattern


def _mode_with_pattern_tiebreak(counts: Counter[int], pattern: list[int]) -> int:
    order = {color: idx for idx, color in enumerate(pattern)}
    best_color = min(counts, key=lambda color: (-counts[color], order[color]))
    return int(best_color)


def _paint_masks(
    base: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    row_to_color: dict[int, int],
    pattern: list[int],
) -> np.ndarray:
    out = base.copy()
    for r0, c0, r1, c1 in rects:
        counts: Counter[int] = Counter()
        for r in range(r0 + 1, r1):
            assert r in row_to_color, "mask interior must lie on stripe rows"
            color = int(row_to_color[r])
            out[r, c0 + 1 : c1] = color
            counts[color] += c1 - c0 - 1
        assert counts, "mask interior cannot be empty"
        border_color = _mode_with_pattern_tiebreak(counts, pattern)
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore masked horizontal stripes and redraw each mask border with the most frequent recovered stripe color.

    Input structure:
    - The background is black.
    - One contiguous band of full-width horizontal stripes occupies some rows; each stripe row is monochrome.
    - The stripe colors follow a repeating sequence with period 2 to 4.
    - One or more solid gray rectangles cover parts of the stripe band without touching the grid edge.

    Rule:
    - Read the visible stripe rows outside the gray masks and infer the shortest repeating stripe sequence.
    - For every gray rectangle, restore its interior cells with the stripe color that should appear on each covered row.
    - Count the restored interior colors of that rectangle and select the most frequent one; if several colors tie, choose the one that appears earlier in the repeating stripe sequence.
    - Recolor the rectangle's outer border with that selected color, while leaving every other non-masked cell unchanged.

    Key insight:
    - The masked cells are repaired row-by-row from the global stripe period, and the border color is a second-order statistic derived from the repaired interior rather than from the original visible border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_to_color, pattern = _stripe_row_map(g)
    rects = _mask_rectangles(g)
    assert rects, "expected at least one gray mask"
    return _paint_masks(g, rects, row_to_color, pattern)


def _candidate_height_options(max_rect_h: int, period: int, want_unique: bool) -> list[int]:
    options: list[int] = []
    for rect_h in range(3, max_rect_h + 1):
        interior_h = rect_h - 2
        is_unique = interior_h % period == 1
        if is_unique == want_unique:
            options.append(rect_h)
    return options


def _choose_rect_height(rng: random.Random, max_rect_h: int, period: int, want_unique: bool) -> int:
    preferred = _candidate_height_options(max_rect_h, period, want_unique)
    if preferred:
        return rng.choice(preferred)
    return rng.randint(3, max_rect_h)


def _place_masks(
    rng: random.Random,
    h: int,
    w: int,
    stripe_top: int,
    stripe_count: int,
    period: int,
    mask_count: int,
    first_wants_unique: bool,
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []

    max_rect_h = min(stripe_count, max(3, min(10, h - 2)))
    max_rect_w = max(3, min(w - 2, 12))
    if max_rect_h < 3 or max_rect_w < 3:
        return None

    for mask_idx in range(mask_count):
        want_unique = first_wants_unique if mask_idx == 0 else rng.random() < 0.5
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = _choose_rect_height(rng, max_rect_h, period, want_unique)
            rect_w = rng.randint(3, max_rect_w)

            row_min = max(stripe_top, 1)
            row_max = min(stripe_top + stripe_count - rect_h, h - rect_h - 1)
            if row_min > row_max:
                continue
            col_min = 1
            col_max = w - rect_w - 1
            if col_min > col_max:
                continue

            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(h - 1, r1 + 1)
            pad_c1 = min(w - 1, c1 + 1)
            if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                continue

            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    rects.sort()
    return rects


def _build_base_grid(h: int, w: int, stripe_top: int, stripe_count: int, pattern: list[int]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for idx in range(stripe_count):
        grid[stripe_top + idx, :] = int(pattern[idx % len(pattern)])
    return grid


def _expected_output(
    base: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    stripe_top: int,
    stripe_count: int,
    pattern: list[int],
) -> np.ndarray:
    row_to_color = {
        stripe_top + idx: int(pattern[idx % len(pattern)])
        for idx in range(stripe_count)
    }
    return _paint_masks(base, rects, row_to_color, pattern)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    period = rng.choice(PERIOD_CHOICES)
    min_stripe_rows = max(6, 2 * period + 1)

    h = rng.randint(min_stripe_rows, 30)
    spare = h - min_stripe_rows
    top_max = min(3, spare)
    top = rng.randint(0, top_max)
    bottom_max = min(3, spare - top)
    bottom = rng.randint(0, bottom_max)
    stripe_count = h - top - bottom
    if stripe_count < min_stripe_rows:
        return None

    w = rng.randint(10, 30)
    pattern = rng.sample(list(STRIPE_COLORS), period)
    base = _build_base_grid(h, w, top, stripe_count, pattern)

    capacity = 1 + int(h >= 12 and w >= 14) + int(h >= 18 and w >= 18)
    mask_count = rng.randint(1, min(max(MASK_COUNT_CHOICES), capacity))
    first_wants_unique = rng.random() < 0.5
    rects = _place_masks(rng, h, w, top, stripe_count, period, mask_count, first_wants_unique)
    if rects is None:
        return None

    inp = base.copy()
    for r0, c0, r1, c1 in rects:
        inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY

    expected = _expected_output(base, rects, top, stripe_count, pattern)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-mask puzzle")
