from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
SIDE_MARGIN = 1
MIN_CONE_GAP = 2
MAX_EXTRA_WIDTH = 5


def _spread_row(active: np.ndarray) -> np.ndarray:
    """One downward step: each lit cell reaches left-down, down, right-down."""
    nxt = active.copy()
    nxt[1:] |= active[:-1]
    nxt[:-1] |= active[1:]
    return nxt


def _find_sources(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != GRAY))
    sources = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    sources.sort()
    return sources


def _cone_interval(source_r: int, source_c: int, row: int) -> tuple[int, int]:
    depth = int(row) - int(source_r)
    return int(source_c) - depth, int(source_c) + depth


def _clear_cone_mask(shape: tuple[int, int], source: tuple[int, int, int]) -> np.ndarray:
    h, w = shape
    sr, sc, _color = source
    mask = np.zeros((h, w), dtype=bool)
    for r in range(sr, h):
        left, right = _cone_interval(sr, sc, r)
        left = max(0, left)
        right = min(w - 1, right)
        mask[r, left : right + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored source downward into a widening cone, with grey blockers carving tapering shadows.

    Input structure:
    - The grid has a black background, one or more single-pixel colored light sources, and grey blocker pixels below them.
    - Every source color is distinct, and valid inputs keep different sources' unobstructed cones disjoint.

    Rule:
    - Treat each source pixel as the only lit cell on its own row.
    - For every lower row, a non-grey cell becomes lit iff at least one of its upper-left, upper, or upper-right cells was lit in the previous row.
    - Paint each lit background cell with that source color, leave blocker pixels grey, and leave unreachable cells black.

    Key insight:
    - The output is not a precomputed static triangle with chunks removed: illumination must be simulated row by row, because darkness behind a blocker narrows as side light walks inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    h, w = g.shape
    sources = _find_sources(g)
    assert sources, "expected at least one colored source pixel"

    for sr, sc, color in sources:
        assert 0 <= sr < h and 0 <= sc < w
        active = np.zeros(w, dtype=bool)
        active[sc] = True

        for r in range(sr + 1, h):
            active = _spread_row(active)
            active &= g[r] != GRAY
            if not active.any():
                break

            touching_other_source = active & (g[r] != BLACK) & (g[r] != GRAY) & (g[r] != color)
            assert not np.any(touching_other_source), "valid inputs keep light cones disjoint"

            paint = active & (g[r] == BLACK)
            conflict = paint & (out[r] != BLACK) & (out[r] != color)
            assert not np.any(conflict), "different light colors must not overlap"
            out[r, paint] = color

    return out


def _pick_source_count() -> int:
    return 1 if random.random() < 0.45 else 2


def _candidate_layouts(count: int) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    layouts: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    heights = range(12, 19) if count == 1 else range(9, 13)

    for height in heights:
        if count == 1:
            max_row = min(4, height - 6)
            for row in range(max_row + 1):
                depth = height - 1 - row
                span = 2 * depth + 1
                if span + 2 * SIDE_MARGIN <= 30:
                    layouts.append((height, (row,), ()))
            continue

        low = max(1, height - 7)
        high = min(5, height // 2, height - 6)
        if low > high:
            continue
        for left_row in range(low, high + 1):
            for right_row in range(left_row, high + 1):
                rows = (left_row, right_row)
                depths = [height - 1 - row for row in rows]
                spans = [2 * depth + 1 for depth in depths]
                base = sum(spans) + 2 * SIDE_MARGIN
                max_gap = min(MIN_CONE_GAP + 2, 30 - base)
                for gap in range(MIN_CONE_GAP, max_gap + 1):
                    layouts.append((height, rows, (gap,)))

    return layouts


def _pack_sources() -> tuple[tuple[int, int], list[tuple[int, int, int]]] | None:
    count = _pick_source_count()
    layouts = _candidate_layouts(count)
    if not layouts:
        return None
    height, rows, gaps = random.choice(layouts)

    depths = [height - 1 - row for row in rows]
    spans = [2 * depth + 1 for depth in depths]
    slot_mins = [SIDE_MARGIN, *gaps, SIDE_MARGIN]
    required_width = sum(spans) + sum(slot_mins)
    if required_width > 30:
        return None

    width = random.randint(required_width, min(30, required_width + MAX_EXTRA_WIDTH))
    extras = [0] * len(slot_mins)
    for _ in range(width - required_width):
        extras[random.randrange(len(extras))] += 1
    slots = [base + extra for base, extra in zip(slot_mins, extras)]

    cursor = slots[0]
    centers = []
    for idx, depth in enumerate(depths):
        centers.append(cursor + depth)
        cursor += 2 * depth + 1
        if idx < count - 1:
            cursor += slots[idx + 1]
    assert width - cursor == slots[-1]

    colors = random.sample(SOURCE_COLORS, count)
    sources = [(row, col, color) for row, col, color in zip(rows, centers, colors)]
    return (height, width), sources


def _pick_bar(shape: tuple[int, int], source: tuple[int, int, int]) -> tuple[int, int, int] | None:
    h, _w = shape
    sr, sc, _color = source
    candidates: list[tuple[int, int, int]] = []

    for row in range(sr + 3, h - 1):
        left, right = _cone_interval(sr, sc, row)
        inner_left = left + 1
        inner_right = right - 1
        usable = inner_right - inner_left + 1
        if usable < 3:
            continue

        max_width = min(5, usable)
        width_choices = [width for width in range(3, max_width + 1) if (h - 1 - row) >= ((width + 1) // 2)]
        if not width_choices:
            continue

        best_width = max(width_choices)
        starts = list(range(inner_left, inner_right - best_width + 2))
        random.shuffle(starts)
        for start in starts:
            candidates.append((row, start, best_width))

    if not candidates:
        return None

    best_width = max(width for _row, _start, width in candidates)
    narrowed = [cand for cand in candidates if cand[2] >= best_width - 1]
    return random.choice(narrowed)


def _bar_mask(shape: tuple[int, int], bar: tuple[int, int, int]) -> np.ndarray:
    row, start, width = bar
    mask = np.zeros(shape, dtype=bool)
    mask[row, start : start + width] = True
    return mask


def _build_scene() -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    packed = _pack_sources()
    if packed is None:
        return None

    shape, sources = packed
    bars: list[tuple[int, int, int]] = []
    occupied = np.zeros(shape, dtype=bool)
    for source in sources:
        bar = _pick_bar(shape, source)
        if bar is None:
            return None
        bar_mask = _bar_mask(shape, bar)
        if np.any(occupied & bar_mask):
            return None
        occupied |= bar_mask
        bars.append(bar)

    grid = np.full(shape, BLACK, dtype=int)
    for row, col, color in sources:
        grid[row, col] = color
    for row, start, bar_width in bars:
        grid[row, start : start + bar_width] = GRAY
    return grid, sources, bars


def _sources_are_disjoint(shape: tuple[int, int], sources: list[tuple[int, int, int]]) -> bool:
    seen = np.zeros(shape, dtype=int)
    for idx, source in enumerate(sources, start=1):
        mask = _clear_cone_mask(shape, source)
        if np.any(seen[mask]):
            return False
        seen[mask] = idx
    return True


def _bar_has_extra_shadow(grid: np.ndarray, bar: tuple[int, int, int], expected: np.ndarray) -> bool:
    test = grid.copy()
    row, start, width = bar
    test[row, start : start + width] = BLACK
    diff = main(test) != expected
    return bool(np.any(diff & ~_bar_mask(grid.shape, bar)))


def _shadow_heals(clear_out: np.ndarray, actual_out: np.ndarray, source: tuple[int, int, int]) -> bool:
    sr, _sc, color = source
    counts = [int(np.count_nonzero((clear_out[r] == color) & (actual_out[r] == BLACK))) for r in range(sr + 1, actual_out.shape[0])]
    positive = [idx for idx, count in enumerate(counts) if count > 0]
    if not positive:
        return False
    last = positive[-1]
    if last + 1 >= len(counts):
        return False
    if counts[last + 1] != 0:
        return False
    return any(counts[idx] > counts[idx + 1] for idx in range(len(counts) - 1))


def _validate_scene(
    grid: np.ndarray,
    sources: list[tuple[int, int, int]],
    bars: list[tuple[int, int, int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(sources) != len(set((r, c) for r, c, _color in sources)):
        return False
    if len(sources) != len(set(color for _r, _c, color in sources)):
        return False
    if not _sources_are_disjoint(grid.shape, sources):
        return False

    expected = main(grid)
    if np.array_equal(expected, grid):
        return False

    no_blockers = grid.copy()
    no_blockers[no_blockers == GRAY] = BLACK
    clear_out = main(no_blockers)

    if not np.any((clear_out != BLACK) & (expected == BLACK)):
        return False

    for source, bar in zip(sources, bars):
        bar_mask = _bar_mask(grid.shape, bar)
        cone_mask = _clear_cone_mask(grid.shape, source)
        if not np.all(cone_mask[bar_mask]):
            return False
        if not _bar_has_extra_shadow(grid, bar, expected):
            return False
        if not _shadow_heals(clear_out, expected, source):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        grid, sources, bars = built
        if _validate_scene(grid, sources, bars):
            return grid
    raise RuntimeError("failed to generate a valid light-shadow scene")
