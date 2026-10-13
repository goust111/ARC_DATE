from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIDE_MAX = 30
FENCE_COUNT_CHOICES = (1, 2, 3, 4)
FENCE_SIZE_MIN = 5
FENCE_SIZE_MAX = 10
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _fence_bbox(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    assert r1 - r0 >= 2 and c1 - c0 >= 2, "each fence must enclose a non-empty interior"
    assert np.all(grid[r0, c0 : c1 + 1] == GRAY), "top fence edge must be solid gray"
    assert np.all(grid[r1, c0 : c1 + 1] == GRAY), "bottom fence edge must be solid gray"
    assert np.all(grid[r0 : r1 + 1, c0] == GRAY), "left fence edge must be solid gray"
    assert np.all(grid[r0 : r1 + 1, c1] == GRAY), "right fence edge must be solid gray"
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GRAY), "fence interior cannot contain gray pixels"

    expected_area = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    assert len(comp) == expected_area, "gray component must be a one-pixel-thick rectangular outline"
    return int(r0), int(c0), int(r1), int(c1)


def _winner_color(interior: np.ndarray) -> int:
    markers = interior[(interior != BLACK) & (interior != GRAY)]
    assert markers.size > 0, "each fence must contain at least one interior marker"

    colors, counts = np.unique(markers, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "each fence must have a unique majority marker color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangular fence by the majority color of its interior markers, then erase those markers.

    Input structure:
    - A black background contains several non-touching, non-nested gray rectangular fences.
    - Every fence is a one-pixel-thick closed axis-aligned rectangle.
    - Inside each fence are scattered single-pixel colored markers on black cells.

    Rule:
    - Detect each gray fence as its own rectangular outline.
    - Count the non-black interior marker colors inside that fence and select the unique most frequent one.
    - Repaint the whole gray fence with that winning color and clear the entire fenced interior to black.
    - Keep all pixels outside every fence unchanged.

    Key insight:
    - The interior colored pixels are only votes for the surrounding fence's new color; the markers themselves disappear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components4(g == GRAY):
        r0, c0, r1, c1 = _fence_bbox(g, comp)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        draw_color = _winner_color(interior)

        out[r0, c0 : c1 + 1] = draw_color
        out[r1, c0 : c1 + 1] = draw_color
        out[r0 : r1 + 1, c0] = draw_color
        out[r0 : r1 + 1, c1] = draw_color
        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK

    return out


def _sample_counts(color_count: int, marker_capacity: int) -> list[int] | None:
    if color_count < 2 or marker_capacity < color_count + 1:
        return None

    max_winner = marker_capacity - (color_count - 1)
    winner_count = random.randint(2, max_winner)

    remaining = marker_capacity - winner_count
    other_counts: list[int] = []
    for idx in range(color_count - 1):
        min_needed_after = color_count - 2 - idx
        upper = min(winner_count - 1, remaining - min_needed_after)
        if upper < 1:
            return None
        count = random.randint(1, upper)
        other_counts.append(count)
        remaining -= count

    extra = random.randint(0, remaining)
    while extra > 0:
        choices = [i for i, count in enumerate(other_counts) if count < winner_count - 1]
        if not choices:
            break
        pick = random.choice(choices)
        other_counts[pick] += 1
        extra -= 1

    counts = [winner_count] + other_counts
    assert max(counts) == winner_count and counts.count(winner_count) == 1
    return counts


def _place_fences(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    reserved = np.zeros((height, width), dtype=bool)
    fences: list[tuple[int, int, int, int]] = []

    for _ in range(count):
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            fh = random.randint(FENCE_SIZE_MIN, min(FENCE_SIZE_MAX, height - 2))
            fw = random.randint(FENCE_SIZE_MIN, min(FENCE_SIZE_MAX, width - 2))
            if height - fh - 1 < 1 or width - fw - 1 < 1:
                continue

            r0 = random.randint(1, height - fh - 1)
            c0 = random.randint(1, width - fw - 1)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = (r0, c0, r1, c1)
            fences.append(placed)
            break
        if placed is None:
            return None

    return fences


def _draw_fence(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _parity_cells(r0: int, c0: int, r1: int, c1: int, parity: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r + c) % 2 == parity
    ]


def _populate_markers(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> int | None:
    parity_cells = [_parity_cells(r0, c0, r1, c1, 0), _parity_cells(r0, c0, r1, c1, 1)]
    capacity = max(len(parity_cells[0]), len(parity_cells[1]))
    if capacity < 3:
        return None

    max_colors = min(4, capacity - 1, len(MARKER_COLORS))
    if max_colors < 2:
        return None
    color_count = random.randint(2, max_colors)
    counts = _sample_counts(color_count, capacity)
    if counts is None:
        return None

    usable = [cells for cells in parity_cells if len(cells) >= sum(counts)]
    if not usable:
        return None
    cells = random.choice(usable)
    chosen_cells = random.sample(cells, sum(counts))

    colors = random.sample(MARKER_COLORS, color_count)
    winner_index = random.randrange(color_count)
    colors[0], colors[winner_index] = colors[winner_index], colors[0]
    winner_color = colors[0]

    start = 0
    for color, count in zip(colors, counts):
        for r, c in chosen_cells[start : start + count]:
            grid[r, c] = color
        start += count

    return int(winner_color)


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    fence_count = random.choice(FENCE_COUNT_CHOICES)
    side_floor = 8 + 4 * fence_count
    height = random.randint(min(side_floor, GRID_SIDE_MAX), GRID_SIDE_MAX)
    width = random.randint(min(side_floor, GRID_SIDE_MAX), GRID_SIDE_MAX)

    fences = _place_fences(height, width, fence_count)
    if fences is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(inp)

    for r0, c0, r1, c1 in fences:
        _draw_fence(inp, r0, c0, r1, c1, GRAY)
        winner_color = _populate_markers(inp, r0, c0, r1, c1)
        if winner_color is None:
            return None
        _draw_fence(expected, r0, c0, r1, c1, winner_color)

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid fenced-vote ARC input")
