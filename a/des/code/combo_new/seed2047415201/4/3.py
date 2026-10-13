from __future__ import annotations

from collections import deque
from fractions import Fraction
import random

import numpy as np


BLACK = 0
GREY = 5
BAR_HEIGHT = 10
BAR_WIDTH = 3
COLOR_VALUES = tuple(range(10))
FOREGROUND_COLORS = tuple(c for c in COLOR_VALUES if c not in (BLACK, GREY))
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue

        q: deque[tuple[int, int]] = deque([(start_r, start_c)])
        seen[start_r, start_c] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

        comps.append(np.array(coords, dtype=int))

    return comps


def _round_half_up(numer: int, denom: int) -> int:
    return int((2 * numer + denom) // (2 * denom))


def _bar_heights(counts: list[int], total: int) -> list[int]:
    heights = [_round_half_up(count * BAR_HEIGHT, total) for count in counts]
    exact = [Fraction(count * BAR_HEIGHT, total) for count in counts]
    diff = BAR_HEIGHT - sum(heights)

    while diff > 0:
        ranked = sorted(
            range(len(heights)),
            key=lambda i: (exact[i] - heights[i], -i),
            reverse=True,
        )
        for i in ranked[:diff]:
            heights[i] += 1
        diff = BAR_HEIGHT - sum(heights)

    while diff < 0:
        ranked = sorted(
            range(len(heights)),
            key=lambda i: (heights[i] - exact[i], heights[i], -i),
            reverse=True,
        )
        changed = False
        for i in ranked:
            if heights[i] > 0:
                heights[i] -= 1
                changed = True
                break
        assert changed, "rounded bar heights cannot be reduced to ten"
        diff = BAR_HEIGHT - sum(heights)

    assert sum(heights) == BAR_HEIGHT
    return heights


def _ordered_colors(window: np.ndarray) -> list[int]:
    colors = [int(c) for c in np.unique(window)]
    first_pos: dict[int, tuple[int, int]] = {}
    for color in colors:
        coords = np.argwhere(window == color)
        assert coords.size > 0
        first_r = int(coords[:, 0].min())
        first_c = int(coords[coords[:, 0] == first_r, 1].min())
        first_pos[color] = (first_r, first_c)
    return sorted(colors, key=lambda color: first_pos[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-encode each mixed-color rectangle as a fixed vertical proportion bar above the baseline.

    Input structure:
    - The canvas background is Black, and the bottom row is a solid Grey baseline.
    - Above the baseline are separated filled rectangular objects.
    - Each object contains a mix of two or three foreground colors, with no Black holes inside the rectangle.

    Rule:
    - Find each foreground rectangle above the Grey baseline and count the pixels of each color inside it.
    - Clear the original rectangles back to Black while keeping the baseline.
    - Above the baseline, centered on each object's original horizontal center, draw a 3-pixel-wide vertical bar of total height 10.
    - The bar is stacked from bottom to top using the colors in order of their first row of appearance in the source object; each segment height is the rounded color proportion scaled to ten pixels.

    Key insight:
    - The two-dimensional color mixture is not copied; each object is reduced to an ordered one-dimensional histogram whose segment heights encode the object's internal color ratios.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= BAR_HEIGHT + 1 and width >= BAR_WIDTH, "grid must have room for bars"

    baseline_color = int(g[-1, 0])
    assert baseline_color != BLACK, "bottom baseline must be non-Black"
    assert np.all(g[-1, :] == baseline_color), "bottom row must be a solid baseline"

    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[:-1, :] = g[:-1, :] != BLACK
    objects = _components(object_mask)
    assert objects, "expected at least one foreground object"

    out = np.full_like(g, BLACK)
    out[-1, :] = baseline_color

    bar_specs: list[tuple[int, int, list[int], list[int]]] = []
    occupied_bars = np.zeros_like(out, dtype=bool)
    for coords in objects:
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        window = g[r0 : r1 + 1, c0 : c1 + 1]

        assert window.size == len(coords), "object bbox must be a filled rectangle"
        assert np.all(window != BLACK), "object rectangles must not contain Black holes"
        colors = _ordered_colors(window)
        assert len(colors) in (2, 3), "each object must contain two or three colors"

        counts = [int(np.count_nonzero(window == color)) for color in colors]
        heights = _bar_heights(counts, int(window.size))
        center_c = (c0 + c1) // 2
        left_c = center_c - BAR_WIDTH // 2
        top_r = height - 1 - BAR_HEIGHT
        assert 0 <= left_c and left_c + BAR_WIDTH <= width, "bar must fit horizontally"
        assert top_r >= 0, "bar must fit above the baseline"

        footprint = occupied_bars[top_r : height - 1, left_c : left_c + BAR_WIDTH]
        assert not np.any(footprint), "bar footprints must not overlap"
        occupied_bars[top_r : height - 1, left_c : left_c + BAR_WIDTH] = True
        bar_specs.append((center_c, left_c, colors, heights))

    for _center_c, left_c, colors, heights in sorted(bar_specs, key=lambda spec: spec[0]):
        bottom = height - 2
        for color, segment_h in zip(colors, heights):
            if segment_h <= 0:
                continue
            top = bottom - segment_h + 1
            out[top : bottom + 1, left_c : left_c + BAR_WIDTH] = int(color)
            bottom = top - 1
        assert bottom == height - BAR_HEIGHT - 2

    return out


# --------------- generator (new task) ---------------


def _valid_count_options(area: int, width: int, obj_h: int, n_colors: int) -> list[tuple[int, ...]]:
    options: list[tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: list[int]) -> None:
        if slots == 1:
            counts = prefix + [remaining]
            if min(counts) <= 0:
                return
            if counts[0] < width:
                return
            for i, count in enumerate(counts):
                if count > (obj_h - i) * width:
                    return
            heights = _bar_heights(counts, area)
            if min(heights) > 0:
                options.append(tuple(counts))
            return

        for value in range(1, remaining - slots + 2):
            rec(remaining - value, slots - 1, prefix + [value])

    rec(area, n_colors, [])
    return options


def _make_mixed_rectangle(
    rng: random.Random,
    obj_h: int,
    obj_w: int,
    colors: list[int],
    counts: tuple[int, ...],
) -> np.ndarray | None:
    rect = np.full((obj_h, obj_w), -1, dtype=int)
    remaining = list(counts)

    rect[0, :] = colors[0]
    remaining[0] -= obj_w
    if remaining[0] < 0:
        return None

    reserved_cols = list(range(obj_w))
    rng.shuffle(reserved_cols)
    for i in range(1, len(colors)):
        if i >= obj_h or remaining[i] <= 0:
            return None
        c = reserved_cols[(i - 1) % obj_w]
        rect[i, c] = colors[i]
        remaining[i] -= 1

    for i in range(len(colors) - 1, -1, -1):
        if remaining[i] == 0:
            continue
        candidates = [
            (r, c)
            for r in range(i, obj_h)
            for c in range(obj_w)
            if rect[r, c] == -1
        ]
        if len(candidates) < remaining[i]:
            return None
        rng.shuffle(candidates)
        for r, c in candidates[: remaining[i]]:
            rect[r, c] = colors[i]

    assert np.all(rect != -1)
    return rect


def _sample_object(rng: random.Random, n_colors: int) -> np.ndarray | None:
    dims = [(3, 3), (4, 3), (5, 3), (4, 5), (5, 5), (6, 5), (7, 5), (5, 7)]
    rng.shuffle(dims)
    colors = rng.sample(list(FOREGROUND_COLORS), n_colors)

    for obj_h, obj_w in dims:
        if obj_h < n_colors:
            continue
        options = _valid_count_options(obj_h * obj_w, obj_w, obj_h, n_colors)
        if not options:
            continue
        counts = rng.choice(options)
        rect = _make_mixed_rectangle(rng, obj_h, obj_w, colors, counts)
        if rect is not None:
            return rect
    return None


def _can_place(
    occupied: np.ndarray,
    top: int,
    left: int,
    obj_h: int,
    obj_w: int,
) -> bool:
    height, width = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(height, top + obj_h + 1)
    c0 = max(0, left - 1)
    c1 = min(width, left + obj_w + 1)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(15, 30)
    width = rng.randint(14, 30)
    n_objects = rng.randint(2, min(5, max(2, (width - 2) // 4)))

    grid = np.full((height, width), BLACK, dtype=int)
    grid[-1, :] = GREY
    occupied = np.zeros((height - 1, width), dtype=bool)
    centers: list[int] = []

    # Use a balanced branch choice so both two-color and three-color rectangles appear often.
    color_counts = [2, 3] * ((n_objects + 1) // 2)
    rng.shuffle(color_counts)

    for i in range(n_objects):
        patch = None
        for _ in range(50):
            patch = _sample_object(rng, color_counts[i])
            if patch is not None and patch.shape[0] <= height - 1 and patch.shape[1] <= width - 2:
                break
        if patch is None:
            return None

        obj_h, obj_w = map(int, patch.shape)
        placed = False
        for _ in range(120):
            top = rng.randint(0, height - 1 - obj_h)
            left = rng.randint(1, width - obj_w - 1)
            center = left + (obj_w - 1) // 2
            if center - 1 < 0 or center + 1 >= width:
                continue
            if any(abs(center - other) < 4 for other in centers):
                continue
            if not _can_place(occupied, top, left, obj_h, obj_w):
                continue
            grid[top : top + obj_h, left : left + obj_w] = patch
            occupied[top : top + obj_h, left : left + obj_w] = True
            centers.append(center)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mixed-rectangle bar-chart input")
