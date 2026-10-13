from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
STRUCTURE_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# Generator config.
COLOR_COUNT_OPTIONS = (3, 4, 5)
COLOR_COUNT_WEIGHTS = (0.25, 0.40, 0.35)
CENTER_PIXEL_PROBABILITY = 0.60
MAX_GENERATION_ATTEMPTS = 300
SLOT_STRIDE = 4
SLOT_USABLE = 3


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in STRUCTURE_8:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors8(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))


def _color_component_info(grid: np.ndarray) -> dict[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    return {color: _find_components(g == color) for color in colors}


def _render_concentric_squares(color_counts: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(color_counts, key=lambda item: (-int(item[1]), int(item[0])))
    side = 2 * int(ordered[0][1]) - 1
    out = np.full((side, side), BG, dtype=int)
    for color, count in ordered:
        inner_side = 2 * int(count) - 1
        pad = (side - inner_side) // 2
        out[pad : pad + inner_side, pad : pad + inner_side] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Encode per-color 8-connected component counts as concentric centered squares.

    Input structure:
    - The input is a black background grid with 3 to 5 non-black colors.
    - Each foreground color appears in 1 to 5 separate 8-connected components.
    - Components are irregular small monochrome shapes scattered apart from each other.

    Rule:
    - Count the number of 8-connected components for each non-background color.
    - Sort colors by component count from largest to smallest.
    - Create a square black output of side `2 * N_max - 1`, then fill centered squares of side `2 * N_i - 1` from outside to inside using the sorted colors.

    Key insight:
    - The input geometry is only a carrier for one number per color: how many disconnected 8-connected pieces that color has; those counts are re-expressed as nested square layers.
    """
    g = np.asarray(grid, dtype=int)
    color_to_components = _color_component_info(g)
    assert 3 <= len(color_to_components) <= 5, "expected 3 to 5 foreground colors"

    color_counts = [(color, len(components)) for color, components in color_to_components.items()]
    counts = [count for _color, count in color_counts]
    assert len(set(counts)) == len(counts), "component counts must be unique"
    assert min(counts) >= 1 and max(counts) <= 8

    return _render_concentric_squares(color_counts)


def _sample_component_counts(rng: np.random.Generator) -> list[int]:
    n_colors = int(rng.choice(COLOR_COUNT_OPTIONS, p=COLOR_COUNT_WEIGHTS))
    want_center_pixel = bool(rng.random() < CENTER_PIXEL_PROBABILITY)

    if want_center_pixel or n_colors == 5:
        max_count_choices = np.arange(n_colors, 6, dtype=int)
        max_weights = np.arange(1, len(max_count_choices) + 1, dtype=float)
        max_weights /= max_weights.sum()
        max_count = int(rng.choice(max_count_choices, p=max_weights))
        chosen = {1, max_count}
        remaining = [count for count in range(2, max_count) if count not in chosen]
    else:
        max_count_choices = np.arange(max(n_colors + 1, 4), 6, dtype=int)
        max_weights = np.arange(1, len(max_count_choices) + 1, dtype=float)
        max_weights /= max_weights.sum()
        max_count = int(rng.choice(max_count_choices, p=max_weights))
        chosen = {max_count}
        remaining = list(range(2, max_count))

    rng.shuffle(remaining)
    while len(chosen) < n_colors:
        chosen.add(int(remaining.pop()))

    counts = list(chosen)
    rng.shuffle(counts)
    return counts


def _random_component_patch(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(2, SLOT_USABLE + 1))
        w = int(rng.integers(2, SLOT_USABLE + 1))
        max_area = min(8, h * w - 1)
        if max_area < 2:
            continue
        target_area = int(rng.integers(2, max_area + 1))

        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        frontier = {cell for cell in _neighbors8(*start) if 0 <= cell[0] < h and 0 <= cell[1] < w}

        while len(cells) < target_area and frontier:
            options = tuple(frontier)
            nxt = options[int(rng.integers(len(options)))]
            frontier.remove(nxt)
            cells.add(nxt)
            for cell in _neighbors8(*nxt):
                if 0 <= cell[0] < h and 0 <= cell[1] < w and cell not in cells:
                    frontier.add(cell)

        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        coords[:, 0] -= int(r0)
        coords[:, 1] -= int(c0)
        if _bbox_area(coords) == int(coords.shape[0]):
            continue

        patch = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
        patch[coords[:, 0], coords[:, 1]] = True
        return patch
    raise RuntimeError("failed to sample irregular component patch")


def _slot_grid_shape(total_components: int, rng: np.random.Generator) -> tuple[int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for rows in range(1, total_components + 1):
        cols = int(math.ceil(total_components / rows))
        candidates.append((abs(rows - cols), rows * cols - total_components, max(rows, cols), rows, cols))
    candidates.sort()
    top = candidates[: min(6, len(candidates))]
    _score0, _score1, _score2, rows, cols = top[int(rng.integers(len(top)))]
    if rng.random() < 0.5:
        rows, cols = cols, rows
    return int(rows), int(cols)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    view[patch] = int(color)


def _apply_random_grid_transform(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(grid, int(rng.integers(4)))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _validate_input_semantics(grid: np.ndarray, planned_counts: dict[int, int]) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30 or _get_bg(g) != BG:
        return False

    color_to_components = _color_component_info(g)
    observed_counts = {color: len(components) for color, components in color_to_components.items()}
    if observed_counts != planned_counts:
        return False
    if not (3 <= len(observed_counts) <= 5):
        return False
    if len(set(observed_counts.values())) != len(observed_counts):
        return False

    for components in color_to_components.values():
        for coords in components:
            area = int(coords.shape[0])
            if not (2 <= area <= 8):
                return False
            if _bbox_area(coords) == area:
                return False

    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG:
                continue
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(g[nr, nc])
                    if other != BG and other != color:
                        return False

    expected = _render_concentric_squares(list(planned_counts.items()))
    actual = main(g)
    if actual.ndim != 2 or max(actual.shape) > 30:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    counts = _sample_component_counts(rng)
    colors = [int(color) for color in rng.choice(np.array(PALETTE, dtype=int), size=len(counts), replace=False)]
    planned_counts = {color: count for color, count in zip(colors, counts)}

    total_components = int(sum(counts))
    slot_rows, slot_cols = _slot_grid_shape(total_components, rng)
    grid = np.full((slot_rows * SLOT_STRIDE + 1, slot_cols * SLOT_STRIDE + 1), BG, dtype=int)

    slots = [(1 + sr * SLOT_STRIDE, 1 + sc * SLOT_STRIDE) for sr in range(slot_rows) for sc in range(slot_cols)]
    rng.shuffle(slots)
    slot_iter = iter(slots)

    for color, component_count in planned_counts.items():
        for _ in range(component_count):
            patch = _random_component_patch(rng)
            ph, pw = patch.shape
            base_r, base_c = next(slot_iter)
            jitter_r = int(rng.integers(0, SLOT_USABLE - ph + 1))
            jitter_c = int(rng.integers(0, SLOT_USABLE - pw + 1))
            _paint_patch(grid, patch, base_r + jitter_r, base_c + jitter_c, color)

    grid = _apply_random_grid_transform(grid, rng)
    if _validate_input_semantics(grid, planned_counts):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC concentric count task")
