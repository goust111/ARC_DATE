from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
DIVIDER_COUNT_RANGE = (1, 4)
MIN_REGION_THICKNESS = 3
SHORT_SIDE_RANGE = (8, 18)
LONG_SIDE_MAX = 24
GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise AssertionError("anchor pairs must be row- or column-aligned")


def _collect_roles(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, tuple[int, int], tuple[int, int]]], dict[tuple[int, int], int]]:
    anchors: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    seeds: dict[tuple[int, int], int] = {}

    counts = Counter(int(value) for value in grid.ravel())
    for color, count in counts.items():
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if count == 2:
            a = tuple(map(int, coords[0]))
            b = tuple(map(int, coords[1]))
            anchors.append((int(color), a, b))
        elif count == 1:
            seeds[tuple(map(int, coords[0]))] = int(color)
        else:
            raise AssertionError("every non-background color must be either an anchor pair or a singleton seed")

    assert anchors, "expected at least one anchor pair"
    assert seeds, "expected at least one singleton seed"
    return anchors, seeds


def _draw_hidden_lines(
    grid: np.ndarray, bg: int, anchors: list[tuple[int, tuple[int, int], tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(grid, copy=True)
    walls = np.zeros(grid.shape, dtype=bool)

    for color, a, b in anchors:
        for r, c in _segment_cells(a, b):
            cell = int(out[r, c])
            if walls[r, c]:
                raise AssertionError("line segments may touch but must not overlap")
            if cell != bg and cell != color:
                raise AssertionError("seed pixels may not lie on hidden line segments")
            out[r, c] = color
            walls[r, c] = True

    return out, walls


def _background_components(passable: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color anchor pair with a straight axis-aligned segment, then
    fill every resulting region whose interior contains exactly one singleton seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    anchors, seeds = _collect_roles(g, bg)
    out, walls = _draw_hidden_lines(g, bg, anchors)

    for component in _background_components(~walls):
        component_seeds = [seeds[cell] for cell in component if cell in seeds]
        assert len(component_seeds) <= 1, "each region must contain at most one singleton seed"
        if not component_seeds:
            continue
        fill_color = int(component_seeds[0])
        for r, c in component:
            if out[r, c] == bg:
                out[r, c] = fill_color

    return out


def _split_with_minimum(total: int, parts: int, minimum: int) -> list[int]:
    assert total >= parts * minimum
    sizes = [minimum] * parts
    for _ in range(total - parts * minimum):
        sizes[random.randrange(parts)] += 1
    return sizes


def _stripe_layout(axis_len: int, divider_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    region_sizes = _split_with_minimum(
        axis_len - divider_count,
        divider_count + 1,
        MIN_REGION_THICKNESS,
    )
    dividers: list[int] = []
    bands: list[tuple[int, int]] = []

    cursor = 0
    for index, size in enumerate(region_sizes):
        bands.append((cursor, cursor + size - 1))
        cursor += size
        if index < divider_count:
            dividers.append(cursor)
            cursor += 1

    assert cursor == axis_len
    return dividers, bands


def _band_seed_coordinate(
    orientation: str, band: tuple[int, int], height: int, width: int
) -> tuple[int, int]:
    if orientation == "vertical":
        c0, c1 = band
        row_lo = 1 if height >= 3 else 0
        row_hi = height - 2 if height >= 3 else height - 1
        col_lo = c0 + 1 if c1 - c0 + 1 >= 3 else c0
        col_hi = c1 - 1 if c1 - c0 + 1 >= 3 else c1
        return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)

    r0, r1 = band
    col_lo = 1 if width >= 3 else 0
    col_hi = width - 2 if width >= 3 else width - 1
    row_lo = r0 + 1 if r1 - r0 + 1 >= 3 else r0
    row_hi = r1 - 1 if r1 - r0 + 1 >= 3 else r1
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _expected_stripe_output(
    grid: np.ndarray,
    orientation: str,
    dividers: list[int],
    divider_colors: list[int],
    bands: list[tuple[int, int]],
    seeded_regions: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    height, width = out.shape

    for color, divider in zip(divider_colors, dividers):
        if orientation == "vertical":
            out[:, divider] = color
        else:
            out[divider, :] = color

    for region_index, seed_color in seeded_regions.items():
        lo, hi = bands[region_index]
        if orientation == "vertical":
            patch = out[:, lo : hi + 1]
        else:
            patch = out[lo : hi + 1, :]
        patch[patch == BG] = seed_color

    return out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    orientation = random.choice(("vertical", "horizontal"))
    divider_count = random.randint(*DIVIDER_COUNT_RANGE)

    min_axis = divider_count + (divider_count + 1) * MIN_REGION_THICKNESS
    if orientation == "vertical":
        height = random.randint(*SHORT_SIDE_RANGE)
        width = random.randint(min_axis, LONG_SIDE_MAX)
        axis_len = width
    else:
        height = random.randint(min_axis, LONG_SIDE_MAX)
        width = random.randint(*SHORT_SIDE_RANGE)
        axis_len = height

    region_count = divider_count + 1
    max_seeded = min(region_count - 1, len(PALETTE) - divider_count)
    if max_seeded < 1:
        return None

    dividers, bands = _stripe_layout(axis_len, divider_count)
    seeded_count = random.randint(1, max_seeded)
    seeded_indices = sorted(random.sample(range(region_count), seeded_count))

    colors = random.sample(PALETTE, divider_count + seeded_count)
    divider_colors = colors[:divider_count]
    seed_colors = colors[divider_count:]

    grid = np.full((height, width), BG, dtype=int)
    for color, divider in zip(divider_colors, dividers):
        if orientation == "vertical":
            grid[0, divider] = color
            grid[height - 1, divider] = color
        else:
            grid[divider, 0] = color
            grid[divider, width - 1] = color

    seeded_regions: dict[int, int] = {}
    for region_index, seed_color in zip(seeded_indices, seed_colors):
        r, c = _band_seed_coordinate(orientation, bands[region_index], height, width)
        assert grid[r, c] == BG
        grid[r, c] = seed_color
        seeded_regions[region_index] = seed_color

    expected = _expected_stripe_output(
        grid,
        orientation,
        dividers,
        divider_colors,
        bands,
        seeded_regions,
    )

    try:
        if np.array_equal(main(grid), expected):
            return grid, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _sample_case()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid input")
