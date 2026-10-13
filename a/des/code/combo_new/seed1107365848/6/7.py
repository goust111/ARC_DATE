from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = random.Random()

# Generator bounds.
GRID_LIMIT = 30
GEN_ATTEMPTS = 400
ZONE_ATTEMPTS = 200
COMPONENT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 250

# Axis and scene variety.
AXIS_COUNT_CHOICES = (1, 2, 3)
ZONE_HEIGHT_RANGE = (8, 13)
ZONE_WIDTH_RANGE = (8, 13)
SOURCE_COMPONENT_COUNT_CHOICES = (1, 2)
SOURCE_SIZE_RANGE = (4, 7)
TARGET_SIZE_MIN = 3
TARGET_MODES = ("empty", "overlap", "separate")
PATTERN_COLORS = tuple(range(1, 10))


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return tuple((int(r + dr), int(c + dc)) for dr, dc in DIRS4)


def _find_axes(grid: np.ndarray) -> tuple[list[dict[str, int]], set[int]]:
    axes: list[dict[str, int]] = []
    marker_colors: set[int] = set()

    colors, counts = np.unique(grid, return_counts=True)
    for raw_color, raw_count in zip(colors.tolist(), counts.tolist()):
        color = int(raw_color)
        count = int(raw_count)
        if color == BG or count != 2:
            continue

        coords = np.argwhere(grid == color)
        (r0, c0), (r1, c1) = [(int(r), int(c)) for r, c in coords]
        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            axes.append(
                {
                    "color": color,
                    "orientation": 0,  # horizontal
                    "fixed": r0,
                    "start": c_lo,
                    "end": c_hi,
                }
            )
            marker_colors.add(color)
        elif c0 == c1:
            r_lo, r_hi = sorted((r0, r1))
            axes.append(
                {
                    "color": color,
                    "orientation": 1,  # vertical
                    "fixed": c0,
                    "start": r_lo,
                    "end": r_hi,
                }
            )
            marker_colors.add(color)

    axes.sort(key=lambda axis: (axis["orientation"], axis["fixed"], axis["start"], axis["end"], axis["color"]))
    assert axes, "expected at least one aligned marker pair"
    return axes, marker_colors


def _erase_markers(grid: np.ndarray, marker_colors: set[int]) -> np.ndarray:
    scene = np.array(grid, dtype=int, copy=True)
    if marker_colors:
        mask = np.isin(scene, list(marker_colors))
        scene[mask] = BG
    return scene


def _partition_axis_pixels(scene: np.ndarray, axis: dict[str, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    side_a: list[tuple[int, int]] = []
    side_b: list[tuple[int, int]] = []

    for rr, cc in np.argwhere(scene != BG):
        r = int(rr)
        c = int(cc)
        if axis["orientation"] == 0:
            if not (axis["start"] <= c <= axis["end"]) or r == axis["fixed"]:
                continue
            if r < axis["fixed"]:
                side_a.append((r, c))
            else:
                side_b.append((r, c))
        else:
            if not (axis["start"] <= r <= axis["end"]) or c == axis["fixed"]:
                continue
            if c < axis["fixed"]:
                side_a.append((r, c))
            else:
                side_b.append((r, c))

    return side_a, side_b


def _reflect(axis: dict[str, int], r: int, c: int) -> tuple[int, int]:
    if axis["orientation"] == 0:
        return int(2 * axis["fixed"] - r), int(c)
    return int(r), int(2 * axis["fixed"] - c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the denser side of each marker-defined fold axis onto the sparser side.

    Input structure:
    - Black background with several same-color marker pairs and additional non-marker foreground patterns.
    - Each marker color appears exactly twice, and the two marker pixels lie on the same row or the same column.
    - The segment joining a marker pair is a horizontal or vertical fold axis.
    - Non-marker foreground pixels appear on one side or both sides of each finite axis segment.

    Rule:
    - Find every color that appears exactly twice in aligned positions and treat the segment between those two pixels as one fold axis.
    - Remove the marker pixels from the scene before counting foreground on the two sides of each axis.
    - For each axis, compare the number of foreground pixels on its two sides within the axis span; the denser side is the source side.
    - Reflect every source-side foreground pixel across the axis and place it onto the opposite side only where the destination cell is still background; existing target-side pixels stay unchanged.

    Key insight:
    - The paired marker pixels define a finite reflection segment, but the fold direction is not encoded by color: it must be inferred from which side contains more foreground pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    axes, marker_colors = _find_axes(g)
    scene = _erase_markers(g, marker_colors)
    out = scene.copy()
    h, w = out.shape

    for axis in axes:
        side_a, side_b = _partition_axis_pixels(scene, axis)
        assert side_a or side_b, "each axis must have at least one nearby foreground pixel"
        assert len(side_a) != len(side_b), "fold direction must be determined by unequal side counts"
        source = side_a if len(side_a) > len(side_b) else side_b

        for r, c in source:
            rr, cc = _reflect(axis, r, c)
            assert 0 <= rr < h and 0 <= cc < w, "reflected pixels must remain in bounds"
            if out[rr, cc] == BG:
                out[rr, cc] = scene[r, c]

    return out


def _sample_axis(zone_h: int, zone_w: int, orientation: int) -> dict[str, int] | None:
    if orientation == 0:
        if zone_h < 8 or zone_w < 6:
            return None
        row = RNG.randint(2, zone_h - 3)
        c0 = RNG.randint(1, zone_w - 5)
        c1 = RNG.randint(c0 + 3, zone_w - 2)
        if min(row, zone_h - 1 - row) < 2:
            return None
        return {"orientation": 0, "fixed": row, "start": c0, "end": c1}

    if zone_h < 6 or zone_w < 8:
        return None
    col = RNG.randint(2, zone_w - 3)
    r0 = RNG.randint(1, zone_h - 5)
    r1 = RNG.randint(r0 + 3, zone_h - 2)
    if min(col, zone_w - 1 - col) < 2:
        return None
    return {"orientation": 1, "fixed": col, "start": r0, "end": r1}


def _axis_sides(zone_h: int, zone_w: int, axis: dict[str, int]) -> tuple[dict[str, set[tuple[int, int]]], tuple[str, str]]:
    if axis["orientation"] == 0:
        max_offset = min(axis["fixed"], zone_h - 1 - axis["fixed"])
        top = {
            (axis["fixed"] - dist, col)
            for dist in range(1, max_offset + 1)
            for col in range(axis["start"], axis["end"] + 1)
        }
        bottom = {
            (axis["fixed"] + dist, col)
            for dist in range(1, max_offset + 1)
            for col in range(axis["start"], axis["end"] + 1)
        }
        return {"top": top, "bottom": bottom}, ("top", "bottom")

    max_offset = min(axis["fixed"], zone_w - 1 - axis["fixed"])
    left = {
        (row, axis["fixed"] - dist)
        for dist in range(1, max_offset + 1)
        for row in range(axis["start"], axis["end"] + 1)
    }
    right = {
        (row, axis["fixed"] + dist)
        for dist in range(1, max_offset + 1)
        for row in range(axis["start"], axis["end"] + 1)
    }
    return {"left": left, "right": right}, ("left", "right")


def _grow_component(available: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    if len(available) < size:
        return None

    seeds = list(available)
    RNG.shuffle(seeds)
    for seed in seeds[: min(len(seeds), COMPONENT_ATTEMPTS)]:
        component = {seed}
        frontier = {nb for nb in _neighbors4(seed) if nb in available}
        while len(component) < size and frontier:
            cell = RNG.choice(tuple(frontier))
            frontier.remove(cell)
            component.add(cell)
            for nb in _neighbors4(cell):
                if nb in available and nb not in component:
                    frontier.add(nb)
        if len(component) == size:
            return component
    return None


def _reserve_gap(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set(cells)
    for cell in list(cells):
        blocked.update(_neighbors4(cell))
    return blocked


def _place_components(allowed: set[tuple[int, int]], sizes: list[int]) -> list[set[tuple[int, int]]] | None:
    available = set(allowed)
    components: list[set[tuple[int, int]]] = []

    for size in sorted(sizes, reverse=True):
        component = _grow_component(available, size)
        if component is None:
            return None
        components.append(component)
        available -= _reserve_gap(component)

    return components


def _largest_component_size(cells: set[tuple[int, int]]) -> int:
    remaining = set(cells)
    best = 0
    while remaining:
        seed = remaining.pop()
        q = deque([seed])
        size = 0
        while q:
            cell = q.popleft()
            size += 1
            for nb in _neighbors4(cell):
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
        best = max(best, size)
    return best


def _paint_cells(zone: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        zone[r, c] = int(color)


def _build_zone(marker_color: int, pattern_palette: tuple[int, ...]) -> np.ndarray | None:
    for _ in range(ZONE_ATTEMPTS):
        zone_h = RNG.randint(*ZONE_HEIGHT_RANGE)
        zone_w = RNG.randint(*ZONE_WIDTH_RANGE)
        orientation = RNG.randint(0, 1)
        axis = _sample_axis(zone_h, zone_w, orientation)
        if axis is None:
            continue

        side_map, side_names = _axis_sides(zone_h, zone_w, axis)
        source_name = RNG.choice(side_names)
        target_name = side_names[1] if source_name == side_names[0] else side_names[0]
        source_allowed = side_map[source_name]
        target_allowed = side_map[target_name]

        source_sizes = [RNG.randint(*SOURCE_SIZE_RANGE) for _ in range(RNG.choice(SOURCE_COMPONENT_COUNT_CHOICES))]
        source_components = _place_components(source_allowed, source_sizes)
        if source_components is None:
            continue

        source_total = sum(len(component) for component in source_components)
        mirrored_cells = {_reflect(axis, r, c) for component in source_components for r, c in component}

        target_mode = RNG.choice(TARGET_MODES)
        target_components: list[set[tuple[int, int]]] = []
        if target_mode != "empty":
            if target_mode == "overlap":
                target_pool = set(mirrored_cells)
            else:
                target_pool = set(target_allowed) - set(mirrored_cells)

            max_target = min(source_total - 1, _largest_component_size(target_pool))
            if max_target < TARGET_SIZE_MIN:
                continue
            target_size = RNG.randint(TARGET_SIZE_MIN, max_target)
            target_components = _place_components(target_pool, [target_size])
            if target_components is None:
                continue

        zone = np.zeros((zone_h, zone_w), dtype=int)
        if axis["orientation"] == 0:
            zone[axis["fixed"], axis["start"]] = marker_color
            zone[axis["fixed"], axis["end"]] = marker_color
        else:
            zone[axis["start"], axis["fixed"]] = marker_color
            zone[axis["end"], axis["fixed"]] = marker_color

        zone_colors = [RNG.choice(pattern_palette) for _ in range(len(source_components))]
        for component, color in zip(source_components, zone_colors):
            _paint_cells(zone, component, int(color))

        if target_components:
            used_source_colors = {
                int(zone[_reflect(axis, r, c)])
                for component in target_components
                for r, c in component
                if zone[_reflect(axis, r, c)] != BG
            }
            choices = tuple(color for color in pattern_palette if color not in used_source_colors) or pattern_palette
            target_color = RNG.choice(choices)
            for component in target_components:
                _paint_cells(zone, component, int(target_color))

        try:
            transformed = main(zone)
        except AssertionError:
            continue
        if transformed.shape != zone.shape or np.array_equal(transformed, zone):
            continue
        return zone

    return None


def _place_zones(zones: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((GRID_LIMIT, GRID_LIMIT), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)

    for zone in sorted(zones, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        zone_h, zone_w = zone.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = RNG.randint(0, GRID_LIMIT - zone_h)
            left = RNG.randint(0, GRID_LIMIT - zone_w)
            r0 = max(0, top - 1)
            r1 = min(GRID_LIMIT, top + zone_h + 1)
            c0 = max(0, left - 1)
            c1 = min(GRID_LIMIT, left + zone_w + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            canvas[top : top + zone_h, left : left + zone_w] = zone
            occupied[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    return canvas


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    points = np.argwhere(canvas != BG)
    assert points.size > 0
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)

    top_margin = RNG.randint(0, min(int(r0), 4))
    left_margin = RNG.randint(0, min(int(c0), 4))
    bottom_margin = RNG.randint(0, min(int(canvas.shape[0] - 1 - r1), 4))
    right_margin = RNG.randint(0, min(int(canvas.shape[1] - 1 - c1), 4))

    return canvas[
        int(r0 - top_margin) : int(r1 + bottom_margin + 1),
        int(c0 - left_margin) : int(c1 + right_margin + 1),
    ].copy()


def _is_valid_generated(grid: np.ndarray) -> bool:
    try:
        axes, marker_colors = _find_axes(grid)
        scene = _erase_markers(grid, marker_colors)
        for axis in axes:
            side_a, side_b = _partition_axis_pixels(scene, axis)
            if not side_a and not side_b:
                return False
            if len(side_a) == len(side_b):
                return False
        out = main(grid)
    except AssertionError:
        return False

    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    axis_count = RNG.choice(AXIS_COUNT_CHOICES)
    marker_colors = RNG.sample(list(PATTERN_COLORS), axis_count)
    pattern_palette = tuple(color for color in PATTERN_COLORS if color not in marker_colors)

    zones: list[np.ndarray] = []
    for marker_color in marker_colors:
        zone = _build_zone(int(marker_color), pattern_palette)
        if zone is None:
            return None
        zones.append(zone)

    canvas = _place_zones(zones)
    if canvas is None:
        return None

    grid = _crop_canvas(canvas)
    if max(grid.shape) > GRID_LIMIT:
        return None
    if not _is_valid_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-axis input")
