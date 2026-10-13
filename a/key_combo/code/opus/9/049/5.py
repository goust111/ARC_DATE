from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Under ARC's visible 0-9 palette and the puzzle's "legend outputs cannot reuse
# object colors" rule, fully valid instances can use 3 or 4 objects.
OBJECT_COUNT_CHOICES = (3, 4)
GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 19)
AREA_CHOICES = tuple(range(3, 16))

GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_hw_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return int(max_r + 1), int(max_c + 1)


def _find_components(grid: np.ndarray, blocked_mask: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            if blocked_mask is not None and blocked_mask[r, c]:
                seen[r, c] = True
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    if blocked_mask is not None and blocked_mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": _bbox(coords),
                }
            )
    return objects


def _detect_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray, list[int], list[int]]:
    h, w = grid.shape
    assert h >= 3 and w >= 3

    c0 = w
    while c0 > 0 and int(grid[0, c0 - 1]) != BG and int(grid[1, c0 - 1]) != BG:
        c0 -= 1
    n = int(w - c0)
    assert 3 <= n <= 5, "legend width must match the object count"

    legend = np.asarray(grid[0:2, c0:w], dtype=int)
    assert legend.shape == (2, n)
    assert bool(np.all(legend != BG)), "legend must be a full 2xN non-background rectangle"
    if c0 > 0:
        assert int(grid[0, c0 - 1]) == BG or int(grid[1, c0 - 1]) == BG
    assert bool(np.all(grid[2, c0:w] == BG)), "legend must be visually isolated below"

    top = [int(v) for v in legend[0]]
    bottom = [int(v) for v in legend[1]]
    assert len(set(top)) == n, "legend keys must be unique"
    assert len(set(bottom)) == n, "legend values must be unique"
    assert set(top).isdisjoint(bottom), "legend outputs may not reuse object colors"

    mask = np.zeros((h, w), dtype=bool)
    mask[0:2, c0:w] = True
    return dict(zip(top, bottom)), mask, top, bottom


def _ring_output(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.full((side, side), int(colors[0]), dtype=int)
    for layer, color in enumerate(colors[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Redraw scattered objects as concentric square rings whose order comes from area and whose colors come from a corner legend.

    Input structure:
    - A black background contains 3-5 monochromatic 4-connected foreground objects with distinct colors and distinct pixel counts.
    - A strict 2xN legend is anchored in the top-right corner; its top row lists the object colors and its bottom row lists replacement colors.
    - The legend is not part of the sortable objects, and its replacement colors do not appear on the objects themselves.

    Rule:
    - Identify the top-right 2xN legend and read each column as object-color -> output-color.
    - Excluding the legend, find all foreground objects, measure their areas, and sort them from largest to smallest.
    - Build a square output of side 2N-1 and draw concentric 1-cell-thick square rings from outside to inside using the mapped color of each object in that area order; the smallest object becomes the center pixel.

    Key insight:
    - Each object contributes two independent signals: its area chooses the ring layer, while its visible color only serves as the lookup key that decides the ring's final color.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask, _legend_top, _legend_bottom = _detect_legend(g)
    objects = _find_components(g, blocked_mask=legend_mask)

    assert len(objects) == len(mapping), "legend and object counts must agree"
    object_colors = {int(obj["color"]) for obj in objects}
    assert object_colors == set(mapping), "legend keys must match the object colors exactly"

    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas)), "object areas must be unique"

    ordered = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    ring_colors = [int(mapping[int(obj["color"])]) for obj in ordered]
    return _ring_output(ring_colors)


def _grow_shape(area: int, rng: random.Random) -> set[tuple[int, int]]:
    feasible = [
        (h, w)
        for h in range(2, 7)
        for w in range(2, 7)
        if h * w >= area and h * w > area and area >= h + w - 1
    ]
    for _ in range(SHAPE_ATTEMPTS):
        h, w = rng.choice(feasible)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < area:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(*cell)

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) != h or len(cols) != w:
            continue
        if len(cells) == h * w:
            continue

        neighbor_counts = [
            sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            for r, c in cells
        ]
        if min(neighbor_counts) >= 3:
            continue

        shape = _normalize(cells)
        if rng.random() < 0.5:
            max_r = max(r for r, _ in shape)
            shape = {(max_r - r, c) for r, c in shape}
        if rng.random() < 0.5:
            max_c = max(c for _, c in shape)
            shape = {(r, max_c - c) for r, c in shape}
        return _normalize(shape)
    raise RuntimeError(f"failed to generate a non-rectangular 4-connected shape of area {area}")


def _place_objects(
    height: int,
    width: int,
    shapes: list[set[tuple[int, int]]],
    colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    blocked[0:3, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    order = sorted(
        range(len(shapes)),
        key=lambda idx: _bbox_hw_from_cells(shapes[idx])[0] * _bbox_hw_from_cells(shapes[idx])[1],
        reverse=True,
    )
    for idx in order:
        shape = shapes[idx]
        color = int(colors[idx])
        sh, sw = _bbox_hw_from_cells(shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(3, height - sh)
            c0 = rng.randint(0, width - sw)
            coords = [(r0 + r, c0 + c) for r, c in shape]
            if any(blocked[r, c] for r, c in coords):
                continue
            for r, c in coords:
                grid[r, c] = color
            for r, c in coords:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if 0 <= rr < height and 0 <= cc < width:
                            blocked[rr, cc] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    mapping, legend_mask, legend_top, legend_bottom = _detect_legend(grid)
    objects = _find_components(grid, blocked_mask=legend_mask)
    assert len(objects) == len(mapping)
    assert {int(obj["color"]) for obj in objects} == set(mapping)

    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas))
    ordered_desc = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    ordered_asc = sorted(objects, key=lambda obj: int(obj["area"]))

    ordered_colors_desc = [int(obj["color"]) for obj in ordered_desc]
    mapped_desc = [int(mapping[c]) for c in ordered_colors_desc]
    mapped_asc = [int(mapping[int(obj["color"])]) for obj in ordered_asc]
    position_order = [
        int(obj["color"])
        for obj in sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    ]

    assert ordered_colors_desc != legend_top, "legend column order must not equal area order"
    assert mapped_desc != legend_bottom, "legend values alone must not determine the ring order"
    assert mapped_desc != ordered_colors_desc, "legend mapping must change the ring colors"
    assert mapped_desc != mapped_asc, "descending area order must matter"
    assert position_order != ordered_colors_desc, "top-left reading order must not equal area order"

    bbox_order = [
        int(obj["color"])
        for obj in sorted(
            objects,
            key=lambda obj: (
                (obj["bbox"][2] - obj["bbox"][0] + 1) * (obj["bbox"][3] - obj["bbox"][1] + 1),
                obj["bbox"][0],
                obj["bbox"][1],
            ),
            reverse=True,
        )
    ]
    assert bbox_order != ordered_colors_desc, "pixel area must not collapse to bbox-area ordering"

    out = main(grid)
    assert np.array_equal(out, _ring_output(mapped_desc))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(OBJECT_COUNT_CHOICES)
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(max(GRID_WIDTH_RANGE[0], n + 8), GRID_WIDTH_RANGE[1])

    object_colors = rng.sample(PALETTE, n)
    remaining = [color for color in PALETTE if color not in object_colors]
    mapped_colors = rng.sample(remaining, n)

    areas = rng.sample(AREA_CHOICES, n)
    shapes = [_grow_shape(area, rng) for area in areas]

    grid = _place_objects(height, width, shapes, object_colors, rng)
    if grid is None:
        return None

    area_order = [
        int(color)
        for _area, color in sorted(zip(areas, object_colors), reverse=True)
    ]
    if area_order == object_colors:
        return None

    legend_top = rng.sample(object_colors, k=n)
    legend_bottom = [mapped_colors[object_colors.index(color)] for color in legend_top]
    if area_order == legend_top:
        return None
    mapped_desc = [mapped_colors[object_colors.index(color)] for color in area_order]
    if mapped_desc == legend_bottom:
        return None
    if mapped_desc == area_order:
        return None

    legend_start = width - n
    grid[0, legend_start:width] = np.array(legend_top, dtype=int)
    grid[1, legend_start:width] = np.array(legend_bottom, dtype=int)

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-order + legend-mapping rings input")
