from __future__ import annotations

from collections import Counter
import random

import numpy as np


COLORS = tuple(range(10))

# Solver assumptions / generation bounds.
GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
OBJECT_COUNT_RANGE = (2, 5)
BORDER_MARGIN = 1
SLOT_GAP = 1
SLOT_MIN_HEIGHT = 4
SLOT_MIN_WIDTH = 4
ODD_LAYER_OPTIONS = (3, 5)
EVEN_LAYER_OPTIONS = (2, 4, 6)
CORE_SMALL_CAP = 4
SINGLE_PIXEL_CORE_PROB = 0.25
MAX_OBJECT_AREA_FRACTION = 0.42
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        r0 = int(start_r)
        c0 = int(start_c)
        if seen[r0, c0]:
            continue

        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            if r > 0 and mask[r - 1, c] and not seen[r - 1, c]:
                seen[r - 1, c] = True
                stack.append((r - 1, c))
            if r + 1 < h and mask[r + 1, c] and not seen[r + 1, c]:
                seen[r + 1, c] = True
                stack.append((r + 1, c))
            if c > 0 and mask[r, c - 1] and not seen[r, c - 1]:
                seen[r, c - 1] = True
                stack.append((r, c - 1))
            if c + 1 < w and mask[r, c + 1] and not seen[r, c + 1]:
                seen[r, c + 1] = True
                stack.append((r, c + 1))
        components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _analyze_layers(subgrid: np.ndarray) -> tuple[int, int]:
    outer_color = int(subgrid[0, 0])
    if np.all(subgrid == outer_color):
        return 1, outer_color

    h, w = subgrid.shape
    assert h >= 3 and w >= 3, "non-uniform nested structure became too thin"
    assert np.all(subgrid[0, :] == outer_color), "top edge must be one shell color"
    assert np.all(subgrid[-1, :] == outer_color), "bottom edge must be one shell color"
    assert np.all(subgrid[:, 0] == outer_color), "left edge must be one shell color"
    assert np.all(subgrid[:, -1] == outer_color), "right edge must be one shell color"

    inner = subgrid[1:-1, 1:-1]
    assert inner.size > 0, "every shell must contain an inner layer"
    inner_layers, core_color = _analyze_layers(inner)
    return inner_layers + 1, core_color


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    structures: list[dict[str, int | tuple[int, int, int, int]]] = []

    for coords in _components4(g != bg):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        filled = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        filled[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert np.all(filled), "nested object occupancy must fill its bbox"

        subgrid = g[r0 : r1 + 1, c0 : c1 + 1]
        layers, core_color = _analyze_layers(subgrid)
        structures.append({"bbox": box, "layers": layers, "core_color": core_color})

    return bg, structures


def _paint_outer_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recognize rectangular nested shell objects. Odd layer counts lose the outer ring;
    even layer counts are filled entirely with the core color.
    """
    g = np.asarray(grid, dtype=int)
    bg, structures = _analyze_grid(g)
    out = g.copy()

    for structure in structures:
        box = structure["bbox"]
        layers = int(structure["layers"])
        core_color = int(structure["core_color"])
        r0, c0, r1, c1 = box

        if layers % 2 == 1:
            _paint_outer_ring(out, (r0, c0, r1, c1), bg)
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = core_color

    return out


def _partition_sizes(total: int, parts: int, min_size: int) -> list[int] | None:
    extra = int(total) - int(parts) * int(min_size)
    if extra < 0:
        return None
    sizes = [int(min_size) + int(x) for x in np.random.multinomial(extra, [1.0 / parts] * parts)]
    random.shuffle(sizes)
    return sizes


def _slot_intervals(length: int, parts: int, min_size: int) -> list[tuple[int, int]] | None:
    usable = int(length) - 2 * BORDER_MARGIN - SLOT_GAP * (int(parts) - 1)
    sizes = _partition_sizes(usable, parts, min_size)
    if sizes is None:
        return None

    pos = BORDER_MARGIN
    intervals: list[tuple[int, int]] = []
    for size in sizes:
        intervals.append((pos, pos + int(size) - 1))
        pos += int(size) + SLOT_GAP
    return intervals


def _build_object(layer_colors: list[int], core_h: int, core_w: int) -> np.ndarray:
    obj = np.full((int(core_h), int(core_w)), int(layer_colors[-1]), dtype=int)
    for color in reversed(layer_colors[:-1]):
        grown = np.full((obj.shape[0] + 2, obj.shape[1] + 2), int(color), dtype=int)
        grown[1:-1, 1:-1] = obj
        obj = grown
    return obj


def _unique_most_common(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    top = max(counts.values())
    return counts[int(bg)] == top and sum(count == top for count in counts.values()) == 1


def _sample_core_extent(limit: int) -> int:
    if limit <= 1:
        return 1
    small_limit = min(limit, CORE_SMALL_CAP)
    extent = random.randint(1, small_limit)
    if limit > small_limit and random.random() < 0.25:
        extent = random.randint(small_limit + 1, limit)
    return int(extent)


def _sample_layer_count(max_layers: int, parity: int) -> int | None:
    options = ODD_LAYER_OPTIONS if parity else EVEN_LAYER_OPTIONS
    valid = [count for count in options if count <= int(max_layers)]
    if not valid:
        return None
    return int(random.choice(valid))


def _layout_candidates(height: int, width: int, object_count: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for rows in range(1, 4):
        for cols in range(1, 4):
            if rows * cols < int(object_count):
                continue
            usable_h = int(height) - 2 * BORDER_MARGIN - SLOT_GAP * (rows - 1)
            usable_w = int(width) - 2 * BORDER_MARGIN - SLOT_GAP * (cols - 1)
            if usable_h >= rows * SLOT_MIN_HEIGHT and usable_w >= cols * SLOT_MIN_WIDTH:
                candidates.append((rows, cols))
    return candidates


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(COLORS)

    max_objects = min(OBJECT_COUNT_RANGE[1], 5 if height * width >= 480 else 4)
    object_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    layouts = _layout_candidates(height, width, object_count)
    if not layouts:
        return None

    rows, cols = random.choice(layouts)
    row_intervals = _slot_intervals(height, rows, SLOT_MIN_HEIGHT)
    col_intervals = _slot_intervals(width, cols, SLOT_MIN_WIDTH)
    if row_intervals is None or col_intervals is None:
        return None

    slots: list[dict[str, tuple[int, int] | int]] = []
    for row_box in row_intervals:
        for col_box in col_intervals:
            slot_h = int(row_box[1] - row_box[0] + 1)
            slot_w = int(col_box[1] - col_box[0] + 1)
            max_layers = 1 + min((slot_h - 1) // 2, (slot_w - 1) // 2)
            slots.append({"rows": row_box, "cols": col_box, "max_layers": max_layers})

    chosen_slots: list[dict[str, tuple[int, int] | int]] | None = None
    odd_target_choices = [object_count // 2] if object_count % 2 == 0 else [object_count // 2, object_count // 2 + 1]
    odd_target_choices = [count for count in odd_target_choices if count >= 1]
    odd_target = random.choice(odd_target_choices)
    for _ in range(40):
        sample = random.sample(slots, object_count)
        odd_capable = [i for i, slot in enumerate(sample) if _sample_layer_count(int(slot["max_layers"]), 1) is not None]
        if len(odd_capable) >= odd_target:
            chosen_slots = sample
            break
    if chosen_slots is None:
        return None

    odd_capable = [i for i, slot in enumerate(chosen_slots) if _sample_layer_count(int(slot["max_layers"]), 1) is not None]
    odd_indices = set(random.sample(odd_capable, odd_target))

    objects: list[tuple[np.ndarray, int, int]] = []
    total_area = 0
    available_colors = [color for color in COLORS if color != bg]

    for idx, slot in enumerate(chosen_slots):
        slot_rows = slot["rows"]
        slot_cols = slot["cols"]
        assert isinstance(slot_rows, tuple) and isinstance(slot_cols, tuple)

        parity = 1 if idx in odd_indices else 0
        layer_count = _sample_layer_count(int(slot["max_layers"]), parity)
        if layer_count is None or len(available_colors) < layer_count:
            return None

        slot_h = int(slot_rows[1] - slot_rows[0] + 1)
        slot_w = int(slot_cols[1] - slot_cols[0] + 1)
        max_core_h = slot_h - 2 * (layer_count - 1)
        max_core_w = slot_w - 2 * (layer_count - 1)
        if max_core_h < 1 or max_core_w < 1:
            return None

        if random.random() < SINGLE_PIXEL_CORE_PROB:
            core_h = 1
            core_w = 1
        else:
            core_h = _sample_core_extent(max_core_h)
            core_w = _sample_core_extent(max_core_w)

        layer_colors = random.sample(available_colors, layer_count)
        obj = _build_object(layer_colors, core_h, core_w)
        obj_h, obj_w = map(int, obj.shape)
        if obj_h > slot_h or obj_w > slot_w:
            return None

        top = random.randint(slot_rows[0], slot_rows[1] - obj_h + 1)
        left = random.randint(slot_cols[0], slot_cols[1] - obj_w + 1)
        objects.append((obj, top, left))
        total_area += obj_h * obj_w

    if total_area > int(height * width * MAX_OBJECT_AREA_FRACTION):
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    for obj, top, left in objects:
        obj_h, obj_w = map(int, obj.shape)
        grid[top : top + obj_h, left : left + obj_w] = obj

    if not _unique_most_common(grid, bg):
        return None

    parsed_bg, structures = _analyze_grid(grid)
    if parsed_bg != bg or len(structures) != object_count:
        return None
    if {int(structure["layers"]) % 2 for structure in structures} != {0, 1}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-shell ARC input")
