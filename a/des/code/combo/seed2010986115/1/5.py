from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
LINE_COLOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400

BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2), (1, 1)),                  # T
    ((0, 0), (1, 0), (2, 0), (2, 1)),                  # L
    ((0, 0), (0, 1), (1, 1), (1, 2)),                  # Z
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),         # P-like
    ((0, 0), (0, 1), (1, 1)),                          # small elbow
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),         # plus
)


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, height, width):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(coords)
    return components


def _monochrome_components(grid: np.ndarray, ignore_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if seen[r, c] or color in ignore_colors:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _shape_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _is_block_component(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return component["area"] == 4 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2


def _is_straight_segment(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return component["area"] == height * width and (height == 1 or width == 1)


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(rows) == 1, "expected a unique full-width separator row"
    row = int(rows[0])
    return row, int(grid[row, 0])


def _bottom_swap_mapping(bottom: np.ndarray, top_colors: set[int], bg: int) -> dict[int, int]:
    components = _monochrome_components(bottom, ignore_colors={bg})
    assert components, "expected a non-empty graph region"

    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    node_components: list[dict] = []
    line_color: int | None = None
    node_colors: set[int] = set()
    for color, color_components in by_color.items():
        all_blocks = all(_is_block_component(component) for component in color_components)
        all_lines = all(
            _is_straight_segment(component) and not _is_block_component(component)
            for component in color_components
        )
        assert all_blocks or all_lines, "bottom region contains an unexpected component shape"
        if all_blocks:
            node_colors.add(color)
            node_components.extend(color_components)
        else:
            assert line_color is None, "expected at most one line color"
            line_color = color

    assert node_colors <= top_colors, "graph block colors must appear in the upper objects"

    node_id = np.full(bottom.shape, -1, dtype=int)
    nodes: list[dict] = []
    for idx, component in enumerate(node_components):
        nodes.append(component)
        for r, c in component["coords"]:
            node_id[r, c] = idx

    mapping: dict[int, int] = {}
    if line_color is None:
        return mapping

    for line_cells in _components_from_mask(bottom == line_color):
        touched: set[int] = set()
        for r, c in line_cells:
            for nr, nc in _neighbors4(r, c, bottom.shape[0], bottom.shape[1]):
                candidate = int(node_id[nr, nc])
                if candidate >= 0:
                    touched.add(candidate)
        assert len(touched) == 2, "every line must connect exactly two blocks"
        node_a, node_b = sorted(touched)
        color_a = int(nodes[node_a]["color"])
        color_b = int(nodes[node_b]["color"])
        assert color_a != color_b, "swapped colors must differ"
        if color_a in mapping:
            assert mapping[color_a] == color_b, "conflicting color swap for first endpoint"
        if color_b in mapping:
            assert mapping[color_b] == color_a, "conflicting color swap for second endpoint"
        mapping[color_a] = color_b
        mapping[color_b] = color_a

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the colors of upper objects according to paired 2x2 color blocks in the lower graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    separator_row, _separator_color = _find_separator_row(g, bg)
    assert 0 < separator_row < g.shape[0] - 1, "separator must split the grid into two non-empty regions"

    top = g[:separator_row, :]
    bottom = g[separator_row + 1 :, :]

    top_components = _monochrome_components(top, ignore_colors={bg})
    assert len(top_components) >= 2, "expected multiple upper objects"
    top_shapes = {_shape_signature(component["coords"]) for component in top_components}
    assert len(top_shapes) == 1, "upper objects must share the same shape"
    top_colors = {int(component["color"]) for component in top_components}

    mapping = _bottom_swap_mapping(bottom, top_colors, bg)

    out = g.copy()
    top_out = out[:separator_row, :]
    for src_color, dst_color in mapping.items():
        top_out[top == src_color] = dst_color
    return out


def _normalize_shape(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _random_shape() -> list[tuple[int, int]]:
    coords = list(random.choice(BASE_SHAPES))
    if random.random() < 0.5:
        coords = [(r, -c) for r, c in coords]
    for _ in range(random.randrange(4)):
        coords = [(-c, r) for r, c in coords]
    return _normalize_shape(coords)


def _shape_size(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(height, r + 2)):
            for nc in range(max(0, c - 1), min(width, c + 2)):
                blocked[nr, nc] = True


def _place_top_objects(
    top_height: int, width: int, shape: list[tuple[int, int]], colors: list[int]
) -> list[tuple[list[tuple[int, int]], int]] | None:
    shape_h, shape_w = _shape_size(shape)
    if top_height < shape_h or width < shape_w:
        return None

    blocked = np.zeros((top_height, width), dtype=bool)
    placements: list[tuple[list[tuple[int, int]], int]] = []
    for color in colors:
        anchors = [(r, c) for r in range(top_height - shape_h + 1) for c in range(width - shape_w + 1)]
        random.shuffle(anchors)
        placed = False
        for base_r, base_c in anchors:
            cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if all(not blocked[r, c] for r, c in cells):
                placements.append((cells, color))
                _mark_blocked(blocked, cells)
                placed = True
                break
        if not placed:
            return None
    return placements


def _single_block_gadget(color: int) -> dict:
    cells = [(r, c, color) for r in range(2) for c in range(2)]
    return {"height": 2, "width": 2, "cells": cells}


def _paired_gadget(color_a: int, color_b: int) -> dict:
    if random.random() < 0.5:
        color_a, color_b = color_b, color_a
    orient = random.choice(("horizontal", "vertical"))
    line_len = random.randint(1, 4)
    attach = random.randint(0, 1)

    cells: list[tuple[int, int, int]] = []
    if orient == "horizontal":
        for r in range(2):
            for c in range(2):
                cells.append((r, c, color_a))
                cells.append((r, line_len + 2 + c, color_b))
        for c in range(2, line_len + 2):
            cells.append((attach, c, LINE_COLOR))
        return {"height": 2, "width": line_len + 4, "cells": cells}

    for r in range(2):
        for c in range(2):
            cells.append((r, c, color_a))
            cells.append((line_len + 2 + r, c, color_b))
    for r in range(2, line_len + 2):
        cells.append((r, attach, LINE_COLOR))
    return {"height": line_len + 4, "width": 2, "cells": cells}


def _place_gadgets(bottom_height: int, width: int, gadgets: list[dict]) -> np.ndarray | None:
    blocked = np.zeros((bottom_height, width), dtype=bool)
    canvas = np.full((bottom_height, width), BG, dtype=int)
    for gadget in sorted(gadgets, key=lambda item: item["height"] * item["width"], reverse=True):
        gh = int(gadget["height"])
        gw = int(gadget["width"])
        if gh > bottom_height or gw > width:
            return None
        anchors = [(r, c) for r in range(bottom_height - gh + 1) for c in range(width - gw + 1)]
        random.shuffle(anchors)
        placed = False
        for base_r, base_c in anchors:
            placed_cells = [(base_r + dr, base_c + dc, color) for dr, dc, color in gadget["cells"]]
            if all(not blocked[r, c] for r, c, _ in placed_cells):
                for r, c, color in placed_cells:
                    canvas[r, c] = color
                _mark_blocked(blocked, [(r, c) for r, c, _ in placed_cells])
                placed = True
                break
        if not placed:
            return None
    return canvas


def _apply_mapping_to_top(grid: np.ndarray, separator_row: int, mapping: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    original_top = grid[:separator_row, :]
    top_out = out[:separator_row, :]
    for src_color, dst_color in mapping.items():
        top_out[original_top == src_color] = dst_color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 26)
    width = random.randint(18, 26)
    separator_row = random.randint(6, height - 8)
    bottom_height = height - separator_row - 1

    shape = _random_shape()
    distinct_color_count = random.randint(3, min(6, len(OBJECT_COLORS)))
    distinct_colors = random.sample(OBJECT_COLORS, distinct_color_count)

    object_count = random.randint(distinct_color_count, min(distinct_color_count + 2, 7))
    top_colors = distinct_colors + [random.choice(distinct_colors) for _ in range(object_count - distinct_color_count)]
    random.shuffle(top_colors)

    top_objects = _place_top_objects(separator_row, width, shape, top_colors)
    if top_objects is None:
        return None

    shuffled_colors = distinct_colors[:]
    random.shuffle(shuffled_colors)
    pair_count = random.randint(1, len(distinct_colors) // 2)
    paired_colors = shuffled_colors[: 2 * pair_count]
    unpaired_colors = shuffled_colors[2 * pair_count :]

    gadgets = [_paired_gadget(paired_colors[i], paired_colors[i + 1]) for i in range(0, len(paired_colors), 2)]
    gadgets.extend(_single_block_gadget(color) for color in unpaired_colors)
    random.shuffle(gadgets)
    bottom = _place_gadgets(bottom_height, width, gadgets)
    if bottom is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for cells, color in top_objects:
        for r, c in cells:
            grid[r, c] = color
    grid[separator_row, :] = SEPARATOR_COLOR
    grid[separator_row + 1 :, :] = bottom

    mapping: dict[int, int] = {}
    for i in range(0, len(paired_colors), 2):
        color_a = int(paired_colors[i])
        color_b = int(paired_colors[i + 1])
        mapping[color_a] = color_b
        mapping[color_b] = color_a

    expected = _apply_mapping_to_top(grid, separator_row, mapping)
    actual = main(grid)
    if np.array_equal(grid, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
