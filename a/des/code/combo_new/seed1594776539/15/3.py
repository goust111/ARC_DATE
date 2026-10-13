from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
PALETTE = tuple(range(1, 10))
COUNT_OPTIONS = (2, 4)
MAX_SIDE = 30
GENERATION_ATTEMPTS = 50
LAYOUT_ATTEMPTS = 80
AREA_GAP = 3
ROW_GAP_RANGE = (2, 4)
COL_GAP_RANGE = (2, 5)
CANVAS_SLACK = 6
COUNT_TO_DIM_BOUNDS = {
    2: (2, 10, 2, 10),
    4: (1, 6, 1, 7),
}
COUNT_TO_ROW_OPTIONS = {
    2: (1, 2),
    4: (2, 3),
}


def _neighbors4(r: int, c: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _colored_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = map(int, grid.shape)
    seen = np.zeros_like(grid, dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "coords": coords})
    return components


def _touches_edge(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in coords)


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG_COLOR):
        return BG_COLOR
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def main(grid: np.ndarray) -> np.ndarray:
    """Reassign loop colors by reversing the order of their enclosed black areas.

    Input structure:
    - Black is the background, and every non-black monochrome component is a one-pixel-thick simple closed loop.
    - The loops are disjoint, not nested, and each loop uses a different color.
    - Every loop encloses exactly one black background region.
    - The enclosed black regions all have different areas.

    Rule:
    - Find every enclosed black component that does not touch the grid edge.
    - Match each enclosed region to the unique loop color touching it.
    - Sort the loops by enclosed black area from smallest to largest.
    - Recolor the loop pixels with the same colors but in reversed sorted order, leaving loop shapes, positions, and black interiors unchanged.

    Key insight:
    - The visible color permutation is controlled by the size ranking of the black regions inside the loops, not by loop positions or color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = _background_color(g)
    loops = _colored_components(g, bg)
    assert len(loops) >= 2 and len(loops) % 2 == 0, "expected an even number of closed loops"
    loop_colors = [int(loop["color"]) for loop in loops]
    assert len(set(loop_colors)) == len(loop_colors), "each loop color must be unique"

    areas_by_color: dict[int, int] = {}
    for coords in _components_from_mask(g == bg):
        if _touches_edge(coords, height, width):
            continue
        neighbor_colors = {
            int(g[nr, nc])
            for r, c in coords
            for nr, nc in _neighbors4(r, c, height, width)
            if int(g[nr, nc]) != bg
        }
        assert len(neighbor_colors) == 1, "each enclosed region must belong to one loop"
        loop_color = neighbor_colors.pop()
        assert loop_color not in areas_by_color, "each loop must enclose exactly one region"
        areas_by_color[loop_color] = len(coords)

    assert len(areas_by_color) == len(loops), "every loop must enclose one black region"
    ordered_loops = sorted(loops, key=lambda loop: areas_by_color[int(loop["color"])])
    ordered_colors = [int(loop["color"]) for loop in ordered_loops]

    out = g.copy()
    for loop, new_color in zip(ordered_loops, reversed(ordered_colors)):
        for r, c in loop["coords"]:
            out[r, c] = new_color
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _sample_rectangles(count: int) -> list[dict[str, int]] | None:
    min_h, max_h, min_w, max_w = COUNT_TO_DIM_BOUNDS[count]
    candidates = [(inner_h, inner_w) for inner_h in range(min_h, max_h + 1) for inner_w in range(min_w, max_w + 1)]
    if len(candidates) < count:
        return None

    for _ in range(LAYOUT_ATTEMPTS):
        dims = random.sample(candidates, count)
        areas = sorted(inner_h * inner_w for inner_h, inner_w in dims)
        if len(set(areas)) != count:
            continue
        if any(b - a < AREA_GAP for a, b in zip(areas, areas[1:])):
            continue
        colors = random.sample(PALETTE, count)
        return [
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "area": inner_h * inner_w,
                "color": color,
            }
            for (inner_h, inner_w), color in zip(dims, colors)
        ]
    return None


def _pack_rows(specs: list[dict[str, int]]) -> tuple[int, int, list[dict[str, int]]] | None:
    count = len(specs)
    row_count = random.choice(COUNT_TO_ROW_OPTIONS[count])
    row_sizes = _positive_composition(count, row_count)

    shuffled = [dict(spec) for spec in specs]
    random.shuffle(shuffled)
    rows: list[dict[str, object]] = []
    index = 0
    max_width = 0
    total_height = 0

    for row_size in row_sizes:
        row_specs = shuffled[index : index + row_size]
        index += row_size
        random.shuffle(row_specs)
        gaps = [random.randint(*COL_GAP_RANGE) for _ in range(max(0, row_size - 1))]
        row_height = max(spec["inner_h"] + 2 for spec in row_specs) + random.randint(0, 2)
        row_width = sum(spec["inner_w"] + 2 for spec in row_specs) + sum(gaps)
        rows.append({"specs": row_specs, "gaps": gaps, "height": row_height, "width": row_width})
        max_width = max(max_width, row_width)
        total_height += row_height

    row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(max(0, row_count - 1))]
    total_height += sum(row_gaps)
    if total_height > MAX_SIDE or max_width > MAX_SIDE:
        return None

    grid_height = total_height + random.randint(0, min(CANVAS_SLACK, MAX_SIDE - total_height))
    grid_width = max_width + random.randint(0, min(CANVAS_SLACK, MAX_SIDE - max_width))
    top_margin = random.randint(0, grid_height - total_height)

    placed: list[dict[str, int]] = []
    cursor_top = top_margin
    for row_index, row in enumerate(rows):
        row_specs = row["specs"]
        row_height = int(row["height"])
        row_width = int(row["width"])
        row_left = random.randint(0, grid_width - row_width)
        cursor_left = row_left
        for spec_index, spec in enumerate(row_specs):
            outer_h = spec["inner_h"] + 2
            outer_w = spec["inner_w"] + 2
            top = cursor_top + random.randint(0, row_height - outer_h)
            placed.append({**spec, "top": top, "left": cursor_left})
            cursor_left += outer_w
            if spec_index < len(row["gaps"]):
                cursor_left += int(row["gaps"][spec_index])
        cursor_top += row_height
        if row_index < len(row_gaps):
            cursor_top += row_gaps[row_index]

    return grid_height, grid_width, placed


def _loop_coords(spec: dict[str, int]) -> list[tuple[int, int]]:
    top = spec["top"]
    left = spec["left"]
    bottom = top + spec["inner_h"] + 1
    right = left + spec["inner_w"] + 1
    coords: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return coords


def _render_scene(height: int, width: int, placed: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for spec in placed:
        color = spec["color"]
        for r, c in _loop_coords(spec):
            grid[r, c] = color
    return grid


def _expected_output(grid: np.ndarray, placed: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    ordered = sorted(placed, key=lambda spec: spec["area"])
    ordered_colors = [spec["color"] for spec in ordered]
    for spec, new_color in zip(ordered, reversed(ordered_colors)):
        for r, c in _loop_coords(spec):
            out[r, c] = new_color
    return out


def _gen_attempt(count: int) -> np.ndarray | None:
    specs = _sample_rectangles(count)
    if specs is None:
        return None

    packed = _pack_rows(specs)
    if packed is None:
        return None
    height, width, placed = packed
    grid = _render_scene(height, width, placed)
    expected = _expected_output(grid, placed)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    count = random.choice(COUNT_OPTIONS)
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid closed-loop recoloring scene")
