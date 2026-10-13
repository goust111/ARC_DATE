import random
from collections import Counter

import numpy as np


GREY = 5
GRID_H_RANGE = (15, 27)
GRID_W_RANGE = (15, 34)
CONTAINER_COUNT_RANGE = (1, 4)
INNER_WIDTH_RANGE = (1, 3)
LAYER_COUNT_RANGE = (2, 5)
SIDE_MARGIN_RANGE = (1, 4)
GAP_RANGE = (2, 5)
VERTICAL_SLACK_RANGE = (2, 8)
HORIZONTAL_SLACK_MAX = 10
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _container_from_component(
    grid: np.ndarray, points: list[tuple[int, int]], color: int, bg: int
) -> dict[str, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    inner_w = outer_w - 2
    layer_count = outer_h - 1
    if inner_w < 1 or inner_w > 3 or layer_count < 1:
        return None

    comp_mask = np.zeros((outer_h, outer_w), dtype=bool)
    for r, c in points:
        comp_mask[r - r0, c - c0] = True

    expected = np.zeros_like(comp_mask)
    expected[:-1, 0] = True
    expected[:-1, -1] = True
    expected[-1, :] = True
    if not np.array_equal(comp_mask, expected):
        return None

    # The interior must be filled with horizontal monochrome rows.
    for r in range(r0, r1):
        row = grid[r, c0 + 1 : c1]
        colors = set(map(int, row))
        if len(colors) != 1:
            return None
        value = next(iter(colors))
        if value == color or value == bg:
            return None

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "frame_color": int(color),
        "layer_count": int(layer_count),
        "inner_w": int(inner_w),
    }


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    containers: list[dict[str, int]] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        for comp in _components(grid == color):
            container = _container_from_component(grid, comp, int(color), int(bg))
            if container is not None:
                containers.append(container)
    containers.sort(key=lambda item: (item["r0"], item["c0"]))
    return containers


def _read_layer_colors(grid: np.ndarray, container: dict[str, int]) -> list[int]:
    r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
    colors: list[int] = []
    for r in range(r0, r1):
        row = grid[r, c0 + 1 : c1]
        value = int(row[0])
        assert bool((row == value).all()), "each layer must fill the container width"
        colors.append(value)
    return colors


def _find_key(grid: np.ndarray, container: dict[str, int], layer_colors: list[int], bg: int) -> tuple[int, int, list[int]]:
    r0, c0, c1 = container["r0"], container["c0"], container["c1"]
    layer_count = len(layer_colors)
    key_top = r0 - layer_count - 1
    assert key_top >= 0, "key must fit above the container"

    candidates: list[tuple[int, int, list[int]]] = []
    for kc in range(c0 + 1, c1):
        if int(grid[r0 - 1, kc]) != int(bg):
            continue
        if key_top > 0 and int(grid[key_top - 1, kc]) != int(bg):
            continue

        seq = [int(grid[r, kc]) for r in range(key_top, r0 - 1)]
        if any(color == int(bg) for color in seq):
            continue
        if Counter(seq) != Counter(layer_colors):
            continue

        clear_elsewhere = True
        for other_c in range(c0 + 1, c1):
            if other_c == kc:
                continue
            if not bool((grid[key_top : r0 - 1, other_c] == int(bg)).all()):
                clear_elsewhere = False
                break
        if clear_elsewhere:
            candidates.append((int(kc), int(key_top), seq))

    assert len(candidates) == 1, "each container must have exactly one matching key column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each container's stacked layers according to the color key above it.

    The key is read top-to-bottom and rewritten into the container top-to-bottom.
    The key column is then erased back to the background color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    containers = _find_containers(g, bg)
    assert containers, "no containers found"

    out = g.copy()
    for container in containers:
        r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
        layer_colors = _read_layer_colors(g, container)
        key_col, key_top, key_colors = _find_key(g, container, layer_colors, bg)

        for idx, r in enumerate(range(r0, r1)):
            out[r, c0 + 1 : c1] = int(key_colors[idx])
        out[key_top : r0 - 1, key_col] = int(bg)

    return out


def _permute_not_identity(values: list[int], rng: random.Random) -> list[int]:
    items = values[:]
    while True:
        rng.shuffle(items)
        if items != values:
            return items[:]


def _choose_dimensions(
    specs: list[dict[str, int]], rng: random.Random
) -> tuple[int, int, list[int], int]:
    outer_widths = [spec["inner_w"] + 2 for spec in specs]
    structure_heights = [2 * spec["layer_count"] + 2 for spec in specs]

    left_margin = rng.randint(*SIDE_MARGIN_RANGE)
    right_margin = rng.randint(*SIDE_MARGIN_RANGE)
    gaps = [rng.randint(*GAP_RANGE) for _ in range(max(0, len(specs) - 1))]
    base_width = left_margin + right_margin + sum(outer_widths) + sum(gaps)
    min_width = max(GRID_W_RANGE[0], base_width)
    max_width = min(GRID_W_RANGE[1], base_width + HORIZONTAL_SLACK_MAX)
    if min_width > max_width:
        raise ValueError("width out of range")
    width = rng.randint(min_width, max_width)
    left_margin += rng.randint(0, width - base_width)

    max_structure_h = max(structure_heights)
    min_height = max(GRID_H_RANGE[0], max_structure_h + 1)
    max_height = min(GRID_H_RANGE[1], max_structure_h + VERTICAL_SLACK_RANGE[1])
    if min_height > max_height:
        raise ValueError("height out of range")
    height = rng.randint(min_height, max_height)
    return height, width, gaps, left_margin


def _paint_container(
    grid: np.ndarray,
    *,
    r0: int,
    c0: int,
    inner_w: int,
    input_top_to_bottom: list[int],
    key_top_to_bottom: list[int],
    key_col_offset: int,
) -> None:
    layer_count = len(input_top_to_bottom)
    c1 = c0 + inner_w + 1
    r1 = r0 + layer_count

    grid[r0:r1 + 1, c0] = GREY
    grid[r0:r1 + 1, c1] = GREY
    grid[r1, c0:c1 + 1] = GREY

    for idx, color in enumerate(input_top_to_bottom):
        grid[r0 + idx, c0 + 1 : c1] = int(color)

    key_col = c0 + 1 + key_col_offset
    key_top = r0 - layer_count - 1
    for idx, color in enumerate(key_top_to_bottom):
        grid[key_top + idx, key_col] = int(color)


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]], bg: int) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        r0 = int(placement["r0"])
        c0 = int(placement["c0"])
        inner_w = int(placement["inner_w"])
        key_col = int(placement["key_col"])
        key_order = list(placement["key_order"])
        layer_count = len(key_order)
        c1 = c0 + inner_w + 1
        key_top = r0 - layer_count - 1

        for idx, color in enumerate(key_order):
            out[r0 + idx, c0 + 1 : c1] = int(color)
        out[key_top : r0 - 1, key_col] = int(bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    container_count = rng.randint(*CONTAINER_COUNT_RANGE)
    specs: list[dict[str, int]] = []
    for _ in range(container_count):
        specs.append(
            {
                "inner_w": rng.randint(*INNER_WIDTH_RANGE),
                "layer_count": rng.randint(*LAYER_COUNT_RANGE),
            }
        )

    try:
        height, width, gaps, left_margin = _choose_dimensions(specs, rng)
    except ValueError:
        return None

    bg_choices = [color for color in range(10) if color != GREY]
    bg = int(rng.choice(bg_choices))
    layer_palette = [color for color in range(10) if color not in (bg, GREY)]
    grid = np.full((height, width), bg, dtype=int)

    placements: list[dict[str, object]] = []
    x = left_margin
    for idx, spec in enumerate(specs):
        inner_w = int(spec["inner_w"])
        layer_count = int(spec["layer_count"])
        outer_w = inner_w + 2
        structure_h = 2 * layer_count + 2
        key_top = rng.randint(0, height - structure_h)
        r0 = key_top + layer_count + 1
        c0 = x
        colors = rng.sample(layer_palette, layer_count)
        input_order = _permute_not_identity(colors, rng)
        key_order = _permute_not_identity(input_order, rng)
        key_col_offset = rng.randint(0, inner_w - 1)

        _paint_container(
            grid,
            r0=r0,
            c0=c0,
            inner_w=inner_w,
            input_top_to_bottom=input_order,
            key_top_to_bottom=key_order,
            key_col_offset=key_col_offset,
        )

        placements.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "inner_w": int(inner_w),
                "key_col": int(c0 + 1 + key_col_offset),
                "key_order": key_order[:],
            }
        )
        x += outer_w
        if idx < len(gaps):
            x += gaps[idx]

    if _bg_color(grid) != bg:
        return None

    expected = _expected_output(grid, placements, bg)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
