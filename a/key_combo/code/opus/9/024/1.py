from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 0
GREY = 5
BAR_COLOR_POOL = [1, 2, 3, 4, 5]
MIN_BAR_COUNT = 3
MAX_BAR_COUNT = 5
MIN_BAR_HEIGHT = 3
MAX_BAR_HEIGHT = 10
MIN_RECT_COUNT = 2
MAX_RECT_COUNT = 3
MAX_GRID_SIDE = 29
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords_arr.shape[0]),
                }
            )

    return components


def _is_filled_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _parse_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "background must be black"

    components = _find_components(g, bg)
    bars: list[dict] = []
    rectangles: list[dict] = []

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        area = int(comp["area"])
        color = int(comp["color"])

        if width == 1 and area == height and MIN_BAR_HEIGHT <= height <= MAX_BAR_HEIGHT:
            bars.append(
                {
                    "color": color,
                    "height": height,
                    "bbox": comp["bbox"],
                    "r0": r0,
                    "r1": r1,
                    "c": c0,
                }
            )
            continue

        if color == GREY and height >= 5 and width >= 5 and area == height * width and _is_filled_rect(g, comp["bbox"], GREY):
            rectangles.append(
                {
                    "bbox": comp["bbox"],
                    "height": height,
                    "width": width,
                }
            )
            continue

        raise AssertionError("unexpected foreground object")

    assert MIN_BAR_COUNT <= len(bars) <= MAX_BAR_COUNT, "expected 3-5 bars"
    assert MIN_RECT_COUNT <= len(rectangles) <= MAX_RECT_COUNT, "expected 2-3 rectangles"
    assert len({bar["color"] for bar in bars}) == len(bars), "bar colors must be distinct"
    assert len({bar["height"] for bar in bars}) == len(bars), "bar heights must be distinct"
    assert len({bar["r1"] for bar in bars}) == 1, "bars must be bottom-aligned"

    bar_right = max(bar["c"] for bar in bars)
    bar_bottom = max(bar["r1"] for bar in bars)
    rect_left = min(rect["bbox"][1] for rect in rectangles)
    rect_top = min(rect["bbox"][0] for rect in rectangles)
    assert bar_right < rect_left or bar_bottom < rect_top, "bars must sit left of or above all rectangles"

    palette_len = len(bars)
    for rect in rectangles:
        max_depth = (min(rect["height"], rect["width"]) - 1) // 2
        assert max_depth >= palette_len, "rectangles must expose the full palette cycle"

    return bars, rectangles


def _palette_from_bars(bars: list[dict]) -> list[int]:
    ranked = sorted(bars, key=lambda bar: (-int(bar["height"]), int(bar["c"])))
    return [int(bar["color"]) for bar in ranked]


def _transform(grid: np.ndarray, bars: list[dict], rectangles: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    palette = _palette_from_bars(bars)
    k = len(palette)

    for bar in bars:
        out[int(bar["r0"]) : int(bar["r1"]) + 1, int(bar["c"])] = BG

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                depth = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = palette[depth % k]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode a palette by ranking vertical bar heights, then repaint gray rectangles with concentric bands from that palette.

    Input structure:
    - The background is black.
    - There are 3-5 one-cell-wide vertical bars with distinct colors and distinct heights; all bars share one bottom row.
    - There are 2-3 solid gray rectangles, each at least 5x5, separated from the bars and from each other.

    Rule:
    - Sort the bars by height from tallest to shortest, breaking ties by left-to-right column order, to obtain the palette.
    - Erase every bar back to black.
    - For each gray rectangle, compute each cell's Chebyshev distance to that rectangle's boundary and recolor the cell with palette[distance mod palette_length].
    - Leave every non-rectangle background cell black.

    Key insight:
    - The bar heights are a global ranking code for palette order; each rectangle then uses that shared order to paint its own concentric frames.
    """
    g = np.asarray(grid, dtype=int)
    bars, rectangles = _parse_objects(g)
    return _transform(g, bars, rectangles)


def _is_monotone(seq: list[int]) -> bool:
    return seq == sorted(seq) or seq == sorted(seq, reverse=True)


def _anti_shortcut_ok(bar_colors: list[int], heights: list[int]) -> bool:
    if _is_monotone(heights):
        return False
    palette = [color for _height, color in sorted(zip(heights, bar_colors), key=lambda pair: -pair[0])]
    return palette != sorted(bar_colors) and palette != sorted(bar_colors, reverse=True)


def _sample_bar_colors(bar_count: int) -> list[int]:
    if bar_count == len(BAR_COLOR_POOL):
        colors = BAR_COLOR_POOL[:]
        random.shuffle(colors)
        return colors

    if random.random() < 0.6:
        others = [color for color in BAR_COLOR_POOL if color != GREY]
        colors = [GREY] + random.sample(others, bar_count - 1)
        random.shuffle(colors)
        return colors

    return random.sample(BAR_COLOR_POOL, bar_count)


def _sample_bars(bar_count: int) -> tuple[list[int], list[int]]:
    for _ in range(200):
        colors = _sample_bar_colors(bar_count)
        heights = random.sample(range(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT + 1), bar_count)
        if _anti_shortcut_ok(colors, heights):
            return colors, heights
    raise RuntimeError("failed to sample informative bars")


@lru_cache(None)
def _can_partition(height: int, width: int, count: int, min_side: int) -> bool:
    if count == 1:
        return height >= min_side and width >= min_side

    for first_count in range(1, count):
        second_count = count - first_count
        for gap in (1, 2):
            for first_h in range(min_side, height - gap - min_side + 1):
                second_h = height - gap - first_h
                if _can_partition(first_h, width, first_count, min_side) and _can_partition(second_h, width, second_count, min_side):
                    return True
            for first_w in range(min_side, width - gap - min_side + 1):
                second_w = width - gap - first_w
                if _can_partition(height, first_w, first_count, min_side) and _can_partition(height, second_w, second_count, min_side):
                    return True

    return False


def _split_region(
    region: tuple[int, int, int, int], orient: str, span: int, gap: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = region
    if orient == "h":
        return (r0, c0, r0 + span - 1, c1), (r0 + span + gap, c0, r1, c1)
    return (r0, c0, r1, c0 + span - 1), (r0, c0 + span + gap, r1, c1)


def _partition_region(
    region: tuple[int, int, int, int], count: int, min_side: int
) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = region
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if count == 1:
        if height >= min_side and width >= min_side:
            return [region]
        return None

    options: list[tuple[str, int, int, int]] = []
    for first_count in range(1, count):
        second_count = count - first_count
        for gap in (1, 2):
            for first_h in range(min_side, height - gap - min_side + 1):
                second_h = height - gap - first_h
                if _can_partition(first_h, width, first_count, min_side) and _can_partition(second_h, width, second_count, min_side):
                    options.append(("h", first_count, first_h, gap))
            for first_w in range(min_side, width - gap - min_side + 1):
                second_w = width - gap - first_w
                if _can_partition(height, first_w, first_count, min_side) and _can_partition(height, second_w, second_count, min_side):
                    options.append(("v", first_count, first_w, gap))

    if not options:
        return None

    orient, first_count, span, gap = random.choice(options)
    left_region, right_region = _split_region(region, orient, span, gap)
    first_parts = _partition_region(left_region, first_count, min_side)
    second_parts = _partition_region(right_region, count - first_count, min_side)
    if first_parts is None or second_parts is None:
        return None

    parts = first_parts + second_parts
    random.shuffle(parts)
    return parts


def _sample_rectangles(
    regions: list[tuple[int, int, int, int]], min_side: int
) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for r0, c0, r1, c1 in regions:
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        rect_h = random.randint(min_side, region_h)
        rect_w = random.randint(min_side, region_w)
        top = random.randint(r0, r1 - rect_h + 1)
        left = random.randint(c0, c1 - rect_w + 1)
        rectangles.append((top, left, top + rect_h - 1, left + rect_w - 1))
    return rectangles


def _build_left_layout(
    height: int, width: int, bar_colors: list[int], heights: list[int], rect_count: int, min_side: int
) -> np.ndarray | None:
    spacing = random.choice([2, 3])
    start_col = random.randint(0, 2)
    columns = [start_col + i * spacing for i in range(len(bar_colors))]
    if columns[-1] >= width:
        return None

    rect_col0 = columns[-1] + random.randint(2, 3)
    right_width = width - rect_col0
    if right_width < min_side or not _can_partition(height, right_width, rect_count, min_side):
        return None

    regions = _partition_region((0, rect_col0, height - 1, width - 1), rect_count, min_side)
    if regions is None:
        return None

    max_height = max(heights)
    bottom = random.randint(max_height - 1, height - 1)
    rects = _sample_rectangles(regions, min_side)

    grid = np.zeros((height, width), dtype=int)
    for color, bar_height, col in zip(bar_colors, heights, columns):
        grid[bottom - bar_height + 1 : bottom + 1, col] = color
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _build_top_layout(
    height: int, width: int, bar_colors: list[int], heights: list[int], rect_count: int, min_side: int
) -> np.ndarray | None:
    spacing = random.choice([2, 3])
    max_start = width - 1 - spacing * (len(bar_colors) - 1)
    if max_start <= 0:
        return None
    start_col = random.randint(0, max_start)
    columns = [start_col + i * spacing for i in range(len(bar_colors))]

    max_height = max(heights)
    bar_bottom_max = min(11, height - min_side - 2)
    if max_height - 1 > bar_bottom_max:
        return None
    bottom = random.randint(max_height - 1, bar_bottom_max)

    rect_row0 = bottom + random.randint(2, 3)
    lower_height = height - rect_row0
    if lower_height < min_side or not _can_partition(lower_height, width, rect_count, min_side):
        return None

    regions = _partition_region((rect_row0, 0, height - 1, width - 1), rect_count, min_side)
    if regions is None:
        return None

    rects = _sample_rectangles(regions, min_side)
    grid = np.zeros((height, width), dtype=int)
    for color, bar_height, col in zip(bar_colors, heights, columns):
        grid[bottom - bar_height + 1 : bottom + 1, col] = color
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bars, rectangles = _parse_objects(grid)
        output = main(grid)
    except AssertionError:
        return False

    expected = _transform(grid, bars, rectangles)
    if not np.array_equal(output, expected):
        return False

    palette = _palette_from_bars(bars)
    if len(set(palette)) != len(palette):
        return False

    heights_by_col = [bar["height"] for bar in sorted(bars, key=lambda bar: bar["c"])]
    colors_by_col = [bar["color"] for bar in sorted(bars, key=lambda bar: bar["c"])]
    if not _anti_shortcut_ok(colors_by_col, heights_by_col):
        return False

    if np.array_equal(output, grid):
        return False

    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        patch = output[r0 : r1 + 1, c0 : c1 + 1]
        if set(np.unique(patch).tolist()) != set(palette):
            return False
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        center_depth = min(center_r - r0, r1 - center_r, center_c - c0, c1 - center_c)
        if int(patch[center_r - r0, center_c - c0]) != int(palette[center_depth % len(palette)]):
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    bar_count = random.randint(MIN_BAR_COUNT, MAX_BAR_COUNT)
    rect_count = random.randint(MIN_RECT_COUNT, MAX_RECT_COUNT)
    bar_colors, heights = _sample_bars(bar_count)
    min_side = 2 * bar_count + 1

    height = random.randint(max(20, min_side + 3), MAX_GRID_SIDE)
    width = random.randint(max(20, min_side + 3), MAX_GRID_SIDE)
    layout = random.choice(["left", "top"])

    if layout == "left":
        grid = _build_left_layout(height, width, bar_colors, heights, rect_count, min_side)
    else:
        grid = _build_top_layout(height, width, bar_colors, heights, rect_count, min_side)

    if grid is None or not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
