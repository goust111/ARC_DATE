from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _parse_vertical_template(
    grid: np.ndarray, coords: list[tuple[int, int]], dot_on_left: bool, bg: int
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if c1 - c0 != 1:
        return None
    dot_col = c0 if dot_on_left else c1
    bar_col = c1 if dot_on_left else c0
    bar_vals = grid[r0 : r1 + 1, bar_col]
    dot_vals = grid[r0 : r1 + 1, dot_col]
    if np.any(bar_vals == bg):
        return None
    if len(set(map(int, bar_vals.tolist()))) != 1:
        return None
    dot_rows = np.flatnonzero(dot_vals != bg)
    if len(dot_rows) != 1:
        return None
    if len(coords) != (r1 - r0 + 1) + 1:
        return None
    return int(r1 - r0 + 1), int(dot_vals[int(dot_rows[0])])


def _parse_horizontal_template(
    grid: np.ndarray, coords: list[tuple[int, int]], dot_on_top: bool, bg: int
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1:
        return None
    dot_row = r0 if dot_on_top else r1
    bar_row = r1 if dot_on_top else r0
    bar_vals = grid[bar_row, c0 : c1 + 1]
    dot_vals = grid[dot_row, c0 : c1 + 1]
    if np.any(bar_vals == bg):
        return None
    if len(set(map(int, bar_vals.tolist()))) != 1:
        return None
    dot_cols = np.flatnonzero(dot_vals != bg)
    if len(dot_cols) != 1:
        return None
    if len(coords) != (c1 - c0 + 1) + 1:
        return None
    return int(c1 - c0 + 1), int(dot_vals[int(dot_cols[0])])


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for comp in _components(grid != bg):
        if not _touches_boundary(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        parsed = None
        if c1 == w - 1:
            parsed = _parse_vertical_template(grid, comp, dot_on_left=False, bg=bg)
        elif c0 == 0:
            parsed = _parse_vertical_template(grid, comp, dot_on_left=True, bg=bg)
        elif r0 == 0:
            parsed = _parse_horizontal_template(grid, comp, dot_on_top=True, bg=bg)
        elif r1 == h - 1:
            parsed = _parse_horizontal_template(grid, comp, dot_on_top=False, bg=bg)
        if parsed is None:
            continue
        count, color = parsed
        if count in mapping:
            assert mapping[count] == color, "template count must map to a unique color"
        mapping[count] = color
    assert mapping, "expected at least one edge template"
    return mapping


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False
    coord_set = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _frame_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    frames: list[tuple[int, list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            if _touches_boundary(comp, h, w):
                continue
            if _is_rectangular_frame(comp):
                frames.append((color, comp))
    assert frames, "expected at least one rectangular fence"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow fence by matching its interior color diversity to an edge legend.

    Input structure:
    - The background is black and fills most of the grid.
    - Several one-pixel-thick hollow rectangular fences sit away from the border; each fence is monochrome, while its interior contains scattered colored pixels.
    - Along one grid edge, small legend templates touch the border: a same-colored bar of length 1..N encodes a count, and the adjacent single pixel gives the recolor target for that count.

    Rule:
    - Read every edge template as count -> color, where the bar length is the count and the neighboring single pixel is the mapped color.
    - For each hollow rectangular fence, count how many distinct interior colors appear inside it, ignoring the background and the fence color itself.
    - Recolor every pixel of that fence to the legend color associated with the counted value.
    - Leave the interior pixels and the background unchanged.

    Key insight:
    - The edge objects are a lookup table, and each fence is keyed not by its size or fill amount but by the number of different colors enclosed inside it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping = _legend_mapping(g, bg)
    out = g.copy()

    for fence_color, comp in _frame_components(g, bg):
        r0, c0, r1, c1 = _bbox(comp)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        distinct = {
            int(value)
            for value in np.unique(interior)
            if int(value) not in (bg, fence_color)
        }
        count = len(distinct)
        assert count in mapping, "every fence count must appear in the legend"
        new_color = int(mapping[count])
        for r, c in comp:
            out[r, c] = new_color

    return out


BACKGROUND = 0
GRID_MIN = 18
GRID_MAX = 30
LEGEND_GAP_CHOICES = (1, 2)
LEGEND_SIDE_CHOICES = ("left", "right", "top", "bottom")
LEGEND_COUNT_MIN = 2
LEGEND_COUNT_MAX = 5
FENCE_COUNT_RANGE = (1, 4)
FENCE_SIZE_MIN = 4
FENCE_SIZE_MAX = 8
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _legend_extent(max_count: int, gap: int) -> int:
    return max_count * (max_count + 1) // 2 + gap * (max_count - 1)


def _valid_legend_counts(length: int, gap: int) -> list[int]:
    return [
        count
        for count in range(LEGEND_COUNT_MIN, LEGEND_COUNT_MAX + 1)
        if _legend_extent(count, gap) <= length - 2
    ]


def _draw_template_entry(
    grid: np.ndarray,
    side: str,
    start_r: int,
    start_c: int,
    count: int,
    bar_color: int,
    out_color: int,
) -> None:
    if side == "right":
        grid[start_r : start_r + count, start_c] = bar_color
        dot_r = start_r + random.randrange(count)
        grid[dot_r, start_c + 1] = out_color
    elif side == "left":
        grid[start_r : start_r + count, start_c + 1] = bar_color
        dot_r = start_r + random.randrange(count)
        grid[dot_r, start_c] = out_color
    elif side == "bottom":
        grid[start_r, start_c : start_c + count] = bar_color
        dot_c = start_c + random.randrange(count)
        grid[start_r + 1, dot_c] = out_color
    else:  # top
        grid[start_r + 1, start_c : start_c + count] = bar_color
        dot_c = start_c + random.randrange(count)
        grid[start_r, dot_c] = out_color


def _place_legend(
    grid: np.ndarray, side: str, max_count: int, gap: int, bar_color: int, mapping: dict[int, int]
) -> None:
    h, w = grid.shape
    extent = _legend_extent(max_count, gap)
    if side in ("left", "right"):
        start_r = random.randint(1, h - extent - 1)
        col = 0 if side == "left" else w - 2
        cursor = start_r
        for count in range(1, max_count + 1):
            _draw_template_entry(grid, side, cursor, col, count, bar_color, mapping[count])
            cursor += count + gap
    else:
        start_c = random.randint(1, w - extent - 1)
        row = 0 if side == "top" else h - 2
        cursor = start_c
        for count in range(1, max_count + 1):
            _draw_template_entry(grid, side, row, cursor, count, bar_color, mapping[count])
            cursor += count + gap


def _reserved_strip(side: str) -> tuple[int, int, int, int]:
    if side == "left":
        return 1, 1, 4, 1
    if side == "right":
        return 1, 1, 1, 4
    if side == "top":
        return 4, 1, 1, 1
    return 1, 4, 1, 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _place_fences(
    grid: np.ndarray, side: str, fence_counts: list[int]
) -> list[tuple[tuple[int, int, int, int], int]]:
    h, w = grid.shape
    top_res, bottom_res, left_res, right_res = _reserved_strip(side)
    row_min = top_res
    row_max = h - 1 - bottom_res
    col_min = left_res
    col_max = w - 1 - right_res
    placements: list[tuple[tuple[int, int, int, int], int]] = []

    for count in fence_counts:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            max_height = min(FENCE_SIZE_MAX, row_max - row_min + 1)
            max_width = min(FENCE_SIZE_MAX, col_max - col_min + 1)
            if max_height < FENCE_SIZE_MIN or max_width < FENCE_SIZE_MIN:
                return []
            height = random.randint(FENCE_SIZE_MIN, max_height)
            width = random.randint(FENCE_SIZE_MIN, max_width)
            if (height - 2) * (width - 2) < count:
                continue
            r0 = random.randint(row_min, row_max - height + 1)
            c0 = random.randint(col_min, col_max - width + 1)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(_boxes_overlap(box, other_box) for other_box, _ in placements):
                continue
            placements.append((box, count))
            placed = True
            break
        if not placed:
            return []
    return placements


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _fill_interior(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    distinct_count: int,
    palette: list[int],
) -> None:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    random.shuffle(cells)
    chosen_colors = random.sample(palette, distinct_count)
    for idx, color in enumerate(chosen_colors):
        rr, cc = cells[idx]
        grid[rr, cc] = color
    extra_limit = min(len(cells) - distinct_count, distinct_count * 2)
    extra_count = random.randint(0, extra_limit) if extra_limit > 0 else 0
    for rr, cc in cells[distinct_count : distinct_count + extra_count]:
        grid[rr, cc] = random.choice(chosen_colors)


def _build_expected(
    grid: np.ndarray, fence_boxes: list[tuple[tuple[int, int, int, int], int]], mapping: dict[int, int], fence_color: int
) -> np.ndarray:
    out = grid.copy()
    for box, count in fence_boxes:
        r0, c0, r1, c1 = box
        out[r0, c0 : c1 + 1] = mapping[count]
        out[r1, c0 : c1 + 1] = mapping[count]
        out[r0 : r1 + 1, c0] = mapping[count]
        out[r0 : r1 + 1, c1] = mapping[count]
    return out


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(LEGEND_SIDE_CHOICES)
    gap = random.choice(LEGEND_GAP_CHOICES)
    axis_length = h if side in ("left", "right") else w
    valid_counts = _valid_legend_counts(axis_length, gap)
    if not valid_counts:
        return None
    max_count = random.choice(valid_counts)

    fence_color = random.randint(1, 9)
    available = [color for color in range(1, 10) if color != fence_color]
    mapping_colors = random.sample(available, max_count)
    mapping = {count: color for count, color in zip(range(1, max_count + 1), mapping_colors, strict=False)}
    bar_choices = [color for color in available if color not in mapping_colors]
    if not bar_choices:
        return None
    bar_color = random.choice(bar_choices)
    interior_palette = [color for color in available if color != bar_color]

    fence_total = random.randint(FENCE_COUNT_RANGE[0], FENCE_COUNT_RANGE[1])
    fence_counts = [random.randint(1, max_count) for _ in range(fence_total)]
    fence_boxes = _place_fences(np.zeros((h, w), dtype=int), side, fence_counts)
    if len(fence_boxes) != fence_total:
        return None

    grid = np.zeros((h, w), dtype=int)
    _place_legend(grid, side, max_count, gap, bar_color, mapping)
    for box, count in fence_boxes:
        _draw_frame(grid, box, fence_color)
        _fill_interior(grid, box, count, interior_palette)

    expected = _build_expected(grid, fence_boxes, mapping, fence_color)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _generate_attempt()
        if sample is None:
            continue
        grid, expected = sample
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
