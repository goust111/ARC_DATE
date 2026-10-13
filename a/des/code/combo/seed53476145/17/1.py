from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
BRIDGE_COLOR = 5

PAIR_COUNT_RANGE = (1, 3)
SINGLETON_COUNT_RANGE = (0, 3)
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (14, 28)
PAIR_BAND_SIZE_RANGE = (4, 8)
PAIR_GAP_RANGE = (1, 4)
RECT_SIDE_RANGE = (2, 6)
SINGLETON_SIDE_RANGE = (1, 4)
MIN_BAND_GAP = 1
GEN_ATTEMPTS = 300
PAIR_ATTEMPTS = 120
SINGLETON_ATTEMPTS = 120


@dataclass(frozen=True)
class Box:
    r0: int
    c0: int
    r1: int
    c1: int


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _solid_rectangles(grid: np.ndarray) -> list[tuple[int, Box]]:
    rects: list[tuple[int, Box]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BACKGROUND:
            continue
        for coords in _components(grid == color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            area = int(coords.shape[0])
            if area != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            rects.append((color, Box(r0, c0, r1, c1)))
    return rects


def _draw_box(grid: np.ndarray, box: Box, color: int) -> None:
    grid[box.r0 : box.r1 + 1, box.c0 : box.c1 + 1] = int(color)


def _swap_box(box: Box) -> Box:
    return Box(box.c0, box.r0, box.c1, box.r1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color rectangle pair with a grey bridge restricted to their
    shared row or column projection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    by_color: dict[int, list[Box]] = defaultdict(list)
    for color, box in _solid_rectangles(g):
        by_color[int(color)].append(box)

    for color, pair in by_color.items():
        if len(pair) != 2:
            continue
        a, b = pair

        row0 = max(a.r0, b.r0)
        row1 = min(a.r1, b.r1)
        col0 = max(a.c0, b.c0)
        col1 = min(a.c1, b.c1)

        left, right = sorted(pair, key=lambda box: (box.c0, box.r0))
        top, bottom = sorted(pair, key=lambda box: (box.r0, box.c0))

        horizontal = row0 <= row1 and left.c1 < right.c0 - 0
        vertical = col0 <= col1 and top.r1 < bottom.r0 - 0

        if horizontal and left.c1 + 1 <= right.c0 - 1:
            out[row0 : row1 + 1, left.c1 + 1 : right.c0] = BRIDGE_COLOR
        elif vertical and top.r1 + 1 <= bottom.r0 - 1:
            out[top.r1 + 1 : bottom.r0, col0 : col1 + 1] = BRIDGE_COLOR
        else:
            _ = color

    return out


def _sample_bands(total: int, count: int, size_range: tuple[int, int], min_gap: int) -> list[tuple[int, int]] | None:
    sizes = [random.randint(*size_range) for _ in range(count)]
    required = sum(sizes) + min_gap * (count - 1)
    if required > total:
        return None

    slack = total - required
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1

    bands: list[tuple[int, int]] = []
    pos = extras[0]
    for i, size in enumerate(sizes):
        bands.append((pos, size))
        pos += size
        if i < count - 1:
            pos += min_gap + extras[i + 1]
    return bands


def _sample_horizontal_pair(height: int, width: int, band_top: int, band_height: int) -> tuple[Box, Box, Box] | None:
    max_rect_side = min(RECT_SIDE_RANGE[1], max(RECT_SIDE_RANGE[0], band_height))
    if width < RECT_SIDE_RANGE[0] * 2 + PAIR_GAP_RANGE[0]:
        return None
    if max_rect_side < RECT_SIDE_RANGE[0]:
        return None

    for _ in range(PAIR_ATTEMPTS):
        w1 = random.randint(*RECT_SIDE_RANGE)
        w2 = random.randint(*RECT_SIDE_RANGE)
        if w1 + w2 + PAIR_GAP_RANGE[0] > width:
            continue
        max_gap = min(PAIR_GAP_RANGE[1], width - w1 - w2)
        if max_gap < PAIR_GAP_RANGE[0]:
            continue
        gap = random.randint(PAIR_GAP_RANGE[0], max_gap)
        total_width = w1 + gap + w2
        start_col = random.randint(0, width - total_width)

        h1 = random.randint(RECT_SIDE_RANGE[0], max_rect_side)
        h2 = random.randint(RECT_SIDE_RANGE[0], max_rect_side)
        overlap = random.randint(1, min(h1, h2))

        extra1 = h1 - overlap
        extra2 = h2 - overlap
        top1_extra = random.randint(0, extra1)
        top2_extra = random.randint(0, extra2)
        bottom1_extra = extra1 - top1_extra
        bottom2_extra = extra2 - top2_extra

        rect1_r0 = -top1_extra
        rect1_r1 = overlap - 1 + bottom1_extra
        rect2_r0 = -top2_extra
        rect2_r1 = overlap - 1 + bottom2_extra

        union_top = min(rect1_r0, rect2_r0)
        union_bottom = max(rect1_r1, rect2_r1)
        union_height = union_bottom - union_top + 1
        if union_height > band_height:
            continue

        offset = random.randint(0, band_height - union_height)
        shift = band_top + offset - union_top

        left = Box(shift + rect1_r0, start_col, shift + rect1_r1, start_col + w1 - 1)
        right_c0 = start_col + w1 + gap
        right = Box(shift + rect2_r0, right_c0, shift + rect2_r1, right_c0 + w2 - 1)
        bridge = Box(shift, left.c1 + 1, shift + overlap - 1, right.c0 - 1)
        return left, right, bridge
    return None


def _sample_vertical_pair(height: int, width: int, band_left: int, band_width: int) -> tuple[Box, Box, Box] | None:
    transposed = _sample_horizontal_pair(width, height, band_left, band_width)
    if transposed is None:
        return None
    a_t, b_t, bridge_t = transposed
    return _swap_box(a_t), _swap_box(b_t), _swap_box(bridge_t)


def _can_place(box: Box, occupied: np.ndarray, reserved: np.ndarray, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, box.r0 - pad)
    cc0 = max(0, box.c0 - pad)
    rr1 = min(h, box.r1 + pad + 1)
    cc1 = min(w, box.c1 + pad + 1)
    return not occupied[rr0:rr1, cc0:cc1].any() and not reserved[rr0:rr1, cc0:cc1].any()


def _render_scene(
    shape: tuple[int, int],
    pairs: list[tuple[int, Box, Box, Box]],
    singletons: list[tuple[int, Box]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BACKGROUND, dtype=int)
    out = np.full(shape, BACKGROUND, dtype=int)

    for color, box_a, box_b, bridge in pairs:
        _draw_box(inp, box_a, color)
        _draw_box(inp, box_b, color)
        _draw_box(out, box_a, color)
        _draw_box(out, box_b, color)
        _draw_box(out, bridge, BRIDGE_COLOR)

    for color, box in singletons:
        _draw_box(inp, box, color)
        _draw_box(out, box, color)

    return inp, out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    orientation = random.choice(("horizontal", "vertical"))
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    if orientation == "horizontal":
        bands = _sample_bands(height, pair_count, PAIR_BAND_SIZE_RANGE, MIN_BAND_GAP)
    else:
        bands = _sample_bands(width, pair_count, PAIR_BAND_SIZE_RANGE, MIN_BAND_GAP)
    if bands is None:
        return None

    palette = [color for color in range(1, 10) if color != BRIDGE_COLOR]
    needed_colors = pair_count + random.randint(*SINGLETON_COUNT_RANGE)
    if needed_colors > len(palette):
        return None
    colors = random.sample(palette, needed_colors)
    pair_colors = colors[:pair_count]
    singleton_colors = colors[pair_count:]

    pairs: list[tuple[int, Box, Box, Box]] = []
    occupied = np.zeros((height, width), dtype=bool)
    reserved = np.zeros((height, width), dtype=bool)

    for color, (band_start, band_size) in zip(pair_colors, bands):
        if orientation == "horizontal":
            sample = _sample_horizontal_pair(height, width, band_start, band_size)
        else:
            sample = _sample_vertical_pair(height, width, band_start, band_size)
        if sample is None:
            return None
        box_a, box_b, bridge = sample
        if occupied[box_a.r0 : box_a.r1 + 1, box_a.c0 : box_a.c1 + 1].any():
            return None
        if occupied[box_b.r0 : box_b.r1 + 1, box_b.c0 : box_b.c1 + 1].any():
            return None
        if reserved[bridge.r0 : bridge.r1 + 1, bridge.c0 : bridge.c1 + 1].any():
            return None

        occupied[box_a.r0 : box_a.r1 + 1, box_a.c0 : box_a.c1 + 1] = True
        occupied[box_b.r0 : box_b.r1 + 1, box_b.c0 : box_b.c1 + 1] = True
        reserved[bridge.r0 : bridge.r1 + 1, bridge.c0 : bridge.c1 + 1] = True
        pairs.append((color, box_a, box_b, bridge))

    singletons: list[tuple[int, Box]] = []
    for color in singleton_colors:
        placed = False
        for _ in range(SINGLETON_ATTEMPTS):
            rect_h = random.randint(*SINGLETON_SIDE_RANGE)
            rect_w = random.randint(*SINGLETON_SIDE_RANGE)
            if rect_h > height or rect_w > width:
                continue
            r0 = random.randint(0, height - rect_h)
            c0 = random.randint(0, width - rect_w)
            box = Box(r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _can_place(box, occupied, reserved, pad=1):
                continue
            occupied[box.r0 : box.r1 + 1, box.c0 : box.c1 + 1] = True
            singletons.append((color, box))
            placed = True
            break
        if not placed:
            return None

    inp, expected = _render_scene((height, width), pairs, singletons)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid grid")
