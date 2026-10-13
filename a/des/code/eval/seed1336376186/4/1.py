from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
L_MASKS = {
    "right": np.array([[1, 0], [1, 1]], dtype=bool),
    "up": np.array([[0, 1], [1, 1]], dtype=bool),
    "left": np.array([[1, 1], [0, 1]], dtype=bool),
    "down": np.array([[1, 1], [1, 0]], dtype=bool),
}
MASK_TO_DIR = {tuple(int(v) for v in mask.ravel()): direction for direction, mask in L_MASKS.items()}

GEN_RETRIES = 400
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (26, 38)
REGION_COUNT_RANGE = (4, 6)
LEGEND_HEIGHT = 2
LEGEND_ENTRY_WIDTH = 3
LEGEND_GAP = 1
LEGEND_SEPARATOR_ROWS = 1
REGION_GAP = 1
PLACEMENT_RETRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return counts.most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_fence(gray_mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    patch = gray_mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(patch, expected)


def _legend_mapping(grid: np.ndarray, bg: int, grey: int) -> dict[int, str]:
    h, w = grid.shape
    if h < 2:
        return {}

    top = h - LEGEND_HEIGHT
    mapping: dict[int, str] = {}
    c = 0
    while c <= w - LEGEND_ENTRY_WIDTH:
        color = int(grid[top, c])
        if color == bg or color == grey or int(grid[top + 1, c]) != color:
            c += 1
            continue

        arrow = grid[top : top + 2, c + 1 : c + 3]
        if arrow.shape != (2, 2):
            c += 1
            continue
        gray_mask = arrow == grey
        if int(gray_mask.sum()) != 3 or not np.all((arrow == bg) | gray_mask):
            c += 1
            continue

        direction = MASK_TO_DIR.get(tuple(int(v) for v in gray_mask.ravel()))
        if direction is None:
            c += 1
            continue

        assert color not in mapping or mapping[color] == direction
        mapping[color] = direction
        c += LEGEND_ENTRY_WIDTH + LEGEND_GAP

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color-to-direction legend on the bottom edge, then grow a same-color
    ray inside each gray rectangular fence. The ray length is the fence's interior
    span along that direction and it stops at the wall.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    counts = Counter(int(v) for v in g.ravel())
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected non-background structure"
    grey = max(non_bg, key=lambda color: counts[color])

    legend = _legend_mapping(g, bg, grey)
    assert legend, "expected at least one legend entry on the bottom edge"

    gray_mask = g == grey
    fences = [cells for cells in _components(gray_mask) if _is_fence(gray_mask, cells)]
    assert fences, "expected at least one closed rectangular fence"

    out = np.array(g, copy=True)
    for fence in fences:
        r0, c0, r1, c1 = _bbox(fence)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_local = np.argwhere((interior != bg) & (interior != grey))
        assert len(seed_local) == 1, "each fence interior must contain exactly one seed"
        sr = int(seed_local[0, 0] + r0 + 1)
        sc = int(seed_local[0, 1] + c0 + 1)
        color = int(g[sr, sc])
        assert color in legend, "every seed color must be present in the legend"

        direction = legend[color]
        dr, dc = DIRS[direction]
        span = (c1 - c0 - 1) if direction in ("left", "right") else (r1 - r0 - 1)

        for step in range(1, span):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if int(g[rr, cc]) == grey:
                break
            if int(out[rr, cc]) not in (bg, color):
                break
            out[rr, cc] = color

    return out


def _legend_width(entries: int) -> int:
    return entries * LEGEND_ENTRY_WIDTH + (entries - 1) * LEGEND_GAP


def _draw_legend(grid: np.ndarray, color_for_direction: dict[str, int], order: list[str]) -> None:
    h, w = grid.shape
    legend_w = _legend_width(len(order))
    start_c = random.randint(1, w - legend_w - 1)
    top = h - LEGEND_HEIGHT
    c = start_c

    for direction in order:
        color = color_for_direction[direction]
        grid[top : top + 2, c] = color
        mask = L_MASKS[direction]
        arrow = grid[top : top + 2, c + 1 : c + 3]
        arrow[mask] = GREY
        c += LEGEND_ENTRY_WIDTH + LEGEND_GAP


def _touches_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = occupied.shape
    er0 = max(0, r0 - pad)
    ec0 = max(0, c0 - pad)
    er1 = min(h - 1, r1 + pad)
    ec1 = min(w - 1, c1 + pad)
    return bool(occupied[er0 : er1 + 1, ec0 : ec1 + 1].any())


def _place_fence(
    grid: np.ndarray,
    occupied: np.ndarray,
    direction: str,
    color: int,
    content_bottom: int,
) -> bool:
    h, w = grid.shape

    if direction in ("left", "right"):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(4, 8)
    else:
        inner_h = random.randint(4, 8)
        inner_w = random.randint(3, 6)

    rect_h = inner_h + 2
    rect_w = inner_w + 2
    max_r0 = content_bottom - rect_h + 1
    max_c0 = w - rect_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, max_c0)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

        if _touches_occupied(occupied, r0, c0, r1, c1, REGION_GAP):
            continue

        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY

        if direction == "right":
            sr = random.randint(r0 + 1, r1 - 1)
            sc = c0 + 1
        elif direction == "left":
            sr = random.randint(r0 + 1, r1 - 1)
            sc = c1 - 1
        elif direction == "down":
            sr = r0 + 1
            sc = random.randint(c0 + 1, c1 - 1)
        else:
            sr = r1 - 1
            sc = random.randint(c0 + 1, c1 - 1)

        grid[sr, sc] = color
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    min_w = max(GRID_W_RANGE[0], _legend_width(4) + 3)
    w = random.randint(min_w, GRID_W_RANGE[1])

    bg_choices = [color for color in range(10) if color != GREY]
    bg = random.choice(bg_choices)
    color_pool = [color for color in range(10) if color not in (bg, GREY)]
    seed_colors = random.sample(color_pool, 4)
    directions = ["up", "down", "left", "right"]
    random.shuffle(directions)
    color_for_direction = {direction: color for direction, color in zip(directions, seed_colors)}

    grid = np.full((h, w), bg, dtype=int)
    content_bottom = h - LEGEND_HEIGHT - LEGEND_SEPARATOR_ROWS - 1
    if content_bottom < 8:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    region_count = random.randint(*REGION_COUNT_RANGE)
    region_dirs = ["up", "down", "left", "right"]
    while len(region_dirs) < region_count:
        region_dirs.append(random.choice(region_dirs[:4]))
    random.shuffle(region_dirs)

    for direction in region_dirs:
        if not _place_fence(grid, occupied, direction, color_for_direction[direction], content_bottom):
            return None

    legend_order = ["up", "down", "left", "right"]
    random.shuffle(legend_order)
    _draw_legend(grid, color_for_direction, legend_order)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
