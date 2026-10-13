from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ALL_COLORS = list(range(1, 10))
LAYOUTS = ("single", "vertical", "horizontal", "cross")
EDGE_SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _detect_templates(grid: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    template_mask = np.zeros((h, w), dtype=bool)
    templates: dict[int, np.ndarray] = {}

    def consider(r0: int, c0: int) -> tuple[int, np.ndarray, list[tuple[int, int]]] | None:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        if patch.shape != (3, 3):
            return None

        checks = [
            (
                [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
                patch[1:, 1:],
                patch[0, 0],
            ),
            (
                [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
                patch[1:, :2],
                patch[0, 2],
            ),
            (
                [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
                patch[:2, 1:],
                patch[2, 0],
            ),
            (
                [(0, 2), (1, 2), (2, 2), (2, 0), (2, 1)],
                patch[:2, :2],
                patch[2, 2],
            ),
        ]
        for border_pts, core, color0 in checks:
            color = int(color0)
            if color == BG:
                continue
            if not all(int(patch[rr, cc]) == color for rr, cc in border_pts):
                continue
            core_mask = core != BG
            if core_mask.sum() == 0 or core_mask.sum() == 4:
                continue
            if np.any(core[core_mask] != color):
                continue
            cells = [(r0 + rr, c0 + cc) for rr in range(3) for cc in range(3)]
            return color, core_mask.copy(), cells
        return None

    anchors: list[tuple[int, int]] = []
    for c0 in range(w - 2):
        anchors.append((0, c0))
        if h > 3:
            anchors.append((h - 3, c0))
    for r0 in range(1, h - 3):
        anchors.append((r0, 0))
        if w > 3:
            anchors.append((r0, w - 3))

    seen_anchor: set[tuple[int, int]] = set()
    for r0, c0 in anchors:
        if (r0, c0) in seen_anchor:
            continue
        seen_anchor.add((r0, c0))
        res = consider(r0, c0)
        if res is None:
            continue
        color, core_mask, cells = res
        if color in templates:
            continue
        if any(template_mask[r, c] for r, c in cells):
            continue
        templates[color] = core_mask
        for r, c in cells:
            template_mask[r, c] = True

    return templates, template_mask


def _marker_mask(grid: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    markers = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if template_mask[r, c] or int(grid[r, c]) == BG:
                continue
            if any(int(grid[nr, nc]) != BG and not template_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                continue
            markers[r, c] = True
    return markers


def _fill_region(
    out: np.ndarray,
    region: list[tuple[int, int]],
    marker_color: int,
    kernel: np.ndarray,
) -> None:
    r0 = min(r for r, _ in region)
    c0 = min(c for _, c in region)
    for r, c in region:
        out[r, c] = marker_color if bool(kernel[(r - r0) % 2, (c - c0) % 2]) else BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed path region by tiling the edge template selected by its interior marker color.

    Input structure:
    - Background is black.
    - One-pixel-wide colored path segments connect into closed boundaries; a boundary may contain multiple colors.
    - Each enclosed region contains exactly one isolated colored marker pixel.
    - One or more 3x3 template blocks sit on the grid edge; each template has a colored L-shaped border and a 2x2 black/non-black kernel keyed by that border color.

    Rule:
    - Detect all edge templates and map each border color to its 2x2 binary kernel.
    - Treat isolated interior singletons as region markers and all other interior non-black pixels as path walls.
    - Find every 4-connected non-wall region that does not touch the outer border.
    - For each such region, read its marker color, select the template with the same border color, and tile that 2x2 kernel from the region's top-left cell, painting kernel 1-cells with the marker color and kernel 0-cells black.
    - Keep path pixels, templates, and exterior background unchanged.

    Key insight:
    - The solver must combine three cues: multi-color walls define the closed geometry, a single interior marker selects the matching edge template by color, and the template contributes only its 2x2 kernel, which must be tiled rather than copied once.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    templates, template_mask = _detect_templates(g)
    markers = _marker_mask(g, template_mask)

    walls = (g != BG) & (~template_mask) & (~markers)
    passable = (~walls) & (~template_mask)
    out = g.copy()

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        marker_pts = [(r, c) for r, c in comp if markers[r, c]]
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        marker_color = int(g[mr, mc])
        kernel = templates.get(marker_color)
        if kernel is None:
            continue
        _fill_region(out, comp, marker_color, kernel)

    return out


def _random_kernel() -> np.ndarray:
    while True:
        mask = np.array([[random.choice([False, True]) for _ in range(2)] for _ in range(2)], dtype=bool)
        if 0 < int(mask.sum()) < 4:
            return mask


def _paint_segment(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _room_rects(layout: str, inner_r0: int, inner_c0: int, room_heights: list[int], room_widths: list[int]) -> list[tuple[int, int, int, int]]:
    if layout == "single":
        return [(inner_r0, inner_c0, room_heights[0], room_widths[0])]
    if layout == "vertical":
        return [
            (inner_r0, inner_c0, room_heights[0], room_widths[0]),
            (inner_r0, inner_c0 + room_widths[0] + 1, room_heights[0], room_widths[1]),
        ]
    if layout == "horizontal":
        return [
            (inner_r0, inner_c0, room_heights[0], room_widths[0]),
            (inner_r0 + room_heights[0] + 1, inner_c0, room_heights[1], room_widths[0]),
        ]
    return [
        (inner_r0, inner_c0, room_heights[0], room_widths[0]),
        (inner_r0, inner_c0 + room_widths[0] + 1, room_heights[0], room_widths[1]),
        (inner_r0 + room_heights[0] + 1, inner_c0, room_heights[1], room_widths[0]),
        (inner_r0 + room_heights[0] + 1, inner_c0 + room_widths[0] + 1, room_heights[1], room_widths[1]),
    ]


def _draw_walls(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    layout: str,
    room_heights: list[int],
    room_widths: list[int],
    path_colors: list[int],
) -> list[tuple[int, int, int, int]]:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    segments = [
        [(top, c) for c in range(left, right + 1)],
        [(bottom, c) for c in range(left, right + 1)],
        [(r, left) for r in range(top, bottom + 1)],
        [(r, right) for r in range(top, bottom + 1)],
    ]
    inner_r0 = top + 1
    inner_c0 = left + 1

    if layout in ("vertical", "cross"):
        div_c = inner_c0 + room_widths[0]
        segments.append([(r, div_c) for r in range(inner_r0, bottom)])
    if layout in ("horizontal", "cross"):
        div_r = inner_r0 + room_heights[0]
        segments.append([(div_r, c) for c in range(inner_c0, right)])

    for cells in segments:
        color = random.choice(path_colors)
        _paint_segment(grid, cells, color)

    if layout == "cross":
        cross_r = inner_r0 + room_heights[0]
        cross_c = inner_c0 + room_widths[0]
        grid[cross_r, cross_c] = random.choice(path_colors)

    return _room_rects(layout, inner_r0, inner_c0, room_heights, room_widths)


def _place_templates(
    grid: np.ndarray,
    side: str,
    colors: list[int],
    kernels: dict[int, np.ndarray],
) -> bool:
    h, w = grid.shape
    count = len(colors)
    if side in ("top", "bottom"):
        if 4 * count - 1 > w:
            return False
        starts = list(range(0, w - 2, 4))
        random.shuffle(starts)
        starts = sorted(starts[:count])
        if len(starts) != count or starts[-1] + 2 >= w:
            return False
        for color, c0 in zip(colors, starts, strict=False):
            r0 = 0 if side == "top" else h - 3
            orientation = random.choice(("left", "right"))
            grid[r0 : r0 + 3, c0 : c0 + 3] = BG
            if side == "top":
                grid[r0, c0 : c0 + 3] = color
                if orientation == "left":
                    grid[r0 : r0 + 3, c0] = color
                    core_r, core_c = r0 + 1, c0 + 1
                else:
                    grid[r0 : r0 + 3, c0 + 2] = color
                    core_r, core_c = r0 + 1, c0
            else:
                grid[r0 + 2, c0 : c0 + 3] = color
                if orientation == "left":
                    grid[r0 : r0 + 3, c0] = color
                    core_r, core_c = r0, c0 + 1
                else:
                    grid[r0 : r0 + 3, c0 + 2] = color
                    core_r, core_c = r0, c0
            kernel = kernels[color]
            for dr in range(2):
                for dc in range(2):
                    grid[core_r + dr, core_c + dc] = color if bool(kernel[dr, dc]) else BG
        return True

    if 4 * count - 1 > h:
        return False
    starts = list(range(0, h - 2, 4))
    random.shuffle(starts)
    starts = sorted(starts[:count])
    if len(starts) != count or starts[-1] + 2 >= h:
        return False
    for color, r0 in zip(colors, starts, strict=False):
        c0 = 0 if side == "left" else w - 3
        orientation = random.choice(("top", "bottom"))
        grid[r0 : r0 + 3, c0 : c0 + 3] = BG
        if side == "left":
            grid[r0 : r0 + 3, c0] = color
            if orientation == "top":
                grid[r0, c0 : c0 + 3] = color
                core_r, core_c = r0 + 1, c0 + 1
            else:
                grid[r0 + 2, c0 : c0 + 3] = color
                core_r, core_c = r0, c0 + 1
        else:
            grid[r0 : r0 + 3, c0 + 2] = color
            if orientation == "top":
                grid[r0, c0 : c0 + 3] = color
                core_r, core_c = r0 + 1, c0
            else:
                grid[r0 + 2, c0 : c0 + 3] = color
                core_r, core_c = r0, c0
        kernel = kernels[color]
        for dr in range(2):
            for dc in range(2):
                grid[core_r + dr, core_c + dc] = color if bool(kernel[dr, dc]) else BG
    return True


def _layout_params(layout: str) -> tuple[list[int], list[int], int, int]:
    if layout == "single":
        room_h = [random.randint(4, 9)]
        room_w = [random.randint(4, 10)]
        return room_h, room_w, room_h[0] + 2, room_w[0] + 2
    if layout == "vertical":
        room_h = [random.randint(4, 9)]
        room_w = [random.randint(4, 7), random.randint(4, 7)]
        return room_h, room_w, room_h[0] + 2, room_w[0] + room_w[1] + 3
    if layout == "horizontal":
        room_h = [random.randint(4, 6), random.randint(4, 6)]
        room_w = [random.randint(4, 10)]
        return room_h, room_w, room_h[0] + room_h[1] + 3, room_w[0] + 2
    room_h = [random.randint(4, 6), random.randint(4, 6)]
    room_w = [random.randint(4, 7), random.randint(4, 7)]
    return room_h, room_w, room_h[0] + room_h[1] + 3, room_w[0] + room_w[1] + 3


def _available_box(h: int, w: int, side: str, outer_h: int, outer_w: int) -> tuple[range, range] | None:
    if side == "top":
        r_range = range(4, h - outer_h - 1)
        c_range = range(1, w - outer_w - 1)
    elif side == "bottom":
        r_range = range(1, h - outer_h - 4)
        c_range = range(1, w - outer_w - 1)
    elif side == "left":
        r_range = range(1, h - outer_h - 1)
        c_range = range(4, w - outer_w - 1)
    else:
        r_range = range(1, h - outer_h - 1)
        c_range = range(1, w - outer_w - 4)
    if len(r_range) == 0 or len(c_range) == 0:
        return None
    return r_range, c_range


def _attempt_generate() -> np.ndarray | None:
    layout = random.choice(LAYOUTS)
    side = random.choice(EDGE_SIDES)
    room_heights, room_widths, outer_h, outer_w = _layout_params(layout)

    h = random.randint(max(outer_h + 6, 12), min(outer_h + 10, 28))
    w = random.randint(max(outer_w + 6, 14), min(outer_w + 12, 30))

    box = _available_box(h, w, side, outer_h, outer_w)
    if box is None:
        return None
    r_range, c_range = box

    grid = np.zeros((h, w), dtype=int)
    top = random.choice(list(r_range))
    left = random.choice(list(c_range))

    region_count = {"single": 1, "vertical": 2, "horizontal": 2, "cross": 4}[layout]
    color_pool = ALL_COLORS[:]
    random.shuffle(color_pool)
    marker_colors = color_pool[:region_count]
    remaining = color_pool[region_count:]
    path_color_count = random.randint(2, min(4, len(remaining) - 1))
    path_colors = remaining[:path_color_count]
    extra_template_colors = remaining[path_color_count : path_color_count + random.randint(0, 1)]

    kernels: dict[int, np.ndarray] = {}
    for color in marker_colors + extra_template_colors:
        kernels[color] = _random_kernel()

    all_template_colors = marker_colors + extra_template_colors
    if not _place_templates(grid, side, all_template_colors, kernels):
        return None

    rooms = _draw_walls(grid, top, left, outer_h, outer_w, layout, room_heights, room_widths, path_colors)

    for (r0, c0, rh, rw), marker_color in zip(rooms, marker_colors, strict=False):
        rr = random.randint(r0, r0 + rh - 1)
        cc = random.randint(c0, c0 + rw - 1)
        grid[rr, cc] = marker_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == BG) == np.count_nonzero(grid == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
