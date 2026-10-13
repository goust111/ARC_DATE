from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive
Coord = tuple[int, int]

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
FRAME_H_RANGE = (9, 11)
FRAME_W_RANGE = (9, 11)
RECT_H_RANGE = (3, 4)
RECT_W_RANGE = (3, 4)
PATTERNS = (
    "diag_main",
    "diag_anti",
    "left_pair",
    "right_pair",
    "top_pair",
    "bottom_pair",
)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if len(_components4(mask)) != 1:
        return False
    return int(mask.sum()) == h * w


def _is_hollow_frame(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(_components4(mask)) != 1:
        return False
    region = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(region, expected)


def _find_scene(grid: np.ndarray) -> tuple[int, int, BBox, int, BBox]:
    bg = _get_bg(grid)
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    solid: tuple[int, BBox] | None = None
    frame: tuple[int, BBox] | None = None
    for color in colors:
        mask = grid == color
        box = _bbox(mask)
        assert box is not None
        if _is_solid_rectangle(mask):
            assert solid is None, "expected one solid rectangle"
            solid = (color, box)
        elif _is_hollow_frame(mask):
            assert frame is None, "expected one hollow frame"
            frame = (color, box)
        else:
            raise AssertionError("foreground object must be a solid rectangle or hollow frame")

    assert solid is not None and frame is not None
    solid_color, solid_box = solid
    frame_color, frame_box = frame

    sr0, sc0, sr1, sc1 = solid_box
    fr0, fc0, fr1, fc1 = frame_box
    assert fr0 < sr0 <= sr1 < fr1 and fc0 < sc0 <= sc1 < fc1, "solid rectangle must lie inside frame"
    return bg, solid_color, solid_box, frame_color, frame_box


def _paint_active_rays(
    grid: np.ndarray,
    bg: int,
    solid_color: int,
    solid_box: BBox,
    frame_color: int,
    frame_box: BBox,
) -> tuple[np.ndarray, list[str]]:
    out = grid.copy()
    sr0, sc0, sr1, sc1 = solid_box
    fr0, fc0, fr1, fc1 = frame_box
    frame_corners = {
        (fr0, fc0),
        (fr0, fc1),
        (fr1, fc0),
        (fr1, fc1),
    }
    rays = [
        ("tl", (sr0, sc0), (-1, -1)),
        ("tr", (sr0, sc1), (-1, 1)),
        ("bl", (sr1, sc0), (1, -1)),
        ("br", (sr1, sc1), (1, 1)),
    ]

    active: list[str] = []
    for name, (r, c), (dr, dc) in rays:
        rr, cc = r + dr, c + dc
        while _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == bg:
            rr += dr
            cc += dc
        if not _in_bounds(grid.shape, rr, cc):
            continue
        if grid[rr, cc] != frame_color or (rr, cc) not in frame_corners:
            continue

        active.append(name)
        rr += dr
        cc += dc
        while _in_bounds(grid.shape, rr, cc):
            if out[rr, cc] == bg:
                out[rr, cc] = solid_color
            rr += dr
            cc += dc

    return out, active


def _exterior_regions(grid: np.ndarray, bg: int, frame_box: BBox) -> list[list[Coord]]:
    mask = grid == bg
    fr0, fc0, fr1, fc1 = frame_box
    if fr1 - fr0 > 1 and fc1 - fc0 > 1:
        mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = False
    return _components4(mask)


def _paint_cells(grid: np.ndarray, cells: list[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonals through aligned frame corners, then recolor the smallest and largest exterior regions.

    Input structure:
    - A black background contains one solid rectangle and one one-pixel-thick hollow rectangular frame in different colors.
    - The solid rectangle lies strictly inside the frame.
    - Some rectangle corners are diagonally aligned with frame corners, while others may hit a frame edge first.

    Rule:
    - From each rectangle corner, trace the outward diagonal until the first non-background pixel.
    - If that first non-background pixel is a frame corner, continue past it on the same diagonal and paint every remaining background cell to the grid edge with the rectangle color.
    - Treat the painted rays, the frame, and the rectangle as barriers; among background regions outside the frame only, recolor every minimum-area region green and every maximum-area region yellow.
    - Keep the frame, rectangle, and any painted rays in their original colors.

    Key insight:
    - Corner-to-corner diagonal alignment decides which rays exist, and those rays change how the frame exterior is partitioned before the area comparison is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg, solid_color, solid_box, frame_color, frame_box = _find_scene(g)
    out, _active = _paint_active_rays(g, bg, solid_color, solid_box, frame_color, frame_box)

    regions = _exterior_regions(out, bg, frame_box)
    assert regions, "expected at least one exterior background region"
    areas = [len(region) for region in regions]
    min_area = min(areas)
    max_area = max(areas)

    for region, area in zip(regions, areas, strict=True):
        if area == min_area:
            _paint_cells(out, region, GREEN)
    for region, area in zip(regions, areas, strict=True):
        if area == max_area:
            _paint_cells(out, region, YELLOW)

    return out


def _pattern_offset_candidates(pattern: str, dy: int, dx: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    if pattern == "diag_main":
        if dy != dx:
            return candidates
        for top_gap in range(1, dy):
            if 2 * top_gap == dx:
                continue
            candidates.append((top_gap, top_gap))
        return candidates

    if pattern == "diag_anti":
        if dy != dx:
            return candidates
        for top_gap in range(1, dy):
            if 2 * top_gap == dx:
                continue
            candidates.append((top_gap, dx - top_gap))
        return candidates

    if pattern == "left_pair":
        if dy % 2 != 0 or dx == dy:
            return candidates
        top_gap = dy // 2
        left_gap = top_gap
        if 1 <= top_gap < dy and 1 <= left_gap < dx:
            candidates.append((top_gap, left_gap))
        return candidates

    if pattern == "right_pair":
        if dy % 2 != 0 or dx == dy:
            return candidates
        top_gap = dy // 2
        left_gap = dx - top_gap
        if 1 <= top_gap < dy and 1 <= left_gap < dx:
            candidates.append((top_gap, left_gap))
        return candidates

    if pattern == "top_pair":
        if dx % 2 != 0 or dx == dy:
            return candidates
        top_gap = dx // 2
        left_gap = top_gap
        if 1 <= top_gap < dy and 1 <= left_gap < dx:
            candidates.append((top_gap, left_gap))
        return candidates

    if pattern == "bottom_pair":
        if dx % 2 != 0 or dx == dy:
            return candidates
        left_gap = dx // 2
        top_gap = dy - left_gap
        if 1 <= top_gap < dy and 1 <= left_gap < dx:
            candidates.append((top_gap, left_gap))
        return candidates

    raise ValueError(f"unknown pattern: {pattern}")


def _central_start(limit: int, span: int) -> int:
    base = (limit - span) // 2
    choices = []
    for delta in (-1, 0, 1):
        pos = base + delta
        if 1 <= pos <= limit - span - 1:
            choices.append(pos)
    if not choices:
        choices = [pos for pos in range(1, limit - span)]
    return int(random.choice(choices))


def _build_input(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    rect_h: int,
    rect_w: int,
    pattern: str,
) -> np.ndarray | None:
    dy = frame_h - rect_h
    dx = frame_w - rect_w
    offset_choices = _pattern_offset_candidates(pattern, dy, dx)
    if not offset_choices:
        return None

    fr0 = _central_start(height, frame_h)
    fc0 = _central_start(width, frame_w)
    top_gap, left_gap = random.choice(offset_choices)

    sr0 = fr0 + top_gap
    sc0 = fc0 + left_gap
    sr1 = sr0 + rect_h - 1
    sc1 = sc0 + rect_w - 1
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((height, width), BLACK, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = RED
    grid[fr1, fc0 : fc1 + 1] = RED
    grid[fr0 : fr1 + 1, fc0] = RED
    grid[fr0 : fr1 + 1, fc1] = RED
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = BLUE
    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        bg, solid_color, solid_box, frame_color, frame_box = _find_scene(grid)
    except AssertionError:
        return False

    if bg != BLACK or solid_color != BLUE or frame_color != RED:
        return False

    sr0, sc0, sr1, sc1 = solid_box
    fr0, fc0, fr1, fc1 = frame_box
    gaps = (sr0 - fr0, fr1 - sr1, sc0 - fc0, fc1 - sc1)
    if len(set(gaps)) == 1:
        return False

    ray_grid, active = _paint_active_rays(grid, bg, solid_color, solid_box, frame_color, frame_box)
    if len(active) < 2:
        return False

    regions = _exterior_regions(ray_grid, bg, frame_box)
    if len(regions) < 2:
        return False

    areas = [len(region) for region in regions]
    if min(areas) == max(areas):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    if np.any(grid == GREEN) or np.any(grid == YELLOW):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)
    min_h = max(GRID_H_RANGE[0], frame_h + 4)
    min_w = max(GRID_W_RANGE[0], frame_w + 4)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_H_RANGE[1])
    width = random.randint(min_w, GRID_W_RANGE[1])
    pattern = random.choice(PATTERNS)
    grid = _build_input(height, width, frame_h, frame_w, rect_h, rect_w, pattern)
    if grid is None:
        return None
    return grid if _validate_generated_case(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
