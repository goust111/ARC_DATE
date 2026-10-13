from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
INPUT_COLORS = (1, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


TEMPLATE_BRANCHES = [
    (
        (1, 3),
        [
            ("##.", ".##", "..#"),
        ],
    ),
    (
        (1, 2),
        [
            ("##.", "###", "..#"),
            ("##..", ".###", "..#."),
        ],
    ),
    (
        (1, 5),
        [
            ("###.", "..##", "...#"),
        ],
    ),
    (
        (2, 4),
        [
            ("###.", "..#.", "..##"),
            ("##..", ".###", "...#"),
        ],
    ),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _parse_template(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    coords = np.argwhere(out)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _component_sizes(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        area = 0
        while q:
            cr, cc = q.popleft()
            area += 1
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        sizes.append(area)
    return sizes


def _single_component(mask: np.ndarray) -> bool:
    sizes = _component_sizes(mask)
    return len(sizes) == 1


def _has_internal_hole(mask: np.ndarray) -> bool:
    background = ~mask
    reachable = np.zeros(mask.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = mask.shape

    for r in range(h):
        for c in (0, w - 1):
            if background[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if background[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and background[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(background & ~reachable))


def _outline_mask(foreground: np.ndarray) -> np.ndarray:
    outline = np.zeros(foreground.shape, dtype=bool)
    for r, c in np.argwhere(foreground):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = rr + dr
                nc = cc + dc
                if _in_bounds(foreground.shape, nr, nc) and not foreground[nr, nc]:
                    outline[nr, nc] = True
    return outline


def _closed_regions(foreground: np.ndarray, use_outline: bool) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    outline = _outline_mask(foreground) if use_outline else np.zeros_like(foreground, dtype=bool)
    passable = (~foreground) & (~outline)
    reachable = np.zeros(foreground.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = foreground.shape

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(foreground.shape, nr, nc) and passable[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    holes = passable & ~reachable
    seen = np.zeros(foreground.shape, dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(holes):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        region: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            region.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(foreground.shape, nr, nc) and holes[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        regions.append(region)
    return outline, regions


def _emergent_area_signature(grid: np.ndarray) -> tuple[int, ...]:
    fg_mask = np.tile(grid != BG, (3, 3))
    _outline, regions = _closed_regions(fg_mask, use_outline=True)
    return tuple(sorted(len(region) for region in regions))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the hole-free input shape, outline the repeated copies, and fill emergent enclosed gaps by area tier.

    Input structure:
    - The input is a small black-background grid with exactly one non-black foreground color.
    - The foreground pixels form one 4-connected shape with no internal hole in the input itself.
    - Red, green, and yellow are reserved output colors and do not appear in the source shape.

    Rule:
    - Repeat the whole input grid three times horizontally and vertically.
    - On the tiled canvas, recolor every background pixel that 8-neighbors any foreground pixel to green.
    - Treat foreground and green outline pixels as blockers, flood-fill the remaining background from the canvas border with 4-connectivity, and identify the unreached closed regions.
    - Fill every closed region of minimum area yellow, fill every larger closed region red, and leave all border-reachable background black.

    Key insight:
    - The source shape has no hole alone; the holes only appear after 3x3 tiling plus the added outer outline, so the solver must compose tiling, outlining, and area-based region filling in that order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert 3 * g.shape[0] <= 30 and 3 * g.shape[1] <= 30, "tiled output must stay within 30x30"

    fg_colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg_color = fg_colors[0]
    assert fg_color not in (RED, GREEN, YELLOW), "input foreground color must not collide with output colors"

    out = np.tile(g, (3, 3))
    foreground = out == fg_color
    outline, regions = _closed_regions(foreground, use_outline=True)

    out[outline] = GREEN
    if not regions:
        return out

    min_area = min(len(region) for region in regions)
    for region in regions:
        fill_color = YELLOW if len(region) == min_area else RED
        for r, c in region:
            out[r, c] = fill_color
    return out


def _sample_branch() -> tuple[tuple[int, int], np.ndarray]:
    target_areas, templates = random.choice(TEMPLATE_BRANCHES)
    template_rows = random.choice(templates)
    return target_areas, _random_transform(_parse_template(template_rows))


def _place_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    r0 = random.randint(0, height - mask.shape[0])
    c0 = random.randint(0, width - mask.shape[1])
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = mask.astype(int)
    return grid


def _valid_generated_input(grid: np.ndarray, target_areas: tuple[int, int]) -> bool:
    if grid.ndim != 2 or not (5 <= grid.shape[0] <= 7 and 5 <= grid.shape[1] <= 7):
        return False

    fg_mask = grid != BG
    fg_count = int(np.sum(fg_mask))
    if not (3 <= fg_count <= 6):
        return False
    if fg_count > int(0.6 * grid.size):
        return False
    if not _single_component(fg_mask):
        return False
    if _has_internal_hole(fg_mask):
        return False

    without_outline = _closed_regions(np.tile(fg_mask, (3, 3)), use_outline=False)[1]
    if without_outline:
        return False

    area_signature = _emergent_area_signature(grid)
    unique_areas = tuple(sorted(set(area_signature)))
    if unique_areas != target_areas:
        return False
    counts_by_area = {area: area_signature.count(area) for area in unique_areas}
    if any(counts_by_area[area] < 2 for area in unique_areas):
        return False

    out = main(grid)
    if out.shape != (grid.shape[0] * 3, grid.shape[1] * 3):
        return False
    fg_color = int(grid[fg_mask][0])
    needed = {BG, GREEN, YELLOW, RED, fg_color}
    if not needed.issubset(set(map(int, np.unique(out)))):
        return False
    if int(np.sum(out == YELLOW)) != area_signature.count(target_areas[0]) * target_areas[0]:
        return False
    if int(np.sum(out == RED)) != area_signature.count(target_areas[1]) * target_areas[1]:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target_areas, mask = _sample_branch()
    height = random.randint(max(5, mask.shape[0]), 7)
    width = random.randint(max(5, mask.shape[1]), 7)
    placed = _place_mask(mask, height, width)
    color = random.choice(INPUT_COLORS)
    grid = np.where(placed, color, BG).astype(int)
    if _valid_generated_input(grid, target_areas):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emergent-hole tiling input")
