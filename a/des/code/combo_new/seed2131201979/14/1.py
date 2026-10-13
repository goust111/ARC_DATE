from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_SIZES = (5, 7, 9)
SIDES = ("up", "down", "left", "right")
MAX_COLORS_PER_FRAME = 3
MAX_COUNT_PER_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_frame(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
        return None

    border = _border_coords(r0, c0, r1, c1)
    coord_set = {tuple(map(int, pt)) for pt in coords}
    if not coord_set.issubset(border):
        return None
    missing = sorted(border - coord_set)
    if len(missing) != 1 or len(coord_set) != len(border) - 1:
        return None

    gap = missing[0]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if gap == (r0, mid_c):
        step = (-1, 0)
    elif gap == (r1, mid_c):
        step = (1, 0)
    elif gap == (mid_r, c0):
        step = (0, -1)
    elif gap == (mid_r, c1):
        step = (0, 1)
    else:
        return None

    return {"bbox": (r0, c0, r1, c1), "gap": gap, "step": step}


def _ordered_color_counts(interior: np.ndarray) -> tuple[list[int], dict[int, int]]:
    order: list[int] = []
    counts: dict[int, int] = {}
    seen: set[int] = set()
    for value in interior.ravel(order="C"):
        color = int(value)
        if color in (BLACK, GREY):
            continue
        counts[color] = counts.get(color, 0) + 1
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded outward branches from gapped gray frames using the seed counts inside each frame.

    Input structure:
    - The background is black.
    - The grid contains one or more gray hollow rectangular frames with border thickness 1.
    - Each frame border has exactly one missing pixel at the midpoint of one side.
    - Inside each frame are scattered colored seed pixels that are neither black nor gray.

    Rule:
    - For each gray frame, locate the missing border pixel and treat its side as the branch direction.
    - Scan the frame interior in row-major order to record the first appearance order of each seed color.
    - Count how many interior pixels belong to each color.
    - Starting at the gap cell, draw a one-cell-wide ray outward; each color occupies a segment whose length equals that color's count, and later colors continue directly after earlier ones.
    - Keep the original frame and all interior seed pixels unchanged.

    Key insight:
    - The missing border pixel is an exit marker, while the interior color multiplicities encode the lengths of consecutive branch segments.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = map(int, g.shape)

    frames: list[dict[str, object]] = []
    for comp in _components(g == GREY):
        frame = _parse_frame(comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic when multiple frames exist

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        dr, dc = frame["step"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        order, counts = _ordered_color_counts(interior)
        if not order:
            continue

        cur_r = int(gap_r)
        cur_c = int(gap_c)
        for color in order:
            for _ in range(int(counts[color])):
                assert 0 <= cur_r < height and 0 <= cur_c < width, "branch exits the grid"
                out[cur_r, cur_c] = int(color)
                cur_r += int(dr)
                cur_c += int(dc)

    return out


def _sample_counts(interior_area: int, n_colors: int) -> list[int]:
    counts: list[int] = []
    remaining = int(interior_area)
    for idx in range(n_colors):
        min_left = int(n_colors - idx - 1)
        max_count = min(MAX_COUNT_PER_COLOR, remaining - min_left)
        count = random.randint(1, max_count)
        counts.append(int(count))
        remaining -= int(count)
    return counts


def _gap_and_step(r0: int, c0: int, h: int, w: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    if side == "up":
        return (r0, mid_c), (-1, 0)
    if side == "down":
        return (r1, mid_c), (1, 0)
    if side == "left":
        return (mid_r, c0), (0, -1)
    return (mid_r, c1), (0, 1)


def _placement_bounds(
    height: int,
    width: int,
    h: int,
    w: int,
    total_len: int,
    side: str,
) -> tuple[int, int, int, int] | None:
    if side == "up":
        r0_min = total_len - 1
        r0_max = height - h
        c0_min = 0
        c0_max = width - w
    elif side == "down":
        r0_min = 0
        r0_max = height - h - total_len + 1
        c0_min = 0
        c0_max = width - w
    elif side == "left":
        r0_min = 0
        r0_max = height - h
        c0_min = total_len - 1
        c0_max = width - w
    else:
        r0_min = 0
        r0_max = height - h
        c0_min = 0
        c0_max = width - w - total_len + 1

    if r0_min > r0_max or c0_min > c0_max:
        return None
    return int(r0_min), int(r0_max), int(c0_min), int(c0_max)


def _expanded_cells(cells: set[tuple[int, int]], height: int, width: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(height, r + pad + 1)):
            for cc in range(max(0, c - pad), min(width, c + pad + 1)):
                expanded.add((rr, cc))
    return expanded


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, gap: tuple[int, int]) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for c in range(c0, c1 + 1):
        if (r0, c) != gap:
            grid[r0, c] = GREY
        if (r1, c) != gap:
            grid[r1, c] = GREY
    for r in range(r0, r1 + 1):
        if (r, c0) != gap:
            grid[r, c0] = GREY
        if (r, c1) != gap:
            grid[r, c1] = GREY


def _place_frame(grid: np.ndarray, blocked: np.ndarray, side: str) -> bool:
    height, width = map(int, grid.shape)
    for _ in range(PLACEMENT_ATTEMPTS):
        h = int(random.choice(FRAME_SIZES))
        w = int(random.choice(FRAME_SIZES))
        interior_area = int((h - 2) * (w - 2))
        n_colors = random.randint(1, min(MAX_COLORS_PER_FRAME, interior_area))
        counts = _sample_counts(interior_area, n_colors)
        total_len = int(sum(counts))

        bounds = _placement_bounds(height, width, h, w, total_len, side)
        if bounds is None:
            continue
        r0_min, r0_max, c0_min, c0_max = bounds
        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)

        gap, step = _gap_and_step(r0, c0, h, w, side)
        ray = {(gap[0] + i * step[0], gap[1] + i * step[1]) for i in range(total_len)}
        if any(not (0 <= r < height and 0 <= c < width) for r, c in ray):
            continue

        frame_area = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
        reserve = _expanded_cells(frame_area | ray, height, width, pad=1)
        if any(bool(blocked[r, c]) for r, c in reserve):
            continue

        _paint_frame(grid, r0, c0, h, w, gap)

        colors = random.sample(PALETTE, n_colors)
        interior_cells = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
        chosen = random.sample(interior_cells, total_len)
        idx = 0
        for color, count in zip(colors, counts):
            for r, c in chosen[idx : idx + count]:
                grid[r, c] = int(color)
            idx += int(count)

        for r, c in reserve:
            blocked[r, c] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    grid = np.full((height, width), BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    area = height * width
    if area < 360:
        max_frames = 2
    elif area < 560:
        max_frames = 3
    else:
        max_frames = 4
    n_frames = random.randint(1, max_frames)
    sides = random.sample(SIDES, n_frames)

    for side in sides:
        if not _place_frame(grid, blocked, side):
            return None

    grey_components = _components(grid == GREY)
    if len(grey_components) != n_frames:
        return None
    if any(_parse_frame(comp) is None for comp in grey_components):
        return None
    if not np.any((grid != BLACK) & (grid != GREY)):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-branch puzzle")
