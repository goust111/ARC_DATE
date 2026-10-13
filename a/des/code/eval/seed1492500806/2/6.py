import random
from typing import Optional

import numpy as np


BG = 0
PLATFORM = 5
CONTAINER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_H, MAX_H = 12, 24
MIN_W, MAX_W = 14, 30
GEN_ATTEMPTS = 400


def _segments_in_row(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for c, value in enumerate(mask.tolist()):
        if value and start is None:
            start = c
        elif not value and start is not None:
            segs.append((start, c - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def _parse_container(grid: np.ndarray, color: int) -> Optional[dict]:
    pts = np.argwhere(grid == color)
    if pts.size == 0:
        return None

    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return None

    bottom_inner = grid[r1, c0 + 1 : c1]
    if bottom_inner.size == 0:
        return None
    if not np.all((bottom_inner == color) | (bottom_inner == BG)):
        return None

    gap_idx = np.where(bottom_inner == BG)[0]
    if gap_idx.size == 0:
        return None
    if int(gap_idx[-1]) - int(gap_idx[0]) + 1 != int(gap_idx.size):
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None
    if not np.all((interior == BG) | (interior == color)):
        return None

    gap0 = c0 + 1 + int(gap_idx[0])
    gap1 = c0 + 1 + int(gap_idx[-1])
    return {
        "color": int(color),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "gap0": gap0,
        "gap1": gap1,
        "sand_count": int(np.count_nonzero(interior == color)),
    }


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, PLATFORM):
            continue
        info = _parse_container(grid, color)
        if info is not None:
            containers.append(info)
    containers.sort(key=lambda item: (item["r0"], item["c0"]))
    return containers


def _find_platform_row(grid: np.ndarray, container: dict) -> int:
    r1 = int(container["r1"])
    c0 = int(container["c0"])
    c1 = int(container["c1"])
    h, _ = grid.shape
    for r in range(r1 + 1, h):
        segs = _segments_in_row(grid[r] == PLATFORM)
        for s0, s1 in segs:
            if s0 <= c0 and c1 <= s1:
                return r
    raise AssertionError("container must have a matching platform below it")


def _settled_positions(container: dict, platform_row: int) -> list[tuple[int, int]]:
    cols = list(range(int(container["gap0"]), int(container["gap1"]) + 1))
    positions: list[tuple[int, int]] = []

    for r in range(platform_row - 1, int(container["r1"]), -1):
        for c in cols:
            positions.append((r, c))

    for r in range(int(container["r1"]) - 1, int(container["r0"]), -1):
        for c in cols:
            positions.append((r, c))

    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate sand leaking from framed containers through bottom openings.

    The generator uses black background/openings (0) and grey platforms (5).
    Each container has a unique visible frame/sand color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for container in _find_containers(g):
        r0 = int(container["r0"])
        c0 = int(container["c0"])
        r1 = int(container["r1"])
        c1 = int(container["c1"])
        color = int(container["color"])
        sand_count = int(container["sand_count"])

        platform_row = _find_platform_row(g, container)
        positions = _settled_positions(container, platform_row)

        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for r, c in positions[:sand_count]:
            out[r, c] = color

    return out


def _draw_container(grid: np.ndarray, spec: dict, rng: random.Random) -> None:
    color = int(spec["color"])
    top = int(spec["top"])
    left = int(spec["left"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    gap0 = int(spec["gap0"])
    gap1 = int(spec["gap1"])
    platform_row = int(spec["platform_row"])
    platform_left = int(spec["platform_left"])
    platform_right = int(spec["platform_right"])
    sand_count = int(spec["sand_count"])

    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[bottom, gap0 : gap1 + 1] = BG
    grid[platform_row, platform_left : platform_right + 1] = PLATFORM

    interior_cells = [
        (r, c)
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
    ]
    for r, c in rng.sample(interior_cells, sand_count):
        grid[r, c] = color


def _make_spec(rng: random.Random, color: int, num_containers: int) -> Optional[dict]:
    if num_containers == 1:
        outer_w = rng.randint(5, 8)
        extra_max = 2
        shaft_max = 6
    elif num_containers == 2:
        outer_w = rng.randint(5, 7)
        extra_max = 2
        shaft_max = 5
    else:
        outer_w = rng.randint(5, 6)
        extra_max = 1
        shaft_max = 4

    outer_h = rng.randint(5, 8)
    interior_w = outer_w - 2
    interior_h = outer_h - 2
    gap_w = rng.randint(1, interior_w)
    gap_start_rel = rng.randint(1, interior_w - gap_w + 1)
    left_extra = rng.randint(0, extra_max)
    right_extra = rng.randint(0, extra_max)
    shaft_depth = rng.randint(2, shaft_max)

    outer_cap = gap_w * shaft_depth
    queue_cap = gap_w * interior_h
    max_sand = min(interior_w * interior_h, outer_cap + queue_cap)
    if max_sand <= 0:
        return None

    want_residual = bool(rng.getrandbits(1))
    if want_residual and max_sand > outer_cap:
        sand_count = rng.randint(outer_cap + 1, max_sand)
        residual = True
    else:
        sand_count = rng.randint(1, min(outer_cap, max_sand))
        residual = False

    return {
        "color": int(color),
        "outer_h": outer_h,
        "outer_w": outer_w,
        "gap_start_rel": gap_start_rel,
        "gap_w": gap_w,
        "left_extra": left_extra,
        "right_extra": right_extra,
        "shaft_depth": shaft_depth,
        "sand_count": sand_count,
        "residual": residual,
    }


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    num_containers = rng.choice([1, 2, 2, 3])
    colors = CONTAINER_COLORS[:]
    rng.shuffle(colors)

    specs = []
    for idx in range(num_containers):
        spec = _make_spec(rng, colors[idx], num_containers)
        if spec is None:
            return None
        specs.append(spec)

    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    cursor = left_margin
    max_row = 0
    branch_seen = set()

    for i, spec in enumerate(specs):
        top = rng.randint(1, 4)
        span_w = int(spec["outer_w"]) + int(spec["left_extra"]) + int(spec["right_extra"])
        if i < len(specs) - 1:
            gap_after = rng.randint(2, 4 if num_containers < 3 else 3)
        else:
            gap_after = 0

        span_left = cursor
        left = span_left + int(spec["left_extra"])
        platform_left = span_left
        platform_right = span_left + span_w - 1
        platform_row = top + int(spec["outer_h"]) + int(spec["shaft_depth"])

        spec["top"] = top
        spec["left"] = left
        spec["platform_left"] = platform_left
        spec["platform_right"] = platform_right
        spec["platform_row"] = platform_row
        spec["gap0"] = left + int(spec["gap_start_rel"]) - 1
        spec["gap1"] = int(spec["gap0"]) + int(spec["gap_w"]) - 1

        cursor = span_left + span_w + gap_after
        max_row = max(max_row, platform_row)
        branch_seen.add(bool(spec["residual"]))

    width = cursor + right_margin
    height = max_row + 1 + bottom_margin
    if not (MIN_H <= height <= MAX_H and MIN_W <= width <= MAX_W):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _draw_container(grid, spec, rng)

    parsed = _find_containers(grid)
    if len(parsed) != num_containers:
        return None
    if not branch_seen:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
