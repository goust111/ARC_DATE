from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
OBSTACLE = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_RANGE = (14, 19)
PAIR_COUNT_CHOICES = (3, 4)
GENERATION_ATTEMPTS = 500


def _bresenham_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)

    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        points.append((int(r0), int(c0)))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
    return points


def _landing_row(grid: np.ndarray, r: int, c: int) -> int:
    H = int(grid.shape[0])
    for rr in range(int(r) + 1, H):
        if int(grid[rr, int(c)]) == OBSTACLE:
            return int(rr - 1)
    return int(H - 1)


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, _W = g.shape
    assert H >= 4, "grid must have room for top markers and falling space"

    pairs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, OBSTACLE}):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each marker color must appear exactly twice"
        endpoints = []
        for r, c in coords:
            assert r in (0, 1), "markers must be in the top two rows"
            endpoints.append((_landing_row(g, r, c), int(c)))
        endpoints.sort(key=lambda rc: (rc[1], rc[0]))
        assert endpoints[0][1] != endpoints[1][1], "a pair must occupy two distinct columns"
        pairs.append(
            {
                "color": int(color),
                "marker_coords": sorted(coords),
                "endpoints": endpoints,
            }
        )

    assert 3 <= len(pairs) <= 4, "expected 3 or 4 marker pairs"
    left_cols = [pair["endpoints"][0][1] for pair in pairs]
    assert len(left_cols) == len(set(left_cols)), "left endpoints must be uniquely ordered"
    pairs.sort(key=lambda pair: (pair["endpoints"][0][1], pair["endpoints"][0][0]))
    return pairs


def _render_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    out[g == OBSTACLE] = OBSTACLE

    cover_order: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pair in pairs:
        a, b = pair["endpoints"]
        for pt in _bresenham_points(a, b):
            cover_order[pt].append(int(pair["color"]))

    for (r, c), colors in cover_order.items():
        if len(colors) % 2 == 1:
            out[int(r), int(c)] = int(colors[-1])
        else:
            out[int(r), int(c)] = int(colors[0])

    out[g == OBSTACLE] = OBSTACLE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop paired top markers onto obstacles or the floor, connect each landed pair, and resolve line crossings by coverage parity.

    Input structure:
    - A black background grid contains scattered grey obstacle pixels.
    - The top two rows contain 3 or 4 color pairs, with each non-grey color
      appearing exactly twice as two single-pixel markers in different columns.

    Rule:
    - Each marker falls straight down within its own column until the cell
      below it is a grey obstacle or it reaches the bottom row.
    - For each same-color pair, draw the inclusive Bresenham line between the
      two landing positions using that pair's color.
    - Draw pairs in order of the landed pair's left endpoint column, breaking
      ties by that left endpoint row.
    - If a pixel is covered by multiple lines, show the last covering line's
      color when the total coverage count is odd, otherwise show the first
      covering line's color.
    - Grey obstacles remain grey in the output, and the original top markers
      disappear because only the landed lines are kept.

    Key insight:
    - Gravity determines the true line endpoints, so the visible crossing
      colors can only be inferred after combining falling, ordered line drawing,
      and the odd-versus-even overwrite rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pairs = _extract_pairs(g)
    return _render_output(g, pairs)


def _line_center_crosses(
    lines: list[list[tuple[int, int]]], center: tuple[int, int]
) -> bool:
    center = (int(center[0]), int(center[1]))
    for pts in lines:
        if center not in set(pts):
            return False
    for i in range(len(lines)):
        hit = False
        set_i = set(lines[i])
        for j in range(len(lines)):
            if i == j:
                continue
            inter = set_i & set(lines[j])
            if center in inter:
                hit = True
            if inter - {center}:
                return False
        if not hit:
            return False
    return True


def _valid_steps(
    center: tuple[int, int], vec: tuple[int, int], H: int, W: int
) -> tuple[list[int], list[int]]:
    cr, cc = map(int, center)
    dr, dc = map(int, vec)
    left_steps: list[int] = []
    right_steps: list[int] = []

    for k in range(2, max(H, W)):
        lr = cr - k * dr
        lc = cc - k * dc
        if 3 <= lr <= H - 1 and 0 <= lc < W:
            left_steps.append(int(k))
        rr = cr + k * dr
        rc = cc + k * dc
        if 3 <= rr <= H - 1 and 0 <= rc < W:
            right_steps.append(int(k))
    return left_steps, right_steps


def _semantic_checks(grid: np.ndarray, out: np.ndarray, pairs: list[dict], center: tuple[int, int]) -> bool:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if not (g.ndim == 2 and out.ndim == 2 and out.shape == g.shape and max(g.shape) <= 30):
        return False

    landing_positions = [endpoint for pair in pairs for endpoint in pair["endpoints"]]
    if len(landing_positions) != len(set(landing_positions)):
        return False

    cover_counts = np.zeros((H, W), dtype=int)
    lines = []
    obstacle_line_overlaps = 0
    for pair in pairs:
        pts = _bresenham_points(pair["endpoints"][0], pair["endpoints"][1])
        lines.append(pts)
        for r, c in pts:
            obstacle_line_overlaps += int(g[r, c] == OBSTACLE)
            cover_counts[r, c] += 1

    if obstacle_line_overlaps > 1:
        return False

    if not _line_center_crosses(lines, center):
        return False
    if int(cover_counts[center]) != len(pairs):
        return False
    if len(pairs) == 3 and int(out[center]) != int(pairs[-1]["color"]):
        return False
    if len(pairs) == 4 and int(out[center]) != int(pairs[0]["color"]):
        return False
    if not any(int(r) == H - 1 for r, _c in landing_positions):
        supported = sum(int(g[r + 1, c]) == OBSTACLE for r, c in landing_positions if r + 1 < H)
        if supported < 5:
            return False

    extra_obstacles = int(np.count_nonzero(g == OBSTACLE))
    if not (5 <= extra_obstacles <= 10):
        return False
    if np.array_equal(g, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    want_bottom = random.random() < 0.5
    specs3 = [(-1, 1, 3, 3), (1, 1, 4, 4), (-1, 3, 2, 2)]
    specs4 = specs3 + [(-3, 1, 2, 2)]
    specs = specs3 if pair_count == 3 else specs4

    if want_bottom:
        center_r = H - (5 if pair_count == 3 else 7)
    else:
        lo = 9 if pair_count == 4 else 7
        hi = H - (7 if pair_count == 4 else 5)
        if lo > hi:
            return None
        center_r = random.randint(lo, hi)
    center_c = random.randint(6, W - 7)
    center = (int(center_r), int(center_c))

    endpoints: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr, dc, kl, kr in specs:
        left = (center[0] - kl * dr, center[1] - kl * dc)
        right = (center[0] + kr * dr, center[1] + kr * dc)
        if not (3 <= left[0] < H and 3 <= right[0] < H and 0 <= left[1] < W and 0 <= right[1] < W):
            return None
        endpoints.append((left, right))

    pairs = []
    colors = random.sample(MARKER_COLORS, pair_count)
    for color, (left, right) in zip(colors, endpoints):
        a, b = sorted([left, right], key=lambda rc: (rc[1], rc[0]))
        pairs.append({"color": int(color), "endpoints": [a, b]})
    pairs.sort(key=lambda pair: (pair["endpoints"][0][1], pair["endpoints"][0][0]))

    lines = [_bresenham_points(pair["endpoints"][0], pair["endpoints"][1]) for pair in pairs]
    if not _line_center_crosses(lines, center):
        return None

    grid = np.full((H, W), BG, dtype=int)
    used_cols = {pair["endpoints"][0][1] for pair in pairs} | {pair["endpoints"][1][1] for pair in pairs}
    for pair in pairs:
        for r, c in pair["endpoints"]:
            if r < H - 1:
                grid[r + 1, c] = OBSTACLE

    line_points = set().union(*(set(pts) for pts in lines))
    for pair in pairs:
        left, right = pair["endpoints"]
        grid[random.randint(0, 1), left[1]] = int(pair["color"])
        grid[random.randint(0, 1), right[1]] = int(pair["color"])

    target_obstacles = random.randint(max(5, int(np.count_nonzero(grid == OBSTACLE))), 10)
    free_candidates = [
        (r, c)
        for r in range(2, H - 1)
        for c in range(W)
        if grid[r, c] == BG
        and (r, c) not in line_points
        and c not in used_cols
    ]
    random.shuffle(free_candidates)
    for r, c in free_candidates:
        if int(np.count_nonzero(grid == OBSTACLE)) >= target_obstacles:
            break
        grid[r, c] = OBSTACLE

    if not (5 <= int(np.count_nonzero(grid == OBSTACLE)) <= 10):
        return None

    try:
        parsed_pairs = _extract_pairs(grid)
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, _render_output(grid, parsed_pairs)):
        return None
    if not _semantic_checks(grid, out, parsed_pairs, center):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
