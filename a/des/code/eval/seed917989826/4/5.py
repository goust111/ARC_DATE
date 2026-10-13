from __future__ import annotations

from collections import Counter
import random

import numpy as np


YELLOW = 4
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

ARROW_SHAPES = {
    "right": {(1, 0), (0, 1), (2, 1)},
    "left": {(0, 0), (1, 1), (2, 0)},
    "up": {(0, 0), (0, 2), (1, 1)},
    "down": {(0, 1), (1, 0), (1, 2)},
}

GRID_MIN = 16
GRID_MAX = 28
REGION_COUNT_RANGE = (3, 5)
BOX_SIZE_RANGE = (4, 6)
GAP_CHOICES = (1, 2)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS8 if conn8 else DIRS4
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
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return {(r - r0, c - c0) for r, c in points}


def _detect_arrow(grid: np.ndarray) -> tuple[str, np.ndarray]:
    arrow_mask = grid == GREY
    comps = _components(arrow_mask, conn8=True)
    assert len(comps) == 1 and len(comps[0]) == 3, "expected a single 3-pixel grey arrow"
    shape = _normalize(comps[0])
    for direction, pattern in ARROW_SHAPES.items():
        if shape == pattern:
            return direction, arrow_mask
    raise AssertionError("unrecognized arrow shape")


def _extract_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    mask = (grid != bg) & (grid != GREY)
    comps = _components(mask, conn8=False)
    regions: list[dict[str, object]] = []
    for comp in comps:
        colors = [int(grid[r, c]) for r, c in comp if int(grid[r, c]) != YELLOW]
        assert colors, "region must contain a visible base color"
        base = Counter(colors).most_common(1)[0][0]
        yellow_count = sum(int(grid[r, c]) == YELLOW for r, c in comp)
        regions.append(
            {
                "cells": set((int(r), int(c)) for r, c in comp),
                "color": int(base),
                "yellow_count": int(yellow_count),
            }
        )
    return regions


def _scan_key(cells: set[tuple[int, int]], direction: str) -> tuple[int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if direction == "right":
        return (min(cols), min(rows), max(rows))
    if direction == "left":
        return (-max(cols), min(rows), max(rows))
    if direction == "down":
        return (min(rows), min(cols), max(cols))
    return (-max(rows), min(cols), max(cols))


def _boundary_cells(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in cells:
                out.append((r, c))
                break
    return out


def _gap_cells_between(
    boundary_a: list[tuple[int, int]],
    cells_a: set[tuple[int, int]],
    cells_b: set[tuple[int, int]],
    occupied: np.ndarray,
    arrow_mask: np.ndarray,
) -> set[tuple[int, int]] | None:
    h, w = occupied.shape
    gaps: set[tuple[int, int]] = set()
    found = False
    for r, c in boundary_a:
        for dr, dc in DIRS4:
            rr, cc = r, c
            path: list[tuple[int, int]] = []
            for _ in range(3):
                rr += dr
                cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                pos = (rr, cc)
                if pos in cells_a:
                    break
                if pos in cells_b:
                    if 1 <= len(path) <= 2:
                        gaps.update(path)
                        found = True
                    break
                if len(path) == 2:
                    break
                if occupied[rr, cc] != 0 or arrow_mask[rr, cc]:
                    break
                path.append(pos)
    return gaps if found else None


def _priority_order(regions: list[dict[str, object]], direction: str) -> list[int]:
    return sorted(
        range(len(regions)),
        key=lambda idx: (
            -int(regions[idx]["yellow_count"]),
            _scan_key(regions[idx]["cells"], direction),
        ),
    )


def _pairwise_gaps(
    regions: list[dict[str, object]], arrow_mask: np.ndarray, shape: tuple[int, int]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    occupied = np.zeros(shape, dtype=int)
    for idx, region in enumerate(regions):
        for r, c in region["cells"]:
            occupied[r, c] = idx + 1

    boundaries = [_boundary_cells(region["cells"], *shape) for region in regions]
    gaps: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            seg = _gap_cells_between(
                boundaries[i],
                regions[i]["cells"],
                regions[j]["cells"],
                occupied,
                arrow_mask,
            )
            if seg:
                gaps[(i, j)] = seg
    return gaps


def _simulate(
    regions: list[dict[str, object]], direction: str, arrow_mask: np.ndarray, shape: tuple[int, int], bg: int
) -> tuple[np.ndarray, int, int]:
    order = _priority_order(regions, direction)
    rank = {idx: pos for pos, idx in enumerate(order)}
    pair_gaps = _pairwise_gaps(regions, arrow_mask, shape)

    alive = [True] * len(regions)
    final_cells = [set(region["cells"]) for region in regions]
    absorption_count = 0

    for idx in order:
        if not alive[idx]:
            continue
        targets: list[tuple[int, set[tuple[int, int]]]] = []
        for other in order:
            if other == idx or not alive[other] or rank[other] <= rank[idx]:
                continue
            key = (idx, other) if idx < other else (other, idx)
            seg = pair_gaps.get(key)
            if seg:
                targets.append((other, seg))
        for other, seg in targets:
            if not alive[other]:
                continue
            alive[other] = False
            absorption_count += 1
            final_cells[idx].update(final_cells[other])
            final_cells[idx].update(seg)

    out = np.full(shape, bg, dtype=int)
    for idx, is_alive in enumerate(alive):
        if not is_alive:
            continue
        color = int(regions[idx]["color"])
        for r, c in final_cells[idx]:
            out[r, c] = color

    return out, absorption_count, sum(alive)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge colored regions by priority from yellow marker counts.

    Priority is determined by:
    1. More yellow markers inside a region.
    2. If tied, the region encountered earlier when scanning in the grey arrow direction.

    Each region absorbs adjacent lower-priority regions across 1-2 cell straight background gaps.
    Yellow markers disappear in the output and the grey arrow is erased back to background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    direction, arrow_mask = _detect_arrow(g)
    regions = _extract_regions(g, bg)
    out, _, _ = _simulate(regions, direction, arrow_mask, g.shape, bg)
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask, conn8=False)
    return len(comps) == 1


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    ok = False
                    break
            if ok:
                cells.append((r, c))
    return cells


def _make_blob(height: int, width: int, needed_markers: int, axis: str) -> np.ndarray:
    while True:
        mask = np.ones((height, width), dtype=bool)
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        random.shuffle(corners)
        for r, c in corners[: random.randint(0, 3)]:
            mask[r, c] = False

        if axis == "horizontal" and width >= 5 and random.random() < 0.6:
            c = random.randint(1, width - 2)
            mask[random.choice([0, height - 1]), c] = False
        if axis == "vertical" and height >= 5 and random.random() < 0.6:
            r = random.randint(1, height - 2)
            mask[r, random.choice([0, width - 1])] = False

        if not _connected(mask):
            continue
        if len(_interior_cells(mask)) < needed_markers:
            continue
        return mask


def _sample_marker_counts(n_regions: int, require_tie: bool) -> list[int]:
    if not require_tie:
        return random.sample(list(range(5)), n_regions)
    while True:
        counts = [random.randint(0, 4) for _ in range(n_regions)]
        if len(set(counts)) < n_regions:
            return counts


def _overlap_len(start_a: int, size_a: int, start_b: int, size_b: int) -> int:
    end_a = start_a + size_a - 1
    end_b = start_b + size_b - 1
    return max(0, min(end_a, end_b) - max(start_a, start_b) + 1)


def _place_chain(
    direction: str, heights: list[int], widths: list[int], gaps: list[int]
) -> tuple[tuple[int, int], list[tuple[int, int]], str] | None:
    n = len(heights)
    axis = "horizontal" if direction in ("left", "right") else "vertical"
    if axis == "horizontal":
        total_w = sum(widths) + sum(gaps)
        w = total_w + random.randint(6, 10)
        h = max(max(heights) + random.randint(8, 12), GRID_MIN)
        if h > GRID_MAX or w > GRID_MAX:
            return None
        left = random.randint(2, w - total_w - 2)
        x = left
        tops: list[int] = []
        prev_top = random.randint(2, h - heights[0] - 3)
        tops.append(prev_top)
        for i in range(1, n):
            feasible = [
                t
                for t in range(2, h - heights[i] - 2)
                if _overlap_len(prev_top, heights[i - 1], t, heights[i]) >= 2
            ]
            if not feasible:
                return None
            prev_top = random.choice(feasible)
            tops.append(prev_top)
        positions = []
        for i in range(n):
            positions.append((tops[i], x))
            x += widths[i]
            if i < n - 1:
                x += gaps[i]
        return (h, w), positions, axis

    total_h = sum(heights) + sum(gaps)
    h = total_h + random.randint(6, 10)
    w = max(max(widths) + random.randint(8, 12), GRID_MIN)
    if h > GRID_MAX or w > GRID_MAX:
        return None
    top = random.randint(2, h - total_h - 2)
    y = top
    lefts: list[int] = []
    prev_left = random.randint(2, w - widths[0] - 3)
    lefts.append(prev_left)
    for i in range(1, n):
        feasible = [
            c
            for c in range(2, w - widths[i] - 2)
            if _overlap_len(prev_left, widths[i - 1], c, widths[i]) >= 2
        ]
        if not feasible:
            return None
        prev_left = random.choice(feasible)
        lefts.append(prev_left)
    positions = []
    for i in range(n):
        positions.append((y, lefts[i]))
        y += heights[i]
        if i < n - 1:
            y += gaps[i]
    return (h, w), positions, axis


def _place_arrow(grid: np.ndarray, direction: str) -> bool:
    pattern = ARROW_SHAPES[direction]
    bh = max(r for r, _ in pattern) + 1
    bw = max(c for _, c in pattern) + 1
    h, w = grid.shape
    corners = [
        (1, 1),
        (1, w - bw - 1),
        (h - bh - 1, 1),
        (h - bh - 1, w - bw - 1),
    ]
    random.shuffle(corners)
    bg = _most_common_color(grid)
    for r0, c0 in corners:
        cells = {(r0 + r, c0 + c) for r, c in pattern}
        ok = True
        for r, c in cells:
            if grid[r, c] != bg:
                ok = False
                break
            for dr, dc in DIRS8 + [(0, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and grid[nr, nc] != bg:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c in cells:
            grid[r, c] = GREY
        return True
    return False


def _attempt() -> np.ndarray | None:
    direction = random.choice(["left", "right", "up", "down"])
    n_regions = random.randint(*REGION_COUNT_RANGE)
    require_tie = random.random() < 0.5
    marker_counts = _sample_marker_counts(n_regions, require_tie)
    heights = [random.randint(*BOX_SIZE_RANGE) for _ in range(n_regions)]
    widths = [random.randint(*BOX_SIZE_RANGE) for _ in range(n_regions)]
    gaps = [random.choice(GAP_CHOICES) for _ in range(n_regions - 1)]

    placement = _place_chain(direction, heights, widths, gaps)
    if placement is None:
        return None
    (h, w), positions, axis = placement

    bg = random.choice([c for c in range(10) if c not in (YELLOW, GREY)])
    palette = [c for c in range(10) if c not in (bg, YELLOW, GREY)]
    if len(palette) < n_regions:
        return None
    region_colors = random.sample(palette, n_regions)

    grid = np.full((h, w), bg, dtype=int)
    region_meta: list[dict[str, object]] = []

    for idx, ((r0, c0), hh, ww, color, yellow_count) in enumerate(
        zip(positions, heights, widths, region_colors, marker_counts, strict=False)
    ):
        local_mask = _make_blob(hh, ww, yellow_count, axis)
        cells: set[tuple[int, int]] = set()
        for rr in range(hh):
            for cc in range(ww):
                if local_mask[rr, cc]:
                    cells.add((r0 + rr, c0 + cc))
        for r, c in cells:
            grid[r, c] = color

        interior = _interior_cells(local_mask)
        if yellow_count:
            for rr, cc in random.sample(interior, yellow_count):
                grid[r0 + rr, c0 + cc] = YELLOW

        region_meta.append(
            {
                "cells": cells,
                "color": color,
                "yellow_count": yellow_count,
            }
        )

    if not _place_arrow(grid, direction):
        return None

    arrow_mask = grid == GREY
    expected, absorptions, survivors = _simulate(region_meta, direction, arrow_mask, grid.shape, bg)
    if absorptions == 0 or survivors <= 1 or survivors >= n_regions:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
