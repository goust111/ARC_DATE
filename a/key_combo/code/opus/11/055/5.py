from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

MARKER_COLORS = (GREEN, PINK)
RANK_COLORS = (YELLOW, BLUE, ORANGE, GREY)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    r, c = map(int, starts[0])
    assert c == 0, "red start pixel must be on the left edge"
    return r, c


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in np.argwhere(np.isin(grid, MARKER_COLORS))]


def _trace_ray_network(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    start_r, start_c = _find_start(g)

    markers = {(int(r), int(c)) for r, c in np.argwhere(np.isin(g, MARKER_COLORS))}
    out = g.copy()
    opened_order: list[tuple[int, int]] = []
    seen_segments: set[tuple[int, int, str]] = set()
    pending: list[tuple[int, int, str]] = [(start_r, start_c, "h")]

    while pending:
        r, c, axis = pending.pop()
        key = (r, c, axis)
        if key in seen_segments:
            continue
        seen_segments.add(key)
        out[r, c] = RED

        if axis == "h":
            directions = [(0, 1)] if key == (start_r, start_c, "h") else [(0, -1), (0, 1)]
            next_axis = "v"
        else:
            directions = [(-1, 0), (1, 0)]
            next_axis = "h"

        new_segments: list[tuple[int, int, str]] = []
        for dr, dc in directions:
            prev_r, prev_c = r, c
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if (rr, cc) in markers:
                    marker = (int(rr), int(cc))
                    if marker not in opened_order:
                        opened_order.append(marker)
                        new_segments.append((prev_r, prev_c, next_axis))
                    break
                if out[rr, cc] == BG:
                    out[rr, cc] = RED
                prev_r, prev_c = rr, cc
                rr += dr
                cc += dc

        pending.extend(new_segments)

    return out, opened_order


def _enclosed_regions(ray_grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = ray_grid.shape
    black = ray_grid == BG
    outside = np.zeros_like(black, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and black[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    seen = np.zeros_like(black, dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not black[r, c] or outside[r, c] or seen[r, c]:
                continue
            pts: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and black[nr, nc]
                        and not outside[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(pts)

    regions.sort(key=lambda pts: (-len(pts), min(pts)))
    return regions


def _solve_with_debug(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[list[tuple[int, int]]]]:
    ray_grid, opened_order = _trace_ray_network(grid)
    regions = _enclosed_regions(ray_grid)
    out = ray_grid.copy()
    for color, region in zip(RANK_COLORS, regions):
        for r, c in region:
            out[r, c] = color
    return out, ray_grid, opened_order, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the red turning-ray network, then fill the enclosed black regions by area rank.

    Input structure:
    - The grid has black background, exactly one red start pixel on the left edge, and 4 to 7 isolated interior marker pixels.
    - Marker pixels are green or pink singletons and remain unchanged in the output.
    - The markers are arranged so the recursively generated orthogonal ray network encloses 2 to 4 black regions.

    Rule:
    - Fire a horizontal red ray rightward from the start pixel; each ray keeps traversing cells until it reaches a marker or the grid boundary, painting only black cells red.
    - When a marker is hit for the first time, emit the perpendicular pair of rays from the last red cell before that marker; later hits on the same marker only stop the incoming ray.
    - After the depth-first ray expansion finishes, find every 4-connected black region that is not connected to the grid border.
    - Rank those enclosed regions by area from largest to smallest and recolor them yellow, then blue, then orange, then grey.

    Key insight:
    - The colored regions are not marked directly: they appear only after the full ray-turn network is completed, and the final fill colors depend on comparing enclosed-region areas rather than positions.
    """
    out, _ray_grid, _opened_order, _regions = _solve_with_debug(np.asarray(grid, dtype=int))
    return out


def _candidate_extra_markers(
    h: int, w: int, start_row: int, left_wall: int, right_wall: int, top_row: int, bottom_row: int, top_marker_row: int, bottom_marker_row: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    for c in range(right_wall + 2, w - 1):
        candidates.append((top_row, c))
        candidates.append((bottom_row, c))
    for c in range(1, left_wall - 1):
        candidates.append((bottom_row, c))
    for r in range(1, top_marker_row):
        candidates.append((r, right_wall))
    for r in range(bottom_marker_row + 1, h - 1):
        candidates.append((r, right_wall))
    for r in range(1, top_row - 1):
        candidates.append((r, left_wall))
    for r in range(start_row + 2, h - 1):
        candidates.append((r, left_wall))

    random.shuffle(candidates)
    return candidates


def _is_isolated(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) <= 1:
                return False
    return True


def _validate_generated_case(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or max(h, w) > 30:
        return False

    marker_pts = _marker_positions(g)
    if not (4 <= len(marker_pts) <= 7):
        return False
    if not _is_isolated(marker_pts):
        return False
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in marker_pts):
        return False

    out, ray_grid, opened_order, regions = _solve_with_debug(g)
    areas = [len(region) for region in regions]
    if len(opened_order) != len(marker_pts):
        return False
    if not (2 <= len(regions) <= 4):
        return False
    if len(set(areas)) != len(areas):
        return False
    if min(areas) < 6 or max(areas) < 12:
        return False

    expected_fill_colors = set(RANK_COLORS[: len(regions)])
    if set(np.unique(out)) & expected_fill_colors != expected_fill_colors:
        return False
    if np.count_nonzero(ray_grid == RED) < (h + w):
        return False

    for r, c in marker_pts:
        edited = g.copy()
        edited[r, c] = BG
        edited_out = main(edited)
        if np.array_equal(edited_out, out):
            return False

    return True


def _make_base_case() -> np.ndarray | None:
    sampled = _sample_base_geometry()
    if sampled is None:
        return None

    grid, _start_row, _left_wall, _right_wall, _top_row, _bottom_row, _top_marker_row, _bottom_marker_row = sampled
    if not _validate_generated_case(grid):
        return None
    return grid


def _sample_base_geometry() -> tuple[np.ndarray, int, int, int, int, int, int, int] | None:
    h = random.randint(19, 24)
    w = random.randint(19, 24)
    grid = np.zeros((h, w), dtype=int)

    start_row = random.randint(h // 3, h // 2 + 1)
    top_marker_row = random.randint(2, max(2, start_row - 5))
    bottom_marker_row = random.randint(start_row + 5, h - 3)
    right_wall = random.randint(max(w // 2, 9), w - 6)
    left_wall = random.randint(2, min(5, right_wall - 6))
    top_row = top_marker_row + 1
    bottom_row = bottom_marker_row - 1

    markers = [
        (start_row, right_wall + 1),
        (top_marker_row, right_wall),
        (top_row, left_wall - 1),
        (bottom_marker_row, right_wall),
    ]
    if len(set(markers)) != 4:
        return None
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in markers):
        return None

    grid[start_row, 0] = RED
    for r, c in markers:
        grid[r, c] = random.choice(MARKER_COLORS)
    return grid, start_row, left_wall, right_wall, top_row, bottom_row, top_marker_row, bottom_marker_row


def _make_case_with_target_count(target_count: int) -> np.ndarray | None:
    assert 5 <= target_count <= 7
    for _ in range(500):
        sampled = _sample_base_geometry()
        if sampled is None:
            continue

        grid, start_row, left_wall, right_wall, top_row, bottom_row, top_marker_row, bottom_marker_row = sampled
        h, w = grid.shape
        base_markers = _marker_positions(grid)

        candidates = _candidate_extra_markers(
            h, w, start_row, left_wall, right_wall, top_row, bottom_row, top_marker_row, bottom_marker_row
        )
        chosen: list[tuple[int, int]] = []
        for p in candidates:
            if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 1 for q in base_markers + chosen):
                continue
            chosen.append(p)
            if len(chosen) == target_count - 4:
                break
        if len(chosen) != target_count - 4:
            continue

        for r, c in chosen:
            grid[r, c] = random.choice(MARKER_COLORS)
        if _validate_generated_case(grid):
            return grid

    return None


def generate_input() -> np.ndarray:
    target_count = random.choice([4, 5, 6, 7])
    for _ in range(1000):
        if target_count == 4:
            grid = _make_base_case()
        else:
            grid = _make_case_with_target_count(target_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-enclosure ARC input")
