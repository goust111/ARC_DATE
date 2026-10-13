from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
CYAN = 8

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_ATTEMPTS = 80
_GEOM_ATTEMPTS = 3000


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Minimal 4-connected component helper using the arc_utils (r,c) convention."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comp = np.zeros_like(mask, dtype=bool)
        rows, cols = zip(*coords)
        comp[rows, cols] = True
        components.append(comp)
    return components


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src = mask[
                max(0, -dr) : h - max(0, dr),
                max(0, -dc) : w - max(0, dc),
            ]
            dst = out[
                max(0, dr) : h - max(0, -dr),
                max(0, dc) : w - max(0, -dc),
            ]
            dst |= src
    return out


def _polyline_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            mask[r0, c_lo : c_hi + 1] = True
        else:
            assert c0 == c1
            r_lo, r_hi = sorted((r0, r1))
            mask[r_lo : r_hi + 1, c0] = True
    return mask


def _channel_from_grid(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    orange = grid == ORANGE
    comps = _component_masks(orange)
    assert len(comps) == 2, "expected exactly two orange components"
    channel = _dilate8(comps[0]) & _dilate8(comps[1])
    return channel, comps


def _marker_dict(grid: np.ndarray, channel: np.ndarray) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((grid == RED) | (grid == GREEN)):
        r = int(r)
        c = int(c)
        assert channel[r, c], "all markers must lie on the channel"
        markers[(r, c)] = int(grid[r, c])
    assert markers, "expected at least one marker"
    return markers


def _select_start(markers: dict[tuple[int, int], int]) -> tuple[int, int]:
    return min(markers, key=lambda rc: (rc[0], rc[1]))


def _trace_channel(
    channel: np.ndarray,
    markers: dict[tuple[int, int], int],
    start: tuple[int, int] | None = None,
    *,
    collect_opportunities: bool = False,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if start is None:
        start = _select_start(markers)
    h, w = channel.shape
    start_color = markers[start]
    visited: list[tuple[int, int]] = []
    visited_set: set[tuple[int, int]] = set()
    opportunities: list[tuple[int, int]] = []
    current = start
    direction: tuple[int, int] | None = None

    while True:
        if current in visited_set:
            break
        visited.append(current)
        visited_set.add(current)
        r, c = current
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            break

        unvisited = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and channel[nr, nc] and (nr, nc) not in visited_set
        ]
        if not unvisited:
            break

        perpendicular: list[tuple[int, int]] = []
        if direction is not None:
            perpendicular = [
                cell
                for cell in unvisited
                if (cell[0] - r) * direction[0] + (cell[1] - c) * direction[1] == 0
            ]
            if collect_opportunities and len(unvisited) >= 2 and perpendicular:
                opportunities.append(current)

        next_cell: tuple[int, int] | None = None
        if direction is not None and markers.get(current) != start_color and perpendicular:
            next_cell = min(perpendicular)
        else:
            next_cell = min(unvisited)

        direction = (next_cell[0] - r, next_cell[1] - c)
        current = next_cell

    return visited, opportunities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the part of the dilated-intersection channel visited by a marker-guided ray.

    Input structure:
    - Black background surrounds exactly two orange single-pixel-width 4-connected path components.
    - Their one-cell 8-neighborhood dilations overlap in a middle channel; all red/green markers lie on that channel.
    - The topmost marker is the ray start, and lower markers may sit on straight segments or at channel branch points.

    Rule:
    - Dilate each orange component by one cell in the 8-neighborhood and intersect the two dilations to obtain the channel.
    - Choose the channel marker closest to the top edge as the start; ties are broken by smaller column.
    - Trace a ray along channel cells, preferring to continue straight; when the ray reaches a marker whose color differs from the start color, it instead takes an available perpendicular branch.
    - Color every visited channel cell cyan, leave untouched channel cells black, and keep all orange paths plus unvisited markers unchanged.

    Key insight:
    - The orange components define the walkable geometry, but the marker colors decide whether the ray passes straight through or makes a forced 90-degree turn inside that geometry.
    """
    g = np.asarray(grid, dtype=int)
    channel, _ = _channel_from_grid(g)
    markers = _marker_dict(g, channel)
    start = _select_start(markers)
    visited, _ = _trace_channel(channel, markers, start)

    out = g.copy()
    for r, c in visited:
        if out[r, c] != ORANGE:
            out[r, c] = CYAN
    return out


def _turn_effective(
    channel: np.ndarray,
    start: tuple[int, int],
    markers: dict[tuple[int, int], int],
    cell: tuple[int, int],
) -> tuple[bool, list[tuple[int, int]]]:
    trial = dict(markers)
    trial[cell] = GREEN
    path, _ = _trace_channel(channel, trial, start)
    base_path, _ = _trace_channel(channel, markers, start)
    return set(path) != set(base_path), path


def _find_endpoint(mask: np.ndarray, *, topmost: bool) -> tuple[int, int] | None:
    h, w = mask.shape
    endpoints = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        degree = sum(
            1
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]
        )
        if degree <= 1 and 0 < r < h - 1 and 0 < c < w - 1:
            endpoints.append((r, c))
    if not endpoints:
        return None
    key = (lambda rc: (rc[0], rc[1])) if topmost else (lambda rc: (-rc[0], rc[1]))
    return min(endpoints, key=key)


def _build_serpentine_paths() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(_GEOM_ATTEMPTS):
        width = random.randint(18, 24)
        top = random.randint(2, 4)
        r1 = top + random.randint(4, 6)
        r2 = r1 + random.randint(6, 9)
        bottom = r2 + random.randint(5, 8)
        height = bottom + 2
        if height > 30:
            continue

        c0 = random.randint(4, width - 10)
        left_out = random.randint(1, 3)
        right_step = random.randint(1, 2)
        left_back = right_step + random.choice([0, 0, 1])
        right_out = random.randint(3, 5)

        left_points = [
            (top, c0),
            (r1, c0),
            (r1, c0 - left_out),
            (r1 + 1, c0 - left_out),
            (r1 + 1, c0 + left_back),
            (bottom, c0 + left_back),
        ]
        right_points = [
            (top, c0 + 2),
            (r1, c0 + 2),
            (r1, c0 + 2 + right_step),
            (r2, c0 + 2 + right_step),
            (r2, c0 + 2 + right_step + right_out),
            (bottom, c0 + 2 + right_step + right_out),
        ]

        if random.random() < 0.5:
            left_points = [(r, width - 1 - c) for r, c in left_points]
            right_points = [(r, width - 1 - c) for r, c in right_points]

        shape = (height, width)
        left_mask = _polyline_mask(shape, left_points)
        right_mask = _polyline_mask(shape, right_points)
        if np.any(left_mask & right_mask):
            continue
        for r, c in np.argwhere(left_mask):
            r = int(r)
            c = int(c)
            if any(
                0 <= nr < height and 0 <= nc < width and right_mask[nr, nc]
                for nr, nc in _neighbors4(r, c)
            ):
                break
        else:
            return left_mask, right_mask
    return None


def _choose_markers(
    channel: np.ndarray,
    blocked: np.ndarray | None = None,
) -> dict[tuple[int, int], int] | None:
    if blocked is None:
        blocked = np.zeros_like(channel, dtype=bool)

    start = _find_endpoint(channel & ~blocked, topmost=True)
    if start is None:
        return None

    markers: dict[tuple[int, int], int] = {start: RED}
    base_path, base_ops = _trace_channel(channel, markers, start, collect_opportunities=True)
    if len(base_path) < 10:
        return None

    green1_candidates = [cell for cell in base_ops if not blocked[cell]]
    if not green1_candidates:
        green1_candidates = [
            cell
            for cell in base_path[2:-2]
            if not blocked[cell] and cell != start
        ]
    if not green1_candidates:
        return None

    green1 = random.choice(green1_candidates)
    markers[green1] = GREEN
    path_after_green1, ops_after_green1 = _trace_channel(
        channel, markers, start, collect_opportunities=True
    )

    want_second_green = random.random() < 0.35
    if want_second_green:
        green2_candidates = [
            cell for cell in path_after_green1[2:-2] if cell not in markers and not blocked[cell]
        ]
        if green2_candidates:
            markers[random.choice(green2_candidates)] = GREEN

    final_path, final_ops = _trace_channel(channel, markers, start, collect_opportunities=True)
    if len(final_path) < 10 or len(final_path) >= int(channel.sum()):
        return None

    want_second_red = random.random() < 0.55
    if want_second_red:
        red_candidates = [
            cell for cell in final_ops if cell not in markers and not blocked[cell]
        ]
        if not red_candidates:
            red_candidates = [
                cell for cell in final_path[2:-2] if cell not in markers and not blocked[cell]
            ]
        if red_candidates:
            markers[random.choice(red_candidates)] = RED

    # Optional untouched marker to show that unvisited markers remain unchanged.
    if random.random() < 0.35:
        used_colors = list(markers.values())
        extra_color = GREEN if used_colors.count(GREEN) < 2 else RED
        if used_colors.count(extra_color) < 2:
            off_path = [
                (int(r), int(c))
                for r, c in np.argwhere(channel)
                if (int(r), int(c)) not in markers
                and not blocked[int(r), int(c)]
                and (int(r), int(c)) not in final_path
                and int(r) > start[0]
            ]
            off_path = [
                cell
                for cell in off_path
                if all(abs(cell[0] - mr) + abs(cell[1] - mc) >= 3 for mr, mc in markers)
            ]
            if off_path:
                markers[random.choice(off_path)] = extra_color

    red_count = sum(color == RED for color in markers.values())
    green_count = sum(color == GREEN for color in markers.values())
    if not (1 <= red_count <= 2 and 1 <= green_count <= 2):
        return None
    if _select_start(markers) != start:
        return None
    return markers


def _pad_grid(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    max_top = max(0, min(3, 30 - h))
    pad_top = random.randint(0, max_top)
    max_bottom = max(0, min(5, 30 - h - pad_top))
    pad_bottom = random.randint(0, max_bottom)
    max_left = max(0, min(3, 30 - w))
    pad_left = random.randint(0, max_left)
    max_right = max(0, min(5, 30 - w - pad_left))
    pad_right = random.randint(0, max_right)
    out = np.zeros((h + pad_top + pad_bottom, w + pad_left + pad_right), dtype=int)
    out[pad_top : pad_top + h, pad_left : pad_left + w] = grid
    return out


def _grid_from_masks(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    markers: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros(left_mask.shape, dtype=int)
    grid[left_mask] = ORANGE
    grid[right_mask] = ORANGE
    for (r, c), color in markers.items():
        grid[r, c] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    channel, comps = _channel_from_grid(grid)
    if len(comps) != 2:
        return False
    if len(_component_masks(channel)) != 1:
        return False

    markers = _marker_dict(grid, channel)
    start = _select_start(markers)
    if markers[start] != RED:
        return False

    visited, ops = _trace_channel(channel, markers, start, collect_opportunities=True)
    if len(visited) < 10 or len(visited) >= int(channel.sum()):
        return False
    if not ops and len(markers) < 3:
        return False
    if not any(color == GREEN for color in markers.values()):
        return False

    out = main(grid)
    visible_visited = sum(grid[r, c] != ORANGE for r, c in visited)
    cyan_count = int(np.sum(out == CYAN))
    if cyan_count != visible_visited:
        return False
    if np.any((grid == ORANGE) & (out != ORANGE)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    built = _build_serpentine_paths()
    if built is None:
        return None
    left_mask, right_mask = built
    grid0 = np.zeros(left_mask.shape, dtype=int)
    grid0[left_mask] = ORANGE
    grid0[right_mask] = ORANGE

    channel, comps = _channel_from_grid(grid0)
    if len(comps) != 2 or len(_component_masks(channel)) != 1:
        return None
    if int(channel.sum()) < 16 or int(channel.sum()) > 60:
        return None

    markers = _choose_markers(channel, blocked=grid0 == ORANGE)
    if markers is None:
        return None
    grid = _grid_from_masks(left_mask, right_mask, markers)
    grid = _pad_grid(grid)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-in-channel puzzle")
