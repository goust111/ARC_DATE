from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_legend_row(row: np.ndarray) -> list[int]:
    colors: list[int] = []
    for value in map(int, row):
        if value == BG:
            break
        colors.append(value)
    assert len(colors) >= 3, "legend must contain at least three colors"
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    assert all(color not in (BG, GREY) for color in colors), "legend colors must exclude background and grey"
    tail = row[len(colors) :]
    assert np.all(tail == BG), "legend tail must be background"
    return colors


def _find_separator_and_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    h, _ = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for r in range(h - 1):
        if np.all(grid[r] == GREY):
            try:
                legend = _parse_legend_row(grid[r + 1])
            except AssertionError:
                continue
            candidates.append((r, legend))
    assert candidates, "expected a full-width grey separator with a legend below it"
    return candidates[-1]


def _distance_field(workspace: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    h, w = workspace.shape
    dist = np.full((h, w), -1, dtype=int)
    blocked = workspace == GREY
    mr, mc = marker
    assert not blocked[mr, mc]
    q: deque[tuple[int, int]] = deque([(mr, mc)])
    dist[mr, mc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_structure(grid: np.ndarray):
    sep, legend = _find_separator_and_legend(grid)
    workspace = np.asarray(grid[:sep, :], dtype=int)
    assert workspace.size > 0, "workspace must be non-empty"
    marker_cells = np.argwhere((workspace != BG) & (workspace != GREY))
    assert marker_cells.shape == (1, 2), "workspace must contain exactly one marker pixel"
    mr, mc = map(int, marker_cells[0])
    assert int(workspace[mr, mc]) == legend[0], "marker color must match the first legend color"
    return sep, legend, workspace, (mr, mc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the workspace with legend-driven cyclic geodesic rings that bend around grey obstacles.

    Input structure:
    - A full-width grey separator line sits near the bottom, with one legend row directly below it.
    - The legend is a left-aligned run of distinct colors followed by background cells.
    - Above the separator is a workspace on background 0 containing grey obstacles and exactly one non-grey marker pixel.

    Rule:
    - Read the legend colors from left to right as a cycle; the marker color equals the first legend color.
    - In the workspace, compute each non-obstacle cell's shortest 4-neighbor path length from the marker while forbidding travel through grey cells.
    - Reachable cells take `legend[distance mod L]`; grey obstacles stay grey; unreachable non-obstacle cells stay background 0; the separator and legend remain unchanged.

    Key insight:
    - The color rings are not plain Manhattan diamonds; they are obstacle-avoiding geodesic level sets whose phase is specified by the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, legend, workspace, marker = _extract_structure(g)
    dist = _distance_field(workspace, marker)

    out = g.copy()
    painted = np.zeros_like(workspace)
    painted[workspace == GREY] = GREY
    reachable = dist >= 0
    if np.any(reachable):
        palette = np.array(legend, dtype=int)
        painted[reachable] = palette[dist[reachable] % len(legend)]
    out[:sep, :] = painted
    return out


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + mh + pad)
    c1 = min(w, left + mw + pad)
    if np.any(occupied[r0:r1, c0:c1] & np.pad(mask, ((top - r0, r1 - top - mh), (left - c0, c1 - left - mw)), constant_values=False)):
        return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    mh, mw = mask.shape
    window = grid[top : top + mh, left : left + mw]
    window[mask] = GREY
    occupied[top : top + mh, left : left + mw] |= mask


def _rect_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _frame_mask(inner_h: int, inner_w: int) -> np.ndarray:
    outer = np.ones((inner_h + 2, inner_w + 2), dtype=bool)
    outer[1:-1, 1:-1] = False
    return outer


def _wall_and_marker(h: int, w: int) -> tuple[np.ndarray, tuple[int, int]]:
    workspace = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    orientation = random.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        thickness = random.randint(1, 2)
        gap_top = random.randint(2, max(2, h // 4))
        gap_bottom = random.randint(2, max(2, h // 4))
        length = h - gap_top - gap_bottom
        if length < 5:
            raise ValueError("workspace too short for vertical wall")
        col_lo = max(3, w // 3)
        col_hi = min(w - thickness - 4, (2 * w) // 3)
        if col_lo > col_hi:
            raise ValueError("workspace too narrow for vertical wall")
        left = random.randint(col_lo, col_hi)
        top = gap_top
        wall = _rect_mask(length, thickness)
        _place_mask(workspace, occupied, wall, top, left)

        side = random.choice(["left", "right"])
        row = random.randint(max(1, top + 1), min(h - 2, top + length - 2))
        if side == "left":
            c_lo, c_hi = 1, left - 2
        else:
            c_lo, c_hi = left + thickness + 1, w - 2
        if c_lo > c_hi:
            raise ValueError("no room for marker beside vertical wall")
        col = random.randint(max(c_lo, c_hi - 2), c_hi)
    else:
        thickness = random.randint(1, 2)
        gap_left = random.randint(2, max(2, w // 4))
        gap_right = random.randint(2, max(2, w // 4))
        length = w - gap_left - gap_right
        if length < 5:
            raise ValueError("workspace too short for horizontal wall")
        row_lo = max(3, h // 3)
        row_hi = min(h - thickness - 4, (2 * h) // 3)
        if row_lo > row_hi:
            raise ValueError("workspace too narrow for horizontal wall")
        top = random.randint(row_lo, row_hi)
        left = gap_left
        wall = _rect_mask(thickness, length)
        _place_mask(workspace, occupied, wall, top, left)

        side = random.choice(["top", "bottom"])
        col = random.randint(max(1, left + 1), min(w - 2, left + length - 2))
        if side == "top":
            r_lo, r_hi = 1, top - 2
        else:
            r_lo, r_hi = top + thickness + 1, h - 2
        if r_lo > r_hi:
            raise ValueError("no room for marker beside horizontal wall")
        row = random.randint(max(r_lo, r_hi - 2), r_hi)

    return workspace, (row, col)


def _try_place_extra_blocks(workspace: np.ndarray, marker: tuple[int, int], want_pocket: bool) -> bool:
    h, w = workspace.shape
    occupied = workspace == GREY
    occupied[max(0, marker[0] - 1) : min(h, marker[0] + 2), max(0, marker[1] - 1) : min(w, marker[1] + 2)] = True

    target_components = random.randint(1, 3)
    placed = 0
    attempts = 0
    while placed < target_components and attempts < 120:
        attempts += 1
        block_h = random.randint(2, min(4, h - 2))
        block_w = random.randint(2, min(4, w - 2))
        top = random.randint(1, h - block_h - 1)
        left = random.randint(1, w - block_w - 1)
        mask = _rect_mask(block_h, block_w)
        if not _can_place_mask(occupied, mask, top, left, pad=1):
            continue
        _place_mask(workspace, occupied, mask, top, left)
        placed += 1

    if placed == 0:
        return False

    if want_pocket:
        attempts = 0
        while attempts < 120:
            attempts += 1
            inner_h = random.randint(2, min(4, h - 6))
            inner_w = random.randint(2, min(4, w - 6))
            top = random.randint(1, h - inner_h - 3)
            left = random.randint(1, w - inner_w - 3)
            mask = _frame_mask(inner_h, inner_w)
            if not _can_place_mask(occupied, mask, top, left, pad=1):
                continue
            _place_mask(workspace, occupied, mask, top, left)
            return True
        return False

    return True


def _stats(grid: np.ndarray) -> dict[str, int]:
    sep, legend, workspace, marker = _extract_structure(grid)
    dist = _distance_field(workspace, marker)
    blocked = workspace == GREY
    rows, cols = np.indices(workspace.shape)
    manhattan = np.abs(rows - marker[0]) + np.abs(cols - marker[1])
    detour = int(np.sum((dist >= 0) & (dist > manhattan)))
    unreachable = int(np.sum((~blocked) & (dist < 0)))
    obstacle_components = _component_count(blocked)
    return {
        "sep": sep,
        "legend_len": len(legend),
        "detour": detour,
        "unreachable": unreachable,
        "obstacle_components": obstacle_components,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 20)
    w = random.randint(13, 24)
    legend_len = random.randint(4, min(7, w - 2))
    legend = random.sample(PALETTE, legend_len)
    want_pocket = random.random() < 0.5

    try:
        workspace, marker = _wall_and_marker(h, w)
    except ValueError:
        return None
    if not _try_place_extra_blocks(workspace, marker, want_pocket):
        return None

    workspace[marker] = legend[0]
    grid = np.zeros((h + 2, w), dtype=int)
    grid[:h, :] = workspace
    grid[h, :] = GREY
    grid[h + 1, :legend_len] = legend

    try:
        out = main(grid)
    except Exception:
        return None

    stats = _stats(grid)
    if stats["legend_len"] != legend_len:
        return None
    if stats["obstacle_components"] < 2:
        return None
    if stats["detour"] == 0:
        return None
    if want_pocket and stats["unreachable"] == 0:
        return None
    if not want_pocket and stats["unreachable"] != 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
