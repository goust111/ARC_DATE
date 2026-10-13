from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)

BRANCHES = ("direct", "up", "down")
GENERATION_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3)
WIDTH_RANGE = (18, 31)
BAND_HEIGHT_RANGE = (7, 10)
BAND_GAP_RANGE = (1, 4)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(shape: tuple[int, int], cell: Coord) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _collect_component(mask: np.ndarray, start: Coord, seen: np.ndarray) -> list[Coord]:
    q: deque[Coord] = deque([start])
    seen[start] = True
    coords: list[Coord] = []
    while q:
        cell = q.popleft()
        coords.append(cell)
        for nbr in _neighbors4(mask.shape, cell):
            if seen[nbr] or not bool(mask[nbr]):
                continue
            seen[nbr] = True
            q.append(nbr)
    return coords


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _segment_cells(
    start: Coord,
    end: Coord,
    *,
    include_start: bool,
    include_end: bool,
) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if start == end:
        return [start] if include_start and include_end else []

    if r0 == r1:
        step = 1 if c1 > c0 else -1
        pts = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        assert c0 == c1, "segments must be orthogonal"
        step = 1 if r1 > r0 else -1
        pts = [(r, c0) for r in range(r0, r1 + step, step)]

    if not include_start:
        pts = pts[1:]
    if not include_end:
        pts = pts[:-1]
    return pts


def _cells_clear(grid: np.ndarray, cells: list[Coord]) -> bool:
    return all(int(grid[r, c]) == BG for r, c in cells)


def _find_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    towers: list[dict] = []
    used_cells: set[Coord] = set()
    seen_colors: set[int] = set()

    for r_raw, c_raw in np.argwhere(g == YELLOW):
        r = int(r_raw)
        c = int(c_raw)
        assert 0 < r < h - 1 and 0 < c < w - 1, "yellow tower marker must have a full 3x3 patch"
        ring_cells = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        ring_colors = {int(g[rr, cc]) for rr, cc in ring_cells}
        assert len(ring_colors) == 1, "tower ring must be monochrome"
        color = ring_colors.pop()
        assert color not in {BG, YELLOW, GREY}, "tower ring color must be a signal color"
        assert color not in seen_colors, "each signal color must appear in exactly one tower"
        seen_colors.add(color)
        tower_cells = set(ring_cells) | {(r, c)}
        assert used_cells.isdisjoint(tower_cells), "towers must not overlap"
        used_cells.update(tower_cells)
        towers.append(
            {
                "center": (r, c),
                "color": color,
                "ring_cells": set(ring_cells),
                "tower_cells": tower_cells,
            }
        )

    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda tower: tower["center"])
    return towers


def _find_receivers(grid: np.ndarray, towers: list[dict]) -> dict[int, Coord]:
    g = np.asarray(grid, dtype=int)
    receivers: dict[int, Coord] = {}
    for tower in towers:
        color = int(tower["color"])
        ring_cells = tower["ring_cells"]
        color_cells = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        receiver_cells = [cell for cell in color_cells if cell not in ring_cells]
        assert len(receiver_cells) == 1, "each tower color must have exactly one receiver pixel"
        receivers[color] = receiver_cells[0]
    return receivers


def _find_obstacles(grid: np.ndarray) -> tuple[list[dict], dict[Coord, int]]:
    mask = np.asarray(grid, dtype=int) == GREY
    seen = np.zeros(mask.shape, dtype=bool)
    obstacles: list[dict] = []
    cell_to_obstacle: dict[Coord, int] = {}

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            coords = _collect_component(mask, (r, c), seen)
            bbox = _bbox(coords)
            r0, c0, r1, c1 = bbox
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == area, "grey obstacles must be solid rectangles"
            idx = len(obstacles)
            for cell in coords:
                cell_to_obstacle[cell] = idx
            obstacles.append({"bbox": bbox, "cells": set(coords)})

    return obstacles, cell_to_obstacle


def _direct_route_cells(center: Coord, receiver: Coord) -> tuple[list[Coord], list[Coord]]:
    r0, c0 = center
    r1, c1 = receiver

    if c1 == c0:
        h_cells: list[Coord] = []
        step = 1 if r1 > r0 else -1
        start = (r0 + 2 * step, c0)
        v_cells = _segment_cells(start, receiver, include_start=True, include_end=False)
        return h_cells, v_cells

    step = 1 if c1 > c0 else -1
    start = (r0, c0 + 2 * step)
    bend = (r0, c1)
    h_cells = _segment_cells(start, bend, include_start=True, include_end=(r0 != r1))
    v_cells = _segment_cells(bend, receiver, include_start=False, include_end=False)
    return h_cells, v_cells


def _row_priority(start_row: int, end_row: int, height: int) -> list[int]:
    rows = list(range(height))
    rows.sort(key=lambda row: (abs(row - start_row), 0 if row < start_row else 1, abs(row - end_row), row))
    return rows


def _route_pair(
    grid: np.ndarray,
    center: Coord,
    receiver: Coord,
    obstacles: list[dict],
    cell_to_obstacle: dict[Coord, int],
) -> tuple[list[Coord], str]:
    g = np.asarray(grid, dtype=int)
    r0, c0 = center
    r1, c1 = receiver

    h_cells, v_cells = _direct_route_cells(center, receiver)
    blocker = next((cell for cell in h_cells if int(g[cell]) != BG), None)
    if blocker is None:
        assert _cells_clear(g, v_cells), "direct final vertical segment must stay in background"
        return h_cells + v_cells, "direct"

    assert int(g[blocker]) == GREY, "only grey obstacles may block the horizontal segment"
    obstacle = obstacles[cell_to_obstacle[blocker]]
    r_top, c_left, r_bottom, c_right = obstacle["bbox"]
    assert c1 != c0, "vertical-only paths should not need horizontal obstacle routing"

    step = 1 if c1 > c0 else -1
    start_col = c0 + 2 * step
    turn_col = c_left - 1 if step == 1 else c_right + 1
    assert (turn_col - start_col) * step >= 0, "obstacle must leave at least one clear exit cell from the tower"

    turn_point = (r0, turn_col)
    start_point = (r0, start_col)
    first_leg = _segment_cells(start_point, turn_point, include_start=True, include_end=True)
    assert _cells_clear(g, first_leg), "pre-obstacle horizontal leg must be clear"

    best: tuple[int, int, int, int, list[Coord], str] | None = None
    for mid_row in _row_priority(r0, r1, g.shape[0]):
        if r_top <= mid_row <= r_bottom:
            continue

        detour_point = (mid_row, turn_col)
        bend_point = (mid_row, c1)
        second_leg = _segment_cells(turn_point, detour_point, include_start=False, include_end=True)
        third_leg = _segment_cells(
            detour_point,
            bend_point,
            include_start=False,
            include_end=(mid_row != r1),
        )
        fourth_leg = _segment_cells(bend_point, receiver, include_start=False, include_end=False)
        tail = second_leg + third_leg + fourth_leg
        if not _cells_clear(g, tail):
            continue

        branch = "up" if mid_row < r0 else "down"
        candidate = (
            abs(mid_row - r0),
            abs(mid_row - r1),
            0 if mid_row < r0 else 1,
            mid_row,
            first_leg + tail,
            branch,
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate

    assert best is not None, "expected a valid single-detour route around the obstacle"
    return best[4], best[5]


def _transform(grid: np.ndarray, *, return_meta: bool) -> tuple[np.ndarray, list[tuple[int, str]]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    towers = _find_towers(g)
    receivers = _find_receivers(g, towers)
    obstacles, cell_to_obstacle = _find_obstacles(g)

    counts = Counter(map(int, g.flat))
    assert counts[BG] == max(counts.values()), "black must be the dominant background"

    out = g.copy()
    meta: list[tuple[int, str]] = []
    for tower in towers:
        color = int(tower["color"])
        receiver = receivers[color]
        route_cells, branch = _route_pair(g, tower["center"], receiver, obstacles, cell_to_obstacle)
        for r, c in route_cells:
            assert int(g[r, c]) == BG, "paths may only paint background cells"
            assert int(out[r, c]) in {BG, color}, "different signal paths must not collide"
            out[r, c] = color
        meta.append((color, branch))

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each marked 3x3 signal tower to its same-colored receiver with a horizontal-first orthogonal route.

    Input structure:
    - The background is black.
    - Several signal towers appear as 3x3 solid color squares whose center cell is yellow.
    - Each tower color appears exactly once more as a single-pixel receiver.
    - Grey pixels form zero or more solid rectangular obstacles.

    Rule:
    - Pair every tower with the receiver of the same color.
    - From the tower center, trace that color toward the receiver using a horizontal-first L path, while leaving the tower body and receiver unchanged.
    - Paint only the background cells along the route; the segment inside the 3x3 tower is implicit and is not redrawn.
    - If the initial horizontal run is blocked by a grey obstacle, turn just before the obstacle, detour above or below it, continue horizontally to the receiver column, and then finish vertically at the receiver row.

    Key insight:
    - The yellow center identifies which 3x3 object is the signal source, and the path rule is not generic shortest-path routing: it is specifically a horizontal-first L path with a single obstacle detour chosen by visible geometry.
    """
    return _transform(grid, return_meta=False)


def _paint_tower(patch: np.ndarray, center: Coord, color: int) -> None:
    r, c = center
    patch[r - 1 : r + 2, c - 1 : c + 2] = color
    patch[r, c] = YELLOW


def _build_local_pair(width: int, height: int, color: int, branch: str, rng: np.random.Generator) -> np.ndarray | None:
    patch = np.full((height, width), BG, dtype=int)

    if branch == "up":
        tower_row_choices = list(range(1, height - 2))
    elif branch == "down":
        tower_row_choices = list(range(2, height - 1))
    else:
        tower_row_choices = list(range(1, height - 1))
    if not tower_row_choices:
        return None

    tower_row = int(rng.choice(tower_row_choices))
    tower_col_min = 2
    tower_col_max = width - 8
    if tower_col_min > tower_col_max:
        return None
    tower_col = int(rng.integers(tower_col_min, tower_col_max + 1))

    if rng.random() < 0.25:
        receiver_row = tower_row
    else:
        receiver_row = int(rng.integers(1, height - 1))

    obstacle_cells: list[tuple[int, int, int, int]] = []
    if branch == "direct":
        receiver_col_min = tower_col + 5
    else:
        obstacle_col0_min = tower_col + 3
        obstacle_col0_max = width - 7
        if obstacle_col0_min > obstacle_col0_max:
            return None
        obstacle_col0 = int(rng.integers(obstacle_col0_min, obstacle_col0_max + 1))
        obstacle_w_max = min(4, width - obstacle_col0 - 3)
        if obstacle_w_max < 2:
            return None
        obstacle_w = int(rng.integers(2, obstacle_w_max + 1))
        obstacle_col1 = obstacle_col0 + obstacle_w - 1

        if branch == "up":
            depth_max = min(3, height - 2 - tower_row)
            if depth_max < 1:
                return None
            depth = int(rng.integers(1, depth_max + 1))
            obstacle_row0 = tower_row
            obstacle_row1 = tower_row + depth
        else:
            depth_max = min(3, tower_row - 1)
            if depth_max < 1:
                return None
            depth = int(rng.integers(1, depth_max + 1))
            obstacle_row0 = tower_row - depth
            obstacle_row1 = tower_row

        obstacle_cells.append((obstacle_row0, obstacle_col0, obstacle_row1, obstacle_col1))
        receiver_col_min = obstacle_col1 + 2

    receiver_col_max = width - 2
    if receiver_col_min > receiver_col_max:
        return None
    receiver_col = int(rng.integers(receiver_col_min, receiver_col_max + 1))

    _paint_tower(patch, (tower_row, tower_col), color)
    patch[receiver_row, receiver_col] = color
    for r0, c0, r1, c1 in obstacle_cells:
        patch[r0 : r1 + 1, c0 : c1 + 1] = GREY

    return patch


def _choose_branches(pair_count: int, rng: np.random.Generator) -> list[str]:
    branches = list(BRANCHES)
    rng.shuffle(branches)
    return branches[:pair_count]


def _sample_layout(pair_count: int, rng: np.random.Generator) -> tuple[int, list[tuple[int, int]]] | None:
    band_heights = [int(rng.integers(*BAND_HEIGHT_RANGE)) for _ in range(pair_count)]
    base_gaps = [int(rng.integers(*BAND_GAP_RANGE)) for _ in range(pair_count - 1)]
    minimum_height = sum(band_heights) + sum(base_gaps)
    if minimum_height > 30:
        return None

    extra = int(rng.integers(0, 31 - minimum_height))
    slot_count = pair_count + 1
    slot_extras = rng.multinomial(extra, [1.0 / slot_count] * slot_count)
    top_pad = int(slot_extras[0])
    bottom_pad = int(slot_extras[-1])
    gaps = [base_gaps[i] + int(slot_extras[i + 1]) for i in range(pair_count - 1)]
    height = minimum_height + extra
    assert height <= 30
    band_specs: list[tuple[int, int]] = []
    row = top_pad
    for idx, band_h in enumerate(band_heights):
        band_specs.append((row, band_h))
        row += band_h
        if idx < len(gaps):
            row += gaps[idx]
    assert row + bottom_pad == height
    return height, band_specs


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    width = int(rng.integers(*WIDTH_RANGE))
    layout = _sample_layout(pair_count, rng)
    if layout is None:
        return None
    height, band_specs = layout

    colors = [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=pair_count, replace=False)]
    branches = _choose_branches(pair_count, rng)
    directions = [1 if rng.random() < 0.5 else -1 for _ in range(pair_count)]

    grid = np.full((height, width), BG, dtype=int)
    for (row, band_h), color, branch, direction in zip(band_specs, colors, branches, directions):
        patch = _build_local_pair(width, band_h, color, branch, rng)
        if patch is None:
            return None
        if direction < 0:
            patch = np.fliplr(patch).copy()
        band = grid[row : row + band_h, :]
        assert np.all(band == BG), "bands must not overlap"
        mask = patch != BG
        band[mask] = patch[mask]

    try:
        out, meta = _transform(grid, return_meta=True)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if Counter(branch for _color, branch in meta) != Counter(branches):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing puzzle")
