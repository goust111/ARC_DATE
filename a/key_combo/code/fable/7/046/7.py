from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SEED_COLORS = {RED, GREY}

MIN_HEIGHT = 9
MAX_HEIGHT = 18
MIN_WIDTH = 8
MAX_WIDTH = 18
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _find_seed_pairs(grid: np.ndarray) -> list[tuple[int, int]]:
    top = np.asarray(grid[0], dtype=int)
    width = int(top.shape[0])
    bg = _background_color(grid)
    pairs: list[tuple[int, int]] = []
    used = np.zeros(width, dtype=bool)

    for col in range(width - 1):
        left = int(top[col])
        right = int(top[col + 1])
        if left not in SEED_COLORS or right not in SEED_COLORS or left == right:
            continue
        if (col > 0 and int(top[col - 1]) != bg) or (col + 2 < width and int(top[col + 2]) != bg):
            continue
        if used[col] or used[col + 1]:
            continue
        start_col = col if left == RED else col + 1
        dc = 1 if left == RED else -1
        pairs.append((start_col, dc))
        used[col] = True
        used[col + 1] = True

    non_bg_top = np.flatnonzero(top != bg)
    assert len(pairs) in (1, 2), "expected one or two bicolor seed pairs"
    assert np.all(used[non_bg_top]), "top-row foreground must be isolated red-grey seed pairs"
    return pairs


def _trace_ray(
    walls: np.ndarray,
    start_col: int,
    dc: int,
) -> dict:
    height, width = map(int, walls.shape)
    row = 1
    col = int(start_col)
    direction = int(dc)
    phase = 0

    path: list[tuple[int, int, int]] = []
    wall_candidates: list[tuple[int, int, int]] = []
    wall_bounces = 0
    boundary_bounces = 0

    while row < height:
        assert 0 <= col < width, "ray column left the grid"
        assert not bool(walls[row, col]), "ray cannot occupy a wall cell"
        color = RED if phase % 2 == 0 else GREY
        path.append((row, col, color))
        if row == height - 1:
            break

        forward = col + direction
        backward = col - direction
        if 0 <= forward < width and 0 <= backward < width and not bool(walls[row + 1, forward]) and not bool(walls[row + 1, backward]):
            wall_candidates.append((row, forward, direction))

        next_col = forward
        bounced = None
        if next_col < 0 or next_col >= width:
            bounced = "boundary"
            direction = -direction
            next_col = col + direction
        elif bool(walls[row + 1, next_col]):
            bounced = "wall"
            direction = -direction
            next_col = col + direction

        assert 0 <= next_col < width, "reflected ray still leaves the grid"
        assert not bool(walls[row + 1, next_col]), "reflected ray still hits a wall"

        if bounced == "wall":
            wall_bounces += 1
        elif bounced == "boundary":
            boundary_bounces += 1

        row += 1
        col = next_col
        phase += 1

    return {
        "path": path,
        "wall_candidates": wall_candidates,
        "wall_bounces": wall_bounces,
        "boundary_bounces": boundary_bounces,
    }


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _background_color(g)
    assert bg == ORANGE, "expected orange background"
    assert np.all(g[-1] != TEAL), "bottom row must stay empty of walls"
    assert np.all((g[1:] == bg) | (g[1:] == TEAL)), "only teal walls may appear below the top row"

    rays = _find_seed_pairs(g)
    walls = g == TEAL
    counts = np.zeros((height, width), dtype=int)
    colors = np.full((height, width), -1, dtype=int)
    traces = []

    for start_col, dc in rays:
        trace = _trace_ray(walls, start_col, dc)
        traces.append(trace)
        for row, col, color in trace["path"]:
            counts[row, col] += 1
            if colors[row, col] == -1:
                colors[row, col] = color

    out = g.copy()
    single = counts == 1
    overlap = counts >= 2
    out[single] = colors[single]
    out[overlap] = YELLOW

    stats = {
        "wall_bounces": sum(int(t["wall_bounces"]) for t in traces),
        "boundary_bounces": sum(int(t["boundary_bounces"]) for t in traces),
        "overlap_cells": int(np.count_nonzero(overlap)),
        "traces": traces,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each top-row red-grey seed pair into a red/grey diagonal ray that bounces off teal walls and side edges, turning crossings yellow.

    Input structure:
    - The background is orange.
    - The top row contains one or two isolated horizontal red-grey seed pairs.
    - The interior may contain vertical teal wall segments, and the bottom row is empty.

    Rule:
    - Each seed pair launches a ray from the red cell's column on the row below the pair.
    - If the red cell is left of grey, the ray starts down-right; if red is right of grey, it starts down-left.
    - The ray paints one cell per row, starting with red and alternating red/grey each step; when the next step would leave the grid or enter a teal wall cell, flip only the horizontal direction and continue the color phase.
    - Stop after painting the bottom row, keep the original seed pairs and teal walls unchanged, and recolor any cell visited by multiple rays to yellow.

    Key insight:
    - The adjacent red-grey order encodes the initial diagonal direction, while both wall reflections and edge reflections preserve the alternating color phase, so yellow cells reveal overlaps between otherwise independent bouncing ray paths.
    """
    out, _stats = _simulate(grid)
    return out


def _paint_seed_pair(grid: np.ndarray, left_col: int, dc: int) -> None:
    if dc == 1:
        grid[0, left_col] = RED
        grid[0, left_col + 1] = GREY
    else:
        grid[0, left_col] = GREY
        grid[0, left_col + 1] = RED


def _sample_seed_layout(height: int, width: int, pair_count: int, need_overlap: bool) -> list[tuple[int, int]] | None:
    all_lefts = list(range(width - 1))
    if pair_count == 1:
        left = random.choice(all_lefts)
        dc = random.choice((-1, 1))
        return [(left, dc)]

    candidates = [(a, b) for a in all_lefts for b in all_lefts if b >= a + 3]
    random.shuffle(candidates)
    for left_a, left_b in candidates:
        if need_overlap:
            start_a = left_a
            start_b = left_b + 1
            diff = start_b - start_a
            if diff % 2 != 0 or diff > 2 * (height - 2):
                continue
            return [(left_a, 1), (left_b, -1)]
        return [(left_a, random.choice((-1, 1))), (left_b, random.choice((-1, 1)))]
    return None


def _build_base_grid(height: int, width: int, pair_count: int, need_overlap: bool) -> np.ndarray | None:
    layout = _sample_seed_layout(height, width, pair_count, need_overlap)
    if layout is None:
        return None
    grid = np.full((height, width), ORANGE, dtype=int)
    for left_col, dc in layout:
        _paint_seed_pair(grid, left_col, dc)
    return grid


def _apply_segment(grid: np.ndarray, row0: int, row1: int, col: int) -> np.ndarray:
    trial = grid.copy()
    trial[row0 : row1 + 1, col] = TEAL
    return trial


def _insert_wall_segments(grid: np.ndarray, target_segments: int) -> tuple[np.ndarray, int]:
    placed = 0
    current = grid.copy()

    for _ in range(target_segments):
        _out, before = _simulate(current)
        options = []
        for trace in before["traces"]:
            options.extend(trace["wall_candidates"])
        random.shuffle(options)

        success = False
        for row, col, _direction in options:
            max_len = min(4, current.shape[0] - 1 - (row + 1))
            if max_len <= 0:
                continue
            lengths = list(range(1, max_len + 1))
            random.shuffle(lengths)
            for length in lengths:
                row0 = row + 1
                row1 = row0 + length - 1
                if np.any(current[row0 : row1 + 1, col] != ORANGE):
                    continue
                trial = _apply_segment(current, row0, row1, col)
                try:
                    _trial_out, after = _simulate(trial)
                except AssertionError:
                    continue
                if after["wall_bounces"] > before["wall_bounces"]:
                    current = trial
                    placed += 1
                    success = True
                    break
            if success:
                break
        if not success:
            break

    return current, placed


def _add_passive_walls(grid: np.ndarray, limit: int) -> np.ndarray:
    current = grid.copy()
    height, width = map(int, current.shape)
    for _ in range(limit):
        col = random.randrange(1, width - 1)
        row0 = random.randrange(2, height - 2)
        row1 = min(height - 2, row0 + random.randint(0, 2))
        if np.any(current[row0 : row1 + 1, col] != ORANGE):
            continue
        trial = _apply_segment(current, row0, row1, col)
        try:
            _simulate(trial)
        except AssertionError:
            continue
        current = trial
    return current


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    pair_count = 2 if random.random() < 0.6 else 1
    need_overlap = pair_count == 2 and random.random() < 0.45
    need_wall_bounce = random.random() < 0.8
    need_boundary_bounce = random.random() < 0.35

    grid = _build_base_grid(height, width, pair_count, need_overlap)
    if grid is None:
        return None

    target_segments = random.randint(1 if need_wall_bounce else 0, 3 if pair_count == 1 else 4)
    grid, placed = _insert_wall_segments(grid, target_segments)
    if need_wall_bounce and placed == 0:
        return None

    if random.random() < 0.4:
        grid = _add_passive_walls(grid, random.randint(0, 2))

    try:
        _out, stats = _simulate(grid)
    except AssertionError:
        return None

    if need_overlap and stats["overlap_cells"] == 0:
        return None
    if need_wall_bounce and stats["wall_bounces"] == 0:
        return None
    if need_boundary_bounce and stats["boundary_bounces"] == 0:
        return None
    if pair_count == 2 and stats["overlap_cells"] > height // 2:
        return None
    if stats["wall_bounces"] + stats["boundary_bounces"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-ray ARC input")
