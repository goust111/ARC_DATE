from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BACKGROUND = Color.GREEN
SEED_COLOR = Color.ORANGE
GROWTH_COLORS = (Color.PINK, Color.ORANGE)
OBSTACLE_COLOR = Color.BLUE
HIT_COLOR = Color.YELLOW

GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (18, 27)
SEED_ROW_OPTIONS = (2, 3, 4)
SEED_COUNT_RANGE = (5, 7)
EXTRA_GAPS = (3, 4)
OFF_PATH_OBSTACLE_RANGE = (1, 4)
GENERATION_RETRIES = 400


def _next_generation_cols(active_cols: list[int] | tuple[int, ...]) -> list[int]:
    cols = sorted(int(c) for c in active_cols)
    return [cols[i] + 1 for i in range(len(cols) - 1) if cols[i + 1] - cols[i] == 2]


def _simulate_growth(
    seed_row: int,
    seed_cols: list[int] | tuple[int, ...],
    obstacle_set: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict[str, object]:
    height, _width = map(int, shape)
    row = int(seed_row)
    active_cols = [int(c) for c in seed_cols]
    potential_generations: list[tuple[int, tuple[int, ...]]] = []
    actual_generations: list[tuple[int, tuple[int, ...], int]] = []
    hit_positions: list[tuple[int, int]] = []

    generation_index = 1
    while row + 2 < height:
        child_row = row + 2
        child_cols = _next_generation_cols(active_cols)
        if not child_cols:
            break

        potential_generations.append((child_row, tuple(child_cols)))
        survivors: list[int] = []
        for col in child_cols:
            pos = (child_row, int(col))
            if pos in obstacle_set:
                hit_positions.append(pos)
            else:
                survivors.append(int(col))

        if not survivors:
            break

        actual_generations.append(
            (child_row, tuple(survivors), int(GROWTH_COLORS[(generation_index - 1) % 2]))
        )
        active_cols = survivors
        row = child_row
        generation_index += 1

    potential_nodes = {
        (int(row0), int(col))
        for row0, cols in potential_generations
        for col in cols
    }
    actual_nodes = {
        (int(row0), int(col))
        for row0, cols, _color in actual_generations
        for col in cols
    }
    return {
        "potential_generations": potential_generations,
        "actual_generations": actual_generations,
        "potential_nodes": potential_nodes,
        "actual_nodes": actual_nodes,
        "hit_positions": hit_positions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an alternating-color tree downward from an orange seed row, marking and pruning branches that collide with blue obstacles.

    Input structure:
    - The background is green, with one upper orange seed row made of separated single pixels.
    - Blue singleton obstacle pixels appear only below the seed row.
    - Seed pixels on the same row may have one to three green cells between neighbors.

    Rule:
    - Treat the orange seed row as generation 0.
    - For every adjacent active pair with exactly one green gap between them on the current generation row, generate one child two rows below at the midpoint column.
    - Alternate generated colors by generation: first pink, then orange, then pink, and so on.
    - If a child would land on a blue obstacle, do not generate that child and recolor that obstacle yellow instead.
    - Only surviving children become the next active generation; untouched blue obstacles and the original seed row stay unchanged.

    Key insight:
    - The tree geometry is determined entirely by one-gap pairs inside each generation, while blue obstacles selectively remove branches by blocking specific child positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seed_rows = np.flatnonzero(np.any(g == SEED_COLOR, axis=1))
    assert seed_rows.size == 1, "expected exactly one orange seed row"
    seed_row = int(seed_rows[0])
    assert np.all(np.isin(g[seed_row], [BACKGROUND, SEED_COLOR])), "seed row must contain only background and orange"

    seed_cols = np.flatnonzero(g[seed_row] == SEED_COLOR).astype(int).tolist()
    assert len(seed_cols) >= 2, "expected multiple seed pixels"

    obstacles = {tuple(map(int, pos)) for pos in np.argwhere(g == OBSTACLE_COLOR)}
    out = g.copy()
    row = seed_row
    active_cols = seed_cols
    generation_index = 1
    height, _width = map(int, g.shape)

    while row + 2 < height:
        child_row = row + 2
        child_cols = _next_generation_cols(active_cols)
        if not child_cols:
            break

        survivors: list[int] = []
        color = int(GROWTH_COLORS[(generation_index - 1) % 2])
        for col in child_cols:
            if (child_row, int(col)) in obstacles:
                out[child_row, int(col)] = HIT_COLOR
            else:
                out[child_row, int(col)] = color
                survivors.append(int(col))

        if not survivors:
            break

        active_cols = survivors
        row = child_row
        generation_index += 1

    return out


def _build_seed_cols(width: int) -> list[int] | None:
    seed_count = random.randint(*SEED_COUNT_RANGE)
    max_block_len = min(6, seed_count)
    block_len = random.randint(4, max_block_len)
    block_start = random.randint(0, seed_count - block_len)

    gaps: list[int] = []
    for idx in range(seed_count - 1):
        if block_start <= idx < block_start + block_len - 1:
            gaps.append(2)
        else:
            gaps.append(random.choice(EXTRA_GAPS))

    span = 1 + sum(gaps)
    if span > width - 2:
        return None

    start_col = random.randint(1, width - span - 1)
    cols = [start_col]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    return cols


def _choose_primary_hit(
    potential_generations: list[tuple[int, tuple[int, ...]]],
    seed_row: int,
    seed_cols: list[int],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    baseline_nodes = _simulate_growth(seed_row, seed_cols, set(), shape)["potential_nodes"]
    candidates: list[tuple[int, int]] = []
    for row, cols in potential_generations[:2]:
        if len(cols) >= 3:
            core = cols[1:-1]
        else:
            core = cols
        for col in core:
            obstacles = {(int(row), int(col))}
            sim = _simulate_growth(seed_row, seed_cols, obstacles, shape)
            if len(sim["actual_generations"]) < 2:
                continue
            deeper_loss = [
                pos
                for pos in baseline_nodes - sim["actual_nodes"] - obstacles
                if pos[0] > row
            ]
            if deeper_loss:
                candidates.append((int(row), int(col)))
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_extra_hits(
    current_obstacles: set[tuple[int, int]],
    seed_row: int,
    seed_cols: list[int],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    result = set(current_obstacles)
    target_extra = random.randint(0, 2)
    for _ in range(target_extra):
        sim = _simulate_growth(seed_row, seed_cols, result, shape)
        if len(sim["actual_generations"]) < 2:
            break
        reachable = []
        for row, cols, _color in sim["actual_generations"]:
            for col in cols:
                reachable.append((int(row), int(col)))
        random.shuffle(reachable)
        added = False
        for pos in reachable:
            trial = set(result)
            trial.add(pos)
            trial_sim = _simulate_growth(seed_row, seed_cols, trial, shape)
            if len(trial_sim["actual_generations"]) >= 2:
                result = trial
                added = True
                break
        if not added:
            break
    return result


def _choose_off_path_obstacles(
    height: int,
    width: int,
    seed_row: int,
    seed_cols: list[int],
    blocked_positions: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    potential_nodes = _simulate_growth(seed_row, seed_cols, set(), (height, width))["potential_nodes"]
    candidates = [
        (r, c)
        for r in range(seed_row + 2, height)
        for c in range(width)
        if (r, c) not in potential_nodes and (r, c) not in blocked_positions
    ]
    if not candidates:
        return None

    random.shuffle(candidates)
    count = random.randint(*OFF_PATH_OBSTACLE_RANGE)
    chosen = set(candidates[:count])
    if not chosen:
        return None
    return chosen


def _render_input(
    height: int,
    width: int,
    seed_row: int,
    seed_cols: list[int],
    obstacle_positions: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[seed_row, seed_cols] = SEED_COLOR
    for row, col in obstacle_positions:
        grid[int(row), int(col)] = OBSTACLE_COLOR
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BACKGROUND, SEED_COLOR, OBSTACLE_COLOR}:
        return False

    seed_rows = np.flatnonzero(np.any(grid == SEED_COLOR, axis=1))
    if seed_rows.size != 1:
        return False
    seed_row = int(seed_rows[0])
    if seed_row not in SEED_ROW_OPTIONS:
        return False
    if not np.all(np.isin(grid[seed_row], [BACKGROUND, SEED_COLOR])):
        return False

    seed_cols = np.flatnonzero(grid[seed_row] == SEED_COLOR).astype(int).tolist()
    if len(seed_cols) < 5:
        return False
    gaps = [seed_cols[i + 1] - seed_cols[i] for i in range(len(seed_cols) - 1)]
    if not all(gap in (2, 3, 4) for gap in gaps):
        return False
    if sum(gap == 2 for gap in gaps) < 3:
        return False

    obstacles = {tuple(map(int, pos)) for pos in np.argwhere(grid == OBSTACLE_COLOR)}
    if not obstacles:
        return False
    if any(row <= seed_row for row, _col in obstacles):
        return False

    no_obstacle_sim = _simulate_growth(seed_row, seed_cols, set(), grid.shape)
    if len(no_obstacle_sim["potential_generations"]) < 3:
        return False

    actual_sim = _simulate_growth(seed_row, seed_cols, obstacles, grid.shape)
    if len(actual_sim["actual_generations"]) < 2:
        return False
    if not actual_sim["hit_positions"]:
        return False

    potential_nodes = set(no_obstacle_sim["potential_nodes"])
    hit_positions = set(actual_sim["hit_positions"])
    untouched = obstacles - hit_positions
    if not untouched:
        return False
    if not any(pos not in potential_nodes for pos in untouched):
        return False

    first_hit_row = min(row for row, _col in hit_positions)
    actual_nodes = set(actual_sim["actual_nodes"])
    deeper_prune = [
        pos
        for pos in potential_nodes - actual_nodes - hit_positions
        if pos[0] > first_hit_row
    ]
    if not deeper_prune:
        return False

    out = main(grid)
    expected = grid.copy()
    for row, cols, color in actual_sim["actual_generations"]:
        expected[row, list(cols)] = color
    for row, col in hit_positions:
        expected[row, col] = HIT_COLOR
    if not np.array_equal(out, expected):
        return False

    if actual_sim["actual_generations"][1][2] != Color.ORANGE:
        return False
    if not np.any(out == HIT_COLOR) or not np.any(out == OBSTACLE_COLOR) or not np.any(out == Color.PINK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    seed_row = random.choice(SEED_ROW_OPTIONS)

    seed_cols = _build_seed_cols(width)
    if seed_cols is None:
        return None

    baseline = _simulate_growth(seed_row, seed_cols, set(), (height, width))
    potential_generations = baseline["potential_generations"]
    if len(potential_generations) < 3:
        return None

    primary_hit = _choose_primary_hit(potential_generations, seed_row, seed_cols, (height, width))
    if primary_hit is None:
        return None

    hit_obstacles = _choose_extra_hits({primary_hit}, seed_row, seed_cols, (height, width))
    off_path = _choose_off_path_obstacles(height, width, seed_row, seed_cols, hit_obstacles)
    if off_path is None:
        return None

    obstacle_positions = set(hit_obstacles) | set(off_path)
    grid = _render_input(height, width, seed_row, seed_cols, obstacle_positions)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interval-growth obstacle-pruning input")
