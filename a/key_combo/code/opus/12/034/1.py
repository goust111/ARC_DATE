from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
FILL_COLORS = (1, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
OBSTACLE_COUNT_RANGE = (3, 5)
GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_marker_top_left(grid: np.ndarray) -> tuple[int, int]:
    mask = grid == BLACK
    assert int(np.count_nonzero(mask)) == 4, "expected exactly one 2x2 black marker"
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 == r0 + 1 and c1 == c0 + 1, "black marker must be a 2x2 block"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "black marker must be solid"
    return r0, c0


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.shape == (1, 2), "expected exactly one red seed pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _cross_mask(shape: tuple[int, int], marker_top_left: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0 = marker_top_left
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + 2, :] = True
    mask[:, c0 : c0 + 2] = True
    return mask


def _spiral_positions(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = seed
    h, w = shape
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg_length = 2 * (max_delta + 1)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction_index = 0
    leg_length = 2
    positions: list[tuple[int, int]] = []

    while leg_length <= max_leg_length:
        dr, dc = directions[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            if _in_bounds(shape, r, c):
                positions.append((r, c))

        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2

    return positions


def _trace_spiral(
    grid: np.ndarray, seed: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    painted: list[tuple[int, int]] = []
    stop_cell: tuple[int, int] | None = None
    for r, c in _spiral_positions(seed, grid.shape):
        if int(grid[r, c]) == GREY:
            stop_cell = (r, c)
            break
        painted.append((r, c))
    return painted, stop_cell


def _apply_cross_erase(grid: np.ndarray, cross: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[cross & (out != RED)] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a blocker-limited spiral that protects red cells from a marker-defined cross erasure.

    Input structure:
    - A fully colored grid contains exactly one solid 2x2 black marker block and
      exactly one red seed pixel.
    - Several grey single-pixel obstacles are scattered elsewhere.
    - The seed lies outside the two marker rows and the two marker columns.

    Rule:
    - First locate the black marker block and remember its two rows and two columns.
    - Starting from the red seed, follow a square spiral with directions up, right,
      down, left and leg lengths 2, 2, 4, 4, 6, 6, ... .
    - Recolor every in-bounds non-grey spiral cell red; out-of-bounds steps still
      count, and the spiral stops before painting the first in-bounds grey blocker.
    - Then recolor every non-red cell in the marker-defined cross to black, while
      leaving red spiral cells inside that cross untouched.

    Key insight:
    - The spiral is not the final object by itself: it creates a protection mask,
      and only the portion that reaches the marker cross survives the later erasure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_top_left = _find_marker_top_left(g)
    seed = _find_seed(g)
    cross = _cross_mask(g.shape, marker_top_left)
    assert not cross[seed], "seed must lie outside the marker cross"

    out = g.copy()
    painted, _ = _trace_spiral(g, seed)
    for r, c in painted:
        out[r, c] = RED

    out[cross & (out != RED)] = BLACK
    return out


def _cross_runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    runs = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _seed_candidates(
    shape: tuple[int, int], marker_top_left: tuple[int, int]
) -> list[dict[str, object]]:
    cross = _cross_mask(shape, marker_top_left)
    h, w = shape
    candidates: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if cross[r, c]:
                continue
            path = _spiral_positions((r, c), shape)
            if len(path) < 10:
                continue

            cross_indices = [i for i, pos in enumerate(path) if cross[pos]]
            if len(cross_indices) < 2:
                continue

            runs = [run for run in _cross_runs(cross_indices) if len(run) >= 2]
            if not runs:
                continue

            first_cross = cross_indices[0]
            last_cross = cross_indices[-1]
            if first_cross < 2:
                continue

            inside_choices: list[int] = []
            for run in runs:
                inside_choices.extend(run[1:])
            after_choices = list(range(last_cross + 1, len(path)))
            if not inside_choices and not after_choices:
                continue

            candidates.append(
                {
                    "seed": (r, c),
                    "path": path,
                    "cross_indices": cross_indices,
                    "inside_choices": inside_choices,
                    "after_choices": after_choices,
                }
            )

    return candidates


def _pick_seed_and_stop(
    shape: tuple[int, int], marker_top_left: tuple[int, int]
) -> tuple[tuple[int, int], list[tuple[int, int]], int, str] | None:
    candidates = _seed_candidates(shape, marker_top_left)
    if not candidates:
        return None

    branch_pool = []
    if any(c["inside_choices"] for c in candidates):
        branch_pool.append("inside")
    if any(c["after_choices"] for c in candidates):
        branch_pool.append("after")
    if not branch_pool:
        return None

    branch = random.choice(branch_pool)
    usable = [c for c in candidates if c[f"{branch}_choices"]]
    if not usable:
        return None

    candidate = random.choice(usable)
    stop_choices = list(candidate[f"{branch}_choices"])
    if branch == "after":
        stop_choices = stop_choices[: min(len(stop_choices), 8)]
    stop_index = random.choice(stop_choices)
    return candidate["seed"], candidate["path"], int(stop_index), branch


def _scatter_extra_obstacles(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    seed: tuple[int, int],
    initial_obstacles: list[tuple[int, int]],
    total_count: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    placed = initial_obstacles[:]
    occupied = set(forbidden) | set(initial_obstacles)

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in occupied and abs(r - seed[0]) + abs(c - seed[1]) >= 2
    ]
    random.shuffle(candidates)

    for cell in candidates:
        if len(placed) >= total_count:
            break
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) < 2 for rr, cc in placed):
            continue
        placed.append(cell)

    if len(placed) != total_count:
        return None
    return placed


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    branch: str,
    chosen_stop: tuple[int, int],
) -> bool:
    try:
        marker_top_left = _find_marker_top_left(inp)
        seed = _find_seed(inp)
    except AssertionError:
        return False

    if int(np.count_nonzero(inp == GREY)) < OBSTACLE_COUNT_RANGE[0]:
        return False

    cross = _cross_mask(inp.shape, marker_top_left)
    if cross[seed]:
        return False

    painted, stop_cell = _trace_spiral(inp, seed)
    if stop_cell != chosen_stop:
        return False
    if stop_cell is None:
        return False
    if branch == "inside" and not cross[stop_cell]:
        return False
    if branch == "after" and cross[stop_cell]:
        return False

    painted_cross = [(r, c) for r, c in painted if cross[r, c]]
    if len(painted_cross) < 2:
        return False

    if not np.any(cross & (out == RED) & (inp != RED)):
        return False
    if not np.any(cross & (out == BLACK) & (inp != BLACK)):
        return False
    if not np.any((out == RED) & ~cross):
        return False

    spiral_only = inp.copy()
    for r, c in painted:
        spiral_only[r, c] = RED
    erase_only = _apply_cross_erase(inp, cross)

    if np.array_equal(out, inp):
        return False
    if np.array_equal(out, spiral_only):
        return False
    if np.array_equal(out, erase_only):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.array(random.choices(FILL_COLORS, k=h * w), dtype=int).reshape(h, w)

    marker_r = random.randint(1, h - 3)
    marker_c = random.randint(1, w - 3)
    marker_top_left = (marker_r, marker_c)
    pick = _pick_seed_and_stop((h, w), marker_top_left)
    if pick is None:
        return None

    seed, path, stop_index, branch = pick
    stop_cell = path[stop_index]

    marker_cells = {(marker_r + dr, marker_c + dc) for dr in range(2) for dc in range(2)}
    forbidden_for_obstacles = marker_cells | {seed} | set(path[:stop_index])
    obstacle_count = random.randint(*OBSTACLE_COUNT_RANGE)
    obstacles = _scatter_extra_obstacles(
        (h, w),
        forbidden_for_obstacles,
        seed,
        [stop_cell],
        obstacle_count,
    )
    if obstacles is None:
        return None

    grid[marker_r : marker_r + 2, marker_c : marker_c + 2] = BLACK
    grid[seed] = RED
    for cell in obstacles:
        grid[cell] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _semantic_checks(grid, out, branch, stop_cell):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-shield input")
