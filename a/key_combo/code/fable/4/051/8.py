from __future__ import annotations

import random

import numpy as np


GREEN = 3
RED = 2
GREY = 5
PINK = 6
ORANGE = 7

GENERATION_RETRIES = 400


def _next_wave(active: np.ndarray) -> np.ndarray:
    width = int(active.shape[0])
    nxt = np.zeros(width, dtype=bool)
    cols = np.flatnonzero(active)
    for col in cols:
        col = int(col)
        if col > 0:
            nxt[col - 1] = ~nxt[col - 1]
        if col + 1 < width:
            nxt[col + 1] = ~nxt[col + 1]
    return nxt


def _simulate_growth(
    seed_row: int,
    seed_cols: list[int],
    height: int,
    width: int,
    obstacles: set[tuple[int, int]],
) -> tuple[dict[int, set[int]], set[tuple[int, int]]]:
    active = np.zeros(width, dtype=bool)
    active[np.asarray(seed_cols, dtype=int)] = True

    grown: dict[int, set[int]] = {}
    hits: set[tuple[int, int]] = set()
    for row in range(seed_row + 1, height):
        candidate = _next_wave(active)
        if not candidate.any():
            break

        next_active = np.zeros(width, dtype=bool)
        for col in np.flatnonzero(candidate):
            col = int(col)
            pos = (row, col)
            if pos in obstacles:
                hits.add(pos)
            else:
                grown.setdefault(row, set()).add(col)
                next_active[col] = True

        active = next_active
        if not active.any():
            break

    return grown, hits


def _has_pair_at_distance_two(cols: list[int]) -> bool:
    return any(b - a == 2 for a, b in zip(cols, cols[1:]))


def _sample_seed_cols(width: int, height: int, seed_row: int) -> list[int] | None:
    max_count = min(6, width // 2)
    for _ in range(200):
        count = random.randint(2, max_count)
        cols = sorted(random.sample(range(1, width - 1), count))
        if not _has_pair_at_distance_two(cols):
            continue

        grown, _hits = _simulate_growth(seed_row, cols, height, width, set())
        if seed_row + 1 not in grown or seed_row + 2 not in grown:
            continue
        if len(grown) < 4:
            continue
        if sum(len(row_cols) for row_cols in grown.values()) < 10:
            continue
        return cols
    return None


def _far_enough(pos: tuple[int, int], others: set[tuple[int, int]], min_l1: int = 2) -> bool:
    row, col = pos
    return all(abs(row - rr) + abs(col - cc) >= min_l1 for rr, cc in others)


def _choose_hit_obstacles(
    seed_row: int,
    seed_cols: list[int],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    obstacles: set[tuple[int, int]] = set()
    used_rows: set[int] = set()
    desired = 0 if random.random() < 0.2 else random.randint(1, 3)
    for _ in range(desired):
        grown, _hits = _simulate_growth(seed_row, seed_cols, height, width, obstacles)
        candidate_rows = [row for row, cols in grown.items() if len(cols) >= 2 and row not in used_rows]
        if not candidate_rows:
            break

        weighted_rows = []
        for row in candidate_rows:
            weight = 2 if row > seed_row + 1 else 1
            weighted_rows.extend([row] * weight)
        row = random.choice(weighted_rows)

        row_candidates = [(row, col) for col in sorted(grown[row]) if _far_enough((row, col), obstacles)]
        if not row_candidates:
            row_candidates = [(row, col) for col in sorted(grown[row])]
        if not row_candidates:
            continue

        choice = random.choice(row_candidates)
        obstacles.add(choice)
        used_rows.add(row)

    return obstacles


def _choose_decoys(
    seed_row: int,
    height: int,
    width: int,
    touched: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    candidates = [
        (row, col)
        for row in range(seed_row + 1, height)
        for col in range(width)
        if (row, col) not in touched and (row, col) not in occupied
    ]
    random.shuffle(candidates)

    desired = random.randint(1, 4)
    decoys: set[tuple[int, int]] = set()
    for pos in candidates:
        if _far_enough(pos, decoys | occupied, min_l1=2):
            decoys.add(pos)
        if len(decoys) >= desired:
            break

    if decoys:
        return decoys

    for pos in candidates:
        decoys.add(pos)
        break
    return decoys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the orange seed row downward by diagonal XOR spread, alternating pink/orange, while hit grey obstacles turn red.

    Input structure:
    - The background is green.
    - Exactly one row contains one or more orange seed pixels.
    - Grey obstacle pixels may appear only below the seed row; all other cells are background.

    Rule:
    - Start from the seed row and propagate one row downward at a time.
    - A cell in the next row is activated iff exactly one of its two parents, upper-left and upper-right, is active in the current row; simultaneous arrival cancels out.
    - If an activated cell is green background, place a pixel there, using pink on the first generated row below the seeds, orange on the next, then alternating by row.
    - If an activated cell is a grey obstacle, do not place a growth pixel there; recolor that obstacle red and stop propagation through that position.
    - Stop when the wave reaches the bottom row or no cells remain active. Keep the original seed row and every untouched obstacle unchanged.

    Key insight:
    - The pattern is a downward Sierpinski-like interference wave: each row is the XOR of the two diagonal parent contributions, and obstacles only react when the wavefront actually lands on them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    height, width = map(int, g.shape)
    seed_rows = np.flatnonzero(np.any(g == ORANGE, axis=1))
    assert len(seed_rows) == 1, "expected exactly one orange seed row"
    seed_row = int(seed_rows[0])

    active = np.zeros(width, dtype=bool)
    active[np.flatnonzero(g[seed_row] == ORANGE)] = True
    assert active.any(), "expected at least one seed pixel"

    out = g.copy()
    for row in range(seed_row + 1, height):
        candidate = _next_wave(active)
        if not candidate.any():
            break

        next_active = np.zeros(width, dtype=bool)
        color = PINK if (row - seed_row) % 2 == 1 else ORANGE
        for col in np.flatnonzero(candidate):
            col = int(col)
            if g[row, col] == GREY:
                out[row, col] = RED
                continue
            out[row, col] = color
            next_active[col] = True

        active = next_active
        if not active.any():
            break

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        height = random.randint(12, 24)
        width = random.randint(10, 24)
        seed_row = random.randint(0, 3)

        seed_cols = _sample_seed_cols(width, height, seed_row)
        if seed_cols is None:
            continue

        hit_obstacles = _choose_hit_obstacles(seed_row, seed_cols, height, width)
        grown, hits = _simulate_growth(seed_row, seed_cols, height, width, hit_obstacles)
        touched = hits | {(row, col) for row, cols in grown.items() for col in cols}

        decoys = _choose_decoys(seed_row, height, width, touched, hit_obstacles)
        if not decoys:
            continue

        grid = np.full((height, width), GREEN, dtype=int)
        grid[seed_row, np.asarray(seed_cols, dtype=int)] = ORANGE
        for row, col in hit_obstacles | decoys:
            grid[row, col] = GREY

        out = main(grid)

        grown_pixels = int(np.count_nonzero(np.isin(out[seed_row + 1 :], [PINK, ORANGE])))
        if grown_pixels < 6:
            continue
        if not np.any(out[seed_row + 1 :] == PINK):
            continue
        if seed_row + 2 >= height or not np.any(out[seed_row + 2 :] == ORANGE):
            continue
        if np.count_nonzero(out == GREY) < 1:
            continue
        if np.count_nonzero(out == RED) != len(hits):
            continue
        if len(hits) == 0 and len(hit_obstacles) > 0:
            continue
        if np.count_nonzero(out != grid) < 6:
            continue

        return grid

    raise RuntimeError("failed to generate a valid interference-growth input")
