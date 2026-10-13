from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
AZURE = 8
MAROON = 9

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_AXIS_LEN = 12
MAX_AXIS_LEN = 30
MIN_LANES = 2
MAX_LANES = 6
MAX_WALLS_PER_LANE = 5
MIN_SEGMENT_LEN = 2
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _infer_source_direction(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    h, w = grid.shape
    preferred: list[tuple[int, int]] = []
    if r == 0:
        preferred.append((1, 0))
    if r == h - 1:
        preferred.append((-1, 0))
    if c == 0:
        preferred.append((0, 1))
    if c == w - 1:
        preferred.append((0, -1))

    for dr, dc in preferred:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) in {BLACK, AZURE, GREEN, MAROON}:
            return dr, dc

    candidates: list[tuple[int, int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if not _in_bounds(grid.shape, rr, cc):
            continue
        if int(grid[rr, cc]) not in {BLACK, AZURE, GREEN, MAROON}:
            continue

        score = 0
        br, bc = r - dr, c - dc
        if not _in_bounds(grid.shape, br, bc) or int(grid[br, bc]) == GREY:
            score += 2

        if dr != 0:
            for oc in (-1, 1):
                nr, nc = r, c + oc
                if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == GREY:
                    score += 1
        else:
            for or_ in (-1, 1):
                nr, nc = r + or_, c
                if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == GREY:
                    score += 1
        candidates.append((score, dr, dc))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][1], candidates[0][2]


def _trace_signal(out: np.ndarray, start_r: int, start_c: int, dr: int, dc: int) -> None:
    r, c = start_r, start_c
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(out.shape, nr, nc):
            return

        value = int(out[nr, nc])
        if value in {BLACK, AZURE}:
            out[nr, nc] = AZURE
            r, c = nr, nc
            continue
        if value == GREEN:
            r, c = nr, nc
            continue
        if value == MAROON:
            out[nr, nc] = RED
            return
        return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each azure signal straight through one-cell-wide channels, crossing green gates and stopping at maroon gates.

    Input structure:
    - Background is black.
    - Grey single-pixel-thick horizontal or vertical wall segments split the grid
      into rectangular channels whose usable width is exactly one cell.
    - Each separating wall contains one green or maroon gate pixel, and each
      active channel has an azure source pixel at the end nearest a grid edge.

    Rule:
    - Start a signal from every azure source and infer its unique inward channel direction.
    - Move straight along that channel, recoloring every reached black channel cell to azure.
    - Green gate pixels stay green but let the signal continue into the next channel.
    - When the signal reaches a maroon gate, recolor that gate red and stop that signal.
    - Unreached black channel cells remain black, while grey walls stay unchanged.

    Key insight:
    - The task is a straight-line cascade: sources do not flood arbitrarily, they
      advance only along their narrow channel, and the gate color alone decides
      whether the cascade continues or terminates with a red warning mark.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r, c in np.argwhere(g == AZURE):
        direction = _infer_source_direction(g, int(r), int(c))
        if direction is None:
            continue
        _trace_signal(out, int(r), int(c), direction[0], direction[1])
    return out


def _split_total(total: int, parts: int, min_part: int) -> list[int]:
    assert parts >= 1
    assert total >= parts * min_part
    if parts == 1:
        return [total]

    extra = total - parts * min_part
    bars = sorted(random.sample(range(extra + parts - 1), parts - 1))
    values: list[int] = []
    prev = -1
    for bar in bars + [extra + parts - 1]:
        values.append(min_part + bar - prev - 1)
        prev = bar
    return values


def _build_lane_pattern(axis_len: int, source_at_start: bool) -> tuple[np.ndarray, list[int]]:
    max_walls = min(MAX_WALLS_PER_LANE, (axis_len - MIN_SEGMENT_LEN) // (MIN_SEGMENT_LEN + 1))
    wall_count = random.randint(1, max_walls)
    segment_lengths = _split_total(
        axis_len - wall_count,
        wall_count + 1,
        min_part=MIN_SEGMENT_LEN,
    )

    pattern = np.full(axis_len, BLACK, dtype=int)
    pattern[0] = AZURE

    gate_positions: list[int] = []
    cursor = segment_lengths[0]
    for seg_len in segment_lengths[1:]:
        gate_positions.append(cursor)
        cursor += 1 + seg_len

    if not source_at_start:
        pattern = pattern[::-1].copy()
        gate_positions = [axis_len - 1 - pos for pos in gate_positions]

    return pattern, gate_positions


def _sample_gate_colors(count: int) -> list[int]:
    if count == 0:
        return []
    if count == 1:
        return [random.choice((GREEN, MAROON))]

    while True:
        colors = [GREEN if random.random() < 0.5 else MAROON for _ in range(count)]
        if GREEN in colors and MAROON in colors:
            return colors


def _generate_horizontal() -> np.ndarray:
    lane_count = random.randint(MIN_LANES, MAX_LANES)
    h = 2 * lane_count + 1
    w = random.randint(MIN_AXIS_LEN, MAX_AXIS_LEN)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[::2, :] = GREY

    gate_coords: list[tuple[int, int]] = []
    for lane_idx in range(lane_count):
        row = 1 + 2 * lane_idx
        source_left = random.random() < 0.5
        pattern, gate_positions = _build_lane_pattern(w, source_at_start=source_left)
        grid[row, :] = pattern
        for col in gate_positions:
            gate_coords.append((row, col))

    for (row, col), color in zip(gate_coords, _sample_gate_colors(len(gate_coords))):
        grid[row, col] = color
    return grid


def _generate_vertical() -> np.ndarray:
    lane_count = random.randint(MIN_LANES, MAX_LANES)
    h = random.randint(MIN_AXIS_LEN, MAX_AXIS_LEN)
    w = 2 * lane_count + 1

    grid = np.full((h, w), BLACK, dtype=int)
    grid[:, ::2] = GREY

    gate_coords: list[tuple[int, int]] = []
    for lane_idx in range(lane_count):
        col = 1 + 2 * lane_idx
        source_top = random.random() < 0.5
        pattern, gate_positions = _build_lane_pattern(h, source_at_start=source_top)
        grid[:, col] = pattern
        for row in gate_positions:
            gate_coords.append((row, col))

    for (row, col), color in zip(gate_coords, _sample_gate_colors(len(gate_coords))):
        grid[row, col] = color
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    gates = int(np.count_nonzero((grid == GREEN) | (grid == MAROON)))
    if gates < 2:
        return False
    if not (np.any(grid == GREEN) and np.any(grid == MAROON)):
        return False

    sources = np.argwhere(grid == AZURE)
    if len(sources) < 2:
        return False
    for r, c in sources:
        if _infer_source_direction(grid, int(r), int(c)) is None:
            return False

    out = main(grid)
    return bool(np.any(out != grid))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_horizontal() if random.random() < 0.5 else _generate_vertical()
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid signal-gate puzzle")
