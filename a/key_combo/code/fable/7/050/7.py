from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
BLOCKER_COLORS = (1, 5, 6, 7, 8, 9)

GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _gcd_list(values: list[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, int(value))
    return g


def _line_value(grid: np.ndarray, orientation: str, fixed: int, pos: int) -> int:
    if orientation == "h":
        return int(grid[fixed, pos])
    return int(grid[pos, fixed])


def _set_line_value(grid: np.ndarray, orientation: str, fixed: int, pos: int, color: int) -> None:
    if orientation == "h":
        grid[fixed, pos] = int(color)
    else:
        grid[pos, fixed] = int(color)


def _track_spec(grid: np.ndarray) -> tuple[int, str, int, list[int], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    yellow_cells = np.argwhere(g == YELLOW)
    assert len(yellow_cells) == 1, "expected a unique yellow target"
    yr, yc = map(int, yellow_cells[0])

    row_mask = g[yr] != bg
    row_mask[yc] = False
    col_mask = g[:, yc] != bg
    col_mask[yr] = False
    row_count = int(row_mask.sum())
    col_count = int(col_mask.sum())

    if row_count >= 2 and col_count < 2:
        orientation = "h"
        fixed = yr
        raw_positions = [int(c) for c in np.where(row_mask)[0]]
        target_coord = yc
    elif col_count >= 2 and row_count < 2:
        orientation = "v"
        fixed = yc
        raw_positions = [int(r) for r in np.where(col_mask)[0]]
        target_coord = yr
    else:
        raise AssertionError("could not determine track orientation")

    raw_positions.sort()
    assert len(raw_positions) >= 2, "track needs at least two visible slots"
    diffs = [b - a for a, b in zip(raw_positions, raw_positions[1:])]
    step = _gcd_list(diffs)
    assert step > 0, "track spacing must be positive"
    assert all(diff % step == 0 for diff in diffs), "track spacing must be regular"

    if target_coord < raw_positions[0]:
        ordered_positions = raw_positions
    elif target_coord > raw_positions[-1]:
        ordered_positions = raw_positions[::-1]
    else:
        raise AssertionError("yellow target must sit outside one end of the track")

    normalized = sorted(raw_positions)
    expected = [normalized[0] + i * step for i in range(len(normalized))]
    assert normalized == expected, "track slots must form a complete arithmetic progression"

    slider_positions = [pos for pos in raw_positions if _line_value(g, orientation, fixed, pos) == RED]
    assert len(slider_positions) == 1, "expected a unique red slider on the track"
    slider_pos = int(slider_positions[0])
    slider_index = ordered_positions.index(slider_pos)

    fuel_count = int(np.count_nonzero(g == RED)) - 1
    assert fuel_count >= 1, "expected at least one red fuel pixel"
    return bg, orientation, fixed, ordered_positions, slider_index, fuel_count


def _advance_destination(
    grid: np.ndarray,
    orientation: str,
    fixed: int,
    ordered_positions: list[int],
    slider_index: int,
    fuel_count: int,
) -> int:
    steps = 0
    for idx in range(slider_index - 1, -1, -1):
        pos = ordered_positions[idx]
        if _line_value(grid, orientation, fixed, pos) == GREEN:
            steps += 1
            if steps == fuel_count:
                return pos
    raise AssertionError("not enough empty slots toward the target")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the red slider along the dotted track toward the yellow target by spending one fuel pixel per empty slot.

    Input structure:
    - A black background contains a straight horizontal or vertical track made of evenly spaced green dots.
    - A single red slider occupies one track slot, some other track slots may be occupied by non-red non-green colors, and a single yellow target sits just outside one end of the track.
    - Additional isolated red pixels elsewhere in the grid act as fuel and are not on the track line.

    Rule:
    - Count the isolated red fuel pixels to get k.
    - Move the red slider toward the yellow target, counting only green empty slots; occupied slots are skipped and do not consume fuel.
    - Remove all fuel pixels, restore the slider's source slot to a green track dot, and place the slider on the k-th reachable empty slot toward the target while leaving the target and other occupied slots unchanged.

    Key insight:
    - The off-track red pixels are consumable step counters, while the non-green occupied slots affect only where the slider lands because they are jumped over without spending steps.
    """
    g = np.asarray(grid, dtype=int)
    bg, orientation, fixed, ordered_positions, slider_index, fuel_count = _track_spec(g)
    slider_pos = ordered_positions[slider_index]
    destination = _advance_destination(g, orientation, fixed, ordered_positions, slider_index, fuel_count)

    out = g.copy()
    red_cells = np.argwhere(out == RED)
    for r, c in red_cells:
        rr, cc = int(r), int(c)
        if orientation == "h" and rr == fixed and cc == slider_pos:
            continue
        if orientation == "v" and cc == fixed and rr == slider_pos:
            continue
        out[rr, cc] = bg

    _set_line_value(out, orientation, fixed, slider_pos, GREEN)
    _set_line_value(out, orientation, fixed, destination, RED)
    return out


def _sample_roles(slot_count: int) -> tuple[int, set[int], int]:
    for _ in range(200):
        slider_index = random.randint(2, slot_count - 2)
        need_skip_branch = random.random() < 0.7

        ahead_candidates = list(range(slider_index))
        max_ahead = min(2, len(ahead_candidates))
        min_ahead = 1 if need_skip_branch and max_ahead >= 1 else 0
        ahead_count = random.randint(min_ahead, max_ahead)
        occupied_ahead = set(random.sample(ahead_candidates, ahead_count))

        behind_candidates = list(range(slider_index + 1, slot_count))
        behind_count = random.randint(0, min(2, len(behind_candidates)))
        occupied_behind = set(random.sample(behind_candidates, behind_count))

        occupied = occupied_ahead | occupied_behind
        available_steps = [idx for idx in range(slider_index) if idx not in occupied]
        if not available_steps:
            continue

        fuel_count = random.randint(1, min(4, len(available_steps)))
        return slider_index, occupied, fuel_count
    raise RuntimeError("failed to sample a valid slider/occupancy layout")


def _pick_dims(orientation: str, slot_count: int, spacing: int) -> tuple[int, int]:
    line_span = 1 + (slot_count - 1) * spacing
    if orientation == "h":
        height = random.randint(10, 18)
        min_width = line_span + 5
        width = random.randint(min_width, min(30, min_width + 8))
    else:
        width = random.randint(10, 18)
        min_height = line_span + 5
        height = random.randint(min_height, min(30, min_height + 8))
    return height, width


def _place_track(
    grid: np.ndarray,
    orientation: str,
    ordered_positions: list[int],
    fixed: int,
    slider_index: int,
    occupied: set[int],
) -> None:
    for idx, pos in enumerate(ordered_positions):
        color = GREEN
        if idx == slider_index:
            color = RED
        elif idx in occupied:
            color = random.choice(BLOCKER_COLORS)
        _set_line_value(grid, orientation, fixed, pos, color)


def _fuel_candidates(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    orientation: str,
    fixed: int,
    yellow_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    yr, yc = yellow_rc
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in blocked:
                continue
            if r == yr or c == yc:
                continue
            if orientation == "h" and r == fixed:
                continue
            if orientation == "v" and c == fixed:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_fuels(
    grid: np.ndarray,
    fuel_count: int,
    blocked: set[tuple[int, int]],
    orientation: str,
    fixed: int,
    yellow_rc: tuple[int, int],
) -> None:
    taken = set(blocked)
    placed = 0
    for r, c in _fuel_candidates(grid.shape, blocked, orientation, fixed, yellow_rc):
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in taken):
            continue
        grid[r, c] = RED
        taken.add((r, c))
        placed += 1
        if placed == fuel_count:
            return
    raise RuntimeError("failed to place isolated fuel pixels")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        orientation = random.choice(("h", "v"))
        slot_count = random.randint(6, 9)
        spacing = random.choice((2, 3))
        height, width = _pick_dims(orientation, slot_count, spacing)

        grid = np.full((height, width), BLACK, dtype=int)
        slider_index, occupied, fuel_count = _sample_roles(slot_count)
        target_near_low_end = random.choice((True, False))

        if orientation == "h":
            row = random.randint(2, height - 3)
            if target_near_low_end:
                start = random.randint(2, width - 2 - (slot_count - 1) * spacing)
                ordered_positions = [start + i * spacing for i in range(slot_count)]
                yellow_rc = (row, start - 1)
            else:
                start = random.randint(1 + (slot_count - 1) * spacing, width - 3)
                ordered_positions = [start - i * spacing for i in range(slot_count)]
                yellow_rc = (row, start + 1)
            fixed = row
        else:
            col = random.randint(2, width - 3)
            if target_near_low_end:
                start = random.randint(2, height - 2 - (slot_count - 1) * spacing)
                ordered_positions = [start + i * spacing for i in range(slot_count)]
                yellow_rc = (start - 1, col)
            else:
                start = random.randint(1 + (slot_count - 1) * spacing, height - 3)
                ordered_positions = [start - i * spacing for i in range(slot_count)]
                yellow_rc = (start + 1, col)
            fixed = col

        _place_track(grid, orientation, ordered_positions, fixed, slider_index, occupied)
        grid[yellow_rc] = YELLOW

        blocked = set(map(tuple, np.argwhere(grid != BLACK)))
        try:
            _place_fuels(grid, fuel_count, blocked, orientation, fixed, yellow_rc)
            out = main(grid)
        except Exception:
            continue

        if grid.shape != out.shape:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid fuel-track input")
