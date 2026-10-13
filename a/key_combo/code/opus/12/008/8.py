import numpy as np


GRID_SIZE = 16
HALF = GRID_SIZE // 2
BG = 0
RED = 2
GRAY = 5
MAGENTA = 6

UL, UR, BL, BR = 0, 1, 2, 3

MODE_BY_ZONE = (
    ("orig", "hflip", "vflip", "dflip"),
    ("hflip", "orig", "dflip", "vflip"),
    ("vflip", "dflip", "orig", "hflip"),
    ("dflip", "vflip", "hflip", "orig"),
)

MODE_DELTAS = {
    "orig": (-1, 1),
    "hflip": (-1, -1),
    "vflip": (1, 1),
    "dflip": (1, -1),
}

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400
FEASIBLE_ZONES = (0, 2, 3)


def _quadrant_of(r: int, c: int) -> int:
    return (2 if r >= HALF else 0) + (1 if c >= HALF else 0)


def _marker_zone(r: int, c: int) -> int:
    return (2 if r >= 4 else 0) + (1 if c >= 4 else 0)


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MAGENTA)
    assert len(coords) == 1, "expected exactly one magenta marker"
    r, c = map(int, coords[0])
    assert r < HALF and c < HALF, "marker must lie in the upper-left quadrant"
    return r, c


def _decode_modes(grid: np.ndarray) -> tuple[str, str, str, str]:
    mr, mc = _find_marker(grid)
    return MODE_BY_ZONE[_marker_zone(mr, mc)]


def _sorted_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    starts = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    assert 1 <= len(starts) <= 2, "expected 1-2 red control pixels"
    assert all(r == GRID_SIZE - 1 for r, _ in starts), "controls must be on the bottom row"
    return sorted(starts, key=lambda rc: rc[1])


def _simulate_paths(
    grid: np.ndarray, *, return_stats: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 16x16 grid"
    modes = _decode_modes(g)
    marker = _find_marker(g)
    starts = _sorted_starts(g)

    out = g.copy()
    state = g.copy()
    path_info: list[dict] = []
    total_crossings = 0
    total_turns = 0

    for sr, sc in starts:
        r, c = sr, sc
        cells = [(r, c)]
        turns = 0
        crossings = 0

        for _ in range(GRID_SIZE * GRID_SIZE):
            quad = _quadrant_of(r, c)
            dr, dc = MODE_DELTAS[modes[quad]]
            fr, fc = r + dr, c

            if 0 <= fr < GRID_SIZE and state[fr, fc] == BG:
                r, c = fr, fc
                state[r, c] = RED
                out[r, c] = RED
                cells.append((r, c))
                if _quadrant_of(r, c) != quad:
                    crossings += 1
                continue

            blocked_forward = not (0 <= fr < GRID_SIZE) or state[fr, fc] == GRAY
            if blocked_forward:
                sr2, sc2 = r, c + dc
                if 0 <= sc2 < GRID_SIZE and state[sr2, sc2] == BG:
                    r, c = sr2, sc2
                    state[r, c] = RED
                    out[r, c] = RED
                    cells.append((r, c))
                    turns += 1
                    if _quadrant_of(r, c) != quad:
                        crossings += 1
                    continue
            break

        total_crossings += crossings
        total_turns += turns
        path_info.append(
            {
                "start": (sr, sc),
                "cells": cells,
                "turns": turns,
                "crossings": crossings,
            }
        )

    out[marker] = BG
    if not return_stats:
        return out
    stats = {
        "paths": path_info,
        "total_crossings": total_crossings,
        "total_turns": total_turns,
        "added_red": int(np.count_nonzero(out == RED) - np.count_nonzero(g == RED)),
        "modes": modes,
        "marker": marker,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each bottom red path through quadrant-specific mirrored movement rules, then erase the magenta encoder.

    Input structure:
    - The grid is a 16x16 black canvas split implicitly into four 8x8 quadrants.
    - Exactly one magenta pixel lies in the upper-left quadrant and its 4x4 sub-block
      position encodes one of four flip assignments for the quadrants.
    - Each quadrant contains several gray obstacle pixels, and the bottom row contains
      1-2 red control pixels from which paths start.

    Rule:
    - Decode the magenta pixel into four quadrant modes: original, horizontal flip,
      vertical flip, or double flip.
    - In a quadrant, original means "try to move up; if blocked by gray or the edge,
      sidestep right"; horizontal flip changes the sidestep to left; vertical flip
      changes the forward direction to down; double flip changes both.
    - Starting from each bottom red control pixel, repeatedly apply the current
      quadrant's rule on the evolving grid: move into a black forward cell if possible,
      otherwise sidestep into a black lateral cell only when the forward cell is gray
      or out of bounds. Stop when neither move is available. Gray obstacles stay gray,
      and the magenta encoder is cleared to black in the output.

    Key insight:
    - The magenta pixel does not move; it selects per-quadrant motion rules, so the
      same obstacle arrangement bends paths differently after they cross quadrant borders.
    """
    return _simulate_paths(grid, return_stats=False)


def _sample_start_column(modes: tuple[str, str, str, str], half_name: str) -> int:
    if half_name == "left":
        dc = MODE_DELTAS[modes[BL]][1]
        candidates = list(range(1, 7)) if dc > 0 else list(range(2, 8))
    else:
        dc = MODE_DELTAS[modes[BR]][1]
        candidates = list(range(9, 15)) if dc < 0 else list(range(8, 14))
    return int(RNG.choice(candidates))


def _cell_ok_for_obstacle(
    r: int,
    c: int,
    marker: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> bool:
    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE):
        return False
    if (r, c) in reserved:
        return False
    mr, mc = marker
    if abs(r - mr) + abs(c - mc) == 1:
        return False
    return True


def _plan_paths(
    modes: tuple[str, str, str, str],
    marker: tuple[int, int],
    starts: list[int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], list[dict]] | None:
    controls = {(GRID_SIZE - 1, c) for c in starts}
    state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    state[marker] = MAGENTA
    for r, c in controls:
        state[r, c] = RED

    obstacle_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set(controls)
    records: list[dict] = []
    obstacle_counts = [0, 0, 0, 0]
    lower_up = MODE_DELTAS[modes[BL]][0] < 0

    for start in sorted(starts):
        r, c = GRID_SIZE - 1, start
        cells = [(r, c)]
        turns = 0
        crossings = 0
        target_turns = int(RNG.integers(1, 4)) if lower_up else 0

        for _ in range(GRID_SIZE * GRID_SIZE):
            quad = _quadrant_of(r, c)
            dr, dc = MODE_DELTAS[modes[quad]]
            fr, fc = r + dr, c
            sr, sc = r, c + dc

            can_turn = (
                target_turns > turns
                and lower_up
                and r > HALF
                and 0 <= fr < GRID_SIZE
                and 0 <= sc < GRID_SIZE
                and state[fr, fc] == BG
                and state[sr, sc] == BG
                and obstacle_counts[_quadrant_of(fr, fc)] < 5
                and _cell_ok_for_obstacle(fr, fc, marker, path_cells | obstacle_cells)
                and RNG.random() < 0.6
            )
            if can_turn:
                state[fr, fc] = GRAY
                obstacle_cells.add((fr, fc))
                obstacle_counts[_quadrant_of(fr, fc)] += 1

            if 0 <= fr < GRID_SIZE and state[fr, fc] == BG:
                r, c = fr, fc
                state[r, c] = RED
                path_cells.add((r, c))
                cells.append((r, c))
                if _quadrant_of(r, c) != quad:
                    crossings += 1
                continue

            blocked_forward = not (0 <= fr < GRID_SIZE) or state[fr, fc] == GRAY
            if blocked_forward and 0 <= sc < GRID_SIZE and state[sr, sc] == BG:
                r, c = sr, sc
                state[r, c] = RED
                path_cells.add((r, c))
                cells.append((r, c))
                turns += 1
                if _quadrant_of(r, c) != quad:
                    crossings += 1
                continue
            break

        if len(cells) <= 1:
            return None
        records.append({"cells": cells, "turns": turns, "crossings": crossings})

    return obstacle_cells, path_cells, records


def _quadrant_bounds(quad: int) -> tuple[int, int, int, int]:
    r0 = 0 if quad < 2 else HALF
    c0 = 0 if quad % 2 == 0 else HALF
    return r0, r0 + HALF, c0, c0 + HALF


def _finish_grid(
    marker: tuple[int, int],
    starts: list[int],
    obstacle_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> np.ndarray | None:
    reserved = set(path_cells) | set(obstacle_cells) | {marker}
    targets = []
    for quad in range(4):
        planned = sum(_quadrant_of(r, c) == quad for r, c in obstacle_cells)
        if planned > 5:
            return None
        targets.append(int(RNG.integers(max(2, planned), 6)))

    extras = set(obstacle_cells)
    for quad in range(4):
        r0, r1, c0, c1 = _quadrant_bounds(quad)
        candidates = [
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if _cell_ok_for_obstacle(r, c, marker, reserved | extras)
        ]
        need = targets[quad] - sum(_quadrant_of(r, c) == quad for r, c in extras)
        if need > len(candidates):
            return None
        if need:
            picks = RNG.choice(len(candidates), size=need, replace=False)
            for idx in np.atleast_1d(picks):
                extras.add(candidates[int(idx)])

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[marker] = MAGENTA
    for c in starts:
        grid[GRID_SIZE - 1, c] = RED
    for r, c in extras:
        grid[r, c] = GRAY
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    marker_cells = np.argwhere(grid == MAGENTA)
    if len(marker_cells) != 1:
        return False
    mr, mc = map(int, marker_cells[0])
    if mr >= HALF or mc >= HALF:
        return False
    starts = _sorted_starts(grid)
    for quad in range(4):
        mask = np.zeros_like(grid, dtype=bool)
        r0, r1, c0, c1 = _quadrant_bounds(quad)
        mask[r0:r1, c0:c1] = True
        count = int(np.count_nonzero((grid == GRAY) & mask))
        if not (2 <= count <= 5):
            return False
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = mr + dr, mc + dc
        if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE and grid[rr, cc] == GRAY:
            return False

    out, stats = _simulate_paths(grid, return_stats=True)
    if stats["added_red"] <= 0 or stats["total_crossings"] <= 0:
        return False
    lower_up = MODE_DELTAS[stats["modes"][BL]][0] < 0
    if lower_up and stats["total_turns"] <= 0:
        return False
    if any(len(info["cells"]) < 3 for info in stats["paths"]):
        return False
    if np.array_equal(out, grid):
        return False
    if out[mr, mc] != BG:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    # Zone 1 makes both bottom quadrants point outward while starting on the bottom
    # edge, so no path can satisfy the required "cross at least one quadrant boundary"
    # generator constraint. Sample uniformly from the feasible encodings instead.
    zone = int(RNG.choice(FEASIBLE_ZONES))
    marker_r = int(RNG.integers(0 if zone < 2 else 4, 4 if zone < 2 else 8))
    marker_c = int(RNG.integers(0 if zone % 2 == 0 else 4, 4 if zone % 2 == 0 else 8))
    marker = (marker_r, marker_c)
    modes = MODE_BY_ZONE[zone]
    lower_up = MODE_DELTAS[modes[BL]][0] < 0

    if lower_up and RNG.random() < 0.45:
        halves = ["left", "right"]
    else:
        halves = [str(RNG.choice(["left", "right"]))]
    starts = [_sample_start_column(modes, half_name) for half_name in halves]
    if len(starts) != len(set(starts)):
        return None

    planned = _plan_paths(modes, marker, starts)
    if planned is None:
        return None
    obstacle_cells, path_cells, _records = planned
    grid = _finish_grid(marker, starts, obstacle_cells, path_cells)
    if grid is None or not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
