from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

N_ZONES = 4
ZONE_WIDTH = 5
SEP_WIDTH = 1
GRID_WIDTH = N_ZONES * ZONE_WIDTH + (N_ZONES - 1) * SEP_WIDTH
HEIGHT_CHOICES = (9, 11, 13, 15)
MAX_GENERATION_ATTEMPTS = 400
DIRECTIONS = ("U", "D", "L", "R")
SURVIVOR_SETS = [
    (0, 1, 2, 3),
    (0, 1, 2, 4),
    (0, 1, 3, 4),
    (0, 2, 3, 4),
    (1, 2, 3, 4),
]
SURVIVOR_SET_WEIGHTS = (3, 3, 2, 2, 1)


def _patterns_for_survivors(target: int) -> list[tuple[int, int, int, int]]:
    patterns: list[tuple[int, int, int, int]] = []
    for u in range(3):
        for d in range(3):
            for l in range(3):
                for r in range(3):
                    counts = (u, d, l, r)
                    if sum(v == 1 for v in counts) != target:
                        continue
                    total = sum(counts)
                    if 3 <= total <= 7:
                        patterns.append(counts)
    return patterns


PATTERNS_BY_SURVIVORS = {k: _patterns_for_survivors(k) for k in range(5)}


def _separator_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _zone_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_cols = _separator_columns(grid)
    assert sep_cols, "expected full-height grey separator columns"
    assert all(sep_cols[i + 1] - sep_cols[i] > 1 for i in range(len(sep_cols) - 1))

    starts = [0] + [c + 1 for c in sep_cols]
    ends = [c - 1 for c in sep_cols] + [grid.shape[1] - 1]
    bounds = [(int(s), int(e)) for s, e in zip(starts, ends) if s <= e]
    widths = {e - s + 1 for s, e in bounds}
    assert len(widths) == 1, "all zones must have equal width"
    assert len(bounds) >= 2, "expected multiple separator-delimited zones"
    return bounds


def _landing_cell(r: int, c: int, mr: int, mc: int) -> tuple[int, int]:
    assert (r == mr) ^ (c == mc), "moving pixel must align with the marker by row xor column"
    if r == mr:
        return mr, mc - 1 if c < mc else mc + 1
    return mr - 1 if r < mr else mr + 1, mc


def _transform_zone(zone: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    markers = np.argwhere(zone == MAROON)
    assert markers.shape[0] == 1, "each zone must contain exactly one maroon marker"
    mr, mc = map(int, markers[0])
    assert 0 < mr < zone.shape[0] - 1 and 0 < mc < zone.shape[1] - 1, "marker needs four neighbors"
    assert np.count_nonzero(zone == GREY) == 0, "grey separators must not appear inside zones"

    landings: dict[tuple[int, int], list[int]] = {}
    initial_count = 0
    for r_raw, c_raw in np.argwhere(zone != BG):
        r = int(r_raw)
        c = int(c_raw)
        color = int(zone[r, c])
        if color == MAROON:
            continue
        initial_count += 1
        rr, cc = _landing_cell(r, c, mr, mc)
        landings.setdefault((rr, cc), []).append(color)

    out = np.zeros_like(zone)
    out[mr, mc] = MAROON
    survivors = 0
    collisions = 0
    for (rr, cc), colors in landings.items():
        assert len(colors) <= 2, "generator guarantees at most two pixels per landing cell"
        if len(colors) == 1:
            out[rr, cc] = int(colors[0])
            survivors += 1
        else:
            collisions += 1
    return out, survivors, initial_count, collisions


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bounds = _zone_bounds(g)
    zone_outputs: list[np.ndarray] = []
    survivor_counts: list[int] = []
    initial_counts: list[int] = []
    collision_counts: list[int] = []
    for start, end in bounds:
        zone = g[:, start : end + 1]
        zone_out, survivors, initial, collisions = _transform_zone(zone)
        zone_outputs.append(zone_out)
        survivor_counts.append(int(survivors))
        initial_counts.append(int(initial))
        collision_counts.append(int(collisions))

    order = sorted(range(len(zone_outputs)), key=lambda idx: (-survivor_counts[idx], idx))
    out = np.zeros_like(g)
    for c in _separator_columns(g):
        out[:, c] = GREY
    for slot, zone_idx in enumerate(order):
        start, end = bounds[slot]
        out[:, start : end + 1] = zone_outputs[zone_idx]
    return out, survivor_counts, initial_counts, collision_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each column-zone's aligned pixels onto its marker, annihilate same-side collisions, then sort the zones by survivors.

    Input structure:
    - Full-height grey columns split the grid into equal-width vertical zones.
    - Each zone contains exactly one maroon singleton marker away from the border.
    - Every other foreground pixel in a zone lies on the marker's row or column.

    Rule:
    - Within each zone, move every non-marker pixel to the cell adjacent to the marker on its shared row or column.
    - If two pixels from the same side of the same axis land on that same adjacent cell, both vanish instead of surviving there.
    - Count the surviving non-marker pixels around each marker, then reorder the whole zones from highest survivor count to lowest, keeping separators between them.

    Key insight:
    - The sorting key is not the original scatter count: it is the post-slide survivor count after pairwise same-side annihilation at the marker-adjacent landing cells.
    """
    out, _survivors, _initial, _collisions = _solve_details(grid)
    return out


def _available_side_cells(height: int) -> dict[str, list[tuple[int, int]]]:
    mid_r = height // 2
    mid_c = ZONE_WIDTH // 2
    return {
        "U": [(r, mid_c) for r in range(mid_r - 1, -1, -1)],
        "D": [(r, mid_c) for r in range(mid_r + 1, height)],
        "L": [(mid_r, c) for c in range(mid_c - 1, -1, -1)],
        "R": [(mid_r, c) for c in range(mid_c + 1, ZONE_WIDTH)],
    }


def _build_zone(height: int, survivor_target: int) -> tuple[np.ndarray, dict[str, object]]:
    zone = np.zeros((height, ZONE_WIDTH), dtype=int)
    mid_r = height // 2
    mid_c = ZONE_WIDTH // 2
    zone[mid_r, mid_c] = MAROON

    pattern = random.choice(PATTERNS_BY_SURVIVORS[survivor_target])
    available = _available_side_cells(height)
    total_pixels = sum(pattern)
    colors = random.sample(PALETTE, total_pixels)
    color_idx = 0
    placements: dict[str, list[tuple[int, int]]] = {}
    for direction, count in zip(DIRECTIONS, pattern):
        chosen = random.sample(available[direction], count)
        placements[direction] = chosen
        for r, c in chosen:
            zone[r, c] = int(colors[color_idx])
            color_idx += 1

    transformed, survivors, initial_count, collisions = _transform_zone(zone)
    assert survivors == survivor_target
    assert initial_count == total_pixels
    meta = {
        "survivors": int(survivors),
        "initial": int(initial_count),
        "collisions": int(collisions),
        "pattern": tuple(int(v) for v in pattern),
        "placements": placements,
        "transformed": transformed,
    }
    return zone, meta


def _assemble_grid(zones: Iterable[np.ndarray]) -> np.ndarray:
    zones = list(zones)
    height = zones[0].shape[0]
    grid = np.zeros((height, GRID_WIDTH), dtype=int)
    for idx, zone in enumerate(zones):
        start = idx * (ZONE_WIDTH + SEP_WIDTH)
        grid[:, start : start + ZONE_WIDTH] = zone
        if idx < len(zones) - 1:
            grid[:, start + ZONE_WIDTH] = GREY
    return grid


def _has_opposite_side_example(patterns: list[tuple[int, int, int, int]]) -> bool:
    for u, d, l, r in patterns:
        if (u == 1 and d == 1) or (l == 1 and r == 1):
            return True
    return False


def _validate_generated_case(grid: np.ndarray, metas: list[dict[str, object]]) -> bool:
    if grid.shape[1] != GRID_WIDTH or grid.shape[0] not in HEIGHT_CHOICES:
        return False
    if len(metas) != N_ZONES:
        return False

    survivor_counts = [int(meta["survivors"]) for meta in metas]
    initial_counts = [int(meta["initial"]) for meta in metas]
    collision_counts = [int(meta["collisions"]) for meta in metas]
    patterns = [tuple(meta["pattern"]) for meta in metas]

    if len(set(survivor_counts)) != N_ZONES:
        return False
    if any(not (3 <= count <= 7) for count in initial_counts):
        return False
    if all(collisions == 0 for collisions in collision_counts):
        return False
    if all(collisions > 0 for collisions in collision_counts):
        return False
    if not any(initial != survivors for initial, survivors in zip(initial_counts, survivor_counts)):
        return False
    if not _has_opposite_side_example(patterns):
        return False

    desired_order = sorted(range(N_ZONES), key=lambda idx: (-survivor_counts[idx], idx))
    if desired_order == list(range(N_ZONES)):
        return False
    shortcut_order = sorted(range(N_ZONES), key=lambda idx: (-initial_counts[idx], idx))
    if shortcut_order == desired_order:
        return False

    try:
        actual_out, actual_survivors, actual_initial, actual_collisions = _solve_details(grid)
    except AssertionError:
        return False
    if actual_survivors != survivor_counts:
        return False
    if actual_initial != initial_counts:
        return False
    if actual_collisions != collision_counts:
        return False

    expected_sorted = [metas[idx]["transformed"] for idx in desired_order]
    expected_out = _assemble_grid(expected_sorted)
    if not np.array_equal(actual_out, expected_out):
        return False
    if np.array_equal(actual_out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(HEIGHT_CHOICES)
    survivor_set = random.choices(SURVIVOR_SETS, weights=SURVIVOR_SET_WEIGHTS, k=1)[0]
    survivor_counts = list(survivor_set)
    random.shuffle(survivor_counts)
    if survivor_counts == sorted(survivor_counts, reverse=True):
        return None

    zones: list[np.ndarray] = []
    metas: list[dict[str, object]] = []
    for count in survivor_counts:
        zone, meta = _build_zone(height, count)
        zones.append(zone)
        metas.append(meta)

    grid = _assemble_grid(zones)
    if not _validate_generated_case(grid, metas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-annihilation sorting puzzle")
