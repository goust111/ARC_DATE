from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
BAR = Color.GREY
REVEAL = Color.RED

GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
BAR_COUNT_RANGE = (3, 5)
RADII = (2, 3, 4, 5)
ORIENTATIONS = ("H", "V")
MAX_GENERATION_ATTEMPTS = 500


def _make_bar(shape: tuple[int, int], orientation: str, radius: int, center: tuple[int, int]) -> dict | None:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    radius = int(radius)
    if radius < 1:
        return None
    if not (radius <= cr < h - radius and radius <= cc < w - radius):
        return None

    if orientation == "H":
        cells = {(cr, c) for c in range(cc - radius, cc + radius + 1)}
    else:
        cells = {(r, cc) for r in range(cr - radius, cr + radius + 1)}

    diamond = {
        (r, c)
        for r in range(cr - radius, cr + radius + 1)
        for c in range(cc - radius, cc + radius + 1)
        if abs(r - cr) + abs(c - cc) <= radius
    }
    if not cells <= diamond:
        return None

    return {
        "orientation": orientation,
        "radius": radius,
        "length": 2 * radius + 1,
        "center": (cr, cc),
        "cells": cells,
        "diamond": diamond,
    }


def _bars_visually_separate(a: dict, b: dict) -> bool:
    for r1, c1 in a["cells"]:
        for r2, c2 in b["cells"]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _background_overlap_cells(bars: list[dict], idx_a: int, idx_b: int) -> set[tuple[int, int]]:
    occupied = set().union(*(bar["cells"] for bar in bars))
    return (bars[idx_a]["diamond"] & bars[idx_b]["diamond"]) - occupied


def _pairwise_overlap_flags(bars: list[dict]) -> list[tuple[int, int, bool, int]]:
    data: list[tuple[int, int, bool, int]] = []
    for i in range(len(bars)):
        r1, c1 = bars[i]["center"]
        for j in range(i + 1, len(bars)):
            r2, c2 = bars[j]["center"]
            dist = abs(r1 - r2) + abs(c1 - c2)
            data.append((i, j, bool(_background_overlap_cells(bars, i, j)), dist))
    return data


def _bar_to_grid(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bar in bars:
        for r, c in bar["cells"]:
            grid[r, c] = BAR
    return grid


def _coverage_counts(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for bar in bars:
        for r, c in bar["diamond"]:
            counts[r, c] += 1
    return counts


def _expected_output(grid: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    counts = _coverage_counts(grid.shape, bars)
    out[(counts >= 2) & (grid == get_bg(grid))] = REVEAL
    return out


def _parse_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BG, "expected black background"
    colors = set(map(int, np.unique(g))) - {bg}
    assert colors == {BAR}, "expected only grey bars on the background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[BAR], return_format="dict")
    assert components, "expected at least one grey bar"

    bars: list[dict] = []
    for comp in components:
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        area = int(comp["area"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert area == len(coords)
        assert height == 1 or width == 1, "each component must be a straight bar"
        length = max(height, width)
        assert length % 2 == 1 and length >= 5, "bar lengths must be odd and at least 5"
        assert area == length, "bar must be solid and one cell thick"

        if height == 1:
            center = (int(r0), int((c0 + c1) // 2))
            orientation = "H"
        else:
            center = (int((r0 + r1) // 2), int(c0))
            orientation = "V"

        bar = _make_bar(g.shape, orientation, (length - 1) // 2, center)
        assert bar is not None
        assert bar["cells"] == {tuple(map(int, rc)) for rc in coords}
        bars.append(bar)

    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal only the background cells where at least two invisible bar-centered diamonds overlap.

    Input structure:
    - The grid has a black background and one or more grey one-cell-thick solid bars.
    - Every bar is horizontal or vertical, and every bar length is odd.
    - Red is absent from the input.

    Rule:
    - Each grey bar implies a Manhattan diamond centered at the bar midpoint, with radius (bar_length - 1) / 2.
    - Color a background cell red iff it lies inside at least two implied diamonds.
    - Leave cells covered by zero or one implied diamond black, and preserve every original grey bar cell.

    Key insight:
    - The diamonds are never drawn directly; the bar length encodes each hidden radius, and only the pairwise-or-more overlap residue becomes visible.
    """
    g = np.asarray(grid, dtype=int)
    bars = _parse_bars(g)
    return _expected_output(g, bars)


def _sample_random_bar(shape: tuple[int, int], orientation: str | None = None, radius: int | None = None) -> dict:
    h, w = map(int, shape)
    orientation = random.choice(ORIENTATIONS) if orientation is None else orientation
    radius = random.choice(RADII) if radius is None else int(radius)
    center = (random.randint(radius, h - radius - 1), random.randint(radius, w - radius - 1))
    bar = _make_bar(shape, orientation, radius, center)
    assert bar is not None
    return bar


def _sample_partner(shape: tuple[int, int], placed: list[dict], must_overlap: bool) -> dict | None:
    target = random.choice(placed)
    desired_orientation = random.choice(ORIENTATIONS)
    if len({bar["orientation"] for bar in placed}) == 1 and random.random() < 0.7:
        desired_orientation = "V" if placed[0]["orientation"] == "H" else "H"

    for _ in range(160):
        radius = random.choice(RADII)
        limit = target["radius"] + radius + 3
        offsets = [
            (dr, dc)
            for dr in range(-limit, limit + 1)
            for dc in range(-limit, limit + 1)
            if 0 < abs(dr) + abs(dc) <= limit
        ]
        random.shuffle(offsets)

        for dr, dc in offsets[:120]:
            candidate = _make_bar(
                shape,
                desired_orientation if random.random() < 0.75 else random.choice(ORIENTATIONS),
                radius,
                (target["center"][0] + dr, target["center"][1] + dc),
            )
            if candidate is None:
                continue
            if any(not _bars_visually_separate(candidate, bar) for bar in placed):
                continue

            trial = placed + [candidate]
            overlaps_existing = any(_background_overlap_cells(trial, idx, len(trial) - 1) for idx in range(len(placed)))
            if overlaps_existing == must_overlap:
                return candidate
    return None


def _sample_near_miss(shape: tuple[int, int], placed: list[dict]) -> dict | None:
    target = random.choice(placed)
    for _ in range(200):
        radius = random.choice(RADII)
        orientation = random.choice(ORIENTATIONS)
        dist = target["radius"] + radius + random.choice((1, 2))
        offsets = [
            (dr, dc)
            for dr in range(-dist, dist + 1)
            for dc in range(-dist, dist + 1)
            if abs(dr) + abs(dc) == dist
        ]
        random.shuffle(offsets)
        for dr, dc in offsets:
            candidate = _make_bar(shape, orientation, radius, (target["center"][0] + dr, target["center"][1] + dc))
            if candidate is None:
                continue
            if any(not _bars_visually_separate(candidate, bar) for bar in placed):
                continue
            trial = placed + [candidate]
            if any(_background_overlap_cells(trial, idx, len(trial) - 1) for idx in range(len(placed))):
                continue
            return candidate
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bars = _parse_bars(grid)
        out = main(grid)
    except Exception:
        return False

    if np.any(grid == REVEAL):
        return False
    if len(bars) < 3:
        return False
    if len({bar["length"] for bar in bars}) < 2:
        return False
    if len({bar["orientation"] for bar in bars}) < 2:
        return False

    expected = _expected_output(grid, bars)
    if not np.array_equal(out, expected):
        return False

    red_count = int(np.count_nonzero(out == REVEAL))
    if red_count < 2 or red_count > grid.size // 4:
        return False

    pair_data = _pairwise_overlap_flags(bars)
    overlap_pairs = [item for item in pair_data if item[2]]
    non_overlap_pairs = [item for item in pair_data if not item[2]]
    if not overlap_pairs or not non_overlap_pairs:
        return False

    isolated = []
    for idx in range(len(bars)):
        if all(not flag for i, j, flag, _dist in pair_data if idx in (i, j)):
            isolated.append(idx)
    if not isolated:
        return False

    if min(dist for _i, _j, _flag, dist in non_overlap_pairs) >= max(dist for _i, _j, _flag, dist in overlap_pairs):
        return False

    counts = _coverage_counts(grid.shape, bars)
    if not np.array_equal((counts >= 2) & (grid == BG), out == REVEAL):
        return False

    if np.any(out[(grid == BAR)] != BAR):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )
    bar_count = random.randint(*BAR_COUNT_RANGE)

    first = _sample_random_bar(shape, radius=random.choice((3, 4, 5)))
    second = _sample_partner(shape, [first], must_overlap=True)
    if second is None:
        return None

    bars = [first, second]
    near_miss = _sample_near_miss(shape, bars)
    if near_miss is None:
        return None
    bars.append(near_miss)

    while len(bars) < bar_count:
        want_overlap = random.random() < 0.5
        candidate = _sample_partner(shape, bars, must_overlap=want_overlap) if want_overlap else _sample_near_miss(shape, bars)
        if candidate is None:
            return None
        bars.append(candidate)

    grid = _bar_to_grid(shape, bars)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implicit-diamond overlap input")
