from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BRIDGE = 5
RIVER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bridge_orientation(cells: list[tuple[int, int]]) -> str:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        return "h"
    if len(cols) == 1:
        return "v"
    raise ValueError("bridge component is not a straight line")


def _trace_river_color(grid: np.ndarray, r: int, c: int, orientation: str) -> int | None:
    if orientation == "h":
        directions = [(-1, 0), (1, 0)]
    else:
        directions = [(0, -1), (0, 1)]

    colors: list[int] = []
    h, w = grid.shape
    for dr, dc in directions:
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            value = int(grid[rr, cc])
            if value == BRIDGE:
                rr += dr
                cc += dc
                continue
            if value == BG:
                break
            colors.append(value)
            break

    if not colors:
        return None
    counts = Counter(colors)
    return max(counts, key=lambda color: (counts[color], -colors.index(color)))


def _bridge_color(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    orientation = _bridge_orientation(cells)
    ordered = sorted(cells, key=lambda rc: (rc[1], rc[0]) if orientation == "h" else (rc[0], rc[1]))
    center = len(ordered) // 2
    probe_order = sorted(range(len(ordered)), key=lambda idx: (abs(idx - center), idx))

    votes: list[int] = []
    for idx in probe_order:
        color = _trace_river_color(grid, ordered[idx][0], ordered[idx][1], orientation)
        if color is None:
            continue
        if idx == center:
            return color
        votes.append(color)

    if votes:
        counts = Counter(votes)
        return max(counts, key=lambda color: (counts[color], -votes.index(color)))
    raise ValueError("unable to infer river color under bridge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey bridge to match the river running underneath it.

    Input structure:
    - The background is black.
    - Non-black, non-grey cells form one-pixel-wide monochrome rivers that run
      orthogonally and may turn.
    - Grey cells form short straight bridge segments that cover a river crossing
      cell and extend into black cells on both sides.

    Rule:
    - Find every 4-connected grey component and treat it as one bridge.
    - Determine whether the bridge is horizontal or vertical from its geometry.
    - Trace perpendicularly away from the bridge, preferring the bridge midpoint,
      to recover the color of the river hidden by the grey overlap.
    - Recolor the entire bridge component to that river color and leave all other
      cells unchanged.

    Key insight:
    The crossing cell's river color is occluded by the bridge, so the solver must
    infer the hidden color from the river continuation perpendicular to the bridge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g == BRIDGE):
        color = _bridge_color(g, comp)
        for r, c in comp:
            out[r, c] = color
    return out


def _random_partition(total: int, parts: int, min_each: int) -> list[int]:
    values = [min_each] * parts
    for _ in range(total - parts * min_each):
        values[random.randrange(parts)] += 1
    return values


def _sample_bands(n_rivers: int) -> tuple[int, list[tuple[int, int]]]:
    band_width = 5
    gaps = [1] + [2] * (n_rivers - 1) + [1]
    min_width = n_rivers * band_width + sum(gaps)
    width = random.randint(min_width, min(30, min_width + 4))
    for _ in range(width - min_width):
        gaps[random.randrange(len(gaps))] += 1

    bands: list[tuple[int, int]] = []
    col = gaps[0]
    for idx in range(n_rivers):
        bands.append((col, col + band_width - 1))
        col += band_width
        if idx < n_rivers - 1:
            col += gaps[idx + 1]
    return width, bands


def _build_river(h: int, band: tuple[int, int], color: int) -> dict | None:
    band_lo, band_hi = band
    start_r = random.randint(1, min(3, h - 9))
    max_turns = min(3, (h - 2 - start_r) // 2 - 1)
    if max_turns < 1:
        return None
    turns = random.randint(1, max_turns)

    end_r_min = start_r + 2 * (turns + 1)
    if end_r_min > h - 2:
        return None
    end_r = random.randint(end_r_min, h - 2)
    vertical_steps = _random_partition(end_r - start_r, turns + 1, 2)

    c = random.randint(band_lo + 1, band_hi - 1)
    r = start_r
    path = [(r, c)]
    vertical_runs: list[list[tuple[int, int]]] = []
    horizontal_runs: list[list[tuple[int, int]]] = []

    for idx, steps in enumerate(vertical_steps):
        run = [(r, c)]
        for _ in range(steps):
            r += 1
            run.append((r, c))
            path.append((r, c))
        vertical_runs.append(run)
        if idx == turns:
            continue

        targets = [target for target in range(band_lo, band_hi + 1) if abs(target - c) >= 2]
        if not targets:
            return None
        target_c = random.choice(targets)
        dc = 1 if target_c > c else -1
        run = [(r, c)]
        while c != target_c:
            c += dc
            run.append((r, c))
            path.append((r, c))
        horizontal_runs.append(run)

    return {
        "color": color,
        "path": path,
        "path_set": set(path),
        "vertical_runs": vertical_runs,
        "horizontal_runs": horizontal_runs,
    }


def _render_rivers(shape: tuple[int, int], rivers: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for river in rivers:
        for r, c in river["path"]:
            grid[r, c] = river["color"]
    return grid


def _bridge_conflicts(cells: list[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in reserved:
                return True
    return False


def _bridge_candidates(grid: np.ndarray, rivers: list[dict]) -> tuple[list[dict], list[dict]]:
    h, w = grid.shape
    horizontal: list[dict] = []
    vertical: list[dict] = []

    for river_idx, river in enumerate(rivers):
        color = river["color"]
        for run in river["vertical_runs"]:
            for r, c in run[1:-1]:
                if c <= 0 or c >= w - 1:
                    continue
                if grid[r, c - 1] == BG and grid[r, c + 1] == BG:
                    horizontal.append(
                        {
                            "orientation": "h",
                            "river_idx": river_idx,
                            "color": color,
                            "cells": [(r, c - 1), (r, c), (r, c + 1)],
                        }
                    )
        for run in river["horizontal_runs"]:
            for r, c in run[1:-1]:
                if r <= 0 or r >= h - 1:
                    continue
                if grid[r - 1, c] == BG and grid[r + 1, c] == BG:
                    vertical.append(
                        {
                            "orientation": "v",
                            "river_idx": river_idx,
                            "color": color,
                            "cells": [(r - 1, c), (r, c), (r + 1, c)],
                        }
                    )
    random.shuffle(horizontal)
    random.shuffle(vertical)
    return horizontal, vertical


def _select_bridges(grid: np.ndarray, rivers: list[dict]) -> list[dict] | None:
    horizontal, vertical = _bridge_candidates(grid, rivers)
    if not horizontal or not vertical:
        return None

    selected: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    for pool in (horizontal, vertical):
        chosen = None
        for candidate in pool:
            if not _bridge_conflicts(candidate["cells"], reserved):
                chosen = candidate
                break
        if chosen is None:
            return None
        selected.append(chosen)
        reserved.update(chosen["cells"])

    extra_target = random.randint(0, len(rivers))
    combined = horizontal + vertical
    random.shuffle(combined)
    for candidate in combined:
        if len(selected) >= 2 + extra_target:
            break
        if _bridge_conflicts(candidate["cells"], reserved):
            continue
        selected.append(candidate)
        reserved.update(candidate["cells"])

    if len({bridge["orientation"] for bridge in selected}) < 2:
        return None
    return selected


def _scene_ok(grid: np.ndarray, rivers: list[dict], bridges: list[dict]) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if any(grid[r, c] == BRIDGE for bridge in bridges for r, c in bridge["cells"]):
        return False
    for i, river in enumerate(rivers):
        for j, other in enumerate(rivers):
            if i >= j:
                continue
            if river["path_set"] & other["path_set"]:
                return False
    return True


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 22)
    n_rivers = random.randint(2, 4)
    w, bands = _sample_bands(n_rivers)
    colors = random.sample(RIVER_COLORS, n_rivers)

    rivers: list[dict] = []
    for band, color in zip(bands, colors):
        river = _build_river(h, band, color)
        if river is None:
            return None
        rivers.append(river)

    out = _render_rivers((h, w), rivers)
    bridges = _select_bridges(out, rivers)
    if bridges is None or not _scene_ok(out, rivers, bridges):
        return None

    for bridge in bridges:
        for r, c in bridge["cells"]:
            out[r, c] = bridge["color"]

    inp = out.copy()
    for bridge in bridges:
        for r, c in bridge["cells"]:
            inp[r, c] = BRIDGE

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _generate_scene()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid bridge-and-river input")
