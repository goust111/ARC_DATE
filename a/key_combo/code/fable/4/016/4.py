"""Mirror reflection puzzle solver/generator."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Dict, Tuple

import numpy as np
import sys
from pathlib import Path

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    _UTIL_ROOT = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            _UTIL_ROOT = candidate
            break
    if _UTIL_ROOT is None:
        raise
    sys.path.append(str(_UTIL_ROOT))
    from arc_utils import get_bg

Edge = Tuple[int, int]
MirrorMap = Dict[Tuple[int, int], str]


def _is_edge(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _inward_direction(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if r == 0 and c == 0:
        return 1, 0
    if r == 0 and c == w - 1:
        return 1, 0
    if r == h - 1 and c == 0:
        return -1, 0
    if r == h - 1 and c == w - 1:
        return -1, 0
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    raise ValueError("inward direction requested for a non-edge cell")


def _reflect(dr: int, dc: int, orientation: str) -> tuple[int, int]:
    """
    Reflect a direction by a mirror orientation.

    "/" reflects (dr,dc) -> (-dc, -dr)
    "\\" reflects (dr,dc) -> (dc, dr)
    """
    if orientation == "/":
        return -dc, -dr
    if orientation == "\\":
        return dc, dr
    raise ValueError(f"invalid mirror orientation: {orientation}")


def _collect_emitters(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    emitters: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if _is_edge(r, c, h, w):
                col = int(grid[r, c])
                if col != bg:
                    emitters.append((r, c, col))
    return emitters


def _collect_mirrors(grid: np.ndarray, bg: int, emitter_cells: set[Edge]) -> MirrorMap:
    """
    Find internal diagonal mirror pixels.

    A mirror is exactly one diagonal-adjacent pixel pair of the same color.
    """
    h, w = grid.shape
    mirror: MirrorMap = {}
    used: set[Edge] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in emitter_cells:
                continue
            if (r, c) in used:
                continue

            color = int(grid[r, c])
            if color == bg:
                continue

            candidates: list[tuple[int, int, str]] = []
            r1, c1 = r + 1, c + 1
            if grid[r1, c1] == color and (r1, c1) not in emitter_cells:
                candidates.append((r1, c1, "\\"))

            r1, c1 = r + 1, c - 1
            if grid[r1, c1] == color and (r1, c1) not in emitter_cells:
                candidates.append((r1, c1, "/"))

            if len(candidates) == 1:
                nr, nc, ori = candidates[0]
                if (nr, nc) in used:
                    continue
                mirror[(r, c)] = ori
                mirror[(nr, nc)] = ori
                used.add((r, c))
                used.add((nr, nc))
            else:
                used.add((r, c))

    return mirror


def _trace_beam(
    out: np.ndarray,
    mirrors: MirrorMap,
    bg: int,
    start_r: int,
    start_c: int,
    color: int,
    dr: int,
    dc: int,
) -> tuple[int, int, bool]:
    """
    Trace one beam in-place.

    Returns:
        (painted_cells, reflections, hit_loop)
    """
    h, w = out.shape
    r, c = start_r, start_c
    painted = 0
    reflections = 0
    hit_loop = False
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        if state in seen:
            hit_loop = True
            break
        seen.add(state)

        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            break

        orientation = mirrors.get((nr, nc))
        if orientation is not None:
            dr, dc = _reflect(dr, dc, orientation)
            reflections += 1
            continue

        r, c = nr, nc
        if out[r, c] == bg:
            out[r, c] = color
            painted += 1

    return painted, reflections, hit_loop


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int | bool]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = out.shape
    bg = int(get_bg(out))

    emitters = _collect_emitters(out, bg)
    emitter_cells = {(r, c) for r, c, _ in emitters}
    mirrors = _collect_mirrors(out, bg, emitter_cells)

    total_reflections = 0
    total_painted = 0
    looped = False
    for r, c, color in emitters:
        dr, dc = _inward_direction(r, c, h, w)
        painted, reflections, hit_loop = _trace_beam(
            out=out,
            mirrors=mirrors,
            bg=bg,
            start_r=r,
            start_c=c,
            color=color,
            dr=dr,
            dc=dc,
        )
        total_painted += painted
        total_reflections += reflections
        if hit_loop:
            looped = True

    stats: dict[str, int | bool] = {
        "total_reflections": total_reflections,
        "painted_cells": total_painted,
        "looped": looped,
        "emitters": len(emitters),
        "mirrors": len(mirrors) // 2,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace rays from edge emitters through diagonal mirror pixels and draw the full beam paths.

    Input structure:
    - Background color is dominant.
    - One or more emitter pixels are on the border.
    - Internal mirrors are diagonal pixel pairs of one color, each pair being either "/" or "\" orientation.

    Rule:
    - Each border emitter projects a beam inward, normal to the side it sits on.
    - Beam endpoints are blocked only by grid bounds; mirrors are not overwritten and are not removed.
    - If the next cell is a mirror pixel, reflect the direction by that mirror orientation at the current cell and continue.
    - Otherwise draw the beam through empty cells using the emitter color.

    Key insight:
    - Mirror logic is fully determined by local 90° reflection geometry on each hit of a diagonal mirror pixel.
    """
    out, _ = _simulate(grid)
    return out


def _edge_positions(h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if h < 3 or w < 3:
        return positions

    for c in range(1, w - 1):
        positions.append((0, c))
        if h > 1:
            positions.append((h - 1, c))
    for r in range(1, h - 1):
        positions.append((r, 0))
        if w > 1:
            positions.append((r, w - 1))
    return positions


def _place_pair(
    grid: np.ndarray,
    bg: int,
    occupancy: set[Edge],
    orientation: str,
    color: int,
) -> bool:
    h, w = grid.shape
    if orientation == "\\":
        if h < 4 or w < 4:
            return False
        r = random.randint(1, h - 3)
        c = random.randint(1, w - 3)
        cells = ((r, c), (r + 1, c + 1))
    else:
        if h < 4 or w < 4:
            return False
        r = random.randint(1, h - 3)
        c = random.randint(2, w - 2)
        cells = ((r, c), (r + 1, c - 1))

    for cell in cells:
        if cell in occupancy:
            return False
        rr, cc = cell
        if grid[rr, cc] != bg:
            return False

    for rr, cc in cells:
        grid[rr, cc] = int(color)
        occupancy.add((rr, cc))
    return True


def _gen_attempt() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(12, 24)
        w = random.randint(12, 24)
        if min(h, w) < 4:
            continue

        bg = random.randint(0, 9)
        grid = np.full((h, w), int(bg), dtype=int)

        edge_cells = _edge_positions(h, w)
        if not edge_cells:
            continue
        emitter_count = random.randint(1, min(4, len(edge_cells)))
        emitter_cells = random.sample(edge_cells, emitter_count)

        palette = [c for c in range(10) if c != bg]
        emitter_colors = random.sample(palette, emitter_count)
        emitters = list(zip(emitter_cells, emitter_colors))

        for (r, c), col in emitters:
            grid[r, c] = col

        occupancy: set[Edge] = set(emitter_cells)
        used_mirror_colors = [c for c in palette if c not in emitter_colors]
        if not used_mirror_colors:
            used_mirror_colors = [c for c in range(10) if c != bg]

        mirror_count = 0 if random.random() < 0.3 else random.randint(1, 5)
        mirror_count = min(mirror_count, (h - 3) * (w - 3) // 2)
        for _ in range(mirror_count):
            if random.random() < 0.5:
                orientation = "/"
            else:
                orientation = "\\"
            color = random.choice(used_mirror_colors)
            placed = False
            for _ in range(120):
                if _place_pair(grid, bg, occupancy, orientation, color):
                    placed = True
                    break
            if not placed:
                break
        else:
            out, stats = _simulate(grid)

            if stats["painted_cells"] == 0:
                continue
            if stats["looped"]:
                continue
            if stats["emitters"] == 0:
                continue
            if np.count_nonzero(grid == bg) <= (h * w) // 2:
                continue
            if random.random() < 0.7 and stats["total_reflections"] == 0:
                continue
            try:
                if not np.array_equal(out, main(grid)):
                    continue
            except Exception:
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 4.py input")
