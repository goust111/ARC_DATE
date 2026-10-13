import math
import random
from typing import Optional

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300
CELL_SIZE_OPTIONS = (5, 7, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _gap_position(cell_size: int, direction: str) -> tuple[int, int]:
    mid = cell_size // 2
    if direction == "up":
        return 0, mid
    if direction == "down":
        return cell_size - 1, mid
    if direction == "left":
        return mid, 0
    if direction == "right":
        return mid, cell_size - 1
    raise ValueError(f"unknown direction: {direction}")


def _target_start(cell_size: int) -> int:
    return cell_size // 2 - 1


def _classify_target(patch: np.ndarray) -> Optional[dict]:
    cell_size = patch.shape[0]
    start = _target_start(cell_size)
    mask = np.zeros_like(patch, dtype=bool)
    mask[start : start + 3, start : start + 3] = True

    block = patch[mask]
    if np.any(block == BG):
        return None
    color = int(block[0])
    if not np.all(block == color):
        return None
    if np.any(patch[~mask] != BG):
        return None
    return {"kind": "target", "color": color}


def _classify_ring(patch: np.ndarray) -> Optional[dict]:
    cell_size = patch.shape[0]
    if np.any(patch[1:-1, 1:-1] != BG):
        return None

    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    non_bg = patch[border][patch[border] != BG]
    if non_bg.size == 0:
        return None
    color = int(non_bg[0])
    if not np.all(non_bg == color):
        return None

    gap_cells = np.argwhere(border & (patch == BG))
    if gap_cells.shape[0] != 1:
        return None
    gap = tuple(map(int, gap_cells[0]))

    direction = None
    for name in DIRS:
        if gap == _gap_position(cell_size, name):
            direction = name
            break
    if direction is None:
        return None

    expected = np.full_like(patch, BG)
    expected[border] = color
    expected[gap] = BG
    if not np.array_equal(patch, expected):
        return None
    return {"kind": "ring", "color": color, "direction": direction}


def _classify_cell(patch: np.ndarray) -> Optional[dict]:
    if np.all(patch == BG):
        return {"kind": "empty"}

    target = _classify_target(patch)
    if target is not None:
        return target

    ring = _classify_ring(patch)
    if ring is not None:
        return ring

    return None


def _analyze_size(grid: np.ndarray, cell_size: int) -> Optional[dict]:
    height, width = grid.shape
    if height % cell_size != 0 or width % cell_size != 0:
        return None

    rows = height // cell_size
    cols = width // cell_size
    cells: list[list[dict]] = []
    ring_count = 0
    target_count = 0

    for cell_r in range(rows):
        row_info = []
        for cell_c in range(cols):
            r0 = cell_r * cell_size
            c0 = cell_c * cell_size
            patch = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            info = _classify_cell(patch)
            if info is None:
                return None
            row_info.append(info)
            if info["kind"] == "ring":
                ring_count += 1
            elif info["kind"] == "target":
                target_count += 1
        cells.append(row_info)

    if ring_count == 0 or target_count == 0:
        return None

    for cell_r in range(rows):
        for cell_c in range(cols):
            info = cells[cell_r][cell_c]
            if info["kind"] != "ring":
                continue
            dr, dc = DIRS[info["direction"]]
            nr = cell_r + dr
            nc = cell_c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                return None
            if cells[nr][nc]["kind"] != "target":
                return None

    return {"cell_size": cell_size, "rows": rows, "cols": cols, "cells": cells}


def _parse_layout(grid: np.ndarray) -> dict:
    height, width = grid.shape
    candidates = []
    for cell_size in range(5, min(height, width) + 1, 2):
        parsed = _analyze_size(grid, cell_size)
        if parsed is not None:
            candidates.append(parsed)
    assert candidates, "expected a valid cell partition"
    return min(candidates, key=lambda item: item["cell_size"])


def _draw_ring(grid: np.ndarray, cell_r: int, cell_c: int, cell_size: int, color: int, direction: str) -> None:
    r0 = cell_r * cell_size
    c0 = cell_c * cell_size
    patch = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    gap_r, gap_c = _gap_position(cell_size, direction)
    patch[gap_r, gap_c] = BG


def _draw_target(grid: np.ndarray, cell_r: int, cell_c: int, cell_size: int, color: int) -> None:
    r0 = cell_r * cell_size
    c0 = cell_c * cell_size
    start = _target_start(cell_size)
    grid[r0 + start : r0 + start + 3, c0 + start : c0 + start + 3] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow ring with the color of the adjacent target block indicated by its gap direction.

    Input structure:
    - The background is black.
    - The canvas is partitioned into equal square cells with no spacing between them.
    - Some cells contain a colored hollow square ring on the cell border, with a one-pixel gap at the center of exactly one side.
    - Some cells contain a centered 3x3 solid color block, and other cells are empty.

    Rule:
    - Infer the common cell size from the repeated cell-local motifs.
    - For each ring cell, read the side containing the border gap to determine which orthogonally adjacent cell it points to.
    - That adjacent cell is a target block; copy its color into every interior pixel of the ring while keeping the border color unchanged.
    - Keep the gap pixel black, preserve target blocks, and leave empty cells unchanged.

    Key insight:
    - The missing border pixel is not damage: it is a direction marker that identifies both which neighboring cell to inspect and which color to use for the ring's interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layout = _parse_layout(g)
    cell_size = layout["cell_size"]
    out = g.copy()

    for cell_r in range(layout["rows"]):
        for cell_c in range(layout["cols"]):
            info = layout["cells"][cell_r][cell_c]
            if info["kind"] != "ring":
                continue

            dr, dc = DIRS[info["direction"]]
            target = layout["cells"][cell_r + dr][cell_c + dc]
            color = int(target["color"])

            r0 = cell_r * cell_size
            c0 = cell_c * cell_size
            out[r0 + 1 : r0 + cell_size - 1, c0 + 1 : c0 + cell_size - 1] = color

    return out


def _dimension_options(cell_size: int) -> list[tuple[int, int]]:
    max_cells = 30 // cell_size
    min_cells = 2 if cell_size == 9 else 3
    options = []
    for rows in range(min_cells, max_cells + 1):
        for cols in range(min_cells, max_cells + 1):
            if math.gcd(rows, cols) == 1:
                options.append((rows, cols))
    return options


def _choose_pairs(rows: int, cols: int, rng: random.Random) -> list[tuple[int, int, str, int, int]]:
    target_pairs = rng.randint(1, min(6, (rows * cols) // 2))
    used: set[tuple[int, int]] = set()
    pairs = []

    for _ in range(target_pairs):
        feasible_dirs = []
        for direction, (dr, dc) in DIRS.items():
            starts = []
            for cell_r in range(rows):
                for cell_c in range(cols):
                    dst = (cell_r + dr, cell_c + dc)
                    if (cell_r, cell_c) in used:
                        continue
                    if not (0 <= dst[0] < rows and 0 <= dst[1] < cols):
                        continue
                    if dst in used:
                        continue
                    starts.append((cell_r, cell_c))
            if starts:
                feasible_dirs.append((direction, dr, dc, starts))
        if not feasible_dirs:
            break

        direction, dr, dc, starts = rng.choice(feasible_dirs)
        cell_r, cell_c = rng.choice(starts)
        nr = cell_r + dr
        nc = cell_c + dc
        used.add((cell_r, cell_c))
        used.add((nr, nc))
        pairs.append((cell_r, cell_c, direction, nr, nc))

    return pairs


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    cell_size = rng.choice(CELL_SIZE_OPTIONS)
    rows, cols = rng.choice(_dimension_options(cell_size))

    pairs = _choose_pairs(rows, cols, rng)
    if not pairs:
        return None

    grid = np.zeros((rows * cell_size, cols * cell_size), dtype=int)
    occupied = {(src_r, src_c) for src_r, src_c, _, _, _ in pairs}
    occupied.update({(dst_r, dst_c) for _, _, _, dst_r, dst_c in pairs})

    palette_size = rng.randint(3, min(7, len(COLORS)))
    palette = rng.sample(COLORS, palette_size)

    for cell_r, cell_c, direction, nr, nc in pairs:
        ring_color = rng.choice(palette)
        target_choices = [color for color in palette if color != ring_color]
        target_color = rng.choice(target_choices)
        _draw_ring(grid, cell_r, cell_c, cell_size, ring_color, direction)
        _draw_target(grid, nr, nc, cell_size, target_color)

    remaining = [
        (cell_r, cell_c)
        for cell_r in range(rows)
        for cell_c in range(cols)
        if (cell_r, cell_c) not in occupied
    ]
    rng.shuffle(remaining)
    extra_targets = rng.randint(0, min(3, len(remaining)))
    for cell_r, cell_c in remaining[:extra_targets]:
        _draw_target(grid, cell_r, cell_c, cell_size, rng.choice(palette))

    parsed = _parse_layout(grid)
    if parsed["cell_size"] != cell_size:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-ring puzzle")
