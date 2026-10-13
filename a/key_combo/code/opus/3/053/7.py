import random

import numpy as np


BLOCK_COLORS = [1, 2, 3]


def _find_tile(grid):
    h, w = grid.shape
    corners = [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
    for size in (3, 2):
        starts = [(0, 0), (0, w - size), (h - size, 0), (h - size, w - size)]
        for r, c in starts:
            patch = grid[r : r + size, c : c + size]
            vals = set(map(int, patch.flatten()))
            if 0 not in vals and len(vals) >= 2:
                return (r, c, size, patch.copy())
    raise ValueError("no corner tile found")


def _find_blocks(grid, tile_cells):
    h, w = grid.shape
    blocks = []
    occupied = set(tile_cells)
    for r in range(h - 2):
        for c in range(w - 2):
            cells = {(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3)}
            if cells & occupied:
                continue
            patch = grid[r : r + 3, c : c + 3]
            color = int(patch[1, 1])
            if color in BLOCK_COLORS and np.all(patch == color):
                blocks.append({"color": color, "center": (r + 1, c + 1), "cells": cells})
                occupied |= cells
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Build cross-ray masks from solid blocks, reset pair intersections, then upscale filled pixels by the corner tile."""
    tr, tc, tile_size, tile = _find_tile(grid)
    tile_cells = {(r, c) for r in range(tr, tr + tile_size) for c in range(tc, tc + tile_size)}
    blocks = _find_blocks(grid, tile_cells)
    work = grid.copy()
    for block in blocks:
        color = block["color"]
        cr, cc = block["center"]
        work[cr, :] = color
        work[:, cc] = color
    for block in blocks:
        for r, c in block["cells"]:
            work[r, c] = block["color"]
    if len(blocks) >= 2:
        for i, a in enumerate(blocks):
            for b in blocks[i + 1 :]:
                for r, c in ((a["center"][0], b["center"][1]), (b["center"][0], a["center"][1])):
                    work[r, c] = 0
    mask = work != 0
    h, w = grid.shape
    out = np.zeros((h * tile_size, w * tile_size), dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                out[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size] = tile
    return out


def generate_input() -> np.ndarray:
    h = w = 10
    grid = np.zeros((h, w), dtype=int)
    tile_size = random.choice([2, 3])
    corner = random.choice([(0, 0), (0, w - tile_size), (h - tile_size, 0), (h - tile_size, w - tile_size)])
    colors = random.sample([4, 5, 6, 8, 9], random.choice([2, 3]))
    for r in range(tile_size):
        for c in range(tile_size):
            grid[corner[0] + r, corner[1] + c] = colors[(r + c) % len(colors)]
    possible = [(2, 2), (2, 7), (7, 2), (7, 7)]
    possible = [p for p in possible if abs(p[0] - corner[0]) > 2 or abs(p[1] - corner[1]) > 2]
    centers = random.sample(possible, random.choice([1, 2]))
    used_rows = set()
    used_cols = set()
    filtered = []
    for r, c in centers:
        if r not in used_rows and c not in used_cols:
            filtered.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    for color, (r, c) in zip(random.sample(BLOCK_COLORS, len(filtered)), filtered):
        grid[r - 1 : r + 2, c - 1 : c + 2] = color
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
