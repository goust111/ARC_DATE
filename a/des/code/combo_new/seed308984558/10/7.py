import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MARKERS = [1, 2, 3, 4, 6, 7, 8, 9]
ART = [2, 3, 4, 6, 7, 8, 9]


def _tile_boxes(grid):
    mask = grid != BLACK
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            out.append((min(rows), min(cols), max(rows), max(cols)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble equal square tiles by matching same-color edge-center connectors.

    Input structure:
    - The input contains equal-sized square tiles scattered on black.
    - Each tile has a grey base, interior art, and optional colored markers at edge
      centers. Every marker color appears exactly twice on complementary edges.

    Rule:
    - Match same-color markers on opposite edge types to establish tile adjacency.
    - Find the tile with no top and no left incoming connection as the layout origin.
    - Arrange tiles into the implied rectangular tiling and remove the connector
      markers by restoring those edge-center cells to grey.

    Key insight:
    - The scattered tile positions are irrelevant noise; the colored edge-center pixels
      encode the true puzzle adjacency graph that reconstructs the final rectangle.
    """
    boxes = _tile_boxes(grid)
    size = boxes[0][2] - boxes[0][0] + 1
    tiles = {}
    marker_map = {}
    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        mids = {
            "top": (0, size // 2),
            "bottom": (size - 1, size // 2),
            "left": (size // 2, 0),
            "right": (size // 2, size - 1),
        }
        links = {}
        for side, (r, c) in mids.items():
            val = int(patch[r, c])
            if val not in (BLACK, GREY):
                links[side] = val
                marker_map.setdefault(val, []).append((idx, side))
                patch[r, c] = GREY
        tiles[idx] = {"patch": patch, "links": links}
    right_of = {}
    below_of = {}
    for color, pair in marker_map.items():
        (a, sa), (b, sb) = pair
        if {sa, sb} == {"left", "right"}:
            left_idx = a if sa == "right" else b
            right_idx = b if sa == "right" else a
            right_of[left_idx] = right_idx
        else:
            top_idx = a if sa == "bottom" else b
            bottom_idx = b if sa == "bottom" else a
            below_of[top_idx] = bottom_idx
    origin = next(idx for idx, t in tiles.items() if "top" not in t["links"] and "left" not in t["links"])
    rows = []
    cur_row = origin
    while True:
        row = [cur_row]
        cur = cur_row
        while cur in right_of:
            cur = right_of[cur]
            row.append(cur)
        rows.append(row)
        if cur_row in below_of:
            cur_row = below_of[cur_row]
        else:
            break
    out = np.zeros((len(rows) * size, len(rows[0]) * size), dtype=int)
    for i, row in enumerate(rows):
        for j, idx in enumerate(row):
            out[i * size : (i + 1) * size, j * size : (j + 1) * size] = tiles[idx]["patch"]
    return out


def generate_input() -> np.ndarray:
    tile = 5
    rows, cols = random.choice([(2, 2), (2, 3), (3, 2)])
    h = rows * (tile + 2) + 6
    w = cols * (tile + 2) + 6
    grid = np.zeros((h, w), dtype=int)
    marker_iter = iter(random.sample(MARKERS, rows * (cols - 1) + cols * (rows - 1)))
    tiles = []
    for i in range(rows):
        for j in range(cols):
            patch = np.full((tile, tile), GREY, dtype=int)
            for _ in range(random.randint(3, 5)):
                patch[random.randint(1, tile - 2), random.randint(1, tile - 2)] = random.choice(ART)
            links = {}
            if j < cols - 1:
                color = next(marker_iter)
                patch[tile // 2, tile - 1] = color
                links["right"] = color
            if i < rows - 1:
                color = next(marker_iter)
                patch[tile - 1, tile // 2] = color
                links["bottom"] = color
            tiles.append((patch, i, j, links))
    for idx, (patch, i, j, links) in enumerate(tiles):
        if j > 0:
            left_patch = tiles[idx - 1][0]
            patch[tile // 2, 0] = left_patch[tile // 2, tile - 1]
        if i > 0:
            top_patch = tiles[idx - cols][0]
            patch[0, tile // 2] = top_patch[tile - 1, tile // 2]
    positions = random.sample([(r, c) for r in range(0, h - tile + 1, tile + 1) for c in range(0, w - tile + 1, tile + 1)], len(tiles))
    for (patch, _, _, _), (r0, c0) in zip(tiles, positions):
        grid[r0 : r0 + tile, c0 : c0 + tile] = patch
    return grid
