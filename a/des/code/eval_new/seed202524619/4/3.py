"""Fill enclosed black regions by keying checkerboard patterns from edge color-bands.

Input structure:
- Background is black. One or more enclosed regions of black are surrounded by maroon wall pixels.
- Each region has exactly one non-maroon, non-black pixel on its top wall (row-key color C1) and one on its left wall (column-key color C2).
- The top row contains a contiguous segment of color C1 with length W, and the left column contains a contiguous segment of color C2 with length H.

Rule:
- Copy the input grid.
- Find each enclosed black connected component (4-connected) that does not touch the grid border.
- For each such component, read C1 from its top wall and C2 from its left wall, then look up W and H from top-row/left-column bands for C1/C2.
- Tile the H×W checkerboard pattern from the component's interior top-left corner:
  positions where (row+col) is even use C1, odd use C2.
- Paint only the component interior cells; keep walls, key pixels, and password bands unchanged.

Key insight:
- The special wall pixels are not noise but selectors; their colors map each region to top-row and left-column code bands, and the selected lengths define the periodic checkerboard used to fill the region.
"""

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


BLACK = 0
WALL = 7


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components_of_black(grid: np.ndarray) -> List[Tuple[List[Tuple[int, int]], bool]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[List[Tuple[int, int]], bool]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for sr in range(h):
        for sc in range(w):
            if grid[sr, sc] != BLACK or visited[sr, sc]:
                continue
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: List[Tuple[int, int]] = []
            touches_border = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if visited[nr, nc] or grid[nr, nc] != BLACK:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            comps.append((cells, touches_border))

    return [c for c, outside in comps if not outside]


def _collect_band_lengths(line: Iterable[int], skip_colors: Tuple[int, int]) -> Dict[int, int]:
    skip_a, skip_b = skip_colors
    arr = list(line)
    n = len(arr)
    lengths: Dict[int, int] = {}

    i = 0
    while i < n:
        color = arr[i]
        if color in (skip_a, skip_b):
            i += 1
            continue
        j = i + 1
        while j < n and arr[j] == color:
            j += 1
        lengths[color] = max(lengths.get(color, 0), j - i)
        i = j

    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.array(grid, copy=True)

    top_band = _collect_band_lengths(grid[0, :].tolist(), (BLACK, WALL))
    left_band = _collect_band_lengths(grid[:, 0].tolist(), (BLACK, WALL))

    regions = _components_of_black(out)
    if not regions:
        return out

    dirs = ((-1, 0), (0, -1))

    for cells in regions:
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)

        top_keys = []
        left_keys = []

        for r, c in cells:
            # Top-wall key
            if r == min_r and _in_bounds(h, w, r - 1, c):
                color = out[r - 1, c]
                if color != BLACK and color != WALL:
                    top_keys.append(color)

            # Left-wall key
            if c == min_c and _in_bounds(h, w, r, c - 1):
                color = out[r, c - 1]
                if color != BLACK and color != WALL:
                    left_keys.append(color)

        if not top_keys or not left_keys:
            continue

        top_key = top_keys[0]
        left_key = left_keys[0]

        if top_key not in top_band or left_key not in left_band:
            continue

        tile_h = max(1, left_band[left_key])
        tile_w = max(1, top_band[top_key])

        for r, c in cells:
            dr = r - min_r
            dc = c - min_c
            out[r, c] = top_key if ((dr + dc) % 2 == 0) else left_key

    return out


def _draw_rect_region(grid: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    # Interior: rows [top, top+h-1], cols [left, left+w-1]
    r0 = top - 1
    r1 = top + h
    c0 = left - 1
    c1 = left + w

    grid[r0, left:left + w] = WALL
    grid[r1, left:left + w] = WALL
    grid[top:top + h, c0] = WALL
    grid[top:top + h, c1] = WALL


def _paint_segment(row: np.ndarray, start: int, length: int, color: int) -> None:
    row[start:start + length] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    H = int(rng.integers(14, 31))
    W = int(rng.integers(14, 31))
    grid = np.zeros((H, W), dtype=int)

    palette = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    rng.shuffle(palette)
    color_i = 0

    occupied = np.zeros((H, W), dtype=bool)
    top_used = np.zeros(W, dtype=bool)
    left_used = np.zeros(H, dtype=bool)

    region_count = int(rng.integers(1, 5))

    for _ in range(region_count):
        if color_i + 2 > len(palette):
            break

        tries = 0
        placed = False

        # Randomized shape parameters and key-band sizes to cover multiple branches.
        h_in = int(rng.integers(4, 9))
        w_in = int(rng.integers(4, 9))
        tile_h = int(rng.integers(2, 7))
        tile_w = int(rng.integers(2, 7))

        # Required key colors: C1 is top-band color, C2 is left-band color.
        c1 = palette[color_i]
        c2 = palette[color_i + 1]
        color_i += 2

        # Place top-band segment on row 0.
        possible_top_x: List[Tuple[int, int]] = []
        for x0 in range(1, W - tile_w):
            if x0 + tile_w > W - 1:
                break
            if tile_w > 0 and np.any(top_used[x0:x0 + tile_w]):
                continue
            possible_top_x.append((x0, tile_w))
        if not possible_top_x:
            break
        top_x, _ = possible_top_x[int(rng.integers(0, len(possible_top_x)))]
        _paint_segment(grid[0], top_x, tile_w, c1)
        top_used[top_x:top_x + tile_w] = True

        # Place left-band segment on col 0.
        possible_left_y: List[Tuple[int, int]] = []
        for y0 in range(1, H - tile_h):
            if y0 + tile_h > H - 1:
                break
            if np.any(left_used[y0:y0 + tile_h]):
                continue
            possible_left_y.append((y0, tile_h))
        if not possible_left_y:
            break
        left_y, _ = possible_left_y[int(rng.integers(0, len(possible_left_y)))]
        grid[left_y:left_y + tile_h, 0] = c2
        left_used[left_y:left_y + tile_h] = True

        # Place the enclosed region so that walls can be drawn safely.
        while tries < 120 and not placed:
            tries += 1
            top = int(rng.integers(2, H - h_in - 1))
            left = int(rng.integers(2, W - w_in - 1))

            r0 = top - 1
            r1 = top + h_in
            c0 = left - 1
            c1 = left + w_in

            if r0 < 1 or r1 >= H - 1 or c0 < 1 or c1 >= W - 1:
                continue

            if np.any(occupied[r0:r1 + 1, c0:c1 + 1]):
                continue

            _draw_rect_region(grid, top, left, h_in, w_in)

            # Insert optional outward bumps (irregular shape branch) to avoid uniformity.
            if rng.random() < 0.5:
                bump_len = int(rng.integers(1, 3))
                side = int(rng.integers(0, 4))
                if side == 0 and top >= 3:  # top side bump
                    x0 = int(rng.integers(left, left + max(1, w_in - bump_len + 1)))
                    if not np.any(occupied[top - 2, x0:x0 + bump_len]):
                        grid[top - 2, x0:x0 + bump_len] = WALL
                elif side == 1 and top + h_in + bump_len < H - 1:  # bottom side bump
                    x0 = int(rng.integers(left, left + max(1, w_in - bump_len + 1)))
                    if not np.any(occupied[top + h_in + 1, x0:x0 + bump_len]):
                        grid[top + h_in + 1, x0:x0 + bump_len] = WALL
                elif side == 2 and left >= 3:  # left side bump
                    y0 = int(rng.integers(top, top + max(1, h_in - bump_len + 1)))
                    if not np.any(occupied[y0:y0 + bump_len, left - 2]):
                        grid[y0:y0 + bump_len, left - 2] = WALL
                elif side == 3 and left + w_in + bump_len < W - 1:  # right side bump
                    y0 = int(rng.integers(top, top + max(1, h_in - bump_len + 1)))
                    if not np.any(occupied[y0:y0 + bump_len, left + w_in + 1]):
                        grid[y0:y0 + bump_len, left + w_in + 1] = WALL

            # Place key pixels on top and left walls.
            top_key_col = left + int(rng.integers(0, w_in))
            left_key_row = top + int(rng.integers(0, h_in))
            grid[top - 1, top_key_col] = c1
            grid[left_key_row, left - 1] = c2

            placed = True
            r0 = top - 1
            r1 = top + h_in
            c0 = left - 1
            c1 = left + w_in
            occupied[r0:r1 + 1, c0:c1 + 1] = True

        if not placed:
            # Roll back the last placed top/left bands so they stay self-consistent.
            grid[0, top_x:top_x + tile_w] = BLACK
            top_used[top_x:top_x + tile_w] = False
            grid[left_y:left_y + tile_h, 0] = BLACK
            left_used[left_y:left_y + tile_h] = False

    return grid
