"""ARC solver and synthetic input generator for puzzle 3.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


BLACK = 0
WALL = 5
AZURE = 8


def _component_bboxes(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Return connected-component bounding boxes as
    (min_row, max_row, min_col, max_col, size).
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    boxes: List[Tuple[int, int, int, int, int]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue

            stack = [(sr, sc)]
            visited[sr, sc] = True
            min_r = max_r = sr
            min_c = max_c = sc
            size = 0

            while stack:
                r, c = stack.pop()
                size += 1
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                nr = r + 1
                nc = c
                if 0 <= nr < h and not visited[nr, nc] and mask[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

                nr = r - 1
                if 0 <= nr < h and not visited[nr, nc] and mask[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

                nc = c + 1
                nr = r
                if 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

                nc = c - 1
                if 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            boxes.append((min_r, max_r, min_c, max_c, size))

    return boxes


def _scale_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize with nearest-neighbor using integer arithmetic."""
    src_h, src_w = arr.shape
    ys = (np.arange(out_h) * src_h) // out_h
    xs = (np.arange(out_w) * src_w) // out_w
    return arr[ys[:, None], xs[None, :]]


def _extract_rooms(grid: np.ndarray) -> List[Tuple[int, int, int, int, Tuple[bool, bool, bool, bool]]]:
    """Locate each 1-cell thick gray room border and its window directions."""
    h, w = grid.shape
    candidates: List[Tuple[int, int, int, int, Tuple[bool, bool, bool, bool]]] = []

    for r0 in range(h - 4):
        for ih in range(3, 7):
            r1 = r0 + ih + 1
            if r1 >= h:
                break
            for c0 in range(w - 4):
                for iw in range(3, 7):
                    c1 = c0 + iw + 1
                    if c1 >= w:
                        break

                    # Keep strict corners as walls for stable room detection.
                    if grid[r0, c0] != WALL or grid[r0, c1] != WALL:
                        continue
                    if grid[r1, c0] != WALL or grid[r1, c1] != WALL:
                        continue

                    top = grid[r0, c0 + 1 : c1]
                    bottom = grid[r1, c0 + 1 : c1]
                    left = grid[r0 + 1 : r1, c0]
                    right = grid[r0 + 1 : r1, c1]
                    if not np.all(np.isin(top, (BLACK, WALL))):
                        continue
                    if not np.all(np.isin(bottom, (BLACK, WALL))):
                        continue
                    if not np.all(np.isin(left, (BLACK, WALL))):
                        continue
                    if not np.all(np.isin(right, (BLACK, WALL))):
                        continue

                    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                    if not np.all(interior == BLACK):
                        continue

                    top_walls = int(np.sum(top == WALL))
                    bot_walls = int(np.sum(bottom == WALL))
                    left_walls = int(np.sum(left == WALL))
                    right_walls = int(np.sum(right == WALL))
                    if top_walls < top.size - 2:
                        continue
                    if bot_walls < bottom.size - 2:
                        continue
                    if left_walls < left.size - 2:
                        continue
                    if right_walls < right.size - 2:
                        continue

                    up = bool(np.any(top == BLACK))
                    down = bool(np.any(bottom == BLACK))
                    left_open = bool(np.any(left == BLACK))
                    right_open = bool(np.any(right == BLACK))
                    candidates.append((r0, r1, c0, c1, (up, down, left_open, right_open)))

    # Remove nested candidates so that one real room generates one hit.
    candidates.sort(key=lambda t: (t[1] - t[0]) * (t[3] - t[2]), reverse=True)
    rooms: List[Tuple[int, int, int, int, Tuple[bool, bool, bool, bool]]] = []
    for cand in candidates:
        r0, r1, c0, c1, _ = cand
        contained = False
        for kept in rooms:
            kr0, kr1, kc0, kc1, _ = kept
            if kr0 <= r0 and r1 <= kr1 and kc0 <= c0 and c1 <= kc1:
                if (kr0, kr1, kc0, kc1) != (r0, r1, c0, c1):
                    contained = True
                    break
        if not contained:
            rooms.append(cand)

    return rooms


def _extract_patterns(grid: np.ndarray) -> Dict[Tuple[bool, bool, bool, bool], np.ndarray]:
    """Extract bottom pattern prototypes and infer their direction signature from azure markers."""
    h, w = grid.shape

    wall_rows = np.argwhere(grid == WALL)
    if wall_rows.size:
        pattern_row_start = int(wall_rows[:, 0].max()) + 1
    else:
        pattern_row_start = h // 2

    mask = (grid != BLACK) & (grid != WALL) & (grid != AZURE)
    if pattern_row_start > 0:
        mask[:pattern_row_start] = False

    patterns: Dict[Tuple[bool, bool, bool, bool], np.ndarray] = {}

    for min_r, max_r, min_c, max_c, _ in _component_bboxes(mask):
        if min_r + 1 > max_r or min_c + 1 > max_c:
            continue
        sample = grid[min_r : max_r + 1, min_c : max_c + 1]

        up = False
        down = False
        left = False
        right = False

        row = min_r - 1
        if 0 <= row < h and np.any(grid[row, min_c : max_c + 1] == AZURE):
            up = True

        row = max_r + 1
        if 0 <= row < h and np.any(grid[row, min_c : max_c + 1] == AZURE):
            down = True

        col = min_c - 1
        if 0 <= col < w and np.any(grid[min_r : max_r + 1, col] == AZURE):
            left = True

        col = max_c + 1
        if 0 <= col < w and np.any(grid[min_r : max_r + 1, col] == AZURE):
            right = True

        key = (up, down, left, right)
        if key not in patterns:
            patterns[key] = sample.copy()

    return patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Fill each gray-walled room interior with the matching sample pattern from the
    bottom palette, keyed by the room's wall-window directions.

    Parameters
    ----------
    grid : np.ndarray
        ARC-like integer image with black(0), gray(5) room borders, azure(8)
        direction markers and bottom color samples.

    Returns
    -------
    np.ndarray
        Output image with interiors replaced by nearest-neighbor-scaled matched
        samples. Walls and window openings remain unchanged.

    Strategy
    --------
    1) Infer candidate 3x3..6x6 inner rectangles from gray walls and infer
       each room's (up,down,left,right) window signature from wall gaps.
    2) Parse pattern tiles in the bottom area and infer their direction
       signatures from nearby azure markers.
    3) Match room signature to pattern signature and scale the sample to the
       room interior using nearest-neighbor indexing.
    """
    out = np.array(grid, dtype=int, copy=True)
    rooms = _extract_rooms(out)
    patterns = _extract_patterns(out)

    for min_r, max_r, min_c, max_c, key in rooms:
        if key not in patterns:
            continue

        patch = patterns[key]
        target_h = max_r - min_r - 1
        target_w = max_c - min_c - 1
        if target_h <= 0 or target_w <= 0:
            continue

        scaled = _scale_nearest(patch, target_h, target_w)
        out[min_r + 1 : max_r, min_c + 1 : max_c] = scaled

    return out


def generate_input() -> np.ndarray:
    """Generate one random valid instance for quick checks and diversity."""
    rng = np.random.default_rng()
    h = int(rng.integers(22, 30))
    w = int(rng.integers(22, 30))
    grid = np.zeros((h, w), dtype=int)

    palette = [1, 2, 3, 4, 6, 7, 9]
    all_keys = [(bool(i & 8), bool(i & 4), bool(i & 2), bool(i & 1)) for i in range(1, 16)]
    rng.shuffle(all_keys)

    pattern_count = int(rng.integers(3, min(6, len(all_keys) + 1)))
    selected_keys = all_keys[:pattern_count]

    # Palette rows: create sample strips at the bottom.
    library_top = h - 8
    if library_top < 12:
        library_top = 12

    next_color_index = 0
    pat_x = 1

    for key in selected_keys:
        ph = int(rng.integers(2, 5))
        pw = int(rng.integers(2, 5))
        color = palette[next_color_index % len(palette)]
        next_color_index += 1

        if pat_x + pw + 3 >= w - 1:
            break

        top = library_top + 1
        bottom = top + ph - 1
        left = pat_x
        right = left + pw - 1

        grid[top : bottom + 1, left : right + 1] = color

        cy = (top + bottom) // 2
        cx = (left + right) // 2

        if key[0] and top - 1 >= 0:
            grid[top - 1, cx] = AZURE
        if key[1] and bottom + 1 < h:
            grid[bottom + 1, cx] = AZURE
        if key[2] and left - 1 >= 0:
            grid[cy, left - 1] = AZURE
        if key[3] and right + 1 < w:
            grid[cy, right + 1] = AZURE

        pat_x = right + 3

    # Place random rooms in the area above the palette.
    room_count = int(rng.integers(3, min(8, pattern_count + 3)))
    occ = grid.copy()
    attempts = 0

    for _ in range(room_count):
        key = selected_keys[int(rng.integers(0, len(selected_keys)))]
        ih = int(rng.integers(3, 7))
        iw = int(rng.integers(3, 7))
        # room border size
        bh = ih + 2
        bw = iw + 2

        placed = False
        while attempts < 500:
            attempts += 1
            r0 = int(rng.integers(1, max(2, library_top - bh)))
            c0 = int(rng.integers(1, max(2, w - bw - 1)))
            r1 = r0 + bh - 1
            c1 = c0 + bw - 1

            if r1 >= library_top or c1 >= w - 1:
                continue
            if np.any(occ[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
                continue

            # Draw the 1-cell border.
            occ[r0, c0 : c1 + 1] = WALL
            occ[r1, c0 : c1 + 1] = WALL
            occ[r0 : r1 + 1, c0] = WALL
            occ[r0 : r1 + 1, c1] = WALL
            occ[r0 + 1 : r1, c0 + 1 : c1] = BLACK

            # Open windows according to direction key.
            top_pos = list(range(c0 + 1, c1))
            bot_pos = list(range(c0 + 1, c1))
            left_pos = list(range(r0 + 1, r1))
            right_pos = list(range(r0 + 1, r1))

            if key[0]:
                k = min(len(top_pos), int(rng.integers(1, max(2, min(3, len(top_pos) + 1)))))
                for c in rng.choice(top_pos, size=k, replace=False):
                    occ[r0, c] = BLACK
            if key[1]:
                k = min(len(bot_pos), int(rng.integers(1, max(2, min(3, len(bot_pos) + 1)))))
                for c in rng.choice(bot_pos, size=k, replace=False):
                    occ[r1, c] = BLACK
            if key[2]:
                k = min(len(left_pos), int(rng.integers(1, max(2, min(3, len(left_pos) + 1)))))
                for r in rng.choice(left_pos, size=k, replace=False):
                    occ[r, c0] = BLACK
            if key[3]:
                k = min(len(right_pos), int(rng.integers(1, max(2, min(3, len(right_pos) + 1)))))
                for r in rng.choice(right_pos, size=k, replace=False):
                    occ[r, c1] = BLACK

            grid = occ
            placed = True
            break

    return grid


if __name__ == "__main__":
    sample = generate_input()
    solved = main(sample)
    print("grid", sample.shape, "changed", int(np.sum(sample != solved)))
