import random
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(comp)
    return groups


def _find_key(grid: np.ndarray) -> Tuple[int, int]:
    h, w = grid.shape
    valid = {1, 2, 3, 4}
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            vals = block.ravel().tolist()
            if set(vals) == valid and len(vals) == 4:
                return r, c
    raise ValueError("color key not found")


def _quadrant(row: int, col: int, active_height: int, width: int) -> str:
    center_row = (active_height - 1) // 2
    center_col = (width - 1) // 2
    upper = row <= center_row
    left = col <= center_col
    if upper and left:
        return "lu"
    if upper and not left:
        return "ru"
    if not upper and left:
        return "ld"
    return "rd"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Read the full gray ground row, the unique 2x2 four-color key, the
        large single-color objects, and the fixed orange marker pixels.
    Analyze:
        Remove the key, settle the large objects under gravity from bottommost
        to topmost, and compute each settled object's integer-floor centroid.
    Transform:
        Use the settled centroid quadrant to look up a replacement color from
        the erased key and recolor the whole settled object accordingly.
    Return:
        Output the modified grid with the ground and orange markers preserved,
        the key erased, and every large object recolored after falling.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    ground_row = next(r for r in range(h - 1, -1, -1) if np.all(grid[r] == 5))
    key_r, key_c = _find_key(grid)
    key_colors: Dict[str, int] = {
        "lu": int(grid[key_r, key_c]),
        "ru": int(grid[key_r, key_c + 1]),
        "ld": int(grid[key_r + 1, key_c]),
        "rd": int(grid[key_r + 1, key_c + 1]),
    }

    key_cells = {(key_r + dr, key_c + dc) for dr in range(2) for dc in range(2)}
    object_mask = np.isin(grid, [1, 2, 3, 4])
    for r, c in key_cells:
        object_mask[r, c] = False

    groups = []
    used = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not object_mask[r, c] or used[r, c]:
                continue
            color = grid[r, c]
            q = deque([(r, c)])
            used[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and object_mask[nr, nc]
                        and not used[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        used[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) > 2:
                groups.append({"coords": comp, "color": int(color)})

    groups.sort(key=lambda item: max(r for r, _ in item["coords"]), reverse=True)

    settled_mask = np.zeros((h, w), dtype=bool)
    out = grid.copy()
    out[key_r : key_r + 2, key_c : key_c + 2] = 0
    for group in groups:
        for r, c in group["coords"]:
            out[r, c] = 0

    settled_groups = []
    for group in groups:
        coords = group["coords"]
        shift = 0
        while True:
            blocked = False
            for r, c in coords:
                nr = r + shift + 1
                if nr == ground_row or settled_mask[nr, c]:
                    blocked = True
                    break
            if blocked:
                break
            shift += 1
        final_coords = [(r + shift, c) for r, c in coords]
        for r, c in final_coords:
            settled_mask[r, c] = True
        settled_groups.append({"coords": final_coords, "color": group["color"]})

    for group in settled_groups:
        rows = [r for r, _ in group["coords"]]
        cols = [c for _, c in group["coords"]]
        centroid_r = sum(rows) // len(rows)
        centroid_c = sum(cols) // len(cols)
        new_color = key_colors[_quadrant(centroid_r, centroid_c, ground_row, w)]
        for r, c in group["coords"]:
            out[r, c] = new_color

    out[grid == 7] = 7
    return out


_SHAPES: Sequence[Sequence[Tuple[int, int]]] = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
)


def _transform_shape(shape: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cells = list(shape)
    if random.random() < 0.5:
        max_r = max(r for r, _ in cells)
        cells = [(max_r - r, c) for r, c in cells]
    if random.random() < 0.5:
        max_c = max(c for _, c in cells)
        cells = [(r, max_c - c) for r, c in cells]
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def generate_input() -> np.ndarray:
    h = 13
    w = 13
    ground_row = h - 1
    left_starts = [3, 5]
    right_starts = [8, 10]

    for _ in range(5000):
        grid = np.zeros((h, w), dtype=int)
        grid[ground_row, :] = 5
        grid[0, 0] = 1
        grid[0, 1] = 2
        grid[1, 0] = 3
        grid[1, 1] = 4

        spans = []
        success = True
        for start_col in left_starts + right_starts:
            shape = _transform_shape(random.choice(_SHAPES))
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            top = random.randint(2, 7 - max_r)
            color_choices = [1, 2, 4] if start_col in left_starts else [1, 2, 3]
            color = random.choice(color_choices)
            for r, c in shape:
                rr, cc = top + r, start_col + c
                if rr >= ground_row or cc >= w or grid[rr, cc] != 0:
                    success = False
                    break
                grid[rr, cc] = color
            if not success:
                break
            spans.append((start_col, start_col + max_c))
        if not success:
            continue

        free_cols = [
            c
            for c in range(w)
            if all(not (start <= c <= end) for start, end in spans) and c > 1
        ]
        if len(free_cols) < 2:
            continue
        marker_count = random.randint(2, min(3, len(free_cols)))
        random.shuffle(free_cols)
        for idx in range(marker_count):
            grid[random.choice([ground_row - 1, ground_row - 2]), free_cols[idx]] = 7

        out = main(grid)
        changed = np.argwhere((grid > 0) & np.isin(grid, [1, 2, 3, 4]))
        if changed.size == 0:
            continue
        if np.any(grid != out):
            return grid
    raise RuntimeError("failed to generate a valid task-2 input")
