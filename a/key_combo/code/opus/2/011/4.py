import random
from collections import deque
from typing import List, Sequence, Tuple

import numpy as np


def _components(grid: np.ndarray) -> List[Tuple[int, List[Tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, List[Tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, comp))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Extract every colored 4-connected sprite and record the top row of its
        tight bounding box.
    Analyze:
        Sort sprites by descending top-row value so the lowest sprite receives
        rank 0, the next lowest rank 1, and so on.
    Transform:
        Shear each ranked sprite rightward inside its local rows by rank times
        row index, then crop the sheared sprite to its new tight box.
    Return:
        Stack the sheared sprites from rank 0 downward in a left-aligned output
        grid sized by the maximum sheared width and total stacked height.
    """
    grid = np.asarray(grid, dtype=int)
    sprites = []
    for color, comp in _components(grid):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        sprites.append(
            {
                "color": color,
                "coords": comp,
                "top": top,
                "bbox": (top, bottom, left, right),
            }
        )
    sprites.sort(key=lambda item: item["top"], reverse=True)

    rendered = []
    max_width = 0
    total_height = 0
    for rank, sprite in enumerate(sprites):
        top, bottom, left, right = sprite["bbox"]
        height = bottom - top + 1
        width = right - left + 1
        extra = rank * (height - 1)
        canvas = np.zeros((height, width + extra), dtype=int)
        for r, c in sprite["coords"]:
            local_r = r - top
            local_c = c - left + rank * local_r
            canvas[local_r, local_c] = sprite["color"]
        rendered.append(canvas)
        max_width = max(max_width, canvas.shape[1])
        total_height += canvas.shape[0]

    out = np.zeros((total_height, max_width), dtype=int)
    row = 0
    for canvas in rendered:
        out[row : row + canvas.shape[0], : canvas.shape[1]] = canvas
        row += canvas.shape[0]
    return out


_SHAPES: Sequence[Sequence[Tuple[int, int]]] = (
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)


def _shape_dims(shape: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def generate_input() -> np.ndarray:
    h = random.randint(10, 12)
    w = random.randint(10, 12)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], random.randint(3, 4))
    top_rows = random.sample(range(0, h - 3), len(colors))
    top_rows.sort()
    placed: List[Tuple[int, int]] = []

    for color, top in zip(colors, top_rows):
        for _ in range(200):
            shape = random.choice(_SHAPES)
            shape_h, shape_w = _shape_dims(shape)
            left = random.randint(0, w - shape_w)
            coords = [(top + r, left + c) for r, c in shape]
            if any((r, c) in placed for r, c in coords):
                continue
            for r, c in coords:
                grid[r, c] = color
                placed.append((r, c))
            break
    return grid
