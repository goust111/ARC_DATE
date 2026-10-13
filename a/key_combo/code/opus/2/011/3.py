import random
from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _components_of_color(grid: np.ndarray, color: int) -> List[List[Tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
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
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Detect each blue or cyan rectangular carrier and the unique red object
        fully embedded inside its border.
    Analyze:
        Slide the red object to the carrier wall implied by the carrier color
        and record the final red-object bounding-box top-left anchor.
    Transform:
        Repaint the vacated carrier pixels, place the slid red object, and fill
        each carrier interior with an anchor-relative checkerboard carve-out.
    Return:
        Emit the updated carrier rectangles with unchanged borders, shifted red
        objects, and interior black cells created by the parity rule.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    carriers = []
    for color in (1, 8):
        for comp in _components_of_color(grid, color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            carriers.append(
                {
                    "color": color,
                    "bbox": (top, bottom, left, right),
                }
            )

    red_components = _components_of_color(grid, 2)
    red_lookup: Dict[int, List[Tuple[int, int]]] = {}
    for idx, carrier in enumerate(carriers):
        top, bottom, left, right = carrier["bbox"]
        for comp in red_components:
            if all(top < r < bottom and left < c < right for r, c in comp):
                red_lookup[idx] = comp
                break
        if idx not in red_lookup:
            raise ValueError("missing red object inside carrier")

    final_reds: Dict[int, List[Tuple[int, int]]] = {}
    for idx, carrier in enumerate(carriers):
        color = carrier["color"]
        top, bottom, left, right = carrier["bbox"]
        red_cells = red_lookup[idx]
        red_rows = [r for r, _ in red_cells]
        red_cols = [c for _, c in red_cells]
        box_left, box_right = min(red_cols), max(red_cols)
        target_left = right - 1 - (box_right - box_left) if color == 1 else left + 1
        shift = target_left - box_left
        for r, c in red_cells:
            out[r, c] = color
        shifted = [(r, c + shift) for r, c in red_cells]
        for r, c in shifted:
            out[r, c] = 2
        final_reds[idx] = shifted

    for idx, carrier in enumerate(carriers):
        color = carrier["color"]
        top, bottom, left, right = carrier["bbox"]
        red_cells = set(final_reds[idx])
        red_rows = [r for r, _ in red_cells]
        red_cols = [c for _, c in red_cells]
        anchor_r, anchor_c = min(red_rows), min(red_cols)
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if (r, c) in red_cells:
                    out[r, c] = 2
                elif (r - anchor_r + c - anchor_c) % 2 == 0:
                    out[r, c] = 0
                else:
                    out[r, c] = color
    return out


_RED_SHAPES: Sequence[Sequence[Tuple[int, int]]] = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def generate_input() -> np.ndarray:
    h = random.randint(14, 16)
    w = 16
    grid = np.zeros((h, w), dtype=int)
    slots = [
        (1, 6, 1, 6),
        (1, 6, 9, 14),
        (8, 13, 2, 7),
        (8, 13, 9, 14),
    ]
    random.shuffle(slots)
    carrier_count = random.randint(2, 4)

    for top, bottom, left, right in slots[:carrier_count]:
        color = random.choice([1, 8])
        grid[top : bottom + 1, left : right + 1] = color
        shape = random.choice(_RED_SHAPES)
        shape_rows = [r for r, _ in shape]
        shape_cols = [c for _, c in shape]
        shape_h = max(shape_rows) + 1
        shape_w = max(shape_cols) + 1
        start_r = random.randint(top + 1, bottom - shape_h)
        start_c = random.randint(left + 1, right - shape_w)
        for r, c in shape:
            grid[start_r + r, start_c + c] = 2
    return grid
