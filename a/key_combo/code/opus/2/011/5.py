import random
from collections import Counter, deque
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _components8(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Find the solid red vertical mirror axis, the yellow sprites on its left,
        and the colored landing markers inside the reflected zones.
    Analyze:
        Reflect every yellow sprite across the axis, then read the unique marker
        color inside each reflected landing zone.
    Transform:
        Conceptually repaint each reflected sprite with its landing marker color
        and count how many reflected sprites use each resulting color.
    Return:
        Emit the sorted frequency-bar grid whose row colors and bar lengths come
        from the reflected color counts rather than from the original layout.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    axis_col = next(c for c in range(w) if np.all(grid[:, c] == 2))

    sprites = []
    left_mask = (grid == 4).copy()
    for c in range(axis_col, w):
        left_mask[:, c] = False
    for comp in _components8(left_mask):
        reflected = [(r, 2 * axis_col - c) for r, c in comp]
        marker_colors = {int(grid[r, c]) for r, c in reflected if grid[r, c] not in (0, 2, 4)}
        if len(marker_colors) != 1:
            raise ValueError("each reflected zone must contain exactly one marker color")
        paint_color = marker_colors.pop()
        sprites.append({"paint": paint_color, "zone": reflected})

    freq = Counter(sprite["paint"] for sprite in sprites)
    first_col: Dict[int, int] = {}
    for color in freq:
        cols = np.argwhere(grid == color)
        first_col[color] = int(cols[:, 1].min())

    order = sorted(freq, key=lambda color: (freq[color], first_col[color]))
    width = max(freq.values())
    out = np.zeros((len(order), width), dtype=int)
    for r, color in enumerate(order):
        out[r, : freq[color]] = color
    return out


_SPRITES: Sequence[Sequence[Tuple[int, int]]] = (
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1)),
    ((0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)


def generate_input() -> np.ndarray:
    h = 14
    w = 14
    for _ in range(5000):
        grid = np.zeros((h, w), dtype=int)
        axis_col = random.choice([6, 7])
        grid[:, axis_col] = 2
        candidate_rows = [0, 3, 6, 9, 12]
        rows = sorted(random.sample(candidate_rows, random.randint(4, 5)))
        ok = True
        for top in rows:
            shape = random.choice(_SPRITES)
            left = axis_col - 2
            coords = [(top + r, left + c) for r, c in shape]
            if any(c >= axis_col or grid[r, c] != 0 for r, c in coords):
                ok = False
                break
            for r, c in coords:
                grid[r, c] = 4
            reflected = [(r, 2 * axis_col - c) for r, c in coords]
            marker_r, marker_c = random.choice(reflected)
            grid[marker_r, marker_c] = random.choice([1, 3, 5, 7])
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid task-5 input")
