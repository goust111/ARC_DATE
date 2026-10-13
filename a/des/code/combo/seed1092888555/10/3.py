import random
from collections import Counter

import numpy as np


BG = 8
FRAME_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
OBJECT_COUNT_RANGE = (2, 5)
FRAME_SIZE_RANGE = (3, 9)
PLACEMENT_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0 = int(pts[:, 0].min())
            r1 = int(pts[:, 0].max())
            c0 = int(pts[:, 1].min())
            c1 = int(pts[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def _is_square_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or not (FRAME_SIZE_RANGE[0] <= side_h <= FRAME_SIZE_RANGE[1]):
        return False
    if len(comp["coords"]) != 4 * side_h - 4:
        return False

    patch = grid[r0:r1 + 1, c0:c1 + 1]
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return False
    if side_h > 2 and not np.all(patch[1:-1, 1:-1] == bg):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow square frame based on the parity of its side length.

    Odd side length:
    - draw a same-color cross on the interior center row and center column.

    Even side length:
    - draw a same-color checkerboard on the interior cells with even local parity.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else _dominant_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one frame"

    out = g.copy()
    for comp in components:
        assert _is_square_frame(g, comp, bg), "all foreground objects must be 1-pixel-thick square frames"
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        side = r1 - r0 + 1

        if side % 2 == 1:
            cr = r0 + side // 2
            cc = c0 + side // 2
            out[cr, c0 + 1:c1] = color
            out[r0 + 1:r1, cc] = color
        else:
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    if ((rr - (r0 + 1)) + (cc - (c0 + 1))) % 2 == 0:
                        out[rr, cc] = color

    return out


def _can_place(occ: np.ndarray, r0: int, c0: int, side: int) -> bool:
    h, w = occ.shape
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    if r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - PLACEMENT_MARGIN)
    cc0 = max(0, c0 - PLACEMENT_MARGIN)
    rr1 = min(h, r1 + PLACEMENT_MARGIN + 1)
    cc1 = min(w, c1 + PLACEMENT_MARGIN + 1)
    return not occ[rr0:rr1, cc0:cc1].any()


def _paint_frame(grid: np.ndarray, occ: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color
    occ[r0:r1 + 1, c0:c1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    count = rng.randint(*OBJECT_COUNT_RANGE)
    odd_slots = rng.randint(1, count - 1)
    even_slots = count - odd_slots

    odd_sizes = [s for s in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1) if s % 2 == 1]
    even_sizes = [s for s in range(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1] + 1) if s % 2 == 0]
    sizes = [rng.choice(odd_sizes) for _ in range(odd_slots)] + [rng.choice(even_sizes) for _ in range(even_slots)]
    rng.shuffle(sizes)

    colors = rng.sample(FRAME_COLORS, count)
    order = sorted(range(count), key=lambda i: sizes[i], reverse=True)

    for idx in order:
        side = sizes[idx]
        color = colors[idx]
        placed = False
        for _ in range(120):
            r0 = rng.randint(0, h - side)
            c0 = rng.randint(0, w - side)
            if _can_place(occ, r0, c0, side):
                _paint_frame(grid, occ, r0, c0, side, color)
                placed = True
                break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
