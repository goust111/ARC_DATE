from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GEN_ATTEMPTS = 500
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _blue_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    red_cells = np.argwhere(grid == RED)
    assert len(red_cells) == 1, "expected one red center"
    cr, cc = map(int, red_cells[0])
    blocks = []
    for comp in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
            continue
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        holes = int(np.sum(crop == BLACK))
        if holes <= 0:
            continue
        center_dist = ((r0 + r1) / 2 - cr) ** 2 + ((c0 + c1) / 2 - cc) ** 2
        blocks.append({"box": (r0, c0, r1, c1), "holes": holes, "dist": center_dist})
    assert len(blocks) >= 4, "expected four blue blocks with holes"
    blocks.sort(key=lambda item: (float(item["dist"]), item["box"]))
    return blocks[:4]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count holes in four blue blocks by center distance, then use those lengths to drive a colored spiral.
    """
    g = np.asarray(grid, dtype=int)
    red_cells = np.argwhere(g == RED)
    assert len(red_cells) == 1, "expected one red center marker"
    cr, cc = map(int, red_cells[0])

    blocks = _blue_blocks(g)
    lengths = [int(block["holes"]) for block in blocks]
    out = g.copy()

    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    r, c = cr, cc
    step_index = 0
    while True:
        if step_index < 4:
            length = lengths[step_index]
        else:
            length = lengths[-1] + (step_index - 3)
        dr, dc = directions[step_index % 4]
        paint = ORANGE if dc else GREEN
        for _ in range(length):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                return out
            r, c = nr, nc
            if out[r, c] not in (BLUE, RED):
                out[r, c] = paint
        step_index += 1


def _draw_block(grid: np.ndarray, r0: int, c0: int, size: int, holes: int) -> None:
    grid[r0 : r0 + size, c0 : c0 + size] = BLUE
    interior = [(r, c) for r in range(r0 + 1, r0 + size - 1) for c in range(c0 + 1, c0 + size - 1)]
    for r, c in random.sample(interior, holes):
        grid[r, c] = BLACK


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(17, 23)
        width = random.randint(17, 23)
        grid = np.full((height, width), BLACK, dtype=int)
        grid[0, 0] = ORANGE
        grid[0, 1] = GREEN

        cr = random.randint(height // 2 - 1, height // 2 + 1)
        cc = random.randint(width // 2 - 1, width // 2 + 1)
        grid[cr, cc] = RED

        size = random.choice((5, 6))
        anchors = [
            (1, random.randint(2, max(2, width // 4))),
            (1, random.randint(width - size - max(2, width // 4), width - size - 1)),
            (random.randint(height - size - max(2, height // 4), height - size - 1), 1),
            (random.randint(height - size - max(2, height // 4), height - size - 1), random.randint(width - size - max(2, width // 4), width - size - 1)),
        ]
        if len(set(anchors)) < 4:
            continue
        hole_counts = random.sample([1, 2, 3, 4, 5], 4)
        random.shuffle(hole_counts)
        for (r0, c0), holes in zip(anchors, hole_counts):
            if r0 < 0 or c0 < 0 or r0 + size > height or c0 + size > width:
                break
            _draw_block(grid, r0, c0, size, holes)
        else:
            if _valid_generated(grid):
                return grid
    raise RuntimeError("failed to generate puzzle 030/8 input")

