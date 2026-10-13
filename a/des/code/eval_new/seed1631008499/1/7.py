from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRECTIONS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (2, 4)
COPY_COUNT_MAX = 4
PLACEMENT_ATTEMPTS = 80
GEN_ATTEMPTS = 400
SHAPE_BOXES = ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3))
COLORS = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _random_polyomino() -> tuple[list[tuple[int, int]], int, int]:
    box_h, box_w = random.choice(SHAPE_BOXES)
    area = random.randint(3, min(6, box_h * box_w))
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    norm, h, w = _normalize(cells)
    if len(norm) < 3:
        return _random_polyomino()
    return norm, h, w


def _side_cells(shape: list[tuple[int, int]], direction: str, h: int, w: int) -> list[tuple[int, int]]:
    if direction == "right":
        edge = w - 1
        return [(r, c) for r, c in shape if c == edge]
    if direction == "left":
        return [(r, c) for r, c in shape if c == 0]
    if direction == "down":
        edge = h - 1
        return [(r, c) for r, c in shape if r == edge]
    return [(r, c) for r, c in shape if r == 0]


def _copy_layout(
    shape: list[tuple[int, int]],
    anchor: tuple[int, int],
    direction: str,
    copies: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], tuple[int, int], set[tuple[int, int]]]:
    dr, dc = DIR_VEC[direction]
    step = w if dc != 0 else h
    ar, ac = anchor
    seed = (ar + dr * (copies + 1), ac + dc * (copies + 1))

    input_cells = set(shape)
    input_cells.add(seed)

    reserve = set(shape)
    for idx in range(copies):
        shift_r = seed[0] - ar + idx * dr * step
        shift_c = seed[1] - ac + idx * dc * step
        for r, c in shape:
            reserve.add((r + shift_r, c + shift_c))
    return input_cells, seed, reserve


def _buffered(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _place_object(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    color: int,
    direction: str,
) -> bool:
    h_grid, w_grid = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        shape, h, w = _random_polyomino()
        step = w if direction in ("left", "right") else h
        min_copies = max(1, step - 1)
        copies = random.randint(min_copies, COPY_COUNT_MAX)
        anchor = random.choice(_side_cells(shape, direction, h, w))
        input_cells, _seed, reserve = _copy_layout(shape, anchor, direction, copies, h, w)

        all_cells = input_cells | reserve
        min_r = min(r for r, _ in all_cells)
        max_r = max(r for r, _ in all_cells)
        min_c = min(c for _, c in all_cells)
        max_c = max(c for _, c in all_cells)
        row_lo = -min_r
        row_hi = h_grid - 1 - max_r
        col_lo = -min_c
        col_hi = w_grid - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(PLACEMENT_ATTEMPTS):
            off_r = random.randint(row_lo, row_hi)
            off_c = random.randint(col_lo, col_hi)
            shifted_input = {(r + off_r, c + off_c) for r, c in input_cells}
            shifted_reserve = {(r + off_r, c + off_c) for r, c in reserve}
            if any((r, c) in blocked for r, c in shifted_reserve):
                continue
            for r, c in shifted_input:
                grid[r, c] = color
            blocked.update(_buffered(shifted_reserve))
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clone each monochrome template outward from its matching singleton seed.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly one small 4-connected template component
      of area 3-6 and one isolated singleton seed of the same color.
    - Every seed sits strictly above, below, left, or right of its template, sharing a
      row or column with an exposed boundary pixel and separated from it only by black cells.
    - Different colors define independent template-seed pairs.

    Rule:
    - Pair each singleton seed with the larger same-color component.
    - The seed's side relative to the template determines the growth direction.
    - On the seed-template sightline, the first template pixel reached from the seed is the
      anchor pixel; the number of black cells between that anchor and the seed is the copy count.
    - Keep the original template, then stamp that many same-color copies with unchanged
      orientation, placing the first copy so its anchor lands on the seed and spacing later
      copies by one template bbox width or height along the growth direction.
    - The seed is overwritten by the anchor pixel of the first clone.

    Key insight:
    - Color matches the seed to its template, while the seed's offset simultaneously encodes
      both the clone direction and how many tightly packed copies must be grown.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h_grid, w_grid = g.shape
    out = g.copy()

    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue

        comps = _component_cells(g == color)
        seeds = [comp for comp in comps if len(comp) == 1]
        templates = [comp for comp in comps if len(comp) > 1]
        assert len(seeds) == 1 and len(templates) == 1, "each color needs one seed and one template"

        seed_r, seed_c = seeds[0][0]
        template = templates[0]
        rows = [r for r, _ in template]
        cols = [c for _, c in template]
        rmin, rmax = min(rows), max(rows)
        cmin, cmax = min(cols), max(cols)
        height = rmax - rmin + 1
        width = cmax - cmin + 1

        candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        if seed_c > cmax:
            row_cells = [(r, c) for r, c in template if r == seed_r]
            if row_cells:
                anchor = max(row_cells, key=lambda rc: rc[1])
                gap = seed_c - anchor[1] - 1
                if gap >= 1:
                    candidates.append((anchor, (0, 1), gap))
        if seed_c < cmin:
            row_cells = [(r, c) for r, c in template if r == seed_r]
            if row_cells:
                anchor = min(row_cells, key=lambda rc: rc[1])
                gap = anchor[1] - seed_c - 1
                if gap >= 1:
                    candidates.append((anchor, (0, -1), gap))
        if seed_r > rmax:
            col_cells = [(r, c) for r, c in template if c == seed_c]
            if col_cells:
                anchor = max(col_cells, key=lambda rc: rc[0])
                gap = seed_r - anchor[0] - 1
                if gap >= 1:
                    candidates.append((anchor, (1, 0), gap))
        if seed_r < rmin:
            col_cells = [(r, c) for r, c in template if c == seed_c]
            if col_cells:
                anchor = min(col_cells, key=lambda rc: rc[0])
                gap = anchor[0] - seed_r - 1
                if gap >= 1:
                    candidates.append((anchor, (-1, 0), gap))

        assert len(candidates) == 1, "seed-template relation must determine one direction"
        anchor, (dr, dc), copies = candidates[0]
        step = width if dc != 0 else height

        for idx in range(copies):
            shift_r = seed_r - anchor[0] + idx * dr * step
            shift_c = seed_c - anchor[1] + idx * dc * step
            for r, c in template:
                rr = r + shift_r
                cc = c + shift_c
                assert 0 <= rr < h_grid and 0 <= cc < w_grid, "clone must stay inside the grid"
                out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h_grid = random.randint(GRID_MIN, GRID_MAX)
        w_grid = random.randint(GRID_MIN, GRID_MAX)
        grid = np.full((h_grid, w_grid), BG, dtype=int)
        blocked: set[tuple[int, int]] = set()

        obj_count = random.randint(*OBJECT_COUNT_RANGE)
        colors = random.sample(COLORS, obj_count)
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        directions = dirs[:obj_count]

        ok = True
        for color, direction in zip(colors, directions):
            if not _place_object(grid, blocked, color, direction):
                ok = False
                break
        if not ok:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
