from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
HOLE_VALUES = (0, 1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_MIN = 15
GRID_SIDE_MAX = 30
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_PLACEMENT_TRIES = 120
OBJECT_MARGIN = 1
OUTER_SIZE_MIN = 3
OUTER_SIZE_MAX = 9


def _components_by_color(grid: np.ndarray, bg: int = BG) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append((color, coords))

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_count(grid: np.ndarray, box: tuple[int, int, int, int], bg: int = BG) -> int:
    r0, c0, r1, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return 0
    return int(np.count_nonzero(inner == bg))


def _square_side(hole_count: int) -> int:
    return 2 * int(hole_count) + 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert hole-marked colored rectangles into concentric canonical squares.

    Input structure:
    - Background is black (0).
    - Each foreground object is a monochrome rectangular ring: a 1-cell border with
      the same color filling the rest of the rectangle.
    - The interior may contain 0-4 black hole pixels, and the rectangles do not overlap.

    Rule:
    - Detect each colored rectangle and count the black pixels strictly inside its
      bounding box.
    - Replace the whole object with a solid square of the same color whose side
      length is 2*H+3, where H is that hole count.
    - Sort the squares by H from large to small and place them concentrically on
      one canvas, so each smaller square overwrites the center of the previous one.
    - The output size is exactly the side length of the largest square.

    Key insight:
    - The holes are not output details; they encode the size class of each object,
      and solving the task means turning those counts into a nested-square layout.
    """
    g = np.asarray(grid, dtype=int)
    objects: list[tuple[int, int, int, int]] = []

    for color, coords in _components_by_color(g, bg=BG):
        box = _bbox(coords)
        holes = _hole_count(g, box, bg=BG)
        r0, c0, _r1, _c1 = box
        objects.append((int(holes), int(r0), int(c0), int(color)))

    if not objects:
        return g.copy()

    objects.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    outer_side = _square_side(objects[0][0])
    out = np.full((outer_side, outer_side), BG, dtype=int)

    for holes, _r0, _c0, color in objects:
        side = _square_side(holes)
        start = (outer_side - side) // 2
        out[start : start + side, start : start + side] = int(color)

    return out


def _valid_inner_shapes(hole_count: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for inner_h in range(1, OUTER_SIZE_MAX - 1):
        for inner_w in range(1, OUTER_SIZE_MAX - 1):
            if inner_h + 2 < OUTER_SIZE_MIN or inner_w + 2 < OUTER_SIZE_MIN:
                continue
            if inner_h + 2 > OUTER_SIZE_MAX or inner_w + 2 > OUTER_SIZE_MAX:
                continue
            if inner_h * inner_w <= hole_count:
                continue
            max_sparse = ((inner_h + 1) // 2) * ((inner_w + 1) // 2)
            if max_sparse < hole_count:
                continue
            shapes.append((int(inner_h), int(inner_w)))
    return shapes


def _sample_holes(inner_h: int, inner_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []

    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(80):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((int(r), int(c)))
                if len(chosen) == hole_count:
                    return chosen
    return None


def _make_rectangle(color: int, hole_count: int) -> np.ndarray | None:
    choices = _valid_inner_shapes(hole_count)
    if not choices:
        return None

    non_square = [(h, w) for h, w in choices if h != w]
    if non_square and random.random() < 0.6:
        inner_h, inner_w = random.choice(non_square)
    else:
        inner_h, inner_w = random.choice(choices)

    holes = _sample_holes(inner_h, inner_w, hole_count)
    if holes is None:
        return None

    obj = np.full((inner_h + 2, inner_w + 2), int(color), dtype=int)
    for r, c in holes:
        obj[r + 1, c + 1] = BG
    return obj


def _candidate_positions(
    occupied: np.ndarray,
    obj_h: int,
    obj_w: int,
    margin: int = OBJECT_MARGIN,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - obj_h):
        for c0 in range(1, w - obj_w):
            r1 = r0 + obj_h
            c1 = c0 + obj_w
            rr0 = max(0, r0 - margin)
            cc0 = max(0, c0 - margin)
            rr1 = min(h, r1 + margin)
            cc1 = min(w, c1 + margin)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((int(r0), int(c0)))
    return positions


def _sample_canvas(objects: list[np.ndarray]) -> tuple[int, int]:
    padded_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for obj in objects)
    tallest = max(int(obj.shape[0]) for obj in objects)
    widest = max(int(obj.shape[1]) for obj in objects)
    side_floor = max(tallest, widest) + 3
    area_floor = int(np.ceil(np.sqrt(padded_area))) + 2
    low = max(GRID_SIDE_MIN, side_floor, area_floor)
    high = GRID_SIDE_MAX
    return random.randint(low, high), random.randint(low, high)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = random.sample(HOLE_VALUES, count)
    colors = random.sample(list(range(1, 10)), count)

    objects: list[np.ndarray] = []
    for color, holes in zip(colors, hole_counts):
        obj = _make_rectangle(color, holes)
        if obj is None:
            return None
        objects.append(obj)

    for _ in range(40):
        height, width = _sample_canvas(objects)
        grid = np.full((height, width), BG, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        order = list(range(count))
        random.shuffle(order)
        placed = True
        for idx in order:
            obj = objects[idx]
            candidates = _candidate_positions(occupied, int(obj.shape[0]), int(obj.shape[1]))
            if not candidates:
                placed = False
                break
            r0, c0 = random.choice(candidates)
            r1 = r0 + int(obj.shape[0])
            c1 = c0 + int(obj.shape[1])
            grid[r0:r1, c0:c1] = obj
            occupied[r0:r1, c0:c1] = True

        if not placed:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if out.ndim != 2 or max(out.shape) > 30:
            continue
        if len(_components_by_color(grid, bg=BG)) != count:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
