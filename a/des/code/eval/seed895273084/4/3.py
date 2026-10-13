from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
SEED = 2
GREEN = 3
YELLOW = 4
OUTLINE = 5
MAGENTA = 6
ORANGE = 7

LAYER_COLORS = (BLUE, GREEN, YELLOW, MAGENTA, ORANGE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
OBJECT_COUNT_VALUES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (2, 4, 3, 1)
MAX_SEEDS = 5
EDGE_MARGIN = 1
OBJECT_GAP = 2
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)

    return comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _layer_distances_from_outline(
    grid: np.ndarray, cells: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    region = set(cells)
    dist: dict[tuple[int, int], int] = {}
    q = deque()

    for r, c in cells:
        if any(int(grid[nr, nc]) == OUTLINE for nr, nc in _neighbors4(r, c, h, w)):
            dist[(r, c)] = 1
            q.append((r, c))

    assert dist, "enclosed region must touch the outline"

    while q:
        r, c = q.popleft()
        nd = dist[(r, c)] + 1
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in region and (nr, nc) not in dist:
                dist[(nr, nc)] = nd
                q.append((nr, nc))

    assert len(dist) == len(region), "region interior must be 4-connected"
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color inward layers inside each grey outline; the number of layers equals the
    number of red seeds inside that enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    out = g.copy()
    passable = (g == BG) | (g == SEED)

    for cells in _components(passable):
        if _touches_edge(cells, h, w):
            continue

        if not any(
            int(g[nr, nc]) == OUTLINE
            for r, c in cells
            for nr, nc in _neighbors4(r, c, h, w)
        ):
            continue

        seed_cells = [(r, c) for r, c in cells if int(g[r, c]) == SEED]
        n_layers = len(seed_cells)
        assert n_layers <= len(LAYER_COLORS), "seed count exceeds available colors"

        distances = _layer_distances_from_outline(g, cells)
        for (r, c), depth in distances.items():
            if 1 <= depth <= n_layers:
                out[r, c] = LAYER_COLORS[depth - 1]

        for r, c in seed_cells:
            out[r, c] = BG

    return out


def _shape_parts(shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(shape[r, c]):
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not bool(shape[nr, nc]):
                    boundary[r, c] = True
                    break
    interior = shape & (~boundary)
    return boundary, interior


def _region_distances(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=int)
    q = deque()

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                    dist[r, c] = 1
                    q.append((r, c))
                    break

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(mask[nr, nc]) and int(dist[nr, nc]) == 0:
                dist[nr, nc] = nd
                q.append((nr, nc))

    return dist


def _make_rectangle(seed_count: int) -> np.ndarray:
    min_side = max(5, 2 * seed_count + 1)
    h = random.randint(min_side, 14)
    w = random.randint(min_side, 14)
    return np.ones((h, w), dtype=bool)


def _make_l_shape(seed_count: int) -> np.ndarray | None:
    min_total = max(7, 2 * seed_count + 3)
    min_arm = 2 * seed_count + 1
    if min_total > 14:
        return None

    h = random.randint(min_total, 14)
    w = random.randint(min_total, 14)
    if min_arm > h - 2 or min_arm > w - 2:
        return None

    arm_w = random.randint(min_arm, w - 2)
    arm_h = random.randint(min_arm, h - 2)
    shape = np.zeros((h, w), dtype=bool)
    shape[:, :arm_w] = True
    shape[h - arm_h :, :] = True
    return np.rot90(shape, random.randint(0, 3)).copy()


def _sample_object(seed_count: int) -> dict[str, object] | None:
    for _ in range(SHAPE_RETRIES):
        kind = random.choice(("rect", "l"))
        if kind == "rect":
            shape = _make_rectangle(seed_count)
        else:
            shape = _make_l_shape(seed_count)
            if shape is None:
                continue

        boundary, interior = _shape_parts(shape)
        if int(np.sum(interior)) < seed_count:
            continue
        if len(_components(interior)) != 1:
            continue

        dist = _region_distances(interior)
        max_depth = int(dist.max())
        if max_depth < seed_count:
            continue

        interior_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        deep_cells = [
            tuple(map(int, rc))
            for rc in np.argwhere(interior & (dist > seed_count))
        ]
        seed_pool = deep_cells if len(deep_cells) >= seed_count else interior_cells
        seeds = random.sample(seed_pool, seed_count)

        return {
            "boundary": boundary,
            "seeds": seeds,
            "height": int(shape.shape[0]),
            "width": int(shape.shape[1]),
        }

    return None


def _place_object(
    grid: np.ndarray, blocked: np.ndarray, obj: dict[str, object]
) -> bool:
    h, w = grid.shape
    obj_h = int(obj["height"])
    obj_w = int(obj["width"])
    if obj_h + 2 * EDGE_MARGIN > h or obj_w + 2 * EDGE_MARGIN > w:
        return False

    positions = [
        (r0, c0)
        for r0 in range(EDGE_MARGIN, h - obj_h - EDGE_MARGIN + 1)
        for c0 in range(EDGE_MARGIN, w - obj_w - EDGE_MARGIN + 1)
    ]
    random.shuffle(positions)

    boundary = obj["boundary"]
    seeds = obj["seeds"]

    for r0, c0 in positions[:PLACEMENT_RETRIES]:
        r1 = r0 + obj_h - 1
        c1 = c0 + obj_w - 1
        rs = max(0, r0 - OBJECT_GAP)
        re = min(h, r1 + OBJECT_GAP + 1)
        cs = max(0, c0 - OBJECT_GAP)
        ce = min(w, c1 + OBJECT_GAP + 1)
        if bool(blocked[rs:re, cs:ce].any()):
            continue

        patch = grid[r0 : r0 + obj_h, c0 : c0 + obj_w]
        if bool(np.any(patch[boundary] != BG)):
            continue

        patch[boundary] = OUTLINE
        for sr, sc in seeds:
            grid[r0 + sr, c0 + sc] = SEED
        blocked[rs:re, cs:ce] = True
        return True

    return False


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_objects = random.choices(OBJECT_COUNT_VALUES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for _ in range(n_objects):
        seed_count = random.randint(1, MAX_SEEDS)
        obj = _sample_object(seed_count)
        if obj is None:
            return None
        if not _place_object(grid, blocked, obj):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
