from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 300
GRID_H_RANGE = (22, 33)
GRID_W_RANGE = (22, 33)
OBJECT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
OBJECT_H_RANGE = (8, 12)
OBJECT_W_RANGE = (8, 12)
PLACEMENT_MARGIN = 2
SHAPE_ATTEMPTS = 80
SEED_ATTEMPTS = 60
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    interior = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if all(bool(mask[nr, nc]) for nr, nc in _neighbors4(r, c)):
                interior[r, c] = True
    return interior


def _normalized(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _bbox_shape(cells: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return max_r + 1, max_c + 1


def _random_connected_shape(rng: random.Random, box_h: int, box_w: int, area: int) -> list[tuple[int, int]] | None:
    for _ in range(40):
        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue
        norm = _normalized(cells)
        h, w = _bbox_shape(norm)
        if h >= 2 and w >= 2:
            return norm
    return None


def _make_region(rng: random.Random, h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        region = np.ones((h, w), dtype=bool)
        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        notch_count = rng.randint(1, 3)
        for side in sides[:notch_count]:
            if side in ("top", "bottom"):
                max_width = max(2, w - 3)
                width = rng.randint(2, max_width)
                start_c = rng.randint(1, w - width - 1)
                depth = rng.randint(1, min(3, h - 4))
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                region[rows, start_c : start_c + width] = False
            else:
                max_height = max(2, h - 3)
                height = rng.randint(2, max_height)
                start_r = rng.randint(1, h - height - 1)
                depth = rng.randint(1, min(3, w - 4))
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                region[start_r : start_r + height, cols] = False

        if rng.random() < 0.5:
            corner_h = rng.randint(1, min(2, h - 4))
            corner_w = rng.randint(1, min(2, w - 4))
            corner = rng.choice(["tl", "tr", "bl", "br"])
            if corner == "tl":
                region[:corner_h, :corner_w] = False
            elif corner == "tr":
                region[:corner_h, w - corner_w :] = False
            elif corner == "bl":
                region[h - corner_h :, :corner_w] = False
            else:
                region[h - corner_h :, w - corner_w :] = False

        comps = _components(region)
        if len(comps) != 1:
            continue

        interior = _erode4(region)
        interior_comps = _components(interior)
        if len(interior_comps) != 1:
            continue
        if int(interior.sum()) < 12:
            continue

        boundary = region & (~interior)
        if not np.any(boundary[0]) or not np.any(boundary[-1]) or not np.any(boundary[:, 0]) or not np.any(boundary[:, -1]):
            continue
        return region, interior
    return None


def _classify_seed_placement(
    interior: np.ndarray, cells: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = interior.shape
    inside = set(map(tuple, np.argwhere(interior)))
    any_positions: list[tuple[int, int]] = []
    edge_positions: list[tuple[int, int]] = []
    open_positions: list[tuple[int, int]] = []
    pat_h, pat_w = _bbox_shape(cells)

    for base_r in range(h - pat_h + 1):
        for base_c in range(w - pat_w + 1):
            placed = {(base_r + dr, base_c + dc) for dr, dc in cells}
            if not placed <= inside:
                continue
            frontier = set()
            blocked = 0
            for r, c in placed:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in placed:
                        continue
                    if (nr, nc) in inside:
                        frontier.add((nr, nc))
                    else:
                        blocked += 1
            if not frontier:
                continue
            any_positions.append((base_r, base_c))
            if blocked > 0:
                edge_positions.append((base_r, base_c))
            else:
                open_positions.append((base_r, base_c))
    return any_positions, edge_positions, open_positions


def _place_seed(
    rng: random.Random, interior: np.ndarray
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(SEED_ATTEMPTS):
        box_h, box_w = rng.choice([(2, 3), (3, 3)])
        area = rng.randint(3, min(5, box_h * box_w - 1))
        shape = _random_connected_shape(rng, box_h, box_w, area)
        if shape is None:
            continue

        any_pos, edge_pos, open_pos = _classify_seed_placement(interior, shape)
        if not any_pos:
            continue
        want_edge = bool(rng.randrange(2))
        if want_edge and edge_pos:
            base_r, base_c = rng.choice(edge_pos)
        elif (not want_edge) and open_pos:
            base_r, base_c = rng.choice(open_pos)
        elif edge_pos:
            base_r, base_c = rng.choice(edge_pos)
        else:
            base_r, base_c = rng.choice(any_pos)

        placed = [(base_r + dr, base_c + dc) for dr, dc in shape]
        return placed, int(want_edge)
    return None


def _grow_once(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    non_grey = g != GREY

    for comp in _components(non_grey):
        if _touches_edge(comp, g.shape):
            continue
        comp_set = set(comp)
        claims: dict[tuple[int, int], set[int]] = {}
        for r, c in comp:
            color = int(g[r, c])
            if color == BLACK:
                continue
            out[r, c] = color
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in comp_set and int(g[nr, nc]) == BLACK:
                    claims.setdefault((nr, nc), set()).add(color)
        for (r, c), colors in claims.items():
            if len(colors) == 1:
                out[r, c] = next(iter(colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each interior seed by one 4-neighbor generation inside its grey outline,
    then erase the outlines.
    """
    return _grow_once(grid)


def _can_place_bbox(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = used.shape
    rr0 = max(0, r0 - PLACEMENT_MARGIN)
    cc0 = max(0, c0 - PLACEMENT_MARGIN)
    rr1 = min(hh, r0 + h + PLACEMENT_MARGIN)
    cc1 = min(ww, c0 + w + PLACEMENT_MARGIN)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _build_object(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(*OBJECT_H_RANGE)
    w = rng.randint(*OBJECT_W_RANGE)
    built = _make_region(rng, h, w)
    if built is None:
        return None
    region, interior = built
    seed = _place_seed(rng, interior)
    if seed is None:
        return None

    seed_cells, _branch = seed
    color = rng.choice(PALETTE)
    obj = np.full((h, w), BLACK, dtype=int)
    boundary = region & (~interior)
    obj[boundary] = GREY
    for r, c in seed_cells:
        obj[r, c] = color

    out = _grow_once(obj)
    seed_count = sum(int(obj[r, c] != BLACK and obj[r, c] != GREY) for r, c in np.argwhere(obj >= 0))
    grown_count = int(np.count_nonzero(out != BLACK))
    if grown_count <= seed_count:
        return None
    return obj, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    n_objects = rng.choice(OBJECT_COUNT_CHOICES)
    colors = list(PALETTE)
    rng.shuffle(colors)

    placed = 0
    for obj_idx in range(n_objects):
        built = None
        for _ in range(SHAPE_ATTEMPTS):
            built = _build_object(rng)
            if built is not None:
                break
        if built is None:
            return None
        obj, _ = built
        oh, ow = obj.shape

        candidate_positions = []
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, h - oh - 1)
            c0 = rng.randint(1, w - ow - 1)
            if _can_place_bbox(used, r0, c0, oh, ow):
                candidate_positions.append((r0, c0))
        if not candidate_positions:
            return None

        r0, c0 = rng.choice(candidate_positions)
        mask = obj != BLACK
        grid[r0 : r0 + oh, c0 : c0 + ow][mask] = obj[mask]
        used[r0 : r0 + oh, c0 : c0 + ow] = True
        placed += 1

    if placed != n_objects:
        return None

    out = _grow_once(grid)
    if np.count_nonzero(grid == GREY) == 0:
        return None
    if np.count_nonzero((grid != BLACK) & (grid != GREY)) == 0:
        return None
    if np.any(out == GREY):
        return None
    if int(np.count_nonzero(out != BLACK)) <= int(np.count_nonzero((grid != BLACK) & (grid != GREY))):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
