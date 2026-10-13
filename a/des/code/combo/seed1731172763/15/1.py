from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
MAGENTA = 6
INPUT_COLORS = (1, 2, 3, 5, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_by_color(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object's outer contour by the parity of the object's area.

    Odd-area objects get a yellow contour; even-area objects get a magenta contour.
    Interior pixels keep their original object color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for comp in _components_by_color(g):
        border_color = YELLOW if len(comp) % 2 == 1 else MAGENTA
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) == BG:
                    out[r, c] = border_color
                    break
    return out


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _shape_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _hole_free(cells: list[tuple[int, int]]) -> bool:
    h, w = _shape_dims(cells)
    solid = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in cells:
        solid[r + 1, c + 1] = True

    seen = np.zeros_like(solid, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not seen[nr, nc] and not solid[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if not solid[r, c] and not seen[r, c]:
                return False
    return True


def _random_rectangle(rng: random.Random, parity: int) -> list[tuple[int, int]] | None:
    choices = [
        (h, w)
        for h in range(1, 6)
        for w in range(1, 6)
        if 1 <= h * w <= 16 and (h * w) % 2 == parity
    ]
    if not choices:
        return None
    h, w = rng.choice(choices)
    return [(r, c) for r in range(h) for c in range(w)]


def _random_corner_cut(rng: random.Random, parity: int) -> list[tuple[int, int]] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for h in range(2, 7):
        for w in range(2, 7):
            for cut_h in range(1, h):
                for cut_w in range(1, w):
                    area = h * w - cut_h * cut_w
                    if 3 <= area <= 16 and area % 2 == parity:
                        candidates.append((h, w, cut_h, cut_w))
    if not candidates:
        return None

    h, w, cut_h, cut_w = rng.choice(candidates)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    cells = {(r, c) for r in range(h) for c in range(w)}

    cut_cells = set()
    for r in range(cut_h):
        for c in range(cut_w):
            if corner == "tl":
                cut_cells.add((r, c))
            elif corner == "tr":
                cut_cells.add((r, w - 1 - c))
            elif corner == "bl":
                cut_cells.add((h - 1 - r, c))
            else:
                cut_cells.add((h - 1 - r, w - 1 - c))

    return _normalize_cells(cells - cut_cells)


def _random_polyomino(rng: random.Random, parity: int) -> list[tuple[int, int]] | None:
    areas = [area for area in range(1, 17) if area % 2 == parity]
    if parity == 0:
        areas = [area for area in areas if area >= 2]
    area = rng.choice(areas)

    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        nxt = rng.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        nr, nc = nxt
        for dr, dc in NEIGHBORS4:
            step = (nr + dr, nc + dc)
            if step not in cells:
                frontier.add(step)

    return _normalize_cells(cells)


def _generate_shape(rng: random.Random, parity: int) -> list[tuple[int, int]]:
    families = [_random_rectangle, _random_corner_cut, _random_polyomino, _random_polyomino]
    for _ in range(200):
        shape = rng.choice(families)(rng, parity)
        if shape is None:
            continue
        area = len(shape)
        h, w = _shape_dims(shape)
        if area % 2 != parity or area > 16 or max(h, w) > 6:
            continue
        if not _hole_free(shape):
            continue
        if area >= 7 and (h == 1 or w == 1) and rng.random() < 0.7:
            continue
        return shape
    raise RuntimeError("failed to generate a valid shape")


def _can_place(blocked: np.ndarray, shape: list[tuple[int, int]], r0: int, c0: int) -> bool:
    h, w = blocked.shape
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
            return False
    return True


def _paint_shape(grid: np.ndarray, blocked: np.ndarray, shape: list[tuple[int, int]], color: int, r0: int, c0: int) -> None:
    h, w = grid.shape
    coords = [(r0 + r, c0 + c) for r, c in shape]
    for rr, cc in coords:
        grid[rr, cc] = color
    for rr, cc in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 18)
    w = rng.randint(10, 18)
    count = rng.randint(2, 6)
    colors = rng.sample(INPUT_COLORS, count)

    parities = [0, 1]
    while len(parities) < count:
        parities.append(rng.randint(0, 1))
    rng.shuffle(parities)

    shapes = [_generate_shape(rng, parity) for parity in parities]
    shapes.sort(key=lambda cells: (_shape_dims(cells)[0] * _shape_dims(cells)[1], len(cells)), reverse=True)

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for shape, color in zip(shapes, colors):
        sh, sw = _shape_dims(shape)
        positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        rng.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            if _can_place(blocked, shape, r0, c0):
                _paint_shape(grid, blocked, shape, color, r0, c0)
                placed = True
                break
        if not placed:
            return None

    comps = _components_by_color(grid)
    if len(comps) != count:
        return None
    if {len(comp) % 2 for comp in comps} != {0, 1}:
        return None
    if np.any(np.isin(grid, (YELLOW, MAGENTA))):
        return None

    out = main(grid)
    if not np.any(out == YELLOW) or not np.any(out == MAGENTA):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
