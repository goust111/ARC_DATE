from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GEN_ATTEMPTS = 500
H_RANGE = (6, 15)
W_RANGE = (6, 15)


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def mirror_quadrants(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.block([[g[::-1, ::-1], g[::-1, :]], [g[:, ::-1], g]])


def exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc) and g[nr, nc] == BLACK and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    outside = exterior_black_mask(g)
    return (g == BLACK) & ~outside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the open green curves into four quadrants, then fill the black regions that become enclosed.

    Input structure:
    - The input is a black background grid containing 1 to 3 disconnected green 4-connected open curves.
    - Each curve has at least one endpoint on the input border.
    - No green curve is closed inside the input by itself.

    Rule:
    - Build a 2x2 output: the original grid goes in the lower-right quadrant, its left-right flip in the lower-left, its up-down flip in the upper-right, and its 180-degree rotation in the upper-left.
    - On the mirrored result, flood from the outer boundary through black cells using 4-connectivity.
    - Recolor every black cell not reached by that flood to yellow, while keeping green cells unchanged.

    Key insight:
    - The fillable interior does not exist in the input; it appears only after the top-edge and left-edge curve endpoints meet their mirrored copies across the central seams.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = mirror_quadrants(g)
    out[enclosed_black_mask(out)] = YELLOW
    return out


def _is_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _curve_is_open_path(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if not (4 <= len(cell_set) <= 10):
        return False
    degrees = {
        cell: sum((nr, nc) in cell_set for nr, nc in neighbors4(*cell)) for cell in cell_set
    }
    if any(deg not in (1, 2) for deg in degrees.values()):
        return False
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    if len(endpoints) != 2:
        return False
    if not any(_is_border(shape, cell) for cell in endpoints):
        return False
    return len(components_mask(_cells_to_mask(shape, [cells]))) == 1


def _cells_to_mask(
    shape: tuple[int, int],
    curves: list[list[tuple[int, int]]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for curve in curves:
        for r, c in curve:
            mask[r, c] = True
    return mask


def _curves_to_grid(shape: tuple[int, int], curves: list[list[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for curve in curves:
        for r, c in curve:
            grid[r, c] = GREEN
    return grid


def _expanded_blocked(
    shape: tuple[int, int],
    curves: list[list[tuple[int, int]]],
) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for curve in curves:
        for r, c in curve:
            blocked.add((r, c))
            for nr, nc in neighbors8(r, c):
                if in_bounds(shape, nr, nc):
                    blocked.add((nr, nc))
    return blocked


def _make_bridge_curve(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    pairs = []
    for r_left in range(1, min(h - 2, 5) + 1):
        for c_top in range(1, min(w - 2, 5) + 1):
            area = r_left + c_top + 1
            if 4 <= area <= 10:
                pairs.append((r_left, c_top))
    random.shuffle(pairs)
    for r_left, c_top in pairs:
        middle = ["D"] * max(0, r_left - 1) + ["L"] * max(0, c_top - 1)
        random.shuffle(middle)
        moves = ["D"] + middle + ["L"]
        r, c = 0, c_top
        cells = [(r, c)]
        ok = True
        for move in moves:
            if move == "D":
                r += 1
            else:
                c -= 1
            if not in_bounds(shape, r, c):
                ok = False
                break
            cells.append((r, c))
        if ok and _curve_is_open_path(shape, cells):
            return cells
    return None


def _random_edge_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _is_border(shape, (r, c))
    ]
    random.shuffle(cells)
    return cells


def _can_extend_path(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    nxt: tuple[int, int],
) -> bool:
    if not in_bounds(shape, *nxt):
        return False
    if nxt in blocked or nxt in path:
        return False
    old_cells = path[:-2]
    return not any(max(abs(nxt[0] - r), abs(nxt[1] - c)) <= 1 for r, c in old_cells)


def _grow_tail_curve(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    targets = list(range(4, 11))
    random.shuffle(targets)
    starts = [cell for cell in _random_edge_cells(shape) if cell not in blocked]
    for target_len in targets:
        for start in starts[: min(len(starts), 24)]:
            path = [start]

            def dfs() -> bool:
                if len(path) == target_len:
                    return _curve_is_open_path(shape, path)
                options = []
                for nxt in neighbors4(*path[-1]):
                    if not _can_extend_path(shape, path, blocked, nxt):
                        continue
                    options.append(nxt)
                random.shuffle(options)
                for nxt in options:
                    path.append(nxt)
                    if dfs():
                        return True
                    path.pop()
                return False

            if dfs():
                return path
    return None


def _semantic_checks(inp: np.ndarray) -> bool:
    shape = inp.shape
    comps = components_mask(inp == GREEN)
    if not (1 <= len(comps) <= 3):
        return False
    for comp in comps:
        if not _curve_is_open_path(shape, comp):
            return False

    if enclosed_black_mask(inp).any():
        return False

    mirrored = mirror_quadrants(inp)
    enclosed = enclosed_black_mask(mirrored)
    if not enclosed.any():
        return False
    if not np.any((mirrored == BLACK) & exterior_black_mask(mirrored)):
        return False
    if np.any(mirrored == YELLOW):
        return False

    out = main(inp)
    if not np.any(out == YELLOW):
        return False
    if np.array_equal(out, mirrored):
        return False
    if not np.array_equal(out == YELLOW, enclosed):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*H_RANGE)
    w = random.randint(*W_RANGE)
    shape = (h, w)

    bridge = _make_bridge_curve(shape)
    if bridge is None:
        return None
    curves = [bridge]

    desired = random.choices([1, 2, 3], weights=[3, 4, 3])[0]
    while len(curves) < desired:
        blocked = _expanded_blocked(shape, curves)
        extra = _grow_tail_curve(shape, blocked)
        if extra is None:
            break
        curves.append(extra)

    grid = _curves_to_grid(shape, curves)
    if max(grid.shape) > 30 or max(main(grid).shape) > 30:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-fill ARC input")
